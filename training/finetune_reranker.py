"""Fine-tune a FinRAG CrossEncoder with an explicit PyTorch loop.

This version avoids CrossEncoderTrainer/Transformers Trainer device placement,
so the model, input tensors, labels, and BCE positive-class weight are all
explicitly placed on the same device. Intended for small pilot datasets on macOS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Некорректный JSONL, строка {line_no}: {exc}") from exc
            if not row.get("query") or not row.get("document_text"):
                continue
            try:
                label = float(row.get("label", -1))
            except (TypeError, ValueError):
                continue
            if label not in {0.0, 1.0}:
                continue
            rows.append({**row, "label": label})
    return rows


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_hf_model(model: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(str(output_dir))
    model.tokenizer.save_pretrained(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "evaluation/data/reranker_pairs_mined.jsonl")
    parser.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/bge_reranker_finrag_cpu_v1")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Smoke test limit; -1 means train for --epochs")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if not args.pairs.exists():
        raise FileNotFoundError(args.pairs)
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_length < 32:
        parser.error("epochs, batch-size должны быть > 0, max-length должен быть >= 32")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("max-steps должен быть -1 или положительным числом")

    try:
        import torch
        import sentence_transformers
        import datasets
        import transformers
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Необходимы torch, datasets, transformers и sentence-transformers") from exc

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Запрошено устройство mps, но torch.backends.mps.is_available() == False")

    rows = _load_pairs(args.pairs)
    train_rows = [r for r in rows if r.get("split") == "train"]
    valid_rows = [r for r in rows if r.get("split") == "validation"]
    for split_name, split_rows in (("train", train_rows), ("validation", valid_rows)):
        counts = Counter(r["label"] for r in split_rows)
        if counts.get(0.0, 0) == 0 or counts.get(1.0, 0) == 0:
            raise ValueError(f"split={split_name} должен содержать label=0 и label=1; сейчас {dict(counts)}")
    if not train_rows or not valid_rows:
        raise ValueError("Нужны пары и в train, и в validation")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Force the Hugging Face backbone onto the requested device.
    cross_encoder = CrossEncoder(
        args.base_model,
        num_labels=1,
        max_length=args.max_length,
        device=args.device,
    )
    hf_model = cross_encoder.model
    hf_model.to(device)
    actual_devices = {str(p.device) for p in hf_model.parameters()}
    if actual_devices != {str(device)}:
        raise RuntimeError(f"Model parameters are on {actual_devices}; expected only {device}")
    print(f"Training device: {device}")
    print(f"Model parameter devices: {sorted(actual_devices)}")

    tokenizer = cross_encoder.tokenizer

    def make_batches(items: list[dict[str, Any]], order: list[int], batch_size: int):
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            yield [items[i] for i in idxs]

    def encode_batch(batch: list[dict[str, Any]]):
        queries = [str(r["query"]) for r in batch]
        docs = [str(r["document_text"]) for r in batch]
        tokens = tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        # Explicitly move every input tensor to the same device as the model.
        return {key: value.to(device) for key, value in tokens.items()}

    pos_count = sum(r["label"] == 1.0 for r in train_rows)
    neg_count = sum(r["label"] == 0.0 for r in train_rows)
    pos_weight_value = max(1.0, neg_count / max(1, pos_count))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(hf_model.parameters(), lr=args.learning_rate)

    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    target_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = int(target_steps * 0.1)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=target_steps
    )

    # Check the encoded lengths up front; tokenizer truncation is explicit.
    max_observed_tokens = 0
    for r in train_rows + valid_rows:
        encoded = tokenizer(
            str(r["query"]), str(r["document_text"]),
            truncation=True, max_length=args.max_length, add_special_tokens=True,
        )
        max_observed_tokens = max(max_observed_tokens, len(encoded["input_ids"]))
    print(f"Prepared {len(train_rows)} train and {len(valid_rows)} validation pairs; "
          f"max_length={args.max_length}; max tokenized pair length={max_observed_tokens}")
    print(f"BCE pos_weight device: {pos_weight.device}; value={pos_weight_value:.4f}")

    global_step = 0
    total_loss = 0.0
    best_val_loss = float("inf")
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    smoke_mode = args.max_steps > 0
    epoch_count = math.ceil(args.epochs)

    for epoch_idx in range(epoch_count):
        hf_model.train()
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch_idx).shuffle(order)
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in make_batches(train_rows, order, args.batch_size):
            if global_step >= target_steps:
                break
            inputs = encode_batch(batch)
            labels = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=device)

            # Guard explicitly against the mixed-device error before forward/loss.
            tensor_devices = {str(t.device) for t in inputs.values()}
            tensor_devices.add(str(labels.device))
            tensor_devices.add(str(pos_weight.device))
            if tensor_devices != {str(device)}:
                raise RuntimeError(f"Mixed devices before forward pass: {tensor_devices}; expected {device}")

            optimizer.zero_grad(set_to_none=True)
            outputs = hf_model(**inputs)
            logits = outputs.logits.reshape(-1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hf_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            step_loss = float(loss.detach().cpu().item())
            epoch_loss_sum += step_loss
            total_loss += step_loss
            epoch_steps += 1
            global_step += 1
            if global_step == 1 or global_step % args.log_every == 0 or global_step == target_steps:
                print(f"step {global_step}/{target_steps} | loss={step_loss:.4f} | device={device}")

        epoch_train_loss = epoch_loss_sum / max(1, epoch_steps)
        record: dict[str, Any] = {
            "epoch": epoch_idx + 1,
            "global_step": global_step,
            "train_loss": epoch_train_loss,
        }

        # On a full run evaluate at the end of each pass; for smoke tests, skip eval
        # and just verify that the explicit CPU/MPS training loop completes.
        if not smoke_mode:
            hf_model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                val_order = list(range(len(valid_rows)))
                for batch in make_batches(valid_rows, val_order, 1):
                    inputs = encode_batch(batch)
                    labels = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=device)
                    outputs = hf_model(**inputs)
                    logits = outputs.logits.reshape(-1)
                    loss = criterion(logits, labels)
                    val_loss_sum += float(loss.detach().cpu().item())
                    val_count += 1
            val_loss = val_loss_sum / max(1, val_count)
            record["validation_loss"] = val_loss
            print(f"epoch {epoch_idx + 1}: train_loss={epoch_train_loss:.4f} validation_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch_idx + 1
                _save_hf_model(cross_encoder, args.output_dir / "model")
                print(f"Saved new best model (validation_loss={best_val_loss:.4f})")
        history.append(record)

        if global_step >= target_steps:
            break

    final_model_dir = args.output_dir / "model"
    # Smoke tests do not run validation; save the resulting test model explicitly.
    if smoke_mode:
        _save_hf_model(cross_encoder, final_model_dir)
    elif not final_model_dir.exists():
        _save_hf_model(cross_encoder, final_model_dir)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "output_model": str(final_model_dir.resolve()),
        "pairs_file": str(args.pairs.resolve()),
        "pairs_sha256": _fingerprint(args.pairs),
        "train_pairs": len(train_rows),
        "validation_pairs": len(valid_rows),
        "train_label_counts": dict(Counter(str(r["label"]) for r in train_rows)),
        "validation_label_counts": dict(Counter(str(r["label"]) for r in valid_rows)),
        "epochs_requested": args.epochs,
        "max_steps": args.max_steps,
        "steps_completed": global_step,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "device": str(device),
        "pos_weight": pos_weight_value,
        "seed": args.seed,
        "python_version": platform.python_version(),
        "sentence_transformers_version": sentence_transformers.__version__,
        "datasets_version": datasets.__version__,
        "transformers_version": transformers.__version__,
        "train_loss_average": total_loss / max(1, global_step),
        "best_validation_loss": None if smoke_mode or math.isinf(best_val_loss) else best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "smoke_test_only": smoke_mode,
        "label_warning": "Training labels include synthetic positives and LLM-judged silver negatives; compare carefully against the fixed human-reviewed benchmark test split.",
    }
    (args.output_dir / "train_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {'smoke-test' if smoke_mode else 'fine-tuned'} reranker to {final_model_dir.resolve()}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
