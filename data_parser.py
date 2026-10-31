"""FinRAG data preparation.

This module converts Telegram posts, financial news, local PDF files, and RFSD
financial statements into one canonical Parquet corpus.

Heavy dependencies (polars, datasets, pdfplumber, huggingface_hub) are imported
inside the functions that need them so helper functions remain easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

# Keep the output filename versioned because the canonical schema is now v3.
DEFAULT_CORPUS_PATH = Path("./data/processed/unified_corpus_v3.parquet")
DEFAULT_RFSD_CARDS_PATH = Path("./data/processed/rfsd_cards.parquet")
DEFAULT_RFSD_ROOT = Path("./data/raw/RFSD/RFSD")
DEFAULT_PDF_DIRS = (Path("./data/raw"), Path("./data/raw/rrncb_docs"))
RFSD_YEARS = (2023, 2024, 2025)
RFSD_TOP_PER_YEAR = 10_000
PDF_CHUNK_SIZE = 800
PDF_CHUNK_OVERLAP = 100
MIN_TELEGRAM_CHARS = 50
MIN_KAGGLE_BODY_CHARS = 30

CORPUS_COLUMNS = (
    "doc_id", "text", "title", "date", "published_date", "period_end_date",
    "reporting_year", "source_type", "source_name", "topic", "url", "metadata",
)

PLACEHOLDERS = {
    "no title", "none", "null", "nan", "без названия", "untitled", "n/a", "-",
}

OKVED_SECTIONS = {
    "A": "сельское, лесное хозяйство, охота, рыболовство и рыбоводство",
    "B": "добыча полезных ископаемых",
    "C": "обрабатывающие производства",
    "D": "обеспечение электрической энергией, газом и паром",
    "E": "водоснабжение и водоотведение",
    "F": "строительство",
    "G": "торговля оптовая и розничная",
    "H": "транспортировка и хранение",
    "I": "гостиницы и общественное питание",
    "J": "деятельность в области информации и связи",
    "K": "финансовая и страховая деятельность",
    "L": "операции с недвижимым имуществом",
    "M": "профессиональная, научная и техническая деятельность",
    "N": "административная деятельность",
    "O": "государственное управление",
    "P": "образование",
    "Q": "здравоохранение и социальные услуги",
    "R": "культура, спорт, досуг",
    "S": "прочие услуги",
    "T": "домашние хозяйства",
    "U": "экстерриториальные организации",
}

OKVED_HINTS = {
    "A": "сельское хозяйство, ферма, растениеводство, животноводство",
    "B": "нефть, газ, добыча, уголь, руда, полезные ископаемые",
    "C": "завод, производство, обрабатывающая промышленность, фабрика",
    "D": "электроэнергия, газ, пар, теплоснабжение",
    "E": "водоснабжение, водоотведение, канализация",
    "F": "строительство, застройщик, стройка, недвижимость",
    "G": "торговля, ритейл, опт, розница, магазин",
    "H": "транспорт, перевозки, логистика, склад",
    "I": "гостиница, отель, ресторан, общепит, туризм",
    "J": "IT, телеком, связь, информация, интернет, разработка",
    "K": "банк, кредитная организация, финансы, страхование, инвестиции",
    "L": "недвижимость, аренда, операции с имуществом",
    "M": "консалтинг, наука, инженерия, право, бухгалтерия",
    "N": "административная деятельность, аренда персонала, клининг",
    "O": "государственное управление, госорган",
    "P": "образование, школа, вуз, обучение",
    "Q": "здравоохранение, медицина, больница, соцуслуги",
    "R": "культура, спорт, досуг, развлечения",
    "S": "прочие услуги, ремонт, бытовые услуги",
}

# Keep aliases from the original notebook and the later fixes in one place.
REGIONS_RU = {
    "moscow": "Москва", "moscow city": "Москва", "moscow_city": "Москва",
    "saint_petersburg": "Санкт-Петербург", "saint petersburg": "Санкт-Петербург",
    "sankt-petersburg": "Санкт-Петербург", "sankt_petersburg": "Санкт-Петербург",
    "moskovskaya": "Московская область", "leningradskaya": "Ленинградская область",
    "kemerovo": "Кемеровская область", "sverdlovsk": "Свердловская область",
    "tatarstan": "Республика Татарстан", "krasnodar": "Краснодарский край",
    "novosibirsk": "Новосибирская область", "rostovskaya": "Ростовская область",
    "bashkortostan": "Республика Башкортостан", "chelyabinsk": "Челябинская область",
    "nizhegorodskaya": "Нижегородская область", "samarskaya": "Самарская область",
    "krasnoyarsk": "Красноярский край", "perm": "Пермский край",
    "voronezh": "Воронежская область", "volgograd": "Волгоградская область",
    "saratov": "Саратовская область", "tyumen": "Тюменская область",
    "penza": "Пензенская область", "sevastopol": "Севастополь",
    "khanty-mansi": "Ханты-Мансийский АО", "yamalo-nenets": "Ямало-Ненецкий АО",
    "chuvash": "Чувашская Республика", "mordovia": "Республика Мордовия",
    "udmurt": "Удмуртская Республика", "mariy el": "Республика Марий Эл",
    "kabardino-balkar": "Кабардино-Балкарская Республика",
    "karachay-cherkess": "Карачаево-Черкесская Республика",
    "north ossetia": "Республика Северная Осетия", "chechnya": "Чеченская Республика",
    "dagestan": "Республика Дагестан", "ingushetia": "Республика Ингушетия",
    "adygea": "Республика Адыгея", "kalmykia": "Республика Калмыкия",
    "karelia": "Республика Карелия", "komi": "Республика Коми",
    "sakha": "Республика Саха (Якутия)", "buryatia": "Республика Бурятия",
    "tyva": "Республика Тыва", "khakassia": "Республика Хакасия",
    "altay": "Республика Алтай", "crimea": "Республика Крым",
}

KEY_COLUMNS = {
    "line_2110": "выручка",
    "line_2120": "себестоимость",
    "line_2100": "валовая прибыль",
    "line_2200": "прибыль от продаж",
    "line_2300": "прибыль до налогообложения",
    "line_2400": "чистая прибыль",
    "line_1600": "активы (баланс)",
    "line_1100": "внеоборотные активы",
    "line_1200": "оборотные активы",
    "line_1300": "капитал и резервы",
    "line_1400": "долгосрочные обязательства",
    "line_1500": "краткосрочные обязательства",
}
EXPENSE_LINES = {"line_2120"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        # Avoid importing numpy just for NaN sentinels from dataframe libraries.
        if type(value).__name__ in {"NaTType", "NAType"}:
            return True
    except Exception:
        pass
    return False


def _clean_string(value: Any) -> str:
    return "" if _is_missing(value) else str(value).strip()


def normalize_iso_date(value: Any) -> str | None:
    """Return YYYY-MM-DD for supported date-like values, else None."""
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "nat", "none", "null", "n/a"}:
        return None
    # Standard ISO dates and timestamp strings are common in all current sources.
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _stable_scalar(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        # Handle common numpy scalar values without importing numpy.
        if hasattr(value, "item"):
            return _stable_scalar(value.item())
    except Exception:
        pass
    return str(value)


def _json_default(value: Any) -> Any:
    stable = _stable_scalar(value)
    if stable is not None:
        return stable
    return None


def corpus_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash canonical corpus fields in row order; parser writes rows sorted by doc_id."""
    hasher = hashlib.sha256()
    for row in rows:
        stable_row = {column: _stable_scalar(row.get(column)) for column in CORPUS_COLUMNS}
        hasher.update(json.dumps(
            stable_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def normalize_region(value: Any) -> str:
    """Normalize an RFSD region code/name to its display name if known."""
    region = _clean_string(value)
    if not region:
        return ""
    return REGIONS_RU.get(region.casefold(), region)


def _json_string(value: Any) -> str:
    if isinstance(value, str):
        # Preserve existing serialized JSON, but ensure it is valid JSON when possible.
        try:
            json.loads(value)
            return value
        except (ValueError, TypeError):
            pass
    return json.dumps(value or {}, ensure_ascii=False, default=_json_default, sort_keys=True)


@dataclass
class Document:
    """Canonical record for text sources. `date` remains for backwards display only."""
    doc_id: str
    text: str
    title: str | None = None
    date: str | None = None
    source_type: str = ""
    source_name: str = ""
    topic: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    published_date: str | None = None
    period_end_date: str | None = None
    reporting_year: int | None = None

    def to_dict(self) -> dict[str, Any]:
        published = normalize_iso_date(self.published_date)
        period_end = normalize_iso_date(self.period_end_date)
        # Existing `date` field is retained for compatibility, but retrieval uses
        # published_date only for publication filters and reporting_year for RFSD.
        legacy_date = normalize_iso_date(self.date) or published or period_end
        reporting_year = self.reporting_year
        if reporting_year is not None:
            try:
                reporting_year = int(reporting_year)
            except (TypeError, ValueError):
                reporting_year = None
        return {
            "doc_id": _clean_string(self.doc_id),
            "text": _clean_string(self.text),
            "title": _clean_string(self.title) or None,
            "date": legacy_date,
            "published_date": published,
            "period_end_date": period_end,
            "reporting_year": reporting_year,
            "source_type": _clean_string(self.source_type),
            "source_name": _clean_string(self.source_name),
            "topic": _clean_string(self.topic) or None,
            "url": _clean_string(self.url) or None,
            "metadata": _json_string(self.metadata),
        }


def make_document_frame(rows: Sequence[Mapping[str, Any]]):
    """Create a Polars DataFrame with stable canonical columns and dtypes."""
    import polars as pl

    schema = {
        "doc_id": pl.String,
        "text": pl.String,
        "title": pl.String,
        "date": pl.String,
        "published_date": pl.String,
        "period_end_date": pl.String,
        "reporting_year": pl.Int32,
        "source_type": pl.String,
        "source_name": pl.String,
        "topic": pl.String,
        "url": pl.String,
        "metadata": pl.String,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(list(rows), infer_schema_length=None, strict=False)
    for column, dtype in schema.items():
        if column not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(column))
        else:
            df = df.with_columns(pl.col(column).cast(dtype, strict=False))
    return df.select(list(CORPUS_COLUMNS))


def _dataset_rows(dataset: Any) -> Iterable[Mapping[str, Any]]:
    """Use the train split when present; otherwise use the first available split."""
    if hasattr(dataset, "keys"):
        keys = list(dataset.keys())
        split = "train" if "train" in keys else (keys[0] if keys else None)
        if split is None:
            return []
        return dataset[split]
    return dataset


def _slug(value: Any, max_length: int = 45) -> str:
    text = _clean_string(value).casefold()
    text = re.sub(r"[^a-zа-я0-9]+", "-", text, flags=re.IGNORECASE).strip("-")
    return (text[:max_length].strip("-") or "unknown")


def convert_telegram(dataset: Any, min_chars: int = MIN_TELEGRAM_CHARS) -> list[Document]:
    """Convert Telegram rows; IDs include the channel to avoid cross-channel collisions."""
    documents: list[Document] = []
    for idx, row in enumerate(_dataset_rows(dataset)):
        text = _clean_string(row.get("message"))
        if len(text) < min_chars:
            continue
        channel = _clean_string(row.get("channel_name")) or "unknown"
        message_id = _clean_string(row.get("message_id")) or str(idx)
        pub_date = normalize_iso_date(row.get("date"))
        documents.append(Document(
            doc_id=f"tg_{_slug(channel)}_{_slug(message_id, 40)}",
            text=text,
            title=None,
            date=pub_date,
            published_date=pub_date,
            source_type="telegram",
            source_name=channel,
            topic=_clean_string(row.get("economic_topic")) or None,
            url=_clean_string(row.get("url") or row.get("message_url")) or None,
            metadata={"viral_final": _stable_scalar(row.get("viral_final"))},
        ))
    return documents


def extract_pseudo_title(body: str, max_len: int = 150) -> str:
    body = _clean_string(body)
    sentence = re.match(r"^(.{20,}?[.!?])(?:\s|$)", body, flags=re.DOTALL)
    if sentence:
        return sentence.group(1).strip()[:max_len]
    excerpt = body[:max_len].strip()
    if len(body) > max_len and " " in excerpt:
        excerpt = excerpt.rsplit(" ", 1)[0]
    excerpt = excerpt.rstrip(".,!?;: ")
    return (excerpt + "…") if len(body) > len(excerpt) else excerpt


def convert_kaggle(
    dataset: Any,
    min_body_chars: int = MIN_KAGGLE_BODY_CHARS,
    short_body_threshold: int = 150,
    ) -> list[Document]:
    
    documents: list[Document] = []
    splits = list(dataset.keys()) if hasattr(dataset, "keys") else ["train"]
    for split in splits:
        rows = dataset[split] if hasattr(dataset, "keys") else dataset
        for idx, row in enumerate(rows):
            title = _clean_string(row.get("title"))
            body = _clean_string(row.get("body"))
            if title.casefold() in PLACEHOLDERS:
                title = ""
            if len(body) < min_body_chars and len(title) < 20:
                continue

            if not title:
                if body and len(body) <= short_body_threshold:
                    title = body.rstrip(".,!?;: ")
                else:
                    title = extract_pseudo_title(body) if body else ""
                title_source = "derived"
            else:
                title_source = "original"

            if body and len(body) <= short_body_threshold:
                # A derived title is the first line of this same short item: keep body once.
                # If a real title exists, retain both title and body instead of dropping details.
                normalized_title = title.casefold().strip(" .,!?:;")
                normalized_body = body.casefold().strip(" .,!?:;")
                if title_source == "derived" or normalized_title == normalized_body:
                    text = body
                elif title:
                    text = f"{title}\n\n{body}"
                else:
                    text = body
            elif title and body:
                text = f"{title}\n\n{body}"
            else:
                text = title or body
            if not text.strip():
                continue

            pub_date = normalize_iso_date(row.get("date"))
            source_name = _clean_string(row.get("source")) or "unknown"
            row_tags = row.get("tags", "")
            source_url = _clean_string(
                row.get("url") or row.get("link") or row.get("article_url")
            ) or None
            documents.append(Document(
                doc_id=f"kaggle_{_slug(split, 24)}_{idx}",
                text=text,
                title=title or None,
                date=pub_date,
                published_date=pub_date,
                source_type="news",
                source_name=source_name,
                url=source_url,
                metadata={"tags": row_tags, "title_source": title_source},
            ))
    return documents


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract and normalize text from a PDF using pdfplumber."""
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    text = "\n".join(pages)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = PDF_CHUNK_SIZE,
    overlap: int = PDF_CHUNK_OVERLAP,
    min_chunk_chars: int = 100,
) -> list[str]:
    """Split text into overlapping character chunks with argument validation."""
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap должен быть >= 0 и меньше chunk_size")
    text = _clean_string(text)
    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if len(chunk) >= min_chunk_chars:
            chunks.append(chunk)
        start += step
    return chunks


def convert_rrncb(
    pdf_dirs: Sequence[str | Path] = DEFAULT_PDF_DIRS,
    chunk_size: int = PDF_CHUNK_SIZE,
    overlap: int = PDF_CHUNK_OVERLAP,
    min_document_chars: int = 200,
) -> list[Document]:
    """Convert local RRNCB PDFs into chunks; no PDF files are downloaded by this function."""
    files: set[Path] = set()
    for directory in pdf_dirs:
        root = Path(directory)
        if root.exists():
            files.update(path.resolve() for path in root.rglob("*.pdf") if path.is_file())
    documents: list[Document] = []
    for pdf_path in sorted(files, key=lambda p: str(p).casefold()):
        try:
            text = extract_pdf_text(pdf_path)
        except Exception as exc:
            print(f"[WARNING] Не удалось прочитать PDF {pdf_path.name}: {exc}")
            continue
        if len(text) < min_document_chars:
            print(f"[WARNING] Пустой/короткий PDF пропущен: {pdf_path.name}")
            continue
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        # Prefer a project-relative path so IDs stay stable when the project folder moves.
        try:
            rel_hint = str(pdf_path.relative_to(Path("./data/raw").resolve()))
        except ValueError:
            rel_hint = pdf_path.name
        path_hash = hashlib.sha1(rel_hint.encode("utf-8")).hexdigest()[:8]
        stem = _slug(pdf_path.stem, 60)
        for chunk_index, chunk in enumerate(chunks):
            documents.append(Document(
                doc_id=f"rrncb_{stem}_{path_hash}_chunk_{chunk_index:04d}",
                text=chunk,
                title=pdf_path.stem,
                source_type="pdf",
                source_name=pdf_path.name,
                metadata={
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                    "source_path": rel_hint,
                },
            ))
    return documents


def fmt_number(value: Any) -> str | None:
    """Human-readable Russian-style magnitude formatter used in RFSD cards."""
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return None
    if abs(number) >= 1e9:
        return f"{number / 1e9:.2f} млрд"
    if abs(number) >= 1e6:
        return f"{number / 1e6:.1f} млн"
    if abs(number) >= 1e3:
        return f"{number / 1e3:.0f} тыс"
    return f"{number:.0f}"


def _format_identifier(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def make_card(row: Mapping[str, Any]) -> str | None:
    """Final RFSD card builder: region names + OКВЭД hints + absolute cost of sales."""
    inn = _format_identifier(row.get("inn"))
    if not inn:
        return None
    year_value = row.get("year")
    try:
        year = int(year_value) if not _is_missing(year_value) else None
    except (TypeError, ValueError):
        year = None

    parts = [f"Организация с ИНН {inn}"]
    if year:
        parts[0] += f", {year} год"
    ogrn = _format_identifier(row.get("ogrn"))
    if ogrn:
        parts[0] += f", ОГРН {ogrn}"

    region = normalize_region(row.get("region"))
    if region:
        parts.append(f"Регион: {region}")

    section_code = _clean_string(row.get("okved_section")).upper()
    if section_code in OKVED_SECTIONS:
        section_name = OKVED_SECTIONS[section_code]
        hint = OKVED_HINTS.get(section_code, "")
        okved = _clean_string(row.get("okved"))
        description = f"Вид деятельности: {section_name}"
        if hint:
            description += f" ({hint})"
        if okved:
            description += f" (ОКВЭД {okved})"
        parts.append(description)

    age = row.get("age")
    if not _is_missing(age):
        try:
            parts.append(f"Возраст: {int(age)} лет")
        except (TypeError, ValueError, OverflowError):
            pass

    metrics: list[str] = []
    for column, label in KEY_COLUMNS.items():
        value = row.get(column)
        if _is_missing(value):
            continue
        if column in EXPENSE_LINES:
            try:
                value = abs(float(value))
            except (TypeError, ValueError):
                pass
        formatted = fmt_number(value)
        if formatted:
            metrics.append(f"{label}: {formatted} руб")
    if not metrics:
        return None
    return ". ".join(parts) + ". Финансовые показатели: " + "; ".join(metrics) + "."


def build_rfsd_cards(
    path: str | Path,
    year: int,
    top_per_year: int = RFSD_TOP_PER_YEAR,
) -> list[dict[str, Any]]:
    """Build RFSD cards from one year, preserving original data-selection policy."""
    import polars as pl

    path_string = str(path)
    path = Path(path_string)
    if any(symbol in path_string for symbol in "*?[]"):
        matches = list(path.parent.glob(path.name))
        if not matches:
            raise FileNotFoundError(f"Файлы RFSD за {year} не найдены по шаблону: {path_string}")
    elif not path.exists():
        raise FileNotFoundError(f"Файл RFSD за {year} не найден: {path}")
    if top_per_year <= 0:
        raise ValueError("top_per_year должен быть положительным")

    lf = pl.scan_parquet(str(path))
    schema = lf.collect_schema().names()
    required = {"inn", "filed", "line_2110"}
    missing = required - set(schema)
    if missing:
        raise ValueError(f"В {path} отсутствуют обязательные колонки: {sorted(missing)}")

    base_columns = ["inn", "ogrn", "region", "okved", "okved_section", "age"]
    financial_columns = [column for column in KEY_COLUMNS if column in schema]
    selected_columns = [column for column in base_columns + financial_columns if column in schema]

    frame = (
        lf.filter(pl.col("filed") == 1.0)
          .filter(pl.col("line_2110").is_not_null())
          .filter(pl.col("line_2110") > 0)
          .sort("line_2110", descending=True)
          .unique(subset=["inn"], keep="first", maintain_order=True)
          .head(top_per_year)
          .select(selected_columns)
          .collect()
    )
    print(f"RFSD {year}: отобрано строк с filed=1 и положительной выручкой: {frame.height:,}")

    cards: list[dict[str, Any]] = []
    period_end = f"{int(year):04d}-12-31"
    for rank, row in enumerate(frame.iter_rows(named=True), start=1):
        row["year"] = int(year)
        text = make_card(row)
        if not text:
            continue
        if len(text) > 1200:
            text = text[:1197].rstrip() + "..."
        inn = _format_identifier(row.get("inn"))
        region_code = _clean_string(row.get("region")) or None
        region_name = normalize_region(region_code) or None
        section_code = _clean_string(row.get("okved_section")).upper() or None
        metadata = {
            "inn": inn,
            "ogrn": _format_identifier(row.get("ogrn")) or None,
            "year": int(year),
            "reporting_year": int(year),
            "okved": _clean_string(row.get("okved")) or None,
            "okved_section": section_code,
            "region": region_code,
            "region_name": region_name,
            "revenue": _stable_scalar(row.get("line_2110")),
            "rank_by_revenue": rank,
        }
        cards.append({
            "doc_id": f"rfsd_{inn}_{int(year)}",
            "text": text,
            "title": f"Отчётность ИНН {inn} за {int(year)}",
            # Legacy date and period_end_date describe the reporting period, NOT publication.
            "date": period_end,
            "published_date": None,
            "period_end_date": period_end,
            "reporting_year": int(year),
            "source_type": "structured",
            "source_name": f"ИНН {inn}",
            "topic": "Финансовая отчётность",
            "url": None,
            "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=_json_default),
        })
    return cards


def download_rfsd_snapshot(
    local_dir: str | Path = "./data/raw/RFSD",
    years: Sequence[int] = RFSD_YEARS,
) -> Path:
    """Download only requested RFSD yearly Parquet shards from the Hugging Face dataset."""
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    patterns = [f"RFSD/year={int(year)}/*.parquet" for year in years]
    snapshot_download(
        repo_id="irlspbru/RFSD",
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=patterns,
    )
    return local_dir / "RFSD"


def download_rrncb_eval_csv():
    """Download the RRNCB public evaluation CSV; it is not part of the indexed corpus."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id="FractalGPT/RRNCBPublic",
        filename="rrncb_public_dataset.csv",
        repo_type="dataset",
    )


def load_source_datasets():
    """Load the two HF text datasets used by the current project."""
    from datasets import load_dataset

    telegram = load_dataset("bruhwalkk/economic-telegram-news-corpus-2025")
    kaggle = load_dataset("Kasymkhan/RussianFinancialNews")
    return telegram, kaggle


def save_rfsd_cards(cards: Sequence[Mapping[str, Any]], path: str | Path = DEFAULT_RFSD_CARDS_PATH) -> Path:
    """Save RFSD cards in the same canonical schema as all other source records."""
    import polars as pl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = make_document_frame(cards)
    frame.write_parquet(path)
    return path


def save_unified_corpus(
    telegram_docs: Sequence[Document],
    news_docs: Sequence[Document],
    pdf_docs: Sequence[Document],
    rfsd_cards_path: str | Path = DEFAULT_RFSD_CARDS_PATH,
    output_path: str | Path = DEFAULT_CORPUS_PATH,
) -> Path:
    """Combine all four sources, validate IDs/text, sort by ID, and save Parquet v3."""
    import polars as pl

    rfsd_cards_path = Path(rfsd_cards_path)
    if not rfsd_cards_path.exists():
        raise FileNotFoundError(f"Не найден Parquet карточек RFSD: {rfsd_cards_path}")
    rfsd_frame = pl.read_parquet(rfsd_cards_path)
    source_rows: list[dict[str, Any]] = []
    for document in list(telegram_docs) + list(news_docs) + list(pdf_docs):
        source_rows.append(document.to_dict())
    # DataFrame construction is driven through the same canonical schema.
    text_frame = make_document_frame(source_rows)
    for column in CORPUS_COLUMNS:
        if column not in rfsd_frame.columns:
            rfsd_frame = rfsd_frame.with_columns(pl.lit(None).alias(column))
    rfsd_frame = rfsd_frame.select(list(CORPUS_COLUMNS))
    text_frame = text_frame.select(list(CORPUS_COLUMNS))
    combined = pl.concat([text_frame, rfsd_frame], how="vertical_relaxed")

    combined = combined.with_columns(
        pl.col("doc_id").cast(pl.String),
        pl.col("text").cast(pl.String),
    )
    # Do not deduplicate silently: it can hide ID-generation bugs or overwrite records.
    invalid = combined.filter(
        pl.col("doc_id").is_null()
        | (pl.col("doc_id").str.strip_chars().str.len_chars() == 0)
        | pl.col("text").is_null()
        | (pl.col("text").str.strip_chars().str.len_chars() == 0)
    )
    if invalid.height:
        raise ValueError(f"В корпусе {invalid.height} строк без doc_id или текста")
    duplicates = (
        combined.group_by("doc_id").len().filter(pl.col("len") > 1)
        .sort("doc_id")
    )
    if duplicates.height:
        sample_ids = duplicates.get_column("doc_id").head(10).to_list()
        raise ValueError(
            f"Найдены повторяющиеся doc_id ({duplicates.height} уникальных ID). "
            f"Примеры: {sample_ids}. Исправьте генерацию ID; автоматическое удаление отключено."
        )

    combined = combined.sort("doc_id")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path)

    print(f"Сохранено: {output_path} ({combined.height:,} документов)")
    print(combined.group_by("source_type").len().sort("source_type"))
    return output_path


def validate_corpus(path: str | Path = DEFAULT_CORPUS_PATH) -> dict[str, Any]:
    """Perform final corpus checks without modifying the file."""
    import polars as pl

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pl.read_parquet(path)
    missing_columns = set(CORPUS_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"В корпусе отсутствуют колонки: {sorted(missing_columns)}")
    duplicate_count = df.height - df.get_column("doc_id").n_unique()
    blank_text_count = df.filter(
        pl.col("text").is_null() | (pl.col("text").str.strip_chars().str.len_chars() == 0)
    ).height
    bad_dates = df.filter(
        pl.col("published_date").is_not_null()
        & (pl.col("published_date").str.len_chars() != 10)
    ).height
    title_placeholder_count = df.filter(
        pl.col("title").is_not_null()
        & pl.col("title").str.to_lowercase().is_in(list(PLACEHOLDERS))
    ).height
    summary = {
        "rows": df.height,
        "duplicate_doc_ids": duplicate_count,
        "blank_texts": blank_text_count,
        "bad_published_dates": bad_dates,
        "placeholder_titles": title_placeholder_count,
        "source_counts": {
            row["source_type"]: row["len"]
            for row in df.group_by("source_type").len().iter_rows(named=True)
        },
    }
    if duplicate_count or blank_text_count or bad_dates:
        raise ValueError(f"Проверка корпуса не пройдена: {summary}")
    return summary
