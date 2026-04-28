import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.column_name_conversion import COLUMN_NAME_CONVERSION, REQUIRED_COLUMNS
from constants import UPLOADS_DIR


def _validate_columns(columns: list[str]) -> list[str]:
    missing = []
    for canonical in REQUIRED_COLUMNS:
        aliases = COLUMN_NAME_CONVERSION[canonical]
        if not any(alias in columns for alias in aliases):
            missing.append(canonical)
    return missing


def _has_gold_label(columns: list[str]) -> bool:
    return any(alias in columns for alias in COLUMN_NAME_CONVERSION["gold_label"])


def _read_csv_columns(path: Path) -> list[str]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def _count_csv_rows(path: Path) -> int:
    with open(path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _save_uploaded_file(uploaded_file) -> Path:
    save_path = UPLOADS_DIR / uploaded_file.name
    save_path.write_bytes(uploaded_file.read())
    return save_path
