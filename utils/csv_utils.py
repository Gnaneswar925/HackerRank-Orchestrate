"""CSV contracts, path parsing, and atomic output persistence."""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import pandas as pd

from models.schemas import FinalDecision


OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
)
REQUIRED_CLAIM_COLUMNS: Final[frozenset[str]] = frozenset(
    {"user_id", "image_paths", "user_claim", "claim_object"}
)


class CSVContractError(ValueError):
    """Raised when an input or output CSV violates its declared contract."""


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    index: int
    user_id: str
    image_paths: tuple[str, ...]
    resolved_image_paths: tuple[str, ...]
    user_claim: str
    claim_object: str

    @property
    def resume_key(self) -> tuple[str, str, str, tuple[str, ...]]:
        return (
            self.user_id,
            self.user_claim,
            self.claim_object.strip().lower(),
            self.image_paths,
        )


def load_claims(
    csv_path: str | Path,
    *,
    images_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[ClaimRecord]]:
    """Load claims and resolve their evidence paths without touching images."""

    path = Path(csv_path).expanduser()
    if not path.exists() or not path.is_file():
        raise CSVContractError(f"claims CSV does not exist or is not a file: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise CSVContractError(f"unable to read claims CSV: {path}") from exc

    normalized = [_normalize_column(column) for column in frame.columns]
    if len(set(normalized)) != len(normalized):
        raise CSVContractError("claims CSV has duplicate normalized columns")
    frame.columns = normalized
    missing = REQUIRED_CLAIM_COLUMNS.difference(frame.columns)
    if missing:
        raise CSVContractError(
            "claims CSV is missing columns: " + ", ".join(sorted(missing))
        )

    root = (
        Path(images_dir).expanduser()
        if images_dir is not None
        else path.parent
    )
    records: list[ClaimRecord] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        raw_paths = parse_list_value(str(row["image_paths"]))
        resolved_paths = tuple(_resolve_image_path(item, root) for item in raw_paths)
        records.append(
            ClaimRecord(
                index=position,
                user_id=str(row["user_id"]).strip(),
                image_paths=tuple(raw_paths),
                resolved_image_paths=resolved_paths,
                user_claim=str(row["user_claim"]).strip(),
                claim_object=str(row["claim_object"]).strip(),
            )
        )
    return frame, records


def parse_list_value(value: object) -> list[str]:
    """Parse JSON, Python-list, or delimited list cells safely."""

    text = str(value).strip()
    if not text:
        return []
    parsed: object = text
    if text[0] in "[({" and text[-1] in "])}":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError) as exc:
                raise CSVContractError(f"invalid list value: {text[:100]}") from exc
    if isinstance(parsed, (list, tuple, set)):
        values = [str(item).strip() for item in parsed]
    elif isinstance(parsed, str):
        delimiter = (
            ";"
            if ";" in parsed
            else "|"
            if "|" in parsed
            else ","
            if "," in parsed
            else None
        )
        values = [item.strip() for item in parsed.split(delimiter)] if delimiter else [parsed]
    else:
        raise CSVContractError("list cell must contain strings")
    if any(not item for item in values):
        raise CSVContractError("list cell contains an empty item")
    return values


def decision_to_row(decision: FinalDecision) -> dict[str, object]:
    data = decision.model_dump(mode="json")
    for column in ("image_paths", "risk_flags", "supporting_image_ids"):
        data[column] = json.dumps(data[column], ensure_ascii=False, separators=(",", ":"))
    return {column: data[column] for column in OUTPUT_COLUMNS}


def load_existing_output(csv_path: str | Path) -> list[dict[str, str]]:
    path = Path(csv_path).expanduser()
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise CSVContractError(f"unable to read resume output: {path}") from exc
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise CSVContractError(
            "existing output columns do not exactly match the required contract"
        )
    return frame.to_dict(orient="records")


def output_row_key(row: Mapping[str, object]) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        str(row["user_id"]).strip(),
        str(row["user_claim"]).strip(),
        str(row["claim_object"]).strip().lower(),
        tuple(parse_list_value(row["image_paths"])),
    )


def atomic_write_output(
    rows: Sequence[Mapping[str, object]], csv_path: str | Path
) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt output."""

    path = Path(csv_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_image_path(value: str, root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve(strict=False))


def _normalize_column(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")
