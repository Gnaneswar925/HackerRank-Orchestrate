"""Deterministic evaluation of object-specific evidence requirements."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

import pandas as pd

from models.schemas import (
    ClaimObject,
    EvidenceResult,
    ImageAnalysis,
    ImageQuality,
    IssueType,
    VisionAnalysis,
    VisionFlag,
)


REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"claim_object", "applies_to"}
)
REQUIREMENT_COLUMN_ALIASES: Final[tuple[str, ...]] = (
    "evidence_requirement",
    "requirement",
    "evidence_standard",
    "minimum_evidence",
)
WILDCARDS: Final[frozenset[str]] = frozenset(
    {"*", "all", "any", "general", "all claims", "all issues", "all objects"}
)
NUMBER_WORDS: Final[dict[str, int]] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}
MAX_REASON_LENGTH: Final[int] = 500


class EvidenceAgentError(RuntimeError):
    """Base error raised by evidence requirement processing."""


class EvidenceRequirementsError(EvidenceAgentError):
    """Raised when the requirements CSV is missing or malformed."""


class RequirementType(str, Enum):
    """Machine-evaluable evidence predicates."""

    VALID_IMAGE = "valid_image"
    CORRECT_OBJECT = "correct_object"
    DAMAGE_VISIBLE = "damage_visible"
    CLAIMED_ISSUE_VISIBLE = "claimed_issue_visible"
    CLAIMED_PART_VISIBLE = "claimed_part_visible"
    ACCEPTABLE_QUALITY = "acceptable_quality"
    MINIMUM_IMAGES = "minimum_images"
    NO_MANIPULATION = "no_manipulation"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    """One normalized row selected from evidence_requirements.csv."""

    requirement_id: str
    description: str
    predicates: tuple[RequirementType, ...]
    minimum_images: int


@dataclass(frozen=True, slots=True)
class RequirementEvaluation:
    requirement: EvidenceRequirement
    met: bool
    reason: str


def load_evidence_requirements(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the evidence-requirements table.

    Required columns are ``claim_object`` and ``applies_to``. A requirement
    description must be supplied using one of the documented aliases. Optional
    columns are ``requirement_id``, ``requirement_type``, and ``minimum_images``.
    """

    path = Path(csv_path).expanduser()
    if not path.exists():
        raise EvidenceRequirementsError(f"requirements CSV does not exist: {path}")
    if not path.is_file():
        raise EvidenceRequirementsError(f"requirements path is not a file: {path}")

    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise EvidenceRequirementsError(
            f"unable to read requirements CSV: {path}"
        ) from exc

    normalized_columns = [_normalize_column(column) for column in frame.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise EvidenceRequirementsError(
            "requirements CSV contains duplicate columns after normalization"
        )
    frame.columns = normalized_columns

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise EvidenceRequirementsError(
            f"requirements CSV is missing columns: {', '.join(sorted(missing))}"
        )
    requirement_column = _find_requirement_column(frame)
    frame = frame.rename(columns={requirement_column: "requirement"}).copy()

    for column in frame.columns:
        frame[column] = frame[column].map(lambda value: str(value).strip())
    if frame.empty:
        raise EvidenceRequirementsError("requirements CSV contains no rows")
    for column in ("claim_object", "applies_to", "requirement"):
        blank_rows = frame.index[frame[column].eq("")].tolist()
        if blank_rows:
            csv_rows = ", ".join(str(index + 2) for index in blank_rows[:5])
            raise EvidenceRequirementsError(
                f"column '{column}' is blank on CSV row(s): {csv_rows}"
            )
    return frame


def evaluate_evidence(
    claim_object: ClaimObject | str,
    issue_type: IssueType | str,
    image_findings: VisionAnalysis,
    evidence_requirements_csv: str | Path | pd.DataFrame,
) -> EvidenceResult:
    """Determine whether the matched minimum evidence standard is satisfied."""

    normalized_object = _parse_enum(ClaimObject, claim_object, "claim_object")
    normalized_issue = _parse_enum(IssueType, issue_type, "issue_type")
    findings = VisionAnalysis.model_validate(image_findings)
    if findings.claim_object != normalized_object:
        raise ValueError("claim_object does not match image_findings.claim_object")

    frame = (
        _validate_preloaded_requirements(evidence_requirements_csv)
        if isinstance(evidence_requirements_csv, pd.DataFrame)
        else load_evidence_requirements(evidence_requirements_csv)
    )
    matched = _match_rows(frame, normalized_object, normalized_issue)
    supporting_ids = _supporting_images(findings, normalized_issue)
    contradicting_ids = _contradicting_images(findings, normalized_issue)

    if matched.empty:
        return EvidenceResult(
            evidence_standard_met=False,
            evidence_standard_met_reason=(
                f"No evidence standard is configured for "
                f"{normalized_object.value}/{normalized_issue.value}."
            ),
            required_evidence=[],
            satisfied_requirements=[],
            missing_requirements=["matching evidence standard"],
            supporting_image_ids=supporting_ids,
            contradicting_image_ids=contradicting_ids,
            evidence_confidence=0.0,
        )

    requirements = [
        _row_to_requirement(row, csv_row=int(index) + 2)
        for index, row in matched.iterrows()
    ]
    evaluations = [
        _evaluate_requirement(
            requirement,
            findings=findings,
            issue_type=normalized_issue,
            supporting_ids=supporting_ids,
        )
        for requirement in requirements
    ]
    standard_met = all(result.met for result in evaluations)
    required = [result.requirement.description for result in evaluations]
    satisfied = [result.requirement.description for result in evaluations if result.met]
    missing_requirements = [
        result.requirement.description for result in evaluations if not result.met
    ]
    reason = _build_reason(evaluations, standard_met, supporting_ids)

    return EvidenceResult(
        evidence_standard_met=standard_met,
        evidence_standard_met_reason=reason,
        required_evidence=required,
        satisfied_requirements=satisfied,
        missing_requirements=missing_requirements,
        supporting_image_ids=supporting_ids,
        contradicting_image_ids=contradicting_ids,
        evidence_confidence=_evidence_confidence(findings, evaluations),
    )


def _match_rows(
    frame: pd.DataFrame,
    claim_object: ClaimObject,
    issue_type: IssueType,
) -> pd.DataFrame:
    object_matches = frame["claim_object"].map(
        lambda value: _matches_selector(value, claim_object.value)
    )
    issue_matches = frame["applies_to"].map(
        lambda value: _matches_selector(value, issue_type.value)
    )
    return frame.loc[object_matches & issue_matches]


def _matches_selector(selector: str, target: str) -> bool:
    values = {
        _normalize_value(value)
        for value in re.split(r"[,;|]", selector)
        if value.strip()
    }
    return bool(values.intersection(WILDCARDS)) or _normalize_value(target) in values


def _row_to_requirement(row: pd.Series, *, csv_row: int) -> EvidenceRequirement:
    description = str(row["requirement"]).strip()
    requirement_id = str(row.get("requirement_id", "")).strip() or f"row_{csv_row}"
    explicit_types = str(row.get("requirement_type", "")).strip()
    predicates = _parse_predicates(explicit_types, description)
    minimum_images = _parse_minimum_images(
        str(row.get("minimum_images", "")).strip(),
        description,
        csv_row=csv_row,
    )
    if minimum_images > 1 and RequirementType.MINIMUM_IMAGES not in predicates:
        predicates = (*predicates, RequirementType.MINIMUM_IMAGES)
    return EvidenceRequirement(
        requirement_id=requirement_id,
        description=description,
        predicates=predicates,
        minimum_images=minimum_images,
    )


def _parse_predicates(
    explicit_types: str, description: str
) -> tuple[RequirementType, ...]:
    if explicit_types:
        parsed: list[RequirementType] = []
        for raw in re.split(r"[,;|]", explicit_types):
            normalized = _normalize_column(raw)
            try:
                predicate = RequirementType(normalized)
            except ValueError as exc:
                raise EvidenceRequirementsError(
                    f"unknown requirement_type: {raw.strip()}"
                ) from exc
            if predicate not in parsed:
                parsed.append(predicate)
        return tuple(parsed)

    text = _normalize_value(description)
    predicates: list[RequirementType] = [RequirementType.VALID_IMAGE]
    if _contains_any(
        text,
        "correct object",
        "claimed object",
        "object visible",
        "item visible",
        "full object",
    ):
        predicates.append(RequirementType.CORRECT_OBJECT)
    if _contains_any(text, "claimed damage", "damage type", "specific damage"):
        predicates.append(RequirementType.CLAIMED_ISSUE_VISIBLE)
    elif "damage" in text:
        predicates.append(RequirementType.DAMAGE_VISIBLE)
    if _contains_any(
        text,
        "object part",
        "claimed part",
        "damaged part",
        "damaged area",
        "affected area",
    ):
        predicates.append(RequirementType.CLAIMED_PART_VISIBLE)
    if _contains_any(text, "clear image", "clear photo", "image quality", "well lit", "in focus"):
        predicates.append(RequirementType.ACCEPTABLE_QUALITY)
    if _contains_any(text, "unedited", "not manipulated", "no manipulation", "authentic image"):
        predicates.append(RequirementType.NO_MANIPULATION)
    has_image_count = _extract_image_count(text) is not None
    if has_image_count:
        predicates.append(RequirementType.MINIMUM_IMAGES)

    # A description that has no evaluable visual meaning is never guessed true.
    explicit_valid_image = _contains_any(
        text,
        "valid image",
        "usable image",
        "readable image",
        "image required",
        "photo required",
    )
    if predicates == [RequirementType.VALID_IMAGE] and not (
        has_image_count or explicit_valid_image
    ):
        return (RequirementType.UNKNOWN,)
    return tuple(dict.fromkeys(predicates))


def _evaluate_requirement(
    requirement: EvidenceRequirement,
    *,
    findings: VisionAnalysis,
    issue_type: IssueType,
    supporting_ids: list[str],
) -> RequirementEvaluation:
    checks: list[tuple[bool, str]] = []
    for predicate in requirement.predicates:
        if predicate is RequirementType.VALID_IMAGE:
            checks.append((findings.valid_image, "no valid image"))
        elif predicate is RequirementType.CORRECT_OBJECT:
            checks.append((findings.object_type_verified, "claimed object not verified"))
        elif predicate is RequirementType.DAMAGE_VISIBLE:
            checks.append((findings.damage_clearly_visible, "damage not clearly visible"))
        elif predicate is RequirementType.CLAIMED_ISSUE_VISIBLE:
            checks.append((bool(supporting_ids), f"{issue_type.value} not clearly visible"))
        elif predicate is RequirementType.CLAIMED_PART_VISIBLE:
            visible = any(
                item.valid_image
                and item.object_matches_claim
                and _part_matches(findings.object_part, item.object_parts)
                for item in findings.image_analyses
            )
            checks.append((visible, f"claimed part '{findings.object_part}' not visible"))
        elif predicate is RequirementType.ACCEPTABLE_QUALITY:
            clear = any(_is_clear_relevant_image(item) for item in findings.image_analyses)
            checks.append((clear, "no clear relevant image"))
        elif predicate is RequirementType.MINIMUM_IMAGES:
            count = sum(
                item.valid_image and item.object_matches_claim
                for item in findings.image_analyses
            )
            checks.append(
                (
                    count >= requirement.minimum_images,
                    f"requires {requirement.minimum_images} usable image(s); found {count}",
                )
            )
        elif predicate is RequirementType.NO_MANIPULATION:
            no_concern = findings.valid_image and VisionFlag.POSSIBLE_MANIPULATION not in findings.flags
            checks.append((no_concern, "possible manipulation flagged"))
        else:
            checks.append((False, "requirement is not machine-evaluable"))

    failures = [reason for passed, reason in checks if not passed]
    return RequirementEvaluation(
        requirement=requirement,
        met=not failures,
        reason="satisfied" if not failures else "; ".join(failures),
    )


def _supporting_images(
    findings: VisionAnalysis, issue_type: IssueType
) -> list[str]:
    return [
        item.image_id
        for item in findings.image_analyses
        if item.valid_image
        and item.object_matches_claim
        and item.damage_clearly_visible
        and issue_type in item.issue_types
    ]


def _contradicting_images(
    findings: VisionAnalysis, issue_type: IssueType
) -> list[str]:
    if issue_type in {IssueType.NONE, IssueType.UNKNOWN}:
        return []
    return [
        item.image_id
        for item in findings.image_analyses
        if _is_clear_relevant_image(item)
        and not item.damage_clearly_visible
        and IssueType.NONE in item.issue_types
    ]


def _is_clear_relevant_image(item: ImageAnalysis) -> bool:
    obscuring_flags = {
        VisionFlag.BLURRY_IMAGE,
        VisionFlag.CROPPED_OR_OBSTRUCTED,
        VisionFlag.LOW_LIGHT_OR_GLARE,
        VisionFlag.WRONG_ANGLE,
        VisionFlag.WRONG_OBJECT,
        VisionFlag.WRONG_OBJECT_PART,
    }
    return (
        item.valid_image
        and item.object_matches_claim
        and item.image_quality in {ImageQuality.GOOD, ImageQuality.ACCEPTABLE}
        and not obscuring_flags.intersection(item.flags)
    )


def _part_matches(claimed_part: str, visible_parts: list[str]) -> bool:
    claimed = set(_normalize_value(claimed_part).split())
    if not claimed or claimed == {"unknown"}:
        return False
    return any(
        claimed.issubset(set(_normalize_value(part).split()))
        or set(_normalize_value(part).split()).issubset(claimed)
        for part in visible_parts
        if _normalize_value(part) != "unknown"
    )


def _build_reason(
    evaluations: list[RequirementEvaluation],
    standard_met: bool,
    supporting_ids: list[str],
) -> str:
    if standard_met:
        image_text = ", ".join(supporting_ids) if supporting_ids else "valid evidence"
        reason = f"Met all {len(evaluations)} requirement(s); supported by {image_text}."
    else:
        failures = [
            f"{item.requirement.description}: {item.reason}"
            for item in evaluations
            if not item.met
        ]
        reason = "Not met; " + " | ".join(failures)
    return reason if len(reason) <= MAX_REASON_LENGTH else reason[:497].rstrip() + "..."


def _evidence_confidence(
    findings: VisionAnalysis, evaluations: list[RequirementEvaluation]
) -> float:
    if not findings.valid_image:
        return 0.0
    evaluable = all(
        RequirementType.UNKNOWN not in item.requirement.predicates
        for item in evaluations
    )
    if not evaluable:
        return 0.0
    relevant = [
        item.analysis_confidence
        for item in findings.image_analyses
        if item.valid_image and item.object_matches_claim
    ]
    return round(sum(relevant) / len(relevant), 4) if relevant else 0.0


def _parse_minimum_images(raw: str, description: str, *, csv_row: int) -> int:
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise EvidenceRequirementsError(
                f"minimum_images must be an integer on CSV row {csv_row}"
            ) from exc
        if value < 1:
            raise EvidenceRequirementsError(
                f"minimum_images must be at least 1 on CSV row {csv_row}"
            )
        return value
    return _extract_image_count(_normalize_value(description)) or 1


def _extract_image_count(text: str) -> int | None:
    number_words = "|".join(NUMBER_WORDS)
    match = re.search(
        rf"\b(?:at least\s+)?(\d+|{number_words})\s+"
        rf"(?:clear\s+)?(?:images?|photos?)\b",
        text,
    )
    if not match:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def _find_requirement_column(frame: pd.DataFrame) -> str:
    matches = [name for name in REQUIREMENT_COLUMN_ALIASES if name in frame.columns]
    if not matches:
        accepted = ", ".join(REQUIREMENT_COLUMN_ALIASES)
        raise EvidenceRequirementsError(
            f"requirements CSV needs one description column: {accepted}"
        )
    if len(matches) > 1:
        raise EvidenceRequirementsError(
            f"requirements CSV has ambiguous description columns: {', '.join(matches)}"
        )
    return matches[0]


def _validate_preloaded_requirements(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*REQUIRED_COLUMNS, "requirement"}
    missing = required.difference(frame.columns)
    if missing:
        raise EvidenceRequirementsError(
            "preloaded requirements are missing columns: "
            + ", ".join(sorted(missing))
        )
    return frame.copy()


def _parse_enum(enum_type: type, value: object, field_name: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _normalize_column(value: object) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(value).strip().lower()
    )
    return re.sub(r"_+", "_", normalized).strip("_")


def _normalize_value(value: object) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).split()
    )


def _contains_any(text: str, *phrases: str) -> bool:
    return any(phrase in text for phrase in phrases)
