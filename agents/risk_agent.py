"""Deterministic user-history risk context for claim verification.

This module intentionally has no access to image evidence or final claim status.
Its output is context for review prioritization and must never override images.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from models.schemas import RiskAssessment, RiskFlag, RiskLevel


USER_ID_ALIASES: Final[tuple[str, ...]] = ("user_id", "userid", "customer_id")
STATUS_ALIASES: Final[tuple[str, ...]] = (
    "claim_status",
    "status",
    "outcome",
    "decision",
)
DATE_ALIASES: Final[tuple[str, ...]] = (
    "claim_date",
    "date",
    "created_at",
    "submitted_at",
    "timestamp",
)
HISTORY_FLAG_ALIASES: Final[tuple[str, ...]] = (
    "suspicious_history_flags",
    "history_flags",
    "risk_flags",
    "suspicious_flag",
    "is_suspicious",
    "fraud_flag",
)
TOTAL_CLAIMS_ALIASES: Final[tuple[str, ...]] = (
    "total_claims",
    "claim_count",
    "historical_claim_count",
)
REJECTED_CLAIMS_ALIASES: Final[tuple[str, ...]] = (
    "rejected_claims",
    "rejection_count",
    "denied_claims",
)
REJECTION_RATE_ALIASES: Final[tuple[str, ...]] = (
    "rejection_rate",
    "rejected_claim_rate",
)
RECENT_CLAIMS_ALIASES: Final[tuple[str, ...]] = (
    "claims_last_30_days",
    "recent_claim_count",
)

REJECTED_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "rejected",
        "denied",
        "declined",
        "contradicted",
        "not supported",
        "fraud",
        "fraudulent",
    }
)
ACCEPTED_STATUSES: Final[frozenset[str]] = frozenset(
    {"supported", "approved", "accepted", "paid"}
)
BENIGN_FLAG_VALUES: Final[frozenset[str]] = frozenset(
    {"", "0", "false", "no", "none", "clean", "normal", "n a", "na", "[]"}
)


class RiskAgentError(RuntimeError):
    """Base error raised while producing history context."""


class UserHistoryError(RiskAgentError):
    """Raised when user_history.csv cannot be loaded or validated."""


@dataclass(frozen=True, slots=True)
class RiskThresholds:
    """Versionable thresholds for deterministic scoring."""

    minimum_claims_for_rejection_rate: int = 3
    high_rejection_rate: float = 0.60
    frequency_window_days: int = 30
    abnormal_claims_in_window: int = 4
    user_history_risk_score: float = 0.40
    manual_review_score: float = 0.65
    rejection_weight: float = 0.45
    frequency_weight: float = 0.40
    suspicious_history_weight: float = 0.65

    def __post_init__(self) -> None:
        if self.minimum_claims_for_rejection_rate < 1:
            raise ValueError("minimum_claims_for_rejection_rate must be positive")
        if self.frequency_window_days < 1 or self.abnormal_claims_in_window < 1:
            raise ValueError("frequency thresholds must be positive")
        bounded = (
            self.high_rejection_rate,
            self.user_history_risk_score,
            self.manual_review_score,
            self.rejection_weight,
            self.frequency_weight,
            self.suspicious_history_weight,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("risk rates, scores, and weights must be within [0, 1]")
        if self.user_history_risk_score > self.manual_review_score:
            raise ValueError("history-risk threshold cannot exceed review threshold")


DEFAULT_THRESHOLDS: Final[RiskThresholds] = RiskThresholds()


@dataclass(frozen=True, slots=True)
class HistoryMetrics:
    historical_claim_count: int
    rejection_rate: float | None
    rejection_sample_size: int
    max_claims_in_window: int | None
    suspicious_history: bool
    suspicious_values: tuple[str, ...]


def load_user_history(csv_path: str | Path) -> pd.DataFrame:
    """Load user history while preserving IDs as strings."""

    path = Path(csv_path).expanduser()
    if not path.exists():
        raise UserHistoryError(f"user history CSV does not exist: {path}")
    if not path.is_file():
        raise UserHistoryError(f"user history path is not a file: {path}")

    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise UserHistoryError(f"unable to read user history CSV: {path}") from exc

    normalized = [_normalize_column(column) for column in frame.columns]
    if len(set(normalized)) != len(normalized):
        raise UserHistoryError(
            "user history CSV contains duplicate columns after normalization"
        )
    frame.columns = normalized
    user_column = _find_single_column(frame, USER_ID_ALIASES, required=True)
    if user_column != "user_id":
        frame = frame.rename(columns={user_column: "user_id"})
    for column in frame.columns:
        frame[column] = frame[column].map(lambda value: str(value).strip())
    if frame["user_id"].eq("").any():
        rows = frame.index[frame["user_id"].eq("")].tolist()
        labels = ", ".join(str(index + 2) for index in rows[:5])
        raise UserHistoryError(f"user_id is blank on CSV row(s): {labels}")
    return frame


def assess_user_risk(
    user_id: str,
    user_history_csv: str | Path | pd.DataFrame,
    *,
    thresholds: RiskThresholds = DEFAULT_THRESHOLDS,
) -> RiskAssessment:
    """Return historical risk context without making a claim decision."""

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    normalized_user_id = user_id.strip()
    if isinstance(user_history_csv, pd.DataFrame):
        if "user_id" not in user_history_csv.columns:
            raise UserHistoryError("preloaded user history is missing user_id")
        frame = user_history_csv.copy()
    else:
        frame = load_user_history(user_history_csv)
    user_rows = frame.loc[frame["user_id"] == normalized_user_id].copy()

    if user_rows.empty:
        return RiskAssessment(
            user_id=normalized_user_id,
            risk_flags=[RiskFlag.NO_HISTORY],
            risk_level=RiskLevel.LOW,
            risk_score=0.0,
            assessment_reason=(
                "No prior history found; no history-based risk signal. "
                "Image evidence remains authoritative."
            ),
            historical_claim_count=0,
        )

    metrics = _calculate_metrics(user_rows, thresholds)
    score = 0.0
    flags: list[RiskFlag] = []
    explanations: list[str] = []

    rejection_triggered = (
        metrics.rejection_rate is not None
        and metrics.rejection_sample_size
        >= thresholds.minimum_claims_for_rejection_rate
        and metrics.rejection_rate >= thresholds.high_rejection_rate
    )
    if rejection_triggered:
        score += thresholds.rejection_weight
        flags.append(RiskFlag.HIGH_REJECTION_RATE)
        explanations.append(
            f"rejection rate {metrics.rejection_rate:.0%} "
            f"across {metrics.rejection_sample_size} decided claim(s)"
        )

    frequency_triggered = (
        metrics.max_claims_in_window is not None
        and metrics.max_claims_in_window >= thresholds.abnormal_claims_in_window
    )
    if frequency_triggered:
        score += thresholds.frequency_weight
        flags.append(RiskFlag.ABNORMAL_CLAIM_FREQUENCY)
        explanations.append(
            f"{metrics.max_claims_in_window} claim(s) within a "
            f"{thresholds.frequency_window_days}-day window"
        )

    if metrics.suspicious_history:
        score += thresholds.suspicious_history_weight
        flags.append(RiskFlag.SUSPICIOUS_HISTORY)
        labels = ", ".join(metrics.suspicious_values[:3])
        explanations.append(f"history flag(s): {labels}")

    score = round(min(score, 1.0), 4)
    if score >= thresholds.user_history_risk_score:
        flags.append(RiskFlag.USER_HISTORY_RISK)
    if score >= thresholds.manual_review_score:
        flags.append(RiskFlag.MANUAL_REVIEW_REQUIRED)

    risk_level = _risk_level(score, thresholds)
    if not flags:
        flags = [RiskFlag.NONE]
        explanations.append("no configured history threshold was exceeded")

    unavailable: list[str] = []
    if metrics.rejection_rate is None:
        unavailable.append("rejection rate unavailable")
    if metrics.max_claims_in_window is None:
        unavailable.append("frequency unavailable")
    explanation = "; ".join([*explanations, *unavailable])

    return RiskAssessment(
        user_id=normalized_user_id,
        risk_flags=_unique_flags(flags),
        risk_level=risk_level,
        risk_score=score,
        assessment_reason=(
            f"History context: {explanation}. Image evidence remains authoritative."
        ),
        historical_claim_count=metrics.historical_claim_count,
    )


def _calculate_metrics(
    rows: pd.DataFrame, thresholds: RiskThresholds
) -> HistoryMetrics:
    total_column = _find_single_column(rows, TOTAL_CLAIMS_ALIASES)
    rejected_column = _find_single_column(rows, REJECTED_CLAIMS_ALIASES)
    rate_column = _find_single_column(rows, REJECTION_RATE_ALIASES)
    status_column = _find_single_column(rows, STATUS_ALIASES)
    date_column = _find_single_column(rows, DATE_ALIASES)
    recent_column = _find_single_column(rows, RECENT_CLAIMS_ALIASES)
    flag_columns = _find_all_columns(rows, HISTORY_FLAG_ALIASES)

    aggregated_total = _sum_nonnegative_integers(rows, total_column)
    historical_count = aggregated_total if aggregated_total is not None else len(rows)

    rejection_rate: float | None = None
    rejection_sample_size = 0
    aggregated_rejected = _sum_nonnegative_integers(rows, rejected_column)
    if aggregated_total is not None and aggregated_total > 0 and aggregated_rejected is not None:
        rejection_sample_size = aggregated_total
        rejection_rate = min(aggregated_rejected / aggregated_total, 1.0)
    elif rate_column:
        rates = [_parse_rate(value, rate_column) for value in rows[rate_column] if value]
        if rates:
            rejection_rate = sum(rates) / len(rates)
            rejection_sample_size = aggregated_total or len(rows)
    elif status_column:
        decided_statuses = REJECTED_STATUSES.union(ACCEPTED_STATUSES)
        statuses = [
            normalized
            for value in rows[status_column]
            if value and (normalized := _normalize_value(value)) in decided_statuses
        ]
        if statuses:
            rejection_sample_size = len(statuses)
            rejection_rate = sum(status in REJECTED_STATUSES for status in statuses) / len(statuses)

    max_claims_in_window: int | None = None
    if recent_column:
        values = [_parse_nonnegative_int(value, recent_column) for value in rows[recent_column] if value]
        if values:
            max_claims_in_window = max(values)
    elif date_column:
        parsed_dates = pd.to_datetime(rows[date_column], errors="coerce", utc=True)
        valid_dates = sorted(date for date in parsed_dates if not pd.isna(date))
        if valid_dates:
            max_claims_in_window = _maximum_window_count(
                valid_dates, thresholds.frequency_window_days
            )

    suspicious_values = _suspicious_values(rows, flag_columns)
    return HistoryMetrics(
        historical_claim_count=historical_count,
        rejection_rate=rejection_rate,
        rejection_sample_size=rejection_sample_size,
        max_claims_in_window=max_claims_in_window,
        suspicious_history=bool(suspicious_values),
        suspicious_values=tuple(suspicious_values),
    )


def _maximum_window_count(dates: list[pd.Timestamp], window_days: int) -> int:
    left = 0
    maximum = 0
    window = pd.Timedelta(days=window_days)
    for right, current in enumerate(dates):
        while current - dates[left] > window:
            left += 1
        maximum = max(maximum, right - left + 1)
    return maximum


def _suspicious_values(rows: pd.DataFrame, columns: list[str]) -> list[str]:
    values: list[str] = []
    for column in columns:
        for raw in rows[column]:
            for token in re.split(r"[,;|]", raw):
                normalized = _normalize_value(token)
                if normalized not in BENIGN_FLAG_VALUES and normalized not in values:
                    values.append(normalized)
    return values


def _risk_level(score: float, thresholds: RiskThresholds) -> RiskLevel:
    if score >= thresholds.manual_review_score:
        return RiskLevel.HIGH
    if score >= thresholds.user_history_risk_score:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _find_single_column(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
    *,
    required: bool = False,
) -> str | None:
    matches = [alias for alias in aliases if alias in frame.columns]
    if len(matches) > 1:
        raise UserHistoryError(
            f"ambiguous equivalent columns: {', '.join(matches)}"
        )
    if not matches:
        if required:
            raise UserHistoryError(
                f"user history CSV needs one of: {', '.join(aliases)}"
            )
        return None
    return matches[0]


def _find_all_columns(frame: pd.DataFrame, aliases: tuple[str, ...]) -> list[str]:
    return [alias for alias in aliases if alias in frame.columns]


def _sum_nonnegative_integers(
    rows: pd.DataFrame, column: str | None
) -> int | None:
    if not column:
        return None
    values = [_parse_nonnegative_int(value, column) for value in rows[column] if value]
    return sum(values) if values else None


def _parse_nonnegative_int(value: str, column: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise UserHistoryError(f"column '{column}' must contain integers") from exc
    if number < 0:
        raise UserHistoryError(f"column '{column}' cannot contain negatives")
    return number


def _parse_rate(value: str, column: str) -> float:
    text = value.strip()
    try:
        rate = float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    except ValueError as exc:
        raise UserHistoryError(f"column '{column}' contains an invalid rate") from exc
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise UserHistoryError(f"column '{column}' rates must be within [0, 1]")
    return rate


def _unique_flags(flags: list[RiskFlag]) -> list[RiskFlag]:
    return list(dict.fromkeys(flags))


def _normalize_column(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def _normalize_value(value: object) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).split()
    )
