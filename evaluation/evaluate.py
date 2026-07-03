"""Evaluate claim-verification predictions against sample_claims.csv labels."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Mapping, Sequence

import pandas as pd


EVALUATED_FIELDS: Final[tuple[str, ...]] = (
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
)
LABEL_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "claim_status": (
        "expected_claim_status",
        "label_claim_status",
        "claim_status_label",
        "ground_truth_claim_status",
        "expected_status",
        "claim_status",
    ),
    "issue_type": (
        "expected_issue_type",
        "label_issue_type",
        "issue_type_label",
        "ground_truth_issue_type",
        "issue_type",
    ),
    "object_part": (
        "expected_object_part",
        "label_object_part",
        "object_part_label",
        "ground_truth_object_part",
        "object_part",
    ),
    "severity": (
        "expected_severity",
        "label_severity",
        "severity_label",
        "ground_truth_severity",
        "severity",
    ),
}
KEY_CANDIDATES: Final[tuple[str, ...]] = (
    "user_id",
    "user_claim",
    "claim_object",
)
MAX_CONFUSIONS: Final[int] = 10
MAX_ERROR_EXAMPLES: Final[int] = 20


class EvaluationError(RuntimeError):
    """Raised when labels and predictions cannot be evaluated reliably."""


@dataclass(frozen=True, slots=True)
class MetricResult:
    field: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def errors(self) -> int:
        return self.total - self.correct


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: tuple[MetricResult, ...]
    exact_match_count: int
    label_count: int
    prediction_count: int
    matched_count: int
    missing_prediction_count: int
    extra_prediction_count: int
    key_columns: tuple[str, ...]
    comparisons: pd.DataFrame


def load_csv(csv_path: str | Path, *, role: str) -> pd.DataFrame:
    """Load a CSV as strings and normalize its column names."""

    path = Path(csv_path).expanduser()
    if not path.exists() or not path.is_file():
        raise EvaluationError(f"{role} CSV does not exist or is not a file: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise EvaluationError(f"unable to read {role} CSV: {path}") from exc
    normalized = [_normalize_column(column) for column in frame.columns]
    if len(set(normalized)) != len(normalized):
        raise EvaluationError(
            f"{role} CSV has duplicate columns after normalization"
        )
    frame.columns = normalized
    return frame


def evaluate(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
) -> EvaluationResult:
    """Align prediction rows to labels and compute the four required metrics."""

    if labels.empty:
        raise EvaluationError("sample_claims.csv contains no labeled rows")
    label_columns = {
        field: _resolve_label_column(labels, field) for field in EVALUATED_FIELDS
    }
    missing_prediction_columns = set(EVALUATED_FIELDS).difference(predictions.columns)
    if missing_prediction_columns:
        raise EvaluationError(
            "predictions are missing columns: "
            + ", ".join(sorted(missing_prediction_columns))
        )

    key_columns = _select_key_columns(labels, predictions)
    prepared_labels = _prepare_for_join(labels, key_columns, prefix="label")
    prepared_predictions = _prepare_for_join(
        predictions, key_columns, prefix="prediction"
    )

    label_projection = [
        *key_columns,
        "_occurrence",
        "_label_row",
        *label_columns.values(),
    ]
    label_projection = list(dict.fromkeys(label_projection))
    prediction_projection = [
        *key_columns,
        "_occurrence",
        "_prediction_row",
        *EVALUATED_FIELDS,
    ]
    merged = prepared_labels[label_projection].merge(
        prepared_predictions[prediction_projection],
        how="left",
        on=[*key_columns, "_occurrence"],
        suffixes=("_label", "_prediction"),
        indicator=True,
        validate="one_to_one",
    )

    comparisons = pd.DataFrame(
        {
            "label_row": merged["_label_row"],
            "prediction_row": merged["_prediction_row"].fillna(""),
            "match_state": merged["_merge"].map(
                {"both": "matched", "left_only": "missing_prediction"}
            ),
        }
    )
    for column in key_columns:
        comparisons[column] = merged[column]

    metrics: list[MetricResult] = []
    correctness_columns: list[str] = []
    for field in EVALUATED_FIELDS:
        label_column = label_columns[field]
        expected_source = _merged_column_name(
            label_column, field, side="label", merged=merged
        )
        predicted_source = _merged_column_name(
            field, label_column, side="prediction", merged=merged
        )
        expected = merged[expected_source].map(lambda value: _canonicalize(field, value))
        predicted = merged[predicted_source].fillna("").map(
            lambda value: _canonicalize(field, value)
        )
        correct = expected.eq(predicted) & merged["_merge"].eq("both")
        comparisons[f"expected_{field}"] = expected
        comparisons[f"predicted_{field}"] = predicted.where(
            merged["_merge"].eq("both"), "<missing>"
        )
        comparisons[f"correct_{field}"] = correct
        correctness_columns.append(f"correct_{field}")
        metrics.append(
            MetricResult(
                field=field,
                correct=int(correct.sum()),
                total=len(labels),
            )
        )

    exact_match = comparisons[correctness_columns].all(axis=1)
    comparisons["exact_match"] = exact_match
    matched_count = int(merged["_merge"].eq("both").sum())
    extra_count = _count_extra_predictions(
        prepared_labels, prepared_predictions, key_columns
    )
    return EvaluationResult(
        metrics=tuple(metrics),
        exact_match_count=int(exact_match.sum()),
        label_count=len(labels),
        prediction_count=len(predictions),
        matched_count=matched_count,
        missing_prediction_count=len(labels) - matched_count,
        extra_prediction_count=extra_count,
        key_columns=tuple(key_columns),
        comparisons=comparisons,
    )


def generate_report(result: EvaluationResult) -> str:
    """Render a deterministic Markdown evaluation report."""

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Damage Claim Verification Evaluation Report",
        "",
        f"Generated: {generated_at}",
        "",
        "## Dataset summary",
        "",
        f"- Labeled claims: {result.label_count}",
        f"- Prediction rows: {result.prediction_count}",
        f"- Matched predictions: {result.matched_count}",
        f"- Missing predictions: {result.missing_prediction_count}",
        f"- Extra predictions: {result.extra_prediction_count}",
        f"- Matching keys: {', '.join(result.key_columns)}",
        "",
        "## 1. Metrics",
        "",
        "| Metric | Correct | Total | Accuracy | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in result.metrics:
        lines.append(
            f"| {_display_name(metric.field)} | {metric.correct} | "
            f"{metric.total} | {metric.accuracy:.2%} | {metric.errors} |"
        )
    exact_accuracy = result.exact_match_count / result.label_count
    lines.extend(
        [
            f"| Exact match (all four fields) | {result.exact_match_count} | "
            f"{result.label_count} | {exact_accuracy:.2%} | "
            f"{result.label_count - result.exact_match_count} |",
            "",
            "Missing predictions are counted as incorrect for every metric. "
            "Extra predictions are reported separately and do not increase the denominator.",
            "",
            "## 2. Error analysis",
            "",
        ]
    )
    lines.extend(_render_confusions(result))
    lines.extend(_render_error_examples(result))
    lines.extend(
        [
            "## 3. Common failure modes",
            "",
            *_common_failure_modes(result),
            "",
            "## 4. Improvement recommendations",
            "",
            *_improvement_recommendations(result),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: str, report_path: str | Path) -> None:
    """Atomically persist the report."""

    path = Path(report_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(report, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render_confusions(result: EvaluationResult) -> list[str]:
    lines: list[str] = []
    for field in EVALUATED_FIELDS:
        errors = result.comparisons.loc[~result.comparisons[f"correct_{field}"]]
        lines.extend([f"### {_display_name(field)} confusions", ""])
        if errors.empty:
            lines.extend(["No errors.", ""])
            continue
        pairs = Counter(
            zip(
                errors[f"expected_{field}"],
                errors[f"predicted_{field}"],
            )
        )
        lines.extend(
            [
                "| Expected | Predicted | Count |",
                "|---|---|---:|",
            ]
        )
        for (expected, predicted), count in pairs.most_common(MAX_CONFUSIONS):
            lines.append(
                f"| {_escape(expected)} | {_escape(predicted)} | {count} |"
            )
        lines.append("")
    return lines


def _render_error_examples(result: EvaluationResult) -> list[str]:
    errors = result.comparisons.loc[~result.comparisons["exact_match"]].head(
        MAX_ERROR_EXAMPLES
    )
    lines = ["### Representative row-level errors", ""]
    if errors.empty:
        return [*lines, "No row-level errors.", ""]
    identity = result.key_columns[0]
    lines.extend(
        [
            f"| Label row | {_display_name(identity)} | Field | Expected | Predicted |",
            "|---:|---|---|---|---|",
        ]
    )
    for _, row in errors.iterrows():
        for field in EVALUATED_FIELDS:
            if not bool(row[f"correct_{field}"]):
                lines.append(
                    f"| {int(row['label_row'])} | {_escape(row[identity])} | "
                    f"{_display_name(field)} | "
                    f"{_escape(row[f'expected_{field}'])} | "
                    f"{_escape(row[f'predicted_{field}'])} |"
                )
    lines.append("")
    return lines


def _common_failure_modes(result: EvaluationResult) -> list[str]:
    metric_by_field = {metric.field: metric for metric in result.metrics}
    modes: list[str] = []
    if result.missing_prediction_count:
        modes.append(
            f"- Pipeline coverage: {result.missing_prediction_count} labeled claim(s) "
            "have no prediction."
        )

    status_errors = result.comparisons.loc[
        ~result.comparisons["correct_claim_status"]
    ]
    predicted_nei = int(
        status_errors["predicted_claim_status"].eq("not_enough_information").sum()
    )
    if predicted_nei:
        modes.append(
            f"- Conservative abstention: {predicted_nei} status error(s) predict "
            "`not_enough_information`."
        )
    polarity_errors = int(
        (
            status_errors["expected_claim_status"].isin(
                {"supported", "contradicted"}
            )
            & status_errors["predicted_claim_status"].isin(
                {"supported", "contradicted"}
            )
        ).sum()
    )
    if polarity_errors:
        modes.append(
            f"- Evidence polarity: {polarity_errors} claim(s) confuse supported "
            "with contradicted."
        )

    unknown_issue_errors = int(
        (
            ~result.comparisons["correct_issue_type"]
            & result.comparisons["predicted_issue_type"].eq("unknown")
        ).sum()
    )
    if unknown_issue_errors:
        modes.append(
            f"- Damage classification uncertainty: {unknown_issue_errors} issue "
            "error(s) resolve to `unknown`."
        )
    unknown_part_errors = int(
        (
            ~result.comparisons["correct_object_part"]
            & result.comparisons["predicted_object_part"].eq("unknown")
        ).sum()
    )
    if unknown_part_errors:
        modes.append(
            f"- Part localization uncertainty: {unknown_part_errors} part error(s) "
            "resolve to `unknown`."
        )

    severity_errors = metric_by_field["severity"].errors
    if severity_errors:
        modes.append(
            f"- Severity calibration: {severity_errors} claim(s) use the wrong "
            "severity band."
        )
    if not modes:
        modes.append("- No common failure mode was observed in this evaluation set.")
    return modes


def _improvement_recommendations(result: EvaluationResult) -> list[str]:
    metric_by_field = {metric.field: metric for metric in result.metrics}
    ranked = sorted(result.metrics, key=lambda metric: (metric.accuracy, metric.field))
    recommendations: list[str] = []
    for metric in ranked:
        if not metric.errors:
            continue
        if metric.field == "claim_status":
            text = (
                "Add regression cases for supported/contradicted/insufficient "
                "boundaries and audit whether evidence requirements or visual "
                "quality gates caused each status error."
            )
        elif metric.field == "issue_type":
            text = (
                "Add object-specific visual examples for the most frequent issue "
                "confusions and reinforce the rule that ambiguous marks remain unknown."
            )
        elif metric.field == "object_part":
            text = (
                "Introduce a canonical part vocabulary per object and normalize "
                "synonyms before comparing claim and image findings."
            )
        else:
            text = (
                "Calibrate severity with labeled visual anchors for none, low, "
                "medium, and high damage, then add boundary-focused tests."
            )
        recommendations.append(f"{len(recommendations) + 1}. {text}")
    if result.missing_prediction_count:
        recommendations.append(
            f"{len(recommendations) + 1}. Investigate row-level pipeline failures "
            "and resume/checkpoint behavior to restore full prediction coverage."
        )
    if result.extra_prediction_count:
        recommendations.append(
            f"{len(recommendations) + 1}. Enforce one output row per input claim "
            "and remove stale or duplicate predictions before evaluation."
        )
    if not recommendations:
        recommendations.append(
            "1. Preserve the current behavior with a regression test for every labeled row."
        )
    worst = ranked[0]
    recommendations.append(
        f"{len(recommendations) + 1}. Prioritize `{worst.field}` in the next iteration "
        f"({worst.accuracy:.2%} accuracy), then rerun this evaluation unchanged."
    )
    return recommendations


def _resolve_label_column(frame: pd.DataFrame, field: str) -> str:
    matches = [alias for alias in LABEL_ALIASES[field] if alias in frame.columns]
    if not matches:
        raise EvaluationError(
            f"labels do not contain a recognized column for {field}: "
            + ", ".join(LABEL_ALIASES[field])
        )
    return matches[0]


def _select_key_columns(
    labels: pd.DataFrame, predictions: pd.DataFrame
) -> list[str]:
    if "claim_id" in labels.columns and "claim_id" in predictions.columns:
        return ["claim_id"]
    common = [
        column
        for column in KEY_CANDIDATES
        if column in labels.columns and column in predictions.columns
    ]
    if "user_id" not in common:
        raise EvaluationError(
            "labels and predictions need a shared claim_id or user_id for alignment"
        )
    return common


def _prepare_for_join(
    frame: pd.DataFrame, key_columns: list[str], *, prefix: str
) -> pd.DataFrame:
    prepared = frame.copy()
    for column in key_columns:
        prepared[column] = prepared[column].map(
            lambda value: _canonicalize_key(column, value)
        )
    prepared["_occurrence"] = prepared.groupby(
        key_columns, dropna=False, sort=False
    ).cumcount()
    prepared[f"_{prefix}_row"] = range(2, len(prepared) + 2)
    return prepared


def _count_extra_predictions(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    key_columns: list[str],
) -> int:
    joined = predictions[[*key_columns, "_occurrence"]].merge(
        labels[[*key_columns, "_occurrence"]],
        how="left",
        on=[*key_columns, "_occurrence"],
        indicator=True,
        validate="one_to_one",
    )
    return int(joined["_merge"].eq("left_only").sum())


def _merged_column_name(
    primary: str,
    conflicting: str,
    *,
    side: str,
    merged: pd.DataFrame,
) -> str:
    candidate = f"{primary}_{side}" if primary == conflicting else primary
    if candidate not in merged.columns:
        raise EvaluationError(f"internal comparison column missing: {candidate}")
    return candidate


def _canonicalize(field: str, value: object) -> str:
    text = str(value).strip().lower()
    if field == "object_part":
        return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split()) or "<blank>"
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text)).strip("_") or "<blank>"


def _canonicalize_key(field: str, value: object) -> str:
    text = " ".join(str(value).strip().split())
    return text.lower() if field in {"claim_object"} else text


def _normalize_column(value: object) -> str:
    return re.sub(
        r"_+",
        "_",
        re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()),
    ).strip("_")


def _display_name(value: str) -> str:
    return value.replace("_", " ").title()


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate output.csv against sample_claims.csv labels."
    )
    parser.add_argument("--labels", default="sample_claims.csv")
    parser.add_argument("--predictions", default="output.csv")
    parser.add_argument(
        "--report",
        default=str(Path("evaluation") / "evaluation_report.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        labels = load_csv(args.labels, role="labels")
        predictions = load_csv(args.predictions, role="predictions")
        result = evaluate(labels, predictions)
        write_report(generate_report(result), args.report)
        print(f"Evaluation report written to {args.report}")
        return 0
    except EvaluationError as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
