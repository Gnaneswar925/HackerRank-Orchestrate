"""Batch entry point for multimodal damage-claim verification."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pandas as pd
from openai import OpenAI

from agents.claim_agent import (
    DEFAULT_MODEL as DEFAULT_CLAIM_MODEL,
    ClaimExtractionError,
    extract_claim,
)
from agents.decision_agent import make_decision
from agents.evidence_agent import evaluate_evidence, load_evidence_requirements
from agents.risk_agent import assess_user_risk, load_user_history
from agents.vision_agent import (
    DEFAULT_MODEL as DEFAULT_VISION_MODEL,
    VisionAnalysisError,
    analyze_images,
)
from models.schemas import ClaimObject, RiskFlag, Severity
from utils.csv_utils import (
    ClaimRecord,
    atomic_write_output,
    decision_to_row,
    load_claims,
    load_existing_output,
    output_row_key,
)


LOGGER = logging.getLogger("claim_verifier")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    claim_model: str
    vision_model: str
    agent_attempts: int
    retry_base_seconds: float
    max_dimension: int


def process_claim(
    record: ClaimRecord,
    *,
    client: OpenAI,
    requirements: pd.DataFrame,
    user_history: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, object]:
    """Run one claim through all five agents in the required order."""

    claim = _with_retry(
        lambda: extract_claim(
            record.user_claim,
            record.claim_object,
            client=client,
            model=config.claim_model,
        ),
        operation="claim extraction",
        attempts=config.agent_attempts,
        base_seconds=config.retry_base_seconds,
        retry_exceptions=(ClaimExtractionError,),
        user_id=record.user_id,
    )
    vision = _with_retry(
        lambda: analyze_images(
            record.resolved_image_paths,
            claim.claim_object,
            claim,
            client=client,
            model=config.vision_model,
            max_dimension=config.max_dimension,
        ),
        operation="vision analysis",
        attempts=config.agent_attempts,
        base_seconds=config.retry_base_seconds,
        retry_exceptions=(VisionAnalysisError,),
        user_id=record.user_id,
    )
    evidence = evaluate_evidence(
        claim.claim_object,
        claim.issue_type,
        vision,
        requirements,
    )
    risk = assess_user_risk(record.user_id, user_history)
    decision = make_decision(claim, vision, evidence, risk)
    row = decision_to_row(decision)
    # Keep the CSV contract faithful to claims.csv while vision uses resolved
    # local paths internally.
    row["image_paths"] = json.dumps(
        list(record.image_paths), ensure_ascii=False, separators=(",", ":")
    )
    return row


def run(args: argparse.Namespace) -> int:
    """Load data, execute pending claims, and checkpoint exact output rows."""

    _, records = load_claims(args.claims, images_dir=args.images_dir)
    requirements = load_evidence_requirements(args.evidence_requirements)
    user_history = load_user_history(args.user_history)
    LOGGER.info(
        "Loaded %d claim(s), %d evidence rule(s), and %d history row(s)",
        len(records),
        len(requirements),
        len(user_history),
    )

    completed: dict[int, dict[str, object]] = {}
    resumed_count = 0
    if not args.no_resume:
        existing_rows = load_existing_output(args.output)
        by_key: dict[tuple[object, ...], deque[dict[str, str]]] = defaultdict(deque)
        for row in existing_rows:
            by_key[output_row_key(row)].append(row)
        for record in records:
            queue = by_key.get(record.resume_key)
            if queue:
                completed[record.index] = queue.popleft()
                resumed_count += 1
        if resumed_count:
            LOGGER.info("Resumed %d completed claim(s) from %s", resumed_count, args.output)

    pending = [record for record in records if record.index not in completed]
    if not pending:
        _checkpoint(completed, args.output)
        LOGGER.info("Nothing to process; output is complete at %s", args.output)
        return 0

    config = PipelineConfig(
        claim_model=args.claim_model,
        vision_model=args.vision_model,
        agent_attempts=args.agent_attempts,
        retry_base_seconds=args.retry_base_seconds,
        max_dimension=args.max_image_dimension,
    )
    client = OpenAI(timeout=args.api_timeout, max_retries=args.sdk_retries)
    failure_count = 0
    processed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for batch_number, batch in enumerate(
            _batches(pending, args.batch_size), start=1
        ):
            LOGGER.info(
                "Starting batch %d with %d claim(s)", batch_number, len(batch)
            )
            futures: dict[Future[tuple[dict[str, object], bool]], ClaimRecord] = {
                executor.submit(
                    _safe_process_claim,
                    record,
                    client=client,
                    requirements=requirements,
                    user_history=user_history,
                    config=config,
                ): record
                for record in batch
            }
            for future in as_completed(futures):
                record = futures[future]
                row, failed = future.result()
                completed[record.index] = row
                processed_count += 1
                failure_count += int(failed)
                LOGGER.info(
                    "Progress %d/%d new claim(s): user_id=%s status=%s",
                    processed_count,
                    len(pending),
                    record.user_id or "<missing>",
                    row["claim_status"],
                )
            _checkpoint(completed, args.output)
            LOGGER.info("Checkpointed batch %d to %s", batch_number, args.output)

    client.close()
    _checkpoint(completed, args.output)
    LOGGER.info(
        "Finished %d claim(s): %d resumed, %d failed safely; output=%s",
        len(records),
        resumed_count,
        failure_count,
        args.output,
    )
    return 0


def _safe_process_claim(
    record: ClaimRecord,
    *,
    client: OpenAI,
    requirements: pd.DataFrame,
    user_history: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[dict[str, object], bool]:
    """Convert any row-level failure into an auditable inconclusive output."""

    try:
        return (
            process_claim(
                record,
                client=client,
                requirements=requirements,
                user_history=user_history,
                config=config,
            ),
            False,
        )
    except Exception as exc:  # row isolation is intentional at the batch boundary
        LOGGER.error(
            "Claim failed safely for user_id=%s: %s: %s",
            record.user_id or "<missing>",
            type(exc).__name__,
            _safe_error(exc),
        )
        return _failure_row(record, exc), True


def _with_retry(
    function: Callable[[], T],
    *,
    operation: str,
    attempts: int,
    base_seconds: float,
    retry_exceptions: tuple[type[Exception], ...],
    user_id: str,
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return function()
        except retry_exceptions as exc:
            if attempt == attempts:
                raise
            delay = base_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            LOGGER.warning(
                "%s failed for user_id=%s (attempt %d/%d); retrying in %.2fs: %s",
                operation,
                user_id or "<missing>",
                attempt,
                attempts,
                delay,
                _safe_error(exc),
            )
            time.sleep(delay)
    raise AssertionError("retry loop ended unexpectedly")


def _failure_row(record: ClaimRecord, error: Exception) -> dict[str, object]:
    try:
        claim_object = ClaimObject(record.claim_object).value
    except (TypeError, ValueError):
        claim_object = ClaimObject.UNKNOWN.value
    return {
        "user_id": record.user_id,
        "image_paths": json.dumps(
            list(record.image_paths), ensure_ascii=False, separators=(",", ":")
        ),
        "user_claim": record.user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": False,
        "evidence_standard_met_reason": (
            "Evidence could not be fully evaluated because pipeline processing failed."
        ),
        "risk_flags": json.dumps(
            [RiskFlag.MANUAL_REVIEW_REQUIRED.value], separators=(",", ":")
        ),
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": (
            "Not enough information: processing error "
            f"({type(error).__name__}: {_safe_error(error)}). Manual review required."
        ),
        "supporting_image_ids": "[]",
        "valid_image": False,
        "severity": Severity.UNKNOWN.value,
    }


def _checkpoint(
    completed: dict[int, dict[str, object]], output_path: str | Path
) -> None:
    rows = [completed[index] for index in sorted(completed)]
    atomic_write_output(rows, output_path)


def _batches(
    records: Sequence[ClaimRecord], batch_size: int
) -> Sequence[list[ClaimRecord]]:
    return [
        list(records[start : start + batch_size])
        for start in range(0, len(records), batch_size)
    ]


def _safe_error(error: Exception) -> str:
    text = " ".join(str(error).replace("\x00", "").split())
    return (text or "unspecified error")[:300]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify multimodal damage claims and generate output.csv."
    )
    parser.add_argument("--claims", default="claims.csv")
    parser.add_argument("--user-history", default="user_history.csv")
    parser.add_argument(
        "--evidence-requirements", default="evidence_requirements.csv"
    )
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--agent-attempts", type=int, default=2)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--sdk-retries", type=int, default=2)
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--max-image-dimension", type=int, default=2048)
    parser.add_argument("--claim-model", default=DEFAULT_CLAIM_MODEL)
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.agent_attempts < 1:
        raise ValueError("--agent-attempts must be at least 1")
    if args.retry_base_seconds < 0:
        raise ValueError("--retry-base-seconds cannot be negative")
    if args.sdk_retries < 0:
        raise ValueError("--sdk-retries cannot be negative")
    if args.api_timeout <= 0:
        raise ValueError("--api-timeout must be positive")
    if args.max_image_dimension < 256:
        raise ValueError("--max-image-dimension must be at least 256")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        _validate_cli_args(args)
        return run(args)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        LOGGER.error("Fatal pipeline error: %s: %s", type(exc).__name__, _safe_error(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
