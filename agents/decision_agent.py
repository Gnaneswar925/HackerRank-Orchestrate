"""Deterministic final policy for multimodal claim verification."""

from __future__ import annotations

import re
from collections.abc import Iterable

from models.schemas import (
    ClaimExtraction,
    ClaimStatus,
    EvidenceResult,
    FinalDecision,
    ImageAnalysis,
    ImageQuality,
    IssueType,
    RiskAssessment,
    RiskFlag,
    Severity,
    VisionAnalysis,
    VisionFlag,
)


DECISION_BLOCKING_FLAGS = frozenset(
    {
        VisionFlag.BLURRY_IMAGE,
        VisionFlag.CROPPED_OR_OBSTRUCTED,
        VisionFlag.LOW_LIGHT_OR_GLARE,
        VisionFlag.WRONG_ANGLE,
        VisionFlag.WRONG_OBJECT,
        VisionFlag.WRONG_OBJECT_PART,
        VisionFlag.POSSIBLE_MANIPULATION,
    }
)

SEVERITY_ORDER = {
    Severity.NONE: 0,
    Severity.UNKNOWN: 1,
    Severity.LOW: 2,
    Severity.MEDIUM: 3,
    Severity.HIGH: 4,
}


class DecisionAgentError(RuntimeError):
    """Raised when upstream results cannot form a coherent decision."""


def make_decision(
    claim: ClaimExtraction,
    vision: VisionAnalysis,
    evidence: EvidenceResult,
    risk: RiskAssessment,
) -> FinalDecision:
    """Apply the image-first decision hierarchy to validated agent outputs.

    Risk context is copied to the output but never participates in status
    selection. This makes it impossible for history alone to support or reject
    a claim.
    """

    claim_result = ClaimExtraction.model_validate(claim)
    vision_result = VisionAnalysis.model_validate(vision)
    evidence_result = EvidenceResult.model_validate(evidence)
    risk_result = RiskAssessment.model_validate(risk)
    _validate_alignment(claim_result, vision_result, evidence_result)

    if claim_result.issue_type is IssueType.UNKNOWN:
        return _not_enough_information(
            claim=claim_result,
            vision=vision_result,
            evidence=evidence_result,
            risk=risk_result,
            supporting_ids=[],
            reasons=["the claimed damage type is unclear"],
        )

    if claim_result.issue_type is IssueType.NONE:
        return _decide_no_damage_claim(
            claim_result, vision_result, evidence_result, risk_result
        )

    matching_images = [
        image
        for image in vision_result.image_analyses
        if _supports_claim(image, claim_result)
    ]
    contradicting_images = [
        image
        for image in vision_result.image_analyses
        if _contradicts_claim(image, claim_result)
    ]
    matching_ids = _image_ids(matching_images)
    contradicting_ids = _image_ids(contradicting_images)

    # Two clean views that materially disagree are not resolved by history or
    # by selecting the more convenient image.
    if matching_images and contradicting_images:
        return _not_enough_information(
            claim=claim_result,
            vision=vision_result,
            evidence=evidence_result,
            risk=risk_result,
            supporting_ids=matching_ids,
            reasons=[
                "clear images conflict: "
                f"{_join_ids(matching_ids)} show the claimed damage while "
                f"{_join_ids(contradicting_ids)} do not"
            ],
        )

    if matching_images and evidence_result.evidence_standard_met:
        justification = (
            f"Supported: {_join_ids(matching_ids)} clearly show "
            f"{claim_result.issue_type.value} on {claim_result.object_part}; "
            f"the minimum evidence standard is met."
        )
        return _build_final(
            claim=claim_result,
            vision=vision_result,
            evidence=evidence_result,
            risk=risk_result,
            status=ClaimStatus.SUPPORTED,
            justification=_append_risk_context(justification, risk_result),
            supporting_ids=matching_ids,
            severity=_maximum_severity(matching_images),
        )

    if contradicting_images:
        justification = (
            f"Contradicted: {_join_ids(contradicting_ids)} clearly show the "
            f"relevant {claim_result.object_part} of the claimed object, but "
            f"{claim_result.issue_type.value} is absent."
        )
        if not evidence_result.evidence_standard_met:
            justification += (
                " The support standard is not met, consistent with the absence "
                "of claim-supporting damage evidence."
            )
        return _build_final(
            claim=claim_result,
            vision=vision_result,
            evidence=evidence_result,
            risk=risk_result,
            status=ClaimStatus.CONTRADICTED,
            justification=_append_risk_context(justification, risk_result),
            supporting_ids=contradicting_ids,
            severity=Severity.NONE,
        )

    reasons = _information_gaps(claim_result, vision_result, evidence_result)
    return _not_enough_information(
        claim=claim_result,
        vision=vision_result,
        evidence=evidence_result,
        risk=risk_result,
        supporting_ids=matching_ids,
        reasons=reasons,
    )


def decide_claim(
    claim: ClaimExtraction,
    vision: VisionAnalysis,
    evidence: EvidenceResult,
    risk: RiskAssessment,
) -> FinalDecision:
    """Backward-friendly public alias for :func:`make_decision`."""

    return make_decision(claim, vision, evidence, risk)


def _decide_no_damage_claim(
    claim: ClaimExtraction,
    vision: VisionAnalysis,
    evidence: EvidenceResult,
    risk: RiskAssessment,
) -> FinalDecision:
    no_damage_images = [
        image
        for image in vision.image_analyses
        if _is_decisive_view(image, claim)
        and not image.damage_clearly_visible
        and IssueType.NONE in image.issue_types
    ]
    damage_images = [
        image
        for image in vision.image_analyses
        if _is_decisive_view(image, claim)
        and image.damage_clearly_visible
        and any(issue not in {IssueType.NONE, IssueType.UNKNOWN} for issue in image.issue_types)
    ]
    no_damage_ids = _image_ids(no_damage_images)
    damage_ids = _image_ids(damage_images)

    if no_damage_images and damage_images:
        return _not_enough_information(
            claim=claim,
            vision=vision,
            evidence=evidence,
            risk=risk,
            supporting_ids=no_damage_ids,
            reasons=["clear images conflict about whether damage is present"],
        )
    if no_damage_images and evidence.evidence_standard_met:
        return _build_final(
            claim=claim,
            vision=vision,
            evidence=evidence,
            risk=risk,
            status=ClaimStatus.SUPPORTED,
            justification=_append_risk_context(
                f"Supported: {_join_ids(no_damage_ids)} clearly show no damage "
                f"on {claim.object_part}, and the evidence standard is met.",
                risk,
            ),
            supporting_ids=no_damage_ids,
            severity=Severity.NONE,
        )
    if damage_images:
        return _build_final(
            claim=claim,
            vision=vision,
            evidence=evidence,
            risk=risk,
            status=ClaimStatus.CONTRADICTED,
            justification=_append_risk_context(
                f"Contradicted: {_join_ids(damage_ids)} clearly show physical "
                f"damage on {claim.object_part}.",
                risk,
            ),
            supporting_ids=damage_ids,
            severity=_maximum_severity(damage_images),
        )
    return _not_enough_information(
        claim=claim,
        vision=vision,
        evidence=evidence,
        risk=risk,
        supporting_ids=no_damage_ids,
        reasons=_information_gaps(claim, vision, evidence),
    )


def _supports_claim(image: ImageAnalysis, claim: ClaimExtraction) -> bool:
    return (
        _is_decisive_view(image, claim, allow_unknown_part=True)
        and image.damage_clearly_visible
        and claim.issue_type in image.issue_types
    )


def _contradicts_claim(image: ImageAnalysis, claim: ClaimExtraction) -> bool:
    claimed_damage_absent = (
        IssueType.NONE in image.issue_types
        or VisionFlag.DAMAGE_NOT_VISIBLE in image.flags
    )
    return (
        _is_decisive_view(image, claim)
        and not image.damage_clearly_visible
        and claimed_damage_absent
    )


def _is_decisive_view(
    image: ImageAnalysis,
    claim: ClaimExtraction,
    *,
    allow_unknown_part: bool = False,
) -> bool:
    if not image.valid_image or not image.object_matches_claim:
        return False
    if image.image_quality not in {ImageQuality.GOOD, ImageQuality.ACCEPTABLE}:
        return False
    if DECISION_BLOCKING_FLAGS.intersection(image.flags):
        return False
    return _claimed_part_visible(
        claim.object_part,
        image.object_parts,
        allow_unknown=allow_unknown_part,
    )


def _claimed_part_visible(
    claimed_part: str,
    visible_parts: list[str],
    *,
    allow_unknown: bool = False,
) -> bool:
    claimed = _tokens(claimed_part)
    if not claimed or claimed == {"unknown"}:
        # A general damage allegation may match a visible issue without naming
        # a part. Absence, however, is never decisive when the target is unclear.
        return allow_unknown and any(
            _tokens(part) and _tokens(part) != {"unknown"}
            for part in visible_parts
        )
    for part in visible_parts:
        visible = _tokens(part)
        if visible and visible != {"unknown"} and (
            claimed.issubset(visible) or visible.issubset(claimed)
        ):
            return True
    return False


def _information_gaps(
    claim: ClaimExtraction,
    vision: VisionAnalysis,
    evidence: EvidenceResult,
) -> list[str]:
    gaps: list[str] = []
    if not vision.valid_image:
        gaps.append("no valid image is available")
    if not vision.object_type_verified:
        gaps.append("the claimed object is not visibly verified")
    relevant = [
        item
        for item in vision.image_analyses
        if item.valid_image and item.object_matches_claim
    ]
    if claim.object_part == "unknown":
        gaps.append("the claimed object part is unclear")
    elif not any(_claimed_part_visible(claim.object_part, item.object_parts) for item in relevant):
        gaps.append(f"the relevant part '{claim.object_part}' is not visible")
    limiting_flags = _ordered_flag_values(
        flag
        for item in relevant
        for flag in item.flags
        if flag in DECISION_BLOCKING_FLAGS
    )
    if limiting_flags:
        gaps.append("image limitations: " + ", ".join(limiting_flags))
    if relevant and all(
        item.image_quality in {ImageQuality.POOR, ImageQuality.UNUSABLE}
        for item in relevant
    ):
        gaps.append("relevant image quality is insufficient")
    if not evidence.evidence_standard_met:
        gaps.append(
            "minimum evidence standard not met: "
            + evidence.evidence_standard_met_reason.rstrip(".")
        )
    if not gaps:
        gaps.append("images do not clearly establish presence or absence of the claimed damage")
    return gaps


def _not_enough_information(
    *,
    claim: ClaimExtraction,
    vision: VisionAnalysis,
    evidence: EvidenceResult,
    risk: RiskAssessment,
    supporting_ids: list[str],
    reasons: list[str],
) -> FinalDecision:
    justification = "Not enough information: " + "; ".join(reasons) + "."
    return _build_final(
        claim=claim,
        vision=vision,
        evidence=evidence,
        risk=risk,
        status=ClaimStatus.NOT_ENOUGH_INFORMATION,
        justification=_append_risk_context(justification, risk),
        supporting_ids=supporting_ids,
        severity=Severity.UNKNOWN,
    )


def _build_final(
    *,
    claim: ClaimExtraction,
    vision: VisionAnalysis,
    evidence: EvidenceResult,
    risk: RiskAssessment,
    status: ClaimStatus,
    justification: str,
    supporting_ids: list[str],
    severity: Severity,
) -> FinalDecision:
    return FinalDecision(
        user_id=risk.user_id,
        image_paths=vision.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=evidence.evidence_standard_met,
        evidence_standard_met_reason=evidence.evidence_standard_met_reason,
        risk_flags=risk.risk_flags,
        issue_type=claim.issue_type,
        object_part=claim.object_part,
        claim_status=status,
        claim_status_justification=justification,
        supporting_image_ids=supporting_ids,
        valid_image=vision.valid_image,
        severity=severity,
    )


def _validate_alignment(
    claim: ClaimExtraction,
    vision: VisionAnalysis,
    evidence: EvidenceResult,
) -> None:
    if claim.claim_object != vision.claim_object:
        raise DecisionAgentError(
            "claim and vision results refer to different object types"
        )
    known_ids = {item.image_id for item in vision.image_analyses}
    evidence_ids = {
        *evidence.supporting_image_ids,
        *evidence.contradicting_image_ids,
    }
    unknown_ids = evidence_ids.difference(known_ids)
    if unknown_ids:
        raise DecisionAgentError(
            "evidence result references unknown image IDs: "
            + ", ".join(sorted(unknown_ids))
        )


def _append_risk_context(text: str, risk: RiskAssessment) -> str:
    material_flags = set(risk.risk_flags).difference(
        {RiskFlag.NONE, RiskFlag.NO_HISTORY}
    )
    if RiskFlag.MANUAL_REVIEW_REQUIRED in material_flags:
        return (
            text
            + " History separately requires manual review but does not change "
            "this image-grounded status."
        )
    if material_flags:
        return text + " History context does not alter this image-grounded status."
    return text


def _maximum_severity(images: list[ImageAnalysis]) -> Severity:
    return max((image.severity for image in images), key=SEVERITY_ORDER.__getitem__)


def _image_ids(images: list[ImageAnalysis]) -> list[str]:
    return [image.image_id for image in images]


def _join_ids(image_ids: list[str]) -> str:
    return ", ".join(image_ids)


def _tokens(value: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _ordered_flag_values(flags: Iterable[VisionFlag]) -> list[str]:
    return [flag.value for flag in dict.fromkeys(flags)]
