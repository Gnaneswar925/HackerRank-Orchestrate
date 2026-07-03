"""Strongly typed data contracts for the damage-claim verification pipeline.

The models in this module are deliberately independent of CSV and OpenAI SDK
types.  They form the validated boundary between agents and can be serialized
with ``model_dump(mode="json")`` before being written to CSV or JSON.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


NonEmptyString = Annotated[str, Field(min_length=1)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class ClaimObject(str, Enum):
    """Objects supported by the claim-verification system."""

    CAR = "car"
    LAPTOP = "laptop"
    PACKAGE = "package"
    UNKNOWN = "unknown"


class ClaimStatus(str, Enum):
    """Permitted final claim outcomes."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class IssueType(str, Enum):
    """Normalized damage categories used across all agents."""

    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"
    TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    NONE = "none"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Normalized damage severity."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class ImageQuality(str, Enum):
    """Whether an image is usable for visual damage verification."""

    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


class VisionFlag(str, Enum):
    """Standardized limitations or concerns detected in evidence images."""

    BLURRY_IMAGE = "blurry_image"
    CROPPED_OR_OBSTRUCTED = "cropped_or_obstructed"
    LOW_LIGHT_OR_GLARE = "low_light_or_glare"
    WRONG_ANGLE = "wrong_angle"
    WRONG_OBJECT = "wrong_object"
    WRONG_OBJECT_PART = "wrong_object_part"
    DAMAGE_NOT_VISIBLE = "damage_not_visible"
    POSSIBLE_MANIPULATION = "possible_manipulation"


class RiskLevel(str, Enum):
    """Aggregate user-history risk level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class RiskFlag(str, Enum):
    """Standardized, explainable risk indicators."""

    FREQUENT_CLAIMS = "frequent_claims"
    RECENT_CLAIM_CLUSTER = "recent_claim_cluster"
    REPEATED_ISSUE = "repeated_issue"
    PRIOR_CONTRADICTED_CLAIMS = "prior_contradicted_claims"
    DUPLICATE_EVIDENCE = "duplicate_evidence"
    INCONSISTENT_HISTORY = "inconsistent_history"
    HIGH_REJECTION_RATE = "high_rejection_rate"
    ABNORMAL_CLAIM_FREQUENCY = "abnormal_claim_frequency"
    SUSPICIOUS_HISTORY = "suspicious_history"
    USER_HISTORY_RISK = "user_history_risk"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    NO_HISTORY = "no_history"
    NONE = "none"
    UNKNOWN = "unknown"


class SchemaModel(BaseModel):
    """Common strict behavior for all inter-agent contracts."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class DamageMention(SchemaModel):
    """One distinct damage allegation found in a claim conversation."""

    issue_type: IssueType
    object_part: NonEmptyString
    claimed_damage: NonEmptyString
    severity: Severity


class ClaimExtraction(SchemaModel):
    """Structured facts extracted from a user's claim conversation.

    ``issue_type``, ``object_part``, and ``claimed_damage`` describe the primary
    allegation. ``damages`` preserves every distinct allegation when the user
    reports more than one.
    """

    user_claim: NonEmptyString
    claim_object: ClaimObject
    issue_type: IssueType
    object_part: NonEmptyString
    claimed_damage: NonEmptyString
    confidence: Confidence
    extracted_summary: NonEmptyString
    damages: list[DamageMention] = Field(min_length=1)
    is_vague: bool
    contradictions: list[NonEmptyString]


class ImageAnalysis(SchemaModel):
    """Grounded findings for one evidence image."""

    image_id: NonEmptyString
    image_path: NonEmptyString
    valid_image: bool
    detected_object: ClaimObject
    object_matches_claim: bool
    issue_types: list[IssueType]
    object_parts: list[NonEmptyString]
    severity: Severity
    image_quality: ImageQuality
    damage_clearly_visible: bool
    flags: list[VisionFlag]
    observations: list[NonEmptyString]
    analysis_confidence: Confidence
    invalid_reason: str | None

    @model_validator(mode="after")
    def validate_image_state(self) -> ImageAnalysis:
        """Require a reason for invalid files and prevent false confidence."""

        if not self.valid_image and not self.invalid_reason:
            raise ValueError("invalid_reason is required when valid_image is false")
        if not self.valid_image and self.analysis_confidence != 0.0:
            raise ValueError(
                "analysis_confidence must be 0.0 when valid_image is false"
            )
        if not self.valid_image and self.image_quality is not ImageQuality.UNUSABLE:
            raise ValueError("invalid images must have unusable image_quality")
        if self.damage_clearly_visible and IssueType.NONE in self.issue_types:
            raise ValueError("visible damage cannot use issue_type 'none'")
        return self


class VisionAnalysis(SchemaModel):
    """Aggregate vision result with auditable per-image findings."""

    claim_object: ClaimObject
    image_paths: list[NonEmptyString] = Field(min_length=1)
    valid_image: bool
    object_type_verified: bool
    issue_type: IssueType
    object_part: NonEmptyString
    severity: Severity
    image_quality: ImageQuality
    damage_clearly_visible: bool
    flags: list[VisionFlag]
    supporting_image_ids: list[NonEmptyString]
    observations: list[NonEmptyString]
    analysis_confidence: Confidence
    image_analyses: list[ImageAnalysis] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aggregate_state(self) -> VisionAnalysis:
        """Keep aggregate fields consistent with the per-image evidence."""

        if len(self.image_paths) != len(self.image_analyses):
            raise ValueError("image_paths and image_analyses must have equal length")
        known_ids = {item.image_id for item in self.image_analyses}
        if len(known_ids) != len(self.image_analyses):
            raise ValueError("image_analyses contains duplicate image_id values")
        if not set(self.supporting_image_ids).issubset(known_ids):
            raise ValueError("supporting_image_ids contains an unknown image_id")
        by_id = {item.image_id: item for item in self.image_analyses}
        if any(
            not by_id[image_id].valid_image
            or not by_id[image_id].damage_clearly_visible
            for image_id in self.supporting_image_ids
        ):
            raise ValueError(
                "supporting images must be valid and show damage clearly"
            )
        if self.valid_image != any(item.valid_image for item in self.image_analyses):
            raise ValueError("valid_image must reflect the per-image analyses")
        if self.object_type_verified != any(
            item.valid_image and item.object_matches_claim
            for item in self.image_analyses
        ):
            raise ValueError(
                "object_type_verified must reflect the per-image analyses"
            )
        if self.damage_clearly_visible != any(
            item.valid_image and item.damage_clearly_visible
            for item in self.image_analyses
        ):
            raise ValueError(
                "damage_clearly_visible must reflect the per-image analyses"
            )
        return self


class EvidenceResult(SchemaModel):
    """Result of matching claim facts, images, and evidence requirements."""

    evidence_standard_met: bool
    evidence_standard_met_reason: NonEmptyString
    required_evidence: list[str] = Field(default_factory=list)
    satisfied_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    supporting_image_ids: list[str] = Field(default_factory=list)
    contradicting_image_ids: list[str] = Field(default_factory=list)
    evidence_confidence: Confidence = 0.0


class RiskAssessment(SchemaModel):
    """User-history context; risk is contextual and never proof by itself."""

    user_id: NonEmptyString
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    risk_score: Confidence = 0.0
    assessment_reason: NonEmptyString
    historical_claim_count: Annotated[int, Field(ge=0)] = 0


class FinalDecision(SchemaModel):
    """Exact output contract for one verified damage claim."""

    user_id: NonEmptyString
    image_paths: list[str] = Field(default_factory=list)
    user_claim: NonEmptyString
    claim_object: ClaimObject
    evidence_standard_met: bool
    evidence_standard_met_reason: NonEmptyString
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    issue_type: IssueType
    object_part: NonEmptyString
    claim_status: ClaimStatus
    claim_status_justification: NonEmptyString
    supporting_image_ids: list[str] = Field(default_factory=list)
    valid_image: bool
    severity: Severity

    @model_validator(mode="after")
    def validate_decision_state(self) -> FinalDecision:
        """Enforce the minimum invariants for decisive outcomes."""

        if self.claim_status is ClaimStatus.SUPPORTED:
            if not self.evidence_standard_met:
                raise ValueError("supported decisions require sufficient evidence")
            if not self.supporting_image_ids:
                raise ValueError("supported decisions require a supporting image")
        if self.claim_status in {
            ClaimStatus.SUPPORTED,
            ClaimStatus.CONTRADICTED,
        }:
            if not self.valid_image:
                raise ValueError("decisive outcomes require a valid image")
            if not self.supporting_image_ids:
                raise ValueError("decisive outcomes require grounded image IDs")
        return self
