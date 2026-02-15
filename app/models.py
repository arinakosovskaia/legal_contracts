from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, AliasChoices


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class Paragraph(BaseModel):
    text: str
    page: int
    paragraph_index: int
    bbox: Optional[dict[str, float]] = None


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    limits: dict[str, Any]


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: Optional[str] = None
    progress: int = 0
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    updated_at: datetime
    filename: Optional[str] = None
    estimated_minutes: Optional[int] = None


RedFlagCategory = str


class RiskAssessment(BaseModel):
    severity_of_consequences: int = Field(default=0, ge=0, le=3)
    degree_of_legal_violation: int = Field(default=0, ge=0, le=3)


class RecommendedRevision(BaseModel):
    revised_clause: str = Field(default="", max_length=2000)
    revision_explanation: str = Field(default="", max_length=800)


class RedFlagFindingLLM(BaseModel):
    category: RedFlagCategory
    quote: str = Field(default="", max_length=2000, validation_alias=AliasChoices("quote", "evidence"))
    reasoning: list[str] = Field(default_factory=list, max_length=3)
    is_unfair: bool = True
    explanation: str = Field(default="", max_length=1200)
    legal_references: list[str] = Field(default_factory=list, max_length=6)
    possible_consequences: str = Field(default="", max_length=800)
    risk_assessment: RiskAssessment = Field(default_factory=RiskAssessment)
    consequences_category: Literal["Invalid", "Unenforceable", "Void", "Voidable", "Nothing"] = "Nothing"
    risk_category: Literal[
        "Penalty",
        "Criminal liability",
        "Failure of the other party to perform obligations on time",
        "Nothing",
    ] = "Nothing"
    recommended_revision: RecommendedRevision = Field(default_factory=RecommendedRevision)
    suggested_follow_up: str = Field(default="", max_length=600)

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, v: Any) -> Any:
        """Try to match category name even if slightly different."""
        if not isinstance(v, str):
            return v
        v_clean = v.strip()
        # Exact match
        allowed = [
            "Termination For Convenience",
            "Uncapped Liability",
            "Irrevocable Or Perpetual License",
            "Most Favored Nation",
            "Audit Rights",
            "Ip Ownership Assignment",
        ]
        for a in allowed:
            if v_clean == a or v_clean.lower() == a.lower():
                return a
        # Fuzzy match - try to find closest
        v_lower = v_clean.lower()
        for a in allowed:
            a_lower = a.lower()
            # Remove common variations
            if "termination" in v_lower and "convenience" in v_lower and "termination" in a_lower and "convenience" in a_lower:
                return a
            if "uncapped" in v_lower and "liability" in v_lower and "uncapped" in a_lower and "liability" in a_lower:
                return a
            if "irrevocable" in v_lower and "perpetual" in v_lower and "license" in v_lower:
                return a
            if "most" in v_lower and "favored" in v_lower and "nation" in v_lower:
                return a
            if "audit" in v_lower and "rights" in v_lower:
                return a
            if "ip" in v_lower and "ownership" in v_lower and "assignment" in v_lower:
                return a
        # If no match, keep the original value (allows non-CUAD categories).
        return v_clean

    @field_validator("consequences_category", mode="before")
    @classmethod
    def _coerce_consequences_category(cls, v: Any) -> Any:
        """Normalize consequences_category to match Literal type exactly."""
        if not isinstance(v, str):
            return "Nothing"
        v_clean = v.strip()
        v_lower = v_clean.lower()
        # Map to exact Literal values (capitalized)
        if v_lower == "unenforceable":
            return "Unenforceable"
        elif v_lower == "invalid":
            return "Invalid"
        elif v_lower == "void":
            return "Void"
        elif v_lower == "voidable":
            return "Voidable"
        elif v_lower == "nothing":
            return "Nothing"
        elif v_clean in ("Invalid", "Unenforceable", "Void", "Voidable", "Nothing"):
            # Already correct format
            return v_clean
        else:
            return "Nothing"

    @field_validator("risk_category", mode="before")
    @classmethod
    def _coerce_risk_category(cls, v: Any) -> Any:
        """Normalize risk_category to match Literal type exactly."""
        if not isinstance(v, str):
            return "Nothing"
        v_clean = v.strip()
        v_lower = v_clean.lower()
        # Map to exact Literal values
        if v_lower == "penalty":
            return "Penalty"
        elif v_lower == "criminal liability":
            return "Criminal liability"
        elif "failure" in v_lower and "obligations" in v_lower:
            return "Failure of the other party to perform obligations on time"
        elif v_lower == "nothing":
            return "Nothing"
        elif v_clean in ("Penalty", "Criminal liability", "Failure of the other party to perform obligations on time", "Nothing"):
            # Already correct format
            return v_clean
        else:
            return "Nothing"


class RedFlagChunkOutput(BaseModel):
    findings: list[RedFlagFindingLLM] = Field(default_factory=list)


class RouterCandidateCategory(BaseModel):
    category: str
    confidence: Literal["low", "medium", "high"]
    trigger_quotes: list[str] = Field(default_factory=list, max_length=3)
    rationale: str = Field(default="", max_length=400)


class RouterCategoriesOutput(BaseModel):
    candidate_categories: list[RouterCandidateCategory] = Field(default_factory=list)


class Finding(BaseModel):
    finding_id: str
    category_id: str
    category_name: str
    severity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    is_unfair: bool
    issue_title: str
    what_is_wrong: str
    explanation: str = ""
    recommendation: str
    evidence_quote: str
    legal_references: list[str] = Field(default_factory=list, max_length=6)
    possible_consequences: str = ""
    risk_assessment: RiskAssessment
    consequences_category: Literal["Invalid", "Unenforceable", "Void", "Voidable", "Nothing"]
    risk_category: Literal[
        "Penalty",
        "Criminal liability",
        "Failure of the other party to perform obligations on time",
        "Nothing",
    ]
    revised_clause: str = ""
    revision_explanation: str = ""
    suggested_follow_up: str = ""
    paragraph_bbox: Optional[dict[str, float]] = None
    location: dict[str, int]


class ResultSummary(BaseModel):
    risk_score: int = Field(ge=0, le=100)
    high: int = 0
    medium: int = 0
    low: int = 0
    top_categories: list[dict[str, Any]] = Field(default_factory=list)


class ResultDocument(BaseModel):
    filename: str
    page_count: int
    paragraph_count: int


class ResultMeta(BaseModel):
    categories_version: str
    prompt_version: str
    models: dict[str, str]
    debug: Optional[dict[str, Any]] = None


class ResultPayload(BaseModel):
    job_id: str
    document: ResultDocument
    summary: ResultSummary
    findings: list[Finding]
    meta: ResultMeta

