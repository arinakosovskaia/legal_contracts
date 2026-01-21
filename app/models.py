from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class Paragraph(BaseModel):
    text: str
    page: int
    paragraph_index: int


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


RedFlagCategory = Literal[
    "Termination For Convenience",
    "Uncapped Liability",
    "Irrevocable Or Perpetual License",
    "Most Favored Nation",
    "Audit Rights",
    "Ip Ownership Assignment",
]


class RedFlagFindingLLM(BaseModel):
    category: RedFlagCategory
    evidence: str = Field(min_length=1, max_length=1200)
    reasoning: list[str] = Field(default_factory=list, min_length=1, max_length=3)


class RedFlagChunkOutput(BaseModel):
    findings: list[RedFlagFindingLLM] = Field(default_factory=list, max_length=2)


class Stage1Prob(BaseModel):
    category_id: str
    prob: float = Field(ge=0.0, le=1.0)

    @field_validator("category_id", mode="before")
    @classmethod
    def _coerce_category_id_to_str(cls, v: Any) -> Any:
        if isinstance(v, int):
            return str(v)
        return v


class Stage1Category(BaseModel):
    category_id: str
    severity: Literal["low", "medium", "high"]
    # Router probability for this category (used for thresholding and cost control).
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_short: str = Field(max_length=220)
    evidence_paragraph_indices: list[int] = Field(default_factory=list)

    @field_validator("category_id", mode="before")
    @classmethod
    def _coerce_category_id_to_str(cls, v: Any) -> Any:
        if isinstance(v, int):
            return str(v)
        return v

    @field_validator("evidence_paragraph_indices")
    @classmethod
    def _limit_evidence_indices(cls, v: list[int]) -> list[int]:
        # Keep small to control cost/verbosity.
        return [int(x) for x in v[:5]]


class Stage1Output(BaseModel):
    """
    Stage 1 router:
    - category_probs: probabilities over categories (multi-label)
    - categories: selected categories (>= threshold) with evidence paragraph indices
    - unclear_prob: "abstain"/unclear probability (0..1)
    """

    category_probs: list[Stage1Prob] = Field(default_factory=list)
    categories: list[Stage1Category] = Field(default_factory=list)
    unclear_prob: float = Field(default=0.0, ge=0.0, le=1.0)


class Stage2Output(BaseModel):
    issue_title: str = Field(max_length=120)
    what_is_wrong: str = Field(max_length=600)
    recommendation: str = Field(max_length=400)
    evidence_quote: str = Field(max_length=400)
    severity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    tags: Optional[list[str]] = Field(default=None, max_length=6)

    @field_validator("tags")
    @classmethod
    def _tags_max_len(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return v
        return v[:6]


class Finding(BaseModel):
    finding_id: str
    category_id: str
    category_name: str
    severity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    issue_title: str
    what_is_wrong: str
    recommendation: str
    evidence_quote: str
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

