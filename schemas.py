"""Data models for the recruiter agent. Pydantic v2.

Core rule: no Claim reaches the final report unless verdict == "supported"
(or "partial", shown with a caveat). Every claim must carry Evidence that
points at a real Crustdata record.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------- Stage 1: parsed request ----------

class Priority(str, Enum):
    MUST = "must"
    NICE = "nice"


class Criterion(BaseModel):
    id: str                                   # "c1", "c2", ...
    text: str                                 # plain-English requirement
    priority: Priority
    category: Literal[
        "skill", "title", "seniority", "location",
        "industry", "company_type", "years_experience", "education", "other",
    ]
    assumed: bool = False                     # True if the agent inferred it
    assumption_reason: Optional[str] = None


class SearchVariant(BaseModel):
    name: Literal["strict", "relaxed", "adjacent"]
    titles: list[str] = []
    skills: list[str] = []
    locations: list[str] = []
    industries: list[str] = []
    min_years_experience: Optional[int] = None
    notes: str = ""                           # why this variant exists


class ParsedRequest(BaseModel):
    role_title: str
    criteria: list[Criterion]
    search_variants: list[SearchVariant]
    assumptions: list[str] = []               # surfaced in the final report
    conflicts: list[str] = []                 # e.g. "Junior title but 8+ yrs"


# ---------- Stage 3/4: evidence and claims ----------

class Evidence(BaseModel):
    record_type: Literal["person", "company", "job", "web"]
    record_id: str                            # Crustdata ID or URL
    field_path: str                           # e.g. "experience[2].company_name"
    value: str                                # raw value copied from the record


class Verdict(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"                       # true but weaker than stated
    UNSUPPORTED = "unsupported"               # evidence doesn't show it
    CONTRADICTED = "contradicted"             # evidence shows the opposite


class Claim(BaseModel):
    id: str
    candidate_id: str
    criterion_id: Optional[str] = None        # None for red-flag claims
    text: str                                 # one atomic, checkable statement
    evidence: list[Evidence]
    # filled in by the verifier:
    verdict: Optional[Verdict] = None
    verifier_note: str = ""
    corrected_text: Optional[str] = None      # verifier's weaker/accurate rewrite

    @property
    def verified(self) -> bool:
        return self.verdict in (Verdict.SUPPORTED, Verdict.PARTIAL)


# ---------- Stage 5: output ----------

class CriterionScore(BaseModel):
    criterion_id: str
    met: Literal["yes", "partial", "no", "unknown"]
    claim_ids: list[str] = []


class RedFlag(BaseModel):
    kind: Literal["gap", "overlap", "title_inflation", "short_tenures",
                  "unverifiable_employer", "other"]
    claim: Claim


class CandidateReport(BaseModel):
    candidate_id: str
    name: str
    score: float = Field(ge=0, le=100)
    criterion_scores: list[CriterionScore]
    claims: list[Claim]                       # only verified ones are rendered
    red_flags: list[RedFlag] = []
    outreach_draft: Optional[str] = None      # DRAFT ONLY, never sent
