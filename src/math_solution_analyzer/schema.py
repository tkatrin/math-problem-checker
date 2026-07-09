from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class StepStatus(str, Enum):
    CORRECT = "correct"
    SUSPICIOUS = "suspicious"
    NEEDS_ATTENTION = "needs_attention"
    INCORRECT = "incorrect"
    INCOMPLETE = "incomplete"


class ParsedProblem(BaseModel):
    """Normalized user input before step splitting."""

    problem: str = Field(..., description="Original problem statement.")
    solution: str = Field(..., description="Original solution text.")
    contains_latex: bool = Field(False, description="Whether input appears to contain LaTeX.")


class SolutionStep(BaseModel):
    index: int = Field(..., ge=1)
    text: str
    source: str = Field(default="user")


class Issue(BaseModel):
    severity: Severity
    title: str
    explanation: str
    recommendation: str


class MLStepPrediction(BaseModel):
    label: StepStatus
    error_type: str = "none"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    model_name: str = "unavailable"
    label_probabilities: dict[str, float] = Field(default_factory=dict)


class StepAnalysis(BaseModel):
    step: SolutionStep
    status: StepStatus
    what_is_correct: list[str] = Field(default_factory=list)
    possible_errors: list[Issue] = Field(default_factory=list)
    missing_steps: list[str] = Field(default_factory=list)
    how_to_fix: list[str] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    ml_prediction: MLStepPrediction | None = None
    llm_explanation: str | None = None


class AnalysisReport(BaseModel):
    problem: str
    steps: list[StepAnalysis]
    summary: str
    what_is_correct: list[str]
    where_possible_error: list[str]
    missing_steps: list[str]
    how_to_fix: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
