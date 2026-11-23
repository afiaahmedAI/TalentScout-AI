"""
Data models for Job Classification System
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class DecisionType(str, Enum):
    """Candidate decision types"""
    SHORTLIST = "SHORTLIST"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class JobDescription(BaseModel):
    """Job Description model"""
    job_id: str
    title: str
    department: str
    required_skills: List[str]
    experience_years_min: int
    experience_years_max: int
    education_degree: str
    education_field: Optional[str] = None
    responsibilities: List[str]
    description: Optional[str] = None


class Candidate(BaseModel):
    """Candidate model"""
    candidate_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    cv_text: str
    cv_path: Optional[str] = None


class ClassificationResult(BaseModel):
    """Result from Agent 1 - Job Classification"""
    candidate_id: str
    candidate_name: str
    classified_role: str
    confidence: float
    all_scores: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SkillsMatchResult(BaseModel):
    """Skills matching evaluation"""
    score: float = Field(ge=0, le=100)
    matched_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]
    reasoning: str


class ExperienceMatchResult(BaseModel):
    """Experience matching evaluation"""
    score: float = Field(ge=0, le=100)
    candidate_years: float
    required_min: int
    required_max: int
    relevance: str  # "Perfect", "Overqualified", "Underqualified"
    reasoning: str


class EducationMatchResult(BaseModel):
    """Education matching evaluation"""
    score: float = Field(ge=0, le=100)
    candidate_degree: str
    required_degree: str
    field_match: bool
    reasoning: str


class EvaluationBreakdown(BaseModel):
    """Detailed evaluation breakdown"""
    skills_match: SkillsMatchResult
    experience_match: ExperienceMatchResult
    education_match: EducationMatchResult


class EvaluationResult(BaseModel):
    """Result from Agent 2 - Candidate Evaluation"""
    candidate_id: str
    candidate_name: str
    classified_role: str
    overall_score: float = Field(ge=0, le=100)
    breakdown: EvaluationBreakdown
    decision: DecisionType
    recommendation: str
    justification: str
    strengths: List[str]
    areas_for_improvement: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProcessingResult(BaseModel):
    """Complete processing result for one candidate"""
    candidate_id: str
    candidate_name: str
    cv_path: Optional[str] = None
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_1_result: ClassificationResult
    agent_2_result: EvaluationResult
    processing_time_seconds: float


class BatchProcessingResult(BaseModel):
    """Batch processing summary"""
    job_description_id: str
    total_candidates: int
    processing_date: datetime = Field(default_factory=datetime.utcnow)
    summary: Dict[str, int]  # {"shortlisted": 12, "review_needed": 18, "rejected": 20}
    results: List[ProcessingResult]
    total_processing_time_seconds: float
