"""
Multi-Agent Orchestrator
Coordinates Agent 1 (Classification) and Agent 2 (Evaluation)
"""
import time
import logging
from typing import List
from pathlib import Path

from agents.agent_1_classifier import JobClassificationAgent
from agents.agent_2_evaluator import CandidateEvaluationAgent
from models.models import (
    JobDescription, Candidate, ProcessingResult,
    BatchProcessingResult
)
from utils.pdf_utils import PDFExtractor, TextCleaner

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """Orchestrates the multi-agent job classification system"""
    
    def __init__(
        self,
        classification_model: str = "facebook/bart-large-mnli",
        evaluation_model: str = "google/flan-t5-base"
    ):
        """
        Initialize orchestrator with both agents
        
        Args:
            classification_model: Model for Agent 1
            evaluation_model: Model for Agent 2
        """
        logger.info("Initializing Multi-Agent Orchestrator")
        
        # Initialize agents
        self.agent_1 = JobClassificationAgent(model_name=classification_model)
        self.agent_2 = CandidateEvaluationAgent(model_name=evaluation_model)
        
        # Initialize utilities
        self.pdf_extractor = PDFExtractor()
        self.text_cleaner = TextCleaner()
        
        logger.info("Orchestrator initialized successfully")
    
    def process_single_candidate(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        available_roles: List[str]
    ) -> ProcessingResult:
        """
        Process a single candidate through both agents
        
        Workflow:
        1. Agent 1: Classify candidate into job role
        2. Agent 2: Evaluate candidate against JD
        3. Return combined result
        
        Args:
            candidate: Candidate object with CV text
            job_description: Job description to match against
            available_roles: List of possible job roles for classification
        
        Returns:
            ProcessingResult with both agent results
        """
        start_time = time.time()
        
        logger.info(f"Processing candidate: {candidate.name}")
        
        try:
            # Step 1: Agent 1 - Classification
            logger.info("Step 1: Classifying job role...")
            agent_1_result = self.agent_1.classify_candidate(
                candidate_id=candidate.candidate_id,
                candidate_name=candidate.name,
                cv_text=candidate.cv_text,
                job_roles=available_roles
            )
            
            # Step 2: Agent 2 - Evaluation
            logger.info("Step 2: Evaluating candidate...")
            agent_2_result = self.agent_2.evaluate_candidate(
                candidate_id=candidate.candidate_id,
                candidate_name=candidate.name,
                classified_role=agent_1_result.classified_role,
                cv_text=candidate.cv_text,
                job_description=job_description
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = ProcessingResult(
                candidate_id=candidate.candidate_id,
                candidate_name=candidate.name,
                cv_path=candidate.cv_path,
                agent_1_result=agent_1_result,
                agent_2_result=agent_2_result,
                processing_time_seconds=round(processing_time, 2)
            )
            
            logger.info(f"Processing complete for {candidate.name} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process candidate {candidate.name}: {e}")
            raise
    
    def process_batch(
        self,
        candidates: List[Candidate],
        job_description: JobDescription,
        available_roles: List[str]
    ) -> BatchProcessingResult:
        """
        Process multiple candidates
        
        Args:
            candidates: List of Candidate objects
            job_description: Job description to match against
            available_roles: List of possible job roles
        
        Returns:
            BatchProcessingResult with summary and all results
        """
        start_time = time.time()
        
        logger.info(f"Starting batch processing of {len(candidates)} candidates")
        
        results = []
        for candidate in candidates:
            try:
                result = self.process_single_candidate(
                    candidate=candidate,
                    job_description=job_description,
                    available_roles=available_roles
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {candidate.name}: {e}")
                continue
        
        # Calculate summary
        summary = {
            "shortlisted": sum(1 for r in results if r.agent_2_result.decision.value == "SHORTLIST"),
            "review_needed": sum(1 for r in results if r.agent_2_result.decision.value == "REVIEW"),
            "rejected": sum(1 for r in results if r.agent_2_result.decision.value == "REJECT")
        }
        
        total_time = time.time() - start_time
        
        batch_result = BatchProcessingResult(
            job_description_id=job_description.job_id,
            total_candidates=len(candidates),
            summary=summary,
            results=results,
            total_processing_time_seconds=round(total_time, 2)
        )
        
        logger.info(f"Batch processing complete in {total_time:.2f}s")
        logger.info(f"Summary: {summary}")
        
        return batch_result
    
    def process_cv_file(
        self,
        cv_path: str,
        candidate_id: str,
        job_description: JobDescription,
        available_roles: List[str]
    ) -> ProcessingResult:
        """
        Process a CV file (PDF)
        
        Args:
            cv_path: Path to PDF file
            candidate_id: Unique identifier
            job_description: Job description
            available_roles: Possible roles
        
        Returns:
            ProcessingResult
        """
        logger.info(f"Processing CV file: {cv_path}")
        
        # Extract text from PDF
        cv_text = self.pdf_extractor.extract_text(cv_path)
        
        if not cv_text:
            raise ValueError(f"Could not extract text from {cv_path}")
        
        # Clean text
        cv_text = self.text_cleaner.clean_text(cv_text)
        
        # Extract metadata
        metadata = self.text_cleaner.extract_metadata(cv_text)
        
        # Create candidate object
        candidate = Candidate(
            candidate_id=candidate_id,
            name=metadata.get("name", "Unknown"),
            email=metadata.get("email"),
            phone=metadata.get("phone"),
            cv_text=cv_text,
            cv_path=cv_path
        )
        
        # Process through agents
        return self.process_single_candidate(
            candidate=candidate,
            job_description=job_description,
            available_roles=available_roles
        )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Sample data
    test_candidate = Candidate(
        candidate_id="CAND001",
        name="John Doe",
        email="john@example.com",
        cv_text="""
        John Doe
        john@example.com
        
        PROFESSIONAL SUMMARY:
        5 years experienced Python developer
        
        SKILLS:
        Python, Django, Flask, PostgreSQL, MongoDB, AWS, Docker, Git
        
        EXPERIENCE:
        Senior Software Engineer | TechCorp | 2021-Present
        - Developed microservices architecture
        - Led team of 3 developers
        
        Software Engineer | StartupXYZ | 2019-2021
        - Built REST APIs using Django
        
        EDUCATION:
        Bachelor of Science in Computer Science
        University of Technology | 2015-2019
        """
    )
    
    test_jd = JobDescription(
        job_id="JD001",
        title="Senior Python Developer",
        department="Engineering",
        required_skills=["Python", "Django", "PostgreSQL", "AWS"],
        experience_years_min=3,
        experience_years_max=5,
        education_degree="Bachelor's",
        education_field="Computer Science",
        responsibilities=["Develop scalable applications"]
    )
    
    available_roles = [
        "Software Engineer",
        "Data Scientist",
        "DevOps Engineer",
        "Product Manager"
    ]
    
    # Process candidate
    result = orchestrator.process_single_candidate(
        candidate=test_candidate,
        job_description=test_jd,
        available_roles=available_roles
    )
    
    print("\n" + "="*60)
    print("PROCESSING RESULT")
    print("="*60)
    print(f"\nCandidate: {result.candidate_name}")
    print(f"Processing Time: {result.processing_time_seconds}s")
    print(f"\nAgent 1 - Classification:")
    print(f"  Classified Role: {result.agent_1_result.classified_role}")
    print(f"  Confidence: {result.agent_1_result.confidence:.2%}")
    print(f"\nAgent 2 - Evaluation:")
    print(f"  Overall Score: {result.agent_2_result.overall_score}")
    print(f"  Decision: {result.agent_2_result.decision.value}")
    print(f"  Skills Score: {result.agent_2_result.breakdown.skills_match.score}")
    print(f"  Experience Score: {result.agent_2_result.breakdown.experience_match.score}")
    print(f"  Education Score: {result.agent_2_result.breakdown.education_match.score}")
    print(f"\n  Recommendation: {result.agent_2_result.recommendation}")
    print(f"\n  Strengths:")
    for strength in result.agent_2_result.strengths:
        print(f"    - {strength}")
    print(f"\n  Areas for Improvement:")
    for area in result.agent_2_result.areas_for_improvement:
        print(f"    - {area}")
