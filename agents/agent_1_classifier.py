"""
Agent 1: Job Title Classification Agent
Uses HuggingFace zero-shot classification
"""
from transformers import pipeline
import logging
from typing import List, Dict
from models.models import ClassificationResult
from datetime import datetime

logger = logging.getLogger(__name__)


class JobClassificationAgent:
    """Agent 1: Classifies candidates into job roles"""
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize classification agent
        
        Args:
            model_name: HuggingFace model for zero-shot classification
                       Default: facebook/bart-large-mnli (good for classification)
                       Alternative: MoritzLaurer/deberta-v3-base-zeroshot-v1
        """
        logger.info(f"Loading classification model: {model_name}")
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1  # CPU (-1), for GPU use 0
            )
            logger.info("Classification model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def classify_candidate(
        self,
        candidate_id: str,
        candidate_name: str,
        cv_text: str,
        job_roles: List[str],
        multi_label: bool = False
    ) -> ClassificationResult:
        """
        Classify candidate into job role
        
        Args:
            candidate_id: Unique candidate identifier
            candidate_name: Candidate's name
            cv_text: CV text content
            job_roles: List of possible job roles
            multi_label: Whether to allow multiple labels
        
        Returns:
            ClassificationResult with classified role and scores
        """
        logger.info(f"Classifying candidate: {candidate_name}")
        
        # Truncate text if too long (model limit is ~512 tokens)
        max_chars = 2000
        if len(cv_text) > max_chars:
            cv_text = cv_text[:max_chars]
        
        try:
            # Perform classification
            result = self.classifier(
                cv_text,
                candidate_labels=job_roles,
                multi_label=multi_label
            )
            
            # Extract results
            classified_role = result['labels'][0]
            confidence = result['scores'][0]
            
            # Create score dictionary
            all_scores = {
                label: score 
                for label, score in zip(result['labels'], result['scores'])
            }
            
            logger.info(f"Classification complete: {classified_role} (confidence: {confidence:.2f})")
            
            return ClassificationResult(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                classified_role=classified_role,
                confidence=confidence,
                all_scores=all_scores,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise
    
    def batch_classify(
        self,
        candidates: List[Dict],
        job_roles: List[str]
    ) -> List[ClassificationResult]:
        """
        Classify multiple candidates
        
        Args:
            candidates: List of dicts with 'candidate_id', 'name', 'cv_text'
            job_roles: List of possible job roles
        
        Returns:
            List of ClassificationResult objects
        """
        results = []
        
        for candidate in candidates:
            try:
                result = self.classify_candidate(
                    candidate_id=candidate['candidate_id'],
                    candidate_name=candidate['name'],
                    cv_text=candidate['cv_text'],
                    job_roles=job_roles
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify {candidate.get('name')}: {e}")
                continue
        
        return results


# Example usage
if __name__ == "__main__":
    # Test the agent
    agent = JobClassificationAgent()
    
    test_cv = """
    John Doe
    Senior Python Developer
    
    SKILLS:
    - Python, Django, Flask
    - PostgreSQL, MongoDB
    - AWS, Docker, Kubernetes
    - REST APIs, Microservices
    
    EXPERIENCE:
    5 years of backend development experience
    Led team of 3 developers
    Built scalable web applications
    """
    
    roles = [
        "Software Engineer",
        "Data Scientist",
        "DevOps Engineer",
        "Product Manager",
        "Frontend Developer"
    ]
    
    result = agent.classify_candidate(
        candidate_id="TEST001",
        candidate_name="John Doe",
        cv_text=test_cv,
        job_roles=roles
    )
    
    print(f"\nClassified Role: {result.classified_role}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nAll Scores:")
    for role, score in result.all_scores.items():
        print(f"  {role}: {score:.2%}")
