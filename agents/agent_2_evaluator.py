"""
Agent 2: Candidate Scoring and Evaluation Agent
Uses HuggingFace models for evaluation and scoring
"""
from transformers import pipeline
import re
import logging
from typing import List, Dict, Tuple
from models.models import (
    EvaluationResult, EvaluationBreakdown,
    SkillsMatchResult, ExperienceMatchResult, EducationMatchResult,
    JobDescription, DecisionType
)
from datetime import datetime

logger = logging.getLogger(__name__)


class CandidateEvaluationAgent:
    """Agent 2: Evaluates and scores candidates"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize evaluation agent
        
        Args:
            model_name: HuggingFace model for text generation
                       Options:
                       - google/flan-t5-base (free, good for reasoning)
                       - google/flan-t5-large (better but slower)
                       - facebook/bart-large (alternative)
        """
        logger.info(f"Loading evaluation model: {model_name}")
        try:
            self.generator = pipeline(
                "text2text-generation",
                model=model_name,
                device=-1,
                max_length=512
            )
            logger.info("Evaluation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from CV text"""
        # Common skill keywords
        skill_indicators = [
            "skills:", "technical skills:", "technologies:",
            "tools:", "programming languages:", "expertise:"
        ]
        
        skills = set()
        lines = text.lower().split('\n')
        
        capture = False
        for line in lines:
            # Check if line contains skill indicator
            if any(indicator in line for indicator in skill_indicators):
                capture = True
                continue
            
            # Stop capturing at next section
            if capture and any(x in line for x in ['experience:', 'education:', 'projects:']):
                break
            
            if capture:
                # Extract words that look like skills
                words = re.findall(r'\b[A-Za-z][A-Za-z0-9+#.]+\b', line)
                skills.update(words)
        
        # Common tech skills (backup extraction)
        common_skills = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#',
            'react', 'angular', 'vue', 'django', 'flask', 'fastapi',
            'sql', 'postgresql', 'mongodb', 'mysql',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'git', 'jenkins', 'terraform', 'ansible'
        ]
        
        text_lower = text.lower()
        for skill in common_skills:
            if skill in text_lower:
                skills.add(skill)
        
        return list(skills)
    
    def extract_experience_years(self, text: str) -> float:
        """Extract years of experience from CV"""
        # Look for patterns like "5 years", "3+ years", etc.
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience[:\s]+(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience'
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))
        
        # Count years from work history (rough estimate)
        year_mentions = re.findall(r'\b(19|20)\d{2}\b', text)
        if len(year_mentions) >= 2:
            years = [int(y) for y in year_mentions]
            return max(years) - min(years)
        
        return 0.0
    
    def extract_education(self, text: str) -> Dict[str, str]:
        """Extract education information"""
        text_lower = text.lower()
        
        # Detect degree
        degree = "Unknown"
        if "phd" in text_lower or "ph.d" in text_lower or "doctorate" in text_lower:
            degree = "PhD"
        elif "master" in text_lower or "msc" in text_lower or "ms" in text_lower:
            degree = "Master's"
        elif "bachelor" in text_lower or "bsc" in text_lower or "bs" in text_lower:
            degree = "Bachelor's"
        elif "associate" in text_lower:
            degree = "Associate"
        
        # Detect field
        field = "Unknown"
        fields = {
            "computer science": ["computer science", "cs", "cse"],
            "engineering": ["engineering", "engineer"],
            "data science": ["data science", "data analytics"],
            "information technology": ["information technology", "it"],
            "mathematics": ["mathematics", "math"],
            "business": ["business", "mba", "management"]
        }
        
        for field_name, keywords in fields.items():
            if any(kw in text_lower for kw in keywords):
                field = field_name
                break
        
        return {"degree": degree, "field": field}
    
    def calculate_skills_match(
        self,
        candidate_skills: List[str],
        required_skills: List[str]
    ) -> SkillsMatchResult:
        """Calculate skills matching score"""
        candidate_skills_lower = [s.lower() for s in candidate_skills]
        required_skills_lower = [s.lower() for s in required_skills]
        
        matched = set(candidate_skills_lower) & set(required_skills_lower)
        missing = set(required_skills_lower) - set(candidate_skills_lower)
        extra = set(candidate_skills_lower) - set(required_skills_lower)
        
        # Calculate score
        if not required_skills:
            score = 100.0
        else:
            match_percentage = (len(matched) / len(required_skills)) * 100
            # Bonus for extra skills (max 10 points)
            bonus = min(len(extra) * 2, 10)
            score = min(match_percentage + bonus, 100.0)
        
        # Generate reasoning
        reasoning = f"Matched {len(matched)} out of {len(required_skills)} required skills. "
        if extra:
            reasoning += f"Candidate has {len(extra)} additional relevant skills. "
        if missing:
            reasoning += f"Missing: {', '.join(list(missing)[:3])}."
        
        return SkillsMatchResult(
            score=round(score, 2),
            matched_skills=list(matched),
            missing_skills=list(missing),
            extra_skills=list(extra),
            reasoning=reasoning
        )
    
    def calculate_experience_match(
        self,
        candidate_years: float,
        required_min: int,
        required_max: int
    ) -> ExperienceMatchResult:
        """Calculate experience matching score"""
        if candidate_years >= required_min and candidate_years <= required_max:
            score = 100.0
            relevance = "Perfect"
            reasoning = f"Candidate has {candidate_years} years, which fits perfectly within the {required_min}-{required_max} year requirement."
        elif candidate_years > required_max:
            excess = candidate_years - required_max
            penalty = min(excess * 5, 20)
            score = max(100 - penalty, 80.0)
            relevance = "Overqualified"
            reasoning = f"Candidate has {candidate_years} years, exceeding the {required_max} year maximum by {excess} years. Overqualified but acceptable."
        else:
            deficit = required_min - candidate_years
            score = max(100 - (deficit * 20), 0.0)
            relevance = "Underqualified"
            reasoning = f"Candidate has {candidate_years} years, which is {deficit} years below the {required_min} year minimum requirement."
        
        return ExperienceMatchResult(
            score=round(score, 2),
            candidate_years=candidate_years,
            required_min=required_min,
            required_max=required_max,
            relevance=relevance,
            reasoning=reasoning
        )
    
    def calculate_education_match(
        self,
        candidate_education: Dict[str, str],
        required_degree: str,
        required_field: str = None
    ) -> EducationMatchResult:
        """Calculate education matching score"""
        candidate_degree = candidate_education.get("degree", "Unknown")
        candidate_field = candidate_education.get("field", "Unknown")
        
        # Degree level hierarchy
        degree_levels = {
            "PhD": 4,
            "Master's": 3,
            "Bachelor's": 2,
            "Associate": 1,
            "Unknown": 0
        }
        
        candidate_level = degree_levels.get(candidate_degree, 0)
        required_level = degree_levels.get(required_degree, 0)
        
        # Calculate score
        if candidate_level >= required_level:
            score = 100.0
        else:
            deficit = required_level - candidate_level
            score = max(100 - (deficit * 30), 0.0)
        
        # Field match
        field_match = False
        if required_field and candidate_field != "Unknown":
            if required_field.lower() in candidate_field.lower() or \
               candidate_field.lower() in required_field.lower():
                field_match = True
                score = min(score + 10, 100.0)  # Bonus for field match
        
        # Reasoning
        reasoning = f"Candidate has {candidate_degree} in {candidate_field}. "
        if candidate_level >= required_level:
            reasoning += f"Meets or exceeds {required_degree} requirement."
        else:
            reasoning += f"Below {required_degree} requirement."
        
        return EducationMatchResult(
            score=round(score, 2),
            candidate_degree=candidate_degree,
            required_degree=required_degree,
            field_match=field_match,
            reasoning=reasoning
        )
    
    def make_decision(
        self,
        overall_score: float,
        skills_score: float
    ) -> Tuple[DecisionType, str]:
        """Make hiring decision based on scores"""
        # Minimum skills threshold
        if skills_score < 50:
            return DecisionType.REJECT, "Insufficient technical skills for the role."
        
        # Overall score-based decision
        if overall_score >= 70:
            return DecisionType.SHORTLIST, "Strong candidate with excellent match to requirements."
        elif overall_score >= 50:
            return DecisionType.REVIEW, "Potentially qualified candidate, requires further review."
        else:
            return DecisionType.REJECT, "Does not meet minimum requirements for the role."
    
    def generate_strengths_and_improvements(
        self,
        breakdown: EvaluationBreakdown
    ) -> Tuple[List[str], List[str]]:
        """Generate strengths and areas for improvement"""
        strengths = []
        improvements = []
        
        # Analyze skills
        if breakdown.skills_match.score >= 80:
            strengths.append(f"Strong technical skills with {len(breakdown.skills_match.matched_skills)} matching skills")
        elif breakdown.skills_match.score < 60:
            improvements.append(f"Need to develop skills in: {', '.join(breakdown.skills_match.missing_skills[:3])}")
        
        # Analyze experience
        if breakdown.experience_match.score >= 90:
            strengths.append(f"Excellent experience level ({breakdown.experience_match.candidate_years} years)")
        elif breakdown.experience_match.relevance == "Underqualified":
            improvements.append("Gain more years of relevant experience")
        
        # Analyze education
        if breakdown.education_match.score >= 90:
            strengths.append("Strong educational background")
        elif breakdown.education_match.score < 70:
            improvements.append("Consider pursuing higher education degree")
        
        # Extra skills
        if breakdown.skills_match.extra_skills:
            strengths.append(f"Additional valuable skills: {', '.join(breakdown.skills_match.extra_skills[:3])}")
        
        return strengths, improvements
    
    def evaluate_candidate(
        self,
        candidate_id: str,
        candidate_name: str,
        classified_role: str,
        cv_text: str,
        job_description: JobDescription
    ) -> EvaluationResult:
        """
        Evaluate candidate against job description
        
        Args:
            candidate_id: Unique candidate identifier
            candidate_name: Candidate's name
            classified_role: Role classified by Agent 1
            cv_text: CV text content
            job_description: Job description to match against
        
        Returns:
            EvaluationResult with scores and recommendation
        """
        logger.info(f"Evaluating candidate: {candidate_name} for role: {classified_role}")
        
        # Extract candidate information
        candidate_skills = self.extract_skills(cv_text)
        candidate_experience = self.extract_experience_years(cv_text)
        candidate_education = self.extract_education(cv_text)
        
        # Calculate individual scores
        skills_match = self.calculate_skills_match(
            candidate_skills,
            job_description.required_skills
        )
        
        experience_match = self.calculate_experience_match(
            candidate_experience,
            job_description.experience_years_min,
            job_description.experience_years_max
        )
        
        education_match = self.calculate_education_match(
            candidate_education,
            job_description.education_degree,
            job_description.education_field
        )
        
        # Create breakdown
        breakdown = EvaluationBreakdown(
            skills_match=skills_match,
            experience_match=experience_match,
            education_match=education_match
        )
        
        # Calculate overall score (weighted average)
        overall_score = (
            skills_match.score * 0.5 +
            experience_match.score * 0.3 +
            education_match.score * 0.2
        )
        
        # Make decision
        decision, justification = self.make_decision(overall_score, skills_match.score)
        
        # Generate strengths and improvements
        strengths, improvements = self.generate_strengths_and_improvements(breakdown)
        
        # Recommendation
        if decision == DecisionType.SHORTLIST:
            recommendation = f"Highly recommend for {classified_role} position"
        elif decision == DecisionType.REVIEW:
            recommendation = f"Consider for {classified_role} with additional screening"
        else:
            recommendation = f"Not recommended for {classified_role} at this time"
        
        logger.info(f"Evaluation complete: {decision.value} (score: {overall_score:.2f})")
        
        return EvaluationResult(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            classified_role=classified_role,
            overall_score=round(overall_score, 2),
            breakdown=breakdown,
            decision=decision,
            recommendation=recommendation,
            justification=justification,
            strengths=strengths,
            areas_for_improvement=improvements,
            timestamp=datetime.utcnow()
        )


# Example usage
if __name__ == "__main__":
    agent = CandidateEvaluationAgent()
    
    test_cv = """
    John Doe
    john.doe@email.com
    
    SKILLS:
    Python, Django, PostgreSQL, Docker, AWS, Git
    
    EXPERIENCE:
    5 years of software development
    
    EDUCATION:
    Bachelor's in Computer Science
    """
    
    jd = JobDescription(
        job_id="JD001",
        title="Senior Python Developer",
        department="Engineering",
        required_skills=["Python", "Django", "PostgreSQL", "AWS"],
        experience_years_min=3,
        experience_years_max=5,
        education_degree="Bachelor's",
        education_field="Computer Science",
        responsibilities=["Develop web applications"]
    )
    
    result = agent.evaluate_candidate(
        candidate_id="TEST001",
        candidate_name="John Doe",
        classified_role="Software Engineer",
        cv_text=test_cv,
        job_description=jd
    )
    
    print(f"\nOverall Score: {result.overall_score}")
    print(f"Decision: {result.decision.value}")
    print(f"\nBreakdown:")
    print(f"  Skills: {result.breakdown.skills_match.score}")
    print(f"  Experience: {result.breakdown.experience_match.score}")
    print(f"  Education: {result.breakdown.education_match.score}")
