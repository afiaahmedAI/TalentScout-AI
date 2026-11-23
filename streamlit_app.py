"""
Streamlit UI for TalentScout AI
Easy-to-use interface for HR teams
"""
import streamlit as st
import sys
import os
import json
import pandas as pd
from pathlib import Path
import time
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.orchestrator import MultiAgentOrchestrator
from models.models import JobDescription, Candidate
from utils.pdf_utils import PDFExtractor, TextCleaner

# Page configuration
st.set_page_config(
    page_title="TalentScout AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        color: #856404;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
if 'job_descriptions' not in st.session_state:
    st.session_state.job_descriptions = {}

# Header
st.markdown('<p class="main-header">🤖 TalentScout AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Candidate Screening & Evaluation</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # ✅ CHANGE 1: Model selection dropdowns REMOVED
    # Fixed models are used directly in MultiAgentOrchestrator defaults
    
    # Initialize system
    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Loading AI models... This may take 1-2 minutes..."):
            try:
                st.session_state.orchestrator = MultiAgentOrchestrator()
                st.success("✅ System initialized successfully!")
                
                # ✅ Auto-load job descriptions on initialize
                jd_file = Path("data/job_descriptions.json")
                if jd_file.exists():
                    with open(jd_file, 'r') as f:
                        jd_data = json.load(f)
                        st.session_state.job_descriptions = {
                            jd['job_id']: JobDescription(**jd)
                            for jd in jd_data['job_descriptions']
                        }
                    st.info(f"✅ Loaded {len(st.session_state.job_descriptions)} job descriptions")
                else:
                    st.warning("⚠️ Job descriptions file not found")
                    
            except Exception as e:
                st.error(f"❌ Failed to initialize: {e}")
    
    # ✅ CHANGE 2: Job Descriptions section in sidebar REMOVED

# Main content
if st.session_state.orchestrator is None:
    st.info("👈 Please initialize the system from the sidebar first")
    
    st.markdown("## 🎯 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Agent 1: Job Classification
        - Uses HuggingFace zero-shot classification
        - Automatically categorizes candidates into job roles
        - Provides confidence scores for each role
        """)
    
    with col2:
        st.markdown("""
        ### Agent 2: Candidate Evaluation
        - Evaluates candidates against job descriptions
        - Scores: Skills, Experience, Education
        - Provides detailed reasoning and recommendations
        """)
    
    st.markdown("---")
    st.markdown("### ✨ Key Features")
    
    features = [
        "🤖 **AI-Powered**: Uses state-of-the-art NLP models",
        "⚡ **Fast**: Process candidates in seconds",
        "📊 **Detailed Scoring**: Breakdown by skills, experience, education",
        "💡 **Explainable**: Clear reasoning for every decision",
        "📈 **Batch Processing**: Handle multiple CVs at once",
        "📋 **Export Results**: Download as CSV or JSON"
    ]
    
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            st.markdown(feature)

else:
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Single Candidate",
        "📁 Batch Processing",
        "📊 Results & Analytics",
        "📋 Job Descriptions"
    ])
    
    # TAB 1: Single Candidate Processing
    with tab1:
        st.header("📄 Process Single Candidate")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # CV Input
            st.subheader("1. Upload CV")
            cv_input_method = st.radio(
                "Choose input method",
                ["Upload PDF", "Paste CV Text", "Load Sample CV"],  # ✅ CHANGE 3: PDF upload added
                horizontal=True
            )
            
            cv_text = ""
            candidate_name = ""
            
            # ✅ CHANGE 3: PDF Upload - fully functional
            if cv_input_method == "Upload PDF":
                uploaded_pdf = st.file_uploader("Upload candidate's CV (PDF)", type=['pdf'])
                if uploaded_pdf:
                    with st.spinner("Extracting text from PDF..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                                tmp.write(uploaded_pdf.getvalue())
                                tmp_path = tmp.name
                            
                            extractor = PDFExtractor()
                            cv_text = extractor.extract_text(tmp_path)
                            cleaner = TextCleaner()
                            cv_text = cleaner.clean_text(cv_text)
                            os.unlink(tmp_path)
                            
                            # Extract only the name from first few lines
                            import re
                            candidate_name = 'Unknown'
                            first_lines = cv_text.strip().split('\n')[:5]
                            for line in first_lines:
                                line = line.strip()
                                # A name: 2-4 words, only letters/spaces, no digits or special chars
                                if line and re.match(r'^[A-Za-z]+(\s[A-Za-z]+){1,3}$', line):
                                    candidate_name = line
                                    break
                            
                            if cv_text:
                                st.success(f"✅ PDF extracted successfully!")
                                with st.expander("📄 Preview Extracted Text"):
                                    st.text_area("", cv_text, height=200, disabled=True)
                            else:
                                st.error("❌ Could not extract text from PDF")
                        except Exception as e:
                            st.error(f"❌ PDF processing failed: {e}")
                
                if cv_text:
                    candidate_name = st.text_input("Candidate Name", value=candidate_name)

            elif cv_input_method == "Paste CV Text":
                cv_text = st.text_area(
                    "Paste CV content here",
                    height=300,
                    placeholder="Paste the candidate's CV text here..."
                )
                candidate_name = st.text_input("Candidate Name", placeholder="John Doe")
            
            elif cv_input_method == "Load Sample CV":
                sample_cvs = {
                    "John Doe - Python Developer": "data/sample_cvs/john_doe.txt",
                    "Sarah Johnson - Data Scientist": "data/sample_cvs/sarah_johnson.txt",
                    "Michael Chen - DevOps Engineer": "data/sample_cvs/michael_chen.txt"
                }
                
                selected_sample = st.selectbox("Choose sample CV", list(sample_cvs.keys()))
                
                if st.button("Load Sample"):
                    sample_path = Path(sample_cvs[selected_sample])
                    if sample_path.exists():
                        with open(sample_path, 'r', encoding='utf-8') as f:
                            cv_text = f.read()
                        candidate_name = selected_sample.split(" - ")[0]
                        
                        # Store in session state
                        st.session_state['loaded_cv_text'] = cv_text
                        st.session_state['loaded_candidate_name'] = candidate_name
                        st.success(f"✅ Loaded: {selected_sample}")
                
                # Display loaded CV from session state
                if 'loaded_cv_text' in st.session_state:
                    cv_text = st.session_state['loaded_cv_text']
                    candidate_name = st.session_state['loaded_candidate_name']
                    st.text_area("CV Content (Loaded)", cv_text, height=200, disabled=True)
                    st.info(f"📝 Candidate: {candidate_name}")
        
        with col2:
            # Job Selection
            st.subheader("2. Select Job")
            
            if st.session_state.job_descriptions:
                job_options = {
                    f"{jd.title} ({jd.job_id})": jd_id
                    for jd_id, jd in st.session_state.job_descriptions.items()
                }
                
                selected_job = st.selectbox(
                    "Job Description",
                    list(job_options.keys())
                )
                
                job_id = job_options[selected_job]
                job_desc = st.session_state.job_descriptions[job_id]
                
                with st.expander("📋 View Job Details"):
                    st.write(f"**Title:** {job_desc.title}")
                    st.write(f"**Department:** {job_desc.department}")
                    st.write(f"**Experience:** {job_desc.experience_years_min}-{job_desc.experience_years_max} years")
                    st.write(f"**Education:** {job_desc.education_degree}")
                    st.write("**Required Skills:**")
                    st.write(", ".join(job_desc.required_skills))
            else:
                st.warning("No job descriptions loaded")
                job_desc = None
            
            # Available Roles
            st.subheader("3. Classification Roles")
            available_roles = st.multiselect(
                "Select possible roles",
                [
                    "Software Engineer",
                    "Data Scientist",
                    "DevOps Engineer",
                    "Frontend Developer",
                    "Product Manager",
                    "Backend Developer",
                    "Full Stack Developer",
                    "ML Engineer"
                ],
                default=["Software Engineer", "Data Scientist", "DevOps Engineer"]
            )
        
        st.markdown("---")
        
        # Process button
        if st.button("🚀 Process Candidate", type="primary", use_container_width=True):
            if not cv_text:
                st.error("❌ Please provide CV text")
            elif not candidate_name:
                st.error("❌ Please provide candidate name")
            elif not job_desc:
                st.error("❌ Please select a job description")
            elif not available_roles:
                st.error("❌ Please select at least one role")
            else:
                with st.spinner("🤖 AI is analyzing the candidate..."):
                    try:
                        # Create candidate object
                        candidate = Candidate(
                            candidate_id=f"CAND_{int(time.time())}",
                            name=candidate_name,
                            cv_text=cv_text
                        )
                        
                        # Process
                        result = st.session_state.orchestrator.process_single_candidate(
                            candidate=candidate,
                            job_description=job_desc,
                            available_roles=available_roles
                        )
                        
                        # Store result
                        st.session_state.processing_results.append(result)
                        
                        # Display results
                        st.success(f"✅ Processing complete in {result.processing_time_seconds}s")
                        
                        # Results display
                        st.markdown("---")
                        st.markdown("## 📊 Evaluation Results")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Overall Score",
                                f"{result.agent_2_result.overall_score:.1f}",
                                f"{result.agent_2_result.decision.value}"
                            )
                        
                        with col2:
                            st.metric(
                                "Skills Match",
                                f"{result.agent_2_result.breakdown.skills_match.score:.1f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Experience",
                                f"{result.agent_2_result.breakdown.experience_match.score:.1f}"
                            )
                        
                        with col4:
                            st.metric(
                                "Education",
                                f"{result.agent_2_result.breakdown.education_match.score:.1f}"
                            )
                        
                        # Classification Result
                        st.markdown("### 🏷️ Agent 1: Job Classification")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Classified Role:** {result.agent_1_result.classified_role}")
                            st.write(f"**Confidence:** {result.agent_1_result.confidence:.1%}")
                        
                        with col2:
                            st.write("**All Role Scores:**")
                            for role, score in sorted(
                                result.agent_1_result.all_scores.items(),
                                key=lambda x: x[1],
                                reverse=True
                            ):
                                st.write(f"- {role}: {score:.1%}")
                        
                        # Evaluation Result
                        st.markdown("### 📋 Agent 2: Detailed Evaluation")
                        
                        # Decision
                        decision_color = {
                            "SHORTLIST": "success",
                            "REVIEW": "warning",
                            "REJECT": "error"
                        }
                        
                        decision_box = decision_color.get(result.agent_2_result.decision.value, "info")
                        
                        if decision_box == "success":
                            st.success(f"**Decision:** {result.agent_2_result.decision.value}")
                        elif decision_box == "warning":
                            st.warning(f"**Decision:** {result.agent_2_result.decision.value}")
                        else:
                            st.error(f"**Decision:** {result.agent_2_result.decision.value}")
                        
                        st.info(f"**Recommendation:** {result.agent_2_result.recommendation}")
                        
                        # Detailed breakdown
                        with st.expander("📊 Detailed Score Breakdown", expanded=True):
                            st.markdown("#### Skills Match")
                            st.write(result.agent_2_result.breakdown.skills_match.reasoning)
                            st.write(f"**Matched:** {', '.join(result.agent_2_result.breakdown.skills_match.matched_skills)}")
                            if result.agent_2_result.breakdown.skills_match.missing_skills:
                                st.write(f"**Missing:** {', '.join(result.agent_2_result.breakdown.skills_match.missing_skills)}")
                            
                            st.markdown("#### Experience Match")
                            st.write(result.agent_2_result.breakdown.experience_match.reasoning)
                            
                            st.markdown("#### Education Match")
                            st.write(result.agent_2_result.breakdown.education_match.reasoning)
                        
                        # Strengths and improvements
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### ✅ Strengths")
                            for strength in result.agent_2_result.strengths:
                                st.write(f"- {strength}")
                        
                        with col2:
                            st.markdown("#### 🎯 Areas for Improvement")
                            for area in result.agent_2_result.areas_for_improvement:
                                st.write(f"- {area}")
                        
                        # Justification
                        with st.expander("💡 AI Justification"):
                            st.write(result.agent_2_result.justification)
                        
                    except Exception as e:
                        st.error(f"❌ Processing failed: {e}")
                        st.exception(e)
    
    # TAB 2: Batch Processing - ✅ CHANGE 4: Fully functional
    with tab2:
        st.header("📁 Batch Processing")
        
        files = st.file_uploader(
            "Upload multiple CV PDFs",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if files:
            st.success(f"✅ {len(files)} file(s) uploaded")
            for i, f in enumerate(files, 1):
                st.write(f"{i}. {f.name}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.session_state.job_descriptions:
                job_options_batch = {
                    f"{jd.title} ({jd.job_id})": jd_id
                    for jd_id, jd in st.session_state.job_descriptions.items()
                }
                selected_job_batch = st.selectbox(
                    "Job Description",
                    list(job_options_batch.keys()),
                    key="batch_job_select"
                )
                job_id_batch = job_options_batch[selected_job_batch]
                job_desc_batch = st.session_state.job_descriptions[job_id_batch]
            else:
                st.warning("No job descriptions loaded")
                job_desc_batch = None
        
        with col2:
            roles_batch = st.multiselect(
                "Select possible roles",
                [
                    "Software Engineer",
                    "Data Scientist",
                    "DevOps Engineer",
                    "Frontend Developer",
                    "Product Manager",
                    "Backend Developer",
                    "Full Stack Developer",
                    "ML Engineer"
                ],
                default=["Software Engineer", "Data Scientist", "DevOps Engineer"],
                key="batch_roles_select"
            )
        
        if st.button("🚀 Process All Candidates", type="primary", use_container_width=True):
            if not files:
                st.error("❌ Please upload at least one PDF file")
            elif not job_desc_batch:
                st.error("❌ Please select a job description")
            elif not roles_batch:
                st.error("❌ Please select at least one role")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                extractor = PDFExtractor()
                cleaner = TextCleaner()
                
                for i, file in enumerate(files):
                    try:
                        status_text.text(f"Processing {i+1}/{len(files)}: {file.name}...")
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                            tmp.write(file.getvalue())
                            tmp_path = tmp.name
                        
                        cv_text_batch = extractor.extract_text(tmp_path)
                        cv_text_batch = cleaner.clean_text(cv_text_batch)
                        meta = cleaner.extract_metadata(cv_text_batch)
                        cand_name = meta.get('name', f"Candidate_{i+1}")
                        os.unlink(tmp_path)
                        
                        if cv_text_batch:
                            candidate = Candidate(
                                candidate_id=f"BATCH_{int(time.time())}_{i}",
                                name=cand_name,
                                cv_text=cv_text_batch
                            )
                            result = st.session_state.orchestrator.process_single_candidate(
                                candidate=candidate,
                                job_description=job_desc_batch,
                                available_roles=roles_batch
                            )
                            batch_results.append(result)
                            st.session_state.processing_results.append(result)
                        else:
                            st.warning(f"⚠️ Could not extract text from: {file.name}")
                        
                        progress_bar.progress((i + 1) / len(files))
                    
                    except Exception as e:
                        st.warning(f"⚠️ Failed to process {file.name}: {e}")
                
                status_text.empty()
                st.success(f"✅ Batch processing complete! Processed {len(batch_results)}/{len(files)} candidates.")
                
                if batch_results:
                    # Summary metrics
                    st.markdown("### 📊 Batch Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Processed", len(batch_results))
                    with col2:
                        shortlisted = sum(1 for r in batch_results if r.agent_2_result.decision.value == "SHORTLIST")
                        st.metric("Shortlisted", shortlisted)
                    with col3:
                        review = sum(1 for r in batch_results if r.agent_2_result.decision.value == "REVIEW")
                        st.metric("Review Needed", review)
                    with col4:
                        rejected = sum(1 for r in batch_results if r.agent_2_result.decision.value == "REJECT")
                        st.metric("Rejected", rejected)
                    
                    # Results table
                    st.markdown("### 📋 Batch Results")
                    results_data = []
                    for r in batch_results:
                        results_data.append({
                            "Name": r.candidate_name,
                            "Classified Role": r.agent_1_result.classified_role,
                            "Confidence": f"{r.agent_1_result.confidence:.1%}",
                            "Overall Score": r.agent_2_result.overall_score,
                            "Skills": r.agent_2_result.breakdown.skills_match.score,
                            "Experience": r.agent_2_result.breakdown.experience_match.score,
                            "Education": r.agent_2_result.breakdown.education_match.score,
                            "Decision": r.agent_2_result.decision.value,
                            "Processing Time": f"{r.processing_time_seconds}s"
                        })
                    
                    df_batch = pd.DataFrame(results_data)
                    st.dataframe(df_batch, use_container_width=True)
                    
                    # Export
                    st.subheader("💾 Export Batch Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_batch = df_batch.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv_batch,
                            "batch_results.csv",
                            "text/csv"
                        )
                    
                    with col2:
                        json_str_batch = json.dumps(
                            [r.dict() for r in batch_results],
                            default=str,
                            indent=2
                        )
                        st.download_button(
                            "📥 Download JSON",
                            json_str_batch,
                            "batch_results.json",
                            "application/json"
                        )
    
    # TAB 3: Results & Analytics (same as original)
    with tab3:
        st.header("📊 Results & Analytics")
        
        if not st.session_state.processing_results:
            st.info("No results yet. Process some candidates first!")
        else:
            results = st.session_state.processing_results
            
            # Summary metrics
            st.subheader("📈 Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Processed", len(results))
            
            with col2:
                shortlisted = sum(1 for r in results if r.agent_2_result.decision.value == "SHORTLIST")
                st.metric("Shortlisted", shortlisted)
            
            with col3:
                review = sum(1 for r in results if r.agent_2_result.decision.value == "REVIEW")
                st.metric("Review Needed", review)
            
            with col4:
                rejected = sum(1 for r in results if r.agent_2_result.decision.value == "REJECT")
                st.metric("Rejected", rejected)
            
            # Results table
            st.subheader("📋 All Results")
            
            results_data = []
            for r in results:
                results_data.append({
                    "Name": r.candidate_name,
                    "Classified Role": r.agent_1_result.classified_role,
                    "Confidence": f"{r.agent_1_result.confidence:.1%}",
                    "Overall Score": r.agent_2_result.overall_score,
                    "Skills": r.agent_2_result.breakdown.skills_match.score,
                    "Experience": r.agent_2_result.breakdown.experience_match.score,
                    "Education": r.agent_2_result.breakdown.education_match.score,
                    "Decision": r.agent_2_result.decision.value,
                    "Processing Time": f"{r.processing_time_seconds}s"
                })
            
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)
            
            # Export options
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "candidate_results.csv",
                    "text/csv"
                )
            
            with col2:
                json_str = json.dumps(
                    [r.dict() for r in results],
                    default=str,
                    indent=2
                )
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    "candidate_results.json",
                    "application/json"
                )
    
    # TAB 4: Job Descriptions (same as original)
    with tab4:
        st.header("📋 Job Descriptions")
        
        if st.session_state.job_descriptions:
            for jd_id, jd in st.session_state.job_descriptions.items():
                with st.expander(f"{jd.title} ({jd_id})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Department:** {jd.department}")
                        st.write(f"**Experience:** {jd.experience_years_min}-{jd.experience_years_max} years")
                        st.write(f"**Education:** {jd.education_degree} in {jd.education_field}")
                    
                    with col2:
                        st.write("**Required Skills:**")
                        for skill in jd.required_skills:
                            st.write(f"- {skill}")
                    
                    st.write("**Description:**")
                    st.write(jd.description)
                    
                    st.write("**Responsibilities:**")
                    for resp in jd.responsibilities:
                        st.write(f"- {resp}")
        else:
            st.warning("No job descriptions loaded")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9rem;'>
        🤖 TalentScout AI | Powered by HuggingFace 🤗
    </div>
    """,
    unsafe_allow_html=True
)
