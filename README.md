# TalentScout AI

TalentScout AI is a Streamlit application that screens candidate CVs against a job description. Two agents work in sequence: the first classifies the CV into a job role, and the second scores the candidate and explains the result. It runs on open HuggingFace models, so no paid API is needed.

## Features

- Classifies a CV into one of several job roles with a confidence score
- Scores skills, experience and education separately, and lists matched and missing skills
- Produces a decision (shortlist, review or reject) with strengths and areas for improvement
- Accepts PDF uploads, pasted text, or the bundled sample CVs
- Processes multiple CVs in one batch and lets you download the results

## How it works

1. **Classifier agent** (`facebook/bart-large-mnli`): zero-shot classification of the CV against the available roles.
2. **Evaluator agent** (`google/flan-t5-base`): extracts skills, years of experience and education, compares them with the selected job description, and generates a short written justification.
3. **Orchestrator**: runs both agents and combines their output into a single result.

The overall score is a weighted average:

```
overall = 0.5 * skills + 0.3 * experience + 0.2 * education
```

| Overall score | Decision |
|---|---|
| 70 or higher | Shortlist |
| 50 to 69 | Review |
| Below 50 | Reject |

The weights and thresholds are defined in `agents/agent_2_evaluator.py` and can be changed there.

## Getting started

Requirements: Python 3.8 or newer and about 4 GB of RAM.

```bash
git clone https://github.com/afiaahmedAI/TalentScout-AI.git
cd TalentScout-AI

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # optional: add a HuggingFace token
streamlit run streamlit_app.py
```

Open http://localhost:8501 and click **Initialize System** in the sidebar. The models are downloaded on the first run (roughly 1 to 2 GB) and cached for later runs.

## Sample data

The `data/` folder includes five job descriptions (Senior Python Developer, Data Scientist, DevOps Engineer, Frontend Developer, Product Manager) and three sample CVs, so the app can be tried without any files of your own.

To add a role, add an entry to `data/job_descriptions.json`.

## Project structure

```
agents/
  agent_1_classifier.py   role classification
  agent_2_evaluator.py    scoring and decisions
  orchestrator.py         coordinates the agents
models/models.py          Pydantic data models
utils/pdf_utils.py        PDF text extraction and cleaning
data/                     sample job descriptions and CVs
streamlit_app.py          user interface
requirements.txt
```

## Limitations

- Skill and education extraction is rule-based, so unusual CV layouts may be parsed poorly.
- Scores are a screening aid and should not replace a human review.
- The first run needs an internet connection to download the models.

## Possible next steps

- REST API
- Interview question generation
- Multi-language CV support
- Better CV parsing
