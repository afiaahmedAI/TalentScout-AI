# 🎯 TalentScout AI

### Stop drowning in résumés. Let two AI agents do the first read for you.

Recruiters spend hours opening CVs that were never a fit. **TalentScout AI** turns that first screening pass into a few clicks: drop in a CV, pick a role, and get back a ranked verdict, a score breakdown, and a plain-English explanation of *why*.

No paid APIs. No black-box scores. Just a transparent pipeline you can read, tune and trust.

---

## ✨ What you get

| | |
|---|---|
| 🧭 **Instant role matching** | Agent 1 reads the CV and places the candidate in the right job family with a confidence score. |
| 📊 **Explainable scoring** | Agent 2 breaks the result into skills, experience and education, listing what matched and what is missing. |
| ✅ **Clear decisions** | Every candidate lands on **SHORTLIST**, **REVIEW** or **REJECT**, with strengths and growth areas spelled out. |
| 📄 **Real CVs welcome** | Upload a PDF, paste text, or load one of the built-in samples. |
| 🗂️ **Batch ready** | Screen a whole stack of candidates at once and download the results. |
| 💸 **Free to run** | Built on open HuggingFace models. No API bill. |

## 🧠 How the agents think

```
   CV (PDF or text)  +  Job description
              │
              ▼
   ┌──────────────────────────┐
   │ Agent 1 · Classifier     │   facebook/bart-large-mnli
   │ "Which role fits best?"  │   zero-shot classification
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ Agent 2 · Evaluator      │   google/flan-t5-base
   │ "How well do they fit?"  │   scoring + written reasoning
   └────────────┬─────────────┘
                ▼
   Score · Decision · Strengths · Gaps
```

**The scoring formula is deliberately simple and fully visible:**

> `Overall = Skills × 50%  +  Experience × 30%  +  Education × 20%`

| Overall score | Decision |
|---|---|
| 70 and above | ✅ SHORTLIST |
| 50 to 69 | 🟡 REVIEW |
| Below 50 | ❌ REJECT |

Want a different hiring philosophy? Change the weights and thresholds in `agents/agent_2_evaluator.py`. It is a few lines, not a retraining project.

## 🚀 Try it in minutes

```bash
git clone https://github.com/afiaahmedAI/TalentScout-AI.git
cd TalentScout-AI

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # optional: add your free HuggingFace token
streamlit run streamlit_app.py
```

Open http://localhost:8501, click **Initialize System**, and screen your first candidate.
The first launch downloads the models (about 1–2 GB, one time only). After that they are cached.

## 🧪 Batteries included

- 5 ready-made job descriptions: Senior Python Developer, Data Scientist, DevOps Engineer, Frontend Developer, Product Manager
- 3 sample CVs so you can see results before you bring your own
- A four-tab Streamlit interface for single candidates, batch runs and exports

## 📁 Project map

```
TalentScout-AI/
├── agents/
│   ├── agent_1_classifier.py   # role classification
│   ├── agent_2_evaluator.py    # scoring, decisions, reasoning
│   └── orchestrator.py         # coordinates both agents
├── models/models.py            # Pydantic data models
├── utils/pdf_utils.py          # PDF text + contact extraction
├── data/                       # sample jobs and CVs
├── streamlit_app.py            # the UI
└── requirements.txt
```

## 🛠️ Make it yours

- **New role?** Add an entry to `data/job_descriptions.json`.
- **Different priorities?** Adjust the scoring weights.
- **Stricter or friendlier screening?** Move the decision thresholds.

## 🗺️ Roadmap

- [ ] REST API endpoints
- [ ] Interview question generator
- [ ] Multi-language CV support
- [ ] Smarter résumé parsing

## 🤝 Contributing

Ideas, bug reports and pull requests are welcome.

---

<sub>Built with Python, Streamlit, LangChain and HuggingFace 🤗</sub>
