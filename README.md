# FAILSAFE — Early Student Failure Risk Detection

> An explainable AI system that predicts which students are at risk of academic failure — before it's too late.


---

##  Live Demo

 **[frabjous-sopapillas-2f9086.netlify.app](https://frabjous-sopapillas-2f9086.netlify.app)**

No setup needed — open the link, upload a CSV, get results instantly.

> First request may take ~60 seconds (Render free tier wakes up from sleep). Wait a moment and retry.

---

## What it does

Upload a student dataset and FAILSAFE will:
- Predict which students are **at risk of failing**
- Assign a **risk score** (0–100%) and level (HIGH / MEDIUM / LOW)
- Show **why** using SHAP feature importance charts
- Suggest **personalised interventions** for each at-risk student

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/ankit-xd03/FAILSAFE.git
cd FAILSAFE
```

### 2. create a virtual environment
```bash
python3 -m venv venv
```

### 3. Activate
```bash
venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Start the backend
```bash
python3 -m uvicorn backend:app --reload
```
Backend runs at `http://127.0.0.1:8000`

### 6. Open the frontend
Open `index.html` directly in your browser — done

---

## CSV Format

Upload any student CSV with these columns:

| Column | Description |
|--------|-------------|
| `school` | School name |
| `sex` | Student gender |
| `age` | Age |
| `absences` | Number of absences |
| `failures` | Past class failures |
| `studytime` | Weekly study hours |

Sample files included: `student-mat.csv` ·

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML Model | XGBoost (5-fold stratified cross-validation) |
| Explainability | SHAP TreeExplainer |
| Backend | FastAPI |
| Frontend | React 18  |
| Backend Hosting | Render |
| Frontend Hosting | Netlify |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Bulk risk prediction |
| `/interventions` | POST | SHAP-based intervention plans |
| `/batch-report` | POST | Download full CSV report |
| `/student/{index}` | GET | Single student detail |

---

## Team

**Ankit Kumar Sinha · Vaidik Maheshwari · Bhawesh Kumar Agrawal**
Problem Statement by Coding Club IIT Guwahati
