import io
import logging
import pickle
from typing import Optional
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("failsafe")
app = FastAPI(
    title="FAILSAFE API",
    description="Student at-risk prediction with Explainable AI (XGBoost + SHAP)",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

MODEL_COLUMNS: list[str] = model.get_booster().feature_names
logger.info(f"Model loaded — {len(MODEL_COLUMNS)} features")

_last_results: list[dict] = []
INTERVENTION_PRIORITY = {
    "failures":  5,
    "absences":  4,
    "studytime": 3,
    "Dalc":      2,
    "Walc":      2,
    "goout":     2,
    "health":    1,
    "famsup":    1,
    "schoolsup": 1,
    "freetime":  1,
}
INTERVENTION_MAP = {
    "absences":  "Attendance monitoring required — extra classes recommended",
    "failures":  "Previous failure history — counselling session recommended",
    "goout":     "Social activity affecting studies — time management counselling recommended",
    "famsup":    "Lack of family support — parent meeting recommended",
    "schoolsup": "No school support enrolled — refer to support and scholarship program",
    "studytime": "Low study time — structured study schedule recommended",
    "health":    "Health issues detected — medical counselling recommended",
    "Dalc":      "Alcohol consumption concern — behavioral counselling recommended",
    "Walc":      "Alcohol consumption concern — behavioral counselling recommended",
    "freetime":  "Excess free time — extracurricular activities recommended",
}
def _risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    if prob >= 0.40:
        return "MEDIUM"
    return "LOW"

def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode and align columns with the trained model's feature set."""
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=MODEL_COLUMNS, fill_value=0)
    return df_encoded

def _validate_csv(df: pd.DataFrame) -> None:
    """
    Raise HTTPException if the uploaded CSV is clearly wrong.
    Checks: not empty, has at least some expected feature columns.
    """
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    expected_raw = {"school", "sex", "age", "absences", "failures",
                    "studytime", "G1", "G2", "G3"}
    present = set(df.columns)
    missing = expected_raw - present
    must_have = {"school", "sex", "age", "absences", "failures", "studytime"}
    truly_missing = must_have - present
    if truly_missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required columns: {sorted(truly_missing)}"
        )
async def _read_csv(file: UploadFile) -> pd.DataFrame:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    _validate_csv(df)
    return df
def _generate_interventions(shap_vals, feature_names: list) -> list[dict]:
    """
    Return a prioritised, deduplicated list of intervention dicts for one student.
    Only positive SHAP values (pushing toward at-risk) are used.
    Each dict: { intervention, shap_impact, feature, priority }
    """
    shap_dict = dict(zip(feature_names, shap_vals))
    risk_drivers = {k: v for k, v in shap_dict.items() if v > 0}
    top_features = sorted(risk_drivers.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    seen: set[str] = set()
    interventions: list[dict] = []

    for feature, shap_value in top_features:
        matched_key = next((k for k in INTERVENTION_MAP if k in feature), None)
        text = (INTERVENTION_MAP[matched_key] if matched_key
                else f"Review factor '{feature}' — faculty attention needed")

        if text not in seen:
            seen.add(text)
            priority = max(
                (INTERVENTION_PRIORITY.get(k, 0) for k in INTERVENTION_PRIORITY if k in feature),
                default=0
            )
            interventions.append({
                "intervention": text,
                "shap_impact": round(float(shap_value), 4),
                "feature": feature,
                "priority": priority,
            })

    interventions.sort(key=lambda x: (-x["priority"], -abs(x["shap_impact"])))
    return interventions

def _shap_chart_data(shap_vals, feature_names: list, top_n: int = 10) -> list[dict]:
    """Top-N features by |SHAP| for waterfall chart rendering on the frontend."""
    shap_dict = dict(zip(feature_names, shap_vals))
    top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return [{"feature": k, "shap_value": round(float(v), 4)} for k, v in top]
def _build_student_result(
    index: int,
    risk_prob: float,
    interventions: list[dict],
    shap_chart: list[dict],
    base_value: float,
) -> dict:
    return {
        "student_index": index,
        "at_risk": "YES" if risk_prob >= 0.5 else "NO",
        "risk_probability": round(float(risk_prob) * 100, 2),
        "risk_level": _risk_level(risk_prob),
        "interventions": interventions,
        "shap_chart_data": shap_chart,
        "base_value": round(float(base_value), 4),
    }

@app.post("/predict", summary="Bulk at-risk prediction")
async def predict(file: UploadFile = File(...)):
    """
    Upload a student CSV and get at-risk predictions with risk level for each student.
    Returns summary counts and per-student results.
    """
    df = await _read_csv(file)
    logger.info(f"/predict — {len(df)} students received")

    df_encoded = _preprocess(df)
    predictions = model.predict(df_encoded)
    probabilities = model.predict_proba(df_encoded)[:, 1]

    results = [
        {
            "student_index": i,
            "at_risk": "YES" if predictions[i] == 1 else "NO",
            "risk_probability": round(float(probabilities[i]) * 100, 2),
            "risk_level": _risk_level(probabilities[i]),
        }
        for i in range(len(df))
    ]

    at_risk = sum(1 for r in results if r["at_risk"] == "YES")
    level_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in results:
        level_counts[r["risk_level"]] += 1

    logger.info(f"/predict done — at_risk={at_risk}/{len(df)}")
    return {
        "total_students": len(results),
        "at_risk_count": at_risk,
        "safe_count": len(results) - at_risk,
        "risk_level_counts": level_counts,
        "predictions": results,
    }
@app.post("/interventions", summary="SHAP-based intervention plans")
async def interventions(file: UploadFile = File(...)):
    """
    Upload a student CSV and get per-student:
    - Prioritised, deduplicated intervention recommendations
    - Top-10 SHAP feature values for waterfall chart (frontend rendering)
    - Risk level (HIGH / MEDIUM / LOW)
    """
    global _last_results
    df = await _read_csv(file)
    logger.info(f"/interventions — {len(df)} students received")

    df_encoded = _preprocess(df)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_encoded)
    feature_names = df_encoded.columns.tolist()
    base_val = float(explainer.expected_value)

    results = []
    for i in range(len(df)):
        risk_prob = float(model.predict_proba(df_encoded.iloc[[i]])[0][1])
        plan = _generate_interventions(shap_values[i], feature_names)
        chart = _shap_chart_data(shap_values[i], feature_names)
        results.append(_build_student_result(i, risk_prob, plan, chart, base_val))

    _last_results = results  
    at_risk = sum(1 for r in results if r["at_risk"] == "YES")
    logger.info(f"/interventions done — at_risk={at_risk}/{len(df)}")

    return {"total_students": len(results), "interventions": results}
@app.post("/batch-report", summary="Download full CSV report")
async def batch_report(file: UploadFile = File(...)):
    """
    Upload a student CSV and download a complete intervention report as CSV.
    Each row = one student with risk level and all recommended interventions.
    Suitable for faculty to print or share with HODs.
    """
    df = await _read_csv(file)
    logger.info(f"/batch-report — {len(df)} students")

    df_encoded = _preprocess(df)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_encoded)
    feature_names = df_encoded.columns.tolist()

    rows = []
    for i in range(len(df)):
        risk_prob = float(model.predict_proba(df_encoded.iloc[[i]])[0][1])
        plan = _generate_interventions(shap_values[i], feature_names)
        top_shap = _shap_chart_data(shap_values[i], feature_names, top_n=3)

        rows.append({
            "student_index": i,
            "at_risk": "YES" if risk_prob >= 0.5 else "NO",
            "risk_level": _risk_level(risk_prob),
            "risk_probability_%": round(risk_prob * 100, 2),
            "top_risk_factor_1": top_shap[0]["feature"] if len(top_shap) > 0 else "",
            "top_risk_factor_2": top_shap[1]["feature"] if len(top_shap) > 1 else "",
            "top_risk_factor_3": top_shap[2]["feature"] if len(top_shap) > 2 else "",
            "intervention_1": plan[0]["intervention"] if len(plan) > 0 else "",
            "intervention_2": plan[1]["intervention"] if len(plan) > 1 else "",
            "intervention_3": plan[2]["intervention"] if len(plan) > 2 else "",
            "intervention_4": plan[3]["intervention"] if len(plan) > 3 else "",
            "intervention_5": plan[4]["intervention"] if len(plan) > 4 else "",
        })

    report_df = pd.DataFrame(rows)
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    logger.info(f"/batch-report CSV ready — {len(rows)} rows")
    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=failsafe_report.csv"},
    )


@app.get("/student/{index}", summary="Single student detail")
async def get_student(index: int):
    """
    Retrieve full intervention detail for a specific student by index.
    Only available after a /interventions call has been made in this session.
    """
    if not _last_results:
        raise HTTPException(
            status_code=404,
            detail="No data in memory. Call /interventions first."
        )
    if index < 0 or index >= len(_last_results):
        raise HTTPException(
            status_code=404,
            detail=f"Student index {index} out of range (0–{len(_last_results)-1})."
        )
    return _last_results[index]
