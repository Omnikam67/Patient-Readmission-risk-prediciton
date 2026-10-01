from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# LOAD MODELS
# ============================================================

preprocessor = joblib.load(
    os.path.join(MODEL_DIR, "preprocessor.pkl")
)

feature_selector = joblib.load(
    os.path.join(MODEL_DIR, "feature_selector.pkl")
)

pca = joblib.load(
    os.path.join(MODEL_DIR, "pca.pkl")
)

xgb_model = joblib.load(
    os.path.join(MODEL_DIR, "xgb_model.pkl")
)

threshold = joblib.load(
    os.path.join(MODEL_DIR, "threshold.pkl")
)


# ============================================================
# EXACT 42 FEATURES USED DURING TRAINING
# ============================================================

FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed"
]


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        probability=None
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # GET DATA FROM FORM
        # ----------------------------------------------------

        data = {}

        for feature in FEATURES:
            data[feature] = request.form.get(feature)


        # ----------------------------------------------------
        # CONVERT NUMERICAL FEATURES
        # ----------------------------------------------------

        numerical_features = [
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses"
        ]

        for feature in numerical_features:
            data[feature] = float(data[feature])


        # ----------------------------------------------------
        # CREATE DATAFRAME
        # EXACT SAME COLUMN ORDER AS TRAINING
        # ----------------------------------------------------

        patient_df = pd.DataFrame(
            [data],
            columns=FEATURES
        )


        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        X_processed = preprocessor.transform(patient_df)


        # ----------------------------------------------------
        # FEATURE SELECTION
        # ----------------------------------------------------

        X_selected = feature_selector.transform(X_processed)


        # ----------------------------------------------------
        # PCA
        # PCA WAS TRAINED ON DENSE DATA
        # ----------------------------------------------------

        if hasattr(X_selected, "toarray"):
            X_selected = X_selected.toarray()

        X_pca = pca.transform(X_selected)


        # ----------------------------------------------------
        # XGBOOST PROBABILITY
        # ----------------------------------------------------

        probability = xgb_model.predict_proba(X_pca)[0][1]


        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        if probability >= threshold:
            prediction = 1
            result = "High Readmission Risk"
        else:
            prediction = 0
            result = "Low Readmission Risk"


        probability_percentage = round(
            probability * 100,
            2
        )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        return render_template(
            "index.html",
            prediction=prediction,
            result=result,
            probability=probability_percentage
        )


    except Exception as e:

        return render_template(
            "index.html",
            prediction=None,
            probability=None,
            error=str(e)
        )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)