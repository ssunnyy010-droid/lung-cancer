import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lung Cancer AI Demo", layout="wide")

# -------------------------
# Model paths
# -------------------------
XGB_PATH = "models/lung_cancer_calibrated_pipeline.joblib"
OLD_CNN_PATH = "models/lung_histology_old_cnn.keras"
NEW_CNN_PATH = "models/lung_histology_cnn.keras"
CV_RESULTS_PATH = "lung_10fold_cv_results.csv"

# -------------------------
# Constants
# -------------------------
IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAMES = ["lung_aca", "lung_n", "lung_scc"]
DISPLAY_LABELS = {
    "lung_aca": "Adenocarcinoma",
    "lung_n": "Benign",
    "lung_scc": "Squamous Cell Carcinoma",
}

XGB_THRESHOLD = 0.50

FEATURE_COLUMNS = [
    "AGE",
    "GENDER",
    "SMOKING",
    "FINGER_DISCOLORATION",
    "MENTAL_STRESS",
    "EXPOSURE_TO_POLLUTION",
    "LONG_TERM_ILLNESS",
    "ENERGY_LEVEL",
    "IMMUNE_WEAKNESS",
    "BREATHING_ISSUE",
    "ALCOHOL_CONSUMPTION",
    "THROAT_DISCOMFORT",
    "OXYGEN_SATURATION",
    "CHEST_TIGHTNESS",
    "FAMILY_HISTORY",
    "SMOKING_FAMILY_HISTORY",
    "STRESS_IMMUNE",
]

# -------------------------
# Load models
# -------------------------
@st.cache_resource
def load_xgb():
    return joblib.load(XGB_PATH)

@st.cache_resource
def load_old_cnn():
    return tf.keras.models.load_model(OLD_CNN_PATH)

@st.cache_resource
def load_new_cnn():
    return tf.keras.models.load_model(NEW_CNN_PATH)

@st.cache_data
def load_cv_results():
    try:
        return pd.read_csv(CV_RESULTS_PATH)
    except Exception:
        return None

xgb_model = load_xgb()
old_cnn = load_old_cnn()
new_cnn = load_new_cnn()
cv_results = load_cv_results()

# -------------------------
# Session state for analysis
# -------------------------
if "last_xgb_prob" not in st.session_state:
    st.session_state.last_xgb_prob = None
if "last_xgb_flag" not in st.session_state:
    st.session_state.last_xgb_flag = None
if "last_xgb_inputs" not in st.session_state:
    st.session_state.last_xgb_inputs = None

if "last_new_probs" not in st.session_state:
    st.session_state.last_new_probs = None
if "last_new_pred_class" not in st.session_state:
    st.session_state.last_new_pred_class = None

if "last_compare_old_probs" not in st.session_state:
    st.session_state.last_compare_old_probs = None
if "last_compare_new_probs" not in st.session_state:
    st.session_state.last_compare_new_probs = None
if "last_compare_ensemble_probs" not in st.session_state:
    st.session_state.last_compare_ensemble_probs = None

# -------------------------
# Helpers
# -------------------------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(image_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return image, arr

def predict_cnn(model, arr):
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    return pred_class, probs

def probs_to_df(probs):
    return pd.DataFrame({
        "Class": [DISPLAY_LABELS[c] for c in CLASS_NAMES],
        "Probability": [float(p) for p in probs]
    })

def cancer_risk_from_probs(probs):
    aca_idx = CLASS_NAMES.index("lung_aca")
    scc_idx = CLASS_NAMES.index("lung_scc")
    return float(probs[aca_idx] + probs[scc_idx])

def format_pct(x):
    return f"{x * 100:.2f}%"

def plot_prob_bar(probs, title="Class Probabilities"):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [DISPLAY_LABELS[c] for c in CLASS_NAMES]
    ax.bar(labels, probs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    plt.xticks(rotation=10)
    return fig

def plot_prob_pie(probs, title="Prediction Distribution"):
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = [DISPLAY_LABELS[c] for c in CLASS_NAMES]
    ax.pie(probs, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    ax.axis("equal")
    return fig

def plot_fold_metric(results_df, metric_col, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(results_df["fold"].astype(str), results_df[metric_col])
    ax.set_title(title)
    ax.set_xlabel("Fold")
    ax.set_ylabel(ylabel)
    return fig

def plot_model_comparison(old_probs, new_probs, ensemble_probs):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    ax.bar(x - width, old_probs, width, label="Old CNN")
    ax.bar(x, new_probs, width, label="New CNN")
    ax.bar(x + width, ensemble_probs, width, label="Ensemble")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS[c] for c in CLASS_NAMES], rotation=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Model Comparison by Class")
    ax.legend()
    return fig

def plot_xgb_risk(prob, threshold):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.barh(["Risk"], [prob])
    ax.axvline(threshold, linestyle="--", color="red", label=f"Threshold = {threshold:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Clinical Risk Score")
    ax.legend()
    return fig

def plot_xgb_pie(prob):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [prob, 1 - prob],
        labels=["Cancer Risk", "Non-Cancer"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("XGBoost Risk Distribution")
    ax.axis("equal")
    return fig

def plot_combined_scores(clinical_score, image_score, combined_score):
    labels = []
    values = []

    if clinical_score is not None:
        labels.append("Clinical")
        values.append(clinical_score)

    if image_score is not None:
        labels.append("Image")
        values.append(image_score)

    labels.append("Combined")
    values.append(combined_score)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Risk Score")
    ax.set_title("Combined Risk Components")
    return fig

def confidence_label(prob):
    if prob >= 0.85:
        return "High"
    if prob >= 0.60:
        return "Moderate"
    return "Low"

# -------------------------
# UI
# -------------------------
st.title("Lung Cancer AI Demo")
st.caption("Research prototype only. Not for clinical diagnosis or treatment.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Clinical Risk (XGBoost)",
    "Histology Prediction (New CNN)",
    "Model Comparison",
    "Data Analysis",
    "Combined Risk (Clinical + Image)"
])

# -------------------------
# Tab 1: XGBoost
# -------------------------
with tab1:
    st.subheader("Clinical Risk Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=0, max_value=120, value=60, key="t1_age")
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="t1_gender")
        smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_smoking")
        finger_discoloration = st.selectbox("Finger Discoloration", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_finger")
        mental_stress = st.selectbox("Mental Stress", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_stress")
        exposure_to_pollution = st.selectbox("Exposure to Pollution", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_pollution")

    with c2:
        long_term_illness = st.selectbox("Long-Term Illness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_long")
        energy_level = st.number_input("Energy Level", min_value=0.0, max_value=100.0, value=50.0, key="t1_energy")
        immune_weakness = st.selectbox("Immune Weakness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_immune")
        breathing_issue = st.selectbox("Breathing Issue", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_breath")
        alcohol_consumption = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_alcohol")
        throat_discomfort = st.selectbox("Throat Discomfort", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_throat")

    with c3:
        oxygen_saturation = st.number_input("Oxygen Saturation", min_value=60.0, max_value=100.0, value=95.0, key="t1_oxygen")
        chest_tightness = st.selectbox("Chest Tightness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_chest")
        family_history = st.selectbox("Family History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_family")
        smoking_family_history = st.selectbox("Smoking Family History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_smoke_family")
        stress_immune = st.selectbox("Stress Immune", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="t1_stress_immune")

    if st.button("Predict Clinical Risk", type="primary"):
        patient_inputs = {
            "AGE": age,
            "GENDER": gender,
            "SMOKING": smoking,
            "FINGER_DISCOLORATION": finger_discoloration,
            "MENTAL_STRESS": mental_stress,
            "EXPOSURE_TO_POLLUTION": exposure_to_pollution,
            "LONG_TERM_ILLNESS": long_term_illness,
            "ENERGY_LEVEL": energy_level,
            "IMMUNE_WEAKNESS": immune_weakness,
            "BREATHING_ISSUE": breathing_issue,
            "ALCOHOL_CONSUMPTION": alcohol_consumption,
            "THROAT_DISCOMFORT": throat_discomfort,
            "OXYGEN_SATURATION": oxygen_saturation,
            "CHEST_TIGHTNESS": chest_tightness,
            "FAMILY_HISTORY": family_history,
            "SMOKING_FAMILY_HISTORY": smoking_family_history,
            "STRESS_IMMUNE": stress_immune,
        }

        patient_df = pd.DataFrame([patient_inputs], columns=FEATURE_COLUMNS)
        prob = float(xgb_model.predict_proba(patient_df)[:, 1][0])
        flag = prob >= XGB_THRESHOLD

        st.session_state.last_xgb_prob = prob
        st.session_state.last_xgb_flag = flag
        st.session_state.last_xgb_inputs = patient_df

        st.metric("Predicted Risk", format_pct(prob))
        st.write(f"Flagged above threshold ({XGB_THRESHOLD:.2f}): **{flag}**")
        st.write(f"Confidence Level: **{confidence_label(prob)}**")
        st.dataframe(patient_df, use_container_width=True)

# -------------------------
# Tab 2: New CNN
# -------------------------
with tab2:
    st.subheader("Histopathology Image Prediction")
    uploaded_file = st.file_uploader(
        "Upload a histology image",
        type=["png", "jpg", "jpeg"],
        key="newcnn"
    )

    if uploaded_file is not None:
        image, arr = preprocess_image(uploaded_file)
        pred_class, probs = predict_cnn(new_cnn, arr)
        risk = cancer_risk_from_probs(probs)

        st.session_state.last_new_probs = probs
        st.session_state.last_new_pred_class = pred_class

        left, right = st.columns([1, 1])

        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with right:
            st.metric("Predicted Class", DISPLAY_LABELS[pred_class])
            st.metric("Estimated Cancer Risk", format_pct(risk))
            st.write(f"Confidence Level: **{confidence_label(np.max(probs))}**")
            st.dataframe(probs_to_df(probs), use_container_width=True)

# -------------------------
# Tab 3: Compare models
# -------------------------
with tab3:
    st.subheader("Old CNN vs New CNN vs Ensemble")
    uploaded_file_compare = st.file_uploader(
        "Upload a histology image",
        type=["png", "jpg", "jpeg"],
        key="compare"
    )

    if uploaded_file_compare is not None:
        image, arr = preprocess_image(uploaded_file_compare)

        old_pred_class, old_probs = predict_cnn(old_cnn, arr)
        new_pred_class, new_probs = predict_cnn(new_cnn, arr)
        ensemble_probs = (old_probs + new_probs) / 2.0
        ensemble_pred_class = CLASS_NAMES[int(np.argmax(ensemble_probs))]

        st.session_state.last_compare_old_probs = old_probs
        st.session_state.last_compare_new_probs = new_probs
        st.session_state.last_compare_ensemble_probs = ensemble_probs

        st.image(image, caption="Uploaded image", use_container_width=False)

        agreement = (old_pred_class == new_pred_class == ensemble_pred_class)

        st.write(f"Model Agreement: **{'Strong' if agreement else 'Mixed'}**")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### Old CNN")
            st.write(f"Prediction: **{DISPLAY_LABELS[old_pred_class]}**")
            st.write(f"Cancer risk: **{format_pct(cancer_risk_from_probs(old_probs))}**")
            st.dataframe(probs_to_df(old_probs), use_container_width=True)

        with c2:
            st.markdown("### New CNN")
            st.write(f"Prediction: **{DISPLAY_LABELS[new_pred_class]}**")
            st.write(f"Cancer risk: **{format_pct(cancer_risk_from_probs(new_probs))}**")
            st.dataframe(probs_to_df(new_probs), use_container_width=True)

        with c3:
            st.markdown("### Ensemble Average")
            st.write(f"Prediction: **{DISPLAY_LABELS[ensemble_pred_class]}**")
            st.write(f"Cancer risk: **{format_pct(cancer_risk_from_probs(ensemble_probs))}**")
            st.dataframe(probs_to_df(ensemble_probs), use_container_width=True)

# -------------------------
# Tab 4: Data Analysis
# -------------------------
with tab4:
    st.subheader("Data Analysis")

    st.markdown("## 1. XGBoost Clinical Dataset Analysis")
    st.write("This section analyzes the structured clinical input model.")
    if st.session_state.last_xgb_prob is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_xgb_risk(st.session_state.last_xgb_prob, XGB_THRESHOLD))
        with c2:
            st.pyplot(plot_xgb_pie(st.session_state.last_xgb_prob))

        st.write(f"Flag above threshold: **{st.session_state.last_xgb_flag}**")
        st.dataframe(st.session_state.last_xgb_inputs, use_container_width=True)
    else:
        st.info("Run a prediction in the Clinical Risk tab to see XGBoost analysis.")

    st.markdown("## 2. New CNN Dataset Analysis")
    st.write("This section analyzes probability outputs from the improved CNN image model.")
    if st.session_state.last_new_probs is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_prob_bar(st.session_state.last_new_probs, title="New CNN Class Probabilities"))
        with c2:
            st.pyplot(plot_prob_pie(st.session_state.last_new_probs, title="New CNN Probability Distribution"))

        st.write(
            "Probability Distribution: The system produces predictions in the form of a "
            "probability distribution displayed in a pie chart, providing insights into "
            "the likelihood of different cancer types based on the input image."
        )
    else:
        st.info("Run a prediction in the Histology Prediction tab to see new CNN analysis.")

    st.markdown("## 3. Old vs New CNN Dataset Comparison")
    st.write("This section compares outputs from the baseline CNN, improved CNN, and ensemble.")
    if (
        st.session_state.last_compare_old_probs is not None
        and st.session_state.last_compare_new_probs is not None
        and st.session_state.last_compare_ensemble_probs is not None
    ):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(plot_prob_pie(st.session_state.last_compare_old_probs, title="Old CNN Distribution"))
        with c2:
            st.pyplot(plot_prob_pie(st.session_state.last_compare_new_probs, title="New CNN Distribution"))
        with c3:
            st.pyplot(plot_prob_pie(st.session_state.last_compare_ensemble_probs, title="Ensemble Distribution"))

        st.pyplot(
            plot_model_comparison(
                st.session_state.last_compare_old_probs,
                st.session_state.last_compare_new_probs,
                st.session_state.last_compare_ensemble_probs
            )
        )

        comparison_df = pd.DataFrame({
            "Class": [DISPLAY_LABELS[c] for c in CLASS_NAMES],
            "Old CNN": st.session_state.last_compare_old_probs,
            "New CNN": st.session_state.last_compare_new_probs,
            "Ensemble": st.session_state.last_compare_ensemble_probs
        })
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("Run a prediction in the Model Comparison tab to see old/new/ensemble analysis.")

    st.markdown("## 4. 10-Fold Cross-Validation Summary")
    st.write("This section summarizes performance on the improved validation dataset.")
    if cv_results is not None:
        st.dataframe(cv_results, use_container_width=True)

        metric_cols = ["accuracy", "macro_f1", "weighted_f1", "auc_ovr_macro"]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Mean Metrics")
            st.dataframe(
                cv_results[metric_cols].mean().to_frame(name="Mean"),
                use_container_width=True
            )

        with c2:
            st.markdown("### Standard Deviation")
            st.dataframe(
                cv_results[metric_cols].std().to_frame(name="Std"),
                use_container_width=True
            )

        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(plot_fold_metric(cv_results, "accuracy", "Fold Accuracy", "Accuracy"))
        with c4:
            st.pyplot(plot_fold_metric(cv_results, "auc_ovr_macro", "Fold AUC", "AUC"))
    else:
        st.warning("Could not load lung_10fold_cv_results.csv")

# -------------------------
# Tab 5: Combined Risk
# -------------------------
with tab5:
    st.subheader("Combined Risk (Clinical + Image)")
    st.write("This page combines patient attributes and histology image analysis into a single risk assessment. It also works if only one input type is provided.")

    st.markdown("### Patient Attributes")
    c1, c2, c3 = st.columns(3)

    with c1:
        c_age = st.number_input("Age", min_value=0, max_value=120, value=60, key="c_age")
        c_gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="c_gender")
        c_smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_smoking")
        c_finger = st.selectbox("Finger Discoloration", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_finger")
        c_stress = st.selectbox("Mental Stress", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_stress")
        c_pollution = st.selectbox("Exposure to Pollution", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_pollution")

    with c2:
        c_long = st.selectbox("Long-Term Illness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_long")
        c_energy = st.number_input("Energy Level", min_value=0.0, max_value=100.0, value=50.0, key="c_energy")
        c_immune = st.selectbox("Immune Weakness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_immune")
        c_breath = st.selectbox("Breathing Issue", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_breath")
        c_alcohol = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_alcohol")
        c_throat = st.selectbox("Throat Discomfort", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_throat")

    with c3:
        c_oxygen = st.number_input("Oxygen Saturation", min_value=60.0, max_value=100.0, value=95.0, key="c_oxygen")
        c_chest = st.selectbox("Chest Tightness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_chest")
        c_family = st.selectbox("Family History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_family")
        c_smoke_family = st.selectbox("Smoking Family History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_smoke_family")
        c_stress_immune = st.selectbox("Stress Immune", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="c_stress_immune")

    use_clinical = st.checkbox("Use patient attributes", value=True, key="use_clinical")

    st.markdown("### Histology Image")
    combined_uploaded_file = st.file_uploader(
        "Upload a histology image (optional)",
        type=["png", "jpg", "jpeg"],
        key="combined_image"
    )
    use_image = st.checkbox("Use image in combined score", value=True, key="use_image")

    if st.button("Calculate Combined Risk", type="primary"):
        clinical_score = None
        image_score = None
        combined_score = None

        # Clinical score
        if use_clinical:
            patient_inputs = {
                "AGE": c_age,
                "GENDER": c_gender,
                "SMOKING": c_smoking,
                "FINGER_DISCOLORATION": c_finger,
                "MENTAL_STRESS": c_stress,
                "EXPOSURE_TO_POLLUTION": c_pollution,
                "LONG_TERM_ILLNESS": c_long,
                "ENERGY_LEVEL": c_energy,
                "IMMUNE_WEAKNESS": c_immune,
                "BREATHING_ISSUE": c_breath,
                "ALCOHOL_CONSUMPTION": c_alcohol,
                "THROAT_DISCOMFORT": c_throat,
                "OXYGEN_SATURATION": c_oxygen,
                "CHEST_TIGHTNESS": c_chest,
                "FAMILY_HISTORY": c_family,
                "SMOKING_FAMILY_HISTORY": c_smoke_family,
                "STRESS_IMMUNE": c_stress_immune,
            }
            patient_df = pd.DataFrame([patient_inputs], columns=FEATURE_COLUMNS)
            clinical_score = float(xgb_model.predict_proba(patient_df)[:, 1][0])

        # Image score from average of old + new CNN
        if use_image and combined_uploaded_file is not None:
            image, arr = preprocess_image(combined_uploaded_file)

            old_pred_class, old_probs = predict_cnn(old_cnn, arr)
            new_pred_class, new_probs = predict_cnn(new_cnn, arr)
            ensemble_probs = (old_probs + new_probs) / 2.0
            ensemble_pred_class = CLASS_NAMES[int(np.argmax(ensemble_probs))]
            image_score = cancer_risk_from_probs(ensemble_probs)

        # Combined logic
        available_scores = [s for s in [clinical_score, image_score] if s is not None]
        if len(available_scores) > 0:
            combined_score = float(np.mean(available_scores))

        # Output
        if combined_score is None:
            st.error("No valid inputs selected. Use patient attributes, upload an image, or both.")
        else:
            c1, c2, c3 = st.columns(3)

            with c1:
                if clinical_score is not None:
                    st.metric("Clinical Score", format_pct(clinical_score))
                else:
                    st.metric("Clinical Score", "Not Used")

            with c2:
                if image_score is not None:
                    st.metric("Image Score (Avg of 2 CNNs)", format_pct(image_score))
                else:
                    st.metric("Image Score", "Not Used")

            with c3:
                st.metric("Combined Risk Score", format_pct(combined_score))

            st.write(f"Combined Confidence Level: **{confidence_label(combined_score)}**")
            st.write(f"Flagged above threshold ({XGB_THRESHOLD:.2f}): **{combined_score >= XGB_THRESHOLD}**")

            st.pyplot(plot_combined_scores(clinical_score, image_score, combined_score))

            if clinical_score is not None and image_score is not None:
                st.write("Combination mode: **Clinical + Image average**")
            elif clinical_score is not None:
                st.write("Combination mode: **Clinical only**")
            elif image_score is not None:
                st.write("Combination mode: **Image only**")

            if use_clinical and clinical_score is not None:
                with st.expander("Show patient attributes used"):
                    st.dataframe(patient_df, use_container_width=True)

            if use_image and combined_uploaded_file is not None and image_score is not None:
                st.markdown("### Image-Based Analysis")
                st.image(image, caption="Uploaded image", use_container_width=False)

                c4, c5, c6 = st.columns(3)
                with c4:
                    st.markdown("#### Old CNN")
                    st.write(f"Prediction: **{DISPLAY_LABELS[old_pred_class]}**")
                    st.dataframe(probs_to_df(old_probs), use_container_width=True)

                with c5:
                    st.markdown("#### New CNN")
                    st.write(f"Prediction: **{DISPLAY_LABELS[new_pred_class]}**")
                    st.dataframe(probs_to_df(new_probs), use_container_width=True)

                with c6:
                    st.markdown("#### Ensemble Image Output")
                    st.write(f"Prediction: **{DISPLAY_LABELS[ensemble_pred_class]}**")
                    st.dataframe(probs_to_df(ensemble_probs), use_container_width=True)

                c7, c8 = st.columns(2)
                with c7:
                    st.pyplot(plot_prob_bar(ensemble_probs, title="Ensemble Class Probabilities"))
                with c8:
                    st.pyplot(plot_prob_pie(ensemble_probs, title="Ensemble Probability Distribution"))