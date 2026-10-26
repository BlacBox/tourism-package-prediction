"""
Visit with Us - Wellness Tourism Package Purchase Predictor
-----------------------------------------------------------
Streamlit app that loads the model committed to tourism_project/deployment/
by the GitHub Actions pipeline, collects the customer inputs into a
dataframe and predicts whether the customer will buy the package.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_tourism_model_v1.joblib"
METRICS_PATH = APP_DIR / "model_metrics.json"

# Same feature order as the training data produced by prep.py
FEATURE_ORDER = [
    "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation", "Gender",
    "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

st.set_page_config(page_title="Wellness Package Predictor", layout="centered")


@st.cache_resource
def load_model():
    """Load the trained pipeline once and reuse it across user sessions."""
    return joblib.load(MODEL_PATH)


def load_metrics():
    """Read the evaluation summary written by train.py (if available)."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


if not MODEL_PATH.exists():
    st.error("Model file not found. Run the GitHub Actions pipeline to train and commit the model.")
    st.stop()

model = load_model()
metrics = load_metrics()

st.title("Wellness Tourism Package - Purchase Prediction")
st.write(
    "Predict whether a customer is likely to purchase the new **Wellness Tourism Package** "
    "so the sales team can prioritise the right customers before contacting them."
)

# Sidebar: information about the deployed model
with st.sidebar:
    st.header("Deployed model")
    if metrics:
        test = metrics["test_metrics"]
        st.write(f"**Algorithm:** {metrics['model']}")
        st.write(f"**Trained:** {metrics['trained_at_utc']}")
        st.metric("Test F1-score", f"{test['f1']:.3f}")
        st.metric("Test recall (buyers found)", f"{test['recall']:.1%}")
        st.metric("Test precision", f"{test['precision']:.1%}")
    st.caption("The model is retrained and redeployed automatically by the GitHub Actions pipeline.")

single_tab, batch_tab = st.tabs(["Single customer", "Batch scoring (CSV)"])

with single_tab:
    with st.form("customer_form"):
        st.subheader("Customer details")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
            designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            monthly_income = st.number_input("Monthly income", min_value=1000, max_value=200000, value=22000, step=500)
            city_tier = st.selectbox("City tier", [1, 2, 3])
        with col2:
            passport = st.radio("Holds a valid passport?", ["No", "Yes"], horizontal=True)
            own_car = st.radio("Owns a car?", ["No", "Yes"], horizontal=True)
            number_of_trips = st.number_input("Average trips per year", min_value=1, max_value=30, value=3)
            persons_visiting = st.number_input("Persons visiting", min_value=1, max_value=10, value=3)
            children_visiting = st.number_input("Children (under 5) visiting", min_value=0, max_value=10, value=1)
            property_star = st.selectbox("Preferred hotel rating (stars)", [3, 4, 5])

        st.subheader("Sales interaction")
        col3, col4 = st.columns(2)
        with col3:
            type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
            product_pitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
            duration_of_pitch = st.number_input("Duration of pitch", min_value=1, max_value=200, value=15)
        with col4:
            number_of_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=3)
            pitch_satisfaction = st.slider("Pitch satisfaction score", min_value=1, max_value=5, value=3)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        # Save the inputs into a single-row dataframe with the training column names
        input_df = pd.DataFrame([{
            "Age": age,
            "TypeofContact": type_of_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": persons_visiting,
            "NumberOfFollowups": number_of_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_of_trips,
            "Passport": 1 if passport == "Yes" else 0,
            "PitchSatisfactionScore": pitch_satisfaction,
            "OwnCar": 1 if own_car == "Yes" else 0,
            "NumberOfChildrenVisiting": children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }])[FEATURE_ORDER]

        st.write("**Customer input**")
        st.dataframe(input_df, hide_index=True)

        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(model.predict(input_df)[0])
        if prediction == 1:
            st.success(f"Likely to BUY the Wellness package (purchase probability: {probability:.1%}). "
                       "Prioritise this customer in the campaign.")
        else:
            st.warning(f"Unlikely to buy the Wellness package (purchase probability: {probability:.1%}). "
                       "Keep this customer in low-cost channels.")

with batch_tab:
    st.write("Upload a CSV file with the same columns as the training data to score a customer list.")
    uploaded = st.file_uploader("Customer list (CSV)", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_ORDER if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            # Same cleaning rule as prep.py for the known gender typo
            batch_df["Gender"] = batch_df["Gender"].replace({"Fe Male": "Female"})
            scored = batch_df.copy()
            scored["PurchaseProbability"] = model.predict_proba(batch_df[FEATURE_ORDER])[:, 1].round(4)
            scored["PredictedProdTaken"] = model.predict(batch_df[FEATURE_ORDER])
            scored = scored.sort_values("PurchaseProbability", ascending=False)
            st.write(f"Scored **{len(scored):,}** customers - "
                     f"**{int(scored['PredictedProdTaken'].sum()):,}** predicted buyers.")
            st.dataframe(scored, hide_index=True)
            st.download_button("Download predictions", scored.to_csv(index=False),
                               file_name="wellness_package_predictions.csv", mime="text/csv")
