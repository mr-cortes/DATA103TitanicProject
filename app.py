import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Titanic ML Dashboard", layout="wide")
st.title("🚢 Titanic Survival Prediction Dashboard")
st.caption("Interactive Machine Learning Dashboard — Predict, Explain, and Audit Fairness")

# ------------------------------------------------------------
# LOAD MODELS AND DATA
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    logit = joblib.load("logistic_model.pkl")
    rf = joblib.load("rf_model.pkl")
    return logit, rf

@st.cache_data
def load_data():
    df = pd.read_csv("titanic_cleaned.csv")
    return df

clf_logit, clf_rf = load_models()
df = load_data()

# ------------------------------------------------------------
# SIDEBAR INPUTS
# ------------------------------------------------------------
st.sidebar.header("🧍 Passenger Features")

sex = st.sidebar.selectbox("Sex", ["male", "female"])
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
age = st.sidebar.slider("Age", 1, 80, 29)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 5, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 5, 0)
fare = st.sidebar.number_input("Fare Paid", 0.0, 512.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
fare_per_person = fare / family_size if family_size > 0 else fare

# ------------------------------------------------------------
# CLASS DESCRIPTIONS
# ------------------------------------------------------------
st.sidebar.header("Passenger Classes")
class_descriptions = {
    1: "**First Class:** Catered to the wealthy with luxurious suites and amenities.",
    2: "**Second Class:** For middle-class professionals and tourists.",
    3: "**Third Class:** For immigrants and working-class people.",
}
st.sidebar.markdown(class_descriptions[pclass])

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

# Derive engineered features EXACTLY like in your notebook

# 1. Title
def infer_title(sex, age):
    if sex == "female":
        return "Miss" if age < 25 else "Mrs"
    else:
        return "Mr"

title = infer_title(sex, age)

# 2. CabinDeck (no cabin input given → set to "Unknown")
cabin_deck = "Unknown"

# 3. CabinKnown (1 = has cabin; 0 = missing)
cabin_known = 0

# 4. TicketPrefix (no ticket input → use "NONE")
ticket_prefix = "NONE"

# 5. AgeBin (same bin edges used in notebook)
def get_age_bin(age):
    bins = [0, 12, 18, 25, 35, 45, 60, 120]
    labels = ["0-12","13-18","19-25","26-35","36-45","46-60","60+"]
    for i in range(len(bins)-1):
        if bins[i] <= age <= bins[i+1]:
            return labels[i]
    return "26-35"

age_bin = get_age_bin(age)

def get_feature_names_from_pipeline(pipeline):
    pre = pipeline.named_steps["pre"]

    # categorical columns passed to OHE
    cat_cols = pre.transformers_[0][2]
    ohe = pre.named_transformers_["cat"]

    if hasattr(ohe, "get_feature_names_out"):
        cat_features = list(ohe.get_feature_names_out(cat_cols))
    else:
        cat_features = list(ohe.get_feature_names(cat_cols))

    # numeric passthrough columns
    num_features = list(pre.transformers_[1][2])

    return cat_features + num_features

# ------------------------------------------------------------
# FINAL INPUT ROW (all required model features)
# ------------------------------------------------------------

input_dict = {
    "Sex": [sex],
    "Pclass": [pclass],
    "Embarked": [embarked],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "FamilySize": [family_size],
    "IsAlone": [is_alone],
    "FarePerPerson": [fare_per_person],
    
    # engineered features below:
    "Title": [title],
    "CabinDeck": [cabin_deck],
    "CabinKnown": [cabin_known],
    "TicketPrefix": [ticket_prefix],
    "AgeBin": [age_bin],
}

input_df = pd.DataFrame(input_dict)

# ------------------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------------------
model_choice = st.sidebar.radio("Select Model", ["Logistic Regression", "Random Forest"])
if model_choice == "Logistic Regression":
    model = clf_logit
else:
    model = clf_rf

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------
pred_proba = model.predict_proba(input_df)[0, 1]
pred_label = int(pred_proba >= 0.5)

st.subheader("🎯 Prediction")
if pred_label:
    st.success("**Prediction:** Survived 🚢")
    st.balloons()
else:
    st.error("**Prediction:** Did Not Survive 💔")
    with st.spinner("Sinking..."):
        time.sleep(1)

st.metric("Survival Probability", f"{pred_proba:.2%}", delta=None)

# ------------------------------------------------------------
# PASSENGER PROFILE
# ------------------------------------------------------------
st.subheader("👤 Passenger Profile")
profile_df = pd.DataFrame([
    {"Feature": "Sex", "Value": sex},
    {"Feature": "Class", "Value": pclass},
    {"Feature": "Age", "Value": age},
    {"Feature": "Family Size", "Value": family_size},
    {"Feature": "Fare", "Value": f"${fare:.2f}"},
    {"Feature": "Embarked", "Value": embarked},
])
st.table(profile_df)

# ------------------------------------------------------------
# SURVIVAL SCENARIOS
# ------------------------------------------------------------
st.subheader("🎭 Survival Scenarios")
scenarios = {
    "Wealthy Woman (1st Class)": {"Sex": "female", "Pclass": 1, "Age": 35, "FamilySize": 2},
    "Poor Man (3rd Class)": {"Sex": "male", "Pclass": 3, "Age": 25, "FamilySize": 1},
    "Child (2nd Class)": {"Sex": "male", "Pclass": 2, "Age": 5, "FamilySize": 3},
}

scenario_data = []
for name, features in scenarios.items():
    scenario_input = input_df.copy()
    for key, value in features.items():
        scenario_input[key] = value
    
    scenario_proba = model.predict_proba(scenario_input)[0, 1]
    scenario_data.append({"Scenario": name, "Survival Probability": scenario_proba})

scenario_df = pd.DataFrame(scenario_data)
st.bar_chart(scenario_df.set_index("Scenario"))

# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------
st.subheader("📈 Model Performance")
y_pred = model.predict(df.drop("Survived", axis=1))
y_true = df["Survived"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, model.predict_proba(df.drop("Survived", axis=1))[:, 1])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Precision", f"{precision:.2%}")
col3.metric("Recall", f"{recall:.2%}")
col4.metric("ROC-AUC", f"{roc_auc:.3f}")

# ------------------------------------------------------------
# FAIRNESS AUDIT
# ------------------------------------------------------------
st.subheader("⚖️ Fairness & Bias Audit")

# Compute fairness metrics on validation-like sample
try:
    preds = model.predict(df.drop("Survived", axis=1))
    y_true = df["Survived"]

    mf = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=preds,
        sensitive_features=df["Sex"]
    )

    fairness_df = mf.by_group.reset_index()
    fairness_df.columns = ["Sex", "Selection Rate"]

    st.write("Selection Rate (Survival Prediction) by Sex:")
    st.bar_chart(fairness_df.set_index("Sex"))

    dp_diff = demographic_parity_difference(y_true, preds, sensitive_features=df["Sex"])
    st.write(f"**Demographic Parity Difference:** {dp_diff:.3f}")
    st.info("A lower Demographic Parity Difference is better. A value of 0 indicates perfect fairness.")

except Exception as e:
    st.warning(f"Fairness audit unavailable: {e}")

# ------------------------------------------------------------
# GLOBAL EDA INSIGHTS
# ------------------------------------------------------------
st.subheader("📊 Exploratory Data Insights")

tab1, tab2, tab3 = st.tabs(["Survival by Category", "Distributions", "Correlation Heatmap"])

with tab1:
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    sns.barplot(data=df, x="Sex", y="Survived", ax=ax[0])
    sns.barplot(data=df, x="Pclass", y="Survived", ax=ax[1])
    sns.barplot(data=df, x="Embarked", y="Survived", ax=ax[2])
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(df["Age"], kde=True, ax=ax[0])
    sns.histplot(df["Fare"], kde=True, ax=ax[1])
    ax[0].set_title("Age Distribution")
    ax[1].set_title("Fare Distribution")
    st.pyplot(fig)

with tab3:
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(fig)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption("Built by Group 5")
