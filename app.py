import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="Titanic Survival App", layout="wide")

st.title("🚢 Titanic Survival Prediction – Interactive App")
st.write(
    "Upload your Titanic **train.csv**, train a model, and try predictions on new passengers."
)

# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------
st.sidebar.header("1. Data")

uploaded = st.sidebar.file_uploader(
    "Upload Titanic train.csv (must have Survived column)",
    type=["csv"],
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists("train.csv"):
    df = pd.read_csv("train.csv")
    st.sidebar.info("Using local train.csv found in this folder.")
else:
    st.warning("Upload a Titanic train.csv file to continue.")
    st.stop()

st.subheader("Raw data preview")
st.dataframe(df.head())

if "Survived" not in df.columns:
    st.error("The file must contain a 'Survived' column.")
    st.stop()

# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING (SIMILAR TO YOUR NOTEBOOK)
# ---------------------------------------------------------------------
st.sidebar.header("2. Preprocessing / Model")

RANDOM_STATE = 42

def extract_title(name):
    m = re.search(r",\s*([^\.]+)\.", str(name))
    return m.group(1).strip() if m else "Unknown"

def feature_engineering(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # Standardize column names (like in your notebook)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Title
    df["Title"] = df["Name"].apply(extract_title)
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Noble", "Countess": "Noble", "Sir": "Noble", "Don": "Noble",
        "Dona": "Noble", "Jonkheer": "Noble",
        "Capt": "Officer", "Col": "Officer", "Major": "Officer",
        "Dr": "Officer", "Rev": "Officer"
    }
    df["Title"] = df["Title"].replace(title_map)

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Ticket prefix
    def ticket_prefix(t):
        t = str(t)
        t = re.sub(r"[0-9./]", "", t).strip().upper()
        return t if t else "NONE"
    df["TicketPrefix"] = df["Ticket"].apply(ticket_prefix)

    # Cabin deck
    def cabin_deck(c):
        if pd.isna(c):
            return "U"
        return str(c)[0]
    df["CabinDeck"] = df["Cabin"].apply(cabin_deck)

    # Embarked: fill missing with mode
    if df["Embarked"].isna().any():
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Age: fill with median per Title x Pclass (as in your P1)
    age_group_median = df.groupby(["Title", "Pclass"])["Age"].median()
    def impute_age(row):
        if pd.isna(row["Age"]):
            return age_group_median.loc[(row["Title"], row["Pclass"])]
        return row["Age"]
    df["Age"] = df.apply(impute_age, axis=1)

    # Fare: fill with median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Select columns for modeling
    target = df["Survived"].astype(int)
    # You can tweak this list to match your notebook exactly
    features = df[[
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "Embarked", "Title", "FamilySize", "IsAlone",
        "TicketPrefix", "CabinDeck"
    ]]

    # One-hot encode categoricals
    cat_cols = ["Pclass", "Sex", "Embarked", "Title", "TicketPrefix", "CabinDeck"]
    features = pd.get_dummies(features, columns=cat_cols, drop_first=False)

    return features, target

X, y = feature_engineering(df)

st.write("After feature engineering, shape:", X.shape)

# ---------------------------------------------------------------------
# 3. TRAIN / VALIDATION SPLIT + MODEL TRAINING
# ---------------------------------------------------------------------
test_size = st.sidebar.slider("Validation size", 0.1, 0.4, 0.2, 0.05)
model_choice = st.sidebar.selectbox(
    "Model",
    ["Logistic Regression", "Decision Tree", "Random Forest"],
)

st.sidebar.write("---")
train_button = st.sidebar.button("🚀 Train / Retrain model")

if "trained" not in st.session_state:
    st.session_state.trained = False

if train_button or not st.session_state.trained:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    else:
        model = RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = dict(
        Accuracy=accuracy_score(y_val, y_pred),
        Precision=precision_score(y_val, y_pred),
        Recall=recall_score(y_val, y_pred),
        ROC_AUC=roc_auc_score(y_val, y_proba),
    )

    st.session_state.model = model
    st.session_state.X_val = X_val
    st.session_state.y_val = y_val
    st.session_state.y_proba = y_proba
    st.session_state.metrics = metrics
    st.session_state.feature_names = list(X.columns)
    st.session_state.trained = True
    st.session_state.model_choice = model_choice

# ---------------------------------------------------------------------
# 4. SHOW METRICS + ROC
# ---------------------------------------------------------------------
st.markdown("## Model performance")

if not st.session_state.trained:
    st.info("Train the model using the sidebar.")
else:
    m = st.session_state.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['Accuracy']:.3f}")
    c2.metric("Precision", f"{m['Precision']:.3f}")
    c3.metric("Recall", f"{m['Recall']:.3f}")
    c4.metric("ROC AUC", f"{m['ROC_AUC']:.3f}")

    # ROC plot
    fpr, tpr, _ = roc_curve(st.session_state.y_val, st.session_state.y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

# ---------------------------------------------------------------------
# 5. SINGLE PASSENGER PREDICTION
# ---------------------------------------------------------------------
st.markdown("## Predict survival for a new passenger")

if not st.session_state.trained:
    st.info("Train a model first.")
else:
    model = st.session_state.model
    feature_names = st.session_state.feature_names

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
            sex = st.selectbox("Sex", ["male", "female"], index=0)
            age = st.number_input("Age", 0.0, 100.0, 30.0)
        with col2:
            sibsp = st.number_input("SibSp", 0, 10, 0)
            parch = st.number_input("Parch", 0, 10, 0)
            fare = st.number_input("Fare", 0.0, 600.0, 32.2)
        with col3:
            embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)
            ticket_prefix = st.text_input("Ticket prefix", value="A/5")
            cabin = st.text_input("Cabin", value="")

        submitted = st.form_submit_button("Predict")

    if submitted:
        family_size = sibsp + parch + 1
        is_alone = int(family_size == 1)

        # Build one-row dataframe similar to original df
        row = {
            "Pclass": [pclass],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked": [embarked],
            "Title": ["Mr"],  # simple default; you can expose this in the form
            "FamilySize": [family_size],
            "IsAlone": [is_alone],
            "TicketPrefix": [ticket_prefix if ticket_prefix.strip() else "NONE"],
            "CabinDeck": [cabin[0] if cabin else "U"],
        }
        df_new = pd.DataFrame(row)

        # Reuse same FE pipeline
        X_new, _ = feature_engineering(
            pd.concat(
                [df.assign(Survived=df["Survived"]),  # original
                 df_new.assign(Survived=0)],         # dummy Survived
                ignore_index=True
            )
        )

        # Last row is the new passenger
        X_new_row = X_new.tail(1)
        # Align columns with training
        X_new_row = X_new_row.reindex(columns=feature_names, fill_value=0)

        proba = model.predict_proba(X_new_row)[0, 1]
        pred = int(proba >= 0.5)

        st.write("---")
        st.subheader("Prediction result")
        c1, c2 = st.columns(2)
        c1.metric("Survival probability", f"{proba:.3f}")
        c2.metric("Predicted class", "Survived" if pred == 1 else "Did not survive")
        st.caption("Threshold = 0.5 (you can change this logic in the code).")
