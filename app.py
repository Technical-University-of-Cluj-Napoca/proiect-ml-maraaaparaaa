import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Premier League ML",
    page_icon="⚽",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD EVERYTHING ONCE
# ─────────────────────────────────────────
@st.cache_resource
def load_all():
    clf_models   = joblib.load('models/clf_tuned_models.pkl')
    clf_scaler   = joblib.load('models/clf_scaler.pkl')
    clf_X_train  = joblib.load('models/X_train_clf.pkl')
    clf_X_test   = joblib.load('models/X_test_clf.pkl')
    clf_y_test   = joblib.load('models/y_test_clf.pkl')
    clf_baseline = joblib.load('models/clf_baseline_results.pkl')
    clf_tuned    = joblib.load('models/clf_tuned_results.pkl')
    clf_lc       = joblib.load('models/lc_results_clf.pkl')
    clf_shap     = joblib.load('models/clf_shap.pkl')

    reg_models   = joblib.load('models/reg_tuned_models.pkl')
    reg_scaler   = joblib.load('models/reg_scaler.pkl')
    reg_X_train  = joblib.load('models/X_train_reg.pkl')
    reg_X_test   = joblib.load('models/X_test_reg.pkl')
    reg_y_test   = joblib.load('models/y_test_reg.pkl')
    reg_baseline = joblib.load('models/reg_baseline_results.pkl')
    reg_tuned    = joblib.load('models/reg_tuned_results.pkl')
    reg_lc       = joblib.load('models/lc_results_reg.pkl')
    reg_shap     = joblib.load('models/reg_shap.pkl')

    feature_names = joblib.load('models/feature_names.pkl')
    df = pd.read_csv('datasets/premier-league-matches.csv')

    return (clf_models, clf_scaler, clf_X_train, clf_X_test,
            clf_y_test, clf_baseline, clf_tuned, clf_lc, clf_shap,
            reg_models, reg_scaler, reg_X_train, reg_X_test,
            reg_y_test, reg_baseline, reg_tuned, reg_lc, reg_shap,
            feature_names, df)

(clf_models, clf_scaler, clf_X_train, clf_X_test,
 clf_y_test, clf_baseline, clf_tuned, clf_lc, clf_shap,
 reg_models, reg_scaler, reg_X_train, reg_X_test,
 reg_y_test, reg_baseline, reg_tuned, reg_lc, reg_shap,
 feature_names, df) = load_all()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.title("⚽ Premier League ML")
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg",
    width=120
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔵 Classification", "🟠 Regression"]
)

# ─────────────────────────────────────────
# HOME
# ─────────────────────────────────────────
if page == "🏠 Home":
    st.title("⚽ Premier League Match Prediction")
    st.markdown("### Machine Learning Project")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", "4,786")
    c2.metric("Seasons", "2019–2024")
    c3.metric("Features", "84")

    st.markdown("---")
    st.markdown("""
    | | Problem | Target | Best Model |
    |--|---------|--------|------------|
    | 🔵 | **Classification** | Win / Draw / Loss | XGBoost |
    | 🟠 | **Regression** | Goals scored | XGBoost |
    """)

    st.markdown("---")
    st.subheader("📊 Dataset Overview")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(5, 3))
        df['result'].value_counts().plot(
            kind='bar', color=['green', 'red', 'gray'], ax=ax)
        ax.set_title('Match Results Distribution')
        ax.set_xticklabels(['W', 'L', 'D'], rotation=0)
        ax.set_ylabel('Count')
        st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df['gf'], bins=10, kde=True,
                     color='steelblue', ax=ax)
        ax.set_title('Goals Scored Distribution')
        ax.set_xlabel('Goals')
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("""
    ### How to use this app
    - Use the **sidebar** to navigate between pages
    - **Classification** — predict match result (W/D/L)
    - **Regression** — predict goals scored
    - Select a model, explore its metrics, then make a prediction!
    """)

# ─────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────
elif page == "🔵 Classification":
    st.title("🔵 Match Result Classification")
    st.caption("Predicting: Win (2) / Draw (1) / Loss (0)")
    st.markdown("---")

    with st.expander("📋 Problem Description", expanded=False):
        st.markdown("""
        **Goal:** Predict the result of a Premier League match —
        Win, Draw, or Loss — from the perspective of a given team.

        **Input features:** venue, xG, possession, shots,
        shots on target, goals conceded, opponent, formation...

        | Class | Meaning |
        |-------|---------|
        | W (2) | Team won the match |
        | D (1) | Match ended in a draw |
        | L (0) | Team lost the match |
        """)

    # EDA
    st.subheader("📊 Key EDA Insights")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3))
        vg = df.groupby('venue')[['gf', 'ga']].mean()
        vg.plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
        ax.set_title('Avg Goals: Home vs Away')
        ax.set_xticklabels(['Away', 'Home'], rotation=0)
        ax.legend(['Scored', 'Conceded'])
        st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3))
        vr = df.groupby(['venue', 'result']).size().unstack()
        vr.plot(kind='bar', ax=ax, color=['gray', 'red', 'green'])
        ax.set_title('Results: Home vs Away')
        ax.set_xticklabels(['Away', 'Home'], rotation=0)
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # Model selector
    st.subheader("🤖 Model Explorer")
    sel = st.selectbox("Select model:", list(clf_models.keys()))
    model = clf_models[sel]

    # Metrics
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📈 Baseline vs Tuned**")
        if sel in clf_baseline and sel in clf_tuned:
            mdf = pd.DataFrame({
                'Baseline': clf_baseline[sel],
                'Tuned': clf_tuned[sel]
            })
            st.dataframe(mdf.style.highlight_max(
                axis=1, color='lightgreen'))

    with c2:
        st.markdown("**🏆 All Models — F1 Score**")
        tdf = pd.DataFrame(clf_tuned).T.sort_values(
            'F1 Score', ascending=False)
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['gold' if m == sel else 'steelblue'
                  for m in tdf.index]
        tdf['F1 Score'].plot(kind='bar', ax=ax, color=colors)
        ax.set_xticklabels(tdf.index, rotation=45, ha='right')
        ax.set_title('F1 Score Comparison')
        st.pyplot(fig); plt.close()

    # Confusion Matrix
    st.markdown("**🔢 Confusion Matrix**")
    X_test_s = clf_scaler.transform(clf_X_test)
    y_pred   = model.predict(X_test_s)
    cm = confusion_matrix(clf_y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['L', 'D', 'W'],
                yticklabels=['L', 'D', 'W'], ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Confusion Matrix — {sel}')
    st.pyplot(fig); plt.close()

    # Hyperparameters
    st.markdown("**⚙️ Best Hyperparameters**")
    try:
        p = {k: v for k, v in model.get_params().items()
             if v is not None}
        st.json(p)
    except Exception as e:
        st.write(str(e))

    # Learning Curve
    st.markdown("**📉 Learning Curve**")
    if sel in clf_lc:
        lc = clf_lc[sel]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lc['train_sizes'],
                lc['train_scores'].mean(axis=1),
                label='Training', color='steelblue', lw=2)
        ax.plot(lc['train_sizes'],
                lc['val_scores'].mean(axis=1),
                label='Validation', color='tomato', lw=2)
        ax.fill_between(lc['train_sizes'],
            lc['train_scores'].mean(1) - lc['train_scores'].std(1),
            lc['train_scores'].mean(1) + lc['train_scores'].std(1),
            alpha=0.15, color='steelblue')
        ax.fill_between(lc['train_sizes'],
            lc['val_scores'].mean(1) - lc['val_scores'].std(1),
            lc['val_scores'].mean(1) + lc['val_scores'].std(1),
            alpha=0.15, color='tomato')
        ax.set_xlabel('Training size')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'Learning Curve — {sel}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()
    else:
        st.info("Learning curve not available for this model.")

    st.markdown("---")

    # Prediction
    st.subheader("🎯 Make a Prediction")
    st.markdown("Enter match features to predict the result:")

    c1, c2, c3 = st.columns(3)
    with c1:
        venue = st.selectbox("Venue", ["Home", "Away"])
        xg    = st.slider("Expected Goals (xG)", 0.0, 5.0, 1.5)
        xga   = st.slider("xG Against (xGA)", 0.0, 5.0, 1.5)
    with c2:
        poss  = st.slider("Possession (%)", 20, 80, 50)
        sh    = st.slider("Total Shots", 0, 35, 12)
        sot   = st.slider("Shots on Target", 0, 15, 5)
    with c3:
        ga    = st.slider("Goals Conceded", 0, 9, 1)
        dist  = st.slider("Shot Distance", 10.0, 35.0, 20.0)
        month = st.selectbox("Month", list(range(1, 13)))
        year  = st.selectbox("Season", [2019, 2020, 2021, 2022, 2023, 2024])

    if st.button("🔮 Predict Result", type="primary"):
        inp = {f: 0 for f in feature_names}
        inp.update({
            'venue': 1 if venue == "Home" else 0,
            'xg': xg, 'xga': xga, 'poss': poss,
            'sh': sh, 'sot': sot, 'ga': ga,
            'dist': dist, 'month': month, 'year': year
        })

        inp_df  = pd.DataFrame([inp])
        inp_s   = clf_scaler.transform(inp_df)
        pred    = model.predict(inp_s)[0]
        proba   = model.predict_proba(inp_s)[0]

        label = {2: "🟢 WIN", 1: "🟡 DRAW", 0: "🔴 LOSS"}
        st.markdown(f"## Result: **{label[int(pred)]}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Loss",  f"{proba[0]:.1%}")
        c2.metric("Draw",  f"{proba[1]:.1%}")
        c3.metric("Win",   f"{proba[2]:.1%}")

        # SHAP
        st.markdown("#### 🔍 Why this prediction? (SHAP)")
        try:
            if sel in ['XGBoost', 'CatBoost', 'Random Forest',
                       'EBM', 'Decision Tree']:
                explainer = shap.Explainer(model)
                sv        = explainer(inp_df)
                vals      = sv.values[0]
                base      = sv.base_values[0]
                if vals.ndim == 2:
                    vals = vals[:, 2]
                    base = base[2] if hasattr(base, '__len__') else base
            else:
                explainer = shap.LinearExplainer(
                    model, clf_scaler.transform(clf_X_train))
                sv   = explainer.shap_values(inp_s)
                base = float(explainer.expected_value)
                vals = np.array(sv).flatten()[:len(feature_names)]

            shap.waterfall_plot(
                shap.Explanation(
                    values=vals,
                    base_values=float(base),
                    data=inp_df.iloc[0],
                    feature_names=feature_names
                ), show=False, max_display=10
            )
            st.pyplot(plt.gcf()); plt.close()
        except Exception as e:
            st.warning(f"SHAP not available: {e}")

# ─────────────────────────────────────────
# REGRESSION
# ─────────────────────────────────────────
elif page == "🟠 Regression":
    st.title("🟠 Goals Scored Regression")
    st.caption("Predicting: Number of goals scored")
    st.markdown("---")

    with st.expander("📋 Problem Description", expanded=False):
        st.markdown("""
        **Goal:** Predict how many goals a team scores in a match.

        **Target variable:** `gf` (Goals For)
        **Unit:** goals | **Range:** 0–9 | **Typical:** 0–3

        **Input features:** venue, xG, possession, shots,
        shots on target, goals conceded...
        """)

    # EDA
    st.subheader("📊 Key EDA Insights")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df['gf'], bins=10, kde=True,
                     color='steelblue', ax=ax)
        ax.set_title('Goals Scored Distribution')
        ax.set_xlabel('Goals')
        st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3))
        df.groupby('venue')['gf'].mean().plot(
            kind='bar', color=['tomato', 'steelblue'], ax=ax)
        ax.set_title('Avg Goals: Home vs Away')
        ax.set_xticklabels(['Away', 'Home'], rotation=0)
        ax.set_ylabel('Average Goals')
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # Model selector
    st.subheader("🤖 Model Explorer")
    sel = st.selectbox("Select model:", list(reg_models.keys()))
    model = reg_models[sel]

    # Metrics
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📈 Baseline vs Tuned**")
        if sel in reg_baseline and sel in reg_tuned:
            mdf = pd.DataFrame({
                'Baseline': reg_baseline[sel],
                'Tuned': reg_tuned[sel]
            })
            st.dataframe(mdf.style.highlight_max(
                axis=1, color='lightgreen'))

    with c2:
        st.markdown("**🏆 All Models — R² Score**")
        tdf = pd.DataFrame(reg_tuned).T.sort_values(
            'R²', ascending=False)
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['gold' if m == sel else 'steelblue'
                  for m in tdf.index]
        tdf['R²'].plot(kind='bar', ax=ax, color=colors)
        ax.set_xticklabels(tdf.index, rotation=45, ha='right')
        ax.set_title('R² Comparison')
        st.pyplot(fig); plt.close()

    # Actual vs Predicted
    st.markdown("**🎯 Actual vs Predicted**")
    X_test_s = reg_scaler.transform(reg_X_test)
    y_pred   = model.predict(X_test_s)
    fig, ax  = plt.subplots(figsize=(6, 4))
    ax.scatter(reg_y_test, y_pred, alpha=0.4,
               color='steelblue', s=20)
    ax.plot([0, 9], [0, 9], 'r--', lw=2)
    ax.set_xlabel('Actual Goals')
    ax.set_ylabel('Predicted Goals')
    ax.set_title(f'Actual vs Predicted — {sel}')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close()

    # Hyperparameters
    st.markdown("**⚙️ Best Hyperparameters**")
    try:
        p = {k: v for k, v in model.get_params().items()
             if v is not None}
        st.json(p)
    except Exception as e:
        st.write(str(e))

    # Learning Curve
    st.markdown("**📉 Learning Curve**")
    if sel in reg_lc:
        lc = reg_lc[sel]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lc['train_sizes'],
                lc['train_scores'].mean(axis=1),
                label='Training', color='steelblue', lw=2)
        ax.plot(lc['train_sizes'],
                lc['val_scores'].mean(axis=1),
                label='Validation', color='tomato', lw=2)
        ax.fill_between(lc['train_sizes'],
            lc['train_scores'].mean(1) - lc['train_scores'].std(1),
            lc['train_scores'].mean(1) + lc['train_scores'].std(1),
            alpha=0.15, color='steelblue')
        ax.fill_between(lc['train_sizes'],
            lc['val_scores'].mean(1) - lc['val_scores'].std(1),
            lc['val_scores'].mean(1) + lc['val_scores'].std(1),
            alpha=0.15, color='tomato')
        ax.set_xlabel('Training size')
        ax.set_ylabel('R² Score')
        ax.set_title(f'Learning Curve — {sel}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()
    else:
        st.info("Learning curve not available for this model.")

    st.markdown("---")

    # Prediction
    st.subheader("🎯 Make a Prediction")
    st.markdown("Enter match features to predict goals scored:")

    c1, c2, c3 = st.columns(3)
    with c1:
        venue = st.selectbox("Venue", ["Home", "Away"])
        xg    = st.slider("Expected Goals (xG)", 0.0, 5.0, 1.5)
        xga   = st.slider("xG Against (xGA)", 0.0, 5.0, 1.5)
    with c2:
        poss  = st.slider("Possession (%)", 20, 80, 50)
        sh    = st.slider("Total Shots", 0, 35, 12)
        sot   = st.slider("Shots on Target", 0, 15, 5)
    with c3:
        ga    = st.slider("Goals Conceded", 0, 9, 1)
        dist  = st.slider("Shot Distance", 10.0, 35.0, 20.0)
        month = st.selectbox("Month", list(range(1, 13)))
        year  = st.selectbox("Season", [2019, 2020, 2021, 2022, 2023, 2024])

    if st.button("🔮 Predict Goals", type="primary"):
        inp = {f: 0 for f in feature_names}
        inp.update({
            'venue': 1 if venue == "Home" else 0,
            'xg': xg, 'xga': xga, 'poss': poss,
            'sh': sh, 'sot': sot, 'ga': ga,
            'dist': dist, 'month': month, 'year': year
        })

        inp_df = pd.DataFrame([inp])
        inp_s  = reg_scaler.transform(inp_df)
        pred   = model.predict(inp_s)[0]

        # Clamp intre 0 si 9
        pred = max(0.0, min(float(pred), 9.0))

        st.markdown(f"## ⚽ Predicted Goals: **{pred:.2f}**")
        st.markdown(f"*Rounded: **{round(pred)} goals***")

        # SHAP
        st.markdown("#### 🔍 Why this prediction? (SHAP)")
        try:
            if sel in ['XGBoost', 'CatBoost', 'Random Forest']:
                explainer = shap.TreeExplainer(model)
                sv        = explainer.shap_values(inp_df)
                base      = float(np.array(
                    explainer.expected_value).flatten()[0])
                vals      = np.array(sv).flatten()[:len(feature_names)]
            elif sel == 'Linear Regression':
                explainer = shap.LinearExplainer(
                    model, reg_scaler.transform(reg_X_train))
                sv   = explainer.shap_values(inp_s)
                base = float(explainer.expected_value)
                vals = np.array(sv).flatten()[:len(feature_names)]
            else:
                explainer = shap.Explainer(model)
                sv   = explainer(inp_df)
                base = float(sv.base_values[0])
                vals = sv.values[0]

            shap.waterfall_plot(
                shap.Explanation(
                    values=vals,
                    base_values=base,
                    data=inp_df.iloc[0],
                    feature_names=feature_names
                ), show=False, max_display=10
            )
            st.pyplot(plt.gcf()); plt.close()
        except Exception as e:
            st.warning(f"SHAP not available: {e}")