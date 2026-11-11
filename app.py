import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Engineering Mathematics Early Warning System",
                   page_icon="📊", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
def load_csv(name):
    return pd.read_csv(os.path.join(BASE, name))

st.title("Engineering Mathematics – Student Early Warning System")
st.caption("Research prototype using synthetic demonstration data. Not for real student decisions.")
metrics = load_csv("model_metrics.csv")
risk = load_csv("instructor_risk_register.csv")
importance = load_csv("feature_importance.csv")
shap_global = load_csv("shap_global_importance.csv")

with st.sidebar:
    st.header("Dashboard filters")
    courses = ["All courses"] + sorted(risk["Course_Name"].dropna().unique().tolist())
    course = st.selectbox("Course", courses)
    terms = ["All terms"] + sorted(risk["Semester"].dropna().unique().tolist())
    term = st.selectbox("Semester", terms)
    risk_filter = st.multiselect("Risk category", ["High", "Moderate", "Low"],
                                 default=["High", "Moderate", "Low"])
    model_name = st.selectbox("Model", metrics["Model"].tolist())

view = risk.copy()
if course != "All courses":
    view = view[view.Course_Name == course]
if term != "All terms":
    view = view[view.Semester == term]
view = view[view.Risk_Level.isin(risk_filter)]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Students / course records", f"{len(view):,}")
c2.metric("High risk", int((view.Risk_Level=="High").sum()))
c3.metric("Moderate risk", int((view.Risk_Level=="Moderate").sum()))
c4.metric("Low risk", int((view.Risk_Level=="Low").sum()))

st.subheader("Risk distribution")
dist = view["Risk_Level"].value_counts().reindex(["High","Moderate","Low"], fill_value=0).rename_axis("Risk level").reset_index(name="Students")
fig = px.bar(dist, x="Risk level", y="Students", color="Risk level",
             category_orders={"Risk level":["High","Moderate","Low"]})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Model performance")
st.dataframe(metrics.style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1-score","ROC-AUC","PR-AUC"]}),
             use_container_width=True, hide_index=True)
st.caption("The selected model is used for feature-importance views; the supplied risk register is scored by the best ROC-AUC model from the experiment.")

st.subheader("Students requiring attention")
display_cols = ["Student_ID","Course_Name","Semester","Week_Number","Predicted_Risk_Probability",
                "Risk_Level","Quiz_Average","Assignment_Average","Attendance_Rate",
                "Activity_Completion_Rate","Recommended_Action"]
st.dataframe(view[display_cols].sort_values("Predicted_Risk_Probability",ascending=False)
             .style.format({"Predicted_Risk_Probability":"{:.1%}"}), use_container_width=True, hide_index=True)

st.subheader("Individual student record")
if len(view):
    ids = view["Student_ID"] + " | " + view["Course_Name"] + " | " + view["Semester"]
    selected = st.selectbox("Select a record", ids.tolist())
    row = view.iloc[ids.tolist().index(selected)]
    a,b,c = st.columns(3)
    a.metric("Predicted risk", f"{row.Predicted_Risk_Probability:.1%}")
    b.metric("Risk level", row.Risk_Level)
    c.metric("Week", int(row.Week_Number))
    st.info(row.Recommended_Action)
    st.write({
        "Quiz average": row.Quiz_Average,
        "Assignment average": row.Assignment_Average,
        "Attendance rate": row.Attendance_Rate,
        "LMS logins": row.Login_Frequency,
        "Activity completion": row.Activity_Completion_Rate
    })

st.subheader("Predictor importance")
tab1,tab2 = st.tabs(["Model feature importance","SHAP global importance"])
with tab1:
    imp = importance[importance.Model == model_name].nlargest(15,"Importance").sort_values("Importance")
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                 title=f"Top predictors – {model_name}")
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    shap_model = st.selectbox("SHAP model", sorted(shap_global.Model.unique()))
    si = shap_global[shap_global.Model == shap_model].nlargest(15,"Mean_Absolute_SHAP").sort_values("Mean_Absolute_SHAP")
    fig = px.bar(si, x="Mean_Absolute_SHAP", y="Feature", orientation="h",
                 title=f"Mean absolute SHAP values – {shap_model}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("SHAP values show each feature's contribution to model output. The global chart ranks average absolute contribution; it does not imply causation.")

#st.warning("Research-use limitation: the underlying records and labels are synthetic. Risk bands are provisional. Validate calibration, thresholds, fairness, and intervention utility with authorized real institutional data before deployment.")
