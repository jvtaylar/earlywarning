import argparse, os, joblib
import numpy as np, pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="engineering_math_early_warning_dummy.csv")
    p.add_argument("--output",default=".")
    a=p.parse_args(); os.makedirs(a.output,exist_ok=True)
    d=pd.read_csv(a.data); s=d[d.Week_Number==8].copy()
    tr=s[s.Academic_Year=="2024-2025"]; te=s[s.Academic_Year=="2025-2026"]
    cat=["Course_Name","Semester","Program"]
    num=["Quiz_Average","Assignment_Average","Prelim_Score","Current_Grade","Assessment_Completion_Rate","Login_Frequency","Total_LMS_Time_Minutes","Learning_Material_Access","Video_Engagement_Time_Minutes","Assignment_Submission_Rate","Late_Submission_Count","Activity_Completion_Rate","Discussion_Forum_Activity","Total_Classes_To_Date","Classes_Attended_To_Date","Absence_Count_To_Date","Attendance_Rate","Tardiness_Count_To_Date","Quiz_Performance_Trend","Attendance_Trend","Academic_Performance_Decline","Learning_Activity_Gap_Days"]
    xtr=tr[cat+num]; xte=te[cat+num]; ytr=tr.At_Risk.astype(int); yte=te.At_Risk.astype(int)
    nlr=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    nt=Pipeline([("impute",SimpleImputer(strategy="median"))])
    cp=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    prep_lr=ColumnTransformer([("num",nlr,num),("cat",cp,cat)])
    prep_tree=ColumnTransformer([("num",nt,num),("cat",cp,cat)])
    models={
      "Logistic Regression":Pipeline([("prep",prep_lr),("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=42))]),
      "Random Forest":Pipeline([("prep",prep_tree),("model",RandomForestClassifier(n_estimators=500,max_depth=8,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1))]),
      "XGBoost":Pipeline([("prep",prep_tree),("model",XGBClassifier(n_estimators=350,max_depth=3,learning_rate=.035,subsample=.85,colsample_bytree=.85,reg_lambda=2,min_child_weight=3,objective="binary:logistic",eval_metric="logloss",random_state=42,n_jobs=-1))])
    }
    rows=[]; predout=te[["Student_ID","Course_Name","Semester","Week_Number","At_Risk"]].copy()
    os.makedirs(os.path.join(a.output,"models"),exist_ok=True)
    for name,m in models.items():
        m.fit(xtr,ytr); pred=m.predict(xte); prob=m.predict_proba(xte)[:,1]
        tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
        rows.append({"Model":name,"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,zero_division=0),"Recall":recall_score(yte,pred,zero_division=0),"F1-score":f1_score(yte,pred,zero_division=0),"ROC-AUC":roc_auc_score(yte,prob),"PR-AUC":average_precision_score(yte,prob),"TN":tn,"FP":fp,"FN":fn,"TP":tp})
        key=name.lower().replace(" ","_"); joblib.dump(m,os.path.join(a.output,"models",key+".joblib"))
        predout[key+"_prediction"]=pred; predout[key+"_risk_probability"]=prob
    pd.DataFrame(rows).to_csv(os.path.join(a.output,"model_metrics.csv"),index=False)
    predout.to_csv(os.path.join(a.output,"test_predictions.csv"),index=False)
    # Native feature importance and global tree SHAP.
    imps=[]; shaps=[]
    for name,m in models.items():
        prep=m.named_steps["prep"]; est=m.named_steps["model"]; fn=prep.get_feature_names_out()
        val=np.abs(est.coef_[0]) if name=="Logistic Regression" else est.feature_importances_
        imps.append(pd.DataFrame({"Model":name,"Feature":fn,"Importance":val}))
        if name in ("Random Forest","XGBoost"):
            xt=prep.transform(xte)
            if hasattr(xt,"toarray"): xt=xt.toarray()
            xt=np.asarray(xt)[:min(180,len(xte))]
            sv=shap.TreeExplainer(est).shap_values(xt)
            if isinstance(sv,list): sv=sv[1]
            sv=np.asarray(sv)
            if sv.ndim==3: sv=sv[:,:,1]
            shaps.append(pd.DataFrame({"Model":name,"Feature":fn,"Mean_Absolute_SHAP":np.mean(np.abs(sv),axis=0)}))
    pd.concat(imps).to_csv(os.path.join(a.output,"feature_importance.csv"),index=False)
    pd.concat(shaps).to_csv(os.path.join(a.output,"shap_global_importance.csv"),index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
