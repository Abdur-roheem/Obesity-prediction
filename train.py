# =============================
# train_logreg_obesity.py
# =============================

import pandas as pd
import joblib


from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# =============================
# 1. Read data
# =============================
DATA_PATH = "/workspaces/Obesity-prediction/ObesityDataSet_raw_and_data_sinthetic.csv"
df2 = pd.read_csv(DATA_PATH)

# =============================
# 2. Manual feature encoding
# =============================

# -------- Target encoding --------
df2.loc[df2['NObeyesdad'] == 'Normal_Weight', 'NObeyesdad'] = 0
df2.loc[df2['NObeyesdad'] == 'Overweight_Level_I', 'NObeyesdad'] = 1
df2.loc[df2['NObeyesdad'] == 'Overweight_Level_II', 'NObeyesdad'] = 2
df2.loc[df2['NObeyesdad'] == 'Obesity_Type_I', 'NObeyesdad'] = 3
df2.loc[df2['NObeyesdad'] == 'Insufficient_Weight', 'NObeyesdad'] = 4
df2.loc[df2['NObeyesdad'] == 'Obesity_Type_II', 'NObeyesdad'] = 5
df2.loc[df2['NObeyesdad'] == 'Obesity_Type_III', 'NObeyesdad'] = 6

# -------- Binary categorical features --------
df2.loc[df2['Gender'] == 'Female', 'Gender'] = 0
df2.loc[df2['Gender'] == 'Male', 'Gender'] = 1

df2.loc[df2['family_history_with_overweight'] == 'no', 'family_history_with_overweight'] = 0
df2.loc[df2['family_history_with_overweight'] == 'yes', 'family_history_with_overweight'] = 1

df2.loc[df2['FAVC'] == 'no', 'FAVC'] = 0
df2.loc[df2['FAVC'] == 'yes', 'FAVC'] = 1

df2.loc[df2['SMOKE'] == 'no', 'SMOKE'] = 0
df2.loc[df2['SMOKE'] == 'yes', 'SMOKE'] = 1

df2.loc[df2['SCC'] == 'no', 'SCC'] = 0
df2.loc[df2['SCC'] == 'yes', 'SCC'] = 1

# -------- Ordinal categorical features --------
df2.loc[df2['CAEC'] == 'no', 'CAEC'] = 0
df2.loc[df2['CAEC'] == 'Sometimes', 'CAEC'] = 1
df2.loc[df2['CAEC'] == 'Frequently', 'CAEC'] = 2
df2.loc[df2['CAEC'] == 'Always', 'CAEC'] = 3

df2.loc[df2['CALC'] == 'no', 'CALC'] = 0
df2.loc[df2['CALC'] == 'Sometimes', 'CALC'] = 1
df2.loc[df2['CALC'] == 'Frequently', 'CALC'] = 2
df2.loc[df2['CALC'] == 'Always', 'CALC'] = 3

df2.loc[df2['MTRANS'] == 'Automobile', 'MTRANS'] = 0
df2.loc[df2['MTRANS'] == 'Motorbike', 'MTRANS'] = 1
df2.loc[df2['MTRANS'] == 'Bike', 'MTRANS'] = 2
df2.loc[df2['MTRANS'] == 'Public_Transportation', 'MTRANS'] = 3
df2.loc[df2['MTRANS'] == 'Walking', 'MTRANS'] = 4

# Convert everything to numeric
df2 = df2.astype('float64')

# =============================
# 3. Split features and target
# =============================
X = df2.drop(columns=["NObeyesdad"])
y = df2["NObeyesdad"].values  # 1D array required by sklearn

# =============================
# 4. Train–test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=4
)

# =============================
# 5. Pipeline definition
# =============================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', OneVsRestClassifier(
        LogisticRegression(
            solver='liblinear',
            max_iter=1000,
            random_state=4
        )
    ))
])
# =============================
# 6. Hyperparameter grid
# =============================
param_grid = {
    'logreg__estimator__C': [0.01, 0.1, 1, 10],
    'logreg__estimator__penalty': ['l1', 'l2']
}

# =============================
# 7. Grid search (ROC-AUC)
# =============================
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc_ovr',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

# =============================
# 8. Evaluation
# =============================
best_model = grid.best_estimator_
y_proba = best_model.predict_proba(X_test)

test_roc_auc = roc_auc_score(
    y_test,
    y_proba,
    multi_class='ovr'
)

print("Best parameters:", grid.best_params_)
print("CV ROC-AUC:", grid.best_score_)
print("Test ROC-AUC:", test_roc_auc)

# =============================
# 9. Save trained model
# =============================
file_name = "logreg_obesity_multiclass.joblib"
joblib.dump(best_model, file_name)
print(f"Model saved successfully: {file_name}")
