import os
import pandas as pd
import numpy as np
import mlflow
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Paths and environment setup
output_dir = "/tmp"
preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")

# AWS Credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'AWS_ACCESS_KEY_ID'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'AWS_SECRET_ACCESS_KEY'

# Read dataset
df = pd.read_csv('fraudTest.csv')

# Preprocessing
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)  # Convert `gender` to 1 (F) or 0 (M)
df['age'] = ((datetime.now() - pd.to_datetime(df['dob'])) / pd.Timedelta(days=365.25)).astype(int)
df = df.drop(columns=['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                      'first', 'last', 'street', 'city', 'state', 'zip', 'city_pop', 'job',
                      'trans_num', 'dob', 'unix_time'])

# Define column order for final DataFrame
final_column_order = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long"]

X = df[final_column_order]
Y = df['is_fraud']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)

# Preprocessor
numeric_features = ['amt', 'lat', 'long', 'age', 'merch_lat', 'merch_long']
categorical_features = ['gender']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', 'passthrough', categorical_features)
])

# Fit preprocessor on training data
preprocessor.fit(X_train)

# Save preprocessor
with open(preprocessor_path, "wb") as f:
    pickle.dump(preprocessor, f)

# Models
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              scale_pos_weight=(Y_train.value_counts()[0] / Y_train.value_counts()[1]))
}

# MLflow setup
mlflow.set_tracking_uri("https://appmanalkadiri-a135200c0fe7.herokuapp.com/")
mlflow.set_experiment("mlflow-AIA")

# Training
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        # Apply preprocessing and restore column order
        transformed_X_train = preprocessor.transform(X_train)
        transformed_X_train_df = pd.DataFrame(transformed_X_train, columns=final_column_order)

        transformed_X_test = preprocessor.transform(X_test)
        transformed_X_test_df = pd.DataFrame(transformed_X_test, columns=final_column_order)

        # Train the model
        clf.named_steps['classifier'].fit(transformed_X_train_df, Y_train)

        # Predictions
        Y_pred = clf.named_steps['classifier'].predict(transformed_X_test_df)
        Y_proba = clf.named_steps['classifier'].predict_proba(transformed_X_test_df)[:, 1] if hasattr(clf.named_steps['classifier'], "predict_proba") else None

        # Metrics
        accuracy = clf.named_steps['classifier'].score(transformed_X_test_df, Y_test)
        auc_roc = roc_auc_score(Y_test, Y_proba) if Y_proba is not None else None
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)

        print(f"{model_name} Results: Accuracy={accuracy}, AUC ROC={auc_roc}, Recall={recall}, F1={f1}")
        mlflow.log_metric("Accuracy", accuracy)
        if auc_roc is not None:
            mlflow.log_metric("AUC ROC", auc_roc)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)

        # Save model and artifacts
        mlflow.sklearn.log_model(clf, model_name)
        mlflow.log_artifact(preprocessor_path)

print("Training completed.")
