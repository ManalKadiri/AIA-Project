import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    # Charger un échantillon de données pour les tests
    df = pd.read_csv("fraudTest.csv").sample(1000)

    # Prétraitement des données (repris de train.py)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)
    df['age'] = ((pd.Timestamp.now() - pd.to_datetime(df['dob'], errors='coerce')) / pd.Timedelta(days=365.25)).astype(int)
    
    columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
                       'first', 'last', 'street', 'city', 'state', 'zip', 'city_pop', 'job',
                       'trans_num', 'dob', 'unix_time']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    final_column_order = ["amt", "gender", "lat", "long", "age", "merch_lat", "merch_long"]
    X = df[final_column_order]
    Y = df['is_fraud']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)

    # Préprocesseur
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

    # Entraîner un modèle simple (par exemple, Logistic Regression)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight="balanced"))
    ])
    model.fit(X_train, Y_train)

    # Prédictions et évaluation
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    # Test pour vérifier la précision
    assert accuracy > 0.7, f"La précision du modèle est insuffisante : {accuracy}"
