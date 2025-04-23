import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from category_encoders.target_encoder import TargetEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import os
import itertools
import joblib

def ReadFile():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(file_path)
    df.drop(columns='customerID', inplace=True)
    return df

def TrainModel(dataFrame, TrainOnColumns, typeOfModel, testSize=0.2):

    categorical_columns = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

    # Convert categorical columns to 'category' data type
    for col in categorical_columns:
        if col in dataFrame.columns:
            dataFrame[col] = dataFrame[col].astype('category')
    dataFrame = dataFrame.fillna(0)
    targetColumn = dataFrame['Churn']
    otherColumns = dataFrame[TrainOnColumns]
    xTrain, xTest, yTrain, yTest = train_test_split(otherColumns, targetColumn, test_size=testSize, stratify=targetColumn, random_state=1)

    # Filter categorical columns based on the selected combination
    selected_categorical_columns = [col for col in categorical_columns if col in TrainOnColumns]

    match typeOfModel:
        case 'XGBoost':
            estimators = [('encoder', TargetEncoder(cols=selected_categorical_columns)),
                          ('clf', XGBClassifier(random_state=1, base_score=0.5))]
            pipe = Pipeline(steps=estimators)
            searchSpace = {
                'clf__max_depth': Integer(2, 6),
                'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
                'clf__subsample': Real(0.5, 1.0),
                'clf__colsample_bytree': Real(0.5, 1.0),
                'clf__colsample_bylevel': Real(0.5, 1.0),
                'clf__colsample_bynode': Real(0.5, 1.0),
                'clf__reg_alpha': Real(0.0, 10.0),
                'clf__reg_lambda': Real(0.0, 10.0),
                'clf__gamma': Real(0.0, 10.0)
            }
            model = BayesSearchCV(pipe, searchSpace, cv=5, n_iter=20, scoring='roc_auc_ovr', random_state=1, refit=True)
            model.fit(xTrain, yTrain)
        case 'Random Forest':
            RFModel = RandomForestClassifier(random_state=1, class_weight='balanced_subsample')
            searchSpace = {
                'max_depth': Integer(2, 6),
                'ccp_alpha': Real(0.0, 10.0),
                'n_estimators': Integer(10, 500),
                'min_samples_split': Integer(2, 10),
                'min_samples_leaf': Integer(2, 10),
            }
            model = BayesSearchCV(estimator=RFModel, search_spaces=searchSpace, n_iter=20, cv=3, random_state=1)
            model.fit(xTrain, yTrain)
        case 'Logistic Regression':
            model = LogisticRegression(max_iter=2000, class_weight='balanced')
            model.fit(xTrain, yTrain)
        case 'Neural Network':
            estimators = [('scaler', StandardScaler()), 
                          ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=2000))]
            pipe = Pipeline(steps=estimators)
            model = pipe.fit(xTrain, yTrain)
    return model

def convert_to_numerical(df):
    # Define the mappings
    mappings = {
        'Yes': 1,
        'No': 0,
        'Female': 0,
        'Male': 1,
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2,
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3,
        'DSL': 0,
        'Fiber optic': 1,
        'No internet service': 2,
        'No phone service': 2
    }

    # Apply the mappings to the relevant columns
    for column in ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']:
        if column in df.columns:
            df[column] = df[column].map(mappings).fillna(df[column])
    
    return df

def generate_and_save_models():
    df = ReadFile()
    df = convert_to_numerical(df)
    columns_to_combine = ['Contract', 'tenure', 'TotalCharges', 'InternetService', 'MonthlyCharges']
    
    models_to_train = ['XGBoost', 'Random Forest', 'Logistic Regression', 'Neural Network']
    models_to_train = ['Random Forest']
    artifacts_dir = 'artifacts'
    
    # Create the artifacts directory if it doesn't exist
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)
    
    for r in range(1, len(columns_to_combine) + 1):
        for combination in itertools.combinations(columns_to_combine, r):
            for model_type in models_to_train:
                filename = os.path.join(artifacts_dir, f"model_{model_type}_{'_'.join(combination)}.joblib")
                if not os.path.exists(filename):
                    model = TrainModel(df, list(combination), model_type)
                    joblib.dump(model, filename)
                else:
                    print(f"Model {filename} already exists. Skipping...")

if __name__ == "__main__":
    generate_and_save_models()
