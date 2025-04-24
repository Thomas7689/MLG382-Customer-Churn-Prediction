import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools

# Define the model types and input columns
model_types = ['XGBoost', 'Random Forest', 'Logistic Regression', 'Neural Network']
input_columns = ['Contract', 'tenure', 'TotalCharges', 'InternetService', 'MonthlyCharges']

def ReadFile():
    df = pd.read_csv('data/test.csv')
    return df

def convert_to_numerical(df):
    # Define the mappings
    mappings = {
        'Yes': 1,
        'No': 0,
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2,
        'DSL': 0,
        'Fiber optic': 1,
        'No internet service': 2,
        'No phone service': 2
    }

    # Apply the mappings to the relevant columns
    for column in ['InternetService', 'Contract', 'Churn']:
        if column in df.columns:
            df[column] = df[column].map(mappings).fillna(df[column])
    df = df.fillna(0)
    return df


def generate_model_performance():
    df = convert_to_numerical(ReadFile())
    y = df['Churn']
    
    performance_data = []
    artifacts_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
    
    for r in range(1, len(input_columns) + 1):
        for combination in itertools.combinations(input_columns, r):
            x = df[list(combination)]
            for model_type in model_types:
                filename = os.path.join(artifacts_folder, f"model_{model_type}_{'_'.join(combination)}.joblib")
                if os.path.exists(filename):
                    try:
                        model = joblib.load(filename)
                        y_pred = model.predict(x)
                        accuracy = accuracy_score(y, y_pred)
                        cm = confusion_matrix(y, y_pred)
                        performance_data.append({
                            'model_type': model_type,
                            'input_columns': combination,
                            'accuracy': accuracy,
                            'confusion_matrix': cm
                        })
                    except KeyError as e:
                        print(f"KeyError: {e} in file {filename}")
    
    # Save the performance data to a file
    performance_data_file = os.path.join(artifacts_folder, 'model_performance_data.joblib')
    joblib.dump(performance_data, performance_data_file)


generate_model_performance()