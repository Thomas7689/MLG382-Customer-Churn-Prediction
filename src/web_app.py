import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import joblib
import os
import plotly.figure_factory as ff
from category_encoders import TargetEncoder

# Define the model types and input columns
model_types = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network']
input_columns = ['Contract', 'tenure', 'TotalCharges', 'InternetService', 'MonthlyCharges']

# Load pre-generated model performance data
artifacts_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
performance_data = joblib.load(os.path.join(artifacts_folder, 'model_performance_data.joblib'))

# Define the mappings for categorical columns to match the ML model's training values
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

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("ML Model Predictions", className='text-center mb-4', style={'fontSize': '3em', 'color': '#007BFF'}), width=12)
    ]),
    
    dcc.Tabs([
        dcc.Tab(label='Predictions', children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Input Columns", className='font-weight-bold'),
                    dbc.Checklist(
                        id='input-columns',
                        options=[{'label': col, 'value': col} for col in input_columns],
                        value=input_columns,
                        inline=True,
                        className='mb-4'
                    ),
                    
                    html.Label("Contract", className='font-weight-bold'),
                    dcc.Dropdown(
                        id='contract',
                        options=[
                            {'label': 'Month-to-month', 'value': 'Month-to-month'},
                            {'label': 'One year', 'value': 'One year'},
                            {'label': 'Two year', 'value': 'Two year'}
                        ],
                        value='Month-to-month',
                        className='mb-4'
                    ),
                    
                    html.Label("Tenure", className='font-weight-bold'),
                    dbc.Input(id='tenure', type='number', value=1, min=0, className='mb-4'), # Ensure non-negative values
                    
                    html.Label("Total Charges", className='font-weight-bold'),
                   .Input(id='total_charges', type='number', value=0, className='mb-4'),
                    
                    html.Label("Internet Service", className='font-weight-bold'),
                    dcc.Dropdown(
                        id='internet_service',
                        options=[
                            {'label': 'DSL', 'value': 'DSL'},
                            {'label': 'Fiber optic', 'value': 'Fiber optic'},
                            {'label': 'No', 'value': 'No'}
                        ],
                        value='DSL',
                        className='mb-4'
                    ),
                    
                    html.Label("Monthly Charges", className='font-weight-bold'),
                    dbc.Input(id='monthly_charges', type='number', value=0, className='mb-4'),
                    
                    dbc.Button('Predict', id='predict-button', color='primary', className='mb-4')
                ], width=4, style={'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'borderRadius': '10px', 'padding': '20px'}),
                
                dbc.Col([
                    html.Div(id='prediction-output', className='p-4', style={'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'borderRadius': '10px'})
                ], width=8)
            ])
        ]),
        dcc.Tab(label='Model Performance', children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Input Columns", className='font-weight-bold'),
                    dbc.Checklist(
                        id='performance-input-columns',
                        options=[{'label': col, 'value': col} for col in input_columns],
                        value=input_columns,
                        inline=True,
                        className='mb-4'
                    ),
                    
                    html.Label("Select Model", className='font-weight-bold'),
                    dcc.Dropdown(
                        id='model-type',
                        options=[{'label': model, 'value': model} for model in model_types],
                        value=model_types[0],
                        className='mb-4'
                    ),
                    
                    html.Div(id='model-performance-output', className='p-4', style={'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'borderRadius': '10px'})
                ], width=12)
            ])
        ])
    ])
], fluid=True)

@app.callback(
    Output('contract', 'disabled'),
    Output('tenure', 'disabled'),
    Output('total_charges', 'disabled'),
    Output('internet_service', 'disabled'),
    Output('monthly_charges', 'disabled'),
    Input('input-columns', 'value')
)
def toggle_inputs(selected_columns):
    return (
        'Contract' not in selected_columns,
        'tenure' not in selected_columns,
        'TotalCharges' not in selected_columns,
        'InternetService' not in selected_columns,
        'MonthlyCharges' not in selected_columns
    )

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input-columns', 'value'),
    State('contract', 'value'),
    State('tenure', 'value'),
    State('total_charges', 'value'),
    State('internet_service', 'value'),
    State('monthly_charges', 'value')
)
def predict(n_clicks, selected_columns, contract, tenure, total_charges, internet_service, monthly_charges):
    if n_clicks is None:
        return ""
    input_data = pd.DataFrame({
        'Contract': [contract],
        'tenure': [tenure],
        'TotalCharges': [total_charges],
        'InternetService': [internet_service],
        'MonthlyCharges': [monthly_charges]
    })
    # Apply the updated mappings to the input data
    for col in ['Contract', 'InternetService']:
        input_data[col] = input_data[col].map(mappings)
    
    predictions = []
    for model_type in model_types:
        filename = os.path.join(artifacts_folder, f"model_{model_type}_{'_'.join(selected_columns)}.joblib")
        if os.path.exists(filename):
            model = joblib.load(filename)
            pred = model.predict(input_data[selected_columns])[0]
            churn_status = "Churn: Yes" if pred == 1 else "Churn: No"
            predictions.append(html.P(f"{model_type} Prediction: {churn_status}", className='lead'))
        else:
            predictions.append(html.P(f"{model_type} model not found for selected columns", className='lead text-danger'))
    return predictions

@app.callback(
    Output('model-performance-output', 'children'),
    Input('model-type', 'value'),
    Input('performance-input-columns', 'value')
)
def display_model_performance(model_type, selected_columns):
    for data in performance_data:
        if data['model_type'] == model_type and set(data['input_columns']) == set(selected_columns):
            accuracy = data['accuracy']
            cm = data['confusion_matrix']
            cm_fig = ff.create_annotated_heatmap(cm, x=['No', 'Yes'], y=['No', 'Yes'], colorscale='Viridis')
            cm_fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
            
            return html.Div([
                html.P(f"Accuracy: {accuracy:.2f}", className='lead'),
                dcc.Graph(figure=cm_fig)
            ])
    return html.P(f"{model_type} model not found for selected columns", className='lead text-danger')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=True)
