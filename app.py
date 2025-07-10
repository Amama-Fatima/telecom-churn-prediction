import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                        'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaperlessBilling', 'PaymentMethod',
                        'tenure-binned', 'MonthlyCharges-binned', 'TotalCharges-binned']
        self.num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = RobustScaler()
        self.fitted = False
        
    def fit(self, X, y=None):
        try:
            X_copy = X.copy()
            if 'TotalCharges' in X_copy.columns:
                X_copy['TotalCharges'] = pd.to_numeric(X_copy['TotalCharges'], errors='coerce')
                X_copy['TotalCharges'] = X_copy['TotalCharges'].fillna(X_copy['TotalCharges'].median())
            
            existing_cat_cols = [col for col in self.cat_cols if col in X_copy.columns]
            existing_num_cols = [col for col in self.num_cols if col in X_copy.columns]
            
            if existing_cat_cols:
                self.ordinal_encoder.fit(X_copy[existing_cat_cols].fillna('Unknown'))
            if existing_num_cols:
                self.scaler.fit(X_copy[existing_num_cols])
            
            self.fitted = True
            return self
        except Exception as e:
            st.error(f"Error in fitting preprocessor: {str(e)}")
            return self
        
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        try:
            X_transformed = X.copy()
            
            if 'TotalCharges' in X_transformed.columns:
                X_transformed['TotalCharges'] = pd.to_numeric(X_transformed['TotalCharges'], errors='coerce')
                X_transformed['TotalCharges'] = X_transformed['TotalCharges'].fillna(X_transformed['TotalCharges'].median())
            
            existing_cat_cols = [col for col in self.cat_cols if col in X_transformed.columns]
            existing_num_cols = [col for col in self.num_cols if col in X_transformed.columns]
            
            if existing_cat_cols:
                X_transformed[existing_cat_cols] = self.ordinal_encoder.transform(
                    X_transformed[existing_cat_cols].fillna('Unknown')
                )
            
            if existing_num_cols:
                X_transformed[existing_num_cols] = self.scaler.transform(X_transformed[existing_num_cols])
            
            return X_transformed
        except Exception as e:
            st.error(f"Error in transforming data: {str(e)}")
            return X

def make_prediction(model, input_data):
    try:
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values
        
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(input_data)
            if predictions.shape[1] == 2:
                return predictions[:, 1] 
            else:
                return predictions.flatten()
        else:
            # For models that don't have predict_proba
            return model.predict(input_data)
    except Exception as e:
        st.error(f"Error in making predictions: {str(e)}")
        return np.zeros(len(input_data))

@st.cache_resource
def load_models():
    models = {}
    
    try:
        # CatBoost
        if os.path.exists('models/model_catboost'):
            catboost = CatBoostClassifier()
            catboost.load_model('models/model_catboost')
            models['catboost'] = catboost
        else:
            st.warning("CatBoost model not found at 'models/model_catboost'")
    except Exception as e:
        st.warning(f"Failed to load CatBoost model: {str(e)}")
    
    try:
        # XGBoost
        if os.path.exists('models/model_xgb.json'):
            xgboost = XGBClassifier()
            xgboost.load_model('models/model_xgb.json')
            models['xgboost'] = xgboost
        else:
            st.warning("XGBoost model not found at 'models/model_xgb.json'")
    except Exception as e:
        st.warning(f"Failed to load XGBoost model: {str(e)}")
    
    try:
        # LightGBM
        if os.path.exists('models/model_lgbm.pkl'):
            lgbm = joblib.load('models/model_lgbm.pkl')
            models['lgbm'] = lgbm
        else:
            st.warning("LightGBM model not found at 'models/model_lgbm.pkl'")
    except Exception as e:
        st.warning(f"Failed to load LightGBM model: {str(e)}")
    
    try:
        # Stacking model
        if os.path.exists('models/stacking_logistic_regression.pkl'):
            stacking_model = joblib.load('models/stacking_logistic_regression.pkl')
            models['stacking'] = stacking_model
        else:
            st.warning("Stacking model not found at 'models/stacking_logistic_regression.pkl'")
    except Exception as e:
        st.warning(f"Failed to load Stacking model: {str(e)}")
    
    return models

def main():
    st.title("Telecom Churn Prediction")
    st.write("Upload customer data to predict churn probability")
    
    models = load_models()
    
    if not models:
        st.error("No models could be loaded. Please check your model files.")
        return
    
    available_models = list(models.keys())
    model_name_mapping = {
        'catboost': 'CatBoost',
        'xgboost': 'XGBoost', 
        'lgbm': 'LightGBM',
        'stacking': 'Stacking Ensemble'
    }
    
    available_model_names = [model_name_mapping.get(model, model) for model in available_models]
    
    model_choice = st.selectbox(
        "Select Model",
        options=available_model_names,
        index=0
    )
    
    model_key = {v: k for k, v in model_name_mapping.items()}[model_choice]
    
    uploaded_file = st.file_uploader("Upload customer data (CSV)", type="csv")
    
    # Optional true labels uploader
    eval_mode = st.checkbox("Enable evaluation mode (provide true labels)")
    true_labels_file = None
    if eval_mode:
        true_labels_file = st.file_uploader("Upload true labels (original data with Churn column)", type="csv")

    if uploaded_file is not None:
        try:
            # Read and display input data info
            input_df = pd.read_csv(uploaded_file)
            st.subheader("Input Data Info")
            st.write(f"Shape: {input_df.shape}")
            st.write("First few rows:")
            st.dataframe(input_df.head())
            
            # Initialize and fit preprocessor
            preprocessor = DataPreprocessor()
            preprocessor.fit(input_df)
            processed_data = preprocessor.transform(input_df)
            
            if model_key == 'stacking' and 'stacking' in models:
                # For stacking, we need base model predictions
                base_predictions = []
                for base_model in ['catboost', 'xgboost', 'lgbm']:
                    if base_model in models:
                        base_pred = make_prediction(models[base_model], processed_data)
                        base_predictions.append(base_pred.reshape(-1, 1))
                
                if len(base_predictions) >= 2:  # Need at least 2 base models
                    stacked_input = np.concatenate(base_predictions, axis=1)
                    pred_probs = make_prediction(models['stacking'], stacked_input)
                else:
                    st.error("Stacking model requires at least 2 base models to be loaded.")
                    return
            else:
                # Single model prediction
                if model_key in models:
                    pred_probs = make_prediction(models[model_key], processed_data)
                else:
                    st.error(f"Selected model {model_choice} is not available.")
                    return
            
            results = input_df.copy()
            results['Churn_Probability'] = pred_probs
            results['Predicted_Churn'] = (pred_probs >= 0.5).astype(int)
            results['Predicted_Churn_Label'] = results['Predicted_Churn'].map({0: 'No', 1: 'Yes'})
            
            if eval_mode and true_labels_file is not None:
                try:
                    true_df = pd.read_csv(true_labels_file)
                    if 'Churn' in true_df.columns:
                        # Handle different formats of true labels
                        if true_df['Churn'].dtype == 'object':
                            true_labels = true_df['Churn'].map({'Yes': 1, 'No': 1, 'yes': 1, 'no': 0}).values
                        else:
                            true_labels = true_df['Churn'].values
                        
                        # Ensure same length
                        if len(true_labels) != len(results):
                            st.error(f"Mismatch in data length: Input data has {len(results)} rows, true labels have {len(true_labels)} rows")
                        else:
                            # Calculate metrics
                            accuracy = accuracy_score(true_labels, results['Predicted_Churn'])
                            conf_matrix = confusion_matrix(true_labels, results['Predicted_Churn'])
                            
                            st.subheader("Evaluation Results")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Accuracy", f"{accuracy:.2%}")
                            
                            with col2:
                                st.write("Confusion Matrix:")
                                conf_df = pd.DataFrame(conf_matrix,
                                                    index=['Actual No', 'Actual Yes'],
                                                    columns=['Predicted No', 'Predicted Yes'])
                                st.dataframe(conf_df)
                            
                            # Add classification report
                            st.subheader("Classification Report")
                            class_report = classification_report(true_labels, results['Predicted_Churn'], output_dict=True)
                            report_df = pd.DataFrame(class_report).transpose()
                            st.dataframe(report_df)
                    else:
                        st.error("True labels file must contain a 'Churn' column")
                except Exception as e:
                    st.error(f"Error processing true labels: {str(e)}")
            
            st.subheader("Predictions")
            
            churn_count = results['Predicted_Churn'].sum()
            total_count = len(results)
            st.write(f"**Summary:** {churn_count} out of {total_count} customers ({churn_count/total_count:.1%}) are predicted to churn")
            
            st.dataframe(results)
            
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please check your data format and ensure all required columns are present.")

if __name__ == "__main__":
    main()