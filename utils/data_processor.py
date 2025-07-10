import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import os

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
            # Handle TotalCharges conversion if it's string
            X_copy = X.copy()
            if 'TotalCharges' in X_copy.columns:
                X_copy['TotalCharges'] = pd.to_numeric(X_copy['TotalCharges'], errors='coerce')
                X_copy['TotalCharges'] = X_copy['TotalCharges'].fillna(X_copy['TotalCharges'].median())
            
            # Get existing columns
            existing_cat_cols = [col for col in self.cat_cols if col in X_copy.columns]
            existing_num_cols = [col for col in self.num_cols if col in X_copy.columns]
            
            # Fit encoders
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
            
            # Handle TotalCharges conversion if it's string
            if 'TotalCharges' in X_transformed.columns:
                X_transformed['TotalCharges'] = pd.to_numeric(X_transformed['TotalCharges'], errors='coerce')
                X_transformed['TotalCharges'] = X_transformed['TotalCharges'].fillna(X_transformed['TotalCharges'].median())
            
            # Get existing columns
            existing_cat_cols = [col for col in self.cat_cols if col in X_transformed.columns]
            existing_num_cols = [col for col in self.num_cols if col in X_transformed.columns]
            
            # Transform categorical columns
            if existing_cat_cols:
                X_transformed[existing_cat_cols] = self.ordinal_encoder.transform(
                    X_transformed[existing_cat_cols].fillna('Unknown')
                )
            
            # Transform numerical columns
            if existing_num_cols:
                X_transformed[existing_num_cols] = self.scaler.transform(X_transformed[existing_num_cols])
            
            return X_transformed
        except Exception as e:
            st.error(f"Error in transforming data: {str(e)}")
            return X

def make_prediction(model, input_data):
    """Make predictions using the given model"""
    try:
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values
        
        # Handle different model types
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(input_data)
            if predictions.shape[1] == 2:
                return predictions[:, 1]  # Return probability of positive class
            else:
                return predictions.flatten()
        else:
            # For models that don't have predict_proba
            return model.predict(input_data)
    except Exception as e:
        st.error(f"Error in making predictions: {str(e)}")
        return np.zeros(len(input_data))

def process_predictions(models, input_df, model_key):
    """Process predictions for given input data"""
    try:
        preprocessor = DataPreprocessor()
        preprocessor.fit(input_df)
        processed_data = preprocessor.transform(input_df)
        
        if model_key == 'stacking' and 'stacking' in models:
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
                return None
        else:
            # Single model prediction
            if model_key in models:
                pred_probs = make_prediction(models[model_key], processed_data)
            else:
                st.error(f"Selected model is not available.")
                return None
        
        results = input_df.copy()
        results['Churn_Probability'] = pred_probs
        results['Predicted_Churn'] = (pred_probs >= 0.5).astype(int)
        results['Predicted_Churn_Label'] = results['Predicted_Churn'].map({0: 'No', 1: 'Yes'})
        
        return results
        
    except Exception as e:
        st.error(f"Error processing predictions: {str(e)}")
        return None

def load_demo_data():
    """Load demo datasets if they exist"""
    demo_data = {}

    try:
        test_path = os.path.join(os.path.dirname(__file__), '../demo_data/test_data_no_churn.csv')
        if os.path.exists(test_path):
            demo_data['test'] = pd.read_csv(test_path)
    except Exception as e:
        st.warning(f"Could not load test demo data: {str(e)}")
    
    try:
        eval_path = os.path.join(os.path.dirname(__file__), '../demo_data/sample_data_with_churn.csv')
        if os.path.exists(eval_path):
            demo_data['eval'] = pd.read_csv(eval_path)
    except Exception as e:
        st.warning(f"Could not load evaluation demo data: {str(e)}")
    
    return demo_data