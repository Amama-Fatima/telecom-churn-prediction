import streamlit as st
import joblib
import os
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_models():
    """Load all available models"""
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