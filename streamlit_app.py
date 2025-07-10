import streamlit as st
from utils.model_loader import load_models
from utils.data_processor import DataPreprocessor
from tabs.upload_tab import upload_tab
from tabs.manual_input_tab import manual_input_tab
from tabs.demo_data_tab import demo_data_tab

def main():
    st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")
    st.title("🔮 Telecom Churn Prediction")
    st.write("Predict customer churn using machine learning models")
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("No models could be loaded. Please check your model files.")
        return
    
    # Show available models
    available_models = list(models.keys())
    model_name_mapping = {
        'catboost': 'CatBoost',
        'xgboost': 'XGBoost', 
        'lgbm': 'LightGBM',
        'stacking': 'Stacking Ensemble'
    }
    
    available_model_names = [model_name_mapping.get(model, model) for model in available_models]
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select Model",
        options=available_model_names,
        index=0
    )
    
    # Map back to model key
    model_key = {v: k for k, v in model_name_mapping.items()}[model_choice]
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload Data", "🎯 Manual Input", "📊 Demo Data"])
    
    with tab1:
        upload_tab(models, model_key)
    
    with tab2:
        manual_input_tab(models, model_key)
    
    with tab3:
        demo_data_tab(models, model_key)

if __name__ == "__main__":
    main()