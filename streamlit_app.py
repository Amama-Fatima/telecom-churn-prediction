import streamlit as st
from utils.model_loader import load_models
from utils.data_processor import DataPreprocessor
from tabs.upload_tab import upload_tab
from tabs.manual_input_tab import manual_input_tab
from tabs.demo_data_tab import demo_data_tab

def add_custom_css():
    """Add custom CSS to make sidebar smaller and style the app"""
    st.markdown("""
    <style>
    /* Make sidebar smaller */
    .css-1d391kg {
        width: 200px !important;
    }
    
    .css-1lcbmhc {
        width: 200px !important;
    }
    
    /* Alternative sidebar selectors for different Streamlit versions */
    section[data-testid="stSidebar"] {
        width: 200px !important;
        min-width: 200px !important;
    }
    
    /* Header styling */
    .header-container {
        background: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        border: 2px solid #34495e;
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(44, 62, 80, 0.9);
        backdrop-filter: blur(10px);
        color: white;
        text-align: center;
        z-index: 999;
        border-top: 3px solid rgba(52, 73, 94, 0.8);
    }
    
    .footer a {
        color: #f0f0f0;
        text-decoration: none;
        margin: 0 15px;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: #ffffff;
        text-decoration: underline;
    }
    
    /* Add bottom padding to main content to prevent footer overlap */
    .main .block-container {
        padding-bottom: 80px;
    }
    
    /* Sidebar header styling */
    .sidebar-header {
        color: white;
        padding: 3px;
        border-radius: 5px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def add_header():
    """Add a custom header to the app"""
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🔮 Telecom Churn Prediction</h1>
        <p class="header-subtitle">Predict customer churn using advanced machine learning models</p>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    """Add a footer with GitHub links"""
    st.markdown("""
    <div class="footer">
        <p>
            <a href="https://github.com/Amama-Fatima/telecom-churn-prediction" target="_blank">📁 Project Repository</a> | 
            <a href="https://github.com/Amama-Fatima" target="_blank">👩‍💻 My GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Telecom Churn Prediction", 
        layout="wide",
        page_icon="🔮",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    add_custom_css()
    
    # Add header
    add_header()
    
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
    
    # Sidebar for model selection with custom styling
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🤖 Model Selection</div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Choose ML Model",
            options=available_model_names,
            index=0,
            help="Select the machine learning model for predictions"
        )
        
        # Add some info in sidebar
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        model_info = {
            'CatBoost': 'Gradient boosting with categorical features',
            'XGBoost': 'Extreme gradient boosting',
            'LightGBM': 'Light gradient boosting machine',
            'Stacking Ensemble': 'Combination of multiple models'
        }
        st.info(model_info.get(model_choice, "Advanced ML model"))
    
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
    
    # Add footer
    add_footer()

if __name__ == "__main__":
    main()