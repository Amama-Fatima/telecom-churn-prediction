import streamlit as st
import pandas as pd
from utils.data_processor import process_predictions

def create_manual_input_form():
    st.subheader("Enter Customer Details")
    
    with st.form("customer_input"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            
            st.write("**Phone Services**")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        
        with col2:
            st.write("**Internet Services**")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            
            st.write("**Streaming Services**")
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        with col3:
            st.write("**Contract & Billing**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", 
                                        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
            st.write("**Financial Details**")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, help="How long the customer has been with the company")
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=0.01)
        
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
        
        if submitted:
            input_data = pd.DataFrame({
                'gender': [gender],
                'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
            })
            
            return input_data
    
    return None

def manual_input_tab(models, model_key):
    """Handle manual input functionality"""
    st.header("Manual Customer Input")
    st.write("Enter individual customer details to predict churn probability")
    
    # Create manual input form
    manual_input = create_manual_input_form()
    
    if manual_input is not None:
        results = process_predictions(models, manual_input, model_key)
        
        if results is not None:
            st.subheader("🎯 Prediction Result")
            
            churn_prob = results['Churn_Probability'].iloc[0]
            churn_label = results['Predicted_Churn_Label'].iloc[0]
            
            # Create result display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Probability", f"{churn_prob:.2%}")
            
            with col2:
                color = "red" if churn_label == "Yes" else "green"
                st.markdown(f"**Prediction:** <span style='color: {color}; font-size: 20px;'>{churn_label}</span>", unsafe_allow_html=True)
            
            with col3:
                # Risk level
                if churn_prob >= 0.7:
                    st.error("⚠️ High Risk")
                elif churn_prob >= 0.3:
                    st.warning("⚡ Medium Risk")
                else:
                    st.success("✅ Low Risk")
            
            st.subheader("📊 Interpretation")
            if churn_prob >= 0.7:
                st.error("🚨 **High churn risk** - Consider immediate retention efforts! This customer is very likely to churn.")
                st.write("**Recommendations:**")
                st.write("- Immediate personal outreach")
                st.write("- Special retention offers")
                st.write("- Account review and optimization")
            elif churn_prob >= 0.3:
                st.warning("⚠️ **Medium churn risk** - Monitor closely and consider preventive measures.")
                st.write("**Recommendations:**")
                st.write("- Proactive engagement")
                st.write("- Customer satisfaction survey")
                st.write("- Service improvement offers")
            else:
                st.success("✅ **Low churn risk** - Customer is likely to stay!")
                st.write("**Recommendations:**")
                st.write("- Continue current service level")
                st.write("- Consider upselling opportunities")
                st.write("- Maintain regular communication")
            
            # Show detailed results
            st.subheader("📋 Customer Details & Full Prediction")
            
            # Create a more readable display
            display_results = results.copy()
            display_results['Churn_Probability'] = display_results['Churn_Probability'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(display_results, use_container_width=True)
            
            # Download option
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Prediction",
                data=csv,
                file_name='single_customer_prediction.csv',
                mime='text/csv'
            )