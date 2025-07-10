import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.data_processor import load_demo_data, process_predictions

def demo_data_tab(models, model_key):
    """Handle demo data functionality"""
    st.header("Demo Data")
    st.write("Try the app with sample datasets from our repository")
    
    demo_data = load_demo_data()
    
    if demo_data:
        has_test_data = 'test' in demo_data
        has_eval_data = 'eval' in demo_data
        
        if not (has_test_data and has_eval_data):
            st.error("❌ Required demo datasets not found. Please ensure both 'test' and 'eval' datasets are available.")
            return
        
        test_data = demo_data['test']
        eval_data = demo_data['eval']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Test Dataset (For Prediction)")
            st.write("*This dataset contains customer features without churn labels*")
            
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.metric("Rows", test_data.shape[0])
            with subcol2:
                st.metric("Columns", test_data.shape[1])
            
            st.write("**Sample Data:**")
            st.dataframe(test_data.head(5))
            
            if st.checkbox("Show full test dataset"):
                st.dataframe(test_data)
        
        with col2:
            st.subheader("📊 Evaluation Dataset (With Labels)")
            st.write("*This dataset contains customer features with actual churn labels*")
            
            # Show eval dataset info
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.metric("Rows", eval_data.shape[0])
            with subcol2:
                st.metric("Columns", eval_data.shape[1])
            
            if 'Churn' in eval_data.columns:
                churn_rate = eval_data['Churn'].value_counts().get('Yes', 0) / len(eval_data)
                st.metric("Churn Rate", f"{churn_rate:.1%}")
            
            st.write("**Sample Data:**")
            st.dataframe(eval_data.head(5))
            
            if st.checkbox("Show full evaluation dataset"):
                st.dataframe(eval_data)
        
        st.divider()
        
        st.subheader("🚀 Model Actions")
        
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            predict_button = st.button(
                "🔮 Run Prediction on Test Data", 
                use_container_width=True,
                help="Make predictions on unlabeled test data"
            )
        
        with button_col2:
            evaluate_button = st.button(
                "🎯 Evaluate Model Performance", 
                use_container_width=True,
                help="Evaluate model accuracy using labeled evaluation data"
            )
        
        # Handle prediction on test data
        if predict_button:
            st.subheader("🔮 Prediction Results")
            
            with st.spinner("Running predictions on test data..."):
                # Process predictions on test data
                results = process_predictions(models, test_data, model_key)
                
                if results is not None:
                    predicted_churn_count = results['Predicted_Churn'].sum()
                    total_customers = len(results)
                    predicted_churn_rate = predicted_churn_count / total_customers
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", total_customers)
                    with col2:
                        st.metric("Predicted Churners", predicted_churn_count)
                    with col3:
                        st.metric("Predicted Churn Rate", f"{predicted_churn_rate:.1%}")
                    
                    # Show prediction results
                    st.write("**Prediction Results:**")
                    
                    # Add readable labels
                    display_results = results.copy()
                    display_results['Churn_Prediction'] = display_results['Predicted_Churn'].map({1: 'Yes', 0: 'No'})
                    display_results['Churn_Probability'] = display_results['Churn_Probability'].round(3)
                    
                    # Show results with option to download
                    st.dataframe(display_results[['Churn_Prediction', 'Churn_Probability']])
                    
                    csv = display_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="test_predictions.csv",
                        mime="text/csv"
                    )
        
        # Handle evaluation
        if evaluate_button:
            st.subheader("🎯 Model Evaluation")
            
            with st.spinner("Evaluating model performance..."):
                if 'Churn' in eval_data.columns:
                    input_df = eval_data.drop('Churn', axis=1)
                    
                    if eval_data['Churn'].dtype == 'object':
                        true_labels = eval_data['Churn'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
                    else:
                        true_labels = eval_data['Churn']
                    
                    if true_labels.isna().any():
                        st.error("❌ Error: Some values in the Churn column could not be mapped. Please ensure all values are 'Yes'/'No' or 1/0.")
                        st.write("**Unique values found in Churn column:**", eval_data['Churn'].unique())
                        return
                    
                    # Process predictions
                    results = process_predictions(models, input_df, model_key)
                    
                    if results is not None:
                        # Check if lengths match
                        if len(true_labels) != len(results):
                            st.error(f"❌ Mismatch in data length: Evaluation data has {len(true_labels)} rows, predictions have {len(results)} rows")
                            return
                        
                        # Calculate evaluation metrics
                        accuracy = accuracy_score(true_labels, results['Predicted_Churn'])
                        conf_matrix = confusion_matrix(true_labels, results['Predicted_Churn'])
                        
                        # Display evaluation metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.2%}")
                        
                        with col2:
                            precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]) if (conf_matrix[1, 1] + conf_matrix[0, 1]) > 0 else 0
                            st.metric("Precision", f"{precision:.2%}")
                        
                        with col3:
                            recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) if (conf_matrix[1, 1] + conf_matrix[1, 0]) > 0 else 0
                            st.metric("Recall", f"{recall:.2%}")
                        
                        with col4:
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            st.metric("F1-Score", f"{f1:.2%}")
                        
                        # Confusion Matrix
                        st.write("**Confusion Matrix:**")
                        conf_df = pd.DataFrame(conf_matrix,
                                            index=['Actual No Churn', 'Actual Churn'],
                                            columns=['Predicted No Churn', 'Predicted Churn'])
                        st.dataframe(conf_df, use_container_width=True)
                        
                        # Performance interpretation
                        st.subheader("📈 Performance Analysis")
                        if accuracy >= 0.85:
                            st.success(f"🎉 Excellent performance! The model achieves {accuracy:.1%} accuracy on this dataset.")
                        elif accuracy >= 0.75:
                            st.info(f"👍 Good performance! The model achieves {accuracy:.1%} accuracy on this dataset.")
                        else:
                            st.warning(f"⚠️ The model achieves {accuracy:.1%} accuracy. Consider model improvement.")
                        
                        # Show detailed results
                        st.write("**Detailed Results:**")
                        detailed_results = pd.DataFrame({
                            'Actual_Churn': pd.Series(true_labels).map({1: 'Yes', 0: 'No'}),
                            'Predicted_Churn': results['Predicted_Churn'].map({1: 'Yes', 0: 'No'}),
                            'Churn_Probability': results['Churn_Probability'].round(3),
                            'Correct_Prediction': pd.Series(true_labels == results['Predicted_Churn']).map({True: '✅', False: '❌'})
                        })
                        
                        st.dataframe(detailed_results)
                        
                        csv = detailed_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Evaluation Results as CSV",
                            data=csv,
                            file_name="evaluation_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("❌ Evaluation dataset doesn't contain 'Churn' column for evaluation.")
    else:
        st.error("❌ No demo data available. Please check your data_model folder.")