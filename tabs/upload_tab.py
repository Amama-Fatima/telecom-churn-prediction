import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.data_processor import process_predictions

def upload_tab(models, model_key):
    """Handle file upload functionality"""
    st.header("Upload Your Data")
    

    uploaded_file = st.file_uploader("Upload customer data (CSV)", type="csv")
    

    eval_mode = st.checkbox("Enable evaluation mode (provide true labels)")
    true_labels_file = None
    if eval_mode:
        true_labels_file = st.file_uploader("Upload true labels (original data with Churn column)", type="csv")

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.subheader("Input Data Info")
            st.write(f"Shape: {input_df.shape}")
            st.write("First few rows:")
            st.dataframe(input_df.head())
            
            results = process_predictions(models, input_df, model_key)
            
            if results is not None:
                if eval_mode and true_labels_file is not None:
                    try:
                        true_df = pd.read_csv(true_labels_file)
                        if 'Churn' in true_df.columns:
                            if true_df['Churn'].dtype == 'object':
                                true_labels = true_df['Churn'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).values
                            else:
                                true_labels = true_df['Churn'].values
                            
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
                
                # Show results
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