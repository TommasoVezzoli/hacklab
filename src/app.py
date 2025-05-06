import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.genai import draft_future_complaint
import streamlit as st
import src.ml as ml
import joblib

if "init" not in st.session_state:
    st.session_state.init = True

    st.session_state.input_complaint = None
    st.session_state.draft_complaint = None
    st.session_state.input_churn_p = None
    st.session_state.draft_churn_p = None

    st.session_state.results = None
    st.session_state.retention_factor = 0.05
    st.session_state.retention_period = 6


st.markdown(
    """
    ### Complaint analysis
    The complaint analysis currently supports two main features:

    ##### 1. Draft future complaints:
    Input all complaints from a customer to get a draft of the potential following complaint based on:
    - The customer's complaints
    - Similar complaints from other customers

    ##### 2. Predict churn:
    Input a complaint to get a prediction of the likelyhood of the customer churning based on  
    a text-classification model trained on other complaints.

    **Note:** The text should follow the format of this sample complaint:
    """
)

with open(os.path.join(os.getcwd(), "sample_complaint.txt"), "r") as file:
    complaint_content = file.read()

st.download_button(
    label="Sample complaint",
    data=complaint_content,
    file_name="sample_complaint.txt",
    mime="text/plain"
)
api_key = st.text_input(
    label="OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Get your API key from https://platform.openai.com/account/api-keys"
)
do_draft = st.checkbox("Draft a future complaint", value=False)
do_churn = st.checkbox("Predict churn", value=False)
complaint = st.text_area(label="Insert the complaint here. Erase it to clear the prediction.")

if complaint:
    draft_complaint_ = st.session_state.draft_complaint
    input_churn_p_ = st.session_state.input_churn_p
    draft_churn_p_ = st.session_state.draft_churn_p
    
    if st.session_state.input_complaint == complaint:
        if not do_draft:
            draft_complaint_ = None
        if not do_churn:
            input_churn_p_ = None
            draft_churn_p_ = None
        else:
            if not do_draft: 
                draft_churn_p_ = None
        pass

    else:
        st.session_state.input_complaint = complaint
        
        with st.spinner("Analysing the complaint..."):
            next_complaint = draft_future_complaint(complaint, api_key)
            st.session_state.draft_complaint = next_complaint

            models_path = os.path.join(os.getcwd(), "models")
            vectorizer = joblib.load(os.path.join(models_path, "tfidf_vectorizer.pkl"))
            model = joblib.load(os.path.join(models_path, "complaint_classifier.pkl"))

            complaint_vec = vectorizer.transform([complaint])
            st.session_state.input_churn_p = model.predict_proba(complaint_vec)[0][1]
            next_complaint_vec = vectorizer.transform([next_complaint])
            st.session_state.draft_churn_p = model.predict_proba(next_complaint_vec)[0][1]

    if draft_complaint_ is not None:
        st.write("Drafted complaint:")
        st.write(draft_complaint_)
    if input_churn_p_ is not None:
        st.write(f"Predicted churn probability for the input complaint: **{input_churn_p_:.2%}**")
    if draft_churn_p_ is not None:
        st.write(f"Updated probability for the drafted one: **{draft_churn_p_:.2%}**")
        
### -------------------------------------------------- ###

# ML-based churn prediction section
st.markdown(
    """
    ---

    ### ML-based churn prediction

    Load a file of customers (CSV or Excel) to get churn predictions based on a classification model  
    trained on customers' data.  
    The model will also provide risk segmentation and profit analysis.

    **Note:** The file should follow the format of this sample:
    """
)

with open(os.path.join(os.getcwd(), "sample_data.csv"), "r") as file:
    csv_content = file.read()
st.download_button(
    label="Sample file",
    data=csv_content,
    file_name="sample_data.csv",
    mime="text/csv"
)

# Retention cost parameters
st.markdown("##### Retention cost parameters")
col1, col2, _ = st.columns((1, 1, 0.3))
with col1:
    retention_factor = st.number_input(
        "Retention cost factor (\% of monthly charges)",
        min_value=0.01,
        max_value=1.0,
        value=st.session_state.retention_factor,
        step=0.01,
        help="This factor multiplies the monthly charges to estimate retention cost (e.g., 0.05 = 5% of monthly charges)"
    )
with col2:
    retention_period = st.number_input(
        "Retention period (months)",
        min_value=1,
        max_value=24,
        value=st.session_state.retention_period,
        step=1,
        help="Number of months over which retention costs are calculated"
    )

# Save parameter values to session state
st.session_state.retention_factor = retention_factor
st.session_state.retention_period = retention_period

# File uploader for ML-based prediction
with col1:
    uploaded_file = st.file_uploader("Upload Customer Data (CSV or Excel)", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    
    # Override the retention cost calculation in ml.py by modifying the module
    ml.retention_factor = retention_factor
    ml.retention_period = retention_period
    
    # Process the file and get results
    results = ml.predict_churn(uploaded_file, st)
    st.session_state.results = results

    if results is not None:
        displayed_results = results.copy()
        # Search functionality
        st.markdown("##### Search by customer ID")
        customer_search = st.text_input("Enter customer ID to search", "")
        
        if customer_search:
            # Filter results by customer ID (case-insensitive partial match)
            filtered_results = results[results["Customer_ID"].astype(str).str.contains(customer_search, case=False)]
            
            if not filtered_results.empty:
                st.write(f"Found {len(filtered_results)} matching customers:")
                # st.dataframe(filtered_results, hide_index=True)  # Remove index when displaying
                displayed_results = filtered_results.copy()
            else:
                st.warning(f"No customers found matching '{customer_search}'")
        
        # Display all results
        st.dataframe(displayed_results, hide_index=True)
        
        # Download button for results
        csv = displayed_results.to_csv(index=False)  # Remove index when exporting
        st.download_button(
            label="Download predictions",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
