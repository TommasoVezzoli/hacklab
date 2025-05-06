# **README**

This repository contains the code for the HackLab challenge made by
- Beatrice Citterio
- Filippo Focaccia
- Giulio Pirotta
- Martina Serandrei
- Tommaso Vezzoli

## Project overview

The goal of this project was to analyze data from a fictitious telecommunications company, focusing on churn prediction.  
The dataset contained information about customers' demographics, subscription details, and the goal was to develop a machine learning model to predict their likelihood of churning.  
To achieve this, we followed a structured approach, including the classic data exploration, preprocessing, and model selection.   
Feature engineering was crucial in this sense to summarize redundant information and create new attributes to improve the model's performance.

Everything was developed with a focus on delivering actionable business insights. For instance, when comparing models with similar performance, we prioritized those that were more interpretable and had a lower false negative rate. This approach minimizes the risk of misclassifying churners as non-churners, helping the company retain customers and avoid potential revenue loss.

We then integrated the machine learning approach with some textual analysis to provide other, more interpretable insights on the customer complaints.  
The techniques used in this section included:
- word frequency examination and topic modeling to identify the main themes in the complaints and verify whether they aligned with the factors identified as leading to churn in the machine learning models
- sentiment analysis to understand how the complaints' tone evolved over time

All features are available in a Streamlit web application, which comprises two sections:

1. **Customer complaint analysis**:  
Perform textual analysis on customer complaints and leverage generative AI tools to
- Draft plausible complaints given a user's complaint history
- Predict the churn probability of a customer from their past complaints

2. **Customer churn prediction**:

## Project structure

    hacklab/
    ├── data/
    ├── docs/                   # Visualization file
    ├── mappings/               # Label encoders for categorical variables
    ├── models/                 # Pre-trained models and vectorizers
    ├── notebooks/              # Jupyter notebooks for data exploration and analysis
    ├── src/
    │   ├── app.py              # Main Streamlit app
    |   ├── complaints.py       # Functions to manipulate the complaints
    │   ├── genai.py            # Functions to draft future complaints
    │   ├── ml.py               # Machine learning utilities for churn prediction
    |   ├── openai_api.py       # Functions to connect to the OpenAI api
    ├── README.md
    ├── requirements.txt
    ├── sample_complaint.txt    # Sample complaint
    └── sample_data.csv         # Sample customer data

## Usage

Using the streamlit web application is extremely simple:

1. Clone the repository
    ```
    git clone https://github.com/beatricecitterio/hacklab.git
    cd hacklab
    ```

2. Install the dependencies
    ```
    pip install -r requirements.txt
    ```

3. Run the web app
    ```
    streamlit run src/app.py
    ```

Alternatively, the web app can be found at [this link](https://hacklab-ai-team.streamlit.app/).

## Visualization

The visualization of topic modeling can be found at [this link](https://rawcdn.githack.com/beatricecitterio/hacklab/refs/heads/master/docs/ldavis5.html).
