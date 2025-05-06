import os
import pandas as pd
from complaints import merge_complaints_by_customer, preprocess_complaint, get_most_similar_complaints
from sentence_transformers import SentenceTransformer
from src.openai_api import send_request, build_prompt


def draft_future_complaint(complaints: str, api_key: str) -> str:

    # Load the complaints
    complaints_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'Complaints.csv'))
    complaints_df['complaint'] = complaints_df['complaint'].astype(str)

    # Merge complaints and build the vectorizer
    merged_complaints = merge_complaints_by_customer(complaints_df)
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # Get the most similar complaints
    similar_complaints = get_most_similar_complaints(
        target=preprocess_complaint(complaints),
        complaints=merged_complaints,
        encoder=encoder
    )

    ### -------------------------------------------------- ###
    ### --- Build the prompt ----------------------------- ###

    instructions = "You are a helpful customer support assistant. " \
    "Given a customer's complaint history and similar complaint histories from other users, your task is to predict the most probable next complaint this customer is likely to make. " \
    "Use patterns you observe in similar users' histories to guide your prediction. " \
    "Be realistic, concise, and use a tone consistent with the customer's prior complaints."

    prompt = f"Customer's complaint history: {complaints}\n" \
    f"Similar complaints from other users: {similar_complaints}\n" \
    "Each history consists of numbered complaints (e.g., 1:, 2:, 3:) in chronological order from a single customer. " \
    "Based on the customer's pattern and the progression of similar customers' complaints, predict the most likely next complaint that this customer might make. " \
    "Only output the predicted complaint, not an explanation."

    prompt = build_prompt(prompt, instructions)

    ### -------------------------------------------------- ###
    ### --- Send the request ----------------------------- ###

    status_code, response = send_request(
        input=prompt,
        model="gpt-4.1-mini",
        temperature=0.9,
        max_tokens=1000,
        api_key=api_key
    )
    if status_code != 200:
            return "Error: Unable to process the request. Please try again later."
    
    next_complaint = response.output[0].content[0].text
    # print("Next complaint prediction:")
    # print(next_complaint)
    return next_complaint


if __name__ == "__main__":
    # Example usage
    complaints = "1: The streaming TV service frequently buffers or crashes, making it impossible for me to watch anything without interruptions. This has been ongoing despite my high monthly charges, and I am very frustrated with the lack of reliability."
    draft_future_complaint(complaints)