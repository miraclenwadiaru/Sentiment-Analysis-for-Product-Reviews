# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:46:11 2024

@author: USER
"""
#import the necessary library
import streamlit as st
import pickle
import os
import spacy

# Load the spaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# Get the user's desktop directory
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

# Specify the model path
model_filename = 'classifier.pkl'
model_path = os.path.join(desktop_path, model_filename)

#load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

#load the vectorizer    
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


def main():
    st.title('Amazon Sentiment Analysis App')

    # Option to enter text
    user_input_option = st.radio("Choose Input Option:", ["Enter Text", "Upload File"])

    if user_input_option == "Enter Text":
        # Text input for user to enter a review
        user_input = st.text_area("Enter your Amazon product review here:")

    elif user_input_option == "Upload File":
        # File upload for user to upload a text file
        uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
        if uploaded_file is not None:
            user_input = uploaded_file.read().decode("utf-8")
        else:
            user_input = ""

    if st.button("Analyze"):
        # Check if the user entered any text
        if user_input:
            # Preprocess the text
            doc = nlp(user_input)
            cleaned_text = ' '.join([token.lemma_ for token in doc])
            
            user_input_vectorized = vectorizer.transform([cleaned_text]).toarray()


            # Make sentiment prediction
            prediction = model.predict(user_input_vectorized)

            # Determine sentiment
            sentiment = "Positive sentiment" if prediction[0] == 1 else "Negative sentiment"

            # Display the prediction result
            st.write("Sentiment Prediction:", sentiment)
        else:
            st.warning("Please enter a product review or upload a text file.")


if __name__ == "__main__":
    main()



