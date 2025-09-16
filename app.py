# import streamlit as st 
# st.title('Hello World') 
# st.write("This is my first web app built using Streamlit!")

# app.py
# Purpose: A Streamlit web app to classify text using the trained PyTorch model.

import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np

# --- 1. Load Model, Vectorizer, and Metadata ---

# Define the exact same neural network architecture as in the training script
class NewsMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Use caching to load the model and vectorizer only once
@st.cache_resource
def load_model_and_vectorizer():
    """Loads the saved model state, vectorizer, and class names."""
    try:
        # Define model parameters
        INPUT_DIM = 5000  # Must match max_features in TfidfVectorizer
        NUM_CLASSES = 20  # 20 newsgroups

        # Instantiate the model
        model = NewsMLP(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
        
        # Load the saved state dictionary
        model.load_state_dict(torch.load('news_classifier_model.pth', map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode

        # Load the fitted vectorizer
        vectorizer = joblib.load('vectorizer.joblib')
        
        # Manually define the class names in order
        class_names = [
            'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
            'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
            'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
            'talk.politics.misc', 'talk.religion.misc'
        ]
        
        return model, vectorizer, class_names
    except FileNotFoundError:
        return None, None, None

# Load the resources
model, vectorizer, class_names = load_model_and_vectorizer()

# --- 2. Prediction Function ---

def predict_category(text):
    """Vectorizes input text and returns the predicted category and confidence."""
    if model is None or vectorizer is None:
        return "Error: Model or vectorizer not found.", 0.0
        
    # 1. Preprocess and vectorize the text
    text_vec = vectorizer.transform([text]).toarray()
    
    # 2. Convert to PyTorch tensor
    text_tensor = torch.tensor(text_vec, dtype=torch.float32)
    
    # 3. Get model prediction (logits)
    with torch.no_grad():
        logits = model(text_tensor)
        
    # 4. Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)
    
    # 5. Get the top prediction
    confidence, predicted_idx = torch.max(probabilities, dim=1)
    predicted_class = class_names[predicted_idx.item()]
    
    return predicted_class, confidence.item()

# --- 3. Streamlit UI ---

st.set_page_config(page_title="20 Newsgroups Classifier", layout="wide")
st.title("📰 20 Newsgroups Document Classifier")
st.write("Enter a piece of text (like a product review or news article snippet), and this AI model will predict which of the 20 newsgroups it belongs to.")

if model is None:
    st.error("Model files not found! Please run `train_model.py` first to generate `news_classifier_model.pth` and `vectorizer.joblib`.")
else:
    # Text input area
    input_text = st.text_area(
        "Enter text to classify:",
        height=200,
        placeholder="e.g., 'The new graphics card offers amazing performance for 3D rendering, but the drivers need some work...'"
    )

    # Classify button
    if st.button("Classify Text", type="primary"):
        if input_text.strip():
            with st.spinner('Analyzing text...'):
                category, confidence = predict_category(input_text)
                st.success(f"**Predicted Category:** `{category}`")
                st.info(f"**Confidence:** `{confidence:.2%}`")
        else:
            st.warning("Please enter some text to classify.")