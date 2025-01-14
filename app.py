import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# Load the trained model
model = load_model("material_classifier.h5")

# Class labels
class_names = {0: "Cardboard", 1: "Plastic", 2: "Metal", 3: "Glass"}

# Function to preprocess image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to generate carbon footprint information
def generate_carbon_footprint_info(material):
    # Example carbon footprint information (you can replace it with static text or other data if needed)
    carbon_footprint_data = {
        "Cardboard": "Cardboard production typically generates around 0.9 kg of CO2 per kg of material.",
        "Plastic": "Plastic production generates roughly 6 kg of CO2 for every kilogram of plastic produced.",
        "Metal": "Metal production, especially steel, can result in around 1.8 kg of CO2 per kg of metal produced.",
        "Glass": "Glass production contributes around 1.2 kg of CO2 per kg of glass produced."
    }
    return carbon_footprint_data.get(material, "Information not available.")

# SDG Goal Images Mapping
sdg_images = {
    "Cardboard": ["sdg goals/12.png", "sdg goals/13.png", "sdg goals/15.png"],
    "Plastic": ["sdg goals/6.jpg", "sdg goals/12.png", "sdg goals/14.png"],
    "Glass": ["sdg goals/12.png", "sdg goals/14.png"],
    "Metal": ["sdg goals/3.png", "sdg goals/6.jpg", "sdg goals/12.png"],
}

# Streamlit App
st.set_page_config(layout="wide")
st.title("Material Classifier & Sustainability Awareness")

st.write("Upload an image to classify it as Cardboard, Plastic, Metal, or Glass.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image with reduced size
   # Display uploaded image with reduced size
    st.image(uploaded_file, caption="Uploaded Image", width=300)  # Adjust the width value to your preference


    # Process the uploaded image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image, target_size=(224, 224))

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence_score = prediction[0][predicted_class_index]

    # Create columns for layout
    col1, col2, col3 = st.columns([2, 3, 2])

    # Display classification result in the first column
    with col1:
        st.subheader("Prediction Results")
        st.write(f"Class: {predicted_class_name}")
        st.write(f"Confidence Score: {confidence_score:.2f}")

    # Display carbon footprint information in the second column
    with col2:
        st.subheader("Carbon Footprint Information")
        info = generate_carbon_footprint_info(predicted_class_name)
        st.write(info)

    # Display SDG goal images in the third column
    with col3:
        st.subheader("Related SDG Goals")
        sdg_cols = st.columns(len(sdg_images[predicted_class_name]))
        for i, img_path in enumerate(sdg_images[predicted_class_name]):
            with sdg_cols[i]:
                st.image(img_path, use_container_width=True, width=250)  # Set width for SDG images