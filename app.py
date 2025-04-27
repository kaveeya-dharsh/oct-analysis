import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import pdfkit
import os
import datetime
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(page_title="AMD Detector", page_icon="🩺", initial_sidebar_state='collapsed')

# Custom CSS for background image and UI styling
st.markdown(
    """
    <style>
        /* Background Image */
        .stApp {
            background: url('https://w0.peakpx.com/wallpaper/459/390/HD-wallpaper-gray-dark-black-black-dark-gradation-gray.jpg') no-repeat center center fixed;
            background-size: cover;
        }

        /* Increase Font Size */
        h1, h2 {
            font-size: 28px !important;
            text-align: center;
        }
        p {
            font-size: 18px;
        }

        /* Centering the Text */
        .stMarkdown {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load DenseNet-121 Model
@st.cache_resource
def load_model():
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(1024, 2)
    model.load_state_dict(torch.load("models/densenet121-oct-5metrics-v1.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Image Preprocessing
def preprocess_image(image):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Model Prediction
def predict(image_tensor, model):
    output = model(image_tensor)
    probabilities = F.softmax(output, dim=1)
    category = torch.argmax(probabilities, dim=1).tolist()
    return category, probabilities.tolist()

# Generate Confidence Graph
def generate_confidence_graph(prediction, confidence):
    labels = ["NORMAL", "ABNORMAL"]
    confidences = [100 - confidence, confidence] if prediction == "ABNORMAL" else [confidence, 100 - confidence]

    plt.figure(figsize=(5, 5))
    plt.bar(labels, confidences, color=['green', 'red'])
    plt.xlabel("Prediction Category")
    plt.ylabel("Confidence (%)")
    plt.title("Model Confidence")
    plt.ylim(0, 100)
    
    graph_path = "confidence_chart.png"
    plt.savefig(graph_path)
    plt.close()
    
    return graph_path

# Doctor Consultation Recommendation
def get_recommendation(prediction, confidence):
    if prediction == "ABNORMAL" and confidence >= 85:
        return "Immediate consultation with an ophthalmologist is advised."
    elif prediction == "ABNORMAL" and confidence < 85:
        return "Consultation is recommended. Consider further clinical tests."
    else:
        return "No immediate concerns detected, but regular eye check-ups are advised."

# Generate PDF Report
def generate_pdf_report(filename, image_path, graph_path, prediction, confidence, patient_name, age, doctor_name, notes):
    pdfkit_config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AMD Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ text-align: center; }}
            p {{ font-size: 16px; }}
            .container {{ max-width: 800px; margin: auto; }}
            .image-container, .graph-container {{ text-align: center; margin-top: 20px; }}
            .image-container img, .graph-container img {{ width: 80%; max-width: 500px; border-radius: 10px; }}
            .normal {{ color: green; font-weight: bold; }}
            .abnormal {{ color: red; font-weight: bold; }}
            .patient-info {{ background: #f3f3f3; padding: 10px; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RetinaVision AI - Prediction Report</h1>
            <p><b>Report Generated:</b> {timestamp}</p>

            <div class="patient-info">
                <h2>Patient Information</h2>
                <p><b>Name:</b> {patient_name}</p>
                <p><b>Age:</b> {age} years</p>
                <p><b>Doctor:</b> {doctor_name}</p>
                <p><b>Medical History:</b> {notes}</p>
                <p><b>Symptoms:</b> {symptoms}</p>
            </div>

            <h2>Model Prediction</h2>
            <p><b>Filename:</b> {filename}</p>
            <p><b>Prediction:</b> <span class="{prediction.lower()}">{prediction}</span></p>
            <p><b>Confidence:</b> {confidence}%</p>
            <p><b>Recommendation:</b> {get_recommendation(prediction, confidence)}</p>

            <div class="image-container">
                <p><b>OCT Image:</b></p>
                <img src="{image_path}" alt="{filename}">
            </div>

            <div class="graph-container">
                <p><b>Confidence Distribution:</b></p>
                <img src="{graph_path}" alt="Confidence Chart">
            </div>

            <p>⚠️ <i>Disclaimer: This tool is for assistive screening only..</i></p>
        </div>
    </body>
    </html>
    """

    pdf_filename = f"{filename.split('.')[0]}.pdf"
    options = {'enable-local-file-access': None, 'encoding': "UTF-8"}

    html_file = f"{filename.split('.')[0]}.html"
    with open(html_file, "w", encoding="utf-8") as file:
        file.write(html_template)

    pdfkit.from_file(html_file, pdf_filename, configuration=pdfkit_config, options=options)

    return pdf_filename

# Streamlit UI
st.title("🩺 Smart OCT Analysis-AMD Detection")
st.markdown("Upload **retinal OCT images** to check for Age-related Macular Degeneration (AMD).")

# User Inputs
patient_name = st.text_input("👤 Patient Name", value="John Doe")
age = st.number_input("📅 Age", min_value=1, max_value=120, value=45)
doctor_name = st.text_input("🩺 Doctor Name", value="Dr. Smith")
notes = st.text_area("📝 Medical History", value="No prior medical history.")
symptoms = st.text_area("👁 Symptoms",value="blur vision")

uploaded_files = st.file_uploader("📤 Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        image_tensor = preprocess_image(image)

        prediction, confidence_scores = predict(image_tensor, model)
        category = "ABNORMAL" if prediction[0] == 0 else "NORMAL"
        confidence = round(float(confidence_scores[0][prediction[0]]) * 100, 2)

        image_path = f"temp_{uploaded_file.name}"
        image.save(image_path)

        graph_path = generate_confidence_graph(category, confidence)

        pdf_filename = generate_pdf_report(uploaded_file.name, image_path, graph_path, category, confidence, patient_name, age, doctor_name, notes)
        
        with open(pdf_filename, "rb") as pdf_file:
            st.download_button(f"📄 Download Report - {uploaded_file.name}", data=pdf_file, file_name=pdf_filename, mime="application/pdf")
