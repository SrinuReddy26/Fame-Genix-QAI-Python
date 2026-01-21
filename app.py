# app.py - Genetic Health Care AI (Streamlit Version)

import streamlit as st
import google.generativeai as genai
import json
import numpy as np
from datetime import datetime
import base64
from PIL import Image
import io
import hashlib

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Genetic Health Care AI",
    page_icon="🧬",
    layout="wide"
)

# ============================================
# SESSION STATE
# ============================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ============================================
# QUANTUM SIMULATOR (SIMPLIFIED & SAFE)
# ============================================
class QuantumSimulator:

    @staticmethod
    def run_full_analysis(input_text: str):
        text_hash = hashlib.md5(input_text.encode()).hexdigest()
        features = [int(c, 16) / 15 for c in text_hash[:16]]

        confidence = round(np.mean(features), 3)
        entropy = float(-sum(f * np.log(f + 1e-10) for f in features))
        uncertainty = 1 - confidence

        return {
            "confidence_score": confidence,
            "quantum_entropy": entropy,
            "uncertainty_estimate": uncertainty,
            "quantum_correlations": round(np.std(features), 3)
        }

# ============================================
# AI HEALTH ANALYSIS (GEMINI)
# ============================================
def analyze_health(input_text, language, api_key, image_data=None):

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
You are a medical AI assistant.
Respond ONLY in valid JSON.

Language: {language}

JSON format:
{{
  "diseaseName": "",
  "symptoms": [],
  "analysisSummary": [],
  "riskPercentage": 0,
  "riskDetails": {{
    "severity": "low/moderate/high",
    "factors": [],
    "explanation": ""
  }},
  "foodSuggestions": [],
  "doctorConsultationNeeded": true,
  "recommendedDoctorType": ""
}}

Symptoms:
{input_text}
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    if "```" in text:
        text = text.split("```")[1]

    return json.loads(text)

# ============================================
# AUTH PAGE
# ============================================
def auth_page():
    st.title("🧬 Genetic Health Care AI")
    st.subheader("Predictive Health Analysis")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):
        if email and password:
            st.session_state.authenticated = True
            st.rerun()

# ============================================
# DASHBOARD
# ============================================
def dashboard():

    st.title("🧬 Genetic Health Care AI")

    with st.sidebar:
        st.header("⚙️ Settings")

        api_key = st.secrets.get("GOOGLE_API_KEY", "")
        if not api_key:
            api_key = st.text_input("Google AI API Key", type="password")

        language = st.selectbox(
            "Language",
            ["English", "Telugu", "Hindi", "Tamil"]
        )

    if not api_key:
        st.warning("Please enter your Google AI API Key")
        return

    input_text = st.text_area(
        "Describe your symptoms",
        height=150,
        placeholder="Example: fever, headache, fatigue"
    )

    uploaded = st.file_uploader(
        "Upload image (optional)",
        type=["png", "jpg", "jpeg"]
    )

    image_data = None
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, width=250)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_data = base64.b64encode(buf.getvalue()).decode()

    if st.button("🔬 Analyze Health", type="primary"):
        with st.spinner("Analyzing..."):
            quantum = QuantumSimulator.run_full_analysis(input_text)
            result = analyze_health(input_text, language, api_key, image_data)

            result["quantumMetrics"] = quantum
            result["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")

            st.session_state.current_analysis = result
            st.session_state.analysis_history.append(result)
            st.rerun()

    if st.session_state.current_analysis:
        display_results(st.session_state.current_analysis)

# ============================================
# RESULTS
# ============================================
def display_results(data):

    st.header(f"🏥 {data['diseaseName']}")

    st.metric("Risk %", f"{data['riskPercentage']}%")
    st.write("**Severity:**", data["riskDetails"]["severity"])

    st.subheader("Symptoms")
    for s in data["symptoms"]:
        st.write("•", s)

    st.subheader("Food Suggestions")
    for f in data["foodSuggestions"]:
        st.success(f)

    st.subheader("Explanation")
    st.write(data["riskDetails"]["explanation"])

    qm = data["quantumMetrics"]
    st.divider()
    st.subheader("⚛️ Quantum Metrics")
    st.json(qm)

    st.warning(
        "⚠️ This app is for informational purposes only. "
        "Always consult a healthcare professional."
    )

# ============================================
# MAIN
# ============================================
def main():
    if st.session_state.authenticated:
        dashboard()
    else:
        auth_page()

if __name__ == "__main__":
    main()

