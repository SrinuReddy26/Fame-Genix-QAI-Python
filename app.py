# app.py - Genetic Health Care AI (Streamlit Version)

import streamlit as st
import google.generativeai as genai
import json
import numpy as np
from datetime import datetime
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

# ============================================
# QUANTUM SIMULATOR (SAFE & LIGHTWEIGHT)
# ============================================
class QuantumSimulator:
    @staticmethod
    def run_full_analysis(input_text: str):
        text_hash = hashlib.md5(input_text.encode()).hexdigest()
        features = [int(c, 16) / 15 for c in text_hash[:16]]

        confidence = round(np.mean(features), 3)
        entropy = float(-sum(f * np.log(f + 1e-10) for f in features))
        uncertainty = round(1 - confidence, 3)

        return {
            "confidence_score": confidence,
            "quantum_entropy": round(entropy, 3),
            "uncertainty_estimate": uncertainty,
            "quantum_correlations": round(np.std(features), 3)
        }

# ============================================
# AI HEALTH ANALYSIS (STABLE GEMINI)
# ============================================
def analyze_health(input_text, language, api_key):

    if not input_text or input_text.strip() == "":
        return {
            "diseaseName": "No input provided",
            "symptoms": [],
            "analysisSummary": ["Please enter symptoms for analysis."],
            "riskPercentage": 0,
            "riskDetails": {
                "severity": "low",
                "factors": [],
                "explanation": "No symptoms were entered."
            },
            "foodSuggestions": [],
            "doctorConsultationNeeded": False,
            "recommendedDoctorType": ""
        }

    genai.configure(api_key=api_key)

    # ✅ STABLE MODEL
    model = genai.GenerativeModel("gemini-pro")

    prompt = f"""
You are a medical AI assistant.
Return ONLY valid JSON. Do not use markdown.

Language: {language}

JSON format:
{{
  "diseaseName": "string",
  "symptoms": ["string"],
  "analysisSummary": ["string"],
  "riskPercentage": number,
  "riskDetails": {{
    "severity": "low or moderate or high",
    "factors": ["string"],
    "explanation": "string"
  }},
  "foodSuggestions": ["string"],
  "doctorConsultationNeeded": true or false,
  "recommendedDoctorType": "string"
}}

User symptoms:
{input_text}
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    # Safety cleanup
    text = text.replace("```json", "").replace("```", "").strip()

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
        placeholder="Example:\nSymptoms: fever, headache, fatigue\nDuration: 2 days\nAge: 22"
    )

    if st.button("🔬 Analyze Health", type="primary"):
        with st.spinner("Analyzing..."):
            quantum = QuantumSimulator.run_full_analysis(input_text)
            result = analyze_health(input_text, language, api_key)

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

    st.su
