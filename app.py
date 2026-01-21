# app.py - Genetic Health Care AI (Streamlit Version)
# Install dependencies: pip install streamlit google-generativeai pillow numpy scipy

import streamlit as st
import google.generativeai as genai
import json
import numpy as np
from scipy import linalg
from datetime import datetime
import base64
from PIL import Image
import io
import hashlib

# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Genetic Health Care AI",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# ============================================
# QUANTUM SIMULATOR (Classical Simulation)
# ============================================
class QuantumSimulator:
    """Classical simulation of quantum-inspired algorithms for health analysis"""
    
    @staticmethod
    def simulate_qpca(input_features: list, target_dimensions: int = 4):
        """Quantum Principal Component Analysis simulation"""
        n = len(input_features)
        if n == 0:
            return {'reduced_features': [], 'eigenvalues': [], 'variance_explained': 0}
        
        # Create density matrix simulation
        features = np.array(input_features)
        features = (features - np.mean(features)) / (np.std(features) + 1e-10)
        
        # Simulate covariance matrix
        cov_matrix = np.outer(features, features) / n
        
        # Eigenvalue decomposition (quantum phase estimation simulation)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Reduce dimensions
        reduced = eigenvectors[:, :target_dimensions].T @ features
        variance_explained = np.sum(eigenvalues[:target_dimensions]) / (np.sum(eigenvalues) + 1e-10)
        
        return {
            'reduced_features': reduced.tolist(),
            'eigenvalues': eigenvalues[:target_dimensions].tolist(),
            'variance_explained': float(variance_explained)
        }
    
    @staticmethod
    def simulate_vqc(features: list, num_classes: int = 5):
        """Variational Quantum Classifier simulation"""
        if len(features) == 0:
            return {'probabilities': [1/num_classes]*num_classes, 'predicted_class': 0, 'confidence': 0.2}
        
        # Simulate parameterized quantum circuit
        features = np.array(features)
        
        # Create rotation angles from features
        theta = features * np.pi
        
        # Simulate quantum state evolution
        amplitudes = np.zeros(num_classes)
        for i in range(num_classes):
            phase = np.sum(theta * (i + 1)) / len(theta)
            amplitudes[i] = np.cos(phase) ** 2 + 0.1 * np.random.random()
        
        # Normalize to probabilities
        probabilities = np.abs(amplitudes) / (np.sum(np.abs(amplitudes)) + 1e-10)
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        return {
            'probabilities': probabilities.tolist(),
            'predicted_class': predicted_class,
            'confidence': confidence
        }
    
    @staticmethod
    def simulate_qnn(input_data: list, hidden_layers: int = 2):
        """Quantum Neural Network simulation"""
        if len(input_data) == 0:
            return {'output_features': [], 'correlations': 0}
        
        data = np.array(input_data)
        layer_outputs = [data]
        
        for layer in range(hidden_layers):
            # Simulate quantum layer with entanglement
            weights = np.random.randn(len(data), len(data)) * 0.5
            transformed = np.tanh(weights @ data)
            
            # Add quantum interference pattern
            interference = np.sin(transformed * np.pi)
            data = (transformed + interference) / 2
            layer_outputs.append(data)
        
        # Calculate quantum correlations
        correlations = float(np.mean([np.corrcoef(layer_outputs[i], layer_outputs[i+1])[0,1] 
                                      for i in range(len(layer_outputs)-1) if len(layer_outputs[i]) > 1]))
        
        return {
            'output_features': data.tolist(),
            'correlations': correlations if not np.isnan(correlations) else 0.5
        }
    
    @staticmethod
    def simulate_vqe(problem_size: int = 8, max_iterations: int = 50):
        """Variational Quantum Eigensolver simulation"""
        # Initialize random parameters
        params = np.random.randn(problem_size) * 0.1
        
        # Simulate Hamiltonian
        hamiltonian = np.random.randn(problem_size, problem_size)
        hamiltonian = (hamiltonian + hamiltonian.T) / 2  # Make symmetric
        
        convergence = []
        for iteration in range(max_iterations):
            # Evaluate energy expectation
            state = np.cos(params)
            energy = float(state @ hamiltonian @ state)
            convergence.append(energy)
            
            # Gradient descent update
            gradient = 2 * hamiltonian @ state * (-np.sin(params))
            params -= 0.1 * gradient
        
        return {
            'optimal_params': params.tolist(),
            'ground_state_energy': float(convergence[-1]),
            'convergence': convergence
        }
    
    @staticmethod
    def run_full_analysis(input_text: str):
        """Run complete quantum-inspired analysis pipeline"""
        # Convert text to numerical features
        text_hash = hashlib.md5(input_text.encode()).hexdigest()
        features = [int(c, 16) / 15.0 for c in text_hash[:16]]
        
        # Run quantum algorithms
        qpca_result = QuantumSimulator.simulate_qpca(features)
        vqc_result = QuantumSimulator.simulate_vqc(qpca_result['reduced_features'])
        qnn_result = QuantumSimulator.simulate_qnn(features)
        vqe_result = QuantumSimulator.simulate_vqe()
        
        # Calculate metrics
        confidence = vqc_result['confidence']
        entropy = -np.sum([p * np.log(p + 1e-10) for p in vqc_result['probabilities']])
        uncertainty = 1 - confidence
        
        return {
            'confidence_score': confidence,
            'quantum_entropy': float(entropy),
            'uncertainty_estimate': uncertainty,
            'eigenvalues': qpca_result['eigenvalues'],
            'classification_probabilities': vqc_result['probabilities'],
            'principal_components': qpca_result['reduced_features'],
            'quantum_correlations': qnn_result['correlations']
        }

# ============================================
# AI HEALTH ANALYSIS
# ============================================
def analyze_health(input_text: str, language: str, api_key: str, image_data: str = None):
    """Perform AI-powered health analysis"""
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    system_prompt = f"""You are an advanced AI health analysis system. Analyze the provided health information and respond in {language}.

IMPORTANT: You must respond with ONLY valid JSON, no markdown, no extra text.

Required JSON structure:
{{
    "diseaseName": "Primary condition identified",
    "symptoms": ["symptom1", "symptom2", "symptom3"],
    "analysisSummary": ["Key finding 1", "Key finding 2", "Key finding 3"],
    "riskPercentage": 0-100,
    "riskDetails": {{
        "severity": "low/moderate/high/critical",
        "factors": ["risk factor 1", "risk factor 2"],
        "explanation": "Detailed explanation"
    }},
    "foodSuggestions": ["food1", "food2", "food3", "food4"],
    "doctorConsultationNeeded": true/false,
    "recommendedDoctorType": "Specialist type if needed"
}}

Guidelines:
- Always recommend consulting healthcare professionals for serious concerns
- Provide evidence-based suggestions
- Be empathetic and clear
- Include appropriate medical disclaimers"""

    try:
        if image_data:
            # Multimodal analysis with image
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            response = model.generate_content([
                system_prompt,
                f"Health concern: {input_text}",
                image
            ])
        else:
            # Text-only analysis
            response = model.generate_content([
                system_prompt,
                f"Health concern: {input_text}"
            ])
        
        # Parse response
        response_text = response.text
        
        # Clean up response if wrapped in markdown
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]
        
        result = json.loads(response_text.strip())
        return result
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

# ============================================
# HELP CHATBOT
# ============================================
def get_chatbot_response(message: str, history: list, api_key: str):
    """Get response from help chatbot"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    system_prompt = """You are a helpful AI assistant for a health analysis application. Your role is to:
1. Help users understand how to use the health analysis features
2. Answer general health-related questions (with appropriate disclaimers)
3. Guide users through uploading files, using voice input, and camera capture
4. Explain analysis results in simple terms
5. Provide general wellness tips

Important: Always recommend consulting healthcare professionals for serious concerns."""

    # Build conversation
    conversation = system_prompt + "\n\n"
    for msg in history:
        role = "User" if msg['role'] == 'user' else "Assistant"
        conversation += f"{role}: {msg['content']}\n"
    conversation += f"User: {message}\nAssistant:"
    
    try:
        response = model.generate_content(conversation)
        return response.text
    except Exception as e:
        return f"I apologize, I encountered an error: {str(e)}"

# ============================================
# AUTHENTICATION (Simplified)
# ============================================
def auth_page():
    """Simple authentication page"""
    st.title("🧬 Genetic Health Care AI")
    st.subheader("Predictive Health Analysis")
    
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
    with tab1:
        with st.form("signin_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")
            
            if submitted and email and password:
                # Simple validation (replace with actual auth)
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.rerun()
    
    with tab2:
        with st.form("signup_form"):
            new_email = st.text_input("Email", key="signup_email")
            full_name = st.text_input("Full Name")
            phone = st.text_input("Phone Number")
            new_password = st.text_input("Password", type="password", key="signup_pass")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Account")
            
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords don't match!")
                elif new_email and new_password:
                    st.session_state.authenticated = True
                    st.session_state.user_email = new_email
                    st.success("Account created successfully!")
                    st.rerun()

# ============================================
# MAIN DASHBOARD
# ============================================
def dashboard():
    """Main dashboard with health analysis"""
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🧬 Genetic Health Care AI")
        st.caption("Predictive Health Analysis")
    with col2:
        if st.button("🚪 Sign Out"):
            st.session_state.authenticated = False
            st.session_state.user_email = None
            st.rerun()
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("Google AI API Key", type="password", 
                                help="Get your API key from Google AI Studio")
        
        language = st.selectbox("Analysis Language", [
            "English", "Telugu", "Hindi", "Tamil", "Kannada", 
            "Malayalam", "Bengali", "Marathi", "Gujarati", "Spanish"
        ])
        
        st.divider()
        
        # Health History
        st.header("📋 Health History")
        if st.session_state.analysis_history:
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
                with st.expander(f"{analysis['diseaseName']} - {analysis['date']}"):
                    st.write(f"Risk: {analysis['riskPercentage']}%")
                    if st.button(f"View Details", key=f"history_{i}"):
                        st.session_state.current_analysis = analysis
        else:
            st.info("No analysis history yet")
    
    # Main content
    if st.session_state.current_analysis:
        display_results(st.session_state.current_analysis)
        if st.button("← New Analysis"):
            st.session_state.current_analysis = None
            st.rerun()
    else:
        analysis_form(api_key, language)
    
    # Help Chatbot
    display_chatbot(api_key)

def analysis_form(api_key: str, language: str):
    """Health analysis input form"""
    
    st.header("📝 Describe Your Health Concern")
    
    # Input methods
    tab1, tab2, tab3 = st.tabs(["✏️ Text Input", "📷 Image Upload", "🎙️ Voice Input"])
    
    with tab1:
        input_text = st.text_area(
            "Describe your symptoms or health concern",
            height=150,
            placeholder="Example: I've been experiencing headaches and fatigue for the past week..."
        )
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload medical reports, X-rays, or photos",
            type=['png', 'jpg', 'jpeg', 'pdf']
        )
        image_data = None
        if uploaded_file:
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=300)
                
                # Convert to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                image_data = base64.b64encode(buffered.getvalue()).decode()
    
    with tab3:
        st.info("🎙️ Voice input requires browser microphone access. Use the text input for now.")
        voice_text = st.text_input("Or type what you would say:")
        if voice_text:
            input_text = voice_text
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔬 Analyze Health", use_container_width=True, type="primary")
    
    if analyze_button:
        if not api_key:
            st.error("Please enter your Google AI API Key in the sidebar")
            return
        if not input_text:
            st.error("Please describe your health concern")
            return
        
        with st.spinner("🧬 Running Quantum-AI Analysis..."):
            # Show quantum processing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Quantum simulation
            status_text.text("⚛️ Running Quantum PCA...")
            progress_bar.progress(20)
            
            quantum_results = QuantumSimulator.run_full_analysis(input_text)
            
            status_text.text("🧠 Running Variational Quantum Classifier...")
            progress_bar.progress(40)
            
            status_text.text("🔗 Running Quantum Neural Network...")
            progress_bar.progress(60)
            
            status_text.text("⚡ Running Variational Quantum Eigensolver...")
            progress_bar.progress(80)
            
            # AI Analysis
            status_text.text("🤖 Performing AI Health Analysis...")
            analysis = analyze_health(input_text, language, api_key, image_data)
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            if analysis:
                # Add quantum metrics and metadata
                analysis['quantumMetrics'] = quantum_results
                analysis['date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                analysis['input_text'] = input_text
                
                # Save to history
                st.session_state.analysis_history.append(analysis)
                st.session_state.current_analysis = analysis
                st.rerun()

def display_results(data: dict):
    """Display analysis results"""
    
    st.header(f"🏥 Analysis Results: {data.get('diseaseName', 'Unknown')}")
    
    # Risk Overview
    risk = data.get('riskPercentage', 0)
    risk_color = "🟢" if risk < 30 else "🟡" if risk < 60 else "🔴"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Risk Level", f"{risk}%", delta=None)
    with col2:
        st.metric("Severity", data.get('riskDetails', {}).get('severity', 'Unknown').title())
    with col3:
        consultation = "Yes ⚠️" if data.get('doctorConsultationNeeded') else "No ✅"
        st.metric("Doctor Needed", consultation)
    
    st.divider()
    
    # Symptoms
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🩺 Symptoms Identified")
        for symptom in data.get('symptoms', []):
            st.write(f"• {symptom}")
        
        st.subheader("📊 Analysis Summary")
        for point in data.get('analysisSummary', []):
            st.info(point)
    
    with col2:
        st.subheader("🍎 Food Recommendations")
        for food in data.get('foodSuggestions', []):
            st.success(f"✓ {food}")
        
        if data.get('doctorConsultationNeeded'):
            st.subheader("👨‍⚕️ Doctor Recommendation")
            st.warning(f"Recommended specialist: **{data.get('recommendedDoctorType', 'General Physician')}**")
    
    st.divider()
    
    # Risk Visualization
    st.subheader("📈 Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart simulation with metrics
        st.write("**Risk Distribution**")
        risk_data = {
            "At Risk": risk,
            "Healthy": 100 - risk
        }
        
        import plotly.express as px
        fig = px.pie(values=list(risk_data.values()), names=list(risk_data.keys()),
                     color_discrete_sequence=['#ef4444', '#22c55e'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk factors
        st.write("**Risk Factors**")
        factors = data.get('riskDetails', {}).get('factors', [])
        for i, factor in enumerate(factors):
            st.progress((len(factors) - i) / len(factors) * 0.8)
            st.caption(factor)
    
    # Quantum Metrics (if available)
    if 'quantumMetrics' in data:
        st.divider()
        st.subheader("⚛️ Quantum Analysis Metrics")
        
        qm = data['quantumMetrics']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Confidence Score", f"{qm.get('confidence_score', 0)*100:.1f}%")
        with col2:
            st.metric("Quantum Entropy", f"{qm.get('quantum_entropy', 0):.3f}")
        with col3:
            st.metric("Uncertainty", f"{qm.get('uncertainty_estimate', 0)*100:.1f}%")
        with col4:
            st.metric("Q-Correlations", f"{qm.get('quantum_correlations', 0):.3f}")
    
    # Explanation
    st.divider()
    st.subheader("📝 Detailed Explanation")
    st.write(data.get('riskDetails', {}).get('explanation', 'No detailed explanation available.'))
    
    # Disclaimer
    st.divider()
    st.warning("""
    ⚠️ **Medical Disclaimer**: This analysis is for informational purposes only and should not replace 
    professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare 
    provider for any health concerns.
    """)

def display_chatbot(api_key: str):
    """Display help chatbot"""
    
    with st.expander("💬 Need Help? Chat with AI Assistant"):
        # Chat messages
        for msg in st.session_state.chat_messages:
            if msg['role'] == 'user':
                st.chat_message("user").write(msg['content'])
            else:
                st.chat_message("assistant").write(msg['content'])
        
        # Chat input
        user_input = st.chat_input("Ask a question about the app or health...")
        
        if user_input and api_key:
            st.session_state.chat_messages.append({'role': 'user', 'content': user_input})
            
            with st.spinner("Thinking..."):
                response = get_chatbot_response(user_input, st.session_state.chat_messages, api_key)
                st.session_state.chat_messages.append({'role': 'assistant', 'content': response})
                st.rerun()
        elif user_input and not api_key:
            st.error("Please add your API key in the sidebar first")

# ============================================
# MAIN APP
# ============================================
def main():
    """Main application entry point"""
    
    # Check if plotly is available, if not show instructions
    try:
        import plotly.express as px
    except ImportError:
        st.error("Please install plotly: `pip install plotly`")
        return
    
    if st.session_state.authenticated:
        dashboard()
    else:
        auth_page()

if __name__ == "__main__":
    main()
