import streamlit as st
import google.generativeai as genai
from PIL import Image
import os

# ======================
# CONFIG & API SETUP
# ======================
# Get your API key from: https://aistudio.google.com/
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="Secure AI FaceID", layout="centered")
st.title("🛡️ Secure FaceID (Anti-Spoofing AI)")

KNOWN_FACES_DIR = "known_faces"

def get_known_faces_context():
    """Loads known identities into the AI's short-term memory."""
    context = []
    if not os.path.exists(KNOWN_FACES_DIR):
        return context
    for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
        p_path = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(p_path):
            for img_file in os.listdir(p_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(os.path.join(p_path, img_file))
                    context.append(f"Known Identity: {person_name}")
                    context.append(img)
                    break 
    return context

# ======================
# UI SECTION
# ======================
uploaded_file = st.file_uploader("Scan Face", type=["jpg", "png", "jpeg"])

if uploaded_file:
    test_img = Image.open(uploaded_file)
    st.image(test_img, caption="Captured Frame", width=400)
    
    if st.button("Authenticate"):
        with st.spinner("Analyzing liveness and identity..."):
            known_context = get_known_faces_context()
            
            # The "Anti-Spoofing" Prompt
            prompt = [
                "SYSTEM INSTRUCTION: You are a high-security biometric sensor. Your first priority is ANTI-SPOOFING.",
                *known_context,
                "TASK: Analyze the image below for authenticity and identity.",
                test_img,
                """
                ANALYSIS RULES:
                1. LIVENESS CHECK: Is this a real human in a 3D environment? 
                   - Look for 'photo-of-a-photo' signs: screen edges, glare on a phone screen, paper edges, pixelation, or moiré patterns. 
                   - Look for 'non-human' signs: statues, drawings, or digital avatars (e.g., Hanuman ji).
                   - If spoofing or non-human is detected, return 'SECURITY_ALERT: [REASON]'.
                
                2. IDENTITY CHECK: If and ONLY IF liveness is confirmed, compare this face to the 'Known Identities' above.
                   - If it matches a known user with high confidence, return 'MATCH: [Name]'.
                   - If it is a real human but doesn't match, return 'UNAUTHORIZED_USER'.
                
                RESPONSE FORMAT: Return only the final status code.
                """
            ]

            try:
                response = model.generate_content(prompt)
                result = response.text.strip()
                
                st.divider()
                if "MATCH:" in result:
                    st.success(f"✅ Access Granted: {result.split(':')[-1]}")
                elif "SECURITY_ALERT" in result:
                    st.error(f"🚨 {result}")
                    st.warning("Detection logic: AI found artifacts suggesting a spoofing attempt.")
                elif "UNAUTHORIZED" in result:
                    st.warning("👤 Human detected, but not in our database.")
                else:
                    st.info(f"Status: {result}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

st.caption("Secured by Gemini 1.5 Flash – Multimodal Liveness Detection")
