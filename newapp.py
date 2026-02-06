import streamlit as st
import google.generativeai as genai
from PIL import Image
import os

# ======================
# CONFIG & API SETUP
# ======================
genai.configure(api_key="YOUR_GEMINI_API_KEY") # Replace with your key
model = genai.GenerativeModel('gemini-1.5-flash') # Flash is faster/cheaper for this

st.set_page_config(page_title="Gemini Face ID", layout="centered")
st.title("🤖 Gemini AI Face Verification")

KNOWN_FACES_DIR = "known_faces"

# ======================
# HELPER: LOAD KNOWN DATA
# ======================
def get_known_faces_data():
    """Returns a list of PIL images + names for the prompt context"""
    context = []
    if not os.path.exists(KNOWN_FACES_DIR):
        return context
    
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(os.path.join(person_path, img_file))
                    context.append(f"This image is {person_name}:")
                    context.append(img)
                    break # Take one image per person
    return context

# ======================
# UI & UPLOAD
# ======================
uploaded_file = st.file_uploader("Upload image to verify", type=["jpg", "png", "jpeg"])

if uploaded_file:
    test_img = Image.open(uploaded_file)
    st.image(test_img, caption="Uploaded Image", width=300)
    
    if st.button("Verify with AI"):
        with st.spinner("Gemini is analyzing..."):
            # Prepare the prompt context
            known_data = get_known_faces_data()
            
            # The Multi-Modal Prompt
            prompt = [
                "You are a strict security system. I will provide images of known users first.",
                *known_data,
                "Now, analyze this last image (the test image).",
                test_img,
                """
                Check the following in order:
                1. Is there a clear human face in the test image? If it's an object, a statue, a drawing, or a deity, return: 'INVALID_PHOTO'.
                2. If it is a human, does the face match any of the known users provided above? 
                3. If it matches a user perfectly, return: 'MATCH: [Name]'.
                4. If it is a human but does not match any known user, return: 'MISMATCH'.
                
                Respond ONLY with the status code string.
                """
            ]

            try:
                response = model.generate_content(prompt)
                result = response.text.strip()
                
                st.divider()
                if "MATCH:" in result:
                    st.success(f"✅ {result}")
                elif "MISMATCH" in result:
                    st.warning("👤 Human detected, but no match found.")
                elif "INVALID_PHOTO" in result:
                    st.error("🚫 Invalid Photo: No real human face detected.")
                else:
                    st.info(f"AI Response: {result}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
