import streamlit as st
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from PIL import Image, ImageEnhance
import pytesseract
import cv2
import numpy as np

# Instantiate Ollama with the llama3 model
llm = Ollama(model="llama3", base_url="http://localhost:11434", verbose=True)

def sendPrompt(prompt, context=None):
    global llm
    if context:
        prompt = f"{context}\n{prompt}"
    response = llm.invoke(prompt)
    return response

def preprocess_image(image):
    # Increase the contrast of the image
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Enhance contrast by a factor of 2

    # Convert PIL Image to OpenCV format
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Enhance the image (increase contrast and brightness)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    
    # Binarization
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Remove table lines
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, (0, 0, 0), 5)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, (0, 0, 0), 5)
    
    # Remove noise
    binary = cv2.fastNlMeansDenoising(binary, h=30)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
    return processed_image

st.title("Chat with Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload an image and ask me a question based on the text in the image!"}
    ]

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    with st.spinner("Extracting text from image..."):
        myconfig = r" --psm 4"
        processed_image = preprocess_image(image)
        extracted_text = pytesseract.image_to_string(processed_image, config=myconfig)
        st.text_area("Extracted Text", extracted_text, height=250)
        st.session_state.extracted_text = extracted_text

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = st.session_state.extracted_text
            response = sendPrompt(prompt, context=context)
            print(response)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
