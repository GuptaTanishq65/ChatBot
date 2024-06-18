from langchain_community.llms import Ollama
import streamlit as st
import docx  # For parsing Word documents
import PyPDF2  # For parsing PDF documents

# Instantiate Ollama with the llama3 model
llm = Ollama(model="llama3", base_url="http://localhost:11434", verbose=True)

# Define a function to extract text from uploaded documents
def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text.strip()

def sendPrompt(prompt, context=None):
    global llm
    if context:
        prompt = f"{context}\n{prompt}"
    response = llm.invoke(prompt)
    return response

st.title("Chat with Ollama")

# Allow the user to upload a document
document_file = st.file_uploader("Upload document", type=["docx", "pdf"])

if document_file:
    document_context = extract_text(document_file)

    # Text box for user questions
    user_question = st.text_area("Enter your question here:")

    # Run button to send the question to Ollama
    if st.button("Run"):
        if user_question:
            with st.spinner("Thinking..."):
                response = sendPrompt(user_question, context=document_context)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
        else:
            st.warning("Please enter a question before running.")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question!"}]
