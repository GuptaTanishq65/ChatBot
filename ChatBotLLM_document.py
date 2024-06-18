from langchain_community.llms import Ollama
import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Instantiate Ollama with the llama3 model
llm = Ollama(model="llama3", base_url="http://localhost:11434", verbose=True)

# Document context
document_context = """
                    We will build 2 containers, Ollama container will be using the host volume to store and load the models (/root/.ollama is mapped to the local ./data/ollama). Ollama container will listen on 11434 (external port, which is internally mapped to 11434)
                        Streamlit chatbot application will listen on 8501 (external port, which is internally mapped to 8501).
                    """

def sendPrompt(prompt, context=None):
    global llm
    if context:
        prompt = f"{context}\n{prompt}"
    response = llm.invoke(prompt)
    return response

st.title("Chat with Ollama")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question!"}
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]): 
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = sendPrompt(prompt, context=document_context)
            print(response)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)