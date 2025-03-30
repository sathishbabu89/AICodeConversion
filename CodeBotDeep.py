import logging
import streamlit as st
import torch
import os
from openai import OpenAI

# Force PyTorch to use CPU
device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = "Deepseek_API"  # Replace with your DeepSeek API key

st.set_page_config(page_title="🚀 AI-Powered Code Analyzer", page_icon="🤖")

# ----------------- AI CHAT FEATURE -----------------
def get_ai_response(user_question, code_context):
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
        # Send chat message with code context
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant, capable of understanding and providing insights on various programming languages."},
                {"role": "user", "content": f"Here is the code:\n{code_context}\n\nQuestion: {user_question}"},
            ],
            stream=False
        )
        
        # If the response is valid, return the message content
        if response:
            logger.info("DeepSeek API call successful.")
            return response.choices[0].message.content
        else:
            logger.error("DeepSeek API call failed. No response received.")
            return "An error occurred while processing your question."
    except Exception as e:
        logger.error(f"Error during AI chat API call: {e}", exc_info=True)
        return "An error occurred while processing your question."

# ----------------- STREAMLIT UI -----------------
st.title("🚀 AI-Powered Code Analyzer")

# Sidebar for file upload
st.sidebar.header("Upload Your Code Files")
uploaded_file = st.sidebar.file_uploader("Upload a code file (Java, Python, JavaScript, etc.)", type=["java", "py", "js", "cpp", "c"])

if uploaded_file:
    # Read the uploaded file
    file_contents = uploaded_file.read().decode("utf-8")

    st.sidebar.success("File uploaded successfully!")

    # Ask user for a question about the uploaded code
    user_question = st.text_input("Enter your question about the uploaded code:")

    if user_question:
        with st.spinner("Thinking... 🤖"):
            # Get AI response based on the uploaded code and the question
            ai_response = get_ai_response(user_question, file_contents)

            st.subheader("🤖 AI Response")
            st.write(ai_response)

# Information section
st.sidebar.markdown("### About")
st.sidebar.write("This AI tool provides answers based on the uploaded code in various programming languages such as Java, Python, JavaScript, and more.")
