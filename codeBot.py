import logging
import streamlit as st
import torch
import os
import base64
from langchain.llms import HuggingFaceEndpoint
from fpdf import FPDF

# Force PyTorch to use CPU
device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = "API_TOKEN"  # Replace with your actual Hugging Face API token

st.set_page_config(page_title="🚀 AI-Powered Code Analyzer", page_icon="🤖")

# ----------------- AI MODEL LOADER -----------------
def load_ai_model():
    return HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        temperature=0.01,
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )

# ----------------- GENERATE REPORTS -----------------
def generate_pdf(text, filename="Code_Analysis.pdf"):
    pdf_path = os.path.join("C:\\temp", filename)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Code Analysis Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)
        pdf.ln(2)

    pdf.output(pdf_path)
    return pdf_path

# ----------------- STREAMLIT UI -----------------
st.title("🚀 AI-Powered Code Analyzer")

st.sidebar.header("Upload Code File")
uploaded_file = st.sidebar.file_uploader("Upload a programming file", type=["py", "js", "java", "cpp", "c", "rb", "php", "html", "css"])

if uploaded_file:
    # Save uploaded file to a temporary location
    uploaded_file_path = os.path.join("C:\\temp", uploaded_file.name)
    with open(uploaded_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File saved at: {uploaded_file_path}")

    # Read the uploaded file contents
    with open(uploaded_file_path, "r", encoding="utf-8") as file:
        file_contents = file.read()

    # Show the contents of the uploaded file
    st.subheader("📄 File Contents")
    st.code(file_contents, language=uploaded_file.name.split('.')[-1])

    # Ask for AI-powered analysis of the uploaded code
    if st.button("Analyze Code"):
        with st.spinner("Analyzing code... 🤖"):
            llm = load_ai_model()
            prompt = f"""
            Perform an analysis on the following programming code:

            **Code**:
            {file_contents}
            """
            response = llm.invoke(prompt)
            st.subheader("🤖 AI-Generated Analysis")
            st.write(response)

            # Generate Reports
            pdf_filename = generate_pdf(response)

            # PDF Download
            with open(pdf_filename, "rb") as f:
                pdf_data = f.read()

            b64_pdf = base64.b64encode(pdf_data).decode()
            href_pdf = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{pdf_filename}">📥 Download PDF Report</a>'
            st.markdown(href_pdf, unsafe_allow_html=True)

# ----------------- AI CHAT FEATURE -----------------
st.subheader("💬 Ask About Your Code")
user_question = st.text_input("Enter your question about the uploaded code:")

if user_question:
    with st.spinner("Thinking... 🤖"):
        llm = load_ai_model()
        prompt = f"""
        Answer the following question based on the uploaded code:

        **Question**: {user_question}

        **Code**: {file_contents}
        """
        response = llm.invoke(prompt)
        st.subheader("🤖 AI Response")
        st.write(response)

st.sidebar.markdown("### About")
st.sidebar.write("This AI tool analyzes code files and helps answer questions about your code.")
