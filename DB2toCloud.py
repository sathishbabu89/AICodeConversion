import logging
import streamlit as st
import re
import torch
import zipfile
import io
import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint

# Force PyTorch to use CPU
device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = "api_token"  # Add your Hugging Face API token

if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

st.set_page_config(page_title="DB2 to Google BigQuery Conversion Tool", page_icon="💻")

# ----------------- SIDEBAR MENU -----------------
page = st.sidebar.selectbox("Choose Page", [
    "File Upload Converter", 
    "Inline Code Converter", 
    "Automated DB2 Code Documentation"
])

# ----------------- AI MODEL LOADER -----------------
def load_ai_model():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
        max_new_tokens=2048,
        top_k=10,
        top_p=0.95,
        temperature=0.01,
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )

def convert_db2_to_bigquery(db2_code, filename, HUGGINGFACE_API_TOKEN):
    try:
        progress_bar = st.progress(0)
        progress_stage = 0

        with st.spinner("Processing and converting code..."):
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 1: Splitting the code into chunks... 💬")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(db2_code)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 2: Generating embeddings... 📊")

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embeddings)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 3: Loading the language model... 🚀")

            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                max_new_tokens=2048,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
                huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
            )

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 4: Converting DB2 to Google BigQuery... 🔄")

            prompt = f"""
Convert the following DB2 SQL code into Google BigQuery SQL.
Ensure that the generated code follows best practices, is optimized for BigQuery, and supports scalability.

### Key Requirements:
1. **Schema Conversion:**
   - Convert DB2 data types to BigQuery data types.
   - Ensure that the schema is compatible with BigQuery.

2. **SQL Syntax Conversion:**
   - Convert DB2-specific SQL syntax to BigQuery SQL syntax.
   - Ensure that the SQL queries are optimized for BigQuery.

3. **Best Practices:**
   - Use BigQuery best practices for query optimization.
   - Ensure that the converted code is modular and maintainable.

### DB2 Code Input:
{db2_code}
"""

            response = llm.invoke(prompt)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 5: Conversion complete! 🎉")

            # Display the converted BigQuery code
            st.code(response, language='sql')

            if re.search(r'\berror\b|\bexception\b|\bsyntax\b|\bmissing\b', response.lower()):
                st.warning("The converted BigQuery code may contain syntax or structural errors. Please review it carefully.")
            else:
                st.success("The BigQuery code is free from basic syntax errors!")

    except Exception as e:
        logger.error(f"An error occurred while converting the code: {e}", exc_info=True)
        st.error("Unable to convert DB2 code to BigQuery.")

# Page 1: File Upload Converter
if page == "File Upload Converter":
    st.header("DB2 to Google BigQuery Conversion Tool with LLM 💻")

    with st.sidebar:
        with st.expander("Upload Your DB2 Code", expanded=True):
            file = st.file_uploader("Upload a DB2 SQL file (.sql) to start analyzing", type="sql")

            if file is not None:
                try:
                    code_content = file.read().decode("utf-8")
                    st.subheader("DB2 Code Preview")
                    st.code(code_content[:5000], language='sql')
                except Exception as e:
                    logger.error(f"An error occurred while reading the code file: {e}", exc_info=True)
                    st.warning("Unable to display code preview.")

    with st.expander("Tutorials & Tips", expanded=True):
        st.write("""### Welcome to the DB2 to Google BigQuery Conversion Tool!
        Here are some tips to help you use this tool effectively:
        - **Code Formatting:** Ensure your DB2 SQL code is properly formatted.
        - **Chunking:** Break large files into smaller parts.
        - **Testing:** Test the BigQuery code after conversion.
        - **Documentation:** Familiarize with DB2 and BigQuery docs.
        """)

    if file is not None:
        if st.button("Convert DB2 to BigQuery"):
            convert_db2_to_bigquery(code_content, file.name, HUGGINGFACE_API_TOKEN)

# Page 2: Inline Code Converter
if page == "Inline Code Converter":
    st.header("Inline DB2 to BigQuery Code Converter 💻")

    db2_code_input = st.text_area("Enter DB2 SQL Code to Convert to BigQuery", height=300)

    if st.button("Convert Inline DB2 to BigQuery"):
        if db2_code_input.strip():
            convert_db2_to_bigquery(db2_code_input, "inline_code_conversion.sql", HUGGINGFACE_API_TOKEN)
        else:
            st.warning("Please enter some DB2 SQL code to convert.")

if page == "Automated DB2 Code Documentation":
    st.header("📄 Automated DB2 Code Documentation")

    file = st.file_uploader("Upload a DB2 SQL file (.sql) to analyze", type="sql")

    if file:
        db2_code = file.read().decode("utf-8")
        st.subheader("DB2 Code Preview")
        st.code(db2_code[:2000], language="sql")  # Show first 2000 characters

        if st.button("Generate Documentation"):
            with st.spinner("Analyzing DB2 Code... 🔍"):
                llm = load_ai_model()

                # AI Prompt for DB2 Analysis
                prompt = f"""
                Analyze the following DB2 SQL code and generate a simple structured documentation.
                Focus on describing the schema, key queries, and data usage.
                
                ### DB2 Code:
                {db2_code}
                """
                response = llm.invoke(prompt)

                # ----------------- DISPLAY DOCUMENTATION -----------------
                st.success("Analysis Complete! 📄")
                st.subheader("📜 AI-Generated DB2 Documentation")
                st.write(response)

# Footer
st.sidebar.markdown("### About")
st.sidebar.write("This tool uses state-of-the-art AI models to assist with DB2 to BigQuery conversion.")
