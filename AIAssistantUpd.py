import logging
import streamlit as st
import torch
import zipfile
import os
import shutil
import base64
import networkx as nx
import matplotlib.pyplot as plt
from langchain.llms import HuggingFaceEndpoint
from fpdf import FPDF

# Force PyTorch to use CPU
device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = "API_TOKEN"  # Replace with your actual Hugging Face API token

st.set_page_config(page_title="🚀 AI-Powered Java Spring Boot Analyzer", page_icon="🤖")

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

# ----------------- ZIP FILE HANDLER -----------------
UPLOAD_DIR = "C:\\temp"

def extract_zip(zip_path):
    extracted_path = os.path.join(UPLOAD_DIR, "extracted_project")
    
    # Ensure the directory exists
    os.makedirs(extracted_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

    return extracted_path

# ----------------- ANALYZE SPRING BOOT PROJECT -----------------
def analyze_spring_boot_project(project_dir):
    """
    Extracts and analyzes key components of the Spring Boot project.
    """
    project_summary = {"controllers": [], "services": [], "repositories": []}

    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "@RestController" in content:
                        project_summary["controllers"].append(file_path)
                    elif "@Service" in content:
                        project_summary["services"].append(file_path)
                    elif "@Repository" in content:
                        project_summary["repositories"].append(file_path)

    return project_summary

# ----------------- AI-POWERED PROJECT ANALYSIS -----------------
def generate_ai_analysis(project_structure):
    try:
        with st.spinner("Analyzing Java Spring Boot Project... 🔍"):
            llm = load_ai_model()
            prompt = f"""
            Perform an AI-powered review of the following **Spring Boot microservices project**:
            
            1. **Project Overview**: Key components like Controllers, Services, Repositories.
            2. **API Endpoints**: List REST APIs with HTTP methods.
            3. **Database Interactions**: Identify repositories and JPA queries.
            4. **Best Practices Check**: Highlight security, performance, and scalability issues.
            5. **Dependency Graph**: Identify interdependencies between microservices.
            
            ### Project Structure:
            {project_structure}
            """
            response = llm.invoke(prompt)
            return response
    except Exception as e:
        logger.error(f"Error in AI analysis: {e}", exc_info=True)
        return "An error occurred while analyzing the project."

# ----------------- UML & DEPENDENCY GRAPH -----------------
def generate_dependency_graph(services):
    """
    Generates a simple dependency graph for Spring Boot microservices.
    """
    if not services:
        return None

    G = nx.DiGraph()
    for service in services:
        service_name = os.path.basename(service).replace(".java", "")
        G.add_node(service_name)
        with open(service, "r", encoding="utf-8") as f:
            content = f.read()
            for other_service in services:
                other_service_name = os.path.basename(other_service).replace(".java", "")
                if other_service_name in content:
                    G.add_edge(service_name, other_service_name)

    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", font_size=10)
    plt.title("Spring Boot Microservices Dependency Graph")
    graph_path = os.path.join(UPLOAD_DIR, "dependency_graph.png")
    plt.savefig(graph_path)
    return graph_path

# ----------------- GENERATE REPORTS -----------------
def generate_pdf(text, filename="SpringBoot_Analysis.pdf"):
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Spring Boot Project Analysis", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)
        pdf.ln(2)

    pdf.output(pdf_path)
    return pdf_path

# ----------------- STREAMLIT UI -----------------
st.title("🚀 AI-Powered Java Spring Boot Analyzer")

st.sidebar.header("Upload Spring Boot Project (ZIP)")
uploaded_file = st.sidebar.file_uploader("Upload a ZIP file", type="zip")

if uploaded_file:
    zip_path = os.path.join(UPLOAD_DIR, "uploaded_project.zip")
    
    # Ensure C:\temp exists before writing
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Save uploaded file
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Check if file exists before extraction
    if os.path.exists(zip_path):
        st.success(f"ZIP file saved at: {zip_path}")

        # Extract ZIP without deleting the parent directory
        extracted_path = extract_zip(zip_path)
        st.success(f"Project extracted to: {extracted_path}")

        # Analyze the extracted project
        project_structure = analyze_spring_boot_project(extracted_path)
        st.subheader("📂 Project Structure Analysis")
        st.json(project_structure)

        if st.button("Analyze Project"):
            ai_analysis = generate_ai_analysis(project_structure)
            st.subheader("📖 AI-Generated Analysis")
            st.text_area("AI Analysis", ai_analysis, height=400)

            # Generate Reports
            pdf_filename = generate_pdf(ai_analysis)

            # PDF Download
            with open(pdf_filename, "rb") as f:
                pdf_data = f.read()

            b64_pdf = base64.b64encode(pdf_data).decode()
            href_pdf = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{pdf_filename}">📥 Download PDF Report</a>'
            st.markdown(href_pdf, unsafe_allow_html=True)

            # Generate Dependency Graph
            graph_path = generate_dependency_graph(project_structure["services"])
            if graph_path:
                st.image(graph_path, caption="Dependency Graph")

# ----------------- AI CHAT FEATURE -----------------
st.subheader("💬 Ask About Your Project")
user_question = st.text_input("Enter your question about the uploaded project:")

if user_question:
    with st.spinner("Thinking... 🤖"):
        llm = load_ai_model()
        prompt = f"""
        Answer the following **Spring Boot microservices** question based on the uploaded project structure:
        
        **Question**: {user_question}
        """
        response = llm.invoke(prompt)
        st.subheader("🤖 AI Response")
        st.write(response)

st.sidebar.markdown("### About")
st.sidebar.write("This AI tool analyzes Java Spring Boot microservices projects for best practices and dependencies.")
