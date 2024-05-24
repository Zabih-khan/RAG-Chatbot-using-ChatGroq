import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import SeleniumURLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Check for API keys
if not groq_api_key or not os.environ["GOOGLE_API_KEY"]:
    st.error("API keys are not set. Please check your .env file.")
    st.stop()

# Sidebar with information and navigation
st.sidebar.image("https://www.xevensolutions.com/wp-content/uploads/2023/08/footer-logo.png", width=200)
st.sidebar.title("Navigation")
st.sidebar.markdown("**Xeven Solutions:** Innovating with cutting-edge digital solutions.")
st.sidebar.markdown("[Visit Xeven Solutions](https://www.xevensolutions.com/)")

# App title and description
st.title("🔍 Chatbot for Xeven Solution")
st.markdown("""
    Welcome to the custom chatbot app! This tool allows you to query and retrieves relevant information from specified URLs and provides accurate responses to user queries.
    First click on the button to create vector embeddings and interact with the chatbot.
""")

# Initialize the chatbot
llm = ChatGroq(api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.urls = ["https://www.xevensolutions.com/",
                                 "https://www.xevensolutions.com/about-us/",
                                 "https://www.xevensolutions.com/careers/",
                                 "https://www.xevensolutions.com/portfolio/"]
        st.session_state.loader = SeleniumURLLoader(urls=st.session_state.urls)
        st.session_state.data = st.session_state.loader.load()
        st.session_state.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.data[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Create embeddings button
if st.button("Create Embedding"):
    vector_embedding()
    st.success("Vector Store DB Is Ready")

# User query input
query = st.text_input("Enter Your Query", placeholder="Ask me anything about the website...")

# Handle query and display response
if query:
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': query})
    response_time = time.process_time() - start_time

    st.markdown(f"**Response time:** {response_time:.2f} seconds")
    st.markdown("### Answer")
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.markdown(f"**Document {i + 1}:**")
            st.write(doc.page_content)
            st.markdown("---")

# Add some custom styling
st.markdown(
    """
    <style>
    .css-18e3th9 {
        padding-top: 3rem;
    }
    .css-1d391kg {
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True
)
