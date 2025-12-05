import streamlit as st
import os
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile


# 1. Load Environment Variables
import streamlit as st
import os

# Try to load dotenv (works locally), skip if missing (works on cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # We are on the cloud, so we don't need this

api_key = os.getenv("GOOGLE_API_KEY")


# 2. Page Configuration
st.set_page_config(page_title="🤖 Chat with PDF (Gemini)", layout="wide")
st.title("🤖 Chat with Your PDF")
st.caption("Powered by Google Gemini & LangChain")

# 3. Sidebar for Upload
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    # Check if we need to process the file
    if uploaded_file and "vector_db" not in st.session_state:
        with st.spinner("Processing PDF... (This may take a minute)"):
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # A. Load PDF
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

                # B. Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # C. Create Embeddings (The "Search Engine" part)
                # We use 'models/text-embedding-004' which is efficient and free
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

                # D. Store in Vector Database (FAISS)
                vector_db = FAISS.from_documents(splits, embeddings)
                
                # Save to session state so we don't reload every time
                st.session_state.vector_db = vector_db
                st.success(f"✅ PDF Processed! ({len(splits)} chunks created)")
                
                # Cleanup
                os.remove(tmp_path)

            except Exception as e:
                st.error(f"Error processing PDF: {e}")

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Question
if prompt := st.chat_input("Ask a question about your PDF..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vector_db" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # A. Setup the Brain (Use the model that worked for you!)
                # If 'gemini-2.5-flash' fails, try 'gemini-1.5-flash'
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3,convert_system_message_to_human=True)

                # B. Create the Chain
                retriever = st.session_state.vector_db.as_retriever()
                
                system_prompt = (
                    "You are a helpful assistant. Use the provided context to answer the question. "
                    "If the answer is not in the context, say you don't know."
                    "\n\n"
                    "{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                # C. Get Answer
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning("⚠️ Please upload a PDF first!")