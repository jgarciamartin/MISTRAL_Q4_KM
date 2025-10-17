# app.py
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from config import *

# Configuración de página
st.set_page_config(page_title="Asistente RAG con Mistral", layout="wide")

st.title("💬 Asistente RAG - Ollama + Mistral")
st.caption("Haz preguntas sobre tus documentos locales (PDF, Word, PowerPoint)")

# Cargar la base vectorial
@st.cache_resource
def load_qa_chain():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model=MODEL_NAME)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa

qa = load_qa_chain()

# Interfaz de chat
query = st.text_input("🔎 Escribe tu pregunta:")

if query:
    with st.spinner("Pensando..."):
        result = qa(query)
        st.write("### 🧠 Respuesta:")
        st.write(result["result"])

        with st.expander("📄 Fuentes utilizadas"):
            for doc in result["source_documents"]:
                st.markdown(f"- **{doc.metadata.get('source', 'Desconocido')}**")

