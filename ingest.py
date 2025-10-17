import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from config import *

def load_docs():
    docs = []
    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif filename.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def main():
    print("📚 Cargando documentos...")
    documents = load_docs()
    print(f"Documentos cargados: {len(documents)}")

    print("🔪 Dividiendo en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Fragmentos creados: {len(chunks)}")

    print("🧠 Generando embeddings y creando base vectorial...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_DB_DIR)
    db.persist()
    print("✅ Base vectorial creada y guardada en", VECTOR_DB_DIR)

if __name__ == "__main__":
    main()