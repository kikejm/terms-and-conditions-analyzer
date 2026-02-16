import streamlit as st
import os
import tempfile
from typing import Optional, Dict, Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.vectorstores import VectorStore

# --- CONSTANTES ---
PAGE_TITLE = "T&C Auditor"
PAGE_ICON = "⚖️"

MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

AUDIT_TOPICS = {
    "privacy": {"q": "Resume las políticas de privacidad, recolección de datos y GDPR. Sé conciso.", "icon": "🔒", "type": "info"},
    "jurisdiction": {"q": "¿Cuál es la jurisdicción, tribunales competentes y ley aplicable? Sé conciso.", "icon": "⚖️", "type": "warning"},
    "termination": {"q": "¿Condiciones para terminar el contrato y penalizaciones? Sé conciso.", "icon": "🚫", "type": "success"}
}

# CORRECCIÓN CSS: Se añade 'color: #31333F' para forzar texto oscuro sobre fondo blanco
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .audit-card {
        padding: 1.5rem; 
        border-radius: 8px; 
        margin-bottom: 1rem;
        border-left: 5px solid; 
        background-color: #ffffff; 
        color: #31333F !important; /* Texto oscuro obligatorio */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .audit-card h4 {
        margin-top: 0;
        color: #000000;
        font-weight: 600;
    }
    .audit-info { border-color: #2196f3; }
    .audit-warning { border-color: #ff9800; }
    .audit-success { border-color: #4caf50; }
</style>
"""

# --- CONFIGURACIÓN ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- LÓGICA ---

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)

@st.cache_resource
def get_chat_model(hf_token: str):
    if not hf_token:
        raise ValueError("Token no proporcionado.")
    
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_REPO_ID,
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        max_new_tokens=512,
        timeout=120,
    )
    
    return ChatHuggingFace(llm=llm)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vector_store: VectorStore, chat_model: Any, k: int = 3) -> Runnable:
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un abogado experto revisando contratos (T&C). Responde basándote ÚNICAMENTE en el contexto proporcionado en español. Si no se menciona, dilo explícitamente."),
        ("human", "Contexto:\n{context}\n\nPregunta: {question}")
    ])
    
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

def process_uploaded_file(uploaded_file) -> Optional[VectorStore]:
    if not uploaded_file: return None
    
    suffix = f".{uploaded_file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name 

    try:
        loader = PyPDFLoader(tmp_path) if uploaded_file.name.endswith('.pdf') else TextLoader(tmp_path, encoding='utf-8')
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = splitter.split_documents(docs)
        embeddings = get_embeddings_model()
        vector_store = FAISS.from_documents(splits, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error procesando archivo: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def perform_audit(vector_store: VectorStore, chat_model: Any) -> Dict[str, Any]:
    rag_chain = create_rag_chain(vector_store, chat_model, k=3)
    results = {}
    progress_bar = st.progress(0)
    
    for i, (key, config) in enumerate(AUDIT_TOPICS.items()):
        try:
            response = rag_chain.invoke(config["q"])
            results[key] = {
                "content": response, 
                "icon": config["icon"], 
                "type": config["type"],
                "title": key.capitalize()
            }
        except Exception as e:
            results[key] = {
                "content": f"Error técnico: {repr(e)}", 
                "icon": "⚠️", 
                "type": "warning",
                "title": key.capitalize()
            }
        progress_bar.progress((i + 1) / len(AUDIT_TOPICS))
    
    progress_bar.empty()
    return results

# --- UI ---

def main():
    if "messages" not in st.session_state: st.session_state.messages = []
    if "vector_store" not in st.session_state: st.session_state.vector_store = None
    if "audit_results" not in st.session_state: st.session_state.audit_results = {}

    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    
    with st.sidebar:
        st.header("Configuración")
        hf_token = st.text_input("HuggingFace Token", type="password")
        uploaded_file = st.file_uploader("Subir Contrato", type=["pdf", "txt"])
        
        if st.button("Auditar Documento", type="primary") and uploaded_file and hf_token:
            with st.spinner("⏳ Analizando..."):
                try:
                    _ = get_embeddings_model()
                    chat_model = get_chat_model(hf_token)
                    vs = process_uploaded_file(uploaded_file)
                    
                    if vs:
                        st.session_state.vector_store = vs
                        st.session_state.audit_results = perform_audit(vs, chat_model)
                        st.success("¡Análisis completado!")
                except Exception as e:
                    st.error(f"Error de conexión: {e}")

    tab1, tab2 = st.tabs(["📊 Resultados", "💬 Chat"])
    
    with tab1:
        if st.session_state.audit_results:
            # CORRECCIÓN LAYOUT: Eliminado st.columns(3) en favor de vista vertical completa
            for res in st.session_state.audit_results.values():
                st.markdown(f"""
                <div class="audit-card audit-{res['type']}">
                    <h4>{res['icon']} {res['title']}</h4>
                    <div>{res['content']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Sube un archivo para comenzar.")

    with tab2:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])
        
        if prompt := st.chat_input("Pregunta sobre el contrato..."):
            if not st.session_state.vector_store:
                st.error("Sube un archivo primero.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").markdown(prompt)
                try:
                    chat_model = get_chat_model(hf_token)
                    chain = create_rag_chain(st.session_state.vector_store, chat_model)
                    with st.spinner("Pensando..."):
                        resp = chain.invoke(prompt)
                        st.chat_message("assistant").markdown(resp)
                        st.session_state.messages.append({"role": "assistant", "content": resp})
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()