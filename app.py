import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURACIÓN UI ---
st.set_page_config(page_title="T&C Auditor (LCEL)", page_icon="⚖️", layout="wide")

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .audit-card {
        padding: 1rem; border-radius: 8px; margin-bottom: 1rem;
        border-left: 4px solid; background-color: #fdfdfd; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .audit-info { border-color: #2196f3; }
    .audit-warning { border-color: #ff9800; }
    .audit-success { border-color: #4caf50; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- ESTADO ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "audit_results" not in st.session_state: st.session_state.audit_results = {}

# --- FUNCIONES DE AYUDA LCEL ---
def format_docs(docs):
    """Función auxiliar para unir los documentos recuperados en un solo string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_llm(hf_token):
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.01,
        huggingfacehub_api_token=hf_token
    )

def process_file(uploaded_file, hf_token):
    if not hf_token: return None
    
    with st.status("⚙️ Indexando documento...", expanded=True) as status:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path) if uploaded_file.name.endswith('.pdf') else TextLoader(tmp_path)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(docs)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(splits, embeddings)
            
            os.remove(tmp_path)
            status.update(label="✅ Listo", state="complete", expanded=False)
            return vector_store
        except Exception as e:
            st.error(f"Error: {e}")
            return None

def run_audit(vector_store, hf_token):
    """Auditoría usando LCEL puro (Sin RetrievalQA)."""
    llm = get_llm(hf_token)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # 1. Definir Prompt
    template = """[INST] Analiza el siguiente contexto legal y responde la pregunta.
    Contexto: {context}
    Pregunta: {question} [/INST]"""
    prompt = PromptTemplate.from_template(template)
    
    # 2. Construir la Cadena (LCEL Pipe)
    # Retriever -> Formatear -> Prompt -> LLM -> Parser Texto
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    audit_topics = {
        "privacy": {"q": "Resume políticas de privacidad y GDPR.", "icon": "🔒", "type": "info"},
        "jurisdiction": {"q": "¿Cuál es la jurisdicción y ley aplicable?", "icon": "⚖️", "type": "warning"},
        "termination": {"q": "¿Cómo se puede terminar el contrato?", "icon": "🚫", "type": "success"}
    }
    
    results = {}
    bar = st.progress(0)
    
    for i, (key, val) in enumerate(audit_topics.items()):
        try:
            # .invoke() es el método estándar de LCEL
            response = rag_chain.invoke(val["q"])
            results[key] = {"content": response, "icon": val["icon"], "type": val["type"]}
        except Exception as e:
            results[key] = {"content": "Error en la consulta.", "icon": "⚠️", "type": "warning"}
        bar.progress((i + 1) / 3)
    
    bar.empty()
    return results

# --- MAIN ---
def main():
    st.title("🛡️ Auditor Legal (LCEL)")
    
    with st.sidebar:
        hf_token = st.text_input("HF Token", type="password")
        uploaded_file = st.file_uploader("Archivo", type=["pdf", "txt"])
        if st.button("Procesar") and uploaded_file:
            st.session_state.vector_store = process_file(uploaded_file, hf_token)
            if st.session_state.vector_store:
                st.session_state.audit_results = run_audit(st.session_state.vector_store, hf_token)

    tab1, tab2 = st.tabs(["Resultados", "Chat"])
    
    with tab1:
        if st.session_state.audit_results:
            cols = st.columns(3)
            idx = 0
            for key, res in st.session_state.audit_results.items():
                with cols[idx]:
                    st.markdown(f"""
                    <div class="audit-card audit-{res['type']}">
                        <h4>{res['icon']} {key.capitalize()}</h4>
                        <small>{res['content']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                idx = (idx + 1) % 3

    with tab2:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])
            
        if prompt_text := st.chat_input("Pregunta..."):
            if st.session_state.vector_store:
                st.session_state.messages.append({"role": "user", "content": prompt_text})
                st.chat_message("user").markdown(prompt_text)
                
                # Cadena para Chat (LCEL)
                llm = get_llm(hf_token)
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                template = "[INST] Contexto: {context}. Pregunta: {question} [/INST]"
                prompt_obj = PromptTemplate.from_template(template)
                
                chat_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_obj
                    | llm
                    | StrOutputParser()
                )
                
                with st.spinner("Pensando..."):
                    resp = chat_chain.invoke(prompt_text)
                    st.chat_message("assistant").markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})

if __name__ == "__main__":
    main()