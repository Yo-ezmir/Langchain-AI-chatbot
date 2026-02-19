import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from loaders.document_loader import load_pdf
from processing.text_splitter import split_documents
from embeddings.embedding_model import get_embeddings
from vectorstore.chroma_store import create_vectorstore
from chains.conversational_chain import build_chain
from tools.web_search import search_web
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }

    /* ── Brand Header ── */
    .brand-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .brand-header .logo {
        font-size: 3rem;
        margin-bottom: 0.25rem;
    }
    .brand-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .brand-header p {
        color: #94a3b8;
        font-size: 0.8rem;
        margin: 0.25rem 0 0;
    }

    /* ── Sidebar Dividers ── */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
        margin: 1rem 0;
    }

    /* ── Stat Cards ── */
    .doc-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0.75rem;
    }
    .stat-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 10px;
        padding: 0.6rem 0.75rem;
        text-align: center;
    }
    .stat-card .stat-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #818cf8;
    }
    .stat-card .stat-label {
        font-size: 0.65rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Provider Badge ── */
    .provider-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        color: #c4b5fd;
        margin-top: 0.5rem;
    }

    /* ── Chat Area ── */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }

    /* ── Welcome Card ── */
    .welcome-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(192,132,252,0.06));
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 600px;
    }
    .welcome-card .icon {
        font-size: 3rem;
        margin-bottom: 0.75rem;
    }
    .welcome-card h3 {
        color: #e2e8f0;
        margin: 0 0 0.5rem;
    }
    .welcome-card p {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* ── Feature Pills ── */
    .features {
        display: flex;
        gap: 0.75rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1.25rem;
    }
    .feature-pill {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.75rem;
        color: #c4b5fd;
    }

    /* ── Success Alert ── */
    .custom-success {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.25);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        color: #86efac;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 3px;
    }

    /* ── Chat message styling ── */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }

    /* ── Web search result card ── */
    .web-results {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .web-results summary {
        color: #93c5fd;
        cursor: pointer;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ─── Provider Config ─────────────────────────────────────────────────────────
PROVIDERS = {
    "OpenAI": {
        "icon": "🟢",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"],
        "key_name": "OPENAI_API_KEY",
        "id": "openai",
    },
    "Google Gemini": {
        "icon": "🔵",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "key_name": "GOOGLE_API_KEY",
        "id": "google",
    },
    "Anthropic": {
        "icon": "🟠",
        "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
        "key_name": "ANTHROPIC_API_KEY",
        "id": "anthropic",
    },
}


# ─── Session State Init ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_info" not in st.session_state:
    st.session_state.doc_info = None


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-header">
        <div class="logo">🧠</div>
        <h1>DocMind AI</h1>
        <p>Intelligent Document Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Provider Selection
    st.markdown("### 🤖 LLM Provider")
    selected_provider = st.selectbox(
        "Choose provider",
        list(PROVIDERS.keys()),
        label_visibility="collapsed",
    )
    provider_cfg = PROVIDERS[selected_provider]

    selected_model = st.selectbox(
        "Model",
        provider_cfg["models"],
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="provider-badge">{provider_cfg["icon"]} {selected_provider} · {selected_model}</div>',
        unsafe_allow_html=True,
    )

    # ── API Key
    env_key = os.getenv(provider_cfg["key_name"], "")
    api_key = st.text_input(
        f"🔑 {provider_cfg['key_name']}",
        value=env_key,
        type="password",
        placeholder="Paste your API key...",
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Document Upload
    st.markdown("### 📄 Document")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload a PDF to start chatting about its content",
    )

    # Show doc info if available
    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.markdown(f"""
        <div class="doc-stats">
            <div class="stat-card">
                <div class="stat-value">{info['pages']}</div>
                <div class="stat-label">Pages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{info['chunks']}</div>
                <div class="stat-label">Chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"📎 {info['filename']}")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Options
    st.markdown("### ⚙️ Options")
    web_search_enabled = st.toggle("🌐 Web Search Fallback", value=False,
                                    help="Search the web if answer isn't in the document")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Actions
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset Document", use_container_width=True):
        for key in ["chain", "vectorstore", "doc_info", "last_file"]:
            st.session_state.pop(key, None)
        st.session_state.messages = []
        st.rerun()


# ─── Main Chat Area ──────────────────────────────────────────────────────────

# Welcome state
if not uploaded_file and not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="icon">📚</div>
        <h3>Welcome to DocMind AI</h3>
        <p>
            Upload a PDF document in the sidebar and start asking questions.
            I'll analyze the content and give you accurate, context-aware answers
            powered by your choice of AI provider.
        </p>
        <div class="features">
            <span class="feature-pill">📄 PDF Analysis</span>
            <span class="feature-pill">💬 Chat Memory</span>
            <span class="feature-pill">🔍 Smart Retrieval</span>
            <span class="feature-pill">🌐 Web Search</span>
            <span class="feature-pill">🤖 Multi-Provider</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Document Processing ────────────────────────────────────────────────────
if uploaded_file:
    if not api_key:
        st.warning(f"⚠️ Please enter your **{provider_cfg['key_name']}** in the sidebar to continue.")
        st.stop()

    # Process document if new file or provider/model changed
    needs_rebuild = (
        "chain" not in st.session_state
        or st.session_state.get("last_file") != uploaded_file.name
        or st.session_state.get("last_provider") != provider_cfg["id"]
        or st.session_state.get("last_model") != selected_model
    )

    if needs_rebuild:
        with st.status("🔄 Processing document...", expanded=True) as status:
            try:
                # Save uploaded file
                st.write("📥 Loading PDF...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    file_path = tmp.name

                # Load & split
                documents = load_pdf(file_path)
                st.write(f"📄 Loaded {len(documents)} pages")

                st.write("✂️ Splitting into chunks...")
                chunks = split_documents(documents)
                st.write(f"🧩 Created {len(chunks)} chunks")

                # Embeddings & vectorstore
                st.write("🧮 Creating embeddings...")
                embed_provider = "google" if provider_cfg["id"] == "google" else "openai"
                embed_key = api_key if provider_cfg["id"] == "google" else api_key
                # For Anthropic, we need OpenAI embeddings key — use env fallback
                if provider_cfg["id"] == "anthropic":
                    embed_key = os.getenv("OPENAI_API_KEY", api_key)

                embeddings = get_embeddings(provider=embed_provider, api_key=embed_key)
                vectorstore = create_vectorstore(chunks, embeddings)
                st.session_state.vectorstore = vectorstore

                # Build chain
                st.write(f"🤖 Connecting to {selected_provider}...")
                chain = build_chain(
                    vectorstore,
                    provider=provider_cfg["id"],
                    model=selected_model,
                    api_key=api_key,
                )
                st.session_state.chain = chain

                # Save state
                st.session_state.last_file = uploaded_file.name
                st.session_state.last_provider = provider_cfg["id"]
                st.session_state.last_model = selected_model
                st.session_state.doc_info = {
                    "filename": uploaded_file.name,
                    "pages": len(documents),
                    "chunks": len(chunks),
                }

                # Clean up temp file
                os.unlink(file_path)

                status.update(label="✅ Document ready!", state="complete")

            except Exception as e:
                status.update(label="❌ Processing failed", state="error")
                st.error(f"**Error:** {str(e)}")
                st.stop()

    # ─── Chat History Display ────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🧠"):
            st.markdown(msg["content"])
            if msg.get("web_results"):
                with st.expander("🌐 Web Search Results"):
                    st.markdown(msg["web_results"])

    # ─── Chat Input ──────────────────────────────────────────────────────────
    if question := st.chat_input("Ask anything about your document..."):
        # Display user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Build chat history for the chain
        chat_history = []
        for msg in st.session_state.messages[:-1]:  # exclude current question
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        # Generate response
        with st.chat_message("assistant", avatar="🧠"):
            try:
                response_placeholder = st.empty()
                full_response = ""

                # Stream the response
                for chunk in st.session_state.chain.stream({
                    "question": question,
                    "chat_history": chat_history,
                }):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                # Web search fallback
                web_results = None
                if web_search_enabled and any(
                    phrase in full_response.lower()
                    for phrase in ["not in the context", "not mentioned", "i don't have", "no information", "cannot find"]
                ):
                    with st.spinner("🌐 Searching the web..."):
                        web_results = search_web(question)
                    if web_results:
                        with st.expander("🌐 Web Search Results", expanded=True):
                            st.markdown(web_results)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "web_results": web_results,
                })

            except Exception as e:
                error_msg = str(e)
                if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                    st.error("🔐 **Authentication failed.** Please check your API key in the sidebar.")
                else:
                    st.error(f"❌ **Error:** {error_msg}")
