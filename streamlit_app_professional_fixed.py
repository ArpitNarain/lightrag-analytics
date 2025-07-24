import os
import streamlit as st
import pandas as pd
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LightRAG Analytics Platform", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Metrics cards */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e4e7;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
    }
    
    .result-title {
        font-weight: 600;
        color: #2c3e50;
        font-size: 1.2rem;
    }
    
    .result-time {
        color: #7f8c8d;
        font-size: 0.9rem;
        background: #ecf0f1;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
    }
    
    /* Query history */
    .query-history {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    
    /* Progress indicators */
    .progress-container {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    /* Document stats */
    .doc-stats {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Error container */
    .error-container {
        background: #fee;
        border: 1px solid #fcc;
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = []
if 'document_stats' not in st.session_state:
    st.session_state.document_stats = {}
if 'last_results' not in st.session_state:
    st.session_state.last_results = {}

# Main header
st.markdown("""
<div class="main-header fade-in">
    <h1>🧠 LightRAG Analytics Platform</h1>
    <p>Advanced RAG Query Analysis with Multi-Modal Search Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Check for OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")
    st.stop()

# Working directory setup with session state persistence
if 'rag_initialized' not in st.session_state:
    WORKING_DIR = "./findings_RAG"
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR, exist_ok=True)
    st.session_state.working_dir = WORKING_DIR
    st.session_state.rag_initialized = False
    st.session_state.current_document_hash = None

WORKING_DIR = st.session_state.working_dir

# Try to import and initialize LightRAG
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_embed
    
    # Initialize RAG system with session state caching
    def initialize_rag():
        if 'rag_instance' not in st.session_state:
            st.session_state.rag_instance = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=gpt_4o_mini_complete,
                llm_model_kwargs={"temperature": 0.0},
                embedding_func=EmbeddingFunc(
                    embedding_dim=1536,
                    max_token_size=8192,
                    func=lambda texts: openai_embed(texts, model="text-embedding-3-small")
                )
            )
        return st.session_state.rag_instance
    
    rag = initialize_rag()
    lightrag_available = True
    
except ImportError as e:
    lightrag_available = False
    st.markdown(f"""
    <div class="error-container">
        <h2>⚠️ LightRAG Installation Issue</h2>
        <p>LightRAG package is being installed. This may take 5-10 minutes on first deployment.</p>
        <p><strong>Error:</strong> {str(e)}</p>
        <p>Please wait for the deployment to complete, then refresh the page.</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        max_tokens = st.number_input("Max Tokens", 100, 4000, 1000)
        model_selection = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    
    st.markdown("---")
    
    # Performance metrics
    if st.session_state.performance_metrics:
        st.markdown("### 📊 Performance Metrics")
        avg_time = sum(m['time'] for m in st.session_state.performance_metrics[-5:]) / min(5, len(st.session_state.performance_metrics))
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
        st.metric("Total Queries", len(st.session_state.performance_metrics))
    
    st.markdown("---")
    
    # Export options
    st.markdown("### 📥 Export Options")
    if st.session_state.last_results:
        if st.button("📄 Export to TXT"):
            # Text export functionality
            export_text = "LightRAG Query Results\n" + "="*50 + "\n\n"
            export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for mode, result in st.session_state.last_results.items():
                export_text += f"{mode.upper()} Results:\n" + "-"*30 + "\n"
                export_text += f"{result}\n\n"
            
            st.download_button(
                label="Download TXT",
                data=export_text,
                file_name=f"lightrag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Query suggestions
    st.markdown("### 💡 Query Suggestions")
    query_templates = [
        "Which models have compliance issues?",
        "What are the key findings related to risk management?",
        "Summarize the main regulatory concerns",
        "List all models mentioned in the document",
        "What are the common themes across findings?"
    ]
    
    selected_template = st.selectbox("Quick Templates", ["Custom"] + query_templates)

# Only show main interface if LightRAG is available
if lightrag_available:
    # Main content area - Two column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Document upload section
        st.markdown("### 📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload your document", 
            type=['txt', 'md', 'csv'],
            help="Drag and drop or click to upload"
        )
        
        # Handle file upload
        if uploaded_file is not None:
            try:
                # Read uploaded file
                if uploaded_file.type == "text/plain":
                    document_text = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "text/csv":
                    document_text = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "text/markdown":
                    document_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.error("Unsupported file type.")
                    st.stop()
                
                # Create document hash to track if it's already processed
                import hashlib
                document_hash = hashlib.md5(document_text.encode()).hexdigest()
                
                # Document statistics
                word_count = len(document_text.split())
                char_count = len(document_text)
                estimated_tokens = char_count // 4
                
                st.session_state.document_stats = {
                    'filename': uploaded_file.name,
                    'size': len(document_text),
                    'word_count': word_count,
                    'char_count': char_count,
                    'estimated_tokens': estimated_tokens
                }
                
                st.markdown(f"""
                <div class="doc-stats">
                    <h4>📊 Document Statistics</h4>
                    <p><strong>File:</strong> {uploaded_file.name}</p>
                    <p><strong>Words:</strong> {word_count:,}</p>
                    <p><strong>Characters:</strong> {char_count:,}</p>
                    <p><strong>Est. Tokens:</strong> {estimated_tokens:,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Document preview
                with st.expander("👁️ Document Preview", expanded=False):
                    st.text_area("First 500 characters:", document_text[:500], height=100, disabled=True)
                
                # Check if we need to process this document
                if (st.session_state.current_document_hash != document_hash or 
                    not st.session_state.rag_initialized):
                    
                    # Insert document into RAG
                    with st.spinner("🔄 Processing document (first time may take longer)..."):
                        start_time = time.time()
                        try:
                            # Force clean working directory
                            import shutil
                            if os.path.exists(WORKING_DIR):
                                shutil.rmtree(WORKING_DIR)
                            os.makedirs(WORKING_DIR, exist_ok=True)
                            
                            # Reinitialize RAG with clean directory
                            if 'rag_instance' in st.session_state:
                                del st.session_state.rag_instance
                            rag = initialize_rag()
                            
                            # Add verbose logging
                            st.info("🔄 Starting fresh document insertion...")
                            st.info(f"📄 Document length: {len(document_text)} characters")
                            st.info(f"📁 Working directory: {WORKING_DIR}")
                            st.info(f"🔑 OpenAI API key exists: {bool(openai_api_key)}")
                            
                            # Check if document is empty or problematic
                            if not document_text or len(document_text.strip()) < 100:
                                st.error("⚠️ Document is too short or empty!")
                                st.stop()
                            
                            # Try to process with more logging
                            try:
                                # Check if we have event loop issues
                                import asyncio
                                try:
                                    loop = asyncio.get_event_loop()
                                    st.info(f"🔄 Event loop: {loop}")
                                except RuntimeError as e:
                                    st.warning(f"⚠️ Event loop issue: {e}")
                                
                                # Try to insert document
                                rag.insert(document_text)
                            except Exception as insert_error:
                                st.error(f"❌ Insert failed: {insert_error}")
                                import traceback
                                st.code(traceback.format_exc())
                                
                            processing_time = time.time() - start_time
                            
                            # Check what files were created
                            created_files = os.listdir(WORKING_DIR) if os.path.exists(WORKING_DIR) else []
                            st.info(f"📁 Files created: {created_files}")
                            
                            # Check if critical files exist
                            critical_files = ['graph_chunk_entity_relation.json', 'entities.json', 'relationships.json']
                            missing_files = [f for f in critical_files if f not in created_files]
                            
                            if missing_files:
                                st.warning(f"⚠️ Missing critical files: {missing_files}")
                                st.info("This explains why queries return [no-context]")
                            else:
                                st.success("✅ All critical knowledge graph files created!")
                            
                        except Exception as e:
                            processing_time = time.time() - start_time
                            st.error(f"⚠️ Document processing error: {str(e)}")
                            st.error(f"Error type: {type(e).__name__}")
                            import traceback
                            st.code(traceback.format_exc())
                        
                        # Mark as processed
                        st.session_state.current_document_hash = document_hash
                        st.session_state.rag_initialized = True
                        st.session_state.processed_document = document_text
                    
                    st.success(f"✅ Document processed in {processing_time:.2f}s!")
                    st.info(f"📋 Debug: Document hash {document_hash[:8]}... | Working dir: {WORKING_DIR}")
                else:
                    st.success("✅ Document already processed - ready for queries!")
                    st.info(f"📋 Debug: Using cached document {document_hash[:8]}... | Working dir: {WORKING_DIR}")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        # Query history
        if st.session_state.query_history:
            st.markdown("### 📜 Query History")
            with st.expander("Recent Queries", expanded=False):
                for i, query in enumerate(reversed(st.session_state.query_history[-5:])):
                    if st.button(f"🔄 {query['query'][:30]}...", key=f"history_{i}"):
                        st.session_state.selected_query = query['query']
    
    with col2:
        # Query interface
        st.markdown("### 🤔 Query Interface")
        
        # Use template or custom query
        if selected_template != "Custom":
            default_query = selected_template
        else:
            default_query = "Which models have compliance issues. Answer strictly based on the document provided. If the relevant information is not present, say I don't know"
        
        user_query = st.text_area(
            "Enter your query:", 
            value=st.session_state.get('selected_query', default_query),
            height=100,
            help="💡 Tip: Press Ctrl+Enter to search",
            key="query_input"
        )
        
        # Character count
        st.caption(f"Characters: {len(user_query)}")
        
        # Search mode selection
        col_modes1, col_modes2 = st.columns(2)
        with col_modes1:
            naive_mode = st.checkbox("🔍 Naive", value=True)
            local_mode = st.checkbox("🏠 Local", value=True)
        with col_modes2:
            global_mode = st.checkbox("🌍 Global", value=True)
            hybrid_mode = st.checkbox("⚡ Hybrid", value=True)
        
        search_modes = []
        if naive_mode: search_modes.append("naive")
        if local_mode: search_modes.append("local")
        if global_mode: search_modes.append("global")
        if hybrid_mode: search_modes.append("hybrid")
        
        # Query execution
        col_search, col_clear = st.columns([3, 1])
        with col_search:
            search_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.last_results = {}
                st.rerun()
        
        # Add OpenAI API test button
        if st.button("🧪 Test OpenAI API"):
            try:
                from lightrag.llm.openai import gpt_4o_mini_complete
                import asyncio
                
                # Test async function properly - don't pass model parameter
                async def test_openai():
                    result = await gpt_4o_mini_complete("Say 'API working!'")
                    return result
                
                # Run async function
                test_result = asyncio.run(test_openai())
                st.success(f"✅ OpenAI API works: {test_result}")
            except Exception as e:
                st.error(f"❌ OpenAI API failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Add force reprocess button for debugging
        if st.button("🔄 Force Clean Reprocess"):
            if 'processed_document' in st.session_state:
                with st.spinner("🔄 Force clean reprocessing document..."):
                    try:
                        # Force clean working directory
                        import shutil
                        if os.path.exists(WORKING_DIR):
                            shutil.rmtree(WORKING_DIR)
                        os.makedirs(WORKING_DIR, exist_ok=True)
                        
                        # Clear the RAG instance and reinitialize
                        if 'rag_instance' in st.session_state:
                            del st.session_state.rag_instance
                        
                        # Reinitialize RAG
                        rag = initialize_rag()
                        
                        # Reprocess document
                        st.info("🔄 Starting completely fresh processing...")
                        rag.insert(st.session_state.processed_document)
                        
                        # Check results
                        created_files = os.listdir(WORKING_DIR) if os.path.exists(WORKING_DIR) else []
                        st.info(f"📁 Files created after clean reprocess: {created_files}")
                        
                        st.success("✅ Document force reprocessed with clean slate!")
                        st.session_state.rag_initialized = True
                    except Exception as e:
                        st.error(f"❌ Error reprocessing: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.warning("No document to reprocess. Upload a document first.")
        
        # Query execution
        if search_clicked:
            if not user_query:
                st.warning("⚠️ Please enter a query.")
            elif not search_modes:
                st.warning("⚠️ Please select at least one search mode.")
            elif not st.session_state.rag_initialized:
                st.warning("⚠️ Please upload and process a document first.")
            else:
                # Add to query history
                st.session_state.query_history.append({
                    'query': user_query,
                    'timestamp': datetime.now(),
                    'modes': search_modes
                })
                
                st.markdown("### 📊 Analysis Results")
                
                results = {}
                performance_data = []
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, mode in enumerate(search_modes):
                    status_text.text(f"Running {mode.upper()} search...")
                    progress_bar.progress((i) / len(search_modes))
                    
                    start_time = time.time()
                    try:
                        result = rag.query(user_query, param=QueryParam(mode=mode))
                        end_time = time.time()
                        
                        # Add debug info for no-context responses
                        if "[no-context]" in result:
                            # Check what files exist in working directory
                            working_dir_files = []
                            if os.path.exists(WORKING_DIR):
                                working_dir_files = os.listdir(WORKING_DIR)
                            
                            result += f"\n\n🔍 Detailed Debug Info:\n- Document processed: {st.session_state.rag_initialized}\n- Working dir exists: {os.path.exists(WORKING_DIR)}\n- Working dir files: {working_dir_files}\n- Query mode: {mode}\n- Document hash: {st.session_state.get('current_document_hash', 'None')[:8]}...\n- Document length: {len(st.session_state.get('processed_document', ''))}"
                        
                        results[mode] = result
                        performance_data.append({
                            'mode': mode,
                            'time': end_time - start_time,
                            'timestamp': datetime.now()
                        })
                        
                    except Exception as e:
                        results[mode] = f"❌ Error: {str(e)}"
                        performance_data.append({
                            'mode': mode,
                            'time': 0,
                            'timestamp': datetime.now()
                        })
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                
                # Store results and metrics
                st.session_state.last_results = results
                st.session_state.performance_metrics.extend(performance_data)
                
                # Display results with professional cards
                for mode, result in results.items():
                    execution_time = next((p['time'] for p in performance_data if p['mode'] == mode), 0)
                    
                    st.markdown(f"""
                    <div class="result-card fade-in">
                        <div class="result-header">
                            <div class="result-title">🔍 {mode.upper()} Search Results</div>
                            <div class="result-time">⏱️ {execution_time:.2f}s</div>
                        </div>
                        <div style="white-space: pre-wrap; line-height: 1.6;">{result}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance comparison
                if len(performance_data) > 1:
                    st.markdown("### ⚡ Performance Comparison")
                    perf_df = pd.DataFrame(performance_data)
                    perf_df['time'] = perf_df['time'].round(3)
                    st.bar_chart(perf_df.set_index('mode')['time'])

else:
    st.markdown("""
    <div style="text-align: center; margin-top: 4rem;">
        <h2>⏳ Setting up your analytics platform...</h2>
        <p>LightRAG is being installed. Please wait and refresh in 5-10 minutes.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p>🧠 <strong>LightRAG Analytics Platform</strong> | Built with Advanced RAG Technology</p>
    <p><em>Empowering Data Scientists with Intelligent Document Analysis</em></p>
</div>
""", unsafe_allow_html=True)
