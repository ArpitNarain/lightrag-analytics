import os
import streamlit as st
import time
from datetime import datetime
from dotenv import load_dotenv
import hashlib

# Load environment variables
load_dotenv()

st.set_page_config(page_title="LightRAG Debug", page_icon="🔍", layout="wide")

st.title("🔍 LightRAG Debug Interface")

# Check for OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ OPENAI_API_KEY not found")
    st.stop()

# Try to import LightRAG
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_embed
    
    st.success("✅ LightRAG imported successfully")
    
    # Initialize working directory
    WORKING_DIR = "./findings_debug"
    
    # Force clean directory every time
    import shutil
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)
    st.info(f"🗂️ Clean working directory created: {WORKING_DIR}")
    
    # Custom LLM wrapper to avoid parameter conflicts
    async def llm_wrapper(prompt, **kwargs):
        return await gpt_4o_mini_complete(prompt)
    
    # Custom embedding wrapper
    def embed_wrapper(texts):
        return openai_embed(texts)
    
    # Initialize RAG (fresh every time)
    st.info("🔄 Initializing fresh LightRAG instance...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_wrapper,
        llm_model_kwargs={"temperature": 0.0},
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embed_wrapper
        )
    )
    st.success("✅ LightRAG initialized")
    
    # Test OpenAI API
    if st.button("🧪 Test OpenAI"):
        try:
            import asyncio
            result = asyncio.run(gpt_4o_mini_complete("Say 'Hello World'"))
            st.success(f"✅ OpenAI works: {result}")
        except Exception as e:
            st.error(f"❌ OpenAI failed: {e}")
    
    # Document upload
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload document", type=['txt'])
    
    if uploaded_file:
        # Read document
        document_text = str(uploaded_file.read(), "utf-8")
        doc_hash = hashlib.md5(document_text.encode()).hexdigest()[:8]
        
        st.info(f"📋 Document: {len(document_text)} chars, hash: {doc_hash}")
        st.text_area("Preview:", document_text[:500], height=150, disabled=True)
        
        # Process document
        if st.button("🚀 Process Document"):
            with st.spinner("Processing..."):
                try:
                    # Check directory before
                    before_files = os.listdir(WORKING_DIR)
                    st.info(f"📁 Before: {before_files}")
                    
                    # Insert document
                    start_time = time.time()
                    rag.insert(document_text)
                    process_time = time.time() - start_time
                    
                    # Check directory after
                    after_files = os.listdir(WORKING_DIR)
                    st.info(f"📁 After: {after_files}")
                    st.info(f"⏱️ Processing took: {process_time:.2f}s")
                    
                    # Check for critical files
                    critical_files = ['graph_chunk_entity_relation.json', 'entities.json', 'relationships.json']
                    created_critical = [f for f in after_files if any(c in f for c in ['graph', 'entities', 'relationships'])]
                    
                    if created_critical:
                        st.success(f"✅ Knowledge graph files created: {created_critical}")
                    else:
                        st.error(f"❌ No knowledge graph files created!")
                        st.info("This explains why queries fail")
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Query interface
    st.header("🤔 Query Interface")
    query = st.text_input("Enter query:", "What are the main models?")
    
    if st.button("🔍 Run Query") and query:
        try:
            start_time = time.time()
            result = rag.query(query, param=QueryParam(mode="hybrid"))
            query_time = time.time() - start_time
            
            st.markdown(f"**Result ({query_time:.2f}s):**")
            st.write(result)
            
            if "[no-context]" in result:
                files = os.listdir(WORKING_DIR)
                st.warning(f"⚠️ No context found. Working dir files: {files}")
                
        except Exception as e:
            st.error(f"❌ Query failed: {e}")

except ImportError as e:
    st.error(f"❌ Import failed: {e}")
    st.info("LightRAG not available")
