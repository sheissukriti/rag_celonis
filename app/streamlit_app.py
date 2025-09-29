import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
import httpx
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title='Customer Support Assistant', 
    page_icon='🤖', 
    layout='wide'
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    color: #1f77b4;
}

.answer-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}

.citation-box {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.25rem;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_config():
    """Load configuration from YAML file."""
    try:
        import yaml
        with open('configs/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        return None

def call_api(endpoint: str, method: str = "GET", data: dict = None, api_url: str = "http://localhost:8000"):
    """Make API calls with proper error handling."""
    try:
        with httpx.Client(timeout=60) as client:
            if method == "GET":
                response = client.get(f"{api_url}{endpoint}")
            elif method == "POST":
                response = client.post(f"{api_url}{endpoint}", json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        st.error("❌ Cannot connect to API server. Please ensure the server is running on http://localhost:8000")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"❌ API Error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Customer Support RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load configuration
    config = load_config()
    if not config:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
    
    # Check API health
    health_status = call_api("/health", api_url=api_url)
    if health_status:
        st.sidebar.success("✅ API Connected")
        with st.sidebar.expander("System Status"):
            st.json(health_status)
    else:
        st.sidebar.error("❌ API Disconnected")
        st.stop()
    
    # Query parameters
    st.sidebar.subheader("Retrieval Parameters")
    top_k = st.sidebar.slider("Documents to Retrieve", 1, 20, config['app']['top_k'])
    top_k_context = st.sidebar.slider("Documents in Context", 1, 10, config['app']['top_k_context'])
    max_tokens = st.sidebar.slider("Max Response Tokens", 50, 1000, config['app']['max_tokens'])
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, config['app']['temperature'], 0.1)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask a Question")
        query = st.text_area(
            "Enter your customer support question:",
            placeholder="e.g., I ordered a laptop but it arrived with a broken screen. What should I do?",
            height=100
        )
    
    with col2:
        st.subheader("📝 Example Queries")
        example_queries = [
            "I need help resetting my password",
            "My order hasn't arrived yet", 
            "Can I return this item?",
            "How do I cancel my subscription?",
            "The product is damaged"
        ]
        
        for example in example_queries:
            if st.button(example, key=f"example_{hash(example)}", use_container_width=True):
                query = example
                st.rerun()
    
    if st.button('🚀 Get Answer', type="primary", use_container_width=True) and query.strip():
        with st.spinner('🔍 Searching knowledge base and generating response...'):
            # Call API
            response_data = call_api(
                "/generate_response",
                method="POST",
                data={
                    "query": query,
                    "top_k": top_k,
                    "top_k_context": top_k_context,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                api_url=api_url
            )
            
            if response_data:
                # Display answer
                st.subheader("📝 Answer")
                st.markdown(f'<div class="answer-box">{response_data["answer"]}</div>', 
                           unsafe_allow_html=True)
                
                # Display metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Response Time", f"{response_data['response_time_seconds']:.2f}s")
                with col2:
                    st.metric("Citations Used", len(response_data['citations']))
                with col3:
                    st.metric("Retriever Type", response_data['retriever_type'])
                
                # Display citations
                st.subheader("📚 Citations")
                
                for i, citation in enumerate(response_data['citations'], 1):
                    with st.expander(f"📄 Document {i} (Score: {citation['score']:.4f})"):
                        st.markdown(f'<div class="citation-box">{citation["text"]}</div>', 
                                   unsafe_allow_html=True)
                        st.caption(f"Document ID: {citation['id']}")
    
    # Footer with system info
    st.markdown("---")
    with st.expander("🔧 System Information"):
        if health_status:
            st.write("**System Health:**")
            components = health_status.get("components", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                retriever_status = components.get("retriever", "unknown")
                color = "🟢" if retriever_status == "healthy" else "🔴"
                st.write(f"{color} **Retriever**: {retriever_status}")
            
            with col2:
                reranker_status = components.get("reranker", "unknown")
                color = "🟢" if reranker_status == "enabled" else "🟡"
                st.write(f"{color} **Reranker**: {reranker_status}")
            
            with col3:
                llm_status = components.get("llm", "unknown")
                color = "🟢" if llm_status == "configured" else "🔴"
                st.write(f"{color} **LLM**: {llm_status}")
        
        st.write("**Dataset**: [Customer Support on Twitter (945k tweets)](https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k)")
        st.write("**API Docs**: http://localhost:8000/docs")

if __name__ == '__main__':
    main()


