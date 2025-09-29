import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import httpx
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import asyncio

# Page configuration
st.set_page_config(
    page_title='Customer Support Assistant',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded'
)
# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
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

async def call_api(endpoint: str, method: str = "GET", data: dict = None, api_url: str = "http://localhost:8000"):
    """Make API calls with proper error handling."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            if method == "GET":
                response = await client.get(f"{api_url}{endpoint}")
            elif method == "POST":
                response = await client.post(f"{api_url}{endpoint}", json=data)
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
    
    # Load configuration
    config = load_config()
    if not config:
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
    
    # Check API health
    health_status = asyncio.run(call_api("/health", api_url=api_url))
    if health_status:
        st.sidebar.success("✅ API Connected")
        with st.sidebar.expander("System Status"):
            st.json(health_status)
    else:
        st.sidebar.error("❌ API Disconnected")
    
    # Query parameters
    st.sidebar.subheader("Retrieval Parameters")
    top_k = st.sidebar.slider("Documents to Retrieve", 1, 20, config['app']['top_k'])
    top_k_context = st.sidebar.slider("Documents in Context", 1, 10, config['app']['top_k_context'])
    max_tokens = st.sidebar.slider("Max Response Tokens", 50, 1000, config['app']['max_tokens'])
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, config['app']['temperature'], 0.1)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Evaluation", "📈 Analytics", "🔧 System Info"])
    
    with tab1:
        st.header("Ask a Question")
        
        # Predefined example queries
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "Enter your customer support question:",
                placeholder="e.g., I ordered a laptop but it arrived with a broken screen. What should I do?",
                height=100
            )
        
        with col2:
            st.write("**Example Queries:**")
            example_queries = [
                "I need help resetting my password",
                "My order hasn't arrived yet",
                "Can I return this item?",
                "How do I cancel my subscription?",
                "The product is damaged"
            ]
            
            for example in example_queries:
                if st.button(example, key=f"example_{hash(example)}"):
                    query = example
                    st.rerun()
        
        if st.button('🚀 Get Answer', type="primary") and query.strip():
            with st.spinner('🔍 Searching knowledge base and generating response...'):
                # Call API
                response_data = asyncio.run(call_api(
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
                ))
                
                if response_data:
                    # Display answer
                    st.subheader("📝 Answer")
                    st.markdown(f"<div class='metric-card'>{response_data['answer']}</div>", 
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
                            st.markdown(f"<div class='citation-box'>{citation['text']}</div>", 
                                       unsafe_allow_html=True)
                            st.caption(f"Document ID: {citation['id']}")
    
    with tab2:
        st.header("📊 System Evaluation")
        st.write("Run comprehensive evaluation on predefined test queries")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🧪 Run Evaluation", type="primary"):
                with st.spinner("Running evaluation... This may take a few minutes."):
                    eval_results = asyncio.run(call_api("/evaluate", method="POST", api_url=api_url))
                    
                    if eval_results:
                        st.success("✅ Evaluation completed!")
                        
                        # Display metrics
                        metrics = eval_results.get("metrics", {})
                        
                        # Overall performance
                        st.subheader("Overall Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Overall Score", f"{metrics.get('avg_overall_score', 0):.3f}")
                        with col2:
                            st.metric("Relevance", f"{metrics.get('avg_relevance_score', 0):.3f}")
                        with col3:
                            st.metric("Coherence", f"{metrics.get('avg_coherence_score', 0):.3f}")
                        with col4:
                            st.metric("Faithfulness", f"{metrics.get('avg_faithfulness_score', 0):.3f}")
                        
                        # Retrieval performance
                        st.subheader("Retrieval Performance")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Precision", f"{metrics.get('avg_retrieval_precision', 0):.3f}")
                        with col2:
                            st.metric("Recall", f"{metrics.get('avg_retrieval_recall', 0):.3f}")
                        with col3:
                            st.metric("F1-Score", f"{metrics.get('avg_retrieval_f1', 0):.3f}")
                        
                        # Performance metrics
                        st.subheader("Performance Metrics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
                        with col2:
                            st.metric("Avg Citations", f"{metrics.get('avg_num_citations', 0):.1f}")
                        with col3:
                            st.metric("Avg Answer Length", f"{metrics.get('avg_answer_length', 0):.0f} words")
        
        with col2:
            # Display test queries
            test_queries_data = asyncio.run(call_api("/test-queries", api_url=api_url))
            if test_queries_data:
                st.subheader("Test Queries")
                for i, query_data in enumerate(test_queries_data["test_queries"], 1):
                    st.write(f"{i}. {query_data['query']}")
    
    with tab3:
        st.header("📈 Analytics Dashboard")
        st.write("System performance analytics and insights")
        
        # Load response logs if available
        log_file = config['app'].get('log_file', 'logs/responses.jsonl')
        
        try:
            if Path(log_file).exists():
                # Read logs
                logs = []
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                
                if logs:
                    df = pd.DataFrame(logs)
                    
                    # Response time distribution
                    st.subheader("Response Time Distribution")
                    fig_time = px.histogram(df, x='response_time_seconds', 
                                          title="Response Time Distribution",
                                          labels={'response_time_seconds': 'Response Time (seconds)'})
                    st.plotly_chart(fig_time, use_container_width=True)
                    
                    # Query types over time
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df['hour'] = df['timestamp'].dt.hour
                        
                        st.subheader("Query Volume by Hour")
                        hourly_counts = df.groupby('hour').size().reset_index(name='count')
                        fig_hourly = px.bar(hourly_counts, x='hour', y='count',
                                          title="Queries by Hour of Day")
                        st.plotly_chart(fig_hourly, use_container_width=True)
                    
                    # Recent queries
                    st.subheader("Recent Queries")
                    recent_logs = df.tail(10)[['timestamp', 'query', 'response_time_seconds']]
                    st.dataframe(recent_logs, use_container_width=True)
                
                else:
                    st.info("📊 No analytics data available yet. Generate some responses to see analytics!")
            else:
                st.info("📊 No log file found. Enable logging in configuration to see analytics.")
                
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
    
    with tab4:
        st.header("🔧 System Information")
        
        # Configuration display
        st.subheader("Current Configuration")
        st.json(config)
        
        # System health
        health_status = asyncio.run(call_api("/health"))
        if health_status:
            st.subheader("System Health")
            
            # Component status
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
        
        # Dataset information
        st.subheader("Dataset Information")
        st.markdown("""
        **Dataset**: [Customer Support on Twitter (945k tweets)](https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k)
        
        This dataset contains real customer support conversations from Twitter, providing:
        - 945,278 customer query-response pairs
        - Diverse customer service scenarios
        - Preprocessed and cleaned interactions
        - English language conversations
        """)
        
        # File status
        st.subheader("File Status")
        files_to_check = [
            ("Configuration", "configs/config.yaml"),
            ("QA Data", config['app']['qa_texts_path']),
            ("Index", config['app']['index_path']),
            ("Response Log", config['app'].get('log_file', 'logs/responses.jsonl'))
        ]
        
        for name, path in files_to_check:
            exists = Path(path).exists()
            status = "✅ Exists" if exists else "❌ Missing"
            size = f"({Path(path).stat().st_size // 1024} KB)" if exists else ""
            st.write(f"**{name}**: {status} {size}")

# Sidebar with additional controls
with st.sidebar:
    st.header("🎛️ Advanced Options")
    
    # Template selection
    template_type = st.selectbox(
        "Response Template",
        ["default", "conversational", "structured", "concise"],
        help="Choose the response style template"
    )
    
    # API testing
    st.subheader("🧪 Quick API Test")
    if st.button("Test API Connection"):
        health = asyncio.run(call_api("/health"))
        if health:
            st.success("API is responsive!")
        else:
            st.error("API connection failed")

if __name__ == '__main__':
    main()
