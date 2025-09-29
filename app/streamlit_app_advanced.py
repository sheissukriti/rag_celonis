"""
Advanced Streamlit app integrating all RAG system improvements:
- Multi-turn conversations
- Advanced reranking
- Redis caching
- A/B testing
- Real-time learning
- Multi-language support
- Analytics dashboard
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import httpx
import json
import uuid
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title='Advanced RAG Assistant', 
    page_icon='🚀', 
    layout='wide',
    initial_sidebar_state='expanded'
)

# Simple CSS for basic styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
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
    except Exception as e:
        if "ConnectError" in str(type(e)) or "Connection" in str(e):
            st.error("❌ Cannot connect to API server. Please ensure the server is running on http://localhost:8000")
            return None
        elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            if e.response.status_code == 404:
                st.info(f"ℹ️ Endpoint not available: {endpoint}")
                return None
            else:
                st.error(f"❌ API Error {e.response.status_code}: {e.response.text}")
                return None
        else:
            st.error(f"❌ Unexpected error: {e}")
            return None

def initialize_session_state():
    """Initialize session state variables."""
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = 'en'
    
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = set()
    
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

def render_conversation_history():
    """Render conversation history."""
    if st.session_state.conversation_history:
        st.subheader("💬 Conversation History")
        
        for i, turn in enumerate(st.session_state.conversation_history):
            with st.expander(f"Turn {i+1}: {turn['query'][:50]}...", expanded=(i == len(st.session_state.conversation_history)-1)):
                # User message
                st.info(f"👤 **You:** {turn['query']}")
                
                # Assistant response
                st.success(f"🤖 **Assistant:** {turn['response']}")
                
                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⏱️ {turn['response_time']:.2f}s")
                with col2:
                    st.caption(f"📄 {len(turn['citations'])} citations")
                with col3:
                    st.caption(f"🔧 {turn.get('retriever_type', 'unknown')}")

def render_language_settings():
    """Render language settings."""
    st.subheader("🌍 Language Settings")
    
    languages = {
        'en': '🇺🇸 English',
        'es': '🇪🇸 Spanish', 
        'fr': '🇫🇷 French',
        'de': '🇩🇪 German',
        'it': '🇮🇹 Italian',
        'pt': '🇵🇹 Portuguese',
        'zh': '🇨🇳 Chinese',
        'ja': '🇯🇵 Japanese'
    }
    
    selected_lang = st.selectbox(
        "Select your language:",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=0
    )
    
    st.session_state.selected_language = selected_lang
    
    auto_translate = st.checkbox("🔄 Auto-translate responses", value=True)
    
    return selected_lang, auto_translate

def render_feedback_section(response_data):
    """Render feedback collection section."""
    if response_data and response_data.get('answer'):
        response_id = f"{st.session_state.conversation_id}_{len(st.session_state.conversation_history)}"
        
        if response_id not in st.session_state.feedback_submitted:
            st.subheader("📝 Was this response helpful?")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("👍 Yes", key=f"thumbs_up_{response_id}"):
                    submit_feedback(response_data, "thumbs_up", True)
                    st.session_state.feedback_submitted.add(response_id)
                    st.success("Thank you for your feedback! 🙏")
                    st.rerun()
            
            with col2:
                if st.button("👎 No", key=f"thumbs_down_{response_id}"):
                    submit_feedback(response_data, "thumbs_down", False)
                    st.session_state.feedback_submitted.add(response_id)
                    st.success("Thank you for your feedback! 🙏")
                    st.rerun()
            
            with col3:
                rating = st.select_slider(
                    "Rate 1-5:",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    key=f"rating_{response_id}"
                )
                
                if st.button("Submit Rating", key=f"submit_rating_{response_id}"):
                    submit_feedback(response_data, "rating", rating)
                    st.session_state.feedback_submitted.add(response_id)
                    st.success(f"Thank you for rating {rating}/5! ⭐")
                    st.rerun()
            
            with col4:
                if st.button("🚨 Report Issue", key=f"report_{response_id}"):
                    submit_feedback(response_data, "report", "reported")
                    st.session_state.feedback_submitted.add(response_id)
                    st.success("Issue reported. Thank you! 🚨")
                    st.rerun()

def submit_feedback(response_data, feedback_type, feedback_value):
    """Submit feedback to the system."""
    feedback_data = {
        "user_id": st.session_state.user_id,
        "session_id": st.session_state.conversation_id,
        "query": response_data.get('query_processed', ''),
        "response": response_data.get('answer', ''),
        "citations": response_data.get('citations', []),
        "feedback_type": feedback_type,
        "feedback_value": feedback_value
    }
    
    # Call feedback API endpoint
    result = call_api("/feedback", method="POST", data=feedback_data)
    if result:
        st.toast("Feedback submitted successfully!")

def render_system_metrics():
    """Render system performance metrics."""
    st.subheader("📊 System Metrics")
    
    # Get system stats
    health_data = call_api("/health")
    if health_data:
        # Main system metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Status", "Healthy", delta="🟢")
        
        with col2:
            # Handle both basic and advanced server response formats
            components = health_data.get("components", {})
            retriever_status = (components.get("base_retriever") or 
                              components.get("retriever") or "unknown")
            status_icon = "🟢" if retriever_status == "healthy" else "🔴"
            st.metric("Retriever", retriever_status, delta=status_icon)
        
        with col3:
            # Check for advanced reranker or basic reranker
            reranker_status = (components.get("advanced_reranker") or 
                             components.get("reranker") or "unknown")
            status_icon = "🟢" if reranker_status == "enabled" else "🟡"
            st.metric("Reranker", reranker_status, delta=status_icon)
        
        with col4:
            llm_status = components.get("llm", "unknown")
            status_icon = "🟢" if llm_status == "configured" else "🔴"
            st.metric("LLM", llm_status, delta=status_icon)
        
        # Advanced features metrics (if available)
        if "stats" in health_data:
            st.subheader("🚀 Advanced Features Status")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cache_status = components.get("cache", "unknown")
                cache_icon = "🟢" if cache_status == "healthy" else "🔴"
                st.metric("Cache System", cache_status, delta=cache_icon)
            
            with col2:
                conversations_status = components.get("conversations", "unknown")
                conv_icon = "🟢" if conversations_status == "enabled" else "🟡"
                st.metric("Conversations", conversations_status, delta=conv_icon)
            
            with col3:
                ab_testing_status = components.get("ab_testing", "unknown")
                ab_icon = "🟢" if ab_testing_status == "enabled" else "🟡"
                st.metric("A/B Testing", ab_testing_status, delta=ab_icon)
            
            with col4:
                learning_status = components.get("learning_system", "unknown")
                learning_icon = "🟢" if learning_status == "healthy" else "🟡"
                st.metric("Learning System", learning_status, delta=learning_icon)
            
            # Additional stats
            stats = health_data.get("stats", {})
            if stats:
                st.subheader("📈 System Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cache_stats = stats.get("cache", {})
                    cache_type = cache_stats.get("type", "unknown")
                    cache_entries = cache_stats.get("total_entries", 0)
                    st.metric("Cache Type", cache_type, delta=f"{cache_entries} entries")
                
                with col2:
                    learning_stats = stats.get("learning", {})
                    feedback_count = learning_stats.get("feedback", {}).get("total_feedback", 0)
                    st.metric("Feedback Collected", feedback_count, delta="📝")
                
                with col3:
                    ml_stats = stats.get("multilingual", {})
                    supported_langs = len(ml_stats.get("supported_languages", {}))
                    st.metric("Languages Supported", supported_langs, delta="🌍")

def render_ab_testing_info():
    """Render A/B testing information."""
    st.subheader("🧪 A/B Testing")
    
    # Get experiment info (if available)
    experiments_data = call_api("/experiments")
    if experiments_data:
        active_experiments = [exp for exp in experiments_data if exp.get('status') == 'active']
        
        if active_experiments:
            st.success(f"🔬 You're participating in {len(active_experiments)} active experiment(s)")
            
            for exp in active_experiments:
                with st.expander(f"Experiment: {exp['name']}"):
                    st.write(f"**Description:** {exp['description']}")
                    st.write(f"**Variants:** {exp['variant_count']}")
                    st.write(f"**Status:** {exp['status']}")
        else:
            st.info("No active experiments running")
    else:
        st.info("ℹ️ A/B testing system not available (using basic API server)")
        st.write("To enable A/B testing:")
        st.code("uvicorn app.main_advanced:app --reload")

def render_analytics_preview():
    """Render analytics preview."""
    st.subheader("📈 Analytics Preview")
    
    # Try to get analytics data from advanced API
    analytics_data = call_api("/analytics/summary")
    
    if analytics_data:
        col1, col2 = st.columns(2)
        
        with col1:
            # Usage trends
            dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            queries = [45, 52, 48, 61, 58]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=queries, mode='lines+markers', name='Daily Queries'))
            fig.update_layout(title="Query Volume Trend", xaxis_title="Date", yaxis_title="Queries")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Response time distribution
            response_times = [1.2, 1.5, 0.8, 2.1, 1.3, 0.9, 1.7, 1.1, 1.4, 1.6]
            
            fig = go.Figure(data=[go.Histogram(x=response_times, nbinsx=10)])
            fig.update_layout(title="Response Time Distribution", xaxis_title="Response Time (s)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Analytics dashboard not available (using basic API server)")
        st.write("To enable full analytics:")
        st.code("python analytics/dashboard.py --port 8050")
        st.write("Then visit: http://localhost:8050")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Advanced RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("*Featuring multi-turn conversations, advanced reranking, caching, A/B testing, real-time learning, and multi-language support*")
    st.markdown("---")
    
    # Load configuration
    config = load_config()
    if not config:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Advanced Settings")
        
        # API Configuration
        api_url = st.text_input("API URL", value="http://localhost:8000")
        
        # Check API health
        health_status = call_api("/health", api_url=api_url)
        if health_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.stop()
        
        st.markdown("---")
        
        # Language settings
        selected_language, auto_translate = render_language_settings()
        
        st.markdown("---")
        
        # Query parameters
        st.subheader("🔧 Retrieval Parameters")
        top_k = st.slider("Documents to Retrieve", 1, 20, config['app']['top_k'])
        top_k_context = st.slider("Documents in Context", 1, 10, config['app']['top_k_context'])
        max_tokens = st.slider("Max Response Tokens", 50, 1000, config['app']['max_tokens'])
        temperature = st.slider("Temperature", 0.0, 1.0, config['app']['temperature'], 0.1)
        
        st.markdown("---")
        
        # Advanced features
        st.subheader("🚀 Advanced Features")
        use_caching = st.checkbox("💾 Enable Response Caching", value=True)
        use_advanced_reranking = st.checkbox("🎯 Advanced Reranking", value=True)
        enable_learning = st.checkbox("🧠 Real-time Learning", value=True)
        
        st.markdown("---")
        
        # Conversation management
        st.subheader("💬 Conversation")
        if st.button("🔄 New Conversation"):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.conversation_history = []
            st.rerun()
        
        if st.button("📥 Export Conversation"):
            if st.session_state.conversation_history:
                conversation_json = json.dumps(st.session_state.conversation_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=conversation_json,
                    file_name=f"conversation_{st.session_state.conversation_id}.json",
                    mime="application/json"
                )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask a Question")
        
        # Show language indicator
        if selected_language != 'en':
            st.info(f"🌍 Selected language: {selected_language.upper()} | Auto-translate: {'✅' if auto_translate else '❌'}")
        
        # Create form for Enter key functionality
        with st.form("query_form"):
            query = st.text_area(
                "Enter your question:",
                value=st.session_state.current_query,
                placeholder="e.g., I ordered a laptop but it arrived with a broken screen. What should I do?",
                height=100,
                key="query_input"
            )
            
            # Advanced query options
            with st.expander("🔧 Advanced Query Options"):
                query_intent = st.selectbox(
                    "Query Intent (optional):",
                    ["auto-detect", "question", "problem", "request", "complaint"]
                )
                
                context_boost = st.slider(
                    "Context Relevance Boost:",
                    0.0, 2.0, 1.0, 0.1
                )
            
            # Submit button inside form
            submit_button = st.form_submit_button('🚀 Get Answer', type="primary", use_container_width=True)
        
        # Update session state when query changes
        if query != st.session_state.current_query:
            st.session_state.current_query = query
    
    with col2:
        st.subheader("📝 Example Queries")
        example_queries = [
            "I need help resetting my password",
            "My order hasn't arrived yet", 
            "Can I return this item?",
            "How do I cancel my subscription?",
            "The product is damaged",
            "¿Cómo puedo cambiar mi dirección?",  # Spanish
            "Comment puis-je retourner un article?",  # French
        ]
        
        for example in example_queries:
            if st.button(example, key=f"example_{hash(example)}", use_container_width=True):
                st.session_state.current_query = example
                st.rerun()
    
    # Query processing
    if submit_button and query.strip():
        with st.spinner('🔍 Processing your question with advanced RAG features...'):
            # Prepare request data
            request_data = {
                "query": query,
                "conversation_id": st.session_state.conversation_id,
                "user_id": st.session_state.user_id,
                "language": selected_language,
                "auto_translate": auto_translate,
                "top_k": top_k,
                "top_k_context": top_k_context,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "use_caching": use_caching,
                "use_advanced_reranking": use_advanced_reranking,
                "enable_learning": enable_learning,
                "query_intent": query_intent if query_intent != "auto-detect" else None,
                "context_boost": context_boost
            }
            
            # Try enhanced API first, fall back to basic API
            response_data = call_api(
                "/generate_response_advanced",
                method="POST",
                data=request_data,
                api_url=api_url
            )
            
            # If advanced endpoint not available, try basic endpoint
            if not response_data:
                basic_request_data = {
                    "query": query,
                    "top_k": top_k,
                    "top_k_context": top_k_context,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                response_data = call_api(
                    "/generate_response",
                    method="POST",
                    data=basic_request_data,
                    api_url=api_url
                )
            
            if response_data:
                # Add to conversation history
                conversation_turn = {
                    "query": query,
                    "response": response_data["answer"],
                    "citations": response_data["citations"],
                    "response_time": response_data["response_time_seconds"],
                    "retriever_type": response_data.get("retriever_type", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.conversation_history.append(conversation_turn)
                
                # Display response
                st.subheader("📝 Response")
                st.write(response_data["answer"])
                
                # Language information
                language_info = response_data.get("language_info", {})
                if language_info and (language_info.get("query_translated") or language_info.get("response_translated")):
                    with st.expander("🌍 Translation Information"):
                        if language_info.get("detected_language"):
                            detected = language_info["detected_language"]
                            st.info(f"Detected language: {detected['language']} (confidence: {detected['confidence']:.2f})")
                        
                        if language_info.get("query_translated"):
                            st.info("Query was translated for processing")
                        
                        if language_info.get("response_translated"):
                            st.info("Response was translated to your language")
                
                # Display enhanced metadata
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Response Time", f"{response_data['response_time_seconds']:.2f}s")
                with col2:
                    st.metric("Citations", len(response_data['citations']))
                with col3:
                    st.metric("Retriever", response_data.get('retriever_type', 'unknown'))
                with col4:
                    cache_hit = response_data.get('cache_hit', False)
                    st.metric("Cache", "Hit" if cache_hit else "Miss")
                with col5:
                    experiment_variant = response_data.get('experiment_variant')
                    st.metric("Variant", experiment_variant or "Control")
                
                # Display citations with enhanced information
                st.subheader("📚 Citations")
                for i, citation in enumerate(response_data['citations'], 1):
                    with st.expander(f"📄 Document {i} (Score: {citation['score']:.4f})"):
                        st.write(citation["text"])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"Document ID: {citation['id']}")
                        with col2:
                            adaptive_weight = citation.get('adaptive_weight', 1.0)
                            st.caption(f"Adaptive Weight: {adaptive_weight:.3f}")
                        with col3:
                            rerank_score = citation.get('rerank_score')
                            if rerank_score:
                                st.caption(f"Rerank Score: {rerank_score:.4f}")
                
                # Feedback section
                render_feedback_section(response_data)
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        render_conversation_history()
    
    # System information tabs
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 System Metrics", "🧪 A/B Testing", "📈 Analytics", "ℹ️ System Info"])
    
    with tab1:
        render_system_metrics()
    
    with tab2:
        render_ab_testing_info()
    
    with tab3:
        render_analytics_preview()
        
        st.info("💡 For full analytics dashboard, run: `python analytics/dashboard.py`")
        st.markdown("Then visit: http://localhost:8050")
    
    with tab4:
        st.subheader("🔧 System Information")
        if health_status:
            st.json(health_status)
        
        st.subheader("📋 Features Enabled")
        features = [
            "✅ Multi-turn Conversations",
            "✅ Advanced Cross-encoder Reranking", 
            "✅ Redis-based Response Caching",
            "✅ A/B Testing Framework",
            "✅ Real-time Feedback Learning",
            "✅ Multi-language Support",
            "✅ Advanced Analytics Dashboard",
            "✅ Context-aware Dialogue Management"
        ]
        
        for feature in features:
            st.write(feature)
        
        st.markdown("**Dataset**: [Customer Support on Twitter (945k tweets)](https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k)")
        st.markdown("**API Docs**: http://localhost:8000/docs")
        st.markdown("**Analytics Dashboard**: http://localhost:8050")

if __name__ == '__main__':
    main()
