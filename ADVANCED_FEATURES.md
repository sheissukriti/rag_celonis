# Advanced RAG System Features

This document describes the advanced features implemented in the RAG system, including multi-turn conversations, advanced reranking, caching, A/B testing, real-time learning, multi-language support, and analytics dashboard.

## 🚀 Overview of New Features

### 1. Multi-turn Conversations
**Context-aware dialogue management that maintains conversation history**

- **Conversation Management**: Persistent conversation storage with automatic cleanup
- **Context-aware Prompts**: Incorporates previous turns for better responses
- **Session Tracking**: User and session-based conversation tracking
- **Conversation Export**: Export conversation history as JSON

**Key Components:**
- `rag/conversation.py` - Core conversation management
- Conversation storage in `store/conversations/`
- Context-aware prompt building

**Usage:**
```python
# Create a new conversation
conversation_id = conversation_manager.create_conversation(user_id="user123")

# Add turns to conversation
conversation_manager.add_turn(
    conversation_id, 
    user_query="I need help with my order",
    assistant_response="I'd be happy to help...",
    citations=[...]
)

# Get conversation context for next response
context = conversation_manager.get_conversation_context(conversation_id)
```

### 2. Advanced Reranking
**Cross-encoder models for superior relevance scoring**

- **Cross-encoder Reranking**: Uses transformer models for pairwise relevance scoring
- **Semantic Similarity**: Alternative embedding-based reranking
- **Hybrid Reranking**: Combines multiple reranking strategies
- **Configurable Models**: Support for different cross-encoder models

**Key Components:**
- `rag/advanced_rerank.py` - Advanced reranking implementations
- Support for Hugging Face cross-encoder models
- Fallback to semantic similarity if cross-encoder unavailable

**Usage:**
```python
# Create cross-encoder reranker
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Rerank documents
reranked_docs = reranker.rerank(query, documents, top_k=10)
```

### 3. Redis-based Caching
**High-performance response caching with Redis backend**

- **Response Caching**: Cache generated responses to reduce latency
- **Redis Integration**: Scalable caching with Redis backend
- **Memory Fallback**: Automatic fallback to in-memory caching
- **Cache Statistics**: Comprehensive caching metrics
- **TTL Management**: Configurable time-to-live for cache entries

**Key Components:**
- `rag/cache.py` - Caching layer implementation
- Redis-based and memory-based cache managers
- Response-specific caching with query normalization

**Usage:**
```python
# Initialize cache manager
cache_manager = CacheManager(cache_config)
response_cache = ResponseCache(cache_manager)

# Get cached response
cached = response_cache.get_response(query, params)

# Cache new response
response_cache.cache_response(query, params, response)
```

### 4. A/B Testing Framework
**Compare different retrieval strategies and configurations**

- **Experiment Management**: Create and manage A/B test experiments
- **Traffic Splitting**: Configurable traffic allocation between variants
- **Metrics Tracking**: Track performance metrics across variants
- **Statistical Analysis**: Compare variant performance with statistical significance
- **Experiment Lifecycle**: Draft → Active → Completed experiment states

**Key Components:**
- `rag/ab_testing.py` - A/B testing framework
- Experiment storage in `store/experiments/`
- Variant assignment and result tracking

**Usage:**
```python
# Create experiment
experiment_id = experiment_manager.create_experiment(
    name="Retriever Comparison",
    description="Compare BM25 vs FAISS retrieval",
    variants=[...],
    metrics=[...]
)

# Start experiment
experiment_manager.start_experiment(experiment_id)

# Assign user to variant
variant = experiment_manager.assign_variant(experiment_id, user_id)
```

### 5. Real-time Learning
**Feedback-based model improvement system**

- **Feedback Collection**: Collect user feedback (thumbs up/down, ratings, corrections)
- **Learning Signals**: Process feedback into actionable learning signals
- **Adaptive Retrieval**: Adjust document weights based on feedback
- **Query-specific Learning**: Learn query-document relevance patterns
- **Continuous Improvement**: Automatic model updates based on feedback

**Key Components:**
- `rag/feedback_learning.py` - Learning system implementation
- Feedback storage in `store/feedback/`
- Adaptive weight storage in `store/adaptive_weights/`

**Usage:**
```python
# Collect feedback
learning_system.collect_feedback(
    user_id, session_id, query, response, citations,
    feedback_type="thumbs_up", feedback_value=True
)

# Search with adaptive weights
results = learning_system.search(query, top_k=10)
```

### 6. Multi-language Support
**Internationalization capabilities with automatic translation**

- **Language Detection**: Automatic detection of query language
- **Translation Services**: Google Translate integration
- **Multilingual Embeddings**: Cross-lingual semantic search
- **Localization**: UI string localization for multiple languages
- **Auto-translation**: Automatic query/response translation

**Key Components:**
- `rag/multilingual.py` - Multi-language support
- Localization files in `locales/`
- Support for 20+ languages

**Usage:**
```python
# Process multilingual query
response = multilingual_system.process_query(
    query="¿Cómo puedo devolver un producto?",
    user_language="es",
    top_k=10
)
```

### 7. Advanced Analytics Dashboard
**Comprehensive system performance monitoring**

- **Performance Metrics**: Response times, query volumes, satisfaction rates
- **Usage Analytics**: Query patterns, peak usage times, user behavior
- **A/B Test Results**: Experiment performance comparison
- **Feedback Analysis**: User satisfaction trends and patterns
- **Real-time Monitoring**: Live dashboard with auto-refresh

**Key Components:**
- `analytics/dashboard.py` - Dashboard implementation
- Plotly/Dash-based interactive dashboard
- Comprehensive metrics collection and visualization

**Usage:**
```bash
# Start analytics dashboard
python analytics/dashboard.py --port 8050

# Visit http://localhost:8050 for dashboard
```

## 🛠️ Installation and Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Redis Setup (Optional but Recommended)
```bash
# Install Redis
# Ubuntu/Debian: sudo apt-get install redis-server
# macOS: brew install redis

# Start Redis server
redis-server

# Test Redis connection
redis-cli ping
```

### 3. Configuration
Update `configs/config.yaml` to enable desired features:

```yaml
features:
  multi_turn_conversations: true
  advanced_reranking: true
  redis_caching: true
  ab_testing: true
  real_time_learning: true
  multi_language_support: true
  analytics_dashboard: true
```

### 4. Start Enhanced API Server
```bash
python app/main_advanced.py
```

### 5. Start Enhanced Streamlit App
```bash
streamlit run app/streamlit_app_advanced.py
```

### 6. Start Analytics Dashboard
```bash
python analytics/dashboard.py
```

## 📊 API Endpoints

### Enhanced Response Generation
```http
POST /generate_response_advanced
Content-Type: application/json

{
  "query": "I need help with my order",
  "conversation_id": "uuid-here",
  "user_id": "user123",
  "language": "en",
  "auto_translate": true,
  "use_caching": true,
  "use_advanced_reranking": true,
  "enable_learning": true
}
```

### Feedback Collection
```http
POST /feedback
Content-Type: application/json

{
  "user_id": "user123",
  "query": "How do I return an item?",
  "response": "You can return items by...",
  "citations": [...],
  "feedback_type": "thumbs_up",
  "feedback_value": true
}
```

### Conversation Management
```http
# Create conversation
POST /conversations
{
  "user_id": "user123"
}

# Get conversation history
GET /conversations/{conversation_id}
```

### A/B Testing
```http
# List experiments
GET /experiments

# Create experiment
POST /experiments
{
  "name": "Retriever Test",
  "description": "Compare retrieval methods",
  "variants": [...],
  "metrics": [...]
}

# Get experiment results
GET /experiments/{experiment_id}/results
```

### System Monitoring
```http
# Enhanced health check
GET /health

# Cache statistics
GET /cache/stats

# Analytics summary
GET /analytics/summary

# Supported languages
GET /languages
```

## 🔧 Configuration Options

### Advanced Features Configuration
```yaml
advanced:
  conversations:
    enabled: true
    storage_path: "store/conversations"
    max_conversation_age_days: 30
    max_turns_in_context: 3
  
  reranking:
    enabled: true
    type: "cross_encoder"
    model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  
  caching:
    enabled: true
    type: "redis"
    redis:
      host: "localhost"
      port: 6379
      default_ttl: 3600
  
  ab_testing:
    enabled: true
    storage_path: "store/experiments"
  
  learning:
    enabled: true
    learning_interval_hours: 1
    min_feedback_for_update: 5
  
  multilingual:
    enabled: true
    auto_translate: true
    target_language: "en"
  
  analytics:
    enabled: true
    dashboard_port: 8050
```

## 📈 Performance Improvements

### Response Time Optimization
- **Caching**: 50-90% reduction in response time for cached queries
- **Advanced Reranking**: Better relevance with minimal latency increase
- **Adaptive Retrieval**: Improved relevance through learning

### Scalability Enhancements
- **Redis Caching**: Scales to multiple server instances
- **Async Processing**: Background tasks for non-critical operations
- **Efficient Storage**: Optimized data structures for conversation and feedback storage

### User Experience Improvements
- **Multi-turn Context**: More natural conversation flow
- **Multi-language Support**: Global accessibility
- **Real-time Learning**: Continuously improving responses
- **Rich Analytics**: Insights into system performance

## 🧪 A/B Testing Examples

### Retriever Comparison
```python
experiment = {
    "name": "Retriever Strategy Comparison",
    "variants": [
        {
            "name": "BM25 Control",
            "config": {"retriever": "bm25"},
            "traffic_percentage": 50.0,
            "is_control": True
        },
        {
            "name": "Dense Retrieval",
            "config": {"retriever": "dense"},
            "traffic_percentage": 50.0,
            "is_control": False
        }
    ],
    "metrics": [
        {"name": "response_time", "higher_is_better": False},
        {"name": "relevance_score", "higher_is_better": True}
    ]
}
```

### Reranking Strategy Test
```python
experiment = {
    "name": "Reranking Strategy Test",
    "variants": [
        {
            "name": "No Reranking",
            "config": {"use_reranker": False},
            "traffic_percentage": 33.3
        },
        {
            "name": "TF-IDF Reranking",
            "config": {"reranker_type": "tfidf"},
            "traffic_percentage": 33.3
        },
        {
            "name": "Cross-encoder Reranking",
            "config": {"reranker_type": "cross_encoder"},
            "traffic_percentage": 33.4
        }
    ]
}
```

## 🔍 Monitoring and Debugging

### Logging
All components include comprehensive logging:
- Request/response logging
- Performance metrics
- Error tracking
- Feature usage statistics

### Health Checks
Enhanced health check endpoint provides status for:
- Base retriever
- Advanced reranker
- Cache system
- Learning system
- Multilingual components
- A/B testing system

### Analytics Dashboard
Real-time monitoring of:
- Query volume and patterns
- Response time distribution
- User satisfaction metrics
- A/B test performance
- Cache hit rates
- Language usage statistics

## 🚨 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Ensure Redis server is running
   - Check Redis configuration in config.yaml
   - System falls back to memory cache automatically

2. **Cross-encoder Model Loading Failed**
   - Check internet connection for model download
   - Verify model name in configuration
   - System falls back to semantic similarity

3. **Translation Service Unavailable**
   - Check Google Translate API availability
   - Verify API keys if required
   - System operates in English-only mode as fallback

4. **Analytics Dashboard Not Loading**
   - Ensure all analytics dependencies are installed
   - Check port availability (default: 8050)
   - Verify log files exist and are readable

### Performance Optimization

1. **Enable Redis Caching**
   - Significantly improves response times
   - Reduces load on retrieval system
   - Scales across multiple instances

2. **Tune A/B Testing**
   - Start with simple experiments
   - Monitor statistical significance
   - Gradually increase complexity

3. **Optimize Learning System**
   - Adjust learning rate based on feedback volume
   - Set appropriate feedback thresholds
   - Monitor adaptive weight distribution

## 🔮 Future Enhancements

### Planned Features
- **Advanced Analytics**: Machine learning-based anomaly detection
- **Enhanced A/B Testing**: Bayesian optimization for experiment design
- **Improved Learning**: Reinforcement learning for response optimization
- **Extended Multilingual**: Support for more languages and dialects
- **Advanced Caching**: Semantic caching based on query similarity
- **Integration APIs**: Webhooks and external system integrations

### Experimental Features
- **Neural Reranking**: Custom neural network rerankers
- **Personalization**: User-specific response customization
- **Voice Support**: Speech-to-text and text-to-speech integration
- **Visual Analytics**: Advanced data visualization components

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Analytics Dashboard**: http://localhost:8050
- **Configuration Reference**: `configs/config.yaml`
- **Example Notebooks**: `examples/` directory
- **Performance Benchmarks**: `evaluation/` directory

## 🤝 Contributing

To contribute to the advanced features:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-enhancement`
3. Add comprehensive tests for new features
4. Update documentation
5. Submit pull request with detailed description

## 📄 License

This enhanced RAG system is available under the same license as the base system. See LICENSE file for details.
