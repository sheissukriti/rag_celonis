# Customer Support RAG Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

A comprehensive **Retrieval-Augmented Generation (RAG)** assistant for customer support, built for the AI Engineer hiring challenge. This system processes customer support queries using state-of-the-art retrieval and generation techniques with advanced features including multi-turn conversations, real-time learning, A/B testing, and multi-language support.

### 🎯 Challenge Requirements Fulfilled

✅ **RAG Implementation**: FAISS/BM25 vector database with multiple retrieval strategies  
✅ **LLM Integration**: Ollama-based response generation with fallback support  
✅ **Explainability**: Citation tracking and retrieval reasoning  
✅ **API Deployment**: FastAPI with comprehensive endpoints  
✅ **Response Logging**: Structured logging to JSON/CSV for analysis  
✅ **Evaluation Framework**: Automated quality assessment system  

### 🗂️ Dataset

**[Customer Support on Twitter (945k tweets)](https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k)**

This dataset contains customer queries and agent responses from Twitter customer support interactions, providing realistic training data for customer service scenarios.

### 🏗️ Architecture

```mermaid
flowchart TD
    A[User Query] -->|POST /generate_response| B[FastAPI Server]
    A -->|Streamlit UI| S[Streamlit App]
    S -->|API calls| B
    B --> C[Retriever]
    C -->|Top-K docs| D[Reranker]
    D -->|Filtered docs| E[Prompt Builder]
    E --> F[LLM (Ollama)]
    F --> G[Response + Citations]
    
    subgraph "Retrieval Options"
      C1[FAISS + TF-IDF-SVD]
      C2[BM25 + TF-IDF]
      C3[Dense Embeddings]
      C4[Hybrid Retrieval]
    end
    
    subgraph "Advanced Features"
      H[Multi-turn Conversations]
      I[Advanced Reranking]
      J[Redis Caching]
      K[A/B Testing]
      L[Real-time Learning]
      M[Multi-language Support]
      N[Analytics Dashboard]
    end
    
    subgraph "Evaluation System"
      O[Relevance Scoring]
      P[Faithfulness Check]
      Q[Coherence Analysis]
    end

    C <--> C1
    C <--> C2
    C <--> C3
    C <--> C4
    B <--> H
    B <--> I
    B <--> J
    B <--> K
    B <--> L
    B <--> M
    G --> O
    G --> P
    G --> Q

    subgraph "Logging & Monitoring"
      R[Response Logs]
      T[Performance Metrics]
      U[Error Tracking]
      V[Analytics Dashboard]
    end
    
    G --> R
    B --> T
    B --> U
    N --> V
```

## 🚀 Key Features

### Core RAG Features
- **Multi-Modal Retrieval**: FAISS + TF-IDF-SVD, BM25, dense embeddings, and hybrid retrieval
- **Intelligent Reranking**: TF-IDF-based and cross-encoder reranking
- **Comprehensive Evaluation**: Automated quality assessment with multiple metrics
- **Explainable AI**: Citation tracking and retrieval reasoning
- **Flexible LLM Backend**: Ollama integration with easy API swapping

### Advanced Features
- **Multi-turn Conversations**: Context-aware dialogue management
- **Advanced Reranking**: Cross-encoder models for superior relevance scoring
- **Redis-based Caching**: High-performance response caching
- **A/B Testing Framework**: Compare different retrieval strategies
- **Real-time Learning**: Feedback-based model improvement
- **Multi-language Support**: Internationalization with automatic translation
- **Analytics Dashboard**: Comprehensive system performance monitoring

### Production Features
- **Docker Support**: Complete containerization with docker-compose
- **Health Monitoring**: System health checks and component status
- **Error Handling**: Robust exception handling and graceful degradation
- **Response Logging**: Structured logging to JSONL for analysis
- **CORS Support**: Cross-origin resource sharing for frontend integration

## 📁 Project Structure

```
rag_celonis/
├── app/
│   ├── main.py                    # Basic FastAPI server
│   ├── main_advanced.py           # Enhanced FastAPI server with all features
│   ├── streamlit_app.py           # Basic Streamlit UI
│   ├── streamlit_app_advanced.py  # Enhanced Streamlit UI
│   └── streamlit_app_enhanced.py  # Alternative enhanced UI
├── rag/
│   ├── ingest.py                  # Dataset ingestion and preprocessing
│   ├── index.py                   # Dense embedding index builder
│   ├── index_tfidf.py             # TF-IDF-SVD + FAISS index builder
│   ├── retrieve.py                # Multi-strategy retrieval system
│   ├── hybrid_retriever.py        # Hybrid retrieval implementation
│   ├── rerank.py                  # Basic TF-IDF reranking
│   ├── advanced_rerank.py         # Cross-encoder reranking
│   ├── chain.py                   # Prompt engineering and context assembly
│   ├── embedder.py                # HuggingFace embedding wrapper
│   ├── evaluation.py              # Comprehensive evaluation framework
│   ├── conversation.py            # Multi-turn conversation management
│   ├── cache.py                   # Redis and memory caching
│   ├── ab_testing.py              # A/B testing framework
│   ├── feedback_learning.py       # Real-time learning system
│   └── multilingual.py            # Multi-language support
├── analytics/
│   └── dashboard.py               # Advanced analytics dashboard
├── scripts/
│   ├── setup.py                   # Automated setup script
│   └── run_evaluation.py          # Comprehensive evaluation runner
├── configs/
│   └── config.yaml                # Comprehensive configuration
├── store/                         # Generated artifacts and data
│   ├── experiments/               # A/B testing experiment data
│   ├── learning/                  # Learning system data
│   └── conversations/             # Conversation history
├── locales/                       # Localization files
├── logs/                          # Application and response logs
├── docker-compose.yml             # Multi-service Docker setup
├── Dockerfile                     # Container configuration
├── run.sh                         # Quick start script
├── run_advanced.sh                # Advanced system startup script
└── requirements.txt               # Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.11+** (tested with Python 3.11 and 3.12)
- **Git** for cloning the repository
- **Docker & Docker Compose** (optional, for containerized deployment)
- **Ollama** (optional, for local LLM inference)
- **Redis** (optional, for advanced caching features)

### Option 1: Quick Start with Docker 🐳

```bash
# Clone the repository
git clone <your-repo-url>
cd rag_celonis

# Start all services with Docker Compose
docker-compose up -d

# The services will be available at:
# - FastAPI: http://localhost:8000
# - Streamlit UI: http://localhost:8501
# - Ollama: http://localhost:11434
```

### Option 2: Quick Start Script 🚀

```bash
# Clone and enter the repository
git clone <your-repo-url>
cd rag_celonis

# Run the quick start script
./run.sh

# This will:
# - Set up virtual environment
# - Install dependencies
# - Create necessary directories
# - Download and process data
# - Start API server and Streamlit UI
```

### Option 3: Advanced System Setup 🔧

```bash
# Clone and enter the repository
git clone <your-repo-url>
cd rag_celonis

# Run the advanced startup script
./run_advanced.sh

# This starts all advanced features:
# - Enhanced API server with all features
# - Advanced Streamlit UI
# - Analytics dashboard
# - Redis caching (if available)
```

### Option 4: Manual Development Setup 💻

```bash
# Clone and enter the repository
git clone <your-repo-url>
cd rag_celonis

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs store evaluation data locales

# Run setup script
python scripts/setup.py --all --limit 10000

# Start basic system
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
streamlit run app/streamlit_app.py &
```

### Option 5: Ollama Setup (for LLM functionality)

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., Mistral 7B)
ollama pull mistral

# Start Ollama server (if not already running)
ollama serve
```

### Option 6: Redis Setup (for Advanced Caching)

```bash
# Install Redis
# Ubuntu/Debian: sudo apt-get install redis-server
# macOS: brew install redis

# Start Redis server
redis-server

# Test Redis connection
redis-cli ping
```

## 🚀 Usage

### 1. Data Preparation

```bash
# Ingest and preprocess the dataset
python -m rag.ingest --limit 10000 --out_qa store/qa_texts_10k.jsonl

# Build TF-IDF + FAISS index (recommended)
python -m rag.index_tfidf store/qa_texts_10k.jsonl store/faiss

# Alternative: Build dense embeddings index
python -m rag.index store/qa_texts_10k.jsonl store/dense store/faiss.index sentence-transformers/all-MiniLM-L6-v2
```

### 2. Start the System

#### Basic System
```bash
# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Launch Streamlit UI (in separate terminal)
streamlit run app/streamlit_app.py
```

#### Advanced System
```bash
# Start enhanced API server
python app/main_advanced.py

# Launch enhanced Streamlit UI (in separate terminal)
streamlit run app/streamlit_app_advanced.py

# Start analytics dashboard (in separate terminal)
python analytics/dashboard.py
```

## 📊 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and API info |
| `/health` | GET | Detailed system health status |
| `/generate_response` | POST | Generate response to customer query |
| `/evaluate` | POST | Run system evaluation with test queries |
| `/test-queries` | GET | Get list of predefined test queries |

### Advanced Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate_response_advanced` | POST | Enhanced response generation with all features |
| `/conversations` | POST/GET | Conversation management |
| `/feedback` | POST | Collect user feedback |
| `/experiments` | GET/POST | A/B testing management |
| `/cache/stats` | GET | Cache statistics |
| `/analytics/summary` | GET | Analytics summary |
| `/languages` | GET | Supported languages |

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Basic response generation
curl -X POST "http://localhost:8000/generate_response" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I ordered a laptop but it arrived with a broken screen. What should I do?",
    "top_k": 10,
    "max_tokens": 256
  }'

# Advanced response generation
curl -X POST "http://localhost:8000/generate_response_advanced" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I need help with my order",
    "conversation_id": "uuid-here",
    "user_id": "user123",
    "language": "en",
    "auto_translate": true,
    "use_caching": true,
    "use_advanced_reranking": true,
    "enable_learning": true
  }'

# Collect feedback
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "query": "How do I return an item?",
    "response": "You can return items by...",
    "feedback_type": "thumbs_up",
    "feedback_value": true
  }'

# Run evaluation
curl -X POST "http://localhost:8000/evaluate"
```

### Response Format

```json
{
  "answer": "I understand your frustration with receiving a damaged laptop. Here's what you should do: [Doc 1] Contact our returns department immediately to report the damage. [Doc 2] Take photos of the damaged screen for documentation. We'll arrange for a replacement or full refund within 5-7 business days.",
  "citations": [
    {
      "id": 1234,
      "score": 0.95,
      "text": "Customer: My laptop arrived broken\n\nAgent: I'm sorry to hear about the damaged laptop. Please contact our returns department..."
    }
  ],
  "response_time_seconds": 2.3,
  "retriever_type": "faiss_tfidf",
  "query_processed": "I ordered a laptop but it arrived with a broken screen. What should I do?"
}
```

## 🔧 Configuration

Edit `configs/config.yaml` to customize the system:

```yaml
app:
  # LLM Configuration
  model: mistral                    # Ollama model name
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  
  # Retrieval Configuration
  retriever: faiss_tfidf           # Options: bm25, dense, faiss_tfidf, hybrid
  top_k: 10                        # Documents to retrieve
  top_k_context: 5                 # Documents to use in prompt
  use_reranker: true               # Enable TF-IDF reranking
  
  # Generation Parameters
  max_tokens: 512
  temperature: 0.7
  
  # Logging
  log_responses: true
  log_file: "logs/responses.jsonl"

# Advanced Features Configuration
advanced:
  # Multi-turn Conversations
  conversations:
    enabled: true
    storage_path: "store/conversations"
    max_conversation_age_days: 30
    max_turns_in_context: 3
  
  # Advanced Reranking
  reranking:
    enabled: true
    type: "cross_encoder"  # Options: cross_encoder, semantic, hybrid
    model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  
  # Caching Configuration
  caching:
    enabled: true
    type: "redis"  # Options: redis, memory
    redis:
      host: "localhost"
      port: 6379
      default_ttl: 3600
  
  # A/B Testing
  ab_testing:
    enabled: true
    storage_path: "store/experiments"
  
  # Real-time Learning
  learning:
    enabled: true
    learning_interval_hours: 1
    min_feedback_for_update: 5
  
  # Multi-language Support
  multilingual:
    enabled: true
    auto_translate: true
    target_language: "en"
  
  # Analytics
  analytics:
    enabled: true
    dashboard_port: 8050

# Feature Flags
features:
  multi_turn_conversations: true
  advanced_reranking: true
  redis_caching: true
  ab_testing: true
  real_time_learning: true
  multi_language_support: true
  analytics_dashboard: true
```

## 📈 Advanced Features

### 1. Multi-turn Conversations
**Context-aware dialogue management that maintains conversation history**

- **Conversation Management**: Persistent conversation storage with automatic cleanup
- **Context-aware Prompts**: Incorporates previous turns for better responses
- **Session Tracking**: User and session-based conversation tracking
- **Conversation Export**: Export conversation history as JSON

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
```

### 2. Advanced Reranking
**Cross-encoder models for superior relevance scoring**

- **Cross-encoder Reranking**: Uses transformer models for pairwise relevance scoring
- **Semantic Similarity**: Alternative embedding-based reranking
- **Hybrid Reranking**: Combines multiple reranking strategies
- **Configurable Models**: Support for different cross-encoder models

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

**Access:**
```bash
# Start analytics dashboard
python analytics/dashboard.py --port 8050

# Visit http://localhost:8050 for dashboard
```

## 📊 Evaluation Framework

The system includes a comprehensive evaluation framework that assesses:

### Metrics Evaluated

1. **Retrieval Quality**
   - Precision: Relevant docs / Retrieved docs
   - Recall: Relevant docs / Total relevant docs  
   - F1-Score: Harmonic mean of precision and recall

2. **Answer Quality**
   - **Relevance**: Semantic similarity between query and answer
   - **Coherence**: Fluency and logical structure of response
   - **Faithfulness**: Alignment between answer and source documents

3. **System Performance**
   - Response time
   - Citation count and quality
   - Answer length and completeness

### Running Evaluation

```bash
# Via API
curl -X POST "http://localhost:8000/evaluate"

# Via script
python scripts/run_evaluation.py

# Via Python
python -c "
from rag.evaluation import RAGEvaluator, create_test_queries
evaluator = RAGEvaluator()
test_queries = create_test_queries()
print(f'Created {len(test_queries)} test queries')
"
```

### Sample Test Queries

The system includes 10 predefined test queries covering common customer support scenarios:

- Product damage/returns
- Account access issues  
- Warranty questions
- Shipping inquiries
- Technical support
- Policy questions

## 🔍 Explainability Features

### Citation Tracking
- Each response includes numbered citations `[Doc 1]`, `[Doc 2]`
- Full source text provided for transparency
- Retrieval scores included for confidence assessment

### Retrieval Reasoning
- Multiple retrieval strategies with configurable weights
- Semantic similarity scores for each retrieved document
- Reranking explanations when enabled

### Evaluation Transparency
- Detailed metrics for each component (retrieval, generation, overall)
- Per-query evaluation results
- Aggregate statistics across test sets

## 🐳 Docker Deployment

### Services Included

- **rag-api**: Main FastAPI application
- **streamlit**: Interactive web UI  
- **ollama**: Local LLM inference (with GPU support)

### Environment Variables

```bash
# .env file (optional)
PYTHONPATH=/app
OLLAMA_HOST=http://ollama:11434
LOG_LEVEL=INFO
```

### Production Deployment

```bash
# Build and start services
docker-compose up -d --build

# Scale API instances
docker-compose up -d --scale rag-api=3

# View logs
docker-compose logs -f rag-api

# Health check
curl http://localhost:8000/health
```

## 📊 Monitoring & Logging

### Log Files

- `logs/app.log`: Application logs with structured format
- `logs/responses.jsonl`: All query-response pairs with metadata
- `store/experiments/`: A/B testing experiment results
- `store/feedback/`: User feedback data
- `store/conversations/`: Conversation history

### Log Format

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "query": "How do I reset my password?",
  "answer": "To reset your password, please follow these steps...",
  "citations": [...],
  "response_time_seconds": 1.45,
  "retriever_type": "faiss_tfidf",
  "model": "mistral"
}
```

## 🧪 Testing

### Test Queries

Run the predefined test suite:

```bash
# Get test queries
curl http://localhost:8000/test-queries

# Run full evaluation
curl -X POST http://localhost:8000/evaluate

# Run evaluation script
python scripts/run_evaluation.py
```

### Custom Testing

```python
from rag.evaluation import RAGEvaluator

evaluator = RAGEvaluator()

# Custom test data
test_data = [
    {
        "query": "Your custom query",
        "answer": "Generated answer",
        "citations": [...],
        "response_time": 1.2
    }
]

metrics = evaluator.evaluate_batch(test_data)
print(f"Overall score: {metrics['avg_overall_score']:.3f}")
```

## 🚧 Troubleshooting

### Common Issues

**1. Ollama Connection Failed**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama
ollama serve

# Pull required model
ollama pull mistral
```

**2. FAISS Index Not Found**
```bash
# Build the index
python -m rag.index_tfidf store/qa_texts_10k.jsonl store/faiss
```

**3. Empty Retrieval Results**
```bash
# Check if data files exist
ls -la store/
# Re-run data ingestion if needed
python -m rag.ingest --limit 10000
```

**4. Memory Issues**
```bash
# Reduce dataset size in config
# Use faiss_tfidf instead of dense retrieval
# Increase Docker memory limits
```

**5. LZMA Module Missing (Python compilation issue)**
```bash
# On macOS with Homebrew:
brew install xz
# Then rebuild Python (if using pyenv):
pyenv install 3.12.7
pyenv local 3.12.7

# Alternative: Use Docker deployment (recommended)
docker-compose up -d
```

**6. Redis Connection Failed**
- Ensure Redis server is running
- Check Redis configuration in config.yaml
- System falls back to memory cache automatically

**7. Cross-encoder Model Loading Failed**
- Check internet connection for model download
- Verify model name in configuration
- System falls back to semantic similarity

**8. Translation Service Unavailable**
- Check Google Translate API availability
- Verify API keys if required
- System operates in English-only mode as fallback

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

## 📋 System Status & Issue Resolution

### Current System Capabilities

The system is **fully functional** for the AI Engineer Challenge requirements:

✅ **RAG Implementation**: Multi-strategy retrieval working  
✅ **LLM Integration**: Ollama integration with fallback messages  
✅ **API Deployment**: FastAPI server with all endpoints  
✅ **Explainability**: Citations and retrieval reasoning  
✅ **Response Logging**: Structured logging to JSONL  
✅ **Error Handling**: Graceful degradation and clear error messages  
✅ **Advanced Features**: All advanced features implemented and working
✅ **Production Ready**: Docker support, monitoring, and comprehensive documentation

### Performance Metrics

- **Response Time**: Typically 2-30 seconds (includes retrieval + generation)
- **Retrieval Quality**: Finding relevant documents with good scores
- **Citation Quality**: 5+ relevant documents retrieved and cited per query
- **API Latency**: Fast response for health checks and simple endpoints
- **Scalability**: Supports multiple concurrent users with caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/transformers/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search  
- [Ollama](https://ollama.ai/) for local LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Streamlit](https://streamlit.io/) for the interactive UI
- [Redis](https://redis.io/) for high-performance caching
- [Plotly & Dash](https://plotly.com/) for analytics dashboard

---

**Built for the AI Engineer Challenge** 🚀

*This project demonstrates production-ready RAG implementation with comprehensive evaluation, monitoring, deployment capabilities, and advanced features including multi-turn conversations, real-time learning, A/B testing, and multi-language support.*

## 🚀 Quick Start Summary

Choose your preferred method to get started:

1. **Docker (Easiest)**: `docker-compose up -d`
2. **Quick Script**: `./run.sh`
3. **Advanced Features**: `./run_advanced.sh`
4. **Manual Setup**: Follow the detailed installation guide above

**Access Points:**
- 🌐 **Web Interface**: http://localhost:8501
- 📚 **API Documentation**: http://localhost:8000/docs
- 📊 **Analytics Dashboard**: http://localhost:8050 (advanced mode)
- 🔧 **API Server**: http://localhost:8000

**Ready for submission and live demo!** 🎉