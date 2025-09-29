# Customer Support RAG Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

A comprehensive **Retrieval-Augmented Generation (RAG)** assistant for customer support, built for the AI Engineer hiring challenge. This system processes customer support queries using state-of-the-art retrieval and generation techniques.

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
    end
    
    subgraph "Evaluation System"
      H[Relevance Scoring]
      I[Faithfulness Check]
      J[Coherence Analysis]
    end

    C <--> C1
    C <--> C2
    C <--> C3
    G --> H
    G --> I
    G --> J

    subgraph "Logging & Monitoring"
      K[Response Logs]
      L[Performance Metrics]
      M[Error Tracking]
    end
    
    G --> K
    B --> L
    B --> M
```

### 🚀 Key Features

- **Multi-Modal Retrieval**: FAISS + TF-IDF-SVD, BM25, and dense embeddings
- **Intelligent Reranking**: TF-IDF-based relevance reranking
- **Comprehensive Evaluation**: Automated quality assessment with multiple metrics
- **Production Ready**: Docker support, logging, error handling, and monitoring
- **Explainable AI**: Citation tracking and retrieval reasoning
- **Flexible LLM Backend**: Ollama integration with easy API swapping

## 📁 Project Structure

```
rag_c-1/
├── app/
│   ├── main.py              # FastAPI server with comprehensive endpoints
│   └── streamlit_app.py     # Interactive Streamlit UI
├── rag/
│   ├── ingest.py           # Dataset ingestion and preprocessing
│   ├── index.py            # Dense embedding index builder
│   ├── index_tfidf.py      # TF-IDF-SVD + FAISS index builder
│   ├── retrieve.py         # Multi-strategy retrieval system
│   ├── rerank.py           # TF-IDF reranking implementation
│   ├── chain.py            # Prompt engineering and context assembly
│   ├── embedder.py         # HuggingFace embedding wrapper
│   └── evaluation.py       # Comprehensive evaluation framework
├── configs/
│   └── config.yaml         # Runtime configuration
├── data/
│   └── preprocessed_data.json  # Local dataset cache
├── store/                  # Generated artifacts (indices, embeddings)
├── logs/                   # Application and response logs
├── docker-compose.yml      # Multi-service Docker setup
├── Dockerfile             # Container configuration
└── requirements.txt        # Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.11+** (tested with Python 3.11 and 3.12)
- **Git** for cloning the repository
- **Docker & Docker Compose** (optional, for containerized deployment)
- **Ollama** (optional, for local LLM inference)

### Option 1: Quick Start with Docker 🐳

```bash
# Clone the repository
git clone <your-repo-url>
cd rag_c-1

# Start all services with Docker Compose
docker-compose up -d

# The services will be available at:
# - FastAPI: http://localhost:8000
# - Streamlit UI: http://localhost:8501
# - Ollama: http://localhost:11434
```

### Option 2: Local Development Setup 💻

```bash
# Clone and enter the repository
git clone <your-repo-url>
cd rag_c-1

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs store evaluation data
```

### Option 3: Ollama Setup (for LLM functionality)

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., Mistral 7B)
ollama pull mistral

# Start Ollama server (if not already running)
ollama serve
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

### 2. Start the API Server

```bash
# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 3. Launch Streamlit UI

```bash
# In a separate terminal
streamlit run app/streamlit_app.py

# UI will be available at http://localhost:8501
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

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Generate response
curl -X POST "http://localhost:8000/generate_response" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I ordered a laptop but it arrived with a broken screen. What should I do?",
    "top_k": 10,
    "max_tokens": 256
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
  retriever: faiss_tfidf           # Options: bm25, dense, faiss_tfidf
  top_k: 10                        # Documents to retrieve
  top_k_context: 5                 # Documents to use in prompt
  use_reranker: true               # Enable TF-IDF reranking
  
  # Generation Parameters
  max_tokens: 512
  temperature: 0.7
  
  # Logging
  log_responses: true
  log_file: "logs/responses.jsonl"
```

## 📈 Evaluation Framework

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
- `evaluation/results/`: Evaluation results and metrics

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

## 🔮 Future Enhancements

- [ ] **Multi-turn Conversations**: Context-aware dialogue management
- [ ] **Advanced Reranking**: Cross-encoder models for better relevance
- [ ] **Caching Layer**: Redis-based response caching
- [ ] **A/B Testing**: Compare different retrieval strategies
- [ ] **Real-time Learning**: Feedback-based model improvement
- [ ] **Multi-language Support**: Internationalization capabilities
- [ ] **Advanced Analytics**: Dashboard for system performance monitoring

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

---

**Built for the AI Engineer Challenge** 🚀

*This project demonstrates production-ready RAG implementation with comprehensive evaluation, monitoring, and deployment capabilities.*
