#!/usr/bin/env python3
"""
Setup script for the RAG Customer Support Assistant.
Downloads data, builds indices, and prepares the system for use.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_dataset(output_path: str = "data/preprocessed_data.json"):
    """Download the preprocessed dataset from HuggingFace."""
    logger.info("Downloading dataset from HuggingFace...")
    
    # Create data directory
    Path(output_path).parent.mkdir(exist_ok=True)
    
    # Dataset URL (using the specific revision mentioned in original code)
    rev = 'edd299bba12cb076030822ea97b4329d6041f218'
    url = f"https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k/resolve/{rev}/preprocessed_data.json?download=true"
    
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Dataset downloaded to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False

def ingest_data(limit: int = 10000):
    """Ingest and preprocess the dataset."""
    logger.info(f"Ingesting dataset with limit={limit}")
    
    try:
        from rag.ingest import main as ingest_main
        ingest_main(
            out_input='store/input_texts.jsonl',
            out_qa='store/qa_texts.jsonl',
            limit=limit
        )
        
        # Create subset files for development
        logger.info("Creating subset files for development...")
        
        # Create 10k subset
        subprocess.run([
            'head', '-n', '10000', 'store/input_texts.jsonl'
        ], stdout=open('store/input_texts_10k.jsonl', 'w'), check=True)
        
        subprocess.run([
            'head', '-n', '10000', 'store/qa_texts.jsonl'
        ], stdout=open('store/qa_texts_10k.jsonl', 'w'), check=True)
        
        # Create 2k subset for quick testing
        subprocess.run([
            'head', '-n', '2000', 'store/input_texts.jsonl'
        ], stdout=open('store/input_texts_2k.jsonl', 'w'), check=True)
        
        subprocess.run([
            'head', '-n', '2000', 'store/qa_texts.jsonl'
        ], stdout=open('store/qa_texts_2k.jsonl', 'w'), check=True)
        
        logger.info("Data ingestion completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return False

def build_indices():
    """Build all required indices."""
    logger.info("Building indices...")
    
    try:
        # Build TF-IDF + FAISS index (recommended)
        logger.info("Building TF-IDF + FAISS index...")
        from rag.index_tfidf import main as index_tfidf_main
        index_tfidf_main(
            input_path='store/input_texts_10k.jsonl',
            qa_path='store/qa_texts_10k.jsonl',
            output_prefix='store/faiss',
            n_components=256
        )
        
        logger.info("Index building completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        return False

def setup_ollama():
    """Setup instructions for Ollama."""
    logger.info("Ollama setup instructions:")
    print("\n" + "="*50)
    print("OLLAMA SETUP")
    print("="*50)
    print("1. Install Ollama:")
    print("   curl -fsSL https://ollama.ai/install.sh | sh")
    print("\n2. Pull a model:")
    print("   ollama pull mistral")
    print("\n3. Start Ollama server:")
    print("   ollama serve")
    print("\n4. Test the model:")
    print("   ollama run mistral")
    print("="*50)

def verify_setup():
    """Verify that all components are properly set up."""
    logger.info("Verifying setup...")
    
    required_files = [
        'configs/config.yaml',
        'store/qa_texts_10k.jsonl',
        'store/faiss.index',
        'store/faiss.ids'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    logger.info("✅ Setup verification passed!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup RAG Customer Support Assistant")
    parser.add_argument("--download-data", action="store_true", help="Download dataset")
    parser.add_argument("--ingest-data", action="store_true", help="Ingest and preprocess data")
    parser.add_argument("--build-indices", action="store_true", help="Build search indices")
    parser.add_argument("--setup-ollama", action="store_true", help="Show Ollama setup instructions")
    parser.add_argument("--verify", action="store_true", help="Verify setup")
    parser.add_argument("--all", action="store_true", help="Run complete setup")
    parser.add_argument("--limit", type=int, default=10000, help="Limit number of documents to process")
    
    args = parser.parse_args()
    
    if args.all:
        # Run complete setup
        logger.info("Running complete setup...")
        
        # Create directories
        for directory in ['logs', 'store', 'evaluation', 'data']:
            Path(directory).mkdir(exist_ok=True)
        
        success = True
        
        # Download data
        if not download_dataset():
            success = False
        
        # Ingest data
        if success and not ingest_data(args.limit):
            success = False
        
        # Build indices
        if success and not build_indices():
            success = False
        
        # Verify setup
        if success and not verify_setup():
            success = False
        
        if success:
            logger.info("🎉 Complete setup finished successfully!")
            setup_ollama()
            print("\nYou can now start the system with:")
            print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
            print("  streamlit run app/streamlit_app.py")
        else:
            logger.error("❌ Setup failed. Please check the logs above.")
            sys.exit(1)
    
    else:
        # Run individual steps
        if args.download_data:
            download_dataset()
        
        if args.ingest_data:
            ingest_data(args.limit)
        
        if args.build_indices:
            build_indices()
        
        if args.setup_ollama:
            setup_ollama()
        
        if args.verify:
            verify_setup()

if __name__ == "__main__":
    main()
