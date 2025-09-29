#!/usr/bin/env python3
"""
Comprehensive evaluation script for the RAG system.
Runs evaluation on test queries and generates detailed reports.
"""

import argparse
import json
import yaml
import asyncio
import httpx
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_evaluation(api_url: str = "http://localhost:8000", output_dir: str = "evaluation/results"):
    """Run comprehensive evaluation of the RAG system."""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            # Get test queries
            logger.info("Fetching test queries...")
            response = await client.get(f"{api_url}/test-queries")
            response.raise_for_status()
            test_data = response.json()
            test_queries = test_data["test_queries"]
            
            logger.info(f"Running evaluation on {len(test_queries)} test queries")
            
            # Run system evaluation
            logger.info("Running system evaluation...")
            eval_response = await client.post(f"{api_url}/evaluate")
            eval_response.raise_for_status()
            evaluation_results = eval_response.json()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results
            results_file = Path(output_dir) / f"evaluation_results_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            
            # Save summary report
            summary_file = Path(output_dir) / f"evaluation_summary_{timestamp}.md"
            generate_summary_report(evaluation_results, summary_file)
            
            logger.info(f"Evaluation completed successfully!")
            logger.info(f"Results saved to: {results_file}")
            logger.info(f"Summary saved to: {summary_file}")
            
            # Print key metrics
            metrics = evaluation_results.get("metrics", {})
            print("\n" + "="*50)
            print("EVALUATION SUMMARY")
            print("="*50)
            print(f"Test Queries: {evaluation_results.get('test_queries_count', 0)}")
            print(f"Overall Score: {metrics.get('avg_overall_score', 0):.3f}")
            print(f"Relevance Score: {metrics.get('avg_relevance_score', 0):.3f}")
            print(f"Coherence Score: {metrics.get('avg_coherence_score', 0):.3f}")
            print(f"Faithfulness Score: {metrics.get('avg_faithfulness_score', 0):.3f}")
            print(f"Avg Response Time: {metrics.get('avg_response_time', 0):.2f}s")
            print("="*50)
            
            return evaluation_results
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

def generate_summary_report(evaluation_results: Dict, output_path: Path):
    """Generate a markdown summary report."""
    
    metrics = evaluation_results.get("metrics", {})
    timestamp = evaluation_results.get("timestamp", datetime.now().isoformat())
    
    report = f"""# RAG System Evaluation Report

**Generated:** {timestamp}  
**Test Queries:** {evaluation_results.get('test_queries_count', 0)}

## Overall Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **Overall Score** | {metrics.get('avg_overall_score', 0):.3f} | Weighted combination of all metrics |
| **Relevance** | {metrics.get('avg_relevance_score', 0):.3f} | How well answers match queries |
| **Coherence** | {metrics.get('avg_coherence_score', 0):.3f} | Fluency and logical structure |
| **Faithfulness** | {metrics.get('avg_faithfulness_score', 0):.3f} | Alignment with source documents |

## Retrieval Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **Precision** | {metrics.get('avg_retrieval_precision', 0):.3f} | Relevant docs / Retrieved docs |
| **Recall** | {metrics.get('avg_retrieval_recall', 0):.3f} | Relevant docs / Total relevant |
| **F1-Score** | {metrics.get('avg_retrieval_f1', 0):.3f} | Harmonic mean of precision/recall |

## System Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Avg Response Time** | {metrics.get('avg_response_time', 0):.2f}s | Time to generate response |
| **Avg Citations** | {metrics.get('avg_num_citations', 0):.1f} | Number of source documents used |
| **Avg Answer Length** | {metrics.get('avg_answer_length', 0):.0f} words | Length of generated responses |

## Score Distribution

- **Overall Score Std Dev:** {metrics.get('std_overall_score', 0):.3f}
- **Relevance Score Std Dev:** {metrics.get('std_relevance_score', 0):.3f}
- **Coherence Score Std Dev:** {metrics.get('std_coherence_score', 0):.3f}
- **Faithfulness Score Std Dev:** {metrics.get('std_faithfulness_score', 0):.3f}

## Recommendations

"""

    # Add recommendations based on scores
    overall_score = metrics.get('avg_overall_score', 0)
    if overall_score >= 0.8:
        report += "✅ **Excellent Performance**: System is performing very well across all metrics.\n\n"
    elif overall_score >= 0.6:
        report += "⚠️ **Good Performance**: System is working well but has room for improvement.\n\n"
    else:
        report += "❌ **Needs Improvement**: System requires optimization in multiple areas.\n\n"

    # Specific recommendations
    if metrics.get('avg_retrieval_f1', 0) < 0.5:
        report += "- **Improve Retrieval**: Consider tuning retrieval parameters or trying different embedding models\n"
    
    if metrics.get('avg_coherence_score', 0) < 0.6:
        report += "- **Enhance Generation**: Consider using a more powerful LLM or improving prompt templates\n"
    
    if metrics.get('avg_faithfulness_score', 0) < 0.7:
        report += "- **Increase Faithfulness**: Review prompt engineering to ensure better adherence to source material\n"
    
    if metrics.get('avg_response_time', 0) > 5.0:
        report += "- **Optimize Performance**: Consider caching, smaller models, or infrastructure improvements\n"

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

async def test_individual_queries(api_url: str, queries: List[str]):
    """Test individual queries and show detailed results."""
    
    async with httpx.AsyncClient(timeout=60) as client:
        for i, query in enumerate(queries, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Query: {query}")
            
            try:
                response = await client.post(f"{api_url}/generate_response", json={
                    "query": query,
                    "top_k": 5,
                    "max_tokens": 256
                })
                response.raise_for_status()
                result = response.json()
                
                print(f"Answer: {result['answer'][:200]}...")
                print(f"Citations: {len(result['citations'])}")
                print(f"Response Time: {result['response_time_seconds']:.2f}s")
                
                # Show top citation
                if result['citations']:
                    top_citation = result['citations'][0]
                    print(f"Top Citation (Score: {top_citation['score']:.3f}): {top_citation['text'][:100]}...")
                
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output-dir", default="evaluation/results", help="Output directory for results")
    parser.add_argument("--test-individual", action="store_true", help="Test individual queries interactively")
    
    args = parser.parse_args()
    
    if args.test_individual:
        # Test individual queries
        test_queries = [
            "I ordered a laptop but it arrived with a broken screen. What should I do?",
            "I need help resetting my password",
            "My cat chewed my phone charger. Is this covered under warranty?",
            "How long does shipping usually take?",
            "Can I cancel my order?"
        ]
        asyncio.run(test_individual_queries(args.api_url, test_queries))
    else:
        # Run full evaluation
        asyncio.run(run_evaluation(args.api_url, args.output_dir))

if __name__ == "__main__":
    main()
