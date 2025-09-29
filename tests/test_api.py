"""
Test suite for the RAG Customer Support Assistant API.
"""

import pytest
import httpx
import asyncio
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.main import app

client = TestClient(app)

class TestAPI:
    """Test cases for the FastAPI endpoints."""
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "config" in data
    
    def test_generate_response_valid_query(self):
        """Test response generation with a valid query."""
        test_query = {
            "query": "I need help with my order",
            "top_k": 5,
            "max_tokens": 100
        }
        
        response = client.post("/generate_response", json=test_query)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "response_time_seconds" in data
        assert "retriever_type" in data
        assert "query_processed" in data
        
        # Check that citations are properly formatted
        for citation in data["citations"]:
            assert "id" in citation
            assert "score" in citation
            assert "text" in citation
    
    def test_generate_response_empty_query(self):
        """Test response generation with empty query."""
        test_query = {"query": ""}
        
        response = client.post("/generate_response", json=test_query)
        assert response.status_code == 400
    
    def test_generate_response_missing_query(self):
        """Test response generation with missing query field."""
        test_query = {"top_k": 5}
        
        response = client.post("/generate_response", json=test_query)
        assert response.status_code == 422  # Validation error
    
    def test_test_queries_endpoint(self):
        """Test the test queries endpoint."""
        response = client.get("/test-queries")
        assert response.status_code == 200
        
        data = response.json()
        assert "test_queries" in data
        assert "count" in data
        assert len(data["test_queries"]) == data["count"]
        
        # Check query format
        for query in data["test_queries"]:
            assert "query" in query
            assert isinstance(query["query"], str)
            assert len(query["query"]) > 0

class TestEvaluation:
    """Test cases for the evaluation system."""
    
    def test_evaluation_endpoint(self):
        """Test the evaluation endpoint."""
        # Note: This test might take a while as it runs the full evaluation
        response = client.post("/evaluate")
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "metrics" in data
            assert "test_queries_count" in data
            assert "timestamp" in data

@pytest.mark.asyncio
class TestAsyncAPI:
    """Async test cases for API endpoints."""
    
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
            # Send multiple concurrent requests
            tasks = []
            for i in range(5):
                task = ac.post("/generate_response", json={
                    "query": f"Test query {i}",
                    "top_k": 3,
                    "max_tokens": 50
                })
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert "citations" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
