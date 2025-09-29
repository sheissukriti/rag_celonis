#!/bin/bash

# Advanced RAG System Startup Script
# This script starts all components of the enhanced RAG system

set -e  # Exit on any error

echo "🚀 Starting Advanced RAG System..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}Port $port is already in use${NC}"
        return 1
    else
        echo -e "${GREEN}Port $port is available${NC}"
        return 0
    fi
}

# Function to check if Redis is running
check_redis() {
    if redis-cli ping >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Redis is running${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Redis is not running. Starting Redis...${NC}"
        # Try to start Redis
        if command -v redis-server >/dev/null 2>&1; then
            redis-server --daemonize yes
            sleep 2
            if redis-cli ping >/dev/null 2>&1; then
                echo -e "${GREEN}✅ Redis started successfully${NC}"
                return 0
            else
                echo -e "${YELLOW}⚠️  Could not start Redis. Will use memory cache as fallback${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}⚠️  Redis not installed. Will use memory cache as fallback${NC}"
            return 1
        fi
    fi
}

# Function to install dependencies if needed
check_dependencies() {
    echo -e "${BLUE}📦 Checking dependencies...${NC}"
    
    if ! python -c "import streamlit" >/dev/null 2>&1; then
        echo -e "${YELLOW}Installing missing dependencies...${NC}"
        pip install -r requirements.txt
    fi
    
    echo -e "${GREEN}✅ Dependencies checked${NC}"
}

# Function to create necessary directories
create_directories() {
    echo -e "${BLUE}📁 Creating necessary directories...${NC}"
    
    directories=(
        "logs"
        "store"
        "store/conversations"
        "store/experiments" 
        "store/feedback"
        "store/learning"
        "store/adaptive_weights"
        "locales"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    echo -e "${GREEN}✅ Directories created${NC}"
}

# Function to start the API server
start_api_server() {
    echo -e "${BLUE}🔧 Starting Enhanced API Server...${NC}"
    
    if check_port 8000; then
        echo "Starting API server on port 8000..."
        python app/main_advanced.py &
        API_PID=$!
        echo $API_PID > .api_pid
        
        # Wait for server to start
        echo "Waiting for API server to start..."
        sleep 5
        
        # Check if server is responding
        if curl -s http://localhost:8000/health >/dev/null; then
            echo -e "${GREEN}✅ API Server started successfully${NC}"
            echo -e "   📍 API Documentation: http://localhost:8000/docs"
        else
            echo -e "${RED}❌ API Server failed to start${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Cannot start API server - port 8000 is in use${NC}"
        return 1
    fi
}

# Function to start the Streamlit app
start_streamlit_app() {
    echo -e "${BLUE}🎨 Starting Enhanced Streamlit App...${NC}"
    
    if check_port 8501; then
        echo "Starting Streamlit app on port 8501..."
        streamlit run app/streamlit_app_advanced.py --server.port 8501 &
        STREAMLIT_PID=$!
        echo $STREAMLIT_PID > .streamlit_pid
        
        echo -e "${GREEN}✅ Streamlit App started successfully${NC}"
        echo -e "   📍 Web Interface: http://localhost:8501"
    else
        echo -e "${RED}❌ Cannot start Streamlit app - port 8501 is in use${NC}"
        return 1
    fi
}

# Function to start the analytics dashboard
start_analytics_dashboard() {
    echo -e "${BLUE}📊 Starting Analytics Dashboard...${NC}"
    
    if check_port 8050; then
        echo "Starting analytics dashboard on port 8050..."
        python analytics/dashboard.py --port 8050 &
        DASHBOARD_PID=$!
        echo $DASHBOARD_PID > .dashboard_pid
        
        echo -e "${GREEN}✅ Analytics Dashboard started successfully${NC}"
        echo -e "   📍 Analytics Dashboard: http://localhost:8050"
    else
        echo -e "${RED}❌ Cannot start Analytics Dashboard - port 8050 is in use${NC}"
        return 1
    fi
}

# Function to display status
show_status() {
    echo ""
    echo -e "${BLUE}🔍 System Status${NC}"
    echo "==============="
    
    # Check API server
    if curl -s http://localhost:8000/health >/dev/null; then
        echo -e "${GREEN}✅ API Server: Running${NC} (http://localhost:8000)"
    else
        echo -e "${RED}❌ API Server: Not responding${NC}"
    fi
    
    # Check Streamlit
    if curl -s http://localhost:8501 >/dev/null; then
        echo -e "${GREEN}✅ Streamlit App: Running${NC} (http://localhost:8501)"
    else
        echo -e "${RED}❌ Streamlit App: Not responding${NC}"
    fi
    
    # Check Analytics Dashboard
    if curl -s http://localhost:8050 >/dev/null; then
        echo -e "${GREEN}✅ Analytics Dashboard: Running${NC} (http://localhost:8050)"
    else
        echo -e "${RED}❌ Analytics Dashboard: Not responding${NC}"
    fi
    
    # Check Redis
    check_redis >/dev/null
}

# Function to stop all services
stop_services() {
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    
    # Stop API server
    if [ -f .api_pid ]; then
        API_PID=$(cat .api_pid)
        if kill -0 $API_PID 2>/dev/null; then
            kill $API_PID
            echo "API server stopped"
        fi
        rm -f .api_pid
    fi
    
    # Stop Streamlit
    if [ -f .streamlit_pid ]; then
        STREAMLIT_PID=$(cat .streamlit_pid)
        if kill -0 $STREAMLIT_PID 2>/dev/null; then
            kill $STREAMLIT_PID
            echo "Streamlit app stopped"
        fi
        rm -f .streamlit_pid
    fi
    
    # Stop Analytics Dashboard
    if [ -f .dashboard_pid ]; then
        DASHBOARD_PID=$(cat .dashboard_pid)
        if kill -0 $DASHBOARD_PID 2>/dev/null; then
            kill $DASHBOARD_PID
            echo "Analytics dashboard stopped"
        fi
        rm -f .dashboard_pid
    fi
    
    echo -e "${GREEN}✅ All services stopped${NC}"
}

# Function to show help
show_help() {
    echo "Advanced RAG System Control Script"
    echo "================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start all services (default)"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  status    Show service status"
    echo "  api       Start only API server"
    echo "  ui        Start only Streamlit app"
    echo "  analytics Start only analytics dashboard"
    echo "  help      Show this help message"
    echo ""
    echo "Features included:"
    echo "  ✅ Multi-turn conversations"
    echo "  ✅ Advanced cross-encoder reranking"
    echo "  ✅ Redis-based caching"
    echo "  ✅ A/B testing framework"
    echo "  ✅ Real-time learning system"
    echo "  ✅ Multi-language support"
    echo "  ✅ Advanced analytics dashboard"
}

# Trap to handle cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🔄 Cleaning up...${NC}"
    stop_services
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main script logic
case "${1:-start}" in
    "start")
        echo -e "${GREEN}🚀 Starting Advanced RAG System${NC}"
        echo ""
        
        check_dependencies
        create_directories
        check_redis
        
        echo ""
        echo -e "${BLUE}Starting services...${NC}"
        
        start_api_server
        sleep 2
        start_streamlit_app
        sleep 2
        start_analytics_dashboard
        
        echo ""
        echo -e "${GREEN}🎉 All services started successfully!${NC}"
        echo ""
        echo "Available interfaces:"
        echo -e "  🌐 Web Interface: ${BLUE}http://localhost:8501${NC}"
        echo -e "  📚 API Documentation: ${BLUE}http://localhost:8000/docs${NC}"
        echo -e "  📊 Analytics Dashboard: ${BLUE}http://localhost:8050${NC}"
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
        
        # Keep script running
        while true; do
            sleep 10
            # Check if services are still running
            if ! curl -s http://localhost:8000/health >/dev/null; then
                echo -e "${RED}⚠️  API server stopped unexpectedly${NC}"
                break
            fi
        done
        ;;
    
    "stop")
        stop_services
        ;;
    
    "restart")
        stop_services
        sleep 2
        $0 start
        ;;
    
    "status")
        show_status
        ;;
    
    "api")
        check_dependencies
        create_directories
        check_redis
        start_api_server
        echo -e "${YELLOW}Press Ctrl+C to stop API server${NC}"
        wait
        ;;
    
    "ui")
        start_streamlit_app
        echo -e "${YELLOW}Press Ctrl+C to stop Streamlit app${NC}"
        wait
        ;;
    
    "analytics")
        start_analytics_dashboard
        echo -e "${YELLOW}Press Ctrl+C to stop analytics dashboard${NC}"
        wait
        ;;
    
    "help"|"-h"|"--help")
        show_help
        ;;
    
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
