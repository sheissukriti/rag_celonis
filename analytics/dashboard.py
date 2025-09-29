"""
Advanced analytics dashboard for RAG system performance monitoring.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)

class RAGAnalytics:
    """Analytics engine for RAG system metrics."""
    
    def __init__(self, logs_path: str = "logs", storage_path: str = "store"):
        self.logs_path = Path(logs_path)
        self.storage_path = Path(storage_path)
        
        # Data sources
        self.response_logs = []
        self.feedback_data = []
        self.experiment_data = []
        self.system_metrics = []
        
        # Load data
        self._load_data()
        
        logger.info("RAGAnalytics initialized")
    
    def _load_data(self):
        """Load data from various sources."""
        # Load response logs
        response_log_file = self.logs_path / "responses.jsonl"
        if response_log_file.exists():
            try:
                with open(response_log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.response_logs.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to load response logs: {e}")
        
        # Load feedback data
        feedback_file = self.storage_path / "feedback" / "feedback.jsonl"
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.feedback_data.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to load feedback data: {e}")
        
        # Load experiment results
        experiment_results_file = self.storage_path / "experiments" / "results.jsonl"
        if experiment_results_file.exists():
            try:
                with open(experiment_results_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.experiment_data.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to load experiment data: {e}")
        
        logger.info(f"Loaded {len(self.response_logs)} responses, {len(self.feedback_data)} feedback items, {len(self.experiment_data)} experiment results")
    
    def get_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get system performance metrics."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent data
        recent_responses = [
            r for r in self.response_logs
            if datetime.fromisoformat(r['timestamp']) > cutoff_date
        ]
        
        recent_feedback = [
            f for f in self.feedback_data
            if datetime.fromisoformat(f['timestamp']) > cutoff_date
        ]
        
        if not recent_responses:
            return {"error": "No recent data available"}
        
        # Calculate metrics
        response_times = [r['response_time_seconds'] for r in recent_responses]
        avg_response_time = sum(response_times) / len(response_times)
        
        # Feedback metrics
        positive_feedback = sum(
            1 for f in recent_feedback
            if f['feedback_type'] in ['thumbs_up', 'rating'] and
            (f['feedback_value'] is True or (isinstance(f['feedback_value'], (int, float)) and f['feedback_value'] >= 4))
        )
        
        total_feedback = len(recent_feedback)
        satisfaction_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
        
        return {
            "total_queries": len(recent_responses),
            "avg_response_time": round(avg_response_time, 3),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "satisfaction_rate": round(satisfaction_rate, 1),
            "queries_per_day": round(len(recent_responses) / days, 1),
            "period_days": days
        }
    
    def get_usage_trends(self, days: int = 30) -> Dict[str, List]:
        """Get usage trends over time."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent data
        recent_responses = [
            r for r in self.response_logs
            if datetime.fromisoformat(r['timestamp']) > cutoff_date
        ]
        
        # Group by date
        daily_counts = {}
        daily_response_times = {}
        
        for response in recent_responses:
            date = datetime.fromisoformat(response['timestamp']).date()
            date_str = date.isoformat()
            
            if date_str not in daily_counts:
                daily_counts[date_str] = 0
                daily_response_times[date_str] = []
            
            daily_counts[date_str] += 1
            daily_response_times[date_str].append(response['response_time_seconds'])
        
        # Calculate daily averages
        dates = sorted(daily_counts.keys())
        counts = [daily_counts[date] for date in dates]
        avg_times = [
            sum(daily_response_times[date]) / len(daily_response_times[date])
            for date in dates
        ]
        
        return {
            "dates": dates,
            "query_counts": counts,
            "avg_response_times": avg_times
        }
    
    def get_retriever_performance(self) -> Dict[str, Any]:
        """Analyze retriever performance."""
        retriever_stats = {}
        
        for response in self.response_logs:
            retriever_type = response.get('retriever_type', 'unknown')
            
            if retriever_type not in retriever_stats:
                retriever_stats[retriever_type] = {
                    'count': 0,
                    'total_time': 0,
                    'citation_counts': []
                }
            
            stats = retriever_stats[retriever_type]
            stats['count'] += 1
            stats['total_time'] += response['response_time_seconds']
            stats['citation_counts'].append(len(response.get('citations', [])))
        
        # Calculate averages
        for retriever_type, stats in retriever_stats.items():
            stats['avg_response_time'] = stats['total_time'] / stats['count']
            stats['avg_citations'] = sum(stats['citation_counts']) / len(stats['citation_counts'])
            del stats['total_time']
            del stats['citation_counts']
        
        return retriever_stats
    
    def get_query_analysis(self) -> Dict[str, Any]:
        """Analyze query patterns."""
        if not self.response_logs:
            return {}
        
        # Query length analysis
        query_lengths = [len(r['query'].split()) for r in self.response_logs]
        
        # Common words analysis
        all_words = []
        for response in self.response_logs:
            words = response['query'].lower().split()
            all_words.extend(words)
        
        word_counts = {}
        for word in all_words:
            if len(word) > 3:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            "total_queries": len(self.response_logs),
            "avg_query_length": sum(query_lengths) / len(query_lengths),
            "min_query_length": min(query_lengths),
            "max_query_length": max(query_lengths),
            "top_words": top_words
        }
    
    def get_feedback_analysis(self) -> Dict[str, Any]:
        """Analyze user feedback patterns."""
        if not self.feedback_data:
            return {"total_feedback": 0}
        
        # Feedback type distribution
        type_counts = {}
        for feedback in self.feedback_data:
            feedback_type = feedback['feedback_type']
            type_counts[feedback_type] = type_counts.get(feedback_type, 0) + 1
        
        # Rating distribution (for rating feedback)
        ratings = [
            f['feedback_value'] for f in self.feedback_data
            if f['feedback_type'] == 'rating' and isinstance(f['feedback_value'], (int, float))
        ]
        
        rating_dist = {}
        for rating in ratings:
            rating_dist[str(int(rating))] = rating_dist.get(str(int(rating)), 0) + 1
        
        return {
            "total_feedback": len(self.feedback_data),
            "feedback_types": type_counts,
            "rating_distribution": rating_dist,
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0
        }

class DashboardApp:
    """Dash-based analytics dashboard."""
    
    def __init__(self, analytics: RAGAnalytics, port: int = 8050):
        self.analytics = analytics
        self.port = port
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Setup layout
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info(f"Dashboard initialized on port {port}")
    
    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("RAG System Analytics Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Time Period"),
                            dcc.Dropdown(
                                id='time-period-dropdown',
                                options=[
                                    {'label': 'Last 24 hours', 'value': 1},
                                    {'label': 'Last 7 days', 'value': 7},
                                    {'label': 'Last 30 days', 'value': 30},
                                    {'label': 'Last 90 days', 'value': 90}
                                ],
                                value=7
                            ),
                            html.Br(),
                            dbc.Button("Refresh Data", id="refresh-button", color="primary")
                        ])
                    ])
                ], width=3),
                
                # Key metrics cards
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(id="total-queries", className="card-title"),
                                    html.P("Total Queries", className="card-text")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(id="avg-response-time", className="card-title"),
                                    html.P("Avg Response Time (s)", className="card-text")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(id="satisfaction-rate", className="card-title"),
                                    html.P("Satisfaction Rate (%)", className="card-text")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(id="total-feedback", className="card-title"),
                                    html.P("Total Feedback", className="card-text")
                                ])
                            ])
                        ], width=3)
                    ])
                ], width=9)
            ], className="mb-4"),
            
            # Charts row 1
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Usage Trends"),
                        dbc.CardBody([
                            dcc.Graph(id="usage-trends-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Response Time Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id="response-time-dist-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Charts row 2
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Retriever Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="retriever-performance-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Feedback Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="feedback-analysis-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Query analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Top Query Terms"),
                        dbc.CardBody([
                            dcc.Graph(id="query-terms-chart")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('total-queries', 'children'),
             Output('avg-response-time', 'children'),
             Output('satisfaction-rate', 'children'),
             Output('total-feedback', 'children'),
             Output('usage-trends-chart', 'figure'),
             Output('response-time-dist-chart', 'figure'),
             Output('retriever-performance-chart', 'figure'),
             Output('feedback-analysis-chart', 'figure'),
             Output('query-terms-chart', 'figure')],
            [Input('time-period-dropdown', 'value'),
             Input('refresh-button', 'n_clicks'),
             Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(time_period, refresh_clicks, n_intervals):
            # Reload data
            self.analytics._load_data()
            
            # Get metrics
            metrics = self.analytics.get_performance_metrics(time_period)
            trends = self.analytics.get_usage_trends(time_period)
            retriever_perf = self.analytics.get_retriever_performance()
            feedback_analysis = self.analytics.get_feedback_analysis()
            query_analysis = self.analytics.get_query_analysis()
            
            # Update metric cards
            total_queries = metrics.get('total_queries', 0)
            avg_response_time = f"{metrics.get('avg_response_time', 0):.2f}"
            satisfaction_rate = f"{metrics.get('satisfaction_rate', 0):.1f}%"
            total_feedback = feedback_analysis.get('total_feedback', 0)
            
            # Usage trends chart
            usage_fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Daily Query Count', 'Average Response Time'),
                vertical_spacing=0.1
            )
            
            if trends.get('dates'):
                usage_fig.add_trace(
                    go.Scatter(
                        x=trends['dates'],
                        y=trends['query_counts'],
                        mode='lines+markers',
                        name='Queries'
                    ),
                    row=1, col=1
                )
                
                usage_fig.add_trace(
                    go.Scatter(
                        x=trends['dates'],
                        y=trends['avg_response_times'],
                        mode='lines+markers',
                        name='Response Time (s)',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
            
            usage_fig.update_layout(height=400, showlegend=False)
            
            # Response time distribution
            if self.analytics.response_logs:
                response_times = [r['response_time_seconds'] for r in self.analytics.response_logs]
                response_time_fig = go.Figure(data=[go.Histogram(x=response_times, nbinsx=20)])
                response_time_fig.update_layout(
                    title="Response Time Distribution",
                    xaxis_title="Response Time (seconds)",
                    yaxis_title="Count"
                )
            else:
                response_time_fig = go.Figure()
            
            # Retriever performance chart
            if retriever_perf:
                retrievers = list(retriever_perf.keys())
                avg_times = [retriever_perf[r]['avg_response_time'] for r in retrievers]
                counts = [retriever_perf[r]['count'] for r in retrievers]
                
                retriever_fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Avg Response Time', 'Usage Count'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                retriever_fig.add_trace(
                    go.Bar(x=retrievers, y=avg_times, name='Avg Time'),
                    row=1, col=1
                )
                
                retriever_fig.add_trace(
                    go.Bar(x=retrievers, y=counts, name='Count'),
                    row=1, col=2
                )
                
                retriever_fig.update_layout(height=400, showlegend=False)
            else:
                retriever_fig = go.Figure()
            
            # Feedback analysis chart
            feedback_types = feedback_analysis.get('feedback_types', {})
            if feedback_types:
                feedback_fig = go.Figure(data=[
                    go.Pie(
                        labels=list(feedback_types.keys()),
                        values=list(feedback_types.values())
                    )
                ])
                feedback_fig.update_layout(title="Feedback Types Distribution")
            else:
                feedback_fig = go.Figure()
            
            # Query terms chart
            top_words = query_analysis.get('top_words', [])
            if top_words:
                words, counts = zip(*top_words[:15])
                query_terms_fig = go.Figure(data=[
                    go.Bar(x=list(counts), y=list(words), orientation='h')
                ])
                query_terms_fig.update_layout(
                    title="Most Common Query Terms",
                    xaxis_title="Frequency",
                    height=500
                )
            else:
                query_terms_fig = go.Figure()
            
            return (
                total_queries, avg_response_time, satisfaction_rate, total_feedback,
                usage_fig, response_time_fig, retriever_fig, feedback_fig, query_terms_fig
            )
    
    def run(self, debug: bool = False):
        """Run the dashboard."""
        self.app.run(debug=debug, port=self.port, host='0.0.0.0')

def create_analytics_dashboard(logs_path: str = "logs", storage_path: str = "store", port: int = 8050):
    """Create and return analytics dashboard."""
    analytics = RAGAnalytics(logs_path, storage_path)
    dashboard = DashboardApp(analytics, port)
    return dashboard

# Standalone dashboard runner
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Analytics Dashboard")
    parser.add_argument("--logs-path", default="logs", help="Path to logs directory")
    parser.add_argument("--storage-path", default="store", help="Path to storage directory")
    parser.add_argument("--port", type=int, default=8050, help="Port to run dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    dashboard = create_analytics_dashboard(args.logs_path, args.storage_path, args.port)
    
    print(f"Starting RAG Analytics Dashboard on http://localhost:{args.port}")
    dashboard.run(debug=args.debug)
