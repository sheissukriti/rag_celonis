"""
A/B Testing framework for comparing different retrieval strategies and configurations.
"""

import json
import uuid
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

@dataclass
class ExperimentVariant:
    """Represents a variant in an A/B test."""
    variant_id: str
    name: str
    description: str
    config: Dict[str, Any]
    traffic_percentage: float
    is_control: bool = False

@dataclass
class ExperimentMetric:
    """Represents a metric to track in experiments."""
    name: str
    description: str
    metric_type: str  # 'counter', 'timer', 'gauge', 'rate'
    higher_is_better: bool = True

@dataclass
class ExperimentResult:
    """Stores results for a single experiment interaction."""
    experiment_id: str
    variant_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    query: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: str

@dataclass
class Experiment:
    """Represents an A/B test experiment."""
    experiment_id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    status: ExperimentStatus
    start_date: Optional[str]
    end_date: Optional[str]
    created_at: str
    updated_at: str
    created_by: Optional[str]
    metadata: Dict[str, Any]
    
    def get_variant_by_id(self, variant_id: str) -> Optional[ExperimentVariant]:
        """Get variant by ID."""
        return next((v for v in self.variants if v.variant_id == variant_id), None)
    
    def get_control_variant(self) -> Optional[ExperimentVariant]:
        """Get the control variant."""
        return next((v for v in self.variants if v.is_control), None)
    
    def validate(self) -> List[str]:
        """Validate experiment configuration."""
        errors = []
        
        # Check traffic allocation
        total_traffic = sum(v.traffic_percentage for v in self.variants)
        if abs(total_traffic - 100.0) > 0.01:
            errors.append(f"Traffic allocation must sum to 100%, got {total_traffic}%")
        
        # Check for control variant
        control_variants = [v for v in self.variants if v.is_control]
        if len(control_variants) != 1:
            errors.append("Experiment must have exactly one control variant")
        
        # Check variant IDs are unique
        variant_ids = [v.variant_id for v in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            errors.append("Variant IDs must be unique")
        
        # Check metrics
        if not self.metrics:
            errors.append("Experiment must have at least one metric")
        
        return errors

class ExperimentManager:
    """Manages A/B test experiments."""
    
    def __init__(self, storage_path: str = "store/experiments"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.experiments: Dict[str, Experiment] = {}
        self.results: List[ExperimentResult] = []
        
        # Load existing experiments
        self._load_experiments()
        
        logger.info(f"ExperimentManager initialized with {len(self.experiments)} experiments")
    
    def create_experiment(self, name: str, description: str, 
                         variants: List[Dict[str, Any]], 
                         metrics: List[Dict[str, Any]],
                         created_by: Optional[str] = None) -> str:
        """Create a new experiment."""
        experiment_id = str(uuid.uuid4())
        
        # Create variant objects
        variant_objects = []
        for variant_data in variants:
            variant = ExperimentVariant(
                variant_id=variant_data.get('variant_id', str(uuid.uuid4())),
                name=variant_data['name'],
                description=variant_data['description'],
                config=variant_data['config'],
                traffic_percentage=variant_data['traffic_percentage'],
                is_control=variant_data.get('is_control', False)
            )
            variant_objects.append(variant)
        
        # Create metric objects
        metric_objects = []
        for metric_data in metrics:
            metric = ExperimentMetric(
                name=metric_data['name'],
                description=metric_data['description'],
                metric_type=metric_data['metric_type'],
                higher_is_better=metric_data.get('higher_is_better', True)
            )
            metric_objects.append(metric)
        
        # Create experiment
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variant_objects,
            metrics=metric_objects,
            status=ExperimentStatus.DRAFT,
            start_date=None,
            end_date=None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            created_by=created_by,
            metadata={}
        )
        
        # Validate experiment
        errors = experiment.validate()
        if errors:
            raise ValueError(f"Experiment validation failed: {', '.join(errors)}")
        
        # Store experiment
        self.experiments[experiment_id] = experiment
        self._save_experiment(experiment)
        
        logger.info(f"Created experiment: {name} ({experiment_id})")
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            logger.error(f"Can only start draft experiments, current status: {experiment.status.value}")
            return False
        
        experiment.status = ExperimentStatus.ACTIVE
        experiment.start_date = datetime.now().isoformat()
        experiment.updated_at = datetime.now().isoformat()
        
        self._save_experiment(experiment)
        
        logger.info(f"Started experiment: {experiment.name}")
        return True
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.ACTIVE:
            logger.error(f"Can only stop active experiments, current status: {experiment.status.value}")
            return False
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.now().isoformat()
        experiment.updated_at = datetime.now().isoformat()
        
        self._save_experiment(experiment)
        
        logger.info(f"Stopped experiment: {experiment.name}")
        return True
    
    def assign_variant(self, experiment_id: str, user_id: Optional[str] = None,
                      session_id: Optional[str] = None) -> Optional[ExperimentVariant]:
        """Assign a user to a variant based on traffic allocation."""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.ACTIVE:
            return None
        
        # Use consistent hashing for user assignment if user_id is provided
        if user_id:
            hash_input = f"{experiment_id}:{user_id}"
            hash_value = hash(hash_input) % 10000  # 0-9999
            percentage = hash_value / 100.0  # 0-99.99
        else:
            # Random assignment for anonymous users
            percentage = random.random() * 100
        
        # Find variant based on traffic allocation
        cumulative_percentage = 0
        for variant in experiment.variants:
            cumulative_percentage += variant.traffic_percentage
            if percentage < cumulative_percentage:
                return variant
        
        # Fallback to control variant
        return experiment.get_control_variant()
    
    def record_result(self, experiment_id: str, variant_id: str, 
                     query: str, metrics: Dict[str, float],
                     user_id: Optional[str] = None,
                     session_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record experiment result."""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        experiment = self.experiments[experiment_id]
        variant = experiment.get_variant_by_id(variant_id)
        
        if not variant:
            logger.error(f"Variant not found: {variant_id}")
            return False
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            session_id=session_id,
            query=query,
            metrics=metrics,
            metadata=metadata or {},
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        self._save_result(result)
        
        logger.debug(f"Recorded result for experiment {experiment_id}, variant {variant_id}")
        return True
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get aggregated results for an experiment."""
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        experiment_results = [r for r in self.results if r.experiment_id == experiment_id]
        
        if not experiment_results:
            return {
                'experiment_id': experiment_id,
                'total_samples': 0,
                'variants': {}
            }
        
        # Aggregate results by variant
        variant_results = {}
        
        for variant in experiment.variants:
            variant_data = [r for r in experiment_results if r.variant_id == variant.variant_id]
            
            if not variant_data:
                variant_results[variant.variant_id] = {
                    'name': variant.name,
                    'is_control': variant.is_control,
                    'sample_size': 0,
                    'metrics': {}
                }
                continue
            
            # Calculate metric statistics
            metric_stats = {}
            for metric in experiment.metrics:
                metric_values = [r.metrics.get(metric.name, 0) for r in variant_data]
                
                if metric_values:
                    metric_stats[metric.name] = {
                        'mean': statistics.mean(metric_values),
                        'median': statistics.median(metric_values),
                        'std': statistics.stdev(metric_values) if len(metric_values) > 1 else 0,
                        'min': min(metric_values),
                        'max': max(metric_values),
                        'count': len(metric_values)
                    }
                else:
                    metric_stats[metric.name] = {
                        'mean': 0, 'median': 0, 'std': 0,
                        'min': 0, 'max': 0, 'count': 0
                    }
            
            variant_results[variant.variant_id] = {
                'name': variant.name,
                'is_control': variant.is_control,
                'sample_size': len(variant_data),
                'metrics': metric_stats
            }
        
        return {
            'experiment_id': experiment_id,
            'experiment_name': experiment.name,
            'status': experiment.status.value,
            'total_samples': len(experiment_results),
            'start_date': experiment.start_date,
            'end_date': experiment.end_date,
            'variants': variant_results
        }
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """List experiments with optional status filter."""
        experiments = []
        
        for experiment in self.experiments.values():
            if status and experiment.status != status:
                continue
            
            experiment_summary = {
                'experiment_id': experiment.experiment_id,
                'name': experiment.name,
                'description': experiment.description,
                'status': experiment.status.value,
                'variant_count': len(experiment.variants),
                'metric_count': len(experiment.metrics),
                'created_at': experiment.created_at,
                'start_date': experiment.start_date,
                'end_date': experiment.end_date
            }
            
            experiments.append(experiment_summary)
        
        return sorted(experiments, key=lambda x: x['created_at'], reverse=True)
    
    def _save_experiment(self, experiment: Experiment):
        """Save experiment to storage."""
        file_path = self.storage_path / f"experiment_{experiment.experiment_id}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(experiment), f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save experiment {experiment.experiment_id}: {e}")
    
    def _save_result(self, result: ExperimentResult):
        """Save result to storage."""
        results_file = self.storage_path / "results.jsonl"
        
        try:
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(result), ensure_ascii=False, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
    
    def _load_experiments(self):
        """Load experiments from storage."""
        if not self.storage_path.exists():
            return
        
        # Load experiments
        for file_path in self.storage_path.glob("experiment_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstruct experiment object
                variants = [ExperimentVariant(**v) for v in data['variants']]
                metrics = [ExperimentMetric(**m) for m in data['metrics']]
                
                experiment = Experiment(
                    experiment_id=data['experiment_id'],
                    name=data['name'],
                    description=data['description'],
                    variants=variants,
                    metrics=metrics,
                    status=ExperimentStatus(data['status']),
                    start_date=data['start_date'],
                    end_date=data['end_date'],
                    created_at=data['created_at'],
                    updated_at=data['updated_at'],
                    created_by=data['created_by'],
                    metadata=data['metadata']
                )
                
                self.experiments[experiment.experiment_id] = experiment
                
            except Exception as e:
                logger.error(f"Failed to load experiment from {file_path}: {e}")
        
        # Load results
        results_file = self.storage_path / "results.jsonl"
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            result = ExperimentResult(**data)
                            self.results.append(result)
            except Exception as e:
                logger.error(f"Failed to load results: {e}")

class ABTestingService:
    """High-level service for A/B testing in RAG system."""
    
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self.active_experiments: Dict[str, str] = {}  # user_id -> experiment_id
        
        logger.info("ABTestingService initialized")
    
    def get_retrieval_config(self, user_id: Optional[str] = None,
                           session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get retrieval configuration based on active experiments."""
        # Find active experiments
        active_experiments = self.experiment_manager.list_experiments(ExperimentStatus.ACTIVE)
        
        if not active_experiments:
            return self._get_default_config()
        
        # For simplicity, use the first active experiment
        # In production, you might want more sophisticated logic
        experiment_id = active_experiments[0]['experiment_id']
        
        # Assign variant
        variant = self.experiment_manager.assign_variant(experiment_id, user_id, session_id)
        
        if not variant:
            return self._get_default_config()
        
        # Store assignment for result recording
        if user_id:
            self.active_experiments[user_id] = experiment_id
        
        # Return variant configuration
        config = variant.config.copy()
        config['_experiment_id'] = experiment_id
        config['_variant_id'] = variant.variant_id
        
        return config
    
    def record_interaction(self, query: str, response_time: float,
                         relevance_score: float, user_satisfaction: Optional[float] = None,
                         user_id: Optional[str] = None,
                         session_id: Optional[str] = None,
                         config: Optional[Dict[str, Any]] = None):
        """Record interaction metrics for A/B testing."""
        if not config or '_experiment_id' not in config:
            return
        
        experiment_id = config['_experiment_id']
        variant_id = config['_variant_id']
        
        # Prepare metrics
        metrics = {
            'response_time': response_time,
            'relevance_score': relevance_score
        }
        
        if user_satisfaction is not None:
            metrics['user_satisfaction'] = user_satisfaction
        
        # Record result
        self.experiment_manager.record_result(
            experiment_id=experiment_id,
            variant_id=variant_id,
            query=query,
            metrics=metrics,
            user_id=user_id,
            session_id=session_id
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default retrieval configuration."""
        return {
            'retriever': 'faiss_tfidf',
            'top_k': 10,
            'top_k_context': 5,
            'use_reranker': True,
            'reranker_type': 'tfidf'
        }

# Example experiment configurations
EXAMPLE_EXPERIMENTS = {
    'retriever_comparison': {
        'name': 'Retriever Strategy Comparison',
        'description': 'Compare BM25 vs FAISS+TF-IDF retrieval strategies',
        'variants': [
            {
                'name': 'BM25 Control',
                'description': 'Traditional BM25 retrieval',
                'config': {'retriever': 'bm25', 'top_k': 10, 'use_reranker': False},
                'traffic_percentage': 50.0,
                'is_control': True
            },
            {
                'name': 'FAISS+TF-IDF',
                'description': 'Dense retrieval with TF-IDF reranking',
                'config': {'retriever': 'faiss_tfidf', 'top_k': 10, 'use_reranker': True},
                'traffic_percentage': 50.0,
                'is_control': False
            }
        ],
        'metrics': [
            {
                'name': 'response_time',
                'description': 'Response time in seconds',
                'metric_type': 'timer',
                'higher_is_better': False
            },
            {
                'name': 'relevance_score',
                'description': 'Average relevance score of retrieved documents',
                'metric_type': 'gauge',
                'higher_is_better': True
            }
        ]
    }
}
