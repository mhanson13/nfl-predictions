"""
Strategy pattern for prediction models.

Provides a flexible interface for different prediction strategies and model types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Result of a prediction operation."""
    
    game_id: str
    home_team: str
    away_team: str
    predictions: Dict[str, float]  # e.g., {'win_prob': 0.65, 'spread': -3.5}
    confidence: float
    model_version: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'game_id': self.game_id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            **self.predictions,
            'confidence': self.confidence,
            'model_version': self.model_version,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }


class PredictionStrategy(ABC):
    """Abstract base class for prediction strategies."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize prediction strategy.
        
        Args:
            name: Strategy name
            version: Model version
        """
        self.name = name
        self.version = version
        self.logger = get_logger(f"{__name__}.{name}")
    
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> List[PredictionResult]:
        """
        Make predictions for given features.
        
        Args:
            features: DataFrame with feature data
            
        Returns:
            List of prediction results
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: Path) -> None:
        """
        Load model from file.
        
        Args:
            model_path: Path to model file
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        pass
    
    def validate_features(self, features: pd.DataFrame) -> bool:
        """
        Validate that features have required columns.
        
        Args:
            features: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_cols = self.get_required_features()
        missing = set(required_cols) - set(features.columns)
        
        if missing:
            self.logger.error(f"Missing required features: {missing}")
            return False
        
        return True
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Get list of required feature names.
        
        Returns:
            List of feature names
        """
        pass


class XGBoostWinProbStrategy(PredictionStrategy):
    """Strategy for XGBoost win probability model."""
    
    def __init__(self, version: str = "1.0.0"):
        super().__init__("xgboost_winprob", version)
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
    
    def load_model(self, model_path: Path) -> None:
        """Load XGBoost model."""
        import xgboost as xgb
        
        self.logger.info(f"Loading XGBoost model from {model_path}")
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        self.feature_names = self.model.feature_names
        if self.feature_names:
            self.logger.info(f"Model loaded with {len(self.feature_names)} features")
    
    def predict(self, features: pd.DataFrame) -> List[PredictionResult]:
        """Make win probability predictions."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not self.validate_features(features):
            raise ValueError("Invalid features")
        
        import xgboost as xgb
        
        # Prepare features
        X = features[self.feature_names]
        dmatrix = xgb.DMatrix(X)
        
        # Make predictions
        predictions = self.model.predict(dmatrix)
        
        # Create results
        results = []
        for idx, row in features.iterrows():
            # Type-safe extraction of row values
            game_id = str(row.get('game_id', f'game_{idx}'))
            home_team = str(row.get('home_team', 'UNK'))
            away_team = str(row.get('away_team', 'UNK'))
            
            # Ensure idx is an integer for array indexing
            idx_int = int(idx) if isinstance(idx, (int, np.integer)) else 0
            
            result = PredictionResult(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                predictions={'win_prob': float(predictions[idx_int])},
                confidence=self._calculate_confidence(predictions[idx_int]),
                model_version=self.version,
                timestamp=datetime.now(),
                metadata={'strategy': self.name}
            )
            results.append(result)
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost model."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        importance = self.model.get_score(importance_type='gain')
        
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        return df
    
    def get_required_features(self) -> List[str]:
        """Get required features."""
        if self.feature_names is None:
            raise ValueError("Model not loaded")
        return self.feature_names
    
    def _calculate_confidence(self, prob: float) -> float:
        """Calculate confidence score based on probability."""
        # Confidence is higher when probability is closer to 0 or 1
        return abs(prob - 0.5) * 2


class XGBoostSpreadStrategy(PredictionStrategy):
    """Strategy for XGBoost spread prediction model."""
    
    def __init__(self, version: str = "1.0.0"):
        super().__init__("xgboost_spread", version)
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
    
    def load_model(self, model_path: Path) -> None:
        """Load XGBoost model."""
        import xgboost as xgb
        
        self.logger.info(f"Loading XGBoost spread model from {model_path}")
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        self.feature_names = self.model.feature_names
        if self.feature_names:
            self.logger.info(f"Model loaded with {len(self.feature_names)} features")
    
    def predict(self, features: pd.DataFrame) -> List[PredictionResult]:
        """Make spread predictions."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not self.validate_features(features):
            raise ValueError("Invalid features")
        
        import xgboost as xgb
        
        # Prepare features
        X = features[self.feature_names]
        dmatrix = xgb.DMatrix(X)
        
        # Make predictions
        predictions = self.model.predict(dmatrix)
        
        # Create results
        results = []
        for idx, row in features.iterrows():
            # Type-safe extraction of row values
            game_id = str(row.get('game_id', f'game_{idx}'))
            home_team = str(row.get('home_team', 'UNK'))
            away_team = str(row.get('away_team', 'UNK'))
            
            # Ensure idx is an integer for array indexing
            idx_int = int(idx) if isinstance(idx, (int, np.integer)) else 0
            
            result = PredictionResult(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                predictions={'spread': float(predictions[idx_int])},
                confidence=self._calculate_confidence(predictions[idx_int]),
                model_version=self.version,
                timestamp=datetime.now(),
                metadata={'strategy': self.name}
            )
            results.append(result)
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost model."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        importance = self.model.get_score(importance_type='gain')
        
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        return df
    
    def get_required_features(self) -> List[str]:
        """Get required features."""
        if self.feature_names is None:
            raise ValueError("Model not loaded")
        return self.feature_names
    
    def _calculate_confidence(self, spread: float) -> float:
        """Calculate confidence score based on spread magnitude."""
        # Higher confidence for larger spreads
        return min(abs(spread) / 20.0, 1.0)


class EnsembleStrategy(PredictionStrategy):
    """Ensemble strategy combining multiple models."""
    
    def __init__(self, strategies: List[PredictionStrategy], weights: Optional[List[float]] = None):
        """
        Initialize ensemble strategy.
        
        Args:
            strategies: List of prediction strategies to ensemble
            weights: Optional weights for each strategy (must sum to 1.0)
        """
        super().__init__("ensemble", "1.0.0")
        self.strategies = strategies
        
        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            if len(weights) != len(strategies):
                raise ValueError("Number of weights must match number of strategies")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights
    
    def load_model(self, model_path: Path) -> None:
        """Load models for all strategies."""
        # Ensemble doesn't have its own model file
        # Individual strategies should be loaded separately
        pass
    
    def predict(self, features: pd.DataFrame) -> List[PredictionResult]:
        """Make ensemble predictions."""
        # Get predictions from all strategies
        all_predictions = []
        for strategy in self.strategies:
            preds = strategy.predict(features)
            all_predictions.append(preds)
        
        # Combine predictions
        results = []
        for i in range(len(all_predictions[0])):
            # Get predictions for this game from all strategies
            game_preds = [preds[i] for preds in all_predictions]
            
            # Combine predictions using weights
            combined_predictions = {}
            for key in game_preds[0].predictions.keys():
                values = [p.predictions[key] for p in game_preds]
                combined_predictions[key] = sum(
                    v * w for v, w in zip(values, self.weights)
                )
            
            # Average confidence
            avg_confidence = sum(
                p.confidence * w for p, w in zip(game_preds, self.weights)
            )
            
            result = PredictionResult(
                game_id=game_preds[0].game_id,
                home_team=game_preds[0].home_team,
                away_team=game_preds[0].away_team,
                predictions=combined_predictions,
                confidence=avg_confidence,
                model_version=self.version,
                timestamp=datetime.now(),
                metadata={
                    'strategy': self.name,
                    'component_strategies': [s.name for s in self.strategies],
                    'weights': self.weights
                }
            )
            results.append(result)
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get combined feature importance."""
        # Combine importance from all strategies
        all_importance = []
        for strategy, weight in zip(self.strategies, self.weights):
            imp = strategy.get_feature_importance()
            imp['importance'] = imp['importance'] * weight
            all_importance.append(imp)
        
        # Combine and aggregate
        combined = pd.concat(all_importance)
        result = combined.groupby('feature')['importance'].sum().reset_index()
        result = result.sort_values('importance', ascending=False)
        
        return result
    
    def get_required_features(self) -> List[str]:
        """Get union of all required features."""
        all_features = set()
        for strategy in self.strategies:
            all_features.update(strategy.get_required_features())
        return list(all_features)


class ModelStrategyFactory:
    """Factory for creating prediction strategies."""
    
    _strategies = {
        'xgboost_winprob': XGBoostWinProbStrategy,
        'xgboost_spread': XGBoostSpreadStrategy,
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, **kwargs) -> PredictionStrategy:
        """
        Create a prediction strategy.
        
        Args:
            strategy_name: Name of strategy to create
            **kwargs: Additional arguments for strategy
            
        Returns:
            Prediction strategy instance
        """
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(**kwargs)
    
    @classmethod
    def create_ensemble(
        cls,
        strategy_names: List[str],
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> EnsembleStrategy:
        """
        Create an ensemble strategy.
        
        Args:
            strategy_names: List of strategy names to ensemble
            weights: Optional weights for each strategy
            **kwargs: Additional arguments for strategies
            
        Returns:
            Ensemble strategy instance
        """
        strategies = [cls.create_strategy(name, **kwargs) for name in strategy_names]
        return EnsembleStrategy(strategies, weights)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> None:
        """
        Register a new strategy.
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        cls._strategies[name] = strategy_class

# Made with Bob
