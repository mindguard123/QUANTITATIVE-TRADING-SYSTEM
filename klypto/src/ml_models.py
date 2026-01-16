"""
Machine Learning Models Module
Implements multiple ML models for price prediction and trading signals.
"""

import warnings
warnings.filterwarnings('ignore')

from typing import Optional, List, Dict, Tuple, Union
import pandas as pd
import numpy as np

# Scikit-learn
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Gradient Boosting
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from .utils import setup_logging, save_model, load_model

logger = setup_logging()


class MLModelTrainer:
    """
    Machine Learning model trainer for price prediction and signal generation.
    
    Supports:
    - XGBoost
    - LightGBM
    - Neural Networks (LSTM/GRU)
    - Ensemble methods
    """
    
    def __init__(self,
                 model_type: str = 'xgboost',
                 task: str = 'classification',
                 random_state: int = 42):
        """
        Initialize ML Model Trainer.
        
        Args:
            model_type: 'xgboost', 'lightgbm', 'neural_network', 'ensemble'
            task: 'classification' (direction) or 'regression' (returns)
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.task = task.lower()
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        
        self.feature_importance = None
        self.training_history = None
        
        logger.info(f"Initialized {model_type} model for {task}")
    
    def prepare_data(self,
                    data: pd.DataFrame,
                    target_column: str = 'target',
                    feature_columns: Optional[List[str]] = None,
                    lookback: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            data: DataFrame with features and target
            target_column: Name of target column
            feature_columns: List of feature columns (auto-select if None)
            lookback: Number of time steps to look back (for LSTM/GRU)
            
        Returns:
            Tuple of (X, y)
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        if feature_columns is None:
            # Auto-select features (exclude target and non-numeric)
            exclude_cols = [
                target_column, 'regime', 'regime_label', 'outlier_flag',
                'open', 'high', 'low', 'close', 'volume'  # Use derived features instead
            ]
            feature_columns = [
                col for col in data.columns 
                if col not in exclude_cols and data[col].dtype in [np.float64, np.float32, np.int64, np.int32]
            ]
        
        self.feature_columns = feature_columns
        logger.info(f"Using {len(feature_columns)} features")
        
        # Extract features and target
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Handle NaN
        if np.isnan(X).any() or np.isnan(y).any():
            logger.warning("NaN values detected. Removing affected rows.")
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
        
        return X, y
    
    def create_target(self,
                     data: pd.DataFrame,
                     method: str = 'direction',
                     horizon: int = 1) -> pd.DataFrame:
        """
        Create target variable for prediction.
        
        Args:
            data: DataFrame with price data
            method: 'direction' (up/down), 'returns', 'binary_threshold'
            horizon: Prediction horizon in days
            
        Returns:
            DataFrame with target column (last 'horizon' rows will be NaN)
        
        Note:
            Fixed lookahead bias: Target is based on FUTURE price movement,
            so we calculate forward returns WITHOUT shifting back.
            This means target at time t predicts t+horizon.
            When training, we align features[t] with target[t] which predicts t+horizon.
        """
        data = data.copy()
        
        if method == 'direction':
            # Predict price direction (1=up, 0=down)
            # Calculate return from t to t+horizon
            future_price = data['close'].shift(-horizon)
            future_returns = (future_price - data['close']) / data['close']
            data['target'] = (future_returns > 0).astype(int)
            
        elif method == 'returns':
            # Predict future returns
            future_price = data['close'].shift(-horizon)
            data['target'] = (future_price - data['close']) / data['close']
            
        elif method == 'binary_threshold':
            # Binary classification with threshold
            future_price = data['close'].shift(-horizon)
            future_returns = (future_price - data['close']) / data['close']
            # Use historical std (calculated on past data only)
            if 'returns' in data.columns:
                threshold = data['returns'].expanding(min_periods=20).std() * 0.5
            else:
                threshold = 0.005  # 0.5% default
            data['target'] = (future_returns > threshold).astype(int)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Remove rows with NaN target (last 'horizon' rows)
        data = data.dropna(subset=['target'])
        
        logger.info(f"Created target using method '{method}', horizon={horizon}")
        logger.info(f"Removed {horizon} rows at end (future unknowns)")
        if method in ['direction', 'binary_threshold']:
            logger.info(f"Target distribution:\n{data['target'].value_counts()}")
        
        return data
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             hyperparameters: Optional[Dict] = None) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            hyperparameters: Model hyperparameters
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        logger.info(f"Training samples: {len(X_train)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Train based on model type
        if self.model_type == 'xgboost':
            metrics = self._train_xgboost(
                X_train_scaled, y_train, X_val_scaled, y_val, hyperparameters
            )
        elif self.model_type == 'lightgbm':
            metrics = self._train_lightgbm(
                X_train_scaled, y_train, X_val_scaled, y_val, hyperparameters
            )
        elif self.model_type == 'neural_network':
            metrics = self._train_neural_network(
                X_train_scaled, y_train, X_val_scaled, y_val, hyperparameters
            )
        elif self.model_type == 'ensemble':
            metrics = self._train_ensemble(
                X_train_scaled, y_train, X_val_scaled, y_val, hyperparameters
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.is_fitted = True
        logger.info(f"Training complete. Metrics: {metrics}")
        
        return metrics
    
    def _train_xgboost(self,
                      X_train, y_train, X_val, y_val,
                      hyperparameters) -> Dict:
        """Train XGBoost model."""
        if hyperparameters is None:
            if self.task == 'classification':
                hyperparameters = {
                    'max_depth': 5,
                    'learning_rate': 0.01,
                    'n_estimators': 500,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'early_stopping_rounds': 50,
                    'random_state': self.random_state
                }
            else:
                hyperparameters = {
                    'max_depth': 5,
                    'learning_rate': 0.01,
                    'n_estimators': 500,
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'early_stopping_rounds': 50,
                    'random_state': self.random_state
                }
        
        early_stopping_rounds = hyperparameters.pop('early_stopping_rounds', 50)
        
        # Create and train model
        self.model = xgb.XGBRegressor(**hyperparameters) if self.task == 'regression' else xgb.XGBClassifier(**hyperparameters)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        metrics = self._calculate_metrics(y_train, train_pred)
        
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)
            metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
        
        return metrics
    
    def _train_lightgbm(self,
                       X_train, y_train, X_val, y_val,
                       hyperparameters) -> Dict:
        """Train LightGBM model."""
        if hyperparameters is None:
            if self.task == 'classification':
                hyperparameters = {
                    'max_depth': 5,
                    'learning_rate': 0.01,
                    'n_estimators': 500,
                    'objective': 'binary',
                    'metric': 'auc',
                    'random_state': self.random_state,
                    'verbose': -1
                }
            else:
                hyperparameters = {
                    'max_depth': 5,
                    'learning_rate': 0.01,
                    'n_estimators': 500,
                    'objective': 'regression',
                    'metric': 'rmse',
                    'random_state': self.random_state,
                    'verbose': -1
                }
        
        # Create and train model
        self.model = lgb.LGBMRegressor(**hyperparameters) if self.task == 'regression' else lgb.LGBMClassifier(**hyperparameters)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set if X_val is not None else None,
            callbacks=[lgb.early_stopping(50, verbose=False)] if X_val is not None else None
        )
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        metrics = self._calculate_metrics(y_train, train_pred)
        
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)
            metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
        
        return metrics
    
    def _train_neural_network(self,
                             X_train, y_train, X_val, y_val,
                             hyperparameters) -> Dict:
        """Train Neural Network model."""
        if hyperparameters is None:
            hyperparameters = {
                'hidden_layers': [128, 64, 32],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32
            }
        
        # Build model
        model = keras.Sequential()
        model.add(layers.Dense(
            hyperparameters['hidden_layers'][0],
            activation='relu',
            input_shape=(X_train.shape[1],)
        ))
        model.add(layers.Dropout(hyperparameters['dropout']))
        
        for units in hyperparameters['hidden_layers'][1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(hyperparameters['dropout']))
        
        # Output layer
        if self.task == 'classification':
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(layers.Dense(1))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
            loss=loss,
            metrics=metrics
        )
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Train
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=hyperparameters['epochs'],
            batch_size=hyperparameters['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        self.model = model
        self.training_history = history.history
        
        # Calculate metrics
        train_pred = model.predict(X_train, verbose=0).flatten()
        metrics_dict = self._calculate_metrics(y_train, train_pred)
        
        if X_val is not None:
            val_pred = model.predict(X_val, verbose=0).flatten()
            val_metrics = self._calculate_metrics(y_val, val_pred)
            metrics_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
        
        return metrics_dict
    
    def _train_ensemble(self,
                       X_train, y_train, X_val, y_val,
                       hyperparameters) -> Dict:
        """Train ensemble of models."""
        logger.info("Training ensemble of XGBoost, LightGBM, and Neural Network...")
        
        models = {}
        predictions = {}
        
        # Train XGBoost
        xgb_trainer = MLModelTrainer('xgboost', self.task, self.random_state)
        xgb_trainer.feature_columns = self.feature_columns
        xgb_trainer.scaler = self.scaler
        xgb_trainer._train_xgboost(X_train, y_train, X_val, y_val, None)
        models['xgboost'] = xgb_trainer.model
        
        # Train LightGBM
        lgb_trainer = MLModelTrainer('lightgbm', self.task, self.random_state)
        lgb_trainer.feature_columns = self.feature_columns
        lgb_trainer.scaler = self.scaler
        lgb_trainer._train_lightgbm(X_train, y_train, X_val, y_val, None)
        models['lightgbm'] = lgb_trainer.model
        
        # Train Neural Network
        nn_trainer = MLModelTrainer('neural_network', self.task, self.random_state)
        nn_trainer.feature_columns = self.feature_columns
        nn_trainer.scaler = self.scaler
        nn_trainer._train_neural_network(X_train, y_train, X_val, y_val, None)
        models['neural_network'] = nn_trainer.model
        
        self.model = models
        
        # Ensemble predictions (simple average)
        train_pred = self._ensemble_predict(X_train)
        metrics_dict = self._calculate_metrics(y_train, train_pred)
        
        if X_val is not None:
            val_pred = self._ensemble_predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)
            metrics_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
        
        return metrics_dict
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        
        for model_name, model in self.model.items():
            if model_name == 'neural_network':
                pred = model.predict(X, verbose=0).flatten()
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'ensemble':
            predictions = self._ensemble_predict(X_scaled)
        elif self.model_type == 'neural_network':
            predictions = self.model.predict(X_scaled, verbose=0).flatten()
        else:
            predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (classification only).
        
        Args:
            X: Features
            
        Returns:
            Probability predictions
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'ensemble':
            proba_list = []
            for model_name, model in self.model.items():
                if model_name == 'neural_network':
                    proba = model.predict(X_scaled, verbose=0).flatten()
                else:
                    proba = model.predict_proba(X_scaled)[:, 1]
                proba_list.append(proba)
            return np.mean(proba_list, axis=0)
        elif self.model_type == 'neural_network':
            return self.model.predict(X_scaled, verbose=0).flatten()
        else:
            return self.model.predict_proba(X_scaled)[:, 1]
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        metrics = {}
        
        if self.task == 'classification':
            # Convert predictions to binary
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
            metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
        else:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance ranking.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not available for this model type")
            return pd.DataFrame()
        
        return self.feature_importance.head(top_n)
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'task': self.task,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history
        }
        
        save_model(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'MLModelTrainer':
        """Load model from disk."""
        model_data = load_model(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.task = model_data['task']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance = model_data.get('feature_importance')
        self.training_history = model_data.get('training_history')
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return self


if __name__ == "__main__":
    # Test the module
    print("Testing ML Models Module")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y_class = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_reg = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1
    
    # Split data
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train_class, y_test_class = y_class[:split], y_class[split:]
    
    # Test classification
    print("\nTesting XGBoost Classification:")
    trainer = MLModelTrainer('xgboost', 'classification')
    trainer.feature_columns = [f'feature_{i}' for i in range(n_features)]
    metrics = trainer.train(X_train, y_train_class, X_test, y_test_class)
    print(f"Metrics: {metrics}")
    
    predictions = trainer.predict(X_test)
    print(f"Sample predictions: {predictions[:10]}")
