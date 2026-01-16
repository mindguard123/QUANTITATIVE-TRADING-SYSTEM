"""
Hidden Markov Model (HMM) Regime Detection Module
Implements HMM for market regime identification (bull/bear/sideways).
"""

import warnings
warnings.filterwarnings('ignore')

from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib

from .utils import setup_logging, save_model, load_model

logger = setup_logging()


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.
    
    Identifies hidden market states based on observable features:
    - Returns
    - Volatility
    - Volume
    
    States typically represent:
    - Bull Market (high returns, low volatility)
    - Bear Market (negative returns, high volatility)
    - Sideways/Consolidation (low returns, medium volatility)
    """
    
    def __init__(self,
                 n_states: int = 3,
                 n_iter: int = 100,
                 random_state: int = 42):
        """
        Initialize HMM Regime Detector.
        
        Args:
            n_states: Number of hidden states (regimes)
            n_iter: Number of iterations for training
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='full',
            n_iter=n_iter,
            random_state=random_state,
            verbose=False
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = []
        self.regime_stats = None
        
        logger.info(f"Initialized HMM with {n_states} states")
    
    def prepare_features(self, 
                        data: pd.DataFrame,
                        feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Prepare features for HMM.
        
        Args:
            data: DataFrame with market data
            feature_columns: List of feature column names (auto-selected if None)
            
        Returns:
            Numpy array of features
        """
        if feature_columns is None:
            # Auto-select features
            feature_columns = []
            
            # Returns (most important)
            if 'returns' in data.columns:
                feature_columns.append('returns')
            elif 'log_returns' in data.columns:
                feature_columns.append('log_returns')
            
            # Volatility
            if 'returns_std_20' in data.columns:
                feature_columns.append('returns_std_20')
            elif 'atr' in data.columns:
                feature_columns.append('atr')
            
            # Volume
            if 'volume_ratio' in data.columns:
                feature_columns.append('volume_ratio')
            elif 'volume_change' in data.columns:
                feature_columns.append('volume_change')
            
            # Additional momentum
            if 'rsi' in data.columns:
                feature_columns.append('rsi')
            
            # Trend
            if 'ema_cross_10_50' in data.columns:
                feature_columns.append('ema_cross_10_50')
        
        if not feature_columns:
            raise ValueError("No suitable features found. Ensure data is feature-engineered.")
        
        self.feature_columns = feature_columns
        logger.info(f"Using features: {feature_columns}")
        
        # Extract features
        features = data[feature_columns].values
        
        # Handle NaN values
        if np.isnan(features).any():
            logger.warning("NaN values detected in features. Filling with forward fill.")
            features = pd.DataFrame(features, columns=feature_columns).ffill().fillna(0).values
        
        return features
    
    def fit(self, 
            data: pd.DataFrame,
            feature_columns: Optional[List[str]] = None) -> 'HMMRegimeDetector':
        """
        Fit HMM to data.
        
        Args:
            data: DataFrame with market data and features
            feature_columns: List of feature column names
            
        Returns:
            Self
        """
        logger.info("Fitting HMM model...")
        
        # Prepare features
        features = self.prepare_features(data, feature_columns)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit HMM
        self.model.fit(features_scaled)
        self.is_fitted = True
        
        logger.info("HMM model fitted successfully")
        logger.info(f"Converged: {self.model.monitor_.converged}")
        logger.info(f"Log-likelihood: {self.model.score(features_scaled):.2f}")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict hidden states.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Array of predicted states
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare features
        features = self.prepare_features(data, self.feature_columns)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict states
        states = self.model.predict(features_scaled)
        
        return states
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict state probabilities.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Array of state probabilities (n_samples, n_states)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare features
        features = self.prepare_features(data, self.feature_columns)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict probabilities
        log_proba = self.model.score_samples(features_scaled)
        
        return np.exp(log_proba)
    
    def fit_predict(self,
                   data: pd.DataFrame,
                   feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit model and add regime predictions to data.
        
        Args:
            data: DataFrame with market data
            feature_columns: List of feature column names
            
        Returns:
            DataFrame with added regime columns
        """
        # Fit model
        self.fit(data, feature_columns)
        
        # Predict states
        states = self.predict(data)
        
        # Add to dataframe
        data = data.copy()
        data['regime'] = states
        
        # Analyze and label regimes
        self.regime_stats = self._analyze_regimes(data)
        data = self._label_regimes(data)
        
        return data
    
    def _analyze_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze characteristics of each regime.
        
        Args:
            data: DataFrame with regime column
            
        Returns:
            DataFrame with regime statistics
        """
        logger.info("Analyzing regime characteristics...")
        
        stats = []
        
        for state in range(self.n_states):
            regime_data = data[data['regime'] == state]
            
            if len(regime_data) == 0:
                continue
            
            stats_dict = {
                'state': state,
                'count': len(regime_data),
                'frequency': len(regime_data) / len(data),
                'mean_return': regime_data['returns'].mean() if 'returns' in regime_data.columns else np.nan,
                'std_return': regime_data['returns'].std() if 'returns' in regime_data.columns else np.nan,
                'sharpe': (regime_data['returns'].mean() / regime_data['returns'].std() * np.sqrt(252)) 
                         if 'returns' in regime_data.columns else np.nan
            }
            
            # Add feature statistics
            for col in self.feature_columns:
                if col in regime_data.columns:
                    stats_dict[f'{col}_mean'] = regime_data[col].mean()
                    stats_dict[f'{col}_std'] = regime_data[col].std()
            
            stats.append(stats_dict)
        
        stats_df = pd.DataFrame(stats)
        
        # Sort by mean return to identify bull/bear/sideways
        stats_df = stats_df.sort_values('mean_return', ascending=False).reset_index(drop=True)
        
        logger.info(f"\nRegime Statistics:\n{stats_df[['state', 'count', 'frequency', 'mean_return', 'std_return']]}")
        
        return stats_df
    
    def _label_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add human-readable labels to regimes.
        
        Args:
            data: DataFrame with regime column
            
        Returns:
            DataFrame with regime_label column
        """
        if self.regime_stats is None:
            return data
        
        # Map states to labels based on returns
        state_to_label = {}
        
        if self.n_states == 3:
            # Standard 3-state model
            bull_state = self.regime_stats.iloc[0]['state']  # Highest returns
            sideways_state = self.regime_stats.iloc[1]['state']  # Medium returns
            bear_state = self.regime_stats.iloc[2]['state']  # Lowest returns
            
            state_to_label = {
                bull_state: 'Bull',
                sideways_state: 'Sideways',
                bear_state: 'Bear'
            }
        elif self.n_states == 4:
            # 4-state model
            state_to_label = {
                self.regime_stats.iloc[0]['state']: 'Strong Bull',
                self.regime_stats.iloc[1]['state']: 'Weak Bull',
                self.regime_stats.iloc[2]['state']: 'Weak Bear',
                self.regime_stats.iloc[3]['state']: 'Strong Bear'
            }
        else:
            # Generic labeling
            for i, row in self.regime_stats.iterrows():
                state_to_label[row['state']] = f'Regime_{i}'
        
        # Apply labels
        data['regime_label'] = data['regime'].map(state_to_label)
        
        logger.info(f"Regime labels: {state_to_label}")
        
        return data
    
    def get_regime_transitions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze regime transitions.
        
        Args:
            data: DataFrame with regime column
            
        Returns:
            DataFrame with transition statistics
        """
        if 'regime' not in data.columns:
            raise ValueError("No regime column found. Run fit_predict() first.")
        
        # Calculate transitions
        data['regime_prev'] = data['regime'].shift(1)
        transitions = data.dropna(subset=['regime_prev'])
        
        # Transition matrix
        transition_counts = pd.crosstab(
            transitions['regime_prev'],
            transitions['regime'],
            normalize='index'
        )
        
        logger.info(f"\nRegime Transition Matrix:\n{transition_counts}")
        
        # Average duration in each regime
        durations = []
        current_regime = data['regime'].iloc[0]
        current_duration = 1
        
        for regime in data['regime'].iloc[1:]:
            if regime == current_regime:
                current_duration += 1
            else:
                durations.append({
                    'regime': current_regime,
                    'duration': current_duration
                })
                current_regime = regime
                current_duration = 1
        
        durations.append({'regime': current_regime, 'duration': current_duration})
        durations_df = pd.DataFrame(durations)
        
        avg_duration = durations_df.groupby('regime')['duration'].agg(['mean', 'median', 'max'])
        logger.info(f"\nAverage Regime Duration (days):\n{avg_duration}")
        
        return transition_counts
    
    def save(self, filepath: str) -> None:
        """
        Save HMM model to disk.
        
        Args:
            filepath: Output file path
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'regime_stats': self.regime_stats,
            'n_states': self.n_states
        }
        
        save_model(model_data, filepath)
        logger.info(f"HMM model saved to {filepath}")
    
    def load(self, filepath: str) -> 'HMMRegimeDetector':
        """
        Load HMM model from disk.
        
        Args:
            filepath: Model file path
            
        Returns:
            Self
        """
        model_data = load_model(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.regime_stats = model_data['regime_stats']
        self.n_states = model_data['n_states']
        self.is_fitted = True
        
        logger.info(f"HMM model loaded from {filepath}")
        
        return self
    
    def get_current_regime(self, data: pd.DataFrame) -> Dict:
        """
        Get current regime and probabilities.
        
        Args:
            data: DataFrame with latest market data
            
        Returns:
            Dictionary with current regime information
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Predict current state
        states = self.predict(data)
        current_state = states[-1]
        
        # Get regime label
        if self.regime_stats is not None:
            regime_info = self.regime_stats[self.regime_stats['state'] == current_state].iloc[0]
        else:
            regime_info = {'state': current_state}
        
        result = {
            'current_state': int(current_state),
            'regime_label': data['regime_label'].iloc[-1] if 'regime_label' in data.columns else f'State_{current_state}',
            'date': data.index[-1],
            'stats': regime_info.to_dict() if isinstance(regime_info, pd.Series) else regime_info
        }
        
        return result


def compare_hmm_models(data: pd.DataFrame,
                      n_states_list: List[int] = [2, 3, 4, 5],
                      feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare HMM models with different numbers of states.
    
    Args:
        data: DataFrame with market data
        n_states_list: List of state counts to try
        feature_columns: Feature columns to use
        
    Returns:
        DataFrame with comparison metrics
    """
    logger.info("Comparing HMM models with different state counts...")
    
    results = []
    
    for n_states in n_states_list:
        logger.info(f"\nTesting {n_states} states...")
        
        hmm_model = HMMRegimeDetector(n_states=n_states)
        data_with_regime = hmm_model.fit_predict(data, feature_columns)
        
        # Prepare features for scoring
        features = hmm_model.prepare_features(data, feature_columns)
        features_scaled = hmm_model.scaler.transform(features)
        
        # Calculate metrics
        log_likelihood = hmm_model.model.score(features_scaled)
        aic = -2 * log_likelihood + 2 * n_states
        bic = -2 * log_likelihood + n_states * np.log(len(data))
        
        results.append({
            'n_states': n_states,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'converged': hmm_model.model.monitor_.converged
        })
    
    results_df = pd.DataFrame(results)
    logger.info(f"\nModel Comparison:\n{results_df}")
    
    # Best model by BIC (lower is better)
    best_idx = results_df['bic'].idxmin()
    best_n_states = results_df.loc[best_idx, 'n_states']
    logger.info(f"\nBest model: {best_n_states} states (by BIC)")
    
    return results_df


if __name__ == "__main__":
    # Test the module
    print("Testing HMM Regime Detection Module")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    # Simulate different regimes
    n = len(dates)
    returns = np.concatenate([
        np.random.normal(0.001, 0.01, n//3),  # Bull
        np.random.normal(-0.001, 0.02, n//3),  # Bear
        np.random.normal(0, 0.01, n - 2*(n//3))  # Sideways
    ])
    
    data = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(returns)),
        'returns': returns,
        'returns_std_20': pd.Series(returns).rolling(20).std().values,
        'volume_ratio': np.random.uniform(0.5, 2, n),
        'rsi': np.random.uniform(30, 70, n)
    }, index=dates)
    
    data = data.dropna()
    
    # Test HMM
    hmm_detector = HMMRegimeDetector(n_states=3)
    data_with_regime = hmm_detector.fit_predict(data)
    
    print(f"\nData shape: {data_with_regime.shape}")
    print(f"\nRegime distribution:\n{data_with_regime['regime_label'].value_counts()}")
    print(f"\nSample data:\n{data_with_regime[['close', 'returns', 'regime', 'regime_label']].tail(10)}")
    
    # Test transitions
    transitions = hmm_detector.get_regime_transitions(data_with_regime)
