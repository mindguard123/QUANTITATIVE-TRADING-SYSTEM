"""
Outlier Detection Module
Statistical methods for detecting and handling anomalies in financial time series.
"""

import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from .utils import setup_logging

logger = setup_logging()


class OutlierDetector:
    """
    Outlier detection for financial time series data.
    
    Methods:
    - Z-score
    - Modified Z-score
    - IQR (Interquartile Range)
    - Isolation Forest
    - Statistical tests
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Outlier Detector.
        
        Args:
            data: DataFrame with market data
        """
        self.data = data.copy()
        self.outliers = pd.DataFrame(index=self.data.index)
        
        logger.info(f"Initialized OutlierDetector with {len(self.data)} rows")
    
    def detect_zscore(self,
                     columns: Optional[List[str]] = None,
                     threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.
        
        Args:
            columns: Columns to check (default: returns, volume)
            threshold: Z-score threshold
            
        Returns:
            DataFrame with outlier flags
        """
        if columns is None:
            columns = ['returns', 'volume'] if 'returns' in self.data.columns else ['close']
        
        logger.info(f"Detecting outliers using Z-score (threshold={threshold})")
        
        for col in columns:
            if col not in self.data.columns:
                continue
            
            z_scores = np.abs(stats.zscore(self.data[col].dropna()))
            outlier_col = f'{col}_zscore_outlier'
            self.outliers[outlier_col] = False
            self.outliers.loc[self.data[col].notna(), outlier_col] = z_scores > threshold
            
            n_outliers = self.outliers[outlier_col].sum()
            logger.info(f"  {col}: {n_outliers} outliers ({n_outliers/len(self.data)*100:.2f}%)")
        
        return self.outliers
    
    def detect_modified_zscore(self,
                              columns: Optional[List[str]] = None,
                              threshold: float = 3.5) -> pd.DataFrame:
        """
        Detect outliers using Modified Z-score (using median).
        
        Args:
            columns: Columns to check
            threshold: Modified Z-score threshold
            
        Returns:
            DataFrame with outlier flags
        """
        if columns is None:
            columns = ['returns', 'volume'] if 'returns' in self.data.columns else ['close']
        
        logger.info(f"Detecting outliers using Modified Z-score (threshold={threshold})")
        
        for col in columns:
            if col not in self.data.columns:
                continue
            
            median = self.data[col].median()
            mad = np.median(np.abs(self.data[col] - median))
            modified_z = 0.6745 * (self.data[col] - median) / mad if mad != 0 else 0
            
            outlier_col = f'{col}_mod_zscore_outlier'
            self.outliers[outlier_col] = np.abs(modified_z) > threshold
            
            n_outliers = self.outliers[outlier_col].sum()
            logger.info(f"  {col}: {n_outliers} outliers ({n_outliers/len(self.data)*100:.2f}%)")
        
        return self.outliers
    
    def detect_iqr(self,
                  columns: Optional[List[str]] = None,
                  multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using IQR method.
        
        Args:
            columns: Columns to check
            multiplier: IQR multiplier (1.5 for outliers, 3.0 for extreme outliers)
            
        Returns:
            DataFrame with outlier flags
        """
        if columns is None:
            columns = ['returns', 'volume'] if 'returns' in self.data.columns else ['close']
        
        logger.info(f"Detecting outliers using IQR (multiplier={multiplier})")
        
        for col in columns:
            if col not in self.data.columns:
                continue
            
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outlier_col = f'{col}_iqr_outlier'
            self.outliers[outlier_col] = (
                (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            )
            
            n_outliers = self.outliers[outlier_col].sum()
            logger.info(f"  {col}: {n_outliers} outliers ({n_outliers/len(self.data)*100:.2f}%)")
        
        return self.outliers
    
    def detect_isolation_forest(self,
                               columns: Optional[List[str]] = None,
                               contamination: float = 0.01) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest.
        
        Args:
            columns: Columns to check
            contamination: Expected proportion of outliers
            
        Returns:
            DataFrame with outlier flags
        """
        if columns is None:
            columns = ['returns', 'volume'] if 'returns' in self.data.columns else ['close']
        
        logger.info(f"Detecting outliers using Isolation Forest (contamination={contamination})")
        
        # Select columns that exist
        valid_cols = [col for col in columns if col in self.data.columns]
        
        if not valid_cols:
            logger.warning("No valid columns found for Isolation Forest")
            return self.outliers
        
        # Prepare data
        X = self.data[valid_cols].ffill().fillna(0)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        predictions = iso_forest.fit_predict(X)
        
        self.outliers['isolation_forest_outlier'] = predictions == -1
        
        n_outliers = self.outliers['isolation_forest_outlier'].sum()
        logger.info(f"  Detected {n_outliers} outliers ({n_outliers/len(self.data)*100:.2f}%)")
        
        return self.outliers
    
    def detect_price_gaps(self, threshold_pct: float = 5.0) -> pd.DataFrame:
        """
        Detect large price gaps.
        
        Args:
            threshold_pct: Gap threshold in percentage
            
        Returns:
            DataFrame with gap outlier flags
        """
        logger.info(f"Detecting price gaps (threshold={threshold_pct}%)")
        
        if 'gap_pct' not in self.data.columns:
            # Calculate gap
            self.data['gap_pct'] = (
                (self.data['open'] - self.data['close'].shift(1)) /
                self.data['close'].shift(1) * 100
            )
        
        self.outliers['gap_outlier'] = np.abs(self.data['gap_pct']) > threshold_pct
        
        n_outliers = self.outliers['gap_outlier'].sum()
        logger.info(f"  Detected {n_outliers} gap outliers ({n_outliers/len(self.data)*100:.2f}%)")
        
        return self.outliers
    
    def detect_volume_spikes(self, threshold: float = 5.0) -> pd.DataFrame:
        """
        Detect volume spikes.
        
        Args:
            threshold: Multiple of average volume
            
        Returns:
            DataFrame with volume spike flags
        """
        logger.info(f"Detecting volume spikes (threshold={threshold}x average)")
        
        if 'volume' not in self.data.columns:
            logger.warning("Volume column not found")
            return self.outliers
        
        avg_volume = self.data['volume'].rolling(window=20).mean()
        self.outliers['volume_spike'] = self.data['volume'] > (avg_volume * threshold)
        
        n_spikes = self.outliers['volume_spike'].sum()
        logger.info(f"  Detected {n_spikes} volume spikes ({n_spikes/len(self.data)*100:.2f}%)")
        
        return self.outliers
    
    def detect_all(self) -> pd.DataFrame:
        """
        Run all outlier detection methods.
        
        Returns:
            DataFrame with all outlier flags
        """
        logger.info("Running all outlier detection methods...")
        
        self.detect_zscore()
        self.detect_modified_zscore()
        self.detect_iqr()
        self.detect_isolation_forest()
        self.detect_price_gaps()
        self.detect_volume_spikes()
        
        # Create combined outlier flag
        outlier_cols = [col for col in self.outliers.columns if 'outlier' in col or 'spike' in col]
        self.outliers['is_outlier'] = self.outliers[outlier_cols].any(axis=1)
        
        total_outliers = self.outliers['is_outlier'].sum()
        logger.info(f"\nTotal outliers detected: {total_outliers} ({total_outliers/len(self.data)*100:.2f}%)")
        
        return self.outliers
    
    def handle_outliers(self,
                       method: str = 'winsorize',
                       columns: Optional[List[str]] = None,
                       percentile: Tuple[float, float] = (1, 99)) -> pd.DataFrame:
        """
        Handle outliers in the data.
        
        Args:
            method: 'remove', 'winsorize', 'cap', 'fill'
            columns: Columns to handle
            percentile: Percentile limits for winsorization/capping
            
        Returns:
            DataFrame with handled outliers
        """
        logger.info(f"Handling outliers using method: {method}")
        
        if columns is None:
            columns = ['returns', 'volume'] if 'returns' in self.data.columns else []
        
        data_clean = self.data.copy()
        
        for col in columns:
            if col not in data_clean.columns:
                continue
            
            if method == 'remove':
                # Remove rows with outliers
                if f'{col}_zscore_outlier' in self.outliers.columns:
                    mask = ~self.outliers[f'{col}_zscore_outlier']
                    data_clean = data_clean[mask]
            
            elif method == 'winsorize':
                # Cap at percentiles
                lower = data_clean[col].quantile(percentile[0] / 100)
                upper = data_clean[col].quantile(percentile[1] / 100)
                data_clean[col] = data_clean[col].clip(lower, upper)
            
            elif method == 'cap':
                # Cap at mean ± 3*std
                mean = data_clean[col].mean()
                std = data_clean[col].std()
                lower = mean - 3 * std
                upper = mean + 3 * std
                data_clean[col] = data_clean[col].clip(lower, upper)
            
            elif method == 'fill':
                # Replace with forward fill
                if f'{col}_zscore_outlier' in self.outliers.columns:
                    outlier_mask = self.outliers[f'{col}_zscore_outlier']
                    data_clean.loc[outlier_mask, col] = np.nan
                    data_clean[col] = data_clean[col].ffill()
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Outlier handling complete. New shape: {data_clean.shape}")
        
        return data_clean
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of outlier detection results.
        
        Returns:
            DataFrame with outlier statistics
        """
        if self.outliers.empty:
            logger.warning("No outliers detected yet. Run detect methods first.")
            return pd.DataFrame()
        
        outlier_cols = [col for col in self.outliers.columns if 'outlier' in col or 'spike' in col]
        
        summary = pd.DataFrame({
            'method': outlier_cols,
            'count': [self.outliers[col].sum() for col in outlier_cols],
            'percentage': [self.outliers[col].sum() / len(self.outliers) * 100 for col in outlier_cols]
        })
        
        return summary


if __name__ == "__main__":
    # Test the module
    print("Testing Outlier Detection Module")
    print("=" * 60)
    
    # Create sample data with outliers
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    returns = np.random.normal(0.001, 0.02, len(dates))
    # Add some outliers
    outlier_indices = np.random.choice(len(returns), 20, replace=False)
    returns[outlier_indices] = returns[outlier_indices] * 5
    
    data = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(returns)),
        'returns': returns,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Detect outliers
    detector = OutlierDetector(data)
    outliers = detector.detect_all()
    
    print(f"\nOutlier Summary:\n{detector.get_summary()}")
    
    # Handle outliers
    data_clean = detector.handle_outliers(method='winsorize')
    print(f"\nOriginal shape: {data.shape}")
    print(f"Clean shape: {data_clean.shape}")
