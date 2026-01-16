"""
Feature Engineering Module
Implements EMA-based features and comprehensive technical indicators.
"""

import warnings
warnings.filterwarnings('ignore')

from typing import List, Optional, Dict
import pandas as pd
import numpy as np

try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using pandas implementation.")

from .utils import setup_logging

logger = setup_logging()


class FeatureEngineer:
    """
    Feature engineering class for creating technical indicators and EMA-based features.
    
    Focuses on:
    - Exponential Moving Averages (EMA)
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Price-based features
    - Volume-based features
    - Volatility features
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize FeatureEngineer.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Validate input data
        missing = set(self.required_columns) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Initialized FeatureEngineer with {len(self.data)} rows")
    
    # ============= EMA Features =============
    
    def add_ema_features(self, 
                        periods: List[int] = [5, 10, 20, 50, 100, 200]) -> pd.DataFrame:
        """
        Add Exponential Moving Average features.
        
        Args:
            periods: List of EMA periods
            
        Returns:
            DataFrame with EMA features
        """
        logger.info(f"Adding EMA features for periods: {periods}")
        
        for period in periods:
            col_name = f'ema_{period}'
            if TALIB_AVAILABLE:
                self.data[col_name] = ta.EMA(self.data['close'], timeperiod=period)
            else:
                self.data[col_name] = self.data['close'].ewm(
                    span=period, adjust=False
                ).mean()
        
        return self.data
    
    def add_ema_crossovers(self, 
                          fast_periods: List[int] = [5, 10, 20],
                          slow_periods: List[int] = [20, 50, 100]) -> pd.DataFrame:
        """
        Add EMA crossover signals.
        
        Args:
            fast_periods: Fast EMA periods
            slow_periods: Slow EMA periods
            
        Returns:
            DataFrame with crossover features
        """
        logger.info("Adding EMA crossover features")
        
        for fast in fast_periods:
            for slow in slow_periods:
                if fast >= slow:
                    continue
                
                fast_col = f'ema_{fast}'
                slow_col = f'ema_{slow}'
                
                # Ensure EMAs exist
                if fast_col not in self.data.columns:
                    self.add_ema_features([fast])
                if slow_col not in self.data.columns:
                    self.add_ema_features([slow])
                
                # Crossover difference
                cross_col = f'ema_cross_{fast}_{slow}'
                self.data[cross_col] = self.data[fast_col] - self.data[slow_col]
                
                # Crossover percentage
                cross_pct = f'ema_cross_pct_{fast}_{slow}'
                self.data[cross_pct] = (
                    (self.data[fast_col] - self.data[slow_col]) / 
                    self.data[slow_col] * 100
                )
                
                # Signal (1 = fast above slow, -1 = fast below slow)
                signal_col = f'ema_signal_{fast}_{slow}'
                self.data[signal_col] = np.where(
                    self.data[fast_col] > self.data[slow_col], 1, -1
                )
        
        return self.data
    
    def add_ema_distance(self, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """
        Add distance of price from EMA.
        
        Args:
            periods: EMA periods to calculate distance from
            
        Returns:
            DataFrame with distance features
        """
        logger.info("Adding EMA distance features")
        
        for period in periods:
            ema_col = f'ema_{period}'
            
            if ema_col not in self.data.columns:
                self.add_ema_features([period])
            
            # Absolute distance
            dist_col = f'price_ema_{period}_dist'
            self.data[dist_col] = self.data['close'] - self.data[ema_col]
            
            # Percentage distance
            dist_pct = f'price_ema_{period}_dist_pct'
            self.data[dist_pct] = (
                (self.data['close'] - self.data[ema_col]) / 
                self.data[ema_col] * 100
            )
        
        return self.data
    
    def add_ema_slope(self, periods: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        Add EMA slope (rate of change).
        
        Args:
            periods: EMA periods to calculate slope for
            
        Returns:
            DataFrame with slope features
        """
        logger.info("Adding EMA slope features")
        
        for period in periods:
            ema_col = f'ema_{period}'
            
            if ema_col not in self.data.columns:
                self.add_ema_features([period])
            
            # Rate of change
            slope_col = f'ema_{period}_slope'
            self.data[slope_col] = self.data[ema_col].pct_change()
            
            # Acceleration (second derivative)
            accel_col = f'ema_{period}_accel'
            self.data[accel_col] = self.data[slope_col].diff()
        
        return self.data
    
    # ============= Momentum Indicators =============
    
    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index.
        
        Args:
            period: RSI period
            
        Returns:
            DataFrame with RSI
        """
        logger.info(f"Adding RSI with period {period}")
        
        if TALIB_AVAILABLE:
            self.data['rsi'] = ta.RSI(self.data['close'], timeperiod=period)
        else:
            # Calculate RSI manually
            delta = self.data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI-based signals
        self.data['rsi_oversold'] = (self.data['rsi'] < 30).astype(int)
        self.data['rsi_overbought'] = (self.data['rsi'] > 70).astype(int)
        
        return self.data
    
    def add_macd(self, 
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).
        
        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with MACD features
        """
        logger.info(f"Adding MACD ({fast}, {slow}, {signal})")
        
        if TALIB_AVAILABLE:
            macd, signal_line, hist = ta.MACD(
                self.data['close'],
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal
            )
            self.data['macd'] = macd
            self.data['macd_signal'] = signal_line
            self.data['macd_hist'] = hist
        else:
            # Calculate MACD manually
            ema_fast = self.data['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = self.data['close'].ewm(span=slow, adjust=False).mean()
            self.data['macd'] = ema_fast - ema_slow
            self.data['macd_signal'] = self.data['macd'].ewm(span=signal, adjust=False).mean()
            self.data['macd_hist'] = self.data['macd'] - self.data['macd_signal']
        
        # MACD crossover signal
        self.data['macd_cross_signal'] = np.where(
            self.data['macd'] > self.data['macd_signal'], 1, -1
        )
        
        return self.data
    
    def add_stochastic(self, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator.
        
        Args:
            k_period: %K period
            d_period: %D period
            
        Returns:
            DataFrame with Stochastic features
        """
        logger.info(f"Adding Stochastic ({k_period}, {d_period})")
        
        if TALIB_AVAILABLE:
            slowk, slowd = ta.STOCH(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                fastk_period=k_period,
                slowk_period=d_period,
                slowd_period=d_period
            )
            self.data['stoch_k'] = slowk
            self.data['stoch_d'] = slowd
        else:
            # Calculate Stochastic manually
            low_min = self.data['low'].rolling(window=k_period).min()
            high_max = self.data['high'].rolling(window=k_period).max()
            self.data['stoch_k'] = 100 * (
                (self.data['close'] - low_min) / (high_max - low_min)
            )
            self.data['stoch_d'] = self.data['stoch_k'].rolling(window=d_period).mean()
        
        return self.data
    
    # ============= Volatility Indicators =============
    
    def add_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands.
        
        Args:
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Bands
        """
        logger.info(f"Adding Bollinger Bands ({period}, {std_dev})")
        
        if TALIB_AVAILABLE:
            upper, middle, lower = ta.BBANDS(
                self.data['close'],
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev
            )
            self.data['bb_upper'] = upper
            self.data['bb_middle'] = middle
            self.data['bb_lower'] = lower
        else:
            # Calculate manually
            self.data['bb_middle'] = self.data['close'].rolling(window=period).mean()
            std = self.data['close'].rolling(window=period).std()
            self.data['bb_upper'] = self.data['bb_middle'] + (std_dev * std)
            self.data['bb_lower'] = self.data['bb_middle'] - (std_dev * std)
        
        # Bandwidth and %B
        self.data['bb_width'] = (
            (self.data['bb_upper'] - self.data['bb_lower']) / 
            self.data['bb_middle'] * 100
        )
        self.data['bb_pct'] = (
            (self.data['close'] - self.data['bb_lower']) / 
            (self.data['bb_upper'] - self.data['bb_lower'])
        )
        
        return self.data
    
    def add_atr(self, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range.
        
        Args:
            period: ATR period
            
        Returns:
            DataFrame with ATR
        """
        logger.info(f"Adding ATR with period {period}")
        
        if TALIB_AVAILABLE:
            self.data['atr'] = ta.ATR(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                timeperiod=period
            )
        else:
            # Calculate True Range
            high_low = self.data['high'] - self.data['low']
            high_close = np.abs(self.data['high'] - self.data['close'].shift())
            low_close = np.abs(self.data['low'] - self.data['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            self.data['atr'] = tr.rolling(window=period).mean()
        
        # ATR percentage
        self.data['atr_pct'] = self.data['atr'] / self.data['close'] * 100
        
        return self.data
    
    # ============= Volume Indicators =============
    
    def add_volume_features(self) -> pd.DataFrame:
        """
        Add volume-based features.
        
        Returns:
            DataFrame with volume features
        """
        logger.info("Adding volume features")
        
        # Volume moving averages
        self.data['volume_sma_20'] = self.data['volume'].rolling(window=20).mean()
        self.data['volume_sma_50'] = self.data['volume'].rolling(window=50).mean()
        
        # Volume ratio
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_sma_20']
        
        # Volume change
        self.data['volume_change'] = self.data['volume'].pct_change()
        
        # On-Balance Volume (OBV)
        obv = np.where(
            self.data['close'] > self.data['close'].shift(1),
            self.data['volume'],
            np.where(
                self.data['close'] < self.data['close'].shift(1),
                -self.data['volume'],
                0
            )
        )
        self.data['obv'] = pd.Series(obv, index=self.data.index).cumsum()
        
        # Price-Volume Trend
        self.data['pvt'] = (
            (self.data['close'].pct_change() * self.data['volume']).cumsum()
        )
        
        return self.data
    
    # ============= Price-Based Features =============
    
    def add_price_features(self) -> pd.DataFrame:
        """
        Add price-based features.
        
        Returns:
            DataFrame with price features
        """
        logger.info("Adding price features")
        
        # Returns
        self.data['returns'] = self.data['close'].pct_change()
        self.data['log_returns'] = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # Intraday range
        self.data['high_low_range'] = self.data['high'] - self.data['low']
        self.data['high_low_pct'] = (
            (self.data['high'] - self.data['low']) / self.data['close'] * 100
        )
        
        # Gap
        self.data['gap'] = self.data['open'] - self.data['close'].shift(1)
        self.data['gap_pct'] = (
            (self.data['open'] - self.data['close'].shift(1)) / 
            self.data['close'].shift(1) * 100
        )
        
        # Close position
        self.data['close_position'] = (
            (self.data['close'] - self.data['low']) / 
            (self.data['high'] - self.data['low'])
        )
        
        # Rolling statistics
        for window in [5, 10, 20]:
            self.data[f'returns_mean_{window}'] = (
                self.data['returns'].rolling(window=window).mean()
            )
            self.data[f'returns_std_{window}'] = (
                self.data['returns'].rolling(window=window).std()
            )
        
        return self.data
    
    def add_lagged_features(self, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Add lagged features.
        
        Args:
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        logger.info(f"Adding lagged features for lags: {lags}")
        
        for lag in lags:
            self.data[f'close_lag_{lag}'] = self.data['close'].shift(lag)
            self.data[f'returns_lag_{lag}'] = self.data['returns'].shift(lag)
            self.data[f'volume_lag_{lag}'] = self.data['volume'].shift(lag)
        
        return self.data
    
    # ============= Master Function =============
    
    def create_all_features(self,
                           ema_periods: List[int] = [5, 10, 20, 50, 100, 200],
                           add_lagged: bool = True) -> pd.DataFrame:
        """
        Create all features at once.
        
        Args:
            ema_periods: EMA periods to use
            add_lagged: Whether to add lagged features
            
        Returns:
            DataFrame with all features
        """
        logger.info("Creating all features...")
        
        # EMA features
        self.add_ema_features(ema_periods)
        self.add_ema_crossovers()
        self.add_ema_distance()
        self.add_ema_slope()
        
        # Momentum indicators
        self.add_rsi()
        self.add_macd()
        self.add_stochastic()
        
        # Volatility indicators
        self.add_bollinger_bands()
        self.add_atr()
        
        # Volume features
        self.add_volume_features()
        
        # Price features
        self.add_price_features()
        
        # Lagged features
        if add_lagged:
            self.add_lagged_features()
        
        # Remove rows with NaN (from indicators)
        initial_len = len(self.data)
        self.data = self.data.dropna()
        logger.info(f"Removed {initial_len - len(self.data)} rows with NaN values")
        
        logger.info(f"Feature engineering complete. Shape: {self.data.shape}")
        logger.info(f"Total features: {len(self.data.columns)}")
        
        return self.data
    
    def get_feature_names(self, category: Optional[str] = None) -> List[str]:
        """
        Get list of feature names, optionally filtered by category.
        
        Args:
            category: Feature category ('ema', 'momentum', 'volatility', 'volume', 'price')
            
        Returns:
            List of feature names
        """
        if category is None:
            return list(self.data.columns)
        
        category_prefixes = {
            'ema': ['ema_'],
            'momentum': ['rsi', 'macd', 'stoch'],
            'volatility': ['bb_', 'atr'],
            'volume': ['volume', 'obv', 'pvt'],
            'price': ['returns', 'gap', 'close_position', 'high_low']
        }
        
        prefixes = category_prefixes.get(category, [])
        features = [
            col for col in self.data.columns 
            if any(col.startswith(prefix) for prefix in prefixes)
        ]
        
        return features


if __name__ == "__main__":
    # Test the module
    print("Testing Feature Engineering Module")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 105,
        'low': np.random.randn(len(dates)).cumsum() + 95,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Ensure OHLC relationships
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    fe = FeatureEngineer(data)
    df_features = fe.create_all_features()
    
    print(f"\nOriginal shape: {data.shape}")
    print(f"Feature shape: {df_features.shape}")
    print(f"\nFeature columns:\n{df_features.columns.tolist()}")
    print(f"\nSample data:\n{df_features.tail()}")
