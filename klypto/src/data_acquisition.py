"""
Data Acquisition Module
Handles downloading and preprocessing of market data from NSE and Yahoo Finance.
"""

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from typing import Optional, List, Union
import time

import pandas as pd
import numpy as np
import yfinance as yf

try:
    from nsepy import get_history
    from nsepy.derivatives import get_expiry_date
    NSE_AVAILABLE = True
except ImportError:
    NSE_AVAILABLE = False
    print("Warning: nsepy not available. Using yfinance only.")

from .utils import save_dataframe, setup_logging


logger = setup_logging()


class DataAcquisition:
    """
    Data acquisition class for fetching market data from multiple sources.
    
    Supports NSE India (via nsepy) and Yahoo Finance.
    Handles data cleaning, validation, and storage.
    """
    
    def __init__(self, 
                 symbol: str,
                 start_date: Union[str, datetime],
                 end_date: Union[str, datetime],
                 source: str = 'auto'):
        """
        Initialize DataAcquisition.
        
        Args:
            symbol: Stock symbol (e.g., 'NIFTY', 'RELIANCE', 'TCS')
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime)
            source: Data source ('nse', 'yahoo', 'auto')
        """
        self.symbol = symbol.upper()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.source = source
        self.data = None
        
        # Symbol mapping for different sources
        self.symbol_map = {
            'NIFTY': {
                'nse': 'NIFTY',
                'yahoo': '^NSEI'
            },
            'BANKNIFTY': {
                'nse': 'BANKNIFTY',
                'yahoo': '^NSEBANK'
            },
            'SENSEX': {
                'nse': 'SENSEX',
                'yahoo': '^BSESN'
            }
        }
        
        logger.info(f"Initialized DataAcquisition for {self.symbol}")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
    
    def fetch_from_nse(self) -> Optional[pd.DataFrame]:
        """
        Fetch data from NSE using nsepy.
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not NSE_AVAILABLE:
            logger.warning("nsepy not available")
            return None
        
        try:
            logger.info(f"Fetching data from NSE for {self.symbol}")
            
            # Map symbol if needed
            nse_symbol = self.symbol
            if self.symbol in self.symbol_map:
                nse_symbol = self.symbol_map[self.symbol]['nse']
            
            # Fetch data based on symbol type
            if self.symbol in ['NIFTY', 'BANKNIFTY']:
                # For indices
                from nsepy import get_index_history
                df = get_index_history(
                    symbol=nse_symbol,
                    start=self.start_date,
                    end=self.end_date
                )
            else:
                # For stocks
                df = get_history(
                    symbol=nse_symbol,
                    start=self.start_date,
                    end=self.end_date
                )
            
            if df is not None and not df.empty:
                # Standardize column names
                df = self._standardize_columns(df)
                logger.info(f"Successfully fetched {len(df)} rows from NSE")
                return df
            else:
                logger.warning("No data returned from NSE")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching from NSE: {str(e)}")
            return None
    
    def fetch_from_yahoo(self) -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance.
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            logger.info(f"Fetching data from Yahoo Finance for {self.symbol}")
            
            # Map symbol if needed
            yahoo_symbol = self.symbol
            if self.symbol in self.symbol_map:
                yahoo_symbol = self.symbol_map[self.symbol]['yahoo']
            else:
                # For Indian stocks, add .NS suffix
                if not yahoo_symbol.startswith('^'):
                    yahoo_symbol = f"{yahoo_symbol}.NS"
            
            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False
            )
            
            if df is not None and not df.empty:
                # Standardize column names
                df = self._standardize_columns(df)
                logger.info(f"Successfully fetched {len(df)} rows from Yahoo Finance")
                return df
            else:
                logger.warning("No data returned from Yahoo Finance")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance: {str(e)}")
            return None
    
    def fetch_data(self, retry_count: int = 3) -> pd.DataFrame:
        """
        Fetch data with automatic source selection and retry logic.
        
        Args:
            retry_count: Number of retry attempts
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ValueError: If no data could be fetched
        """
        attempts = 0
        df = None
        
        while attempts < retry_count and df is None:
            if self.source == 'nse' or (self.source == 'auto' and NSE_AVAILABLE):
                df = self.fetch_from_nse()
                if df is not None:
                    break
            
            if self.source == 'yahoo' or self.source == 'auto':
                df = self.fetch_from_yahoo()
                if df is not None:
                    break
            
            attempts += 1
            if df is None and attempts < retry_count:
                logger.warning(f"Retry attempt {attempts + 1}/{retry_count}")
                time.sleep(2)
        
        if df is None:
            raise ValueError(f"Failed to fetch data for {self.symbol} after {retry_count} attempts")
        
        # Clean and validate data
        df = self._clean_data(df)
        self.data = df
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across different data sources.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            DataFrame with standardized columns
        """
        # Column mapping
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            'Turnover': 'turnover',
            'Trades': 'trades',
            'Deliverable Volume': 'deliverable_volume',
            '%Deliverble': 'pct_deliverable'
        }
        
        # Rename columns
        df = df.rename(columns=column_map)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        # Remove duplicates
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} duplicate rows")
        
        # Handle missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            # Forward fill then backward fill
            df = df.ffill().bfill()
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['close']) |
            (df['high'] < df['open']) |
            (df['low'] > df['close']) |
            (df['low'] > df['open'])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
            # Fix by adjusting high/low
            df.loc[invalid_ohlc, 'high'] = df.loc[invalid_ohlc, [
                'open', 'high', 'low', 'close'
            ]].max(axis=1)
            df.loc[invalid_ohlc, 'low'] = df.loc[invalid_ohlc, [
                'open', 'high', 'low', 'close'
            ]].min(axis=1)
        
        # Remove zero volume days (if not an index)
        if 'volume' in df.columns and self.symbol not in ['NIFTY', 'BANKNIFTY', 'SENSEX']:
            zero_vol = df['volume'] == 0
            if zero_vol.any():
                logger.warning(f"Removing {zero_vol.sum()} zero volume days")
                df = df[~zero_vol]
        
        # Remove extreme outliers (price changes > 50% in a day)
        if len(df) > 1:
            price_change = df['close'].pct_change().abs()
            extreme = price_change > 0.5
            if extreme.any():
                logger.warning(f"Found {extreme.sum()} days with extreme price changes (>50%)")
                # Don't remove, but flag for investigation
                df.loc[extreme, 'outlier_flag'] = True
        
        logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        
        return df
    
    def save_data(self, filepath: str, format: str = 'csv') -> None:
        """
        Save fetched data to file.
        
        Args:
            filepath: Output file path
            format: File format ('csv', 'parquet', 'pickle')
        """
        if self.data is None:
            raise ValueError("No data to save. Call fetch_data() first.")
        
        save_dataframe(self.data, filepath, format=format)
        logger.info(f"Data saved to {filepath}")
    
    def get_summary(self) -> dict:
        """
        Get summary statistics of the data.
        
        Returns:
            Dictionary with summary information
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        df = self.data
        
        summary = {
            'symbol': self.symbol,
            'start_date': df.index[0],
            'end_date': df.index[-1],
            'total_days': len(df),
            'trading_days': len(df),
            'columns': list(df.columns),
            'price_range': {
                'min': df['close'].min(),
                'max': df['close'].max(),
                'mean': df['close'].mean()
            },
            'volume_stats': {
                'mean': df['volume'].mean() if 'volume' in df.columns else None,
                'total': df['volume'].sum() if 'volume' in df.columns else None
            },
            'missing_values': df.isnull().sum().to_dict(),
            'outliers': df.get('outlier_flag', pd.Series(False, index=df.index)).sum()
        }
        
        return summary


def fetch_multiple_symbols(symbols: List[str],
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime],
                           output_dir: Optional[str] = None) -> dict:
    """
    Fetch data for multiple symbols.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date
        end_date: End date
        output_dir: Optional directory to save data
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    results = {}
    
    for symbol in symbols:
        try:
            logger.info(f"Processing {symbol}...")
            da = DataAcquisition(symbol, start_date, end_date)
            df = da.fetch_data()
            results[symbol] = df
            
            if output_dir:
                filepath = f"{output_dir}/{symbol}_data.csv"
                da.save_data(filepath)
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {str(e)}")
            results[symbol] = None
    
    return results


if __name__ == "__main__":
    # Test the module
    print("Testing Data Acquisition Module")
    print("=" * 60)
    
    # Test with NIFTY
    da = DataAcquisition(
        symbol='NIFTY',
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    df = da.fetch_data()
    print(f"\nFetched {len(df)} rows for NIFTY")
    print(df.head())
    print(df.tail())
    
    summary = da.get_summary()
    print(f"\nSummary: {summary}")
