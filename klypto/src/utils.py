"""
Utility Functions for Quantitative Trading System
Provides helper functions for data processing, validation, and common operations.
"""

import os
import pickle
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Configure logging
def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to save logs
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('quantitative_trading')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Path utilities
def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Data I/O utilities
def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path], 
                   format: str = 'csv') -> None:
    """
    Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        format: File format ('csv', 'parquet', 'pickle')
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    if format == 'csv':
        df.to_csv(filepath, index=True)
    elif format == 'parquet':
        df.to_parquet(filepath, index=True)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(filepath: Union[str, Path], 
                   format: Optional[str] = None) -> pd.DataFrame:
    """
    Load DataFrame from file.
    
    Args:
        filepath: Input file path
        format: File format (auto-detected if None)
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix[1:]  # Remove leading dot
    
    if format == 'csv':
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif format == 'parquet':
        return pd.read_parquet(filepath)
    elif format == 'pickle':
        return pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """
    Save model to disk using pickle.
    
    Args:
        model: Model object to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: Union[str, Path]) -> Any:
    """
    Load model from disk.
    
    Args:
        filepath: Model file path
        
    Returns:
        Loaded model object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_config(config: Dict, filepath: Union[str, Path]) -> None:
    """
    Save configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(filepath: Union[str, Path]) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        filepath: Config file path
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


# Data validation
def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str],
                       check_nulls: bool = True) -> bool:
    """
    Validate DataFrame has required columns and no nulls.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        check_nulls: Whether to check for null values
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for nulls
    if check_nulls:
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            null_cols = null_counts[null_counts > 0]
            raise ValueError(f"Null values found:\n{null_cols}")
    
    return True


def check_date_continuity(df: pd.DataFrame, 
                          date_column: Optional[str] = None,
                          max_gap_days: int = 5) -> pd.DataFrame:
    """
    Check for gaps in date sequence and report them.
    
    Args:
        df: DataFrame with date index or column
        date_column: Name of date column (None if index)
        max_gap_days: Maximum acceptable gap in days
        
    Returns:
        DataFrame with gap information
    """
    if date_column is None:
        dates = pd.to_datetime(df.index)
    else:
        dates = pd.to_datetime(df[date_column])
    
    date_diffs = dates.to_series().diff()
    gaps = date_diffs[date_diffs > pd.Timedelta(days=max_gap_days)]
    
    if len(gaps) > 0:
        gap_info = pd.DataFrame({
            'gap_start': dates[gaps.index - 1],
            'gap_end': dates[gaps.index],
            'gap_days': gaps.dt.days
        })
        return gap_info
    
    return pd.DataFrame()


# Statistical utilities
def calculate_returns(prices: pd.Series, 
                     method: str = 'simple',
                     periods: int = 1) -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log'
        periods: Number of periods for return calculation
        
    Returns:
        Returns series
    """
    if method == 'simple':
        returns = prices.pct_change(periods=periods)
    elif method == 'log':
        returns = np.log(prices / prices.shift(periods))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return returns


def calculate_volatility(returns: pd.Series, 
                        window: int = 20,
                        annualize: bool = True) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Returns series
        window: Rolling window size
        annualize: Whether to annualize volatility
        
    Returns:
        Volatility series
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)  # 252 trading days
    
    return vol


def normalize_data(data: Union[pd.DataFrame, pd.Series],
                   method: str = 'standard',
                   fit_data: Optional[Union[pd.DataFrame, pd.Series]] = None) -> tuple:
    """
    Normalize data using specified method.
    
    Args:
        data: Data to normalize
        method: 'standard' (z-score) or 'minmax'
        fit_data: Optional data to fit scaler on (for train/test split)
        
    Returns:
        Tuple of (normalized_data, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reshape if Series
    is_series = isinstance(data, pd.Series)
    if is_series:
        data_values = data.values.reshape(-1, 1)
        if fit_data is not None:
            fit_values = fit_data.values.reshape(-1, 1)
    else:
        data_values = data.values
        fit_values = fit_data.values if fit_data is not None else None
    
    # Fit and transform
    if fit_data is not None:
        scaler.fit(fit_values)
    else:
        scaler.fit(data_values)
    
    normalized = scaler.transform(data_values)
    
    # Convert back to original format
    if is_series:
        result = pd.Series(normalized.flatten(), index=data.index, name=data.name)
    else:
        result = pd.DataFrame(normalized, index=data.index, columns=data.columns)
    
    return result, scaler


# Performance metrics
def calculate_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.06,
                          periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    return sharpe


def calculate_sortino_ratio(returns: pd.Series,
                           risk_free_rate: float = 0.06,
                           periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sortino ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return np.nan
    
    sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    return sortino


def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, Any]:
    """
    Calculate maximum drawdown and related metrics.
    
    Args:
        equity_curve: Portfolio value over time
        
    Returns:
        Dictionary with drawdown metrics
    """
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # Find peak before drawdown
    peak_date = equity_curve[:max_dd_date].idxmax()
    peak_value = equity_curve[peak_date]
    trough_value = equity_curve[max_dd_date]
    
    # Find recovery date (if any)
    recovery_date = None
    if max_dd_date < equity_curve.index[-1]:
        future_data = equity_curve[max_dd_date:]
        recovery_mask = future_data >= peak_value
        if recovery_mask.any():
            recovery_date = future_data[recovery_mask].index[0]
    
    return {
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd * 100,
        'peak_date': peak_date,
        'trough_date': max_dd_date,
        'recovery_date': recovery_date,
        'peak_value': peak_value,
        'trough_value': trough_value,
        'drawdown_duration_days': (max_dd_date - peak_date).days if peak_date else 0
    }


def calculate_calmar_ratio(returns: pd.Series,
                          equity_curve: pd.Series,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Args:
        returns: Returns series
        equity_curve: Portfolio value over time
        periods_per_year: Number of periods in a year
        
    Returns:
        Calmar ratio
    """
    annual_return = returns.mean() * periods_per_year
    max_dd = abs(calculate_max_drawdown(equity_curve)['max_drawdown'])
    
    if max_dd == 0:
        return np.nan
    
    return annual_return / max_dd


# Time utilities
def get_trading_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Get trading dates (weekdays) between start and end dates.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        DatetimeIndex of trading dates
    """
    dates = pd.bdate_range(start=start_date, end=end_date)
    return dates


def split_train_test_by_date(df: pd.DataFrame,
                             train_ratio: float = 0.7,
                             validation_ratio: float = 0.15) -> tuple:
    """
    Split data into train/validation/test sets by date.
    
    Args:
        df: DataFrame with datetime index
        train_ratio: Proportion for training
        validation_ratio: Proportion for validation
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df


def walk_forward_split(df: pd.DataFrame, 
                       n_splits: int = 5,
                       train_size: int = 252,
                       test_size: int = 63) -> list:
    """
    Create walk-forward (rolling window) train/test splits.
    
    Args:
        df: DataFrame with datetime index
        n_splits: Number of splits to create
        train_size: Number of periods in training window (252 = ~1 year)
        test_size: Number of periods in test window (63 = ~1 quarter)
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    splits = []
    total_size = train_size + test_size
    step_size = len(df) // (n_splits + 1)
    
    for i in range(n_splits):
        start_idx = i * step_size
        train_end = start_idx + train_size
        test_end = train_end + test_size
        
        if test_end > len(df):
            break
            
        train_indices = range(start_idx, train_end)
        test_indices = range(train_end, test_end)
        
        splits.append((train_indices, test_indices))
    
    return splits


def calculate_statistical_significance(returns: pd.Series, 
                                       benchmark_returns: pd.Series = None) -> Dict:
    """
    Calculate statistical significance tests for trading strategy returns.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns (e.g., buy-and-hold)
        
    Returns:
        Dictionary with statistical test results
    """
    from scipy import stats
    
    results = {}
    
    # T-test: Is mean return significantly different from zero?
    t_stat, p_value = stats.ttest_1samp(returns.dropna(), 0)
    results['t_statistic'] = t_stat
    results['p_value_vs_zero'] = p_value
    results['significant_vs_zero'] = p_value < 0.05
    
    # Shapiro-Wilk test for normality
    if len(returns.dropna()) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(returns.dropna()[:5000])  # Limit to 5000 samples
        results['normality_test_stat'] = shapiro_stat
        results['normality_p_value'] = shapiro_p
        results['returns_normal'] = shapiro_p > 0.05
    
    # Jarque-Bera test for normality (better for larger samples)
    jb_stat, jb_p = stats.jarque_bera(returns.dropna())
    results['jarque_bera_stat'] = jb_stat
    results['jarque_bera_p'] = jb_p
    
    # If benchmark provided, test for alpha
    if benchmark_returns is not None:
        aligned_returns = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned_returns) > 1:
            excess_returns = aligned_returns.iloc[:, 0] - aligned_returns.iloc[:, 1]
            t_stat_alpha, p_value_alpha = stats.ttest_1samp(excess_returns, 0)
            results['alpha_t_statistic'] = t_stat_alpha
            results['alpha_p_value'] = p_value_alpha
            results['significant_alpha'] = p_value_alpha < 0.05
    
    return results


def bootstrap_sharpe_ci(returns: pd.Series, 
                        n_bootstrap: int = 1000,
                        confidence: float = 0.95) -> Dict:
    """
    Calculate confidence interval for Sharpe ratio using bootstrap.
    
    Args:
        returns: Strategy returns
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 = 95%)
        
    Returns:
        Dictionary with Sharpe ratio and confidence intervals
    """
    returns_clean = returns.dropna()
    
    # Original Sharpe
    sharpe = calculate_sharpe_ratio(returns_clean)
    
    # Bootstrap
    sharpe_samples = []
    n = len(returns_clean)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(returns_clean, size=n, replace=True)
        sample_sharpe = (sample.mean() / sample.std()) * np.sqrt(252) if sample.std() > 0 else 0
        sharpe_samples.append(sample_sharpe)
    
    sharpe_samples = np.array(sharpe_samples)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(sharpe_samples, lower_percentile)
    ci_upper = np.percentile(sharpe_samples, upper_percentile)
    
    return {
        'sharpe_ratio': sharpe,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence,
        'bootstrap_samples': n_bootstrap,
        'bootstrap_mean': sharpe_samples.mean(),
        'bootstrap_std': sharpe_samples.std()
    }


def compare_strategies(strategy_results: Dict[str, Dict], 
                      baseline_name: str = 'buy_hold') -> pd.DataFrame:
    """
    Compare multiple trading strategies with statistical tests.
    
    Args:
        strategy_results: Dictionary of {strategy_name: backtest_results}
        baseline_name: Name of baseline strategy for comparison
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for name, results in strategy_results.items():
        metrics = results.get('metrics', {})
        equity_curve = results.get('equity_curve', pd.Series())
        
        row = {
            'Strategy': name,
            'Total Return (%)': metrics.get('total_return_pct', 0),
            'Annual Return (%)': metrics.get('annualized_return_pct', 0),
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Sortino Ratio': metrics.get('sortino_ratio', 0),
            'Max Drawdown (%)': metrics.get('max_drawdown_pct', 0),
            'Calmar Ratio': metrics.get('calmar_ratio', 0),
            'Win Rate (%)': metrics.get('win_rate_pct', 0),
            'Profit Factor': metrics.get('profit_factor', 0),
            'Volatility (%)': metrics.get('volatility_pct', 0),
            'Total Trades': metrics.get('total_trades', 0)
        }
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by Sharpe ratio
    df = df.sort_values('Sharpe Ratio', ascending=False)
    
    return df


# Display utilities
def print_summary_stats(df: pd.DataFrame, title: str = "Summary Statistics") -> None:
    """
    Print formatted summary statistics.
    
    Args:
        df: DataFrame to summarize
        title: Title for the summary
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape}")
    print(f"Date Range: {df.index[0]} to {df.index[-1]}")
    print(f"Total Days: {len(df)}")
    print(f"\nColumns: {', '.join(df.columns)}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDescriptive Statistics:\n{df.describe()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Test utilities
    logger = setup_logging()
    logger.info("Utilities module loaded successfully")
    
    # Test path utilities
    root = get_project_root()
    logger.info(f"Project root: {root}")
