"""
Visualization and Reporting Module
Creates plots and reports for analysis and backtesting results.
"""

import warnings
warnings.filterwarnings('ignore')

from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .utils import setup_logging, ensure_dir

logger = setup_logging()

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """
    Visualization class for trading system analysis.
    """
    
    def __init__(self, output_dir: str = 'plots'):
        """
        Initialize Visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        
        logger.info(f"Initialized Visualizer, output dir: {self.output_dir}")
    
    def plot_price_and_signals(self,
                               data: pd.DataFrame,
                               signals: Optional[pd.Series] = None,
                               title: str = 'Price and Trading Signals',
                               filename: Optional[str] = None) -> None:
        """Plot price with trading signals."""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot price
        ax.plot(data.index, data['close'], label='Close Price', linewidth=1.5)
        
        # Plot signals if provided
        if signals is not None:
            buy_signals = data[signals == 1]
            sell_signals = data[signals == -1]
            
            ax.scatter(buy_signals.index, buy_signals['close'],
                      marker='^', color='green', s=100, label='Buy', zorder=5)
            ax.scatter(sell_signals.index, sell_signals['close'],
                      marker='v', color='red', s=100, label='Sell', zorder=5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        plt.close()
    
    def plot_equity_curve(self,
                         equity_curve: pd.Series,
                         title: str = 'Equity Curve',
                         filename: Optional[str] = None) -> None:
        """Plot equity curve."""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        ax.plot(equity_curve.index, equity_curve.values, linewidth=2, color='blue')
        ax.fill_between(equity_curve.index, equity_curve.values,
                        alpha=0.3, color='blue')
        
        # Add drawdown shading
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        ax2 = ax.twinx()
        ax2.fill_between(equity_curve.index, 0, drawdown * 100,
                        alpha=0.3, color='red', label='Drawdown')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        plt.close()
    
    def plot_returns_distribution(self,
                                 returns: pd.Series,
                                 title: str = 'Returns Distribution',
                                 filename: Optional[str] = None) -> None:
        """Plot returns distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(returns.dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(returns.mean(), color='red', linestyle='--',
                   label=f'Mean: {returns.mean():.4f}')
        ax1.axvline(returns.median(), color='green', linestyle='--',
                   label=f'Median: {returns.median():.4f}')
        ax1.set_title('Returns Histogram', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Returns', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        plt.close()
    
    def plot_regime_analysis(self,
                            data: pd.DataFrame,
                            title: str = 'Regime Analysis',
                            filename: Optional[str] = None) -> None:
        """Plot HMM regime detection results."""
        if 'regime' not in data.columns:
            logger.warning("No regime column found")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Plot 1: Price with regime coloring
        for regime in data['regime'].unique():
            regime_data = data[data['regime'] == regime]
            label = regime_data['regime_label'].iloc[0] if 'regime_label' in data.columns else f'Regime {regime}'
            axes[0].scatter(regime_data.index, regime_data['close'],
                          s=10, label=label, alpha=0.6)
        
        axes[0].set_title('Price by Market Regime', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Close Price', fontsize=10)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Returns with regime coloring
        if 'returns' in data.columns:
            for regime in data['regime'].unique():
                regime_data = data[data['regime'] == regime]
                label = regime_data['regime_label'].iloc[0] if 'regime_label' in data.columns else f'Regime {regime}'
                axes[1].scatter(regime_data.index, regime_data['returns'],
                              s=10, label=label, alpha=0.6)
            
            axes[1].set_title('Returns by Market Regime', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Returns', fontsize=10)
            axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Regime timeline
        axes[2].plot(data.index, data['regime'], linewidth=2)
        axes[2].set_title('Regime Timeline', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Date', fontsize=10)
        axes[2].set_ylabel('Regime State', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        plt.close()
    
    def plot_feature_importance(self,
                               importance_df: pd.DataFrame,
                               top_n: int = 20,
                               title: str = 'Feature Importance',
                               filename: Optional[str] = None) -> None:
        """Plot feature importance."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_df.head(top_n)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        plt.close()
    
    def plot_correlation_matrix(self,
                               data: pd.DataFrame,
                               title: str = 'Feature Correlation Matrix',
                               filename: Optional[str] = None) -> None:
        """Plot correlation matrix."""
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr = data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        plt.close()
    
    def plot_ema_analysis(self,
                         data: pd.DataFrame,
                         ema_periods: List[int] = [20, 50, 200],
                         title: str = 'EMA Analysis',
                         filename: Optional[str] = None) -> None:
        """Plot price with multiple EMAs."""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot price
        ax.plot(data.index, data['close'], label='Close Price',
               linewidth=1.5, alpha=0.8)
        
        # Plot EMAs
        colors = ['orange', 'blue', 'red', 'green', 'purple']
        for i, period in enumerate(ema_periods):
            ema_col = f'ema_{period}'
            if ema_col in data.columns:
                ax.plot(data.index, data[ema_col],
                       label=f'EMA {period}',
                       linewidth=1.5,
                       alpha=0.7,
                       color=colors[i % len(colors)])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        plt.close()
    
    def plot_outliers(self,
                     data: pd.DataFrame,
                     outliers: pd.DataFrame,
                     column: str = 'returns',
                     title: str = 'Outlier Detection',
                     filename: Optional[str] = None) -> None:
        """Plot outliers in data."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot 1: Time series with outliers
        ax1.plot(data.index, data[column], label='Data', alpha=0.7)
        
        if 'is_outlier' in outliers.columns:
            outlier_points = data[outliers['is_outlier']]
            ax1.scatter(outlier_points.index, outlier_points[column],
                       color='red', s=50, label='Outliers', zorder=5)
        
        ax1.set_title(f'{column.capitalize()} with Outliers', fontsize=12, fontweight='bold')
        ax1.set_ylabel(column.capitalize(), fontsize=10)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot
        ax2.boxplot([data[column].dropna()], vert=False)
        ax2.set_title('Box Plot', fontsize=12, fontweight='bold')
        ax2.set_xlabel(column.capitalize(), fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        plt.close()
    
    def create_backtest_report(self,
                              results: Dict,
                              data: pd.DataFrame,
                              prefix: str = 'backtest') -> None:
        """Create complete backtest report with multiple plots."""
        logger.info("Creating comprehensive backtest report...")
        
        # 1. Equity curve
        self.plot_equity_curve(
            results['equity_curve'],
            title='Portfolio Equity Curve',
            filename=f'{prefix}_equity_curve.png'
        )
        
        # 2. Returns distribution
        returns = results['equity_curve'].pct_change().dropna()
        self.plot_returns_distribution(
            returns,
            title='Portfolio Returns Distribution',
            filename=f'{prefix}_returns_dist.png'
        )
        
        # 3. Price and signals
        signals = pd.Series(0, index=data.index)
        if len(results['trades']) > 0:
            for _, trade in results['trades'].iterrows():
                if trade['type'] == 'BUY':
                    signals.loc[trade['date']] = 1
                elif 'SELL' in trade['type']:
                    signals.loc[trade['date']] = -1
        
        self.plot_price_and_signals(
            data,
            signals,
            title='Price with Trading Signals',
            filename=f'{prefix}_signals.png'
        )
        
        logger.info(f"Backtest report created with prefix '{prefix}'")


if __name__ == "__main__":
    # Test the module
    print("Testing Visualization Module")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    returns = np.random.normal(0.001, 0.02, len(dates))
    
    data = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(returns)),
        'returns': returns
    }, index=dates)
    
    # Create visualizer
    viz = Visualizer(output_dir='test_plots')
    
    # Test equity curve
    equity = pd.Series(1000000 * np.exp(np.cumsum(returns * 10)), index=dates)
    viz.plot_equity_curve(equity, filename='test_equity.png')
    
    # Test returns distribution
    viz.plot_returns_distribution(data['returns'], filename='test_returns.png')
    
    print("\nTest plots created in 'test_plots' directory")
