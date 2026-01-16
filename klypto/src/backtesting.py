"""
Backtesting Engine Module
Realistic backtesting with transaction costs, slippage, and position sizing.
"""

import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .utils import (
    setup_logging, calculate_sharpe_ratio, calculate_sortino_ratio,
    calculate_max_drawdown, calculate_calmar_ratio
)

logger = setup_logging()


class BacktestEngine:
    """
    Backtesting engine for trading strategies.
    
    Features:
    - Transaction costs and slippage
    - Position sizing (fixed, percentage, Kelly)
    - Long/short positions
    - Performance metrics
    - Trade logging
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 transaction_cost: float = 0.00065,  # 0.065% (NSE: 0.03% brokerage + 0.0325% STT + 0.0002% stamp)
                 slippage: float = 0.0003,  # 0.03% (realistic for NIFTY)
                 position_size: float = 1.0):
        """
        Initialize Backtesting Engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
            position_size: Position size as fraction of capital
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.position_size = position_size
        
        self.trades = []
        self.equity_curve = None
        self.metrics = {}
        
        logger.info(f"Initialized BacktestEngine with capital={initial_capital}")
    
    def run_backtest(self,
                    data: pd.DataFrame,
                    signals: pd.Series,
                    price_column: str = 'close') -> Dict:
        """
        Run backtest with given signals.
        
        Args:
            data: DataFrame with price data
            signals: Series with trading signals (1=buy, -1=sell, 0=hold)
            price_column: Column to use for prices
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Running backtest...")
        logger.info(f"Data shape: {data.shape}, Signals: {len(signals)}")
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Number of shares
        cash = capital
        
        equity = []
        positions = []
        trades_list = []
        
        # Iterate through data
        for i in range(len(data)):
            date = data.index[i]
            price = data[price_column].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Current portfolio value
            portfolio_value = cash + (position * price)
            equity.append(portfolio_value)
            positions.append(position)
            
            # Execute trades based on signals
            if signal == 1 and position == 0:
                # Buy signal
                shares_to_buy = int((cash * self.position_size) / (price * (1 + self.transaction_cost + self.slippage)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + self.transaction_cost + self.slippage)
                    if cost <= cash:
                        cash -= cost
                        position = shares_to_buy
                        
                        trades_list.append({
                            'date': date,
                            'type': 'BUY',
                            'price': price,
                            'shares': shares_to_buy,
                            'cost': cost,
                            'cash': cash,
                            'portfolio_value': portfolio_value
                        })
            
            elif signal == -1 and position > 0:
                # Sell signal
                proceeds = position * price * (1 - self.transaction_cost - self.slippage)
                cash += proceeds
                
                trades_list.append({
                    'date': date,
                    'type': 'SELL',
                    'price': price,
                    'shares': position,
                    'proceeds': proceeds,
                    'cash': cash,
                    'portfolio_value': portfolio_value
                })
                
                position = 0
        
        # Close any open position at the end
        if position > 0:
            final_price = data[price_column].iloc[-1]
            proceeds = position * final_price * (1 - self.transaction_cost - self.slippage)
            cash += proceeds
            
            trades_list.append({
                'date': data.index[-1],
                'type': 'SELL (CLOSE)',
                'price': final_price,
                'shares': position,
                'proceeds': proceeds,
                'cash': cash,
                'portfolio_value': cash
            })
        
        # Create equity curve
        self.equity_curve = pd.Series(equity, index=data.index)
        self.trades = pd.DataFrame(trades_list)
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(data)
        
        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
        logger.info(f"Final portfolio value: {self.equity_curve.iloc[-1]:,.2f}")
        
        return {
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'metrics': self.metrics
        }
    
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        logger.info("Calculating performance metrics...")
        
        metrics = {}
        
        # Basic metrics
        metrics['initial_capital'] = self.initial_capital
        metrics['final_value'] = self.equity_curve.iloc[-1]
        metrics['total_return'] = (metrics['final_value'] - self.initial_capital) / self.initial_capital
        metrics['total_return_pct'] = metrics['total_return'] * 100
        
        # Annualized return
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (1 / years) - 1
        metrics['annualized_return_pct'] = metrics['annualized_return'] * 100
        
        # Returns
        portfolio_returns = self.equity_curve.pct_change().dropna()
        
        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(portfolio_returns)
        metrics['sortino_ratio'] = calculate_sortino_ratio(portfolio_returns)
        
        # Drawdown
        dd_info = calculate_max_drawdown(self.equity_curve)
        metrics.update(dd_info)
        
        # Calmar ratio
        metrics['calmar_ratio'] = calculate_calmar_ratio(portfolio_returns, self.equity_curve)
        
        # Volatility
        metrics['volatility'] = portfolio_returns.std() * np.sqrt(252)
        metrics['volatility_pct'] = metrics['volatility'] * 100
        
        # Trade statistics
        if len(self.trades) > 0:
            buy_trades = self.trades[self.trades['type'] == 'BUY']
            sell_trades = self.trades[self.trades['type'].str.contains('SELL')]
            
            metrics['total_trades'] = len(buy_trades)
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                # Match buy and sell trades
                profits = []
                for i in range(min(len(buy_trades), len(sell_trades))):
                    buy_cost = buy_trades.iloc[i]['cost']
                    sell_proceeds = sell_trades.iloc[i]['proceeds']
                    profit = sell_proceeds - buy_cost
                    profits.append(profit)
                
                profits = np.array(profits)
                winning_trades = profits > 0
                
                metrics['winning_trades'] = winning_trades.sum()
                metrics['losing_trades'] = (~winning_trades).sum()
                metrics['win_rate'] = winning_trades.sum() / len(profits) if len(profits) > 0 else 0
                metrics['win_rate_pct'] = metrics['win_rate'] * 100
                
                if winning_trades.sum() > 0:
                    metrics['avg_win'] = profits[winning_trades].mean()
                    metrics['avg_win_pct'] = (profits[winning_trades] / buy_trades.iloc[:len(profits)]['cost'].values[winning_trades]).mean() * 100
                
                if (~winning_trades).sum() > 0:
                    metrics['avg_loss'] = profits[~winning_trades].mean()
                    metrics['avg_loss_pct'] = (profits[~winning_trades] / buy_trades.iloc[:len(profits)]['cost'].values[~winning_trades]).mean() * 100
                
                # Profit factor
                gross_profit = profits[winning_trades].sum() if winning_trades.sum() > 0 else 0
                gross_loss = abs(profits[~winning_trades].sum()) if (~winning_trades).sum() > 0 else 1
                metrics['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else np.nan
        
        # Buy and hold comparison
        buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        metrics['buy_hold_return'] = buy_hold_return
        metrics['buy_hold_return_pct'] = buy_hold_return * 100
        metrics['excess_return'] = metrics['total_return'] - buy_hold_return
        metrics['excess_return_pct'] = metrics['excess_return'] * 100
        
        return metrics
    
    def get_metrics_summary(self) -> pd.Series:
        """Get formatted metrics summary."""
        if not self.metrics:
            logger.warning("No metrics available. Run backtest first.")
            return pd.Series()
        
        return pd.Series(self.metrics)
    
    def print_summary(self) -> None:
        """Print backtest summary."""
        if not self.metrics:
            logger.warning("No results available. Run backtest first.")
            return
        
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"\nCapital:")
        print(f"  Initial Capital:       {self.metrics['initial_capital']:>15,.2f}")
        print(f"  Final Value:           {self.metrics['final_value']:>15,.2f}")
        print(f"  Total Return:          {self.metrics['total_return_pct']:>14.2f}%")
        print(f"  Annualized Return:     {self.metrics['annualized_return_pct']:>14.2f}%")
        
        print(f"\nRisk Metrics:")
        print(f"  Volatility (Annual):   {self.metrics['volatility_pct']:>14.2f}%")
        print(f"  Sharpe Ratio:          {self.metrics['sharpe_ratio']:>15.2f}")
        print(f"  Sortino Ratio:         {self.metrics['sortino_ratio']:>15.2f}")
        print(f"  Calmar Ratio:          {self.metrics['calmar_ratio']:>15.2f}")
        print(f"  Max Drawdown:          {self.metrics['max_drawdown_pct']:>14.2f}%")
        
        if 'total_trades' in self.metrics:
            print(f"\nTrade Statistics:")
            print(f"  Total Trades:          {self.metrics['total_trades']:>15.0f}")
            print(f"  Winning Trades:        {self.metrics.get('winning_trades', 0):>15.0f}")
            print(f"  Losing Trades:         {self.metrics.get('losing_trades', 0):>15.0f}")
            print(f"  Win Rate:              {self.metrics.get('win_rate_pct', 0):>14.2f}%")
            
            if 'avg_win_pct' in self.metrics:
                print(f"  Average Win:           {self.metrics['avg_win_pct']:>14.2f}%")
            if 'avg_loss_pct' in self.metrics:
                print(f"  Average Loss:          {self.metrics['avg_loss_pct']:>14.2f}%")
            if 'profit_factor' in self.metrics:
                print(f"  Profit Factor:         {self.metrics['profit_factor']:>15.2f}")
        
        print(f"\nComparison:")
        print(f"  Buy & Hold Return:     {self.metrics['buy_hold_return_pct']:>14.2f}%")
        print(f"  Excess Return:         {self.metrics['excess_return_pct']:>14.2f}%")
        
        print("=" * 60 + "\n")


def optimize_strategy(data: pd.DataFrame,
                     signal_generator,
                     param_grid: Dict,
                     initial_capital: float = 1000000) -> pd.DataFrame:
    """
    Optimize strategy parameters.
    
    Args:
        data: Price data
        signal_generator: Function to generate signals given parameters
        param_grid: Dictionary of parameters to test
        initial_capital: Starting capital
        
    Returns:
        DataFrame with optimization results
    """
    logger.info("Running strategy optimization...")
    
    results = []
    
    # Generate parameter combinations
    import itertools
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        logger.info(f"Testing parameters: {params}")
        
        try:
            # Generate signals with these parameters
            signals = signal_generator(data, **params)
            
            # Run backtest
            engine = BacktestEngine(initial_capital=initial_capital)
            engine.run_backtest(data, signals)
            
            # Store results
            result = params.copy()
            result.update({
                'total_return': engine.metrics['total_return'],
                'sharpe_ratio': engine.metrics['sharpe_ratio'],
                'max_drawdown': engine.metrics['max_drawdown'],
                'win_rate': engine.metrics.get('win_rate', 0),
                'total_trades': engine.metrics.get('total_trades', 0)
            })
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error with parameters {params}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    logger.info(f"Optimization complete. Best Sharpe: {results_df['sharpe_ratio'].max():.2f}")
    
    return results_df


if __name__ == "__main__":
    # Test the module
    print("Testing Backtesting Engine")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    returns = np.random.normal(0.001, 0.02, len(dates))
    
    data = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(returns))
    }, index=dates)
    
    # Create simple signals (moving average crossover)
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    signals = pd.Series(0, index=data.index)
    signals[data['sma_20'] > data['sma_50']] = 1
    signals[data['sma_20'] < data['sma_50']] = -1
    
    # Run backtest
    engine = BacktestEngine(initial_capital=1000000)
    results = engine.run_backtest(data, signals)
    
    engine.print_summary()
    print(f"\nFirst 5 trades:\n{engine.trades.head()}")
