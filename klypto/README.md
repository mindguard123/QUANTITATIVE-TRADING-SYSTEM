# Quantitative Trading System for Indian Markets (NIFTY)

A complete, production-grade quantitative trading system implementing advanced machine learning techniques for Indian equity markets.

## 🎯 Project Overview

This system implements a comprehensive quantitative trading strategy that combines:
- **EMA-based Technical Indicators**: Exponential Moving Averages and derived features
- **Hidden Markov Models (HMM)**: Market regime detection (bull/bear/sideways)
- **Machine Learning Models**: XGBoost, LightGBM, Neural Networks for price prediction
- **Backtesting Engine**: Realistic simulation with transaction costs and slippage
- **Outlier Analysis**: Statistical methods to detect and handle anomalies

## 📁 Project Structure

```
klypto/
├── data/
│   ├── raw/              # Raw market data from NSE/Yahoo Finance
│   ├── interim/          # Intermediate processed data
│   └── processed/        # Final feature-engineered data
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py      # Data downloading and cleaning
│   ├── feature_engineering.py   # EMA and technical indicators
│   ├── hmm_regime.py            # Hidden Markov Model implementation
│   ├── ml_models.py             # ML model training and prediction
│   ├── backtesting.py           # Backtesting engine
│   ├── outlier_detection.py     # Outlier analysis and handling
│   ├── visualization.py         # Plotting and reporting
│   └── utils.py                 # Helper functions
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_eda_feature_engineering.ipynb
│   ├── 03_hmm_regime_detection.ipynb
│   ├── 04_ml_model_training.ipynb
│   ├── 05_backtesting_evaluation.ipynb
│   └── 06_main_execution.ipynb
├── models/               # Saved trained models
├── plots/                # Generated visualizations
├── results/              # Performance metrics and reports
├── configs/              # Configuration files
├── tests/                # Unit tests
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone or navigate to the repository:**
```bash
cd d:\klypto
```

2. **Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install TA-Lib (Windows):**
- Download TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
- Install: `pip install TA_Lib‑0.4.28‑cp310‑cp310‑win_amd64.whl`

## 📊 Data Sources

- **Primary**: NSE India via `nsepy` library
- **Fallback**: Yahoo Finance via `yfinance`
- **Coverage**: NIFTY 50 Index and constituent stocks
- **Frequency**: Daily OHLCV data
- **History**: Minimum 5 years for robust backtesting

## 🔬 Methodology

### 1. Feature Engineering
- **EMA Features**: 
  - EMA 5, 10, 20, 50, 100, 200
  - EMA crossovers and divergence
  - Rate of change in EMAs
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - ATR (Average True Range)
  - Volume-based indicators
- **Price Features**:
  - Returns (log and simple)
  - Volatility (rolling std)
  - High-Low spread
  - Gaps

### 2. Hidden Markov Model (HMM)
- **States**: 3-4 hidden states representing market regimes
  - Bull Market (high returns, low volatility)
  - Bear Market (negative returns, high volatility)
  - Sideways/Consolidation (low returns, medium volatility)
- **Observations**: Returns, volatility, volume
- **Training**: Baum-Welch algorithm (EM)
- **Inference**: Viterbi algorithm for most likely state sequence

### 3. Machine Learning Models
- **XGBoost**: Gradient boosting with tree-based learners
- **LightGBM**: Fast gradient boosting framework
- **Neural Networks**: Deep learning with LSTM/GRU layers
- **Ensemble**: Weighted combination of models
- **Target**: Next-day returns or direction (classification)

### 4. Backtesting
- **Walk-forward optimization**: Rolling window approach
- **Transaction costs**: 0.05% per trade (realistic for Indian markets)
- **Slippage**: 0.02% market impact
- **Position sizing**: Fixed fractional or Kelly criterion
- **Risk management**: Stop-loss and take-profit levels
- **Performance metrics**:
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Win Rate
  - Profit Factor
  - Calmar Ratio

### 5. Outlier Detection
- **Statistical methods**:
  - Z-score analysis
  - IQR (Interquartile Range)
  - Modified Z-score
- **Time-series specific**:
  - Seasonal decomposition
  - ARIMA residuals
- **Handling**:
  - Winsorization
  - Capping at percentiles
  - Removal with forward-fill

## 📈 Usage

### Quick Start

1. **Run the complete pipeline:**
   Open and execute notebooks in order:
   ```
   01_data_acquisition.ipynb → ... → 06_main_execution.ipynb
   ```

2. **Or use the main execution notebook:**
   ```
   06_main_execution.ipynb
   ```

### Module Usage

```python
# Data Acquisition
from src.data_acquisition import DataAcquisition
da = DataAcquisition(symbol='NIFTY', start_date='2018-01-01', end_date='2024-01-01')
df = da.fetch_data()

# Feature Engineering
from src.feature_engineering import FeatureEngineer
fe = FeatureEngineer(df)
df_features = fe.create_all_features()

# HMM Regime Detection
from src.hmm_regime import HMMRegimeDetector
hmm = HMMRegimeDetector(n_states=3)
df_with_regime = hmm.fit_predict(df_features)

# ML Model Training
from src.ml_models import MLModelTrainer
ml = MLModelTrainer(model_type='xgboost')
ml.train(X_train, y_train)
predictions = ml.predict(X_test)

# Backtesting
from src.backtesting import BacktestEngine
bt = BacktestEngine(initial_capital=1000000)
results = bt.run_backtest(df_with_signals)
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v --cov=src
```

## 📊 Results

Results are saved in the `results/` directory:
- `backtest_metrics.csv`: Performance metrics
- `trade_log.csv`: Individual trade details
- `equity_curve.png`: Portfolio value over time
- `regime_performance.csv`: Performance by market regime

## ⚠️ Risk Disclaimer

This software is for educational and research purposes only. 
- **Not financial advice**: Do not use for actual trading without proper due diligence
- **Past performance**: Does not guarantee future results
- **Market risk**: Trading involves substantial risk of loss
- **Test thoroughly**: Always paper trade before live deployment

## 🔧 Configuration

Edit `configs/config.yaml` to customize:
- Data parameters (symbols, date ranges)
- Feature engineering settings
- Model hyperparameters
- Backtesting parameters
- Risk management rules

## 📚 References

- **HMM**: Lawrence R. Rabiner, "A Tutorial on Hidden Markov Models"
- **EMA**: Technical Analysis of Financial Markets by John Murphy
- **XGBoost**: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System"
- **Backtesting**: Prado, "Advances in Financial Machine Learning"

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

MIT License - See LICENSE file for details

## 👨‍💻 Author

Senior Quantitative Researcher & ML Engineer
Specialized in systematic trading strategies for Indian markets

## 📧 Contact

For questions or collaboration opportunities, please open an issue.

---

**Last Updated**: January 2026
**Version**: 1.0.0
**Status**: Production-Ready
