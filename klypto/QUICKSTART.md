# Quick Start Guide - Quantitative Trading System

## 🚀 Installation and Setup

### Step 1: Environment Setup

```powershell
# Navigate to project directory
cd d:\klypto

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 2: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt
```

**Note for TA-Lib (Windows):**
If TA-Lib installation fails, download the wheel file:
1. Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Download `TA_Lib‑0.4.28‑cp310‑cp310‑win_amd64.whl` (or appropriate version)
3. Install: `pip install TA_Lib‑0.4.28‑cp310‑cp310‑win_amd64.whl`

### Step 3: Verify Installation

```powershell
# Test imports
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, tensorflow, hmmlearn; print('All packages installed successfully!')"
```

## 📊 Running the System

### Option 1: Complete Pipeline (Recommended)

Run the main execution notebook that performs all steps:

```powershell
jupyter notebook notebooks/06_main_execution.ipynb
```

Then execute all cells in sequence (Cell → Run All).

### Option 2: Step-by-Step Execution

Run notebooks in order:

1. **Data Acquisition**
   ```powershell
   jupyter notebook notebooks/01_data_acquisition.ipynb
   ```

2. **Feature Engineering** (create notebook 02)
3. **HMM Regime Detection** (create notebook 03)
4. **ML Model Training** (create notebook 04)
5. **Backtesting** (create notebook 05)
6. **Main Execution** (notebook 06)

### Option 3: Python Scripts

Use the modules directly in Python:

```python
from src.data_acquisition import DataAcquisition
from src.feature_engineering import FeatureEngineer
from src.hmm_regime import HMMRegimeDetector
from src.ml_models import MLModelTrainer
from src.backtesting import BacktestEngine

# Example: Quick backtest
da = DataAcquisition('NIFTY', '2020-01-01', '2024-01-01')
df = da.fetch_data()

fe = FeatureEngineer(df)
df_features = fe.create_all_features()

# ... continue with modeling and backtesting
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Key parameters to adjust:
data:
  symbol: "NIFTY"          # Change symbol
  start_date: "2018-01-01" # Change date range

ml_models:
  target:
    method: "direction"    # or "returns"
    horizon: 1             # prediction days ahead

backtesting:
  initial_capital: 1000000 # Starting capital in INR
  transaction_cost: 0.0005 # 0.05% per trade

strategy:
  use_regime_filter: true  # Trade only in specific regimes
  trade_regimes: ["Bull", "Sideways"]
```

## 📁 Project Structure

```
klypto/
├── data/
│   ├── raw/              # Downloaded market data
│   ├── interim/          # Intermediate processing
│   └── processed/        # Final feature-engineered data
├── src/
│   ├── data_acquisition.py      # Data fetching
│   ├── feature_engineering.py   # EMA + indicators
│   ├── hmm_regime.py            # Regime detection
│   ├── ml_models.py             # ML models
│   ├── backtesting.py           # Backtest engine
│   ├── outlier_detection.py     # Anomaly detection
│   ├── visualization.py         # Plotting
│   └── utils.py                 # Helper functions
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   └── 06_main_execution.ipynb
├── models/               # Saved trained models
├── plots/                # Generated charts
├── results/              # Performance reports
├── configs/
│   └── config.yaml       # Configuration file
├── requirements.txt
└── README.md
```

## 🔍 What Each Module Does

### 1. Data Acquisition (`data_acquisition.py`)
- Fetches historical data from NSE (nsepy) or Yahoo Finance
- Handles missing data and outliers
- Validates OHLCV relationships
- Saves raw data in multiple formats

### 2. Feature Engineering (`feature_engineering.py`)
- **EMA Features**: 5, 10, 20, 50, 100, 200 periods
- **EMA Crossovers**: Multiple fast/slow combinations
- **Technical Indicators**: RSI, MACD, Stochastic, Bollinger Bands, ATR
- **Volume Indicators**: OBV, PVT, volume ratios
- **Price Features**: Returns, gaps, ranges
- **Lagged Features**: Historical lookback

### 3. HMM Regime Detection (`hmm_regime.py`)
- Identifies 3-4 hidden market states
- Bull/Bear/Sideways classification
- Transition probability analysis
- Current regime identification
- Regime-specific performance metrics

### 4. Machine Learning Models (`ml_models.py`)
- **XGBoost**: Gradient boosting classifier
- **LightGBM**: Fast gradient boosting
- **Neural Networks**: Deep learning with TensorFlow
- **Ensemble**: Weighted combination of models
- Feature importance ranking
- Probability-based signals

### 5. Backtesting Engine (`backtesting.py`)
- Realistic transaction costs (0.05% for Indian markets)
- Slippage modeling (0.02%)
- Position sizing strategies
- Performance metrics:
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Calmar Ratio
  - Win Rate, Profit Factor
- Trade-by-trade logging

### 6. Outlier Detection (`outlier_detection.py`)
- Z-score method
- Modified Z-score (robust)
- IQR (Interquartile Range)
- Isolation Forest
- Multiple handling strategies:
  - Winsorization
  - Capping
  - Removal
  - Forward fill

### 7. Visualization (`visualization.py`)
- Equity curves with drawdowns
- Price charts with signals
- Returns distribution
- Regime analysis plots
- Feature importance charts
- Correlation matrices

## 📈 Expected Outputs

After running the complete pipeline:

### Data Files
- `data/raw/NIFTY_data.csv` - Raw market data
- `data/processed/nifty_features.csv` - All features
- `data/processed/nifty_with_regime.csv` - With HMM regimes

### Models
- `models/xgboost_model.pkl` - Trained XGBoost
- `models/lightgbm_model.pkl` - Trained LightGBM
- `models/nn_model.pkl` - Trained Neural Network
- `models/hmm_regime_model.pkl` - HMM regime detector

### Results
- `results/backtest_metrics.csv` - Performance metrics
- `results/trade_log.csv` - All executed trades
- `results/equity_curve.csv` - Portfolio value over time
- `results/trading_signals.csv` - Generated signals

### Plots
- `plots/final_backtest_equity_curve.png`
- `plots/final_backtest_signals.png`
- `plots/regime_analysis.png`
- `plots/feature_importance.png`
- `plots/ema_analysis.png`

## 🎯 Key Performance Metrics

The system calculates comprehensive performance metrics:

| Metric | Description |
|--------|-------------|
| Total Return | Overall profit/loss percentage |
| Annualized Return | Return normalized to yearly basis |
| Sharpe Ratio | Risk-adjusted return (>1 is good, >2 is excellent) |
| Sortino Ratio | Downside risk-adjusted return |
| Maximum Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | Return / Max Drawdown |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / Gross loss |

## ⚠️ Important Notes

1. **Data Quality**: Ensure stable internet for data download
2. **Computational Resources**: ML training requires significant CPU/GPU
3. **Time**: Full pipeline may take 15-30 minutes depending on hardware
4. **Memory**: Ensure at least 8GB RAM available
5. **Date Range**: Minimum 2 years of data recommended for robust results

## 🐛 Troubleshooting

### Issue: Module not found
```powershell
# Ensure you're in the virtual environment
.\venv\Scripts\activate

# Verify Python is using venv
python -c "import sys; print(sys.prefix)"
```

### Issue: TA-Lib installation fails
```powershell
# Use the pre-compiled wheel (see Step 2)
# Or skip TA-Lib features (system will use pandas implementation)
```

### Issue: Jupyter kernel not found
```powershell
python -m ipykernel install --user --name=venv --display-name="Python (Klypto)"
```

### Issue: Out of memory
```powershell
# Reduce date range in config.yaml
# Or process in smaller chunks
```

## 📚 Next Steps After Setup

1. **Run the main notebook** (`06_main_execution.ipynb`)
2. **Review the results** in `results/` directory
3. **Analyze plots** in `plots/` directory
4. **Fine-tune parameters** in `configs/config.yaml`
5. **Experiment with features** in `feature_engineering.py`
6. **Optimize hyperparameters** for ML models
7. **Implement risk management** rules
8. **Paper trade** before live deployment

## 📞 Support

For issues or questions:
1. Check the main README.md
2. Review module docstrings
3. Examine test code in each module's `__main__` block

## ⚡ Performance Tips

1. **Use Parquet format** for large datasets (faster than CSV)
2. **Enable caching** in config for repeated runs
3. **Use parallel processing** (`n_jobs=-1` in config)
4. **GPU acceleration** for Neural Networks (configure TensorFlow)
5. **Reduce features** if training is too slow

## 🎓 Learning Resources

- **HMM**: Research papers on Hidden Markov Models in finance
- **XGBoost**: Official documentation and tutorials
- **Backtesting**: "Advances in Financial Machine Learning" by Marcos Lopez de Prado
- **Technical Analysis**: "Technical Analysis of Financial Markets" by John Murphy

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Status**: Production Ready  

🎉 **Happy Trading!** (Remember: Past performance ≠ Future results)
