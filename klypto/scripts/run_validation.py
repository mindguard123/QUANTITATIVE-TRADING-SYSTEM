"""
Run Complete System Validation
Executes the full pipeline and generates results
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

# Add src to path - import directly, not as package
sys.path.insert(0, 'D:/klypto/src')

# Import custom modules (direct imports, no package structure)
import data_acquisition as da_module
import feature_engineering as fe_module
import hmm_regime as hmm_module
import ml_models as ml_module
import backtesting as bt_module
import outlier_detection as od_module
import visualization as viz_module
import utils

DataAcquisition = da_module.DataAcquisition
FeatureEngineer = fe_module.FeatureEngineer
HMMRegimeDetector = hmm_module.HMMRegimeDetector
MLModelTrainer = ml_module.MLModelTrainer
BacktestEngine = bt_module.BacktestEngine
OutlierDetector = od_module.OutlierDetector
Visualizer = viz_module.Visualizer
setup_logging = utils.setup_logging
load_config = utils.load_config
save_dataframe = utils.save_dataframe
load_dataframe = utils.load_dataframe
split_train_test_by_date = utils.split_train_test_by_date
calculate_statistical_significance = utils.calculate_statistical_significance
bootstrap_sharpe_ci = utils.bootstrap_sharpe_ci
compare_strategies = utils.compare_strategies

# Setup
logger = setup_logging()
config = load_config('D:/klypto/configs/config.yaml')

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("=" * 80)
print(" " * 20 + "QUANTITATIVE TRADING SYSTEM VALIDATION")
print(" " * 20 + "Fixed Data Leakage - Production Ready")
print("=" * 80)
print(f"\n⏰ Execution started: {datetime.now()}")
print("\n✅ All modules loaded successfully\n")

# ============================================================================
# STEP 1: DATA ACQUISITION
# ============================================================================
print("\n" + "=" * 60)
print("STEP 1: DATA ACQUISITION")
print("=" * 60)

data_fetcher = DataAcquisition(
    source='yahoo',
    cache_dir='D:/klypto/data/raw'
)

start_date = config['data']['start_date']
end_date = config['data']['end_date']
symbol = config['data']['symbol']

print(f"Fetching {symbol} from {start_date} to {end_date}...")
df = data_fetcher.fetch_data(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date
)

print(f"\n✓ Data fetched: {len(df)} rows")
print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

save_dataframe(df, 'D:/klypto/data/raw/nifty_raw.csv')
print("✓ Raw data saved")

# ============================================================================
# STEP 2: FEATURE ENGINEERING (NO LOOKAHEAD)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

engineer = FeatureEngineer(df)
print("Creating technical features...")

df_features = engineer.create_all_features(
    ema_periods=config['features']['ema_periods'],
    add_momentum=True,
    add_volatility=True,
    add_volume=True,
    add_price_features=True
)

print(f"\n✓ Features created: {df_features.shape[1]} columns, {len(df_features)} rows")

save_dataframe(df_features, 'D:/klypto/data/interim/nifty_features.csv')
print("✓ Feature data saved")

# ============================================================================
# STEP 3: OUTLIER DETECTION
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: OUTLIER DETECTION")
print("=" * 60)

outlier_detector = OutlierDetector(df_features)
outliers = outlier_detector.detect_all()

print("\nOutlier Detection Summary:")
print(outlier_detector.get_summary())

df_clean = outlier_detector.handle_outliers(
    method='clip',
    columns=['returns', 'volume'],
    percentile=(1, 99)
)

print(f"\n✓ Data shape after outlier handling: {df_clean.shape}")
save_dataframe(df_clean, 'D:/klypto/data/processed/nifty_clean.csv')
print("✓ Cleaned data saved")

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT (TEMPORAL - NO LOOKAHEAD)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: TRAIN/TEST SPLIT")
print("=" * 60)

train_df, val_df, test_df = split_train_test_by_date(
    df_clean,
    train_ratio=0.7,
    validation_ratio=0.15
)

print(f"\nData Split:")
print(f"  Training:   {len(train_df)} samples ({train_df.index[0].date()} to {train_df.index[-1].date()})")
print(f"  Validation: {len(val_df)} samples ({val_df.index[0].date()} to {val_df.index[-1].date()})")
print(f"  Test:       {len(test_df)} samples ({test_df.index[0].date()} to {test_df.index[-1].date()})")

print("\n⚠️  CRITICAL: HMM will be fitted ONLY on training data")

# ============================================================================
# STEP 5: HMM REGIME DETECTION (FITTED ON TRAIN ONLY)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: HMM REGIME DETECTION")
print("=" * 60)

hmm_detector = HMMRegimeDetector(
    n_states=config['hmm']['n_states'],
    n_iter=config['hmm']['n_iter'],
    random_state=config['execution']['random_state']
)

print("Fitting HMM on training data only...")
hmm_detector.fit(train_df)

train_regimes = hmm_detector.predict(train_df)
val_regimes = hmm_detector.predict(val_df)
test_regimes = hmm_detector.predict(test_df)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df['regime'] = train_regimes
val_df['regime'] = val_regimes
test_df['regime'] = test_regimes

print(f"\n✓ Regime detection complete")
print(f"  Train regime distribution:\n{pd.Series(train_regimes).value_counts()}")

hmm_detector.save('D:/klypto/models/hmm_model.pkl')
print("✓ HMM model saved")

# ============================================================================
# STEP 6: ML MODEL TRAINING (FIXED TARGET CREATION)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: ML MODEL TRAINING")
print("=" * 60)

ml_trainer = MLModelTrainer(
    model_type='xgboost',
    task='classification',
    random_state=config['execution']['random_state']
)

print("Creating targets (NO lookahead bias)...")
train_ml = ml_trainer.create_target(train_df, method='direction', horizon=1)
val_ml = ml_trainer.create_target(val_df, method='direction', horizon=1)
test_ml = ml_trainer.create_target(test_df, method='direction', horizon=1)

print(f"\nTarget creation complete:")
print(f"  Train: {len(train_ml)} samples")
print(f"  Val:   {len(val_ml)} samples")
print(f"  Test:  {len(test_ml)} samples")

X_train, y_train = ml_trainer.prepare_data(train_ml, 'target')
X_val, y_val = ml_trainer.prepare_data(val_ml, 'target')
X_test, y_test = ml_trainer.prepare_data(test_ml, 'target')

print(f"\nFeature matrix: {X_train.shape}")
print(f"Number of features: {len(ml_trainer.feature_columns)}")
print(f"Target distribution (train): {pd.Series(y_train).value_counts().to_dict()}")

print("\nTraining XGBoost...")
xgb_metrics = ml_trainer.train(
    X_train, y_train, X_val, y_val,
    hyperparameters=config['ml_models']['xgboost']
)

print(f"\n✓ XGBoost Training Metrics:")
for k, v in xgb_metrics.items():
    print(f"  {k}: {v:.4f}")

ml_trainer.save('D:/klypto/models/xgboost_model.pkl')
print("\n✓ XGBoost model saved")

# ============================================================================
# STEP 7: GENERATE TRADING SIGNALS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 7: GENERATE TRADING SIGNALS")
print("=" * 60)

test_predictions = ml_trainer.predict(X_test)
test_proba = ml_trainer.predict_proba(X_test)

signals_df = test_ml.copy()
signals_df['prediction'] = test_predictions
signals_df['prediction_proba'] = test_proba[:, 1] if test_proba.ndim > 1 else test_proba

threshold = config['ml_models']['prediction_threshold']
signals_df['signal'] = 0
signals_df.loc[signals_df['prediction_proba'] > threshold, 'signal'] = 1
signals_df.loc[signals_df['prediction_proba'] < (1 - threshold), 'signal'] = -1

print(f"\nSignal Distribution:")
print(signals_df['signal'].value_counts())

save_dataframe(signals_df, 'D:/klypto/results/trading_signals.csv')
print("\n✓ Trading signals saved")

# ============================================================================
# STEP 8: BACKTESTING (REALISTIC COSTS)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 8: BACKTESTING")
print("=" * 60)

backtest = BacktestEngine(
    initial_capital=1000000,
    transaction_cost=0.00065,  # Realistic NSE
    slippage=0.0003,
    position_size=0.95
)

print(f"\nBacktest Configuration:")
print(f"  Initial Capital: ₹{backtest.initial_capital:,.0f}")
print(f"  Transaction Cost: {backtest.transaction_cost*100:.3f}%")
print(f"  Slippage: {backtest.slippage*100:.3f}%")

print("\nRunning backtest on test set...")
results = backtest.run_backtest(
    data=signals_df,
    signals=signals_df['signal'],
    price_column='close'
)

print("\n✓ Backtest complete!")
print("\n" + "=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)

metrics = results['metrics']
for key, value in metrics.items():
    if isinstance(value, (int, float)):
        if 'pct' in key or 'rate' in key:
            print(f"{key:30s}: {value:>10.2f}")
        else:
            print(f"{key:30s}: {value:>10.4f}")

save_dataframe(results['equity_curve'], 'D:/klypto/results/equity_curve.csv')
save_dataframe(results['trades'], 'D:/klypto/results/trades.csv')
print("\n✓ Results saved")

# ============================================================================
# STEP 9: BASELINE COMPARISON (BUY & HOLD)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 9: BASELINE COMPARISON")
print("=" * 60)

buyhold_backtest = BacktestEngine(
    initial_capital=1000000,
    transaction_cost=0.00065,
    slippage=0.0003,
    position_size=0.95
)

buyhold_signals = pd.Series(0, index=signals_df.index)
buyhold_signals.iloc[0] = 1

print("Running buy-and-hold baseline...")
buyhold_results = buyhold_backtest.run_backtest(
    data=signals_df,
    signals=buyhold_signals,
    price_column='close'
)

print("\n✓ Buy-and-hold backtest complete")

strategy_comparison = compare_strategies({
    'ML Strategy (XGBoost)': results,
    'Buy & Hold': buyhold_results
})

print("\n" + "=" * 60)
print("STRATEGY COMPARISON")
print("=" * 60)
print(strategy_comparison.to_string(index=False))

excess_return = metrics['total_return_pct'] - buyhold_results['metrics']['total_return_pct']
print(f"\n📊 Excess Return over Buy & Hold: {excess_return:.2f}%")

if excess_return > 0:
    print("✅ Strategy OUTPERFORMS buy-and-hold")
else:
    print("⚠️  Strategy UNDERPERFORMS buy-and-hold")

strategy_comparison.to_csv('D:/klypto/results/strategy_comparison.csv', index=False)
print("\n✓ Comparison saved")

# ============================================================================
# STEP 10: STATISTICAL SIGNIFICANCE TESTS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 10: STATISTICAL SIGNIFICANCE TESTS")
print("=" * 60)

strategy_returns = results['equity_curve'].pct_change().dropna()
buyhold_returns = buyhold_results['equity_curve'].pct_change().dropna()

print("\nRunning statistical tests...")
sig_tests = calculate_statistical_significance(
    strategy_returns,
    buyhold_returns
)

print("\n" + "=" * 60)
print("STATISTICAL TESTS")
print("=" * 60)

print(f"\n1. T-Test (Returns vs Zero):")
print(f"   T-Statistic: {sig_tests['t_statistic']:.4f}")
print(f"   P-Value: {sig_tests['p_value_vs_zero']:.4f}")
print(f"   Significant (p<0.05): {sig_tests['significant_vs_zero']}")

if sig_tests['significant_vs_zero']:
    print("   ✅ Strategy returns are statistically significant")
else:
    print("   ⚠️  Strategy returns are NOT statistically significant")

print(f"\n2. Normality Test (Jarque-Bera):")
print(f"   JB Statistic: {sig_tests['jarque_bera_stat']:.4f}")
print(f"   P-Value: {sig_tests['jarque_bera_p']:.4f}")

if 'alpha_t_statistic' in sig_tests:
    print(f"\n3. Alpha Test (vs Buy & Hold):")
    print(f"   T-Statistic: {sig_tests['alpha_t_statistic']:.4f}")
    print(f"   P-Value: {sig_tests['alpha_p_value']:.4f}")
    print(f"   Significant Alpha (p<0.05): {sig_tests['significant_alpha']}")
    
    if sig_tests['significant_alpha']:
        if sig_tests['alpha_t_statistic'] > 0:
            print("   ✅ Strategy has statistically significant POSITIVE alpha")
        else:
            print("   ❌ Strategy has statistically significant NEGATIVE alpha")
    else:
        print("   ⚠️  Alpha is not statistically significant (could be luck)")

with open('D:/klypto/results/statistical_tests.json', 'w') as f:
    tests_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                         for k, v in sig_tests.items()}
    json.dump(tests_serializable, f, indent=2)

print("\n✓ Statistical tests saved")

# ============================================================================
# STEP 11: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 11: BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 60)

print("\nCalculating bootstrap confidence intervals (1000 samples)...")
bootstrap_results = bootstrap_sharpe_ci(
    strategy_returns,
    n_bootstrap=1000,
    confidence=0.95
)

print("\n" + "=" * 60)
print("SHARPE RATIO CONFIDENCE INTERVAL")
print("=" * 60)

print(f"\nSharpe Ratio: {bootstrap_results['sharpe_ratio']:.4f}")
print(f"95% Confidence Interval: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
print(f"Bootstrap Mean: {bootstrap_results['bootstrap_mean']:.4f}")
print(f"Bootstrap Std: {bootstrap_results['bootstrap_std']:.4f}")

if bootstrap_results['ci_lower'] > 0:
    print("\n✅ Sharpe ratio is SIGNIFICANTLY POSITIVE (95% CI does not include 0)")
else:
    print("\n⚠️  Sharpe ratio confidence interval includes 0 (not significantly positive)")

with open('D:/klypto/results/bootstrap_results.json', 'w') as f:
    bootstrap_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                             for k, v in bootstrap_results.items()}
    json.dump(bootstrap_serializable, f, indent=2)

print("\n✓ Bootstrap results saved")

# ============================================================================
# STEP 12: VISUALIZATION
# ============================================================================
print("\n" + "=" * 60)
print("STEP 12: VISUALIZATION")
print("=" * 60)

visualizer = Visualizer()

# 1. Equity curves comparison
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(results['equity_curve'].index, results['equity_curve'].values, 
        label='ML Strategy', linewidth=2)
ax.plot(buyhold_results['equity_curve'].index, buyhold_results['equity_curve'].values,
        label='Buy & Hold', linewidth=2, alpha=0.7)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Portfolio Value (₹)', fontsize=12)
ax.set_title('Strategy Comparison: Equity Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:/klypto/plots/equity_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Equity curves plot saved")

# 2. Drawdown analysis
visualizer.plot_drawdown(results, save_path='D:/klypto/plots/drawdown_analysis.png')
print("✓ Drawdown plot saved")

# 3. Returns distribution
visualizer.plot_returns_distribution(results, save_path='D:/klypto/plots/returns_distribution.png')
print("✓ Returns distribution plot saved")

# 4. Feature importance
if ml_trainer.feature_importance is not None:
    top_features = ml_trainer.feature_importance.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 20 Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('D:/klypto/plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature importance plot saved")

print("\n✅ All visualizations generated")

# ============================================================================
# STEP 13: FINAL SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print(" " * 20 + "FINAL VALIDATION REPORT")
print("=" * 80)

print("\n🔧 FIXES APPLIED:")
print("  ✅ Removed lookahead bias in target creation")
print("  ✅ HMM fitted only on training data")
print("  ✅ Temporal train/test split (no data leakage)")
print("  ✅ Realistic NSE transaction costs (0.065%)")
print("  ✅ Fixed deprecated pandas methods")

print("\n📊 PERFORMANCE SUMMARY:")
print(f"  Strategy: ML XGBoost")
print(f"  Test Period: {signals_df.index[0].date()} to {signals_df.index[-1].date()}")
print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
print(f"  Annual Return: {metrics['annualized_return_pct']:.2f}%")
print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
print(f"  Sortino Ratio: {metrics['sortino_ratio']:.4f}")
print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"  Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
print(f"  Profit Factor: {metrics.get('profit_factor', 0):.4f}")

print("\n📈 VS BENCHMARK:")
print(f"  Buy & Hold Return: {buyhold_results['metrics']['total_return_pct']:.2f}%")
print(f"  Excess Return: {excess_return:.2f}%")
print(f"  Outperformance: {'YES ✅' if excess_return > 0 else 'NO ❌'}")

print("\n🔬 STATISTICAL VALIDATION:")
print(f"  Returns Significant: {'YES ✅' if sig_tests['significant_vs_zero'] else 'NO ❌'}")
print(f"  P-Value: {sig_tests['p_value_vs_zero']:.4f}")
if 'significant_alpha' in sig_tests:
    print(f"  Alpha Significant: {'YES ✅' if sig_tests['significant_alpha'] else 'NO ❌'}")
print(f"  Sharpe 95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
print(f"  CI Excludes Zero: {'YES ✅' if bootstrap_results['ci_lower'] > 0 else 'NO ❌'}")

print("\n💼 INTERVIEW READINESS:")
interview_score = 0
if metrics['sharpe_ratio'] > 0.5:
    interview_score += 2
    print("  ✅ Sharpe ratio > 0.5 (good)")
elif metrics['sharpe_ratio'] > 0:
    interview_score += 1
    print("  ⚠️  Sharpe ratio positive but < 0.5")
else:
    print("  ❌ Sharpe ratio negative")

if sig_tests['significant_vs_zero']:
    interview_score += 2
    print("  ✅ Statistically significant returns")
else:
    print("  ❌ Returns not statistically significant")

if excess_return > 0:
    interview_score += 1
    print("  ✅ Outperforms buy-and-hold")
else:
    print("  ⚠️  Underperforms buy-and-hold")

if bootstrap_results['ci_lower'] > 0:
    interview_score += 2
    print("  ✅ Sharpe CI excludes zero (robust)")
else:
    print("  ⚠️  Sharpe CI includes zero")

if metrics.get('max_drawdown_pct', 100) < 20:
    interview_score += 1
    print("  ✅ Max drawdown < 20%")
else:
    print("  ⚠️  Max drawdown >= 20%")

print(f"\n🎯 OVERALL INTERVIEW SCORE: {interview_score}/8")

if interview_score >= 7:
    print("   Rating: 9/10 - EXCELLENT, Interview Ready ⭐⭐⭐")
elif interview_score >= 5:
    print("   Rating: 7-8/10 - GOOD, Strong Candidate ⭐⭐")
elif interview_score >= 3:
    print("   Rating: 6/10 - PASS, Needs Improvement ⭐")
else:
    print("   Rating: 4-5/10 - WEAK, More Work Needed")

print("\n" + "=" * 80)
print(f"⏰ Execution completed: {datetime.now()}")
print("=" * 80)

print("\n✅ ALL STEPS COMPLETED SUCCESSFULLY!")
print("\nResults saved to:")
print("  - D:/klypto/results/equity_curve.csv")
print("  - D:/klypto/results/trades.csv")
print("  - D:/klypto/results/trading_signals.csv")
print("  - D:/klypto/results/strategy_comparison.csv")
print("  - D:/klypto/results/statistical_tests.json")
print("  - D:/klypto/results/bootstrap_results.json")
print("  - D:/klypto/plots/*.png")
