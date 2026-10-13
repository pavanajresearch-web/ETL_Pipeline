import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

# Suppress LightGBM categorical warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def calculate_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Calculates the pinball loss (quantile loss) for a given tau."""
    err = y_true - y_pred
    return np.mean(np.maximum(tau * err, (tau - 1) * err))

def evaluate_metrics(y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> dict:
    """Computes quantile-specific metrics: Pinball loss, MAE, and coverage."""
    # Ensure monotonicity for evaluation purposes
    p10_adj = np.minimum(p10, p50)
    p90_adj = np.maximum(p90, p50)
    
    # Coverage calculation
    coverage = np.mean((y_true >= p10_adj) & (y_true <= p90_adj)) * 100
    
    return {
        'pinball_10': calculate_pinball_loss(y_true, p10_adj, 0.10),
        'pinball_50': calculate_pinball_loss(y_true, p50, 0.50),
        'pinball_90': calculate_pinball_loss(y_true, p90_adj, 0.90),
        'mae_50': np.mean(np.abs(y_true - p50)),
        'coverage_pct': coverage
    }

def train_quantile_models(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> dict:
    """Trains the three required P10, P50, and P90 LightGBM models."""
    base_params = {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1
    }
    if params:
        base_params.update(params)
        
    models = {}
    for tau, name in zip([0.10, 0.50, 0.90], ['p10', 'p50', 'p90']):
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=tau,
            **base_params
        )
        model.fit(X_train, y_train)
        models[name] = model
        
    return models

def prepare_features(df: pd.DataFrame) -> tuple:
    """Extracts feature matrix X and target y, defining categorical columns."""
    df = df.copy()
    
    # Convert market to category for native LightGBM handling
    if 'market' in df.columns and not pd.api.types.is_categorical_dtype(df['market']):
        df['market'] = df['market'].astype('category')
        
    # Exclude non-feature or leaky columns
    exclude_cols = ['date_resolved', 'target_price_30d', 'is_future_row']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    return df, feature_cols

def walk_forward_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes year-level walk-forward validation.
    Enforces a strict 30-day boundary buffer to prevent label leakage.
    """
    df, feature_cols = prepare_features(df)
    
    # Drop rows without targets (e.g., the last 30 days of the dataset and future rows)
    # This is standard for training/validation sets.
    valid_df = df.dropna(subset=['target_price_30d'])
    
    test_years = [2023, 2024, 2025, 2026]
    results = []
    
    for year in test_years:
        test_start_date = pd.Timestamp(f'{year}-01-01')
        
        # STRICT LEAKAGE PREVENTION:
        # A row at test_start_date - 1 day predicts for +29 days into the test year.
        # To prevent the target of the training set overlapping with the test period,
        # training must end exactly 30 days BEFORE the test start date.
        train_end_date = test_start_date - pd.Timedelta(days=30)
        
        train_mask = valid_df['date_resolved'] < train_end_date
        test_mask = valid_df['date_resolved'].dt.year == year
        
        train_df = valid_df[train_mask]
        test_df = valid_df[test_mask]
        
        if len(train_df) == 0 or len(test_df) == 0:
            continue
            
        X_train, y_train = train_df[feature_cols], train_df['target_price_30d']
        X_test, y_test = test_df[feature_cols], test_df['target_price_30d']
        
        # Train
        models = train_quantile_models(X_train, y_train)
        
        # Predict
        p10_pred = models['p10'].predict(X_test)
        p50_pred = models['p50'].predict(X_test)
        p90_pred = models['p90'].predict(X_test)
        
        # Evaluate
        metrics = evaluate_metrics(y_test.values, p10_pred, p50_pred, p90_pred)
        metrics['fold_year'] = year
        metrics['train_size'] = len(X_train)
        metrics['test_size'] = len(X_test)
        results.append(metrics)
        
    results_df = pd.DataFrame(results)
    print("\n--- Walk-Forward Validation Results ---")
    print(results_df[['fold_year', 'pinball_10', 'pinball_50', 'pinball_90', 'mae_50', 'coverage_pct']].round(2).to_string(index=False))
    
    return results_df

def train_final_models(df: pd.DataFrame) -> dict:
    """
    Trains the final production models on all historical data.
    These models are shipped to inference.
    """
    df, feature_cols = prepare_features(df)
    
    # Exclude rows lacking targets (last 30 days of actuals + future_rows)
    train_df = df.dropna(subset=['target_price_30d'])
    
    X_train = train_df[feature_cols]
    y_train = train_df['target_price_30d']
    
    print(f"\nTraining final production models on {len(X_train)} historical records...")
    models = train_quantile_models(X_train, y_train)
    
    # Store feature columns in the dict for easy inference alignment
    models['feature_cols'] = feature_cols
    
    return models

if __name__ == "__main__":
    print("Module 3 ready. Exposes: walk_forward_validation(df), train_final_models(df)")