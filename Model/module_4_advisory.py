import pandas as pd
import numpy as np

def enforce_monotonicity(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray):
    """Guarantees P10 <= P50 <= P90 to fix any quantile crossing at the tails."""
    p10_adj = np.minimum(p10, p50)
    p90_adj = np.maximum(p90, p50)
    return p10_adj, p50, p90_adj

def determine_decision(p10: float, p50: float, breakeven: float) -> tuple:
    """Applies Talebian negatively-convex agent rules for ruin avoidance."""
    buffer_threshold = 0.85 * breakeven
    
    # Strictly handle all boundary logic explicitly
    if p10 > breakeven:
        decision = "GO"
        rationale = "Proceed. Even the worst-case (10th percentile) expected price clears your break-even cost."
    elif p50 > breakeven and p10 > buffer_threshold:
        decision = "CAUTION"
        rationale = "Expected price is profitable, but there is some tail risk of falling slightly below break-even."
    else:
        # This catches p10 <= 0.85*breakeven OR p50 <= breakeven
        decision = "ABSTAIN"
        rationale = "Do not rear. High risk of market prices falling well below your break-even threshold."
        
    return decision, rationale

def generate_single_advisory(models: dict, feature_row: pd.DataFrame, breakeven: float):
    """Runs inference on a single row and prints a formatted advisory report."""
    X = feature_row[models['feature_cols']]
    
    # Predict
    booster = getattr(models['p10'], 'booster_', models['p10'])
    print("Saved pandas categories:", getattr(booster, 'pandas_categorical', None))
    print("Current X dtypes:\n", X.dtypes)
    import pandas as pd

    booster = getattr(models['p10'], 'booster_', models['p10'])
    if hasattr(booster, 'pandas_categorical') and booster.pandas_categorical:
    # booster.pandas_categorical contains list of category lists in order of categorical features
              X['market'] = pd.Categorical(X['market'], categories=booster.pandas_categorical[0])
    p10_raw = models['p10'].predict(X)[0]
    p50_raw = models['p50'].predict(X)[0]
    p90_raw = models['p90'].predict(X)[0]
    
    # Enforce constraints
    p10, p50, p90 = enforce_monotonicity(p10_raw, p50_raw, p90_raw)
    decision, rationale = determine_decision(p10, p50, breakeven)
    
    # Meta info
    start_date = feature_row['date_resolved'].iloc[0]
    harvest_date = start_date + pd.Timedelta(days=30)
    market = feature_row['market'].iloc[0].capitalize()
    silk_type = "Bivoltine (BV)" if feature_row['is_BV'].iloc[0] == 1 else "Multivoltine (MV)"
    expected_profit = p50 - breakeven
    
    print("\n" + "="*55)
    print(" 🐛 SERICULTURE FARMER PRICE ADVISORY 🐛 ")
    print("="*55)
    print(f"Rearing Start Date : {start_date.strftime('%Y-%m-%d')}")
    print(f"Expected Harvest   : {harvest_date.strftime('%Y-%m-%d')}")
    print(f"Market             : {market}")
    print(f"Silk Type          : {silk_type}")
    print(f"Break-even Price   : ₹{breakeven:.0f}/kg")
    print("-" * 55)
    print("PRICE FORECAST (in 30 days):")
    print(f"  Floor (P10)      : ₹{p10:.0f}/kg")
    print(f"  Expected (P50)   : ₹{p50:.0f}/kg")
    print(f"  Ceiling (P90)    : ₹{p90:.0f}/kg")
    print("-" * 55)
    print(f"Expected Profit    : ₹{expected_profit:.0f}/kg (at P50)")
    print(f"DECISION           : {decision}")
    print(f"Rationale          : {rationale}")
    print("="*55 + "\n")

def run_bulk_backtest(models: dict, df: pd.DataFrame, current_breakeven: float, inflation_rate: float = 0.07):
    """Evaluates the advisory system over historical rows using an inflation-adjusted break-even."""
    test_df = df.dropna(subset=['target_price_30d']).copy()
    
    # VERY IMPORTANT: Ensure categorical types perfectly match training for bulk inference
    if 'market' in models['feature_cols'] and not isinstance(test_df['market'].dtype, pd.CategoricalDtype):
        test_df['market'] = test_df['market'].astype('category')
        
    X = test_df[models['feature_cols']]
    
    p10_raw = models['p10'].predict(X)
    p50_raw = models['p50'].predict(X)
    p90_raw = models['p90'].predict(X)
    
    p10,p50,p90 = enforce_monotonicity(p10_raw, p50_raw, p90_raw)
    
    # 1. DYNAMIC BREAK-EVEN CALCULATION
    # Anchor the calculation to the most recent date in the dataset
    anchor_date = test_df['date_resolved'].max()
    
    # Calculate exactly how many years ago each row occurred
    years_ago = (anchor_date - test_df['date_resolved']).dt.days / 365.25
    
    # Discount the current break-even backward in time
    test_df['dynamic_breakeven'] = current_breakeven / ((1 + inflation_rate) ** years_ago)
    
    # 2. VECTORIZED DECISION LOGIC
    buffer_threshold = 0.85 * test_df['dynamic_breakeven']
    
    conditions = [
        (p10 > test_df['dynamic_breakeven']),
        (p50 > test_df['dynamic_breakeven']) & (p10 > buffer_threshold)
    ]
    choices = ['GO', 'CAUTION']
    test_df['decision'] = np.select(conditions, choices, default='ABSTAIN')
    
    # 3. EVALUATION LOGIC
    test_df['actual_profitable'] = test_df['target_price_30d'] >= test_df['dynamic_breakeven']
    test_df['advised_proceed'] = test_df['decision'].isin(['GO', 'CAUTION'])
    
    correct_calls = (test_df['actual_profitable'] == test_df['advised_proceed']).sum()
    total_calls = len(test_df)
    accuracy = (correct_calls / total_calls) * 100
    
    ruin_events = (test_df['actual_profitable'] == False).sum()
    ruin_prevented = ((test_df['actual_profitable'] == False) & (test_df['advised_proceed'] == False)).sum()
    ruin_prevention_rate = (ruin_prevented / ruin_events * 100) if ruin_events > 0 else 100
    
    # Display sample of dynamic break-evens for sanity check
    print("\n--- Break-even Historical Sample (Inflation Adj) ---")
    years_sample = test_df.groupby(test_df['date_resolved'].dt.year)['dynamic_breakeven'].mean()
    for year, be in years_sample.items():
        print(f"  {year}: ₹{be:.0f}/kg")

    print("\n--- Bulk Backtest Results ---")
    print(f"Total historical decisions tested : {total_calls}")
    print(f"Overall Directional Accuracy      : {accuracy:.1f}%")
    print(f"Ruin Events Prevented (Recall)    : {ruin_prevented}/{ruin_events} ({ruin_prevention_rate:.1f}%)")
    print("\nHistorical Decision Distribution:")
    print(test_df['decision'].value_counts().to_string())
    print("-" * 30)

if __name__ == "__main__":
    print("Module 4 ready. Exposes: generate_single_advisory(), run_bulk_backtest()")
