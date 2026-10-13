import pandas as pd
import numpy as np

def compute_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes temporal variables to sine/cosine coordinates."""
    doys = df['date_resolved'].dt.dayofyear
    
    df['sin_doy'] = np.sin(2 * np.pi * doys / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * doys / 365.25)
    
    # 6 crop cycles per year (~60 days)
    df['sin_crop_cycle'] = np.sin(2 * np.pi * (doys % 60) / 60.0)
    df['cos_crop_cycle'] = np.cos(2 * np.pi * (doys % 60) / 60.0)
    
    df['month'] = df['date_resolved'].dt.month
    df['post_glut_recovery'] = (df['date_resolved'].dt.year >= 2025).astype(int)
    return df

def compute_cross_market_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """H. Divergence: CV of price across markets per day/silk-type."""
    def calc_cv(x):
        return x.std() / x.mean() if len(x) > 1 and x.mean() > 0 else 0.0

    divergence = df.groupby(['date_resolved', 'is_BV'])['avg_price'].apply(calc_cv).reset_index()
    divergence.rename(columns={'avg_price': 'cross_market_cv'}, inplace=True)
    
    return df.merge(divergence, on=['date_resolved', 'is_BV'], how='left')

def compute_rolling_and_lags(group: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all windowed features. 
    Applies shift(1) explicitly to ALL features prior to any rolling operation 
    to absolutely guarantee no lookahead leakage.
    """
    group = group.sort_values('date_resolved').copy()
    
    # Set DateIndex to allow time-aware rolling windows (safeguards against missing weekend/holiday data)
    group.set_index('date_resolved', inplace=True)
    
    # Strict shift(1) step: All operations from here on look at "yesterday's" data
    shifted_price = group['avg_price'].shift(1)
    shifted_qty = group['qty_kg'].shift(1)
    
    # C. Lags (Exact days, merged via index)
    for lag in [7, 14, 21, 30]:
        group[f'price_lag_{lag}d'] = shifted_price.shift(lag - 1, freq='D')
        group[f'qty_lag_{lag}d']   = shifted_qty.shift(lag - 1, freq='D')
        
    # D. Rolling Stats (7d, 14d, 21d)
    for window in [7, 14, 21]:
        roll_p = shifted_price.rolling(f'{window}D')
        
        group[f'price_mean_{window}d'] = roll_p.mean()
        group[f'price_std_{window}d'] = roll_p.std()
        group[f'price_cv_{window}d'] = group[f'price_std_{window}d'] / group[f'price_mean_{window}d']

    # B. Market tightness: 30d rolling normalized price rank
    # Note: custom rolling apply to get percentile rank over the last 30 days of data
    group['market_tightness_30d'] = shifted_price.rolling('30D').apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1]) if len(x) > 1 else 0.5, raw=False
    )
    
    # E. Supply Momentum (Velocity and Acceleration)
    velocity = shifted_qty - group['qty_lag_7d']
    group['qty_velocity_7d'] = velocity
    
    # Acceleration is velocity today minus velocity 7 days ago
    shifted_velocity_7d = velocity.shift(7, freq='D')
    group['qty_accel_7d'] = velocity - shifted_velocity_7d
    
    return group.reset_index()

def merge_yoy_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    G. Year-over-year features (364 days ago)
    I. Target Variable: Exact price 30 days in the future.
    """
    # Create target lookup mapping
    target_lookup = df[['date_resolved', 'market', 'is_BV', 'avg_price']].copy()
    
    # To get future price, we subtract 30 days from the lookup date, 
    # so it merges onto today's date.
    target_lookup['date_resolved'] = target_lookup['date_resolved'] - pd.Timedelta(days=30)
    target_lookup.rename(columns={'avg_price': 'target_price_30d'}, inplace=True)
    
    # Create YoY lookup mapping (364 days for day-of-week alignment)
    yoy_lookup = df[['date_resolved', 'market', 'is_BV', 'avg_price', 'qty_kg']].copy()
    yoy_lookup['date_resolved'] = yoy_lookup['date_resolved'] + pd.Timedelta(days=364)
    yoy_lookup.rename(columns={'avg_price': 'price_yoy_lag', 'qty_kg': 'qty_yoy_lag'}, inplace=True)
    
    # Merge them back
    df = df.merge(target_lookup, on=['date_resolved', 'market', 'is_BV'], how='left')
    df = df.merge(yoy_lookup, on=['date_resolved', 'market', 'is_BV'], how='left')
    
    return df

def generate_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Master feature engineering pipeline."""
    df = clean_df.copy()
    
    # Sanity check: Ensure data wasn't wiped out in cleaning
    if df.empty:
        raise ValueError("DataFrame is empty after cleaning! Check your CSV data or filters.")
    
    # A. Price spread (Intraday volatility)
    df['price_spread'] = df['max_price'] - df['min_price']
    
    # Temporal & Cyclical Features
    df = compute_cyclical_features(df)
    
    # H. Cross-market divergence
    df = compute_cross_market_divergence(df)
    
    # B, C, D, E: Rolling, Lags, and Momentum 
    # STRICT BYPASS: Avoid pandas .apply() column-dropping bugs by using a manual loop
    processed_groups = []
    for (market, is_bv), group in df.groupby(['market', 'is_BV']):
        # Compute features for this specific group
        processed = compute_rolling_and_lags(group)
        
        # Forcefully ensure grouping columns are retained
        processed['market'] = market
        processed['is_BV'] = is_bv
        processed_groups.append(processed)
        
    # Recombine the data safely
    df = pd.concat(processed_groups, ignore_index=True)
    
    # G, I: YoY and Target mapping
    df = merge_yoy_and_targets(df)
    
    return df.sort_values(['market', 'is_BV', 'date_resolved']).reset_index(drop=True)

if __name__ == "__main__":
    print("Module 2 ready. Exposes: generate_features(clean_df)")

if __name__ == "__main__":
    print("Module 2 ready. Exposes: generate_features(clean_df)")