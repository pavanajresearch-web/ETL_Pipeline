import pandas as pd
import numpy as np
import unicodedata
from datetime import datetime

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parses date_resolved with ISO first, falling back to dayfirst."""
    if 'date_resolved' not in df.columns:
        return df
    
    # If already datetime-like, simply drop NaTs and return
    if pd.api.types.is_datetime64_any_dtype(df['date_resolved']):
        return df.dropna(subset=['date_resolved'])
        
    s = df['date_resolved'].astype(str).str.strip().str.replace('/', '-', regex=False)
    # Attempt ISO8601 parsing first
    parsed_iso = pd.to_datetime(s, format='ISO8601', errors='coerce')
    
    # Identify failures and try dayfirst=True
    failed_mask = parsed_iso.isna() & df['date_resolved'].notna()
    if failed_mask.any():
        parsed_iso[failed_mask] = pd.to_datetime(
            s[failed_mask], 
            dayfirst=True, 
            errors='coerce'
        )
        
    df['date_resolved'] = parsed_iso
    return df.dropna(subset=['date_resolved'])

def derive_market_name(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstructs and standardizes a single clean 'market' column for grouped operations."""
    df = df.copy()
    
    # 1. If 'market' column exists, clean, normalize unicode and strip hidden whitespace
    if 'market' in df.columns:
        market_clean = df['market'].astype(str).apply(
            lambda x: unicodedata.normalize('NFKC', str(x)).strip().lower() if pd.notna(x) else ""
        )
        market_clean = market_clean.str.replace(r'[^a-z]', '', regex=True)
        valid_mask = market_clean.isin(['kanakapura', 'kolar', 'ramanagara'])
        df['market'] = np.where(valid_mask, market_clean, 'unknown')
    else:
        df['market'] = 'unknown'
        
    # 2. For unknown rows, reconstruct from one-hot dummy flags if present
    unknown_mask = df['market'] == 'unknown'
    if unknown_mask.any():
        for m in ['kanakapura', 'kolar', 'ramanagara']:
            col = f'market_{m}'
            if col in df.columns:
                is_m = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int) == 1
                df.loc[unknown_mask & is_m, 'market'] = m
                unknown_mask = df['market'] == 'unknown'
                
    # 3. Fallback to source_url substring matching if still unknown
    if unknown_mask.any() and 'source_url' in df.columns:
        for m in ['kanakapura', 'kolar', 'ramanagara']:
            url_match = df['source_url'].astype(str).str.lower().str.contains(m, na=False)
            df.loc[unknown_mask & url_match, 'market'] = m
            unknown_mask = df['market'] == 'unknown'
            
    filtered_df = df[df['market'].isin(['kanakapura', 'kolar', 'ramanagara'])].copy()
    filtered_df['market'] = filtered_df['market'].astype(str)
    
    # Warning check to help diagnose if data is entirely wiped out
    if filtered_df.empty:
        print("⚠️ WARNING: All rows were dropped in derive_market_name! Check your 0/1 market flags.")
        
    return filtered_df

def remove_qty_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drops extreme qty_kg outliers using a per-market 10x IQR fence."""
    if df.empty:
        return df
        
    # Using .transform() instead of .apply() guarantees the DataFrame structure 
    # (and the 'market' column) is never accidentally dropped by pandas.
    Q1 = df.groupby('market')['qty_kg'].transform(lambda x: x.quantile(0.25))
    Q3 = df.groupby('market')['qty_kg'].transform(lambda x: x.quantile(0.75))
    IQR = Q3 - Q1
    upper_bound = Q3 + 10 * IQR
    
    # Keep rows under the fence, or rows where a fence couldn't be calculated
    return df[(df['qty_kg'] <= upper_bound) | upper_bound.isna()]

def clean_data(raw_df: pd.DataFrame, summary_path: str = "summary.txt") -> pd.DataFrame:
    """Master cleaning pipeline applying all constraint rules and logging a summary."""
    initial_rows = len(raw_df)
    
    # 1. Parse dates and create standardized market column
    df = parse_dates(raw_df.copy())
    df = derive_market_name(df)
    
    # 2. Coerce numeric columns safely
    for num_col in ['avg_price', 'min_price', 'max_price', 'qty_kg', 'lots']:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
            
    # 3. Ensure is_BV is strictly an integer flag (0 or 1)
    if 'is_BV' in df.columns:
        df['is_BV'] = pd.to_numeric(df['is_BV'], errors='coerce').fillna(0).astype(int)
        df['is_BV'] = np.where(df['is_BV'] == 1, 1, 0)
    
    # 4. Flag future-dated rows
    df['is_future_row'] = df['date_resolved'] > pd.Timestamp('2026-09-27')
    
    # 5. Extreme date offset filter: safely drop only rows where offset > 30 days
    # Do NOT drop rows with NaN offset (which occur on resolved/imputed dates)
    if 'date_offset_days' in df.columns:
        offset_num = pd.to_numeric(df['date_offset_days'], errors='coerce')
        extreme_offset_mask = offset_num.notna() & (offset_num.abs() > 30)
        offset_drops = int(extreme_offset_mask.sum())
        df = df[~extreme_offset_mask]
    else:
        offset_drops = 0
    
    zero_mask = (df['avg_price'] <= 0) | (df['qty_kg'] <= 0) | (df['lots'] <= 0) | df['avg_price'].isna() | df['qty_kg'].isna() | df['lots'].isna()
    zero_drops = int(zero_mask.sum())
    df = df[~zero_mask]
    
    logic_mask = (df['min_price'] > df['avg_price']) | (df['avg_price'] > df['max_price']) | df['min_price'].isna() | df['max_price'].isna()
    logic_drops = int(logic_mask.sum())
    df = df[~logic_mask]
    
    dup_drops = int(df['flag_duplicate'].sum()) if 'flag_duplicate' in df.columns else 0
    if 'flag_duplicate' in df.columns:
        df = df[~df['flag_duplicate'].fillna(False).astype(bool)]
        
    pre_iqr_count = len(df)
    df = remove_qty_outliers(df)
    iqr_drops = pre_iqr_count - len(df)
    
    # Ensure required columns exist before filtering
    cols_to_keep = [
        'date_resolved', 'market', 'is_BV', 'avg_price', 
        'min_price', 'max_price', 'qty_kg', 'lots', 'is_future_row'
    ]
    # Only keep columns that actually exist in the dataframe to prevent KeyError
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    
    cleaned_df = df[cols_to_keep].sort_values(['market', 'is_BV', 'date_resolved']).reset_index(drop=True)
    
    # Final type enforcement
    cleaned_df['market'] = cleaned_df['market'].astype(str)
    if 'is_BV' in cleaned_df.columns:
        cleaned_df['is_BV'] = cleaned_df['is_BV'].astype(int)
    
    # GENERATE CLEANING SUMMARY LOG
    final_rows = len(cleaned_df)
    summary_text = f"""=======================================================
 🧹 DATA CLEANING AUDIT SUMMARY 🧹
 Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=======================================================
Raw Input Rows       : {initial_rows}
Cleaned Output Rows   : {final_rows}
Total Dropped Rows   : {initial_rows - final_rows}

DROPPED BREAKDOWN:
  - Date Offset (>30d) : {offset_drops}
  - Zero Artifacts     : {zero_drops}
  - Price Logic Errors : {logic_drops}
  - Duplicates         : {dup_drops}
  - Quantity Outliers  : {iqr_drops}

DATASET BOUNDS:
  - Earliest Date      : {cleaned_df['date_resolved'].min().strftime('%Y-%m-%d') if not cleaned_df.empty else 'N/A'}
  - Latest Market Date : {cleaned_df[~cleaned_df['is_future_row']]['date_resolved'].max().strftime('%Y-%m-%d') if not cleaned_df.empty else 'N/A'}
  - Future Rows Kept   : {cleaned_df['is_future_row'].sum() if not cleaned_df.empty else 0}
=======================================================
"""
    # Write to summary_path
    with open(summary_path, "w") as f:
        f.write(summary_text)
        
    return cleaned_df

if __name__ == "__main__":
    # Test logic
    print("Module 1 ready. Exposes: clean_data(raw_df)")