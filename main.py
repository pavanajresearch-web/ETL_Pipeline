import argparse
import pandas as pd
import subprocess
import sys
from pathlib import Path

# Import ML Pipeline Modules from Model/ folder
from Model.module_1_cleaning import clean_data
from Model.module_2_features import generate_features
from Model.module_3_model import train_final_models
from Model.module_4_advisory import generate_single_advisory, run_bulk_backtest

def parse_args():
    parser = argparse.ArgumentParser(description="Sericulture Farmer Price Advisory System")
    parser.add_argument("--breakeven", type=float, required=True, help="Farmer break-even price (₹/kg)")
    parser.add_argument("--mode", type=str, choices=['single', 'bulk'], default='single', help="single or bulk mode")
    parser.add_argument("--market", type=str, choices=['kanakapura', 'kolar', 'ramanagara'], help="Required for single mode")
    parser.add_argument("--silk_type", type=str, choices=['BV', 'MV'], help="Required for single mode")
    parser.add_argument("--report", action="store_true", help="Print top-15 feature importance")
    
    args = parser.parse_args()
    
    if args.mode == 'single' and (not args.market or not args.silk_type):
        parser.error("--market and --silk_type are required when --mode is 'single'")
        
    return args

def print_feature_importance(models):
    importances = models['p50'].feature_importances_
    features = models['feature_cols']
    
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)
    
    print("\n--- Top 15 Feature Importances (P50 Model) ---")
    print(feat_df.to_string(index=False))

def main():
    args = parse_args()
    
    # ---------------------------------------------------------
    # STAGE 1: SCRAPER (Market_data/)
    # ---------------------------------------------------------
    print("\n=== STAGE 1: SCRAPING NEW DATA ===")
    try:
        subprocess.run([sys.executable, "Market_data/scraper_code.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Error during scraping. Exiting pipeline.")
        sys.exit(1)
        
    # ---------------------------------------------------------
    # STAGE 2: PASS 1 DOMAIN CLEANER (Cleaner/)
    # ---------------------------------------------------------
    print("\n=== STAGE 2: DOMAIN CLEANING & MERGING ===")
    try:
        subprocess.run([sys.executable, "Cleaner/cocoon_data_clean.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Error during primary cleaning. Exiting pipeline.")
        sys.exit(1)

    # ---------------------------------------------------------
    # STAGE 3: PASS 2 ML CLEANING & FEATURE ENGINEERING (Model/)
    # ---------------------------------------------------------
    print("\n=== STAGE 3: ML GUARDRAIL CLEANING & MODELING ===")
    raw_merged_path = Path("Cleaner/data.csv")
    
    if not raw_merged_path.exists():
        print(f"❌ Error: {raw_merged_path} not found. The cleaner did not output the file.")
        sys.exit(1)
        
    print("Loading merged dataset...")
    merged_df = pd.read_csv(raw_merged_path)
    
    print("Applying ML-specific guardrails (Pass 2 cleaning)...")
    clean_df = clean_data(merged_df, summary_path="Cleaner/summary.txt")
    
    print("Engineering temporal and momentum features...")
    feature_df = generate_features(clean_df)
    
    print("Training final production models...")
    models = train_final_models(feature_df)
    
    if args.report:
        print_feature_importance(models)
        
    # ---------------------------------------------------------
    # STAGE 4: ADVISORY INFERENCE
    # ---------------------------------------------------------
    if args.mode == 'bulk':
        run_bulk_backtest(models, feature_df, args.breakeven)
        
    elif args.mode == 'single':
        # --- DIAGNOSTIC PRINT BLOCK ---
        print("\n--- DATASET DIAGNOSTICS ---")
        print("Total rows in feature_df:", len(feature_df))
        print("Unique markets present:", feature_df['market'].unique() if 'market' in feature_df.columns else "COL 'market' MISSING")
        print("Unique is_BV values present:", feature_df['is_BV'].unique() if 'is_BV' in feature_df.columns else "COL 'is_BV' MISSING")
        print("Date range:", feature_df['date_resolved'].min(), "to", feature_df['date_resolved'].max())
        print("Future rows count:", feature_df['is_future_row'].sum() if 'is_future_row' in feature_df.columns else "COL 'is_future_row' MISSING")
        print("----------------------------\n")
        # -----------------------------

        is_bv_flag = 1 if args.silk_type.upper() == 'BV' else 0
        mask = (feature_df['market'].astype(str).str.lower() == args.market.lower()) & (feature_df['is_BV'] == is_bv_flag)
        
        target_df = feature_df[mask & ~feature_df['is_future_row']].copy()
        
        if target_df.empty:
            print(f"❌ Still empty after mask. Let's check rows matching ONLY market '{args.market}':")
            market_only = feature_df[feature_df['market'].astype(str).str.lower() == args.market.lower()]
            print(market_only[['date_resolved', 'market', 'is_BV', 'avg_price']].tail(5))
            sys.exit(1)
            
        latest_idx = target_df['date_resolved'].idxmax()
        feature_row = target_df.loc[[latest_idx]]
        
        generate_single_advisory(models, feature_row, args.breakeven)
if __name__ == "__main__":
    main()