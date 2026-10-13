# Sericulture ETL & ML Pipeline - Comprehensive Correction & Audit Log

**Author:** Antigravity AI Engineering  
**Date:** 2026-10-12  
**Target Repository:** `/Users/pavanajmadhu/documents/ETL`  
**Primary Files Modified:**  
1. `Cleaner/cocoon_data_clean.py` (Pass 1 Domain Cleaning, Standardization & Merging)  
2. `Model/module_1_cleaning.py` (Pass 2 ML Guardrails, Type Casting & Outlier Filtering)  

---

## 1. Executive Summary & Problem Isolation

### 1.1 The Issue
When running the master orchestrator (`main.py`) in `single` inference mode:
```bash
python3 main.py --breakeven 500 --market ramanagara --silk_type BV --mode single
```
The pipeline crashed at Stage 4 with an empty DataFrame error:
```
❌ Still empty after mask. Let's check rows matching ONLY market 'ramanagara':
Empty DataFrame
Columns: [date_resolved, market, is_BV, avg_price]
Index: []
```
Despite raw scraper data having **2,880 rows for Ramanagara** and **2,246 rows for Kolar**, the final cleaned dataset contained **0 rows for Ramanagara** and **0 rows for Kolar**, leaving only **201 rows of Kanakapura**.

---

## 2. Root Cause Analysis (RCA)

Our investigation diagnosed four interconnected defects across the cleaning layers:

```
[Raw Scraped Data]
 ├── Kanakapura: Uses '-' in date (DD-MM-YYYY, e.g., 02-01-2022)
 ├── Ramanagara: Uses '/' in date (DD/MM/YYYY, e.g., 01/01/2023)
 └── Kolar:      Uses '/' in date (DD/MM/YYYY, e.g., 01/01/2024)
          │
          ▼ (pd.concat in Pass 1: Kanakapura rows appear first)
[pd.to_datetime(combined_df['date'], errors='coerce')]
 ├── Infers format from first rows ('%d-%m-%Y')
 ├── Silently coerces all '/' rows (Ramanagara & Kolar) to NaT!
 └── Without dayfirst=True, '02-01-2022' parses as Feb 1 instead of Jan 2 (offset -335 vs -365)
          │
          ▼
[date_offset_days = (date_parsed - url_date).dt.days]
 └── Evaluates to NaN for 100% of Ramanagara and Kolar rows!
          │
          ▼
[pd.get_dummies(combined_df, columns=['market'])]
 └── Drops the original canonical 'market' string column!
          │
          ▼
[Pass 2: df = df[df['date_offset_days'].abs() <= 30]]
 └── IN PANDAS: NaN <= 30 EVALUATES TO FALSE!
 └── Silently drops ALL 2,544 Ramanagara rows and ALL 2,212 Kolar rows!
          │
          ▼
[Target Output: 0 Ramanagara rows → main.py query mask returns Empty DataFrame]
```

### Defect 1: Mixed Date Delimiters & Silent `NaT` Coercion (`cocoon_data_clean.py`)
* **Cause:** Kanakapura CSV dates use hyphens (`-`), whereas Ramanagara and Kolar use slashes (`/`). In pandas 2.0+, `pd.to_datetime()` on a Series with `errors='coerce'` infers the format from the first non-null element (`%d-%m-%Y`).
* **Impact:** Every single row with slashes (`01/01/2023`) failed format matching and was silently coerced to `NaT`.
* **Secondary Date Bug:** Without `dayfirst=True`, Kanakapura's `02-01-2022` (January 2nd) was parsed as `2022-02-01` (February 1st). The year-bug rule looking for offset `-365` failed to match (got `-335`), leaving 727 Kanakapura rows uncorrected.

### Defect 2: Lethal Date Offset Filter (`module_1_cleaning.py`)
* **Cause:** Line 68 of `Model/module_1_cleaning.py` had:
  ```python
  if 'date_offset_days' in df.columns:
      df = df[df['date_offset_days'].abs() <= 30]
  ```
* **Impact:** In pandas, `np.nan.abs() <= 30` evaluates to `False`. Because `date_offset_days` was `NaN` for all Ramanagara and Kolar rows, **this filter dropped 100% of Ramanagara and Kolar data**. (The intended logic in the cleaner comments had been: `df['date_offset_days'].isna() | (df['date_offset_days'].abs() <= threshold)`).

### Defect 3: Premature Dropping of the Canonical `market` Column
* **Cause:** `pd.get_dummies(combined_df, columns=['market'], prefix='market')` removed the `market` string column.
* **Impact:** Pass 2 had to redundantly reconstruct `market` via `derive_market_name()`. Any whitespace, missing one-hot flags, or type coercion failures resulted in `'unknown'`, silently dropping rows.

### Defect 4: Categorical Type Casting & Whitespace Inconsistencies
* **Cause:** `variety` was mapped with a rigid check (`x == 'BV'`), vulnerable to `'bv'`, `'Bi-Voltine'`, or trailing spaces (`'BV '`). Furthermore, `is_BV` was not strictly cast to an integer flag (`int64`), causing potential mismatch in `feature_df['is_BV'] == 1`.

---

## 3. Changes Made

### 3.1 Changes in `Cleaner/cocoon_data_clean.py`
1. **BOM & Encoding Safety:**
   - Switched CSV readers to `encoding='utf-8-sig'` to handle any hidden UTF-8 Byte Order Marks (`\xef\xbb\xbf`).
   - Added `sanitize_text` to normalize Unicode (NFKC) and strip trailing whitespace, tabs, and carriage returns (`\r`).
2. **Robust Multi-Format URL Date Extraction:**
   - Updated regex from `\d{2}` to `\d{1,2}` to handle single-digit days.
   - Tested formats `%d-%m-%Y`, `%d-%B-%Y`, and `%d-%b-%Y`.
3. **Delimiter Harmonization & Day-First Date Parsing:**
   - Replaced all `/` with `-` in `combined_df['date']` before parsing.
   - Explicitly used `dayfirst=True` for Indian market conventions.
   - Imputed missing dates from `url_date`.
4. **Year-Bug & Offset Alignment:**
   - Handled year bugs with offset in `[-365, -366]` (leap years).
   - Set effective `date_offset_days` to `0.0` for corrected rows so Pass 2 does not discard them.
5. **Preserved Canonical Columns:**
   - Standardized `combined_df['market']` as lowercase clean strings.
   - Generated `market_kanakapura`, `market_kolar`, and `market_ramanagara` without dropping `market`.
   - Mapped `variety` robustly to `is_BV` (strict integer `0` or `1`).

### 3.2 Changes in `Model/module_1_cleaning.py`
1. **Datetime Type Agnostic `parse_dates`:**
   - Checks `pd.api.types.is_datetime64_any_dtype(df['date_resolved'])` to avoid `ValueError: Cannot use 'format' when passing a datetime-like Series`.
   - Delimiter harmonization with fallback to `dayfirst=True`.
2. **Hardened `derive_market_name`:**
   - First normalizes existing `market` column if present (NFKC, regex non-alpha strip).
   - Falls back to `market_*` one-hot flags using numeric coercion.
   - Falls back to `source_url` substring matching.
   - Guarantees `market` is a clean `str` type.
3. **Safe Offset Filtering:**
   - Replaced `df['date_offset_days'].abs() <= 30` with:
     ```python
     offset_num = pd.to_numeric(df['date_offset_days'], errors='coerce')
     extreme_offset_mask = offset_num.notna() & (offset_num.abs() > 30)
     df = df[~extreme_offset_mask]
     ```
   - Retains rows where `date_offset_days` is `NaN` (resolved/imputed dates).
4. **Strict Categorical & Numeric Guardrails:**
   - Coerces `avg_price`, `min_price`, `max_price`, `qty_kg`, `lots` to numeric.
   - Enforces `is_BV` strictly as integer `0` or `1`.

---

## 4. Execution Logs & Task History

### Task 1: Environment & File Inspection
* **Task:** Initial diagnostic run on CSV headers and structure.
* **Findings:** Raw datasets contain:
  - `Market_data/kanakapura/cocoon_prices.csv`: 2,994 rows, columns: `['market', 'date', 'variety', 'lots', 'qty_kg', 'min_price', 'max_price', 'avg_price', 'source_url']`
  - `Market_data/ramanagara/cocoon_prices.csv`: 2,880 rows
  - `Market_data/kolar/cocoon_prices.csv`: 2,246 rows

### Task 2: Step-by-Step Filter Simulation
* **Script:** `scratch/test_pipeline_steps.py`
* **Log Output:**
  ```
  Loaded data.csv, shape: (7716, 24)
  After derive_market_name: {'kanakapura': 2960, 'ramanagara': 2544, 'kolar': 2212}
  --- With old offset filter (df['date_offset_days'].abs() <= 30) ---
  After date_offset <= 30: {'kanakapura': 232}   <-- RAMANAGARA & KOLAR COMPLETELY WIPED OUT!
  
  --- With corrected offset filter (~(notna & abs > 30)) ---
  After proper offset filter: {'ramanagara': 2544, 'kanakapura': 2233, 'kolar': 2212}
  After zero drops: {'ramanagara': 2544, 'kanakapura': 2232, 'kolar': 2212}
  After logic drops: {'ramanagara': 2501, 'kanakapura': 2223, 'kolar': 2208}
  After dup drops: {'ramanagara': 2501, 'kanakapura': 1870, 'kolar': 1818}
  ```

### Task 3: Delimiter Failure Reproduction
* **Test:**
  ```python
  s = pd.Series(['02-01-2022', '03-01-2022', '01/01/2023', '01/02/2024'])
  pd.to_datetime(s, errors='coerce')
  ```
* **Result:**
  ```
  0   2022-02-01
  1   2022-03-01
  2          NaT    <-- SILENT COERCION OF '/'
  3          NaT
  ```

### Task 4: Full Execution of `Cleaner/cocoon_data_clean.py`
* **Command:** `python3 Cleaner/cocoon_data_clean.py`
* **Execution Log:**
  ```
  Loading files from Google Drive...
  Shape of df1: (2994, 9)
  Shape of df2: (2880, 9)
  Shape of df3: (2246, 9)
  Shape of combined_df: (8120, 9)

  Count of rows where url_date was successfully extracted: 8084
  Count of rows where url_date could not be extracted (unresolvable): 36
  Number of nulls in 'date' column after imputation: 4

  Value counts for date_correction_applied:
  none                  7956
  kept_col_small_lag      86
  corrected_year_bug      78

  Unique values in 'variety' before encoding: ['CB', 'BV']
  Value counts of 'is_BV' after encoding:
  is_BV
  0    4070
  1    4050

  Final DataFrame Shape: (8116, 23)
  Rows with |offset| > 30 days: 0

  === MARKET COVERAGE PER YEAR ===
  year  market    
  2022  kanakapura    517
        kolar         327
        ramanagara    481
  2023  kanakapura    587
        kolar         248
        ramanagara    497
  2024  kanakapura    590
        kolar         498
        ramanagara    582
  2025  kanakapura    620
        kolar         555
        ramanagara    632
  2026  kanakapura    678
        kolar         618
        ramanagara    686
  ```

### Task 5: Pass 2 Clean Data Verification
* **Test:** `clean_data(pd.read_csv('Cleaner/data.csv'))`
* **Execution Log:**
  ```
  Clean df shape: (6705, 9)
  Markets: {'kanakapura': 2519, 'ramanagara': 2363, 'kolar': 1823}
  BV split:
  is_BV          0     1
  market                
  kanakapura  1273  1246
  kolar        913   910
  ramanagara  1178  1185
  ```

### Task 6: Feature Engineering Verification
* **Test:** `generate_features(clean_df)`
* **Execution Log:**
  ```
  feature_df shape: (6705, 40)
  Market kanakapura, silk_type CB: 1273 matching inference rows
  Market kanakapura, silk_type BV: 1246 matching inference rows
  Market ramanagara, silk_type CB: 1178 matching inference rows
  Market ramanagara, silk_type BV: 1185 matching inference rows
  Market kolar, silk_type CB: 913 matching inference rows
  Market kolar, silk_type BV: 910 matching inference rows
  ```

### Task 7: End-to-End Orchestrator Run (`main.py`)
* **Command:** `python3 main.py --breakeven 500 --market ramanagara --silk_type BV --mode single`
* **Execution Log:**
  ```
  === STAGE 1: SCRAPING NEW DATA ===
  [KOLAR] Scraping data newer than 2026-09-27...
    ...Extracted 13/13 articles (26 rows written)
  [RAMANAGARA] Scraping data newer than 2026-09-27...
    ...Extracted 15/15 articles (30 rows written)
  [KANAKAPURA] Scraping data newer than 2026-09-27...
    ...Extracted 13/13 articles (26 rows written)

  === STAGE 2: DOMAIN CLEANING & MERGING ===
  Final processed data exported to 'Cleaner/data.csv'

  === STAGE 3: ML GUARDRAIL CLEANING & MODELING ===
  Loading merged dataset...
  Applying ML-specific guardrails (Pass 2 cleaning)...
  Engineering temporal and momentum features...
  Training final production models...
  [LightGBM] [Info] Number of data points in the train set: 4920, number of used features: 37

  --- DATASET DIAGNOSTICS ---
  Total rows in feature_df: 6787
  Unique markets present: ['kanakapura', 'kolar', 'ramanagara']
  Unique is_BV values present: [0, 1]
  Date range: 2022-02-10 to 2026-10-12
  Future rows count: 82
  ----------------------------

  =======================================================
   🐛 SERICULTURE FARMER PRICE ADVISORY 🐛 
  =======================================================
  Rearing Start Date : 2026-09-26
  Expected Harvest   : 2026-10-26
  Market             : Ramanagara
  Silk Type          : Bivoltine (BV)
  Break-even Price   : ₹500/kg
  -------------------------------------------------------
  PRICE FORECAST (in 30 days):
    Floor (P10)      : ₹715/kg
    Expected (P50)   : ₹804/kg
    Ceiling (P90)    : ₹820/kg
  -------------------------------------------------------
  Expected Profit    : ₹304/kg (at P50)
  DECISION           : GO
  Rationale          : Proceed. Even the worst-case (10th percentile) expected price clears your break-even cost.
  =======================================================
  ```

---

## 5. Verification Checklist

| Check | Expected Behavior | Status |
|---|---|---|
| **BOM Headers** | Read using `utf-8-sig` across all sources | Verified |
| **Market Whitespace** | Stripped and lowercased (`'ramanagara'`) | Verified |
| **Date Delimiters** | Normalized `/` and `-` with `dayfirst=True` | Verified |
| **Year-Bug Resolution** | Offsets in `[-365, -366]` aligned to `url_date` | Verified |
| **Pass 2 Offset Filter** | Only drops `notna & abs > 30`; preserves `NaN` | Verified |
| **Market Column** | Retained in `Cleaner/data.csv` and `clean_df` | Verified |
| **Categorical Flag** | `is_BV` strictly integer `0` (CB) or `1` (BV) | Verified |
| **Ramanagara BV Query** | 1,185 matching rows found; advisory succeeds | Verified |

---
*Report generated and committed to project root as `AGY_correction.md`.*
