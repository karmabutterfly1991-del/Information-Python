#!/usr/bin/env python3
"""
Train Local AI models from real DB data over a date range.
Usage (PowerShell):
  python train_from_db.py 2024-12-01 2025-01-31
"""

import sys
from datetime import datetime, timedelta, date
import pandas as pd

from db_config import get_connection
from analysis import train_local_ai, get_local_ai_status, _calculate_scores

def fetch_daily_aggregates(conn, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch all rows and columns from table [dbPayment].[dbo].[Vdetails_CPC_TRUC] for training."""
    query = """
        SELECT *
        FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC]
        WHERE print_q = '5' 
            AND reportdate BETWEEN ? AND ?
            AND (cane_type = '1' OR cane_type = '2')
            AND PRINT_W = 'y' 
            AND WGT_NET > 0 
            AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')
        ORDER BY reportdate, WGT_OUT_DT;
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    return df

def add_comparison_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7-day rolling baselines (excluding current day) for average tons and fresh percent."""
    df = df.copy()
    
    # Debug: Print column names to see what's available
    print(f"Available columns: {list(df.columns)}")
    
    # Find the correct date column name
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'report' in col.lower()]
    print(f"Date-related columns: {date_columns}")
    
    # Use the first available date column
    if date_columns:
        date_col = date_columns[0]
        print(f"Using date column: {date_col}")
        df['report_date'] = pd.to_datetime(df[date_col]).dt.date
    else:
        # Fallback: try common column names
        for col_name in ['reportdate', 'report_date', 'date', 'Date']:
            if col_name in df.columns:
                print(f"Using fallback date column: {col_name}")
                df['report_date'] = pd.to_datetime(df[col_name]).dt.date
                break
        else:
            raise KeyError("No date column found. Available columns: " + str(list(df.columns)))
    
    # Calculate daily aggregates from raw data
    daily_agg = df.groupby('report_date').agg({
        'WGT_NET': ['sum', 'count'],
        'CANE_TYPE': lambda x: (x == '1').sum()  # Count fresh cane
    }).reset_index()
    
    # Flatten column names
    daily_agg.columns = ['report_date', 'total_tons', 'total_trucks', 'fresh_trucks']
    daily_agg['fresh_tons'] = df.groupby('report_date').apply(
        lambda x: x[x['CANE_TYPE'] == '1']['WGT_NET'].sum()
    ).reset_index(drop=True)
    daily_agg['fresh_percent'] = (daily_agg['fresh_tons'] / daily_agg['total_tons'] * 100).fillna(0)
    
    # Add comparison baselines
    daily_agg['avg_daily_tons'] = 0.0
    daily_agg['avg_fresh_percent'] = 0.0
    daily_agg['has_comparison_data'] = False
    
    for idx, row in daily_agg.iterrows():
        current_day = row['report_date']
        start = current_day - timedelta(days=7)
        window = daily_agg[(daily_agg['report_date'] >= start) & (daily_agg['report_date'] < current_day)]
        if not window.empty and window['total_tons'].sum() > 0 and window.shape[0] >= 3:
            avg_daily = window['total_tons'].sum() / window.shape[0]
            avg_fresh_pct = (window['fresh_tons'].sum() / window['total_tons'].sum()) * 100
            daily_agg.at[idx, 'avg_daily_tons'] = avg_daily
            daily_agg.at[idx, 'avg_fresh_percent'] = avg_fresh_pct
            daily_agg.at[idx, 'has_comparison_data'] = True
    
    # Merge back to original dataframe
    df = df.merge(daily_agg[['report_date', 'total_tons', 'fresh_tons', 'fresh_percent', 
                           'avg_daily_tons', 'avg_fresh_percent', 'has_comparison_data']], 
                  on='report_date', how='left')
    
    return df

def add_trend_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7-day vs previous 7-day trend metrics for tons and fresh percent."""
    df = df.copy()
    df['tons_trend_percent'] = 0.0
    df['fresh_trend_percent'] = 0.0
    
    # Get unique dates for trend calculation
    unique_dates = df['report_date'].unique()
    
    for date in unique_dates:
        day = pd.to_datetime(date).date()
        cur_start = day - timedelta(days=7)
        cur_end = day - timedelta(days=1)
        prev_start = day - timedelta(days=14)
        prev_end = day - timedelta(days=8)

        # Get current and previous period data
        cur_data = df[(df['report_date'] >= cur_start) & (df['report_date'] <= cur_end)]
        prev_data = df[(df['report_date'] >= prev_start) & (df['report_date'] <= prev_end)]

        if not cur_data.empty and not prev_data.empty:
            # Calculate current period metrics
            cur_total_tons = cur_data['WGT_NET'].sum()
            cur_fresh_tons = cur_data[cur_data['CANE_TYPE'] == '1']['WGT_NET'].sum()
            cur_fresh_pct = (cur_fresh_tons / cur_total_tons * 100) if cur_total_tons > 0 else 0
            
            # Calculate previous period metrics
            prev_total_tons = prev_data['WGT_NET'].sum()
            prev_fresh_tons = prev_data[prev_data['CANE_TYPE'] == '1']['WGT_NET'].sum()
            prev_fresh_pct = (prev_fresh_tons / prev_total_tons * 100) if prev_total_tons > 0 else 0
            
            # Calculate trends
            if prev_total_tons > 0:
                tons_trend = ((cur_total_tons - prev_total_tons) / prev_total_tons * 100)
            else:
                tons_trend = 0
                
            fresh_trend = cur_fresh_pct - prev_fresh_pct
            
            # Update all rows for this date
            mask = df['report_date'] == date
            df.loc[mask, 'tons_trend_percent'] = tons_trend
            df.loc[mask, 'fresh_trend_percent'] = fresh_trend
    
    return df

def build_training_records(df: pd.DataFrame):
    """Convert raw DataFrame into training records for train_local_ai."""
    records = []
    
    # Group by date to create daily records
    daily_groups = df.groupby('report_date')
    
    for date, group in daily_groups:
        # Calculate daily statistics
        total_tons = group['WGT_NET'].sum()
        fresh_tons = group[group['CANE_TYPE'] == '1']['WGT_NET'].sum()
        fresh_percent = (fresh_tons / total_tons * 100) if total_tons > 0 else 0
        
        # Calculate hourly statistics
        group['hour'] = pd.to_datetime(group['WGT_OUT_DT']).dt.hour
        hourly_tons = group.groupby('hour')['WGT_NET'].sum()
        hours_processed = len(hourly_tons)
        peak_hour_tons = hourly_tons.max() if not hourly_tons.empty else 0
        
        # Get the first row for baseline data
        first_row = group.iloc[0]
        
        stats = {
            'today_total': float(total_tons),
            'avg_daily_tons': float(first_row.get('avg_daily_tons', 0)),
            'type_1_percent': float(fresh_percent),
            'avg_fresh_percent': float(first_row.get('avg_fresh_percent', 0)),
            'has_comparison_data': bool(first_row.get('has_comparison_data', False))
        }
        
        exec_summary = {
            'hours_processed': int(hours_processed),
            'peak_hour_tons': float(peak_hour_tons),
            'latest_volume_time': str(group['WGT_OUT_DT'].max()) if not group.empty else 'N/A',
            'latest_volume_tons': float(group['WGT_NET'].iloc[-1]) if not group.empty else 0.0,
            'peak_hour_time': str(hourly_tons.idxmax()) if not hourly_tons.empty else 'N/A',
            'forecasted_total': float(total_tons)
        }
        
        trend_data = {
            'has_trend_data': True,
            'tons_trend_percent': float(first_row.get('tons_trend_percent', 0)),
            'fresh_trend_percent': float(first_row.get('fresh_trend_percent', 0))
        }
        
        # Derive scores using the same logic used in runtime
        scores = _calculate_scores(stats, exec_summary)

        records.append({
            'date': date,
            'stats': stats,
            'exec_summary': exec_summary,
            'trend_data': trend_data,
            'scores': scores
        })
    
    return records

def main():
    if len(sys.argv) != 3:
        print("Usage: python train_from_db.py YYYY-MM-DD YYYY-MM-DD")
        print("Example: python train_from_db.py 2024-12-01 2025-01-31")
        sys.exit(1)

    start_date = datetime.strptime(sys.argv[1], '%Y-%m-%d').date()
    end_date = datetime.strptime(sys.argv[2], '%Y-%m-%d').date()

    if end_date < start_date:
        print("End date must be >= start date")
        sys.exit(1)

    print(f"Connecting to DB and fetching ALL ROWS AND COLUMNS from {start_date} to {end_date} ...")
    conn = get_connection()
    try:
        raw_data = fetch_daily_aggregates(conn, start_date, end_date)
    finally:
        conn.close()

    if raw_data.empty:
        print("No data returned from DB in this range.")
        sys.exit(0)

    print(f"Processing {len(raw_data)} raw records...")
    processed_data = add_comparison_baselines(raw_data)
    processed_data = add_trend_context(processed_data)

    # Keep only days with valid comparison baselines for better model quality
    filtered = processed_data[processed_data['has_comparison_data'] == True].reset_index(drop=True)
    if filtered.shape[0] < 10:
        print(f"Not enough days with comparison baselines. Found: {filtered.shape[0]} (need >= 10)")
        sys.exit(0)

    print(f"Building training records from {len(filtered['report_date'].unique())} days with {len(filtered)} total records...")
    records = build_training_records(filtered)

    print("Training local AI models...")
    ok = train_local_ai(records)
    print("Training result:", ok)

    if ok:
        status = get_local_ai_status()
        print("Local AI Status:", status)
        print("Models saved under:", status['model_path'])
        # Write metadata and append training history
        try:
            import json, os, time
            os.makedirs(status["model_path"], exist_ok=True)
            meta = {
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "train_range": {"start": str(start_date), "end": str(end_date)},
                "days_used": int(len(filtered['report_date'].unique())),
                "total_records_used": int(len(filtered)),
                "data_source": "ALL ROWS AND COLUMNS from Vdetails_CPC_TRUC",
                "model_path": status["model_path"]
            }
            with open(os.path.join(status["model_path"], "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            hist_path = os.path.join(status["model_path"], "training_history.jsonl")
            with open(hist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            print("Saved metadata and appended training history.")
        except Exception as e:
            print("Warning: failed to write training metadata:", e)
        print("Done.")
    else:
        print("Training failed.")

if __name__ == "__main__":
    main()
