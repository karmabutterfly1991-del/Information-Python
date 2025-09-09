import os
import pyodbc
import configparser
import pandas as pd
from datetime import datetime, timedelta, time
from flask import Flask, render_template, request, jsonify
import traceback

# Import the new analysis function
# Make sure you have the analysis.py file in the same directory
from analysis import generate_analysis
from advanced_hourly_analysis import analyze_hourly_data_advanced, predict_hourly_performance

# Weather Impact Model removed

# --- Database Configuration ---
def get_connection():
    """Creates and returns a database connection."""
    config = configparser.ConfigParser()
    # Assuming config.ini is in the same directory as the script
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.ini file not found. Please ensure it is in the same folder.")
    
    config.read(config_path, encoding='utf-8')
    db_config = config['DATABASE']

    required_fields = ['SERVER', 'DATABASE', 'UID']
    if not all(field in db_config for field in required_fields):
        raise ValueError("Missing required settings (SERVER, DATABASE, UID) in config.ini")

    conn_str = (
        f"DRIVER={db_config.get('DRIVER', '{ODBC Driver 17 for SQL Server}')};"
        f"SERVER={db_config['SERVER']};DATABASE={db_config['DATABASE']};UID={db_config['UID']};"
        f"PWD={db_config.get('PWD', '')};TrustServerCertificate=yes;"
    )
    timeout = db_config.getint('TIMEOUT', 10)
    
    try:
        return pyodbc.connect(conn_str, timeout=timeout)
    except pyodbc.Error as e:
        print(f"Database connection error: {e}")
        raise

# --- Flask Application ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sugar_cane_monitoring_2025'

def get_season_statistics(conn):
    """Fetches season-wide cumulative statistics with a single query."""
    query = """
    SELECT 
        (SELECT [SUMcanetypesum] FROM [dbPayment].[dbo].[VsumcanetypeSum]) AS total_cane_all,
        SUM(CASE WHEN Prod_line = 'A' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) AS total_line_a,
        SUM(CASE WHEN Prod_line = 'B' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) AS total_line_b,
        SUM(CASE WHEN cane_type = '1' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) AS total_type_1,
        SUM(CASE WHEN cane_type = '2' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) AS total_type_2,
        SUM(CASE WHEN Prod_line = 'A' THEN 1 ELSE 0 END) AS total_trucks_a,
        SUM(CASE WHEN Prod_line = 'B' THEN 1 ELSE 0 END) AS total_trucks_b,
        COUNT(*) as total_trucks_all
    FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC] 
    WHERE print_q = '5' 
        AND (cane_type = '1' OR cane_type = '2')
        AND PRINT_W = 'y' 
        AND WGT_NET > 0 
        AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')
    """
    try:
        df = pd.read_sql(query, conn)
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as e:
        print(f"Error getting season statistics: {e}")
        return {}

def get_historical_comparison_stats(conn, report_date):
    """Fetches statistics from the past 7 days for comparison."""
    end_date = report_date - timedelta(days=1)
    start_date = end_date - timedelta(days=6) # 7-day period
    query = """
        SELECT
            SUM(CAST(wgt_net AS DECIMAL(18,2))) AS total_tons,
            SUM(CASE WHEN cane_type = '1' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) AS total_fresh,
            COUNT(DISTINCT reportdate) AS days_with_data
        FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC]
        WHERE print_q = '5' 
            AND reportdate BETWEEN ? AND ?
            AND (cane_type = '1' OR cane_type = '2')
            AND PRINT_W = 'y' 
            AND WGT_NET > 0 
            AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81');
    """
    try:
        df = pd.read_sql(query, conn, params=(start_date, end_date))
        if df.empty or df.iloc[0]['days_with_data'] == 0:
            return {'avg_daily_tons': 0, 'avg_fresh_percent': 0, 'has_comparison_data': False}

        stats = df.iloc[0]
        avg_daily_tons = stats['total_tons'] / stats['days_with_data']
        avg_fresh_percent = (stats['total_fresh'] / stats['total_tons']) * 100 if stats['total_tons'] > 0 else 0
        return {'avg_daily_tons': avg_daily_tons, 'avg_fresh_percent': avg_fresh_percent, 'has_comparison_data': True}
    except Exception as e:
        print(f"Error fetching comparison data: {e}")
        return {'avg_daily_tons': 0, 'avg_fresh_percent': 0, 'has_comparison_data': False}

def get_hourly_stats(conn, report_date, start_time, end_time):
    """
    Helper function to fetch stats for a single hour, matching VB.NET logic.
    """
    query = """
        SELECT
            ISNULL(SUM(CASE WHEN Prod_line = 'A' THEN 1 ELSE 0 END), 0) AS A_Count,
            ISNULL(SUM(CASE WHEN Prod_line = 'A' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END), 0) AS A_Tons,
            ISNULL(SUM(CASE WHEN Prod_line = 'B' THEN 1 ELSE 0 END), 0) AS B_Count,
            ISNULL(SUM(CASE WHEN Prod_line = 'B' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END), 0) AS B_Tons,
            ISNULL(SUM(CASE WHEN cane_type = '1' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END), 0) AS Fresh_Tons,
            ISNULL(SUM(CASE WHEN cane_type = '2' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END), 0) AS Burnt_Tons
        FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC]
        WHERE 
            print_q = '5' 
            AND reportdate = ?
            AND CAST(WGT_OUT_DT AS TIME) BETWEEN ? AND ?
            AND (cane_type = '1' OR cane_type = '2')
            AND PRINT_W = 'y' 
            AND WGT_NET > 0 
            AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')
    """
    df = pd.read_sql(query, conn, params=(report_date, start_time, end_time))
    
    if not df.empty:
        stats = df.iloc[0].to_dict()
        stats['Total_Count'] = stats['A_Count'] + stats['B_Count']
        stats['Total_Tons'] = stats['A_Tons'] + stats['B_Tons']
        return stats
    else:
        return {'A_Count': 0, 'A_Tons': 0, 'B_Count': 0, 'B_Tons': 0, 'Fresh_Tons': 0, 'Burnt_Tons': 0, 'Total_Count': 0, 'Total_Tons': 0}

def get_trend_analysis_stats(conn, report_date):
    """Fetches stats for the last 14 days to calculate a 7-day vs 7-day trend."""
    
    current_end = report_date - timedelta(days=1)
    current_start = report_date - timedelta(days=7)
    
    previous_end = report_date - timedelta(days=8)
    previous_start = report_date - timedelta(days=14)

    # --- CORRECTED SQL QUERY USING A CTE ---
    # A CTE simplifies the logic for the database engine.
    # 1. We first create a temporary result set called 'PeriodData' where we assign a 'period' to each relevant row.
    # 2. We then run the final aggregation against that simplified result set.
    query = """
        WITH PeriodData AS (
            SELECT
                reportdate,
                wgt_net,
                cane_type,
                (CASE 
                    WHEN reportdate BETWEEN ? AND ? THEN 'current'
                    WHEN reportdate BETWEEN ? AND ? THEN 'previous'
                END) as period
            FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC]
            WHERE print_q = '5' 
                AND (reportdate BETWEEN ? AND ? OR reportdate BETWEEN ? AND ?)
                AND (cane_type = '1' OR cane_type = '2')
                AND PRINT_W = 'y' 
                AND WGT_NET > 0 
                AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')
        )
        SELECT 
            period,
            SUM(CAST(wgt_net AS DECIMAL(18,2))) AS total_tons,
            SUM(CASE WHEN cane_type = '1' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) AS total_fresh,
            COUNT(DISTINCT reportdate) AS days_with_data
        FROM PeriodData
        WHERE period IS NOT NULL
        GROUP BY period;
    """
    # Parameters now match the simplified query structure
    params = (
        current_start, current_end, previous_start, previous_end,
        current_start, current_end, previous_start, previous_end
    )

    try:
        df = pd.read_sql(query, conn, params=params)
        if df.shape[0] < 2:
            return {'has_trend_data': False}

        # Set 'period' as the index for easy row lookup
        df.set_index('period', inplace=True)

        current_stats = df.loc['current']
        previous_stats = df.loc['previous']

        avg_tons_current = current_stats['total_tons'] / current_stats['days_with_data'] if current_stats['days_with_data'] > 0 else 0
        avg_tons_previous = previous_stats['total_tons'] / previous_stats['days_with_data'] if previous_stats['days_with_data'] > 0 else 0
        
        avg_fresh_current = (current_stats['total_fresh'] / current_stats['total_tons'] * 100) if current_stats['total_tons'] > 0 else 0
        avg_fresh_previous = (previous_stats['total_fresh'] / previous_stats['total_tons'] * 100) if previous_stats['total_tons'] > 0 else 0
        
        tons_trend = ((avg_tons_current - avg_tons_previous) / avg_tons_previous * 100) if avg_tons_previous > 0 else 0
        fresh_trend = avg_fresh_current - avg_fresh_previous

        return {
            'has_trend_data': True,
            'tons_trend_percent': tons_trend,
            'fresh_trend_percent': fresh_trend
        }
    except Exception as e:
        print(f"Error getting trend data: {e}")
        return {'has_trend_data': False}
    """Fetches stats for the last 14 days to calculate a 7-day vs 7-day trend."""
    
    current_end = report_date - timedelta(days=1)
    current_start = report_date - timedelta(days=7)
    
    previous_end = report_date - timedelta(days=8)
    previous_start = report_date - timedelta(days=14)

    query = """
        SELECT 
            (CASE 
                WHEN reportdate BETWEEN ? AND ? THEN 'current'
                WHEN reportdate BETWEEN ? AND ? THEN 'previous'
            END) as period,
            SUM(CAST(wgt_net AS DECIMAL(18,2))) AS total_tons,
            SUM(CASE WHEN cane_type = '1' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) AS total_fresh,
            COUNT(DISTINCT reportdate) AS days_with_data
        FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC]
        WHERE print_q = '5' 
            AND (reportdate BETWEEN ? AND ? OR reportdate BETWEEN ? AND ?)
            AND (cane_type = '1' OR cane_type = '2')
            AND PRINT_W = 'y' 
            AND WGT_NET > 0 
            AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')
        GROUP BY (CASE 
                    WHEN reportdate BETWEEN ? AND ? THEN 'current'
                    WHEN reportdate BETWEEN ? AND ? THEN 'previous'
                END)
    """
    params = (
        current_start, current_end, previous_start, previous_end,
        current_start, current_end, previous_start, previous_end,
        current_start, current_end, previous_start, previous_end
    )

    try:
        df = pd.read_sql(query, conn, params=params)
        if df.shape[0] < 2:
            return {'has_trend_data': False}

        current_stats = df[df['period'] == 'current'].iloc[0]
        previous_stats = df[df['period'] == 'previous'].iloc[0]

        avg_tons_current = current_stats['total_tons'] / current_stats['days_with_data'] if current_stats['days_with_data'] > 0 else 0
        avg_tons_previous = previous_stats['total_tons'] / previous_stats['days_with_data'] if previous_stats['days_with_data'] > 0 else 0
        
        avg_fresh_current = (current_stats['total_fresh'] / current_stats['total_tons'] * 100) if current_stats['total_tons'] > 0 else 0
        avg_fresh_previous = (previous_stats['total_fresh'] / previous_stats['total_tons'] * 100) if previous_stats['total_tons'] > 0 else 0
        
        tons_trend = ((avg_tons_current - avg_tons_previous) / avg_tons_previous * 100) if avg_tons_previous > 0 else 0
        fresh_trend = avg_fresh_current - avg_fresh_previous

        return {
            'has_trend_data': True,
            'tons_trend_percent': tons_trend,
            'fresh_trend_percent': fresh_trend
        }
    except Exception as e:
        print(f"Error getting trend data: {e}")
        return {'has_trend_data': False}

def get_historical_performance_index(conn, report_date, hours_processed, current_total_tons):
    """Calculates a performance index by comparing current performance to historical averages for the same timeframe."""
    if hours_processed == 0:
        return None

    end_date = report_date - timedelta(days=1)
    start_date = report_date - timedelta(days=14)
    
    try:
        simple_query = "SELECT SUM(CAST(wgt_net as DECIMAL(18,2)))/COUNT(DISTINCT reportdate) FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC] WHERE print_q='5' AND reportdate BETWEEN ? AND ? AND (cane_type = '1' OR cane_type = '2') AND PRINT_W = 'y' AND WGT_NET > 0 AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')"
        df_avg_daily = pd.read_sql(simple_query, conn, params=(start_date, end_date))
        
        if df_avg_daily.empty or df_avg_daily.iloc[0,0] is None:
             return None

        historical_avg_daily_tons = df_avg_daily.iloc[0,0]
        historical_avg_hourly_tons = historical_avg_daily_tons / 24
        
        current_avg_hourly_tons = current_total_tons / hours_processed
        
        return current_avg_hourly_tons / historical_avg_hourly_tons if historical_avg_hourly_tons > 0 else None

    except Exception as e:
        print(f"Error calculating performance index: {e}")
        return None

def get_dashboard_data(selected_date):
    """
    Gathers all necessary data for the dashboard, ensuring percentage calculations are balanced and correct.
    """
    try:
        conn = get_connection()
    except Exception as e:
        return {'error': f"Database connection failed: {e}"}, 500

    try:
        mydate = datetime.strptime(selected_date, '%Y-%m-%d')
        report_date_param = mydate.date()
        
        day_start_anchor = (mydate - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        
        hourly_records = []
        for i in range(24):
            start_dt = day_start_anchor + timedelta(hours=i)
            end_dt = day_start_anchor + timedelta(hours=i + 1)
            
            start_time_param = start_dt.time()
            end_time_param = (end_dt - timedelta(seconds=1)).time()

            time_label = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
            
            hourly_stats = get_hourly_stats(conn, report_date_param, start_time_param, end_time_param)
            hourly_stats['Time'] = time_label
            hourly_records.append(hourly_stats)

        df_full = pd.DataFrame(hourly_records)
        summary_stats = df_full.sum(numeric_only=True).to_dict()
        total_tons_today = summary_stats.get('Total_Tons', 0)

        season_stats = get_season_statistics(conn)
        comparison_stats = get_historical_comparison_stats(conn, mydate.date())
        trend_data = get_trend_analysis_stats(conn, mydate.date())
        
        comparison_period_days = 7

        hourly_data_list = df_full.to_dict('records')
        summary_row = {col: f"{summary_stats.get(col, 0):,.1f}" if 'Count' in col else f"{summary_stats.get(col, 0):,.2f}" for col in df_full.columns if col != 'Time'}
        summary_row.update({
            'Time': 'สรุปรวม',
            **summary_stats,
            'is_summary': True
        })
        all_data_for_frontend = hourly_data_list + [summary_row]
        
        valid_hours = [h for h in hourly_data_list if h['Total_Tons'] > 0]
        peak_hour_data = max(valid_hours, key=lambda x: x['Total_Tons']) if valid_hours else {'Time': 'N/A', 'Total_Tons': 0}
        exec_summary_data = {
            'latest_volume_time': valid_hours[-1]['Time'] if valid_hours else "N/A",
            'latest_volume_tons': valid_hours[-1]['Total_Tons'] if valid_hours else 0,
            'peak_hour_data': peak_hour_data,
            'peak_hour_time': peak_hour_data['Time'],
            'peak_hour_tons': peak_hour_data['Total_Tons'],
            'hours_processed': len(valid_hours),
            'forecasted_total': (total_tons_today / len(valid_hours)) * 24 if valid_hours else 0
        }

        perf_index = get_historical_performance_index(conn, mydate.date(), exec_summary_data['hours_processed'], total_tons_today)
        
        # --- CORRECTED LOGIC ---
        # Create a specific total based on the sum of categorized cane types.
        # This ensures the percentages will always add up to 100%.
        total_cane_by_type_today = summary_stats.get('Fresh_Tons', 0) + summary_stats.get('Burnt_Tons', 0)
        # -----------------------

        total_ab_tons_season = season_stats.get('total_line_a', 0) + season_stats.get('total_line_b', 0)
        total_type_tons_season = season_stats.get('total_type_1', 0) + season_stats.get('total_type_2', 0)
        total_ab_trucks_season = season_stats.get('total_trucks_a', 0) + season_stats.get('total_trucks_b', 0)
        
        stats_panel_data = {
            'today_total': total_tons_today,
            'today_truck_count': summary_stats.get('Total_Count', 0),
            'today_type_1': summary_stats.get('Fresh_Tons', 0),
            'today_type_2': summary_stats.get('Burnt_Tons', 0),
            # Use the corrected total for percentage calculation
            'type_1_percent': (summary_stats.get('Fresh_Tons', 0) / total_cane_by_type_today * 100) if total_cane_by_type_today > 0 else 0,
            'type_2_percent': (summary_stats.get('Burnt_Tons', 0) / total_cane_by_type_today * 100) if total_cane_by_type_today > 0 else 0,
            'total_cane_all': season_stats.get('total_cane_all', 0),
            'total_type_1': season_stats.get('total_type_1', 0),
            'total_type_2': season_stats.get('total_type_2', 0),
            'total_trucks_a_season': season_stats.get('total_trucks_a', 0),
            'total_trucks_b_season': season_stats.get('total_trucks_b', 0),
            'total_tons_a_season': season_stats.get('total_line_a', 0),
            'total_tons_b_season': season_stats.get('total_line_b', 0),
            'percent_tons_a_season': (season_stats.get('total_line_a', 0) / total_ab_tons_season * 100) if total_ab_tons_season > 0 else 0,
            'percent_tons_b_season': (season_stats.get('total_line_b', 0) / total_ab_tons_season * 100) if total_ab_tons_season > 0 else 0,
            'type_1_percent_overall': (season_stats.get('total_type_1', 0) / total_type_tons_season * 100) if total_type_tons_season > 0 else 0,
            'type_2_percent_overall': (season_stats.get('total_type_2', 0) / total_type_tons_season * 100) if total_type_tons_season > 0 else 0,
            'percent_trucks_a_season': (season_stats.get('total_trucks_a', 0) / total_ab_trucks_season * 100) if total_ab_trucks_season > 0 else 0,
            'percent_trucks_b_season': (season_stats.get('total_trucks_b', 0) / total_ab_trucks_season * 100) if total_ab_trucks_season > 0 else 0,
            'perf_index': perf_index,
            'comparison_period_days': comparison_period_days, 
            **comparison_stats
        }

        # Weather impact analysis removed
        contextual_data = {}

        # Pass analysis_mode explicitly based on whether the selected date is before today
        analysis_mode = 'historical' if mydate.date() < datetime.now().date() else 'current'
        full_analysis = generate_analysis(mydate, stats_panel_data, exec_summary_data, trend_data=trend_data, comparison_period_days=comparison_period_days, contextual_data=contextual_data, analysis_mode=analysis_mode)

        return {
            'summary': {
                'a1sum': f"{summary_stats.get('A_Count', 0):,.0f}", 
                'b1sum': f"{summary_stats.get('B_Count', 0):,.0f}",
                'a2sum': f"{summary_stats.get('A_Tons', 0):,.2f}", 
                'b2sum': f"{summary_stats.get('B_Tons', 0):,.2f}",
                'daily_avg_a_count': f"{summary_stats.get('A_Count', 0) / 24:,.2f}", 
                'daily_avg_b_count': f"{summary_stats.get('B_Count', 0) / 24:,.2f}",
                'daily_avg_a_tons': f"{summary_stats.get('A_Tons', 0) / 24:,.2f}", 
                'daily_avg_b_tons': f"{summary_stats.get('B_Tons', 0) / 24:,.2f}",
                'total_fresh': f"{summary_stats.get('Fresh_Tons', 0):,.2f}", 
                'total_burnt': f"{summary_stats.get('Burnt_Tons', 0):,.2f}"
            },
            'hourly_data': all_data_for_frontend,
            'cumulative': {
                'times': df_full['Time'].tolist(),
                'a_tons': df_full['A_Tons'].cumsum().tolist(), 'b_tons': df_full['B_Tons'].cumsum().tolist(),
                'fresh': df_full['Fresh_Tons'].cumsum().tolist(), 'burnt': df_full['Burnt_Tons'].cumsum().tolist()
            },
            'statistics': stats_panel_data,
            'analysis': full_analysis,
            # Weather analysis removed
        }, 200
    except Exception as e:
        print(f"Error during data retrieval: {e}")
        traceback.print_exc()
        return {'error': f"Data retrieval failed: {e}"}, 500
    finally:
        if 'conn' in locals() and conn:
            conn.close()
    """
    Gathers all necessary data for the dashboard, ensuring percentage calculations are balanced and correct.
    """
    try:
        conn = get_connection()
    except Exception as e:
        return {'error': f"Database connection failed: {e}"}, 500

    try:
        mydate = datetime.strptime(selected_date, '%Y-%m-%d')
        report_date_param = mydate.date()
        
        day_start_anchor = (mydate - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        
        hourly_records = []
        for i in range(24):
            start_dt = day_start_anchor + timedelta(hours=i)
            end_dt = day_start_anchor + timedelta(hours=i + 1)
            
            start_time_param = start_dt.time()
            end_time_param = (end_dt - timedelta(seconds=1)).time()

            time_label = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
            
            hourly_stats = get_hourly_stats(conn, report_date_param, start_time_param, end_time_param)
            hourly_stats['Time'] = time_label
            hourly_records.append(hourly_stats)

        df_full = pd.DataFrame(hourly_records)
        summary_stats = df_full.sum(numeric_only=True).to_dict()
        total_tons_today = summary_stats.get('Total_Tons', 0)

        season_stats = get_season_statistics(conn)
        comparison_stats = get_historical_comparison_stats(conn, mydate.date())
        trend_data = get_trend_analysis_stats(conn, mydate.date())
        
        comparison_period_days = 7

        hourly_data_list = df_full.to_dict('records')
        summary_row = {col: f"{summary_stats.get(col, 0):,.1f}" if 'Count' in col else f"{summary_stats.get(col, 0):,.2f}" for col in df_full.columns if col != 'Time'}
        summary_row.update({
            'Time': 'สรุปรวม',
            **summary_stats,
            'is_summary': True
        })
        all_data_for_frontend = hourly_data_list + [summary_row]
        
        valid_hours = [h for h in hourly_data_list if h['Total_Tons'] > 0]
        peak_hour_data = max(valid_hours, key=lambda x: x['Total_Tons']) if valid_hours else {'Time': 'N/A', 'Total_Tons': 0}
        exec_summary_data = {
            'latest_volume_time': valid_hours[-1]['Time'] if valid_hours else "N/A",
            'latest_volume_tons': valid_hours[-1]['Total_Tons'] if valid_hours else 0,
            'peak_hour_data': peak_hour_data,
            'peak_hour_time': peak_hour_data['Time'],
            'peak_hour_tons': peak_hour_data['Total_Tons'],
            'hours_processed': len(valid_hours),
            'forecasted_total': (total_tons_today / len(valid_hours)) * 24 if valid_hours else 0
        }

        perf_index = get_historical_performance_index(conn, mydate.date(), exec_summary_data['hours_processed'], total_tons_today)

        # FIX: total_displayed_cane_today is removed as it was redundant.
        # Calculations now correctly use total_tons_today.
        total_ab_tons_season = season_stats.get('total_line_a', 0) + season_stats.get('total_line_b', 0)
        total_type_tons_season = season_stats.get('total_type_1', 0) + season_stats.get('total_type_2', 0)
        total_ab_trucks_season = season_stats.get('total_trucks_a', 0) + season_stats.get('total_trucks_b', 0)
        
        stats_panel_data = {
            'today_total': total_tons_today,
            'today_truck_count': summary_stats.get('Total_Count', 0),
            'today_type_1': summary_stats.get('Fresh_Tons', 0),
            'today_type_2': summary_stats.get('Burnt_Tons', 0),
            # --- CORRECTED CALCULATION ---
            'type_1_percent': (summary_stats.get('Fresh_Tons', 0) / total_tons_today * 100) if total_tons_today > 0 else 0,
            'type_2_percent': (summary_stats.get('Burnt_Tons', 0) / total_tons_today * 100) if total_tons_today > 0 else 0,
            # -----------------------------
            'total_cane_all': season_stats.get('total_cane_all', 0),
            'total_type_1': season_stats.get('total_type_1', 0),
            'total_type_2': season_stats.get('total_type_2', 0),
            'total_trucks_a_season': season_stats.get('total_trucks_a', 0),
            'total_trucks_b_season': season_stats.get('total_trucks_b', 0),
            'total_tons_a_season': season_stats.get('total_line_a', 0),
            'total_tons_b_season': season_stats.get('total_line_b', 0),
            'percent_tons_a_season': (season_stats.get('total_line_a', 0) / total_ab_tons_season * 100) if total_ab_tons_season > 0 else 0,
            'percent_tons_b_season': (season_stats.get('total_line_b', 0) / total_ab_tons_season * 100) if total_ab_tons_season > 0 else 0,
            'type_1_percent_overall': (season_stats.get('total_type_1', 0) / total_type_tons_season * 100) if total_type_tons_season > 0 else 0,
            'type_2_percent_overall': (season_stats.get('total_type_2', 0) / total_type_tons_season * 100) if total_type_tons_season > 0 else 0,
            'percent_trucks_a_season': (season_stats.get('total_trucks_a', 0) / total_ab_trucks_season * 100) if total_ab_trucks_season > 0 else 0,
            'percent_trucks_b_season': (season_stats.get('total_trucks_b', 0) / total_ab_trucks_season * 100) if total_ab_trucks_season > 0 else 0,
            'perf_index': perf_index,
            'comparison_period_days': comparison_period_days, 
            **comparison_stats
        }

        full_analysis = generate_analysis(mydate, stats_panel_data, exec_summary_data, trend_data=trend_data, comparison_period_days=comparison_period_days)

        return {
            'summary': {
                'a1sum': f"{summary_stats.get('A_Count', 0):,.0f}", 
                'b1sum': f"{summary_stats.get('B_Count', 0):,.0f}",
                'a2sum': f"{summary_stats.get('A_Tons', 0):,.2f}", 
                'b2sum': f"{summary_stats.get('B_Tons', 0):,.2f}",
                'daily_avg_a_count': f"{summary_stats.get('A_Count', 0) / 24:,.2f}", 
                'daily_avg_b_count': f"{summary_stats.get('B_Count', 0) / 24:,.2f}",
                'daily_avg_a_tons': f"{summary_stats.get('A_Tons', 0) / 24:,.2f}", 
                'daily_avg_b_tons': f"{summary_stats.get('B_Tons', 0) / 24:,.2f}",
                'total_fresh': f"{summary_stats.get('Fresh_Tons', 0):,.2f}", 
                'total_burnt': f"{summary_stats.get('Burnt_Tons', 0):,.2f}"
            },
            'hourly_data': all_data_for_frontend,
            'cumulative': {
                'times': df_full['Time'].tolist(),
                'a_tons': df_full['A_Tons'].cumsum().tolist(), 'b_tons': df_full['B_Tons'].cumsum().tolist(),
                'fresh': df_full['Fresh_Tons'].cumsum().tolist(), 'burnt': df_full['Burnt_Tons'].cumsum().tolist()
            },
            'statistics': stats_panel_data,
            'analysis': full_analysis
        }, 200
    except Exception as e:
        print(f"Error during data retrieval: {e}")
        traceback.print_exc()
        return {'error': f"Data retrieval failed: {e}"}, 500
    finally:
        if 'conn' in locals() and conn:
            conn.close()
    """
    Gathers all necessary data for the dashboard, ensuring percentage calculations are balanced and correct.
    """
    try:
        conn = get_connection()
    except Exception as e:
        return {'error': f"Database connection failed: {e}"}, 500

    try:
        mydate = datetime.strptime(selected_date, '%Y-%m-%d')
        report_date_param = mydate.date()
        
        day_start_anchor = (mydate - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        
        hourly_records = []
        for i in range(24):
            start_dt = day_start_anchor + timedelta(hours=i)
            end_dt = day_start_anchor + timedelta(hours=i + 1)
            
            start_time_param = start_dt.time()
            end_time_param = (end_dt - timedelta(seconds=1)).time()

            time_label = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
            
            hourly_stats = get_hourly_stats(conn, report_date_param, start_time_param, end_time_param)
            hourly_stats['Time'] = time_label
            hourly_records.append(hourly_stats)

        df_full = pd.DataFrame(hourly_records)
        summary_stats = df_full.sum(numeric_only=True).to_dict()
        total_tons_today = summary_stats.get('Total_Tons', 0)

        season_stats = get_season_statistics(conn)
        comparison_stats = get_historical_comparison_stats(conn, mydate.date())
        trend_data = get_trend_analysis_stats(conn, mydate.date())
        
        # Define the comparison period days here
        comparison_period_days = 7

        hourly_data_list = df_full.to_dict('records')
        summary_row = {col: f"{summary_stats.get(col, 0):,.1f}" if 'Count' in col else f"{summary_stats.get(col, 0):,.2f}" for col in df_full.columns if col != 'Time'}
        summary_row.update({
            'Time': 'สรุปรวม',
            **summary_stats,
            'is_summary': True
        })
        all_data_for_frontend = hourly_data_list + [summary_row]
        
        valid_hours = [h for h in hourly_data_list if h['Total_Tons'] > 0]
        peak_hour_data = max(valid_hours, key=lambda x: x['Total_Tons']) if valid_hours else {'Time': 'N/A', 'Total_Tons': 0}
        exec_summary_data = {
            'latest_volume_time': valid_hours[-1]['Time'] if valid_hours else "N/A",
            'latest_volume_tons': valid_hours[-1]['Total_Tons'] if valid_hours else 0,
            'peak_hour_data': peak_hour_data,
            'peak_hour_time': peak_hour_data['Time'],
            'peak_hour_tons': peak_hour_data['Total_Tons'],
            'hours_processed': len(valid_hours),
            'forecasted_total': (total_tons_today / len(valid_hours)) * 24 if valid_hours else 0
        }

        perf_index = get_historical_performance_index(conn, mydate.date(), exec_summary_data['hours_processed'], total_tons_today)

        total_displayed_cane_today = summary_stats.get('Fresh_Tons', 0) + summary_stats.get('Burnt_Tons', 0)
        total_ab_tons_season = season_stats.get('total_line_a', 0) + season_stats.get('total_line_b', 0)
        total_type_tons_season = season_stats.get('total_type_1', 0) + season_stats.get('total_type_2', 0)
        total_ab_trucks_season = season_stats.get('total_trucks_a', 0) + season_stats.get('total_trucks_b', 0)
        
        stats_panel_data = {
            'today_total': total_tons_today,
            'today_truck_count': summary_stats.get('Total_Count', 0),
            'today_type_1': summary_stats.get('Fresh_Tons', 0),
            'today_type_2': summary_stats.get('Burnt_Tons', 0),
            'type_1_percent': (summary_stats.get('Fresh_Tons', 0) / total_displayed_cane_today * 100) if total_displayed_cane_today > 0 else 0,
            'type_2_percent': (summary_stats.get('Burnt_Tons', 0) / total_displayed_cane_today * 100) if total_displayed_cane_today > 0 else 0,
            'total_cane_all': season_stats.get('total_cane_all', 0),
            'total_type_1': season_stats.get('total_type_1', 0),
            'total_type_2': season_stats.get('total_type_2', 0),
            'total_trucks_a_season': season_stats.get('total_trucks_a', 0),
            'total_trucks_b_season': season_stats.get('total_trucks_b', 0),
            'total_tons_a_season': season_stats.get('total_line_a', 0),
            'total_tons_b_season': season_stats.get('total_line_b', 0),
            'percent_tons_a_season': (season_stats.get('total_line_a', 0) / total_ab_tons_season * 100) if total_ab_tons_season > 0 else 0,
            'percent_tons_b_season': (season_stats.get('total_line_b', 0) / total_ab_tons_season * 100) if total_ab_tons_season > 0 else 0,
            'type_1_percent_overall': (season_stats.get('total_type_1', 0) / total_type_tons_season * 100) if total_type_tons_season > 0 else 0,
            'type_2_percent_overall': (season_stats.get('total_type_2', 0) / total_type_tons_season * 100) if total_type_tons_season > 0 else 0,
            'percent_trucks_a_season': (season_stats.get('total_trucks_a', 0) / total_ab_trucks_season * 100) if total_ab_trucks_season > 0 else 0,
            'percent_trucks_b_season': (season_stats.get('total_trucks_b', 0) / total_ab_trucks_season * 100) if total_ab_trucks_season > 0 else 0,
            'perf_index': perf_index,
            'comparison_period_days': comparison_period_days, 
            **comparison_stats
        }

        full_analysis = generate_analysis(mydate, stats_panel_data, exec_summary_data, trend_data=trend_data, comparison_period_days=comparison_period_days)

        return {
            'summary': {
                'a1sum': f"{summary_stats.get('A_Count', 0):,.0f}", 
                'b1sum': f"{summary_stats.get('B_Count', 0):,.0f}",
                'a2sum': f"{summary_stats.get('A_Tons', 0):,.2f}", 
                'b2sum': f"{summary_stats.get('B_Tons', 0):,.2f}",
                'daily_avg_a_count': f"{summary_stats.get('A_Count', 0) / 24:,.2f}", 
                'daily_avg_b_count': f"{summary_stats.get('B_Count', 0) / 24:,.2f}",
                'daily_avg_a_tons': f"{summary_stats.get('A_Tons', 0) / 24:,.2f}", 
                'daily_avg_b_tons': f"{summary_stats.get('B_Tons', 0) / 24:,.2f}",
                'total_fresh': f"{summary_stats.get('Fresh_Tons', 0):,.2f}", 
                'total_burnt': f"{summary_stats.get('Burnt_Tons', 0):,.2f}"
            },
            'hourly_data': all_data_for_frontend,
            'cumulative': {
                'times': df_full['Time'].tolist(),
                'a_tons': df_full['A_Tons'].cumsum().tolist(), 'b_tons': df_full['B_Tons'].cumsum().tolist(),
                'fresh': df_full['Fresh_Tons'].cumsum().tolist(), 'burnt': df_full['Burnt_Tons'].cumsum().tolist()
            },
            'statistics': stats_panel_data,
            'analysis': full_analysis
        }, 200
    except Exception as e:
        print(f"Error during data retrieval: {e}")
        traceback.print_exc()
        return {'error': f"Data retrieval failed: {e}"}, 500
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# --- Routes ---
@app.route('/health')
def health_check():
    try:
        conn = get_connection()
        conn.cursor().execute("SELECT 1")
        conn.close()
        return jsonify({'status': 'healthy', 'database': 'connected'}), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'database': 'disconnected', 'error': str(e)}), 500

@app.route('/')
def index():
    """Main dashboard route."""
    try:
        # Get current date
        today = datetime.now().date()
        
        # Get system configuration
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
        config.read(config_path, encoding='utf-8')
        
        # Get timeout settings
        session_timeout = config.getint('SYSTEM', 'SESSION_TIMEOUT', fallback=30)
        warning_timeout = config.getint('SYSTEM', 'WARNING_TIMEOUT', fallback=25)
        auto_refresh_interval = config.getint('SYSTEM', 'AUTO_REFRESH_INTERVAL', fallback=60)
        
        # Calculate historical date range (last 30 days)
        historical_start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        historical_end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
        
        return render_template('index.html', 
                             today_date=today.strftime('%Y-%m-%d'),
                             historical_start_date=historical_start_date,
                             historical_end_date=historical_end_date,
                             session_timeout=session_timeout,
                             warning_timeout=warning_timeout,
                             auto_refresh_interval=auto_refresh_interval * 1000)  # Convert to milliseconds
    except Exception as e:
        print(f"Error in index route: {e}")
        traceback.print_exc()
        return "เกิดข้อผิดพลาดในการโหลดหน้าเว็บ", 500

@app.route('/advanced_hourly_analysis')
def advanced_hourly_analysis():
    """Advanced hourly analysis page route."""
    try:
        # Get current date
        today = datetime.now().date()
        
        return render_template('advanced_hourly_analysis.html', 
                             today_date=today.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"Error in advanced hourly analysis route: {e}")
        traceback.print_exc()
        return "เกิดข้อผิดพลาดในการโหลดหน้าเว็บ", 500

@app.route('/get_data', methods=['GET'])
def get_data():
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter is missing'}), 400
    try:
        datetime.strptime(selected_date, '%Y-%m-%d')
        data, status_code = get_dashboard_data(selected_date)
        return jsonify(data), status_code
    except ValueError:
        return jsonify({'error': f'Invalid date format: {selected_date} (must be YYYY-MM-DD)'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/get_historical_data', methods=['GET'])
def get_historical_data():
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    if not start_date_str or not end_date_str:
        return jsonify({'error': 'Please provide start_date and end_date'}), 400

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

    try:
        conn = get_connection()
        query = """
            SELECT
                CONVERT(varchar, reportdate, 23) as report_date,
                SUM(CASE WHEN Prod_line = 'A' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as total_a_tons,
                SUM(CASE WHEN Prod_line = 'B' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as total_b_tons,
                SUM(CAST(wgt_net AS DECIMAL(18,2))) as total_tons
            FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC]
            WHERE print_q = '5' AND reportdate BETWEEN ? AND ?
            GROUP BY reportdate
            ORDER BY reportdate;
        """
        df = pd.read_sql(query, conn, params=(start_date, end_date))
        conn.close()
        
        # Convert to list of dictionaries and ensure numeric values
        historical_data = []
        for _, row in df.iterrows():
            # Debug print to see what we're getting from database
            print(f"Row data: report_date={row['report_date']}, total_tons={row['total_tons']}, type={type(row['total_tons'])}")
            
            # Ensure we have valid numeric values
            total_tons = row['total_tons']
            if total_tons is None or str(total_tons).lower() in ['none', 'null', '']:
                total_tons = 0.0
            else:
                try:
                    total_tons = float(total_tons)
                except (ValueError, TypeError):
                    total_tons = 0.0
            
            total_a_tons = row['total_a_tons']
            if total_a_tons is None or str(total_a_tons).lower() in ['none', 'null', '']:
                total_a_tons = 0.0
            else:
                try:
                    total_a_tons = float(total_a_tons)
                except (ValueError, TypeError):
                    total_a_tons = 0.0
            
            total_b_tons = row['total_b_tons']
            if total_b_tons is None or str(total_b_tons).lower() in ['none', 'null', '']:
                total_b_tons = 0.0
            else:
                try:
                    total_b_tons = float(total_b_tons)
                except (ValueError, TypeError):
                    total_b_tons = 0.0
            
            historical_data.append({
                'report_date': row['report_date'],
                'total_a_tons': total_a_tons,
                'total_b_tons': total_b_tons,
                'total_tons': total_tons
            })
            
            print(f"Processed: report_date={row['report_date']}, total_tons={total_tons}")
        
        return jsonify(historical_data), 200
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return jsonify({'error': f'Could not fetch historical data: {str(e)}'}), 500

# NEW ROUTE FOR HEATMAP DATA
@app.route('/get_heatmap_data', methods=['GET'])
def get_heatmap_data():
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    if not start_date_str or not end_date_str:
        return jsonify({'error': 'Please provide start_date and end_date'}), 400
    
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        conn = get_connection()
        query = """
            SELECT
                CONVERT(varchar, reportdate, 23) as report_date,
                DATEPART(hour, WGT_OUT_DT) as hour_of_day,
                SUM(CAST(wgt_net AS DECIMAL(18,2))) as total_tons
            FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC]
            WHERE print_q = '5' AND reportdate BETWEEN ? AND ?
            GROUP BY reportdate, DATEPART(hour, WGT_OUT_DT)
            ORDER BY report_date, hour_of_day;
        """
        df = pd.read_sql(query, conn, params=(start_date, end_date))
        conn.close()

        if df.empty:
            return jsonify({'x': [], 'y': [], 'z': []}), 200

        # Pivot data for heatmap format
        heatmap_pivot = df.pivot_table(index='report_date', columns='hour_of_day', values='total_tons', fill_value=0)
        
        # Ensure all 24 hours are present
        for hour in range(24):
            if hour not in heatmap_pivot.columns:
                heatmap_pivot[hour] = 0
        heatmap_pivot = heatmap_pivot.reindex(sorted(heatmap_pivot.columns), axis=1)

        # Format for Plotly
        response_data = {
            'y': heatmap_pivot.index.tolist(),
            'x': heatmap_pivot.columns.tolist(),
            'z': heatmap_pivot.values.tolist()
        }
        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error fetching heatmap data: {e}")
        return jsonify({'error': f'Could not fetch heatmap data: {str(e)}'}), 500

# NEW ROUTE FOR MONTHLY DATA
@app.route('/get_monthly_data', methods=['GET'])
def get_monthly_data():
    try:
        conn = get_connection()
        query = """
            SELECT
                FORMAT(reportdate, 'yyyy-MM') as report_month,
                SUM(CAST(wgt_net AS DECIMAL(18,2))) as total_tons,
                SUM(CASE WHEN cane_type = '1' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as fresh_tons,
                SUM(CASE WHEN cane_type = '2' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as burnt_tons,
                SUM(CASE WHEN Prod_line = 'A' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as line_a_tons,
                SUM(CASE WHEN Prod_line = 'B' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as line_b_tons,
                COUNT(*) as total_trucks,
                SUM(CASE WHEN cane_type = '1' THEN 1 ELSE 0 END) as fresh_trucks,
                SUM(CASE WHEN cane_type = '2' THEN 1 ELSE 0 END) as burnt_trucks
            FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC]
            WHERE print_q = '5' 
                AND (cane_type = '1' OR cane_type = '2')
                AND PRINT_W = 'y' 
                AND WGT_NET > 0 
                AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')
            GROUP BY FORMAT(reportdate, 'yyyy-MM')
            ORDER BY report_month;
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return jsonify(df.to_dict(orient='records')), 200
    except Exception as e:
        print(f"Error fetching monthly data: {e}")
        return jsonify({'error': f'Could not fetch monthly data: {str(e)}'}), 500

# NEW ROUTE FOR DETAILED ANALYTICS
@app.route('/get_detailed_analytics', methods=['GET'])
def get_detailed_analytics():
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter is missing'}), 400
    
    try:
        mydate = datetime.strptime(selected_date, '%Y-%m-%d')
        conn = get_connection()
        
        # Get detailed hourly analysis
        report_date_param = mydate.date()
        day_start_anchor = (mydate - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        
        hourly_analysis = []
        for i in range(24):
            start_dt = day_start_anchor + timedelta(hours=i)
            end_dt = day_start_anchor + timedelta(hours=i + 1)
            
            start_time_param = start_dt.time()
            end_time_param = (end_dt - timedelta(seconds=1)).time()
            
            hourly_stats = get_hourly_stats(conn, report_date_param, start_time_param, end_time_param)
            hourly_stats['hour'] = i
            hourly_stats['time_label'] = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
            
            # Calculate efficiency for this hour
            if hourly_stats['Total_Tons'] > 0:
                hourly_stats['efficiency'] = 'high' if hourly_stats['Total_Tons'] > 100 else 'normal' if hourly_stats['Total_Tons'] > 50 else 'low'
            else:
                hourly_stats['efficiency'] = 'none'
            
            hourly_analysis.append(hourly_stats)
        
        # Get season trends
        season_query = """
        SELECT 
            FORMAT(reportdate, 'yyyy-MM') as month,
            SUM(CAST(wgt_net AS DECIMAL(18,2))) as total_tons,
            AVG(CAST(wgt_net AS DECIMAL(18,2))) as avg_daily_tons,
            COUNT(*) as total_records,
            SUM(CASE WHEN cane_type = '1' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as fresh_tons,
            SUM(CASE WHEN cane_type = '2' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as burnt_tons
        FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC] 
        WHERE print_q = '5' 
            AND reportdate >= DATEADD(month, -6, GETDATE())
            AND (cane_type = '1' OR cane_type = '2')
            AND PRINT_W = 'y' 
            AND WGT_NET > 0 
            AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')
        GROUP BY FORMAT(reportdate, 'yyyy-MM')
        ORDER BY month DESC
        """
        
        season_df = pd.read_sql(season_query, conn)
        season_trends = season_df.to_dict('records') if not season_df.empty else []
        
        # Get performance comparison
        comparison_query = """
        SELECT 
            AVG(CAST(wgt_net AS DECIMAL(18,2))) as avg_tons,
            STDEV(CAST(wgt_net AS DECIMAL(18,2))) as std_tons,
            MIN(CAST(wgt_net AS DECIMAL(18,2))) as min_tons,
            MAX(CAST(wgt_net AS DECIMAL(18,2))) as max_tons
        FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC] 
        WHERE print_q = '5' 
            AND reportdate >= DATEADD(day, -30, ?)
            AND (cane_type = '1' OR cane_type = '2')
            AND PRINT_W = 'y' 
            AND WGT_NET > 0 
            AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')
        """
        
        comparison_df = pd.read_sql(comparison_query, conn, params=(report_date_param,))
        performance_stats = comparison_df.iloc[0].to_dict() if not comparison_df.empty else {}
        
        # Get quality trends
        quality_query = """
        SELECT 
            reportdate,
            SUM(CASE WHEN cane_type = '1' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) * 100.0 / 
            SUM(CAST(wgt_net AS DECIMAL(18,2))) as fresh_percentage
        FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC] 
        WHERE print_q = '5' 
            AND reportdate >= DATEADD(day, -14, ?)
            AND (cane_type = '1' OR cane_type = '2')
            AND PRINT_W = 'y' 
            AND WGT_NET > 0 
            AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')
        GROUP BY reportdate
        ORDER BY reportdate
        """
        
        quality_df = pd.read_sql(quality_query, conn, params=(report_date_param,))
        quality_trends = quality_df.to_dict('records') if not quality_df.empty else []
        
        conn.close()
        
        return jsonify({
            'hourly_analysis': hourly_analysis,
            'season_trends': season_trends,
            'performance_stats': performance_stats,
            'quality_trends': quality_trends,
            'analysis_date': selected_date
        }), 200
        
    except Exception as e:
        print(f"Error fetching detailed analytics: {e}")
        return jsonify({'error': f'Could not fetch detailed analytics: {str(e)}'}), 500

# NEW ROUTE FOR EXPORT ANALYSIS REPORT
@app.route('/export_analysis_report', methods=['GET'])
def export_analysis_report():
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter is missing'}), 400
    
    try:
        mydate = datetime.strptime(selected_date, '%Y-%m-%d')
        
        # Get the analysis data
        data, status_code = get_dashboard_data(selected_date)
        if status_code != 200:
            return jsonify({'error': 'Could not generate analysis data'}), 500
        
        # Create report structure
        report = {
            'report_date': selected_date,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_tons': data['statistics']['today_total'],
                'total_trucks': data['statistics']['today_truck_count'],
                'fresh_percentage': data['statistics']['type_1_percent'],
                'burnt_percentage': data['statistics']['type_2_percent']
            },
            'analysis': {
                'status': data['analysis']['guru_analysis']['headline']['text'],
                'score': data['analysis']['guru_analysis']['scores']['overall_score_display'],
                'comment': data['analysis']['guru_analysis']['comment'],
                'recommendation': data['analysis']['guru_analysis']['recommendation']
            },
            'efficiency_metrics': data['analysis']['guru_analysis'].get('efficiency_metrics', {}),
            'operational_insights': data['analysis']['guru_analysis'].get('operational_insights', []),
            'predictions': data['analysis']['guru_analysis'].get('predictive_insights', [])
        }
        
        return jsonify(report), 200
        
    except Exception as e:
        print(f"Error exporting analysis report: {e}")
        return jsonify({'error': f'Could not export analysis report: {str(e)}'}), 500


# Weather impact routes removed

# NEW ROUTE FOR DAILY WEIGHT BAR CHART
@app.route('/get_daily_weight_data', methods=['GET'])
def get_daily_weight_data():
    """Get daily weight data for bar chart display"""
    try:
        # Get date range parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Default to last 30 days if no dates provided
        if not start_date:
            end_date_obj = datetime.now().date()
            start_date_obj = end_date_obj - timedelta(days=29)
        else:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            if not end_date:
                end_date_obj = start_date_obj + timedelta(days=29)
            else:
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        conn = get_connection()
        
        # Query to get daily weight data
        query = """
            SELECT 
                CAST(reportdate AS DATE) as report_date,
                SUM(CAST(wgt_net AS DECIMAL(18,2))) as total_weight,
                COUNT(*) as truck_count,
                SUM(CASE WHEN cane_type = '1' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as fresh_weight,
                SUM(CASE WHEN cane_type = '2' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as burnt_weight,
                SUM(CASE WHEN Prod_line = 'A' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as line_a_weight,
                SUM(CASE WHEN Prod_line = 'B' THEN CAST(wgt_net AS DECIMAL(18,2)) ELSE 0 END) as line_b_weight
            FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC]
            WHERE print_q = '5' 
                AND CAST(reportdate AS DATE) BETWEEN ? AND ?
                AND (cane_type = '1' OR cane_type = '2')
                AND PRINT_W = 'y' 
                AND WGT_NET > 0 
                AND NameLan NOT IN ('10', '20', '30', '31', '40', '41', '50', '80', '81')
            GROUP BY CAST(reportdate AS DATE)
            ORDER BY report_date
        """
        
        df = pd.read_sql(query, conn, params=(start_date_obj, end_date_obj))
        conn.close()
        
        if df.empty:
            return jsonify({
                'message': 'ไม่พบข้อมูลน้ำหนักในช่วงวันที่ที่เลือก',
                'data': [],
                'date_range': {
                    'start_date': start_date_obj.strftime('%Y-%m-%d'),
                    'end_date': end_date_obj.strftime('%Y-%m-%d')
                }
            }), 200
        
        # Convert to list of dictionaries for JSON response
        daily_data = []
        for _, row in df.iterrows():
            daily_data.append({
                'date': row['report_date'].strftime('%Y-%m-%d'),
                'total_weight': float(row['total_weight']) if row['total_weight'] else 0,
                'truck_count': int(row['truck_count']) if row['truck_count'] else 0,
                'fresh_weight': float(row['fresh_weight']) if row['fresh_weight'] else 0,
                'burnt_weight': float(row['burnt_weight']) if row['burnt_weight'] else 0,
                'line_a_weight': float(row['line_a_weight']) if row['line_a_weight'] else 0,
                'line_b_weight': float(row['line_b_weight']) if row['line_b_weight'] else 0
            })
        
        return jsonify({
            'data': daily_data,
            'date_range': {
                'start_date': start_date_obj.strftime('%Y-%m-%d'),
                'end_date': end_date_obj.strftime('%Y-%m-%d')
            },
            'summary': {
                'total_days': len(daily_data),
                'total_weight': sum(item['total_weight'] for item in daily_data),
                'avg_daily_weight': sum(item['total_weight'] for item in daily_data) / len(daily_data) if daily_data else 0,
                'max_daily_weight': max(item['total_weight'] for item in daily_data) if daily_data else 0,
                'min_daily_weight': min(item['total_weight'] for item in daily_data) if daily_data else 0
            }
        }), 200
        
    except Exception as e:
        print(f"Error fetching daily weight data: {e}")
        return jsonify({'error': f'Could not fetch daily weight data: {str(e)}'}), 500

# NEW ROUTE FOR DAILY WEIGHT COMPARISON
@app.route('/get_daily_weight_comparison', methods=['GET'])
def get_daily_weight_comparison():
    """Get daily weight comparison between current and previous periods"""
    try:
        # Get base date parameter
        base_date = request.args.get('date')
        if not base_date:
            base_date = datetime.now().strftime('%Y-%m-%d')
        
        base_date_obj = datetime.strptime(base_date, '%Y-%m-%d').date()
        
        # Calculate comparison periods (current week vs previous week)
        current_start = base_date_obj - timedelta(days=6)
        current_end = base_date_obj
        previous_start = current_start - timedelta(days=7)
        previous_end = current_start - timedelta(days=1)
        
        conn = get_connection()
        
        # Query for comparison data
        query = """
            SELECT 
                CAST(reportdate AS DATE) as report_date,
                SUM(CAST(wgt_net AS DECIMAL(18,2))) as total_weight,
                COUNT(*) as truck_count,
                CASE 
                    WHEN CAST(reportdate AS DATE) BETWEEN ? AND ? THEN 'current'
                    WHEN CAST(reportdate AS DATE) BETWEEN ? AND ? THEN 'previous'
                END as period
            FROM [dbPayment].[dbo].[Vdetails_CPC_TRUC]
            WHERE print_q = '5' 
                AND (CAST(reportdate AS DATE) BETWEEN ? AND ? OR CAST(reportdate AS DATE) BETWEEN ? AND ?)
            GROUP BY CAST(reportdate AS DATE), 
                     CASE 
                         WHEN CAST(reportdate AS DATE) BETWEEN ? AND ? THEN 'current'
                         WHEN CAST(reportdate AS DATE) BETWEEN ? AND ? THEN 'previous'
                     END
            ORDER BY report_date
        """
        
        params = (
            current_start, current_end, previous_start, previous_end,
            current_start, current_end, previous_start, previous_end,
            current_start, current_end, previous_start, previous_end
        )
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return jsonify({
                'message': 'ไม่พบข้อมูลสำหรับการเปรียบเทียบ',
                'comparison': {
                    'current_period': [],
                    'previous_period': [],
                    'summary': {}
                }
            }), 200
        
        # Separate data by period
        current_data = df[df['period'] == 'current'].to_dict('records')
        previous_data = df[df['period'] == 'previous'].to_dict('records')
        
        # Calculate summary statistics
        current_total = sum(item['total_weight'] for item in current_data) if current_data else 0
        previous_total = sum(item['total_weight'] for item in previous_data) if previous_data else 0
        current_avg = current_total / len(current_data) if current_data else 0
        previous_avg = previous_total / len(previous_data) if previous_data else 0
        
        weight_change = ((current_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
        
        return jsonify({
            'comparison': {
                'current_period': [
                    {
                        'date': item['report_date'].strftime('%Y-%m-%d'),
                        'weight': float(item['total_weight']),
                        'truck_count': int(item['truck_count'])
                    } for item in current_data
                ],
                'previous_period': [
                    {
                        'date': item['report_date'].strftime('%Y-%m-%d'),
                        'weight': float(item['total_weight']),
                        'truck_count': int(item['truck_count'])
                    } for item in previous_data
                ],
                'summary': {
                    'current_avg': current_avg,
                    'previous_avg': previous_avg,
                    'weight_change_percent': weight_change,
                    'current_total': current_total,
                    'previous_total': previous_total
                }
            }
        }), 200
        
    except Exception as e:
        print(f"Error fetching daily weight comparison: {e}")
        return jsonify({'error': f'Could not fetch daily weight comparison: {str(e)}'}), 500

# NEW ROUTE FOR ADVANCED HOURLY ANALYSIS
@app.route('/get_advanced_hourly_analysis', methods=['GET'])
def get_advanced_hourly_analysis():
    """Get advanced hourly analysis for a specific date"""
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter is missing'}), 400
    
    try:
        mydate = datetime.strptime(selected_date, '%Y-%m-%d')
        conn = get_connection()
        
        # Get hourly data for the selected date
        report_date_param = mydate.date()
        day_start_anchor = (mydate - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        
        hourly_data = []
        for i in range(24):
            start_dt = day_start_anchor + timedelta(hours=i)
            end_dt = day_start_anchor + timedelta(hours=i + 1)
            
            start_time_param = start_dt.time()
            end_time_param = (end_dt - timedelta(seconds=1)).time()
            
            hourly_stats = get_hourly_stats(conn, report_date_param, start_time_param, end_time_param)
            hourly_stats['hour'] = i
            hourly_stats['time_label'] = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
            hourly_data.append(hourly_stats)
        
        conn.close()
        
        # Perform advanced analysis
        advanced_analysis = analyze_hourly_data_advanced(hourly_data)
        
        return jsonify({
            'date': selected_date,
            'hourly_data': hourly_data,
            'advanced_analysis': advanced_analysis
        }), 200
        
    except Exception as e:
        print(f"Error fetching advanced hourly analysis: {e}")
        return jsonify({'error': f'Could not fetch advanced hourly analysis: {str(e)}'}), 500

# NEW ROUTE FOR HOURLY PERFORMANCE PREDICTION
@app.route('/get_hourly_prediction', methods=['GET'])
def get_hourly_prediction():
    """Get prediction for next hour performance"""
    selected_date = request.args.get('date')
    current_hour = request.args.get('current_hour', type=int)
    
    if not selected_date:
        return jsonify({'error': 'Date parameter is missing'}), 400
    
    if current_hour is None or current_hour < 0 or current_hour > 23:
        return jsonify({'error': 'Invalid current_hour parameter (0-23)'}), 400
    
    try:
        mydate = datetime.strptime(selected_date, '%Y-%m-%d')
        conn = get_connection()
        
        # Get hourly data up to current hour
        report_date_param = mydate.date()
        day_start_anchor = (mydate - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        
        hourly_data = []
        for i in range(current_hour + 1):  # Include current hour
            start_dt = day_start_anchor + timedelta(hours=i)
            end_dt = day_start_anchor + timedelta(hours=i + 1)
            
            start_time_param = start_dt.time()
            end_time_param = (end_dt - timedelta(seconds=1)).time()
            
            hourly_stats = get_hourly_stats(conn, report_date_param, start_time_param, end_time_param)
            hourly_stats['hour'] = i
            hourly_stats['time_label'] = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
            hourly_data.append(hourly_stats)
        
        conn.close()
        
        # Generate prediction
        prediction = predict_hourly_performance(hourly_data, current_hour)
        
        return jsonify({
            'date': selected_date,
            'current_hour': current_hour,
            'prediction': prediction,
            'historical_data': hourly_data
        }), 200
        
    except Exception as e:
        print(f"Error generating hourly prediction: {e}")
        return jsonify({'error': f'Could not generate hourly prediction: {str(e)}'}), 500

# NEW ROUTE FOR HOURLY PERFORMANCE COMPARISON
@app.route('/get_hourly_performance_comparison', methods=['GET'])
def get_hourly_performance_comparison():
    """Compare hourly performance between different dates"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not start_date or not end_date:
        return jsonify({'error': 'Please provide start_date and end_date'}), 400
    
    try:
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        conn = get_connection()
        
        # Get hourly data for the date range
        comparison_data = []
        current_date = start_date_obj
        
        while current_date <= end_date_obj:
            mydate = datetime.combine(current_date, datetime.min.time())
            report_date_param = current_date
            day_start_anchor = (mydate - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
            
            daily_hourly_data = []
            for i in range(24):
                start_dt = day_start_anchor + timedelta(hours=i)
                end_dt = day_start_anchor + timedelta(hours=i + 1)
                
                start_time_param = start_dt.time()
                end_time_param = (end_dt - timedelta(seconds=1)).time()
                
                hourly_stats = get_hourly_stats(conn, report_date_param, start_time_param, end_time_param)
                hourly_stats['hour'] = i
                hourly_stats['time_label'] = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
                daily_hourly_data.append(hourly_stats)
            
            # Perform advanced analysis for this day
            daily_analysis = analyze_hourly_data_advanced(daily_hourly_data)
            
            comparison_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'hourly_data': daily_hourly_data,
                'analysis': daily_analysis
            })
            
            current_date += timedelta(days=1)
        
        conn.close()
        
        # Calculate comparison metrics
        total_days = len(comparison_data)
        if total_days > 0:
            # Calculate averages across all days
            avg_performance_index = sum(
                day['analysis']['analysis']['overall_performance_index']['overall_index'] 
                for day in comparison_data 
                if 'overall_performance_index' in day['analysis']['analysis']
            ) / total_days
            
            avg_efficiency = sum(
                day['analysis']['analysis']['performance_metrics']['efficiency_ratio'] 
                for day in comparison_data 
                if 'performance_metrics' in day['analysis']['analysis']
            ) / total_days
            
            # Find best and worst performing days
            best_day = max(comparison_data, 
                          key=lambda x: x['analysis']['analysis'].get('overall_performance_index', {}).get('overall_index', 0))
            worst_day = min(comparison_data, 
                           key=lambda x: x['analysis']['analysis'].get('overall_performance_index', {}).get('overall_index', 0))
        else:
            avg_performance_index = 0
            avg_efficiency = 0
            best_day = None
            worst_day = None
        
        return jsonify({
            'date_range': {
                'start_date': start_date,
                'end_date': end_date,
                'total_days': total_days
            },
            'comparison_summary': {
                'avg_performance_index': avg_performance_index,
                'avg_efficiency_ratio': avg_efficiency,
                'best_performing_day': {
                    'date': best_day['date'] if best_day else None,
                    'performance_index': best_day['analysis']['analysis'].get('overall_performance_index', {}).get('overall_index', 0) if best_day else 0
                },
                'worst_performing_day': {
                    'date': worst_day['date'] if worst_day else None,
                    'performance_index': worst_day['analysis']['analysis'].get('overall_performance_index', {}).get('overall_index', 0) if worst_day else 0
                }
            },
            'daily_data': comparison_data
        }), 200
        
    except Exception as e:
        print(f"Error fetching hourly performance comparison: {e}")
        return jsonify({'error': f'Could not fetch hourly performance comparison: {str(e)}'}), 500

# NEW ROUTE FOR HOURLY EFFICIENCY REPORT
@app.route('/get_hourly_efficiency_report', methods=['GET'])
def get_hourly_efficiency_report():
    """Generate comprehensive hourly efficiency report"""
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'Date parameter is missing'}), 400
    
    try:
        mydate = datetime.strptime(selected_date, '%Y-%m-%d')
        conn = get_connection()
        
        # Get hourly data
        report_date_param = mydate.date()
        day_start_anchor = (mydate - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        
        hourly_data = []
        for i in range(24):
            start_dt = day_start_anchor + timedelta(hours=i)
            end_dt = day_start_anchor + timedelta(hours=i + 1)
            
            start_time_param = start_dt.time()
            end_time_param = (end_dt - timedelta(seconds=1)).time()
            
            hourly_stats = get_hourly_stats(conn, report_date_param, start_time_param, end_time_param)
            hourly_stats['hour'] = i
            hourly_stats['time_label'] = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
            hourly_data.append(hourly_stats)
        
        conn.close()
        
        # Perform advanced analysis
        advanced_analysis = analyze_hourly_data_advanced(hourly_data)
        
        # Generate efficiency report
        efficiency_report = {
            'report_date': selected_date,
            'generated_at': datetime.now().isoformat(),
            'executive_summary': {
                'total_tons': advanced_analysis['analysis']['performance_metrics']['total_tons'],
                'total_trucks': advanced_analysis['analysis']['performance_metrics']['total_trucks'],
                'active_hours': advanced_analysis['analysis']['performance_metrics']['active_hours'],
                'efficiency_ratio': advanced_analysis['analysis']['performance_metrics']['efficiency_ratio'],
                'overall_performance_index': advanced_analysis['analysis']['overall_performance_index']['overall_index'],
                'performance_level': advanced_analysis['analysis']['overall_performance_index']['performance_level']
            },
            'detailed_analysis': advanced_analysis['analysis'],
            'insights': advanced_analysis['insights'],
            'recommendations': advanced_analysis['recommendations'],
            'hourly_breakdown': hourly_data
        }
        
        return jsonify(efficiency_report), 200
        
    except Exception as e:
        print(f"Error generating hourly efficiency report: {e}")
        return jsonify({'error': f'Could not generate hourly efficiency report: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)