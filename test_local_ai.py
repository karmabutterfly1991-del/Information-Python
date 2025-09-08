#!/usr/bin/env python3
"""
Test Local AI Functionality for Sugar Cane Analysis
==================================================

This script demonstrates how to use the local AI-enhanced analysis system.
"""

import sys
import os
from datetime import datetime, timedelta
import random

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from analysis import generate_analysis, train_local_ai, get_local_ai_status
    from local_ai_config import LocalAISetup
    print("✅ Successfully imported local AI modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)

def generate_sample_data(num_days=30):
    """Generate sample historical data for training"""
    print(f"Generating {num_days} days of sample data...")
    
    historical_data = []
    base_date = datetime.now() - timedelta(days=num_days)
    
    for i in range(num_days):
        date = base_date + timedelta(days=i)
        
        # Generate realistic sugar cane data
        base_quantity = 800 + random.uniform(-100, 100)
        base_quality = 75 + random.uniform(-10, 10)
        
        # Add some seasonal variation
        month = date.month
        if month in [12, 1, 2, 3, 4]:  # Crushing season
            quantity_multiplier = 1.0 + random.uniform(-0.1, 0.2)
            quality_multiplier = 1.0 + random.uniform(-0.05, 0.1)
        else:  # Off season
            quantity_multiplier = 0.1 + random.uniform(0, 0.2)
            quality_multiplier = 0.8 + random.uniform(-0.1, 0.2)
        
        today_total = base_quantity * quantity_multiplier
        type_1_percent = max(0, min(100, base_quality * quality_multiplier))
        
        # Generate executive summary
        hours_processed = random.randint(8, 16)
        peak_hour_tons = today_total / hours_processed * random.uniform(1.2, 2.0)
        
        # Generate trend data
        tons_trend = random.uniform(-15, 15)
        fresh_trend = random.uniform(-8, 8)
        
        # Calculate scores
        quantity_score = 3
        if today_total > 900: quantity_score = 5
        elif today_total > 850: quantity_score = 4
        elif today_total < 700: quantity_score = 1
        elif today_total < 750: quantity_score = 2
        
        quality_score = 3
        if type_1_percent > 80: quality_score = 5
        elif type_1_percent > 75: quality_score = 4
        elif type_1_percent < 65: quality_score = 1
        elif type_1_percent < 70: quality_score = 2
        
        stability_score = 3
        peak_ratio = peak_hour_tons / (today_total / hours_processed) if hours_processed > 0 else 1
        if peak_ratio < 1.3: stability_score = 5
        elif peak_ratio < 1.6: stability_score = 4
        elif peak_ratio > 2.5: stability_score = 1
        elif peak_ratio > 2.0: stability_score = 2
        
        data_point = {
            'date': date,
            'stats': {
                'today_total': today_total,
                'avg_daily_tons': 850,
                'type_1_percent': type_1_percent,
                'avg_fresh_percent': 75,
                'has_comparison_data': True
            },
            'exec_summary': {
                'hours_processed': hours_processed,
                'peak_hour_tons': peak_hour_tons,
                'latest_volume_time': f"{random.randint(6, 18):02d}:00",
                'latest_volume_tons': today_total * 0.8,
                'peak_hour_time': f"{random.randint(8, 16):02d}:00",
                'forecasted_total': today_total * 1.1
            },
            'trend_data': {
                'has_trend_data': True,
                'tons_trend_percent': tons_trend,
                'fresh_trend_percent': fresh_trend
            },
            'scores': {
                'quantity': quantity_score,
                'quality': quality_score,
                'stability': stability_score
            }
        }
        
        historical_data.append(data_point)
    
    print(f"✅ Generated {len(historical_data)} data points")
    return historical_data

def test_local_ai_training():
    """Test local AI training functionality"""
    print("\n🤖 Testing Local AI Training...")
    print("=" * 40)
    
    # Generate sample data
    historical_data = generate_sample_data(20)
    
    # Train local AI
    print("Training local AI models...")
    success = train_local_ai(historical_data)
    
    if success:
        print("✅ Local AI training successful!")
        
        # Check status
        status = get_local_ai_status()
        print(f"AI Status: {status}")
        
        return True
    else:
        print("❌ Local AI training failed")
        return False

def test_local_ai_analysis():
    """Test local AI analysis with realistic data"""
    print("\n🔍 Testing Local AI Analysis...")
    print("=" * 40)
    
    # Use a date within sugar cane season (December-April)
    test_date = datetime(2024, 12, 15)  # December 15, 2024
    
    test_stats = {
        'today_total': 2500,  # Realistic daily production
        'avg_daily_tons': 2300,  # Historical average
        'type_1_percent': 85.5,  # Fresh cane percentage
        'avg_fresh_percent': 82.0,  # Historical fresh cane average
        'has_comparison_data': True,  # Has historical data
        'line_a_total': 1200,
        'line_b_total': 1300,
        'type_1_total': 2137,
        'type_2_total': 363
    }
    
    test_exec_summary = {
        'hours_processed': 12,  # Half day data
        'latest_volume_time': '12:00',
        'latest_volume_tons': 2500,
        'peak_hour_time': '10:00',
        'peak_hour_tons': 250,
        'forecasted_total': 4800  # Forecast for full day
    }
    
    test_trend_data = {
        'has_trend_data': True,
        'tons_trend_percent': 8.2,
        'fresh_trend_percent': 4.0
    }
    
    # Generate analysis
    print("Generating AI-enhanced analysis...")
    result = generate_analysis(
        test_date, 
        test_stats, 
        test_exec_summary, 
        test_trend_data
    )
    
    # Display results
    guru_analysis = result['guru_analysis']
    
    print(f"\n📊 Analysis Results:")
    print(f"Headline: {guru_analysis['headline']['text']}")
    print(f"AI Enhanced: {guru_analysis['ai_enhanced']}")
    print(f"Overall Score: {guru_analysis['scores']['overall_score_display']}")
    
    if guru_analysis.get('ai_insights'):
        print(f"\n🤖 AI Insights:")
        for insight in guru_analysis['ai_insights']:
            print(f"  • {insight}")
    
    if guru_analysis.get('trend_prediction'):
        print(f"\n🔮 Trend Prediction:")
        print(f"  {guru_analysis['trend_prediction']}")
        print(f"  Confidence: {guru_analysis['prediction_confidence']}")
    
    if guru_analysis.get('anomalies'):
        print(f"\n⚠️  Anomalies Detected:")
        for anomaly in guru_analysis['anomalies']:
            print(f"  • {anomaly}")
    
    print(f"\n💡 Recommendation:")
    print(f"  {guru_analysis['recommendation']}")
    
    return result

def test_no_data_analysis():
    """Test AI analysis when there's no data for today"""
    print("\n🔍 Testing AI Analysis with No Data...")
    print("=" * 40)
    
    # Use a date within sugar cane season
    test_date = datetime(2024, 12, 15)
    
    # Test data with no production today
    test_stats = {
        'today_total': 0,  # No data today
        'avg_daily_tons': 2300,  # Historical average
        'type_1_percent': 0,  # No fresh cane data
        'avg_fresh_percent': 82.0,
        'has_comparison_data': True,
        'line_a_total': 0,
        'line_b_total': 0,
        'type_1_total': 0,
        'type_2_total': 0
    }
    
    test_exec_summary = {
        'hours_processed': 0,  # No hours processed
        'latest_volume_time': 'N/A',
        'latest_volume_tons': 0,
        'peak_hour_time': 'N/A',
        'peak_hour_tons': 0,
        'forecasted_total': 0
    }
    
    test_trend_data = {
        'has_trend_data': True,
        'tons_trend_percent': 0,
        'fresh_trend_percent': 0
    }
    
    # Generate analysis
    print("Generating AI analysis with no data...")
    result = generate_analysis(
        test_date, 
        test_stats, 
        test_exec_summary, 
        test_trend_data
    )
    
    # Display results
    guru_analysis = result['guru_analysis']
    
    print(f"\n📊 Analysis Results (No Data):")
    print(f"Headline: {guru_analysis['headline']['text']}")
    print(f"AI Enhanced: {guru_analysis['ai_enhanced']}")
    print(f"Overall Score: {guru_analysis['scores']['overall_score_display']}")
    print(f"Comment: {guru_analysis['comment']}")
    print(f"Recommendation: {guru_analysis['recommendation']}")
    
    # Check if efficiency metrics are empty
    if guru_analysis.get('efficiency_metrics'):
        print(f"\n📈 Efficiency Metrics:")
        for key, value in guru_analysis['efficiency_metrics'].items():
            print(f"  {key}: {value}")
    
    # Check if operational insights are empty
    if guru_analysis.get('operational_insights'):
        print(f"\n💡 Operational Insights:")
        for insight in guru_analysis['operational_insights']:
            print(f"  • {insight}")
    
    return result

def test_local_ai_setup():
    """Test local AI setup and configuration"""
    print("\n⚙️  Testing Local AI Setup...")
    print("=" * 40)
    
    try:
        # Test LocalAISetup
        ai_setup = LocalAISetup()
        ai_setup.ensure_model_directory()
        
        print(f"✅ Model directory created: {ai_setup.model_path}")
        print(f"✅ Min data points: {ai_setup.min_data_points}")
        print(f"✅ Prediction horizon: {ai_setup.prediction_horizon} days")
        
        # Test model loading
        models_loaded = ai_setup.load_models()
        print(f"✅ Models loaded: {models_loaded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

def test_anomaly_detection_scenarios():
    """Test various anomaly detection scenarios with detailed explanations"""
    print("\n🔍 Testing Anomaly Detection Scenarios...")
    print("=" * 50)
    
    test_date = datetime(2024, 12, 15)
    
    # Scenario 1: High volume anomaly
    print("\n📊 Scenario 1: ปริมาณสูงผิดปกติ")
    test_stats_1 = {
        'today_total': 3500,  # สูงกว่าค่าเฉลี่ยมาก
        'avg_daily_tons': 2300,
        'type_1_percent': 85.5,
        'avg_fresh_percent': 82.0,
        'has_comparison_data': True,
        'line_a_total': 1800,
        'line_b_total': 1700,
        'type_1_total': 2992,
        'type_2_total': 508
    }
    
    test_exec_1 = {
        'hours_processed': 12,
        'latest_volume_time': '12:00',
        'latest_volume_tons': 3500,
        'peak_hour_time': '10:00',
        'peak_hour_tons': 400,  # สูงผิดปกติ
        'forecasted_total': 7000
    }
    
    result_1 = generate_analysis(test_date, test_stats_1, test_exec_1, {'has_trend_data': True})
    print(f"Anomalies: {result_1['guru_analysis'].get('anomalies', [])}")
    
    # Scenario 2: Low quality anomaly
    print("\n📊 Scenario 2: คุณภาพต่ำผิดปกติ")
    test_stats_2 = {
        'today_total': 2300,
        'avg_daily_tons': 2300,
        'type_1_percent': 65.0,  # ต่ำกว่าค่าเฉลี่ยมาก
        'avg_fresh_percent': 82.0,
        'has_comparison_data': True,
        'line_a_total': 1200,
        'line_b_total': 1100,
        'type_1_total': 1495,
        'type_2_total': 805
    }
    
    test_exec_2 = {
        'hours_processed': 12,
        'latest_volume_time': '12:00',
        'latest_volume_tons': 2300,
        'peak_hour_time': '10:00',
        'peak_hour_tons': 250,
        'forecasted_total': 4600
    }
    
    result_2 = generate_analysis(test_date, test_stats_2, test_exec_2, {'has_trend_data': True})
    print(f"Anomalies: {result_2['guru_analysis'].get('anomalies', [])}")
    
    # Scenario 3: Line imbalance anomaly
    print("\n📊 Scenario 3: ความไม่สมดุลของราง")
    test_stats_3 = {
        'today_total': 2300,
        'avg_daily_tons': 2300,
        'type_1_percent': 82.0,
        'avg_fresh_percent': 82.0,
        'has_comparison_data': True,
        'line_a_total': 2000,  # ราง A มากผิดปกติ
        'line_b_total': 300,   # ราง B น้อยผิดปกติ
        'type_1_total': 1886,
        'type_2_total': 414
    }
    
    test_exec_3 = {
        'hours_processed': 12,
        'latest_volume_time': '12:00',
        'latest_volume_tons': 2300,
        'peak_hour_time': '10:00',
        'peak_hour_tons': 250,
        'forecasted_total': 4600
    }
    
    result_3 = generate_analysis(test_date, test_stats_3, test_exec_3, {'has_trend_data': True})
    print(f"Anomalies: {result_3['guru_analysis'].get('anomalies', [])}")
    
    # Scenario 4: High fresh cane ratio anomaly
    print("\n📊 Scenario 4: อ้อยสดสูงผิดปกติ")
    test_stats_4 = {
        'today_total': 2300,
        'avg_daily_tons': 2300,
        'type_1_percent': 98.0,  # อ้อยสดสูงมาก
        'avg_fresh_percent': 82.0,
        'has_comparison_data': True,
        'line_a_total': 1200,
        'line_b_total': 1100,
        'type_1_total': 2254,  # อ้อยสดเกือบทั้งหมด
        'type_2_total': 46     # อ้อยไฟน้อยมาก
    }
    
    test_exec_4 = {
        'hours_processed': 12,
        'latest_volume_time': '12:00',
        'latest_volume_tons': 2300,
        'peak_hour_time': '10:00',
        'peak_hour_tons': 250,
        'forecasted_total': 4600
    }
    
    result_4 = generate_analysis(test_date, test_stats_4, test_exec_4, {'has_trend_data': True})
    print(f"Anomalies: {result_4['guru_analysis'].get('anomalies', [])}")
    
    return [result_1, result_2, result_3, result_4]

def test_anomalies_in_web_data():
    """Test that anomalies are properly included in the analysis data"""
    print("\n🔍 Testing Anomalies in Web Data...")
    print("=" * 40)
    
    test_date = datetime(2024, 12, 15)
    
    # Create data that will definitely trigger anomalies
    test_stats = {
        'today_total': 4000,  # Very high volume
        'avg_daily_tons': 2300,
        'type_1_percent': 95.0,  # Very high quality
        'avg_fresh_percent': 82.0,
        'has_comparison_data': True,
        'line_a_total': 3500,  # Line A imbalance
        'line_b_total': 500,
        'type_1_total': 3800,
        'type_2_total': 200
    }
    
    test_exec_summary = {
        'hours_processed': 8,  # Short hours
        'latest_volume_time': '08:00',
        'latest_volume_tons': 4000,
        'peak_hour_time': '10:00',
        'peak_hour_tons': 600,  # Very high peak
        'forecasted_total': 12000
    }
    
    test_trend_data = {
        'has_trend_data': True,
        'tons_trend_percent': 15.0,
        'fresh_trend_percent': 8.0
    }
    
    # Generate analysis
    print("Generating analysis with guaranteed anomalies...")
    result = generate_analysis(
        test_date, 
        test_stats, 
        test_exec_summary, 
        test_trend_data
    )
    
    # Check anomalies
    guru_analysis = result['guru_analysis']
    anomalies = guru_analysis.get('anomalies', [])
    
    print(f"\n📊 Analysis Results:")
    print(f"Headline: {guru_analysis['headline']['text']}")
    print(f"AI Enhanced: {guru_analysis['ai_enhanced']}")
    print(f"Overall Score: {guru_analysis['scores']['overall_score_display']}")
    
    print(f"\n⚠️  Anomalies Found ({len(anomalies)}):")
    for i, anomaly in enumerate(anomalies, 1):
        print(f"  {i}. {anomaly}")
    
    # Check if anomalies are in the result structure
    print(f"\n🔍 Data Structure Check:")
    print(f"  - Anomalies key exists: {'anomalies' in guru_analysis}")
    print(f"  - Anomalies is list: {isinstance(anomalies, list)}")
    print(f"  - Anomalies length: {len(anomalies)}")
    
    if anomalies:
        print(f"  - First anomaly: {anomalies[0][:100]}...")
    
    return result

def test_web_interface_anomalies():
    """Test if anomalies are properly sent to web interface"""
    print("\n🌐 Testing Web Interface Anomalies...")
    print("=" * 40)
    
    # Import Flask app for testing
    try:
        from app import app
        with app.test_client() as client:
            # Test with a date that should have anomalies
            response = client.get('/get_data?date=2024-12-15')
            
            if response.status_code == 200:
                data = response.get_json()
                analysis = data.get('analysis', {})
                guru_analysis = analysis.get('guru_analysis', {})
                anomalies = guru_analysis.get('anomalies', [])
                
                print(f"✅ Web interface test successful!")
                print(f"📊 Anomalies found: {len(anomalies)}")
                for i, anomaly in enumerate(anomalies, 1):
                    print(f"  {i}. {anomaly[:100]}...")
                
                return len(anomalies) > 0
            else:
                print(f"❌ Web interface test failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Web interface test error: {e}")
        return False

def main():
    """Main test function"""
    print("🤖 Local AI Sugar Cane Analysis Test")
    print("=" * 50)
    
    # Test setup
    setup_ok = test_local_ai_setup()
    if not setup_ok:
        print("❌ Setup failed, exiting...")
        return
    
    # Test training
    training_ok = test_local_ai_training()
    if not training_ok:
        print("❌ Training failed, but continuing with analysis test...")
    
    # Test anomalies in web data
    try:
        result = test_anomalies_in_web_data()
        print("\n✅ Anomalies test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Anomalies test failed: {e}")
    
    # Test web interface
    try:
        web_ok = test_web_interface_anomalies()
        if web_ok:
            print("\n✅ Web interface anomalies test completed successfully!")
        else:
            print("\n⚠️  Web interface anomalies test - no anomalies found")
        
    except Exception as e:
        print(f"\n❌ Web interface test failed: {e}")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
