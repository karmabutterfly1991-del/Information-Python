#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for enhanced analysis.py with human-like intelligence
"""

import sys
import os
from datetime import datetime, timedelta
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis import generate_analysis, _get_time_based_context, _generate_human_like_greeting

def test_time_based_context():
    """Test time-based context generation"""
    print("=== Testing Time-Based Context ===")
    
    # Test different times of day
    test_times = [
        (datetime.now().replace(hour=8, minute=0), "Morning"),
        (datetime.now().replace(hour=14, minute=0), "Afternoon"),
        (datetime.now().replace(hour=19, minute=0), "Evening"),
        (datetime.now().replace(hour=23, minute=0), "Night"),
    ]
    
    for test_date, time_label in test_times:
        context = _get_time_based_context(test_date)
        print(f"\n{time_label} Context:")
        print(f"  Time of day: {context['time_of_day']}")
        print(f"  Greeting: {context['time_greeting']}")
        print(f"  Urgency level: {context['urgency_level']}")
        print(f"  Time-specific advice: {context['time_specific_advice']}")

def test_human_like_greeting():
    """Test human-like greeting generation"""
    print("\n=== Testing Human-Like Greeting ===")
    
    test_date = datetime.now()
    context = _get_time_based_context(test_date)
    
    personas = ["CRITICAL", "EXCELLENT", "WEAK_START", "STEADY_PERFORMANCE"]
    
    for persona in personas:
        greeting = _generate_human_like_greeting(context, persona)
        print(f"\n{persona} Persona Greeting:")
        print(f"  {greeting}")

def test_enhanced_analysis():
    """Test enhanced analysis generation"""
    print("\n=== Testing Enhanced Analysis ===")
    
    # Sample data - use a date during sugarcane season (Dec-Apr)
    selected_date = datetime.now().replace(month=1, day=15)  # January 15th
    statistics = {
        'today_total': 1200,
        'avg_daily_tons': 1000,
        'type_1_percent': 85,
        'avg_fresh_percent': 80,
        'has_comparison_data': True
    }
    
    executive_summary = {
        'hours_processed': 12,
        'peak_hour_tons': 150,
        'latest_volume_time': '14:30',
        'latest_volume_tons': 1200,
        'peak_hour_time': '10:00',
        'forecasted_total': 1400
    }
    
    trend_data = {
        'has_trend_data': True,
        'tons_trend_percent': 5.2,
        'fresh_trend_percent': 2.1
    }
    
    try:
        result = generate_analysis(selected_date, statistics, executive_summary, trend_data)
        
        if result and 'guru_analysis' in result:
            analysis = result['guru_analysis']
            
            print(f"\nHeadline: {analysis['headline']['text']}")
            print(f"Comment: {analysis['comment'][:200]}...")
            print(f"Recommendation: {analysis['recommendation'][:200]}...")
            print(f"Score: {analysis['scores']['overall_score_display']}")
            
            # Check for new enhanced features
            if 'human_like_intelligence' in analysis:
                print(f"\n✅ Human-like intelligence enabled")
            
            if 'practical_insights' in analysis:
                print(f"✅ Practical insights: {len(analysis['practical_insights'])} items")
                for insight in analysis['practical_insights'][:2]:
                    print(f"  - {insight}")
            
            if 'actionable_advice' in analysis:
                print(f"✅ Actionable advice: {len(analysis['actionable_advice'])} items")
                for advice in analysis['actionable_advice'][:2]:
                    print(f"  - {advice}")
            
            if 'learning_points' in analysis:
                print(f"✅ Learning points: {len(analysis['learning_points'])} items")
                for point in analysis['learning_points'][:2]:
                    print(f"  - {point}")
            
            if 'time_context' in analysis:
                print(f"✅ Time context: {analysis['time_context']['time_of_day']}")
            
        else:
            print("❌ Analysis generation failed")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def test_different_scenarios():
    """Test different performance scenarios"""
    print("\n=== Testing Different Scenarios ===")
    
    # Use dates during sugarcane season
    base_date = datetime.now().replace(month=1, day=15)  # January 15th
    
    scenarios = [
        {
            'name': 'Critical Situation',
            'stats': {
                'today_total': 400,
                'avg_daily_tons': 1000,
                'type_1_percent': 70,
                'avg_fresh_percent': 80,
                'has_comparison_data': True
            },
            'exec_summary': {
                'hours_processed': 8,
                'peak_hour_tons': 80,
                'latest_volume_time': '15:00',
                'latest_volume_tons': 400,
                'peak_hour_time': '09:00',
                'forecasted_total': 600
            }
        },
        {
            'name': 'Excellent Performance',
            'stats': {
                'today_total': 1400,
                'avg_daily_tons': 1000,
                'type_1_percent': 90,
                'avg_fresh_percent': 80,
                'has_comparison_data': True
            },
            'exec_summary': {
                'hours_processed': 12,
                'peak_hour_tons': 180,
                'latest_volume_time': '14:30',
                'latest_volume_tons': 1400,
                'peak_hour_time': '11:00',
                'forecasted_total': 1600
            }
        }
    ]
    
    trend_data = {
        'has_trend_data': True,
        'tons_trend_percent': 0,
        'fresh_trend_percent': 0
    }
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        try:
            result = generate_analysis(
                base_date, 
                scenario['stats'], 
                scenario['exec_summary'], 
                trend_data
            )
            
            if result and 'guru_analysis' in result:
                analysis = result['guru_analysis']
                print(f"Persona: {analysis['headline']['text']}")
                print(f"Comment preview: {analysis['comment'][:150]}...")
                print(f"Score: {analysis['scores']['overall_score_display']}")
                
                if 'practical_insights' in analysis and analysis['practical_insights']:
                    print(f"Key insight: {analysis['practical_insights'][0]}")
                
            else:
                print("❌ Analysis failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main test function"""
    print("🧠 Testing Enhanced Analysis with Human-Like Intelligence")
    print("=" * 60)
    
    test_time_based_context()
    test_human_like_greeting()
    test_enhanced_analysis()
    test_different_scenarios()
    
    print("\n" + "=" * 60)
    print("✅ Enhanced analysis testing completed!")
    print("\nKey improvements:")
    print("• Human-like greetings and responses")
    print("• Time-aware context and recommendations")
    print("• Emotional intelligence in communication")
    print("• Practical insights and actionable advice")
    print("• Learning points for continuous improvement")
    print("• Adaptive recommendations based on time and performance")

if __name__ == "__main__":
    main()
