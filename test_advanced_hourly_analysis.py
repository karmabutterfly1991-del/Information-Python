"""
Test script for Advanced Hourly Analysis
ทดสอบการทำงานของโมดูลการวิเคราะห์ข้อมูลรายชั่วโมงแบบ Advanced
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_hourly_analysis import AdvancedHourlyAnalyzer, analyze_hourly_data_advanced, predict_hourly_performance
import json
from datetime import datetime, timedelta

def create_sample_hourly_data():
    """สร้างข้อมูลตัวอย่างสำหรับการทดสอบ"""
    sample_data = []
    
    # สร้างข้อมูล 24 ชั่วโมง
    for i in range(24):
        # สร้างข้อมูลที่สมจริง
        base_tons = 50 + (i * 2)  # เพิ่มขึ้นตามชั่วโมง
        if 6 <= i <= 18:  # ช่วงเวลาทำงาน
            base_tons += 30
        
        # เพิ่มความแปรปรวน
        import random
        variation = random.uniform(-10, 10)
        total_tons = max(0, base_tons + variation)
        
        # แบ่งเป็น Line A และ B
        a_ratio = 0.6 + random.uniform(-0.2, 0.2)
        a_tons = total_tons * a_ratio
        b_tons = total_tons * (1 - a_ratio)
        
        # แบ่งเป็น Fresh และ Burnt
        fresh_ratio = 0.7 + random.uniform(-0.1, 0.1)
        fresh_tons = total_tons * fresh_ratio
        burnt_tons = total_tons * (1 - fresh_ratio)
        
        # คำนวณจำนวนรถ
        avg_tons_per_truck = 15
        total_count = max(1, int(total_tons / avg_tons_per_truck))
        a_count = int(total_count * a_ratio)
        b_count = total_count - a_count
        
        sample_data.append({
            'Time': f"{i:02d}:00-{i+1:02d}:00",
            'A_Count': a_count,
            'A_Tons': round(a_tons, 2),
            'B_Count': b_count,
            'B_Tons': round(b_tons, 2),
            'Total_Count': total_count,
            'Total_Tons': round(total_tons, 2),
            'Fresh_Tons': round(fresh_tons, 2),
            'Burnt_Tons': round(burnt_tons, 2),
            'hour': i,
            'time_label': f"{i:02d}:00-{i+1:02d}:00"
        })
    
    return sample_data

def test_advanced_analyzer():
    """ทดสอบ AdvancedHourlyAnalyzer"""
    print("=" * 60)
    print("ทดสอบ AdvancedHourlyAnalyzer")
    print("=" * 60)
    
    # สร้างข้อมูลตัวอย่าง
    sample_data = create_sample_hourly_data()
    
    # สร้าง analyzer
    analyzer = AdvancedHourlyAnalyzer()
    
    # ทดสอบการวิเคราะห์ประสิทธิภาพ
    print("\n1. ทดสอบการวิเคราะห์ประสิทธิภาพ...")
    performance_result = analyzer.analyze_hourly_performance(sample_data)
    
    print(f"   - จำนวนข้อมูล: {len(sample_data)} ชั่วโมง")
    print(f"   - ดัชนีประสิทธิภาพรวม: {performance_result['overall_performance_index']['overall_index']:.1f}")
    print(f"   - ระดับประสิทธิภาพ: {performance_result['overall_performance_index']['performance_level']}")
    
    # ทดสอบการสร้างข้อมูลเชิงลึก
    print("\n2. ทดสอบการสร้างข้อมูลเชิงลึก...")
    insights = analyzer.generate_hourly_insights(performance_result)
    print(f"   - จำนวนข้อมูลเชิงลึก: {len(insights)}")
    for i, insight in enumerate(insights[:3], 1):  # แสดง 3 ข้อแรก
        print(f"   {i}. {insight}")
    
    # ทดสอบการสร้างคำแนะนำ
    print("\n3. ทดสอบการสร้างคำแนะนำ...")
    recommendations = analyzer.generate_hourly_recommendations(performance_result)
    print(f"   - จำนวนคำแนะนำ: {len(recommendations)}")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['title']} ({rec['priority']})")
    
    # ทดสอบการพยากรณ์
    print("\n4. ทดสอบการพยากรณ์...")
    prediction = analyzer.predict_next_hour_performance(sample_data, 15)
    if 'error' not in prediction:
        print(f"   - พยากรณ์ชั่วโมงถัดไป: {prediction['predicted_tons']:.1f} ตัน")
        print(f"   - ระดับความเชื่อมั่น: {prediction['confidence_level']}")
    else:
        print(f"   - ข้อผิดพลาด: {prediction['error']}")
    
    return performance_result

def test_main_functions():
    """ทดสอบฟังก์ชันหลัก"""
    print("\n" + "=" * 60)
    print("ทดสอบฟังก์ชันหลัก")
    print("=" * 60)
    
    # สร้างข้อมูลตัวอย่าง
    sample_data = create_sample_hourly_data()
    
    # ทดสอบ analyze_hourly_data_advanced
    print("\n1. ทดสอบ analyze_hourly_data_advanced...")
    result = analyze_hourly_data_advanced(sample_data)
    
    print(f"   - สถานะ: {'สำเร็จ' if 'error' not in result else 'ล้มเหลว'}")
    if 'error' not in result:
        print(f"   - จำนวนข้อมูลเชิงลึก: {len(result['insights'])}")
        print(f"   - จำนวนคำแนะนำ: {len(result['recommendations'])}")
        print(f"   - เวลาสร้าง: {result['generated_at']}")
    
    # ทดสอบ predict_hourly_performance
    print("\n2. ทดสอบ predict_hourly_performance...")
    prediction = predict_hourly_performance(sample_data, 20)
    
    if 'error' not in prediction:
        print(f"   - พยากรณ์ชั่วโมงที่ 21: {prediction['predicted_tons']:.1f} ตัน")
        print(f"   - ช่วงความเชื่อมั่น: {prediction['confidence_interval']['lower']:.1f} - {prediction['confidence_interval']['upper']:.1f}")
    else:
        print(f"   - ข้อผิดพลาด: {prediction['error']}")
    
    return result

def test_performance_metrics():
    """ทดสอบการคำนวณเมตริกประสิทธิภาพ"""
    print("\n" + "=" * 60)
    print("ทดสอบการคำนวณเมตริกประสิทธิภาพ")
    print("=" * 60)
    
    # สร้างข้อมูลตัวอย่าง
    sample_data = create_sample_hourly_data()
    
    # สร้าง analyzer
    analyzer = AdvancedHourlyAnalyzer()
    
    # ทดสอบการคำนวณเมตริกต่างๆ
    import pandas as pd
    df = pd.DataFrame(sample_data)
    
    print("\n1. ทดสอบการคำนวณเมตริกประสิทธิภาพ...")
    perf_metrics = analyzer._calculate_performance_metrics(df)
    print(f"   - น้ำหนักรวม: {perf_metrics['total_tons']:.1f} ตัน")
    print(f"   - จำนวนรถรวม: {perf_metrics['total_trucks']} คัน")
    print(f"   - ชั่วโมงที่ทำงาน: {perf_metrics['active_hours']}/24")
    print(f"   - อัตราประสิทธิภาพ: {perf_metrics['efficiency_ratio']:.1%}")
    
    print("\n2. ทดสอบการวิเคราะห์แนวโน้ม...")
    trend_analysis = analyzer._analyze_hourly_trends(df)
    if 'error' not in trend_analysis:
        print(f"   - ทิศทางแนวโน้ม: {trend_analysis['trend_direction']}")
        print(f"   - ความแรง: {trend_analysis['trend_strength']}")
        print(f"   - การเปลี่ยนแปลง: {trend_analysis['change_percentage']:.1f}%")
    
    print("\n3. ทดสอบการวิเคราะห์ความแปรปรวน...")
    variability = analyzer._analyze_variability(df)
    if 'error' not in variability:
        print(f"   - ระดับความเสถียร: {variability['stability_level']}")
        print(f"   - ค่าสัมประสิทธิ์การแปรผัน: {variability['coefficient_of_variation']:.1f}%")
        print(f"   - ค่าผิดปกติ: {variability['outliers_count']} ชั่วโมง")
    
    print("\n4. ทดสอบการวิเคราะห์ความสัมพันธ์...")
    correlation = analyzer._analyze_hourly_correlations(df)
    if 'error' not in correlation:
        print(f"   - ความสัมพันธ์ A-B: {correlation['ab_correlation']:.3f}")
        print(f"   - การประเมินความสมดุล: {correlation['balance_assessment']}")
        print(f"   - อัตราส่วน A:B: {correlation['balance_ratio']:.2f}:1")

def display_sample_results():
    """แสดงผลลัพธ์ตัวอย่าง"""
    print("\n" + "=" * 60)
    print("ผลลัพธ์ตัวอย่าง")
    print("=" * 60)
    
    # สร้างข้อมูลตัวอย่าง
    sample_data = create_sample_hourly_data()
    
    # วิเคราะห์ข้อมูล
    result = analyze_hourly_data_advanced(sample_data)
    
    if 'error' not in result:
        analysis = result['analysis']
        
        print("\n📊 สรุปผลการวิเคราะห์:")
        print(f"   • ดัชนีประสิทธิภาพรวม: {analysis['overall_performance_index']['overall_index']:.1f}/100")
        print(f"   • ระดับประสิทธิภาพ: {analysis['overall_performance_index']['performance_level']}")
        print(f"   • น้ำหนักรวม: {analysis['performance_metrics']['total_tons']:.1f} ตัน")
        print(f"   • อัตราประสิทธิภาพ: {analysis['performance_metrics']['efficiency_ratio']:.1%}")
        
        print("\n💡 ข้อมูลเชิงลึก:")
        for i, insight in enumerate(result['insights'], 1):
            print(f"   {i}. {insight}")
        
        print("\n🎯 คำแนะนำ:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"   {i}. {rec['title']} ({rec['priority']})")
            print(f"      {rec['description']}")
    
    else:
        print(f"❌ เกิดข้อผิดพลาด: {result['error']}")

def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ"""
    print("🚀 เริ่มทดสอบ Advanced Hourly Analysis Module")
    print("=" * 60)
    
    try:
        # ทดสอบ AdvancedHourlyAnalyzer
        test_advanced_analyzer()
        
        # ทดสอบฟังก์ชันหลัก
        test_main_functions()
        
        # ทดสอบการคำนวณเมตริกประสิทธิภาพ
        test_performance_metrics()
        
        # แสดงผลลัพธ์ตัวอย่าง
        display_sample_results()
        
        print("\n" + "=" * 60)
        print("✅ การทดสอบเสร็จสิ้น - ทุกฟังก์ชันทำงานได้ปกติ")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาดในการทดสอบ: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
