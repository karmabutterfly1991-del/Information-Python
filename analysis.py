import math
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import random
import traceback

warnings.filterwarnings('ignore')

# --- Configuration ---
class LocalAIConfig:
    def __init__(self):
        self.model_path = "ai_models/"
        self.scaler_path = "ai_models/"
        self.min_data_points = 10
        
    def ensure_model_directory(self):
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.scaler_path, exist_ok=True)

local_ai_config = LocalAIConfig()
local_ai_config.ensure_model_directory()

def format_num(num, digits=2, locale='th'):
    """
    Enhanced number formatting function with improved readability
    
    Args:
        num: Number to format
        digits: Decimal places (default: 2)
        locale: Locale for formatting (default: 'th' for Thai)
    
    Returns:
        Formatted number string with comma separators
    """
    if not isinstance(num, (int, float)) or num is None:
        return str(num) if num is not None else "ไม่มีข้อมูล"
    
    # Handle special cases
    if num == 0:
        return "0"
    
    # Always use comma-separated format for better readability
    if abs(num) < 0.01 and num != 0:
        # Very small numbers
        return f"{num:.{digits+2}f}"
    else:
        # Regular formatting with comma separators
        return f"{num:,.{digits}f}"

def format_percentage(num, digits=1, show_sign=True):
    """
    Enhanced percentage formatting with proper sign handling
    
    Args:
        num: Percentage value
        digits: Decimal places (default: 1)
        show_sign: Whether to show + sign for positive values
    
    Returns:
        Formatted percentage string
    """
    if not isinstance(num, (int, float)) or num is None:
        return "ไม่มีข้อมูล"
    
    if num == 0:
        return "0%"
    
    # Format with proper sign
    if show_sign and num > 0:
        return f"+{num:.{digits}f}%"
    else:
        return f"{num:.{digits}f}%"

def format_quantity_with_unit(value, unit="ตัน", digits=1):
    """
    Format quantity values with proper units and Thai formatting
    
    Args:
        value: Numeric value
        unit: Unit of measurement (default: "ตัน")
        digits: Decimal places
    
    Returns:
        Formatted quantity string
    """
    if not isinstance(value, (int, float)) or value is None:
        return f"ไม่มีข้อมูล {unit}"
    
    if value == 0:
        return f"0 {unit}"
    
    # Use enhanced formatting for large numbers
    formatted_num = format_num(value, digits, 'th')
    
    # Add unit
    if unit in ["ตัน", "กิโลกรัม", "ลิตร"]:
        return f"{formatted_num} {unit}"
    else:
        return f"{formatted_num} {unit}"

# --- ENHANCED PERSONA DICTIONARY V7 (Ultra Concise) ---
PERSONAS = {
    "CRITICAL": {
        "status": "สถานการณ์น่าเป็นห่วง", 
        "color": "text-danger", 
        "icon": "bi-exclamation-diamond-fill",
        "narrative_template": "สถานการณ์ไม่ดี ปริมาณและคุณภาพต่ำกว่าเกณฑ์",
        "base_recommendation": "ต้องแก้ไขเร่งด่วน เรียกประชุมทีมทันที",
        "efficiency_insights": ["ประสิทธิภาพการรับอ้อยต่ำกว่าเกณฑ์มาก", "การส่งมอบอ้อยอาจติดขัด", "ควรตรวจสอบเครื่องจักรและกำลังคน"],
        "operational_focus": ["การแก้ไขจุดที่งานติดขัด", "การสื่อสารกับทีมขนส่งและทีมไร่", "การตรวจสอบเครื่องจักรเร่งด่วน"],
        "ai_priority": "critical",
        "risk_level": "high",
        "response_time": "immediate",
        "time_awareness": "ช่วงเวลานี้สำคัญมาก ต้องแก้ไขให้เร็วที่สุด"
    },
    "EXCELLENT": {
        "status": "ผลงานยอดเยี่ยม", 
        "color": "text-success", 
        "icon": "bi-trophy-fill",
        "narrative_template": "ผลงานยอดเยี่ยม ทำได้เหนือเป้าหมายทุกมิติ",
        "base_recommendation": "ควรถอดบทเรียนและสรุปปัจจัยความสำเร็จ",
        "efficiency_insights": ["ประสิทธิภาพโดยรวมสูงกว่าเป้าหมายชัดเจน", "การประสานงานระหว่างทีมเป็นไปอย่างราบรื่น", "เครื่องจักรทำงานได้เต็มศักยภาพ"],
        "operational_focus": ["การรักษามาตรฐานระดับสูง", "การถอดบทเรียนความสำเร็จ", "การวางแผนเชิงรุกสำหรับวันถัดไป"],
        "ai_priority": "low",
        "risk_level": "minimal",
        "response_time": "normal",
        "time_awareness": "ช่วงเวลาที่ดีแบบนี้ควรใช้เป็นโอกาสในการปรับปรุง"
    },
    "QUANTITY_PUSH": {
        "status": "เน้นปริมาณ คุณภาพต้องเฝ้าระวัง", 
        "color": "text-warning", 
        "icon": "bi-truck-front-fill",
        "narrative_template": "ปริมาณดี แต่คุณภาพลดลง ต้องเฝ้าระวัง",
        "base_recommendation": "ปริมาณดีแล้ว ควรเน้นควบคุมคุณภาพ",
        "efficiency_insights": ["ปริมาณอ้อยเข้าหีบสูงตามเป้า", "คุณภาพอ้อยสดมีแนวโน้มลดลง", "อาจมีสัดส่วนอ้อยไฟสูงขึ้น"],
        "operational_focus": ["การปรับปรุงคุณภาพอ้อย", "การสื่อสารนโยบายคุณภาพ", "การสร้างสมดุลระหว่างปริมาณและคุณภาพ"],
        "time_awareness": "ช่วงเวลานี้เป็นโอกาสในการปรับปรุงคุณภาพ"
    },
    "QUALITY_FOCUS": {
        "status": "คุณภาพเยี่ยม แต่ปริมาณต้องเร่งขึ้น", 
        "color": "text-primary", 
        "icon": "bi-gem",
        "narrative_template": "คุณภาพยอดเยี่ยม แต่ปริมาณยังไม่ถึงเป้า",
        "base_recommendation": "คุณภาพดี ควรเน้นเพิ่มปริมาณและรักษามาตรฐาน",
        "efficiency_insights": ["คุณภาพอ้อยสดสูงกว่าค่าเฉลี่ย", "ปริมาณอ้อยเข้าหีบต่ำกว่าเป้าหมาย", "การจัดการคุณภาพหน้างานทำได้ดี"],
        "operational_focus": ["การเพิ่มปริมาณอ้อย", "การเพิ่มประสิทธิภาพการขนส่ง", "การรักษามาตรฐานคุณภาพ"],
        "time_awareness": "ช่วงเวลาที่คุณภาพดีควรใช้เป็นโอกาสในการเพิ่มปริมาณ"
    },
    "VOLATILE_PERFORMANCE": {
        "status": "ผลงานดี แต่มีความผันผวน", 
        "color": "text-info", 
        "icon": "bi-activity",
        "narrative_template": "ตัวเลขรวมดี แต่มีความผันผวนสูง",
        "base_recommendation": "ภาพรวมดี ควรทำให้การรับอ้อยสม่ำเสมอขึ้น",
        "efficiency_insights": ["ปริมาณรวมดีแต่การไหลเข้าไม่สม่ำเสมอ", "มีช่วงที่รับอ้อยหนาแน่นกว่าปกติ", "ควรปรับปรุงการจัดการคิวรถ"],
        "operational_focus": ["การสร้างความสม่ำเสมอในการรับอ้อย", "การบริหารจัดการคิวรถ", "การสื่อสารกับทีมขนส่ง"],
        "time_awareness": "ช่วงเวลาที่มีความผันผวนควรหาทางทำให้สม่ำเสมอ"
    },
    "STABLE_RECOVERY": {
        "status": "กลับมามีเสถียรภาพ", 
        "color": "text-success", 
        "icon": "bi-graph-up-arrow",
        "narrative_template": "การดำเนินงานกลับมาปกติและเสถียร",
        "base_recommendation": "ขอชื่นชมทีมงาน ควรรักษาระดับนี้ไว้",
        "efficiency_insights": ["การดำเนินงานกลับมาสู่เกณฑ์มาตรฐาน", "ประสิทธิภาพเริ่มฟื้นตัว", "การแก้ไขปัญหาได้ผล"],
        "operational_focus": ["การรักษาความต่อเนื่อง", "การติดตามข้อมูล", "การป้องกันปัญหาเกิดซ้ำ"],
        "time_awareness": "ช่วงเวลาที่ฟื้นตัวควรใช้เป็นโอกาสในการสร้างความแข็งแกร่ง"
    },
    "CONCERNING_TREND": {
        "status": "แนวโน้มที่ควรจับตา", 
        "color": "text-warning", 
        "icon": "bi-graph-down-arrow",
        "narrative_template": "ผลงานยังอยู่ในเกณฑ์ แต่แนวโน้มเริ่มชะลอ",
        "base_recommendation": "ยังไม่น่ากังวล แต่ควรวิเคราะห์หาสาเหตุ",
        "efficiency_insights": ["แนวโน้มปริมาณหรือคุณภาพเริ่มลดลง", "ประสิทธิภาพเริ่มชะลอตัว", "ควรวิเคราะห์ข้อมูลเชิงลึก"],
        "operational_focus": ["การวิเคราะห์แนวโน้ม", "การค้นหาสาเหตุล่วงหน้า", "การวางแผนป้องกันปัญหา"],
        "time_awareness": "ช่วงเวลานี้เป็นโอกาสในการป้องกันปัญหาใหญ่"
    },
    "STEADY_PERFORMANCE": {
        "status": "การดำเนินงานมีเสถียรภาพ", 
        "color": "text-info", 
        "icon": "bi-check2-circle",
        "narrative_template": "ดำเนินงานราบรื่นตามมาตรฐาน",
        "base_recommendation": "ทุกอย่างตามแผน ควรรักษาระดับนี้ไว้",
        "efficiency_insights": ["การดำเนินงานเป็นไปตามมาตรฐาน", "ประสิทธิภาพอยู่ในระดับที่คาดการณ์ได้", "การประสานงานเป็นไปอย่างดี"],
        "operational_focus": ["การรักษามาตรฐานการทำงาน", "การหาจุดปรับปรุง", "การเพิ่มประสิทธิภาพ"],
        "time_awareness": "ช่วงเวลาที่เสถียรเหมาะสำหรับการพัฒนาต่อยอด"
    },
    "WEAK_START": {
        "status": "ช่วงเริ่มต้นยังต่ำกว่าเป้า", 
        "color": "text-warning", 
        "icon": "bi-hourglass-split",
        "narrative_template": "เริ่มต้นช้า แต่ยังมีเวลาชดเชย",
        "base_recommendation": "เริ่มต้นช้า แต่ยังมีโอกาสเร่งรัด ควรเตรียมพร้อม",
        "efficiency_insights": ["การรับอ้อยในช่วงแรกช้ากว่าปกติ", "ยังมีโอกาสในการปรับปรุง", "ต้องเตรียมพร้อมสำหรับช่วงที่อ้อยเข้ามาก"],
        "operational_focus": ["การเร่งรัดประสิทธิภาพ", "การเตรียมพร้อม", "การวิเคราะห์สาเหตุการเริ่มต้นที่ล่าช้า"],
        "time_awareness": "ช่วงเวลาที่เหลือเป็นโอกาสในการชดเชย"
    }
}

# --- ENHANCED ROOT CAUSE SUGGESTION DICTIONARY ---
ROOT_CAUSE_SUGGESTIONS = {
    "QUANTITY_PUSH": [
        "อาจมีการเร่งตัดอ้อยในบางพื้นที่จนละเลยการทำความสะอาด", 
        "อาจเป็นอ้อยจากโซนที่ประสบปัญหาภัยแล้ง ทำให้คุณภาพไม่ดีเท่าที่ควร", 
        "อาจต้องทบทวนขั้นตอนการตรวจรับคุณภาพที่หน้าโรงงาน",
        "การขนส่งอ้อยไฟอาจมีปัญหาในการจัดการความร้อน"
    ],
    "QUALITY_FOCUS": [
        "อาจมีปัญหาการจราจรหรือคอขวดในการขนส่งจากบางโซน", 
        "โควต้าจากบางพื้นที่อาจจะยังเข้ามาไม่เต็มที่",
        "การประสานงานกับทีมขนส่งอาจยังไม่เต็มประสิทธิภาพ",
        "สภาพอากาศอาจส่งผลต่อการขนส่งอ้อยสด"
    ],
    "CRITICAL": [
        "อาจมีฝนตกหนักในพื้นที่สำคัญซึ่งเป็นอุปสรรคต่อการตัดและขนส่ง", 
        "ควรตรวจสอบว่ามีปัญหาการสื่อสารระหว่างทีมส่งเสริมและทีมขนส่งหรือไม่",
        "อาจมีปัญหาความพร้อมของเครื่องจักรหรือระบบหยุดทำงานกะทันหัน",
        "การจัดการแรงงานอาจมีปัญหาในช่วงเปลี่ยนกะทำให้งานไม่ต่อเนื่อง"
    ],
    "VOLATILE_PERFORMANCE": [
        "การกระจายเวลาเข้าหีบของรถยังไม่เหมาะสม อาจกระจุกตัวแค่บางช่วงเวลา",
        "การประสานงานเรื่องตารางเดินรถกับผู้รับเหมาขนส่งอาจมีช่องว่าง",
        "สภาพการจราจรหรือปัญหาระหว่างเส้นทางอาจส่งผลกระทบเป็นช่วงๆ",
        "ระบบการจัดการคิวรถหน้าโรงงานอาจยังไม่คล่องตัวเท่าที่ควร"
    ]
}

# --- LOCAL AI ENGINE ---
class LocalAIEngine:
    def __init__(self):
        self.quantity_model, self.quality_model, self.persona_classifier = None, None, None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, stats, exec_summary, trend_data) -> np.ndarray:
        today_total, avg_total = stats.get('today_total', 0), stats.get('avg_daily_tons', 0)
        hours = exec_summary.get('hours_processed', 0)
        
        features = [
            today_total, avg_total,
            stats.get('type_1_percent', 0), stats.get('avg_fresh_percent', 0),
            hours, exec_summary.get('peak_hour_tons', 0),
            trend_data.get('tons_trend_percent', 0), trend_data.get('fresh_trend_percent', 0),
            (today_total / avg_total) if avg_total > 0 else 1.0,
            (stats.get('type_1_percent', 0) / stats.get('avg_fresh_percent', 0)) if stats.get('avg_fresh_percent', 0) > 0 else 1.0,
            (exec_summary.get('peak_hour_tons', 0) / (today_total / hours)) if today_total > 0 and hours > 0 else 1.0,
            abs(trend_data.get('tons_trend_percent', 0)),
            abs(trend_data.get('fresh_trend_percent', 0)),
            1.0 if datetime.now().month in [12, 1, 2, 3, 4] else 0.0
        ]
        return np.array(features).reshape(1, -1)

    def train_models(self, historical_data):
        if len(historical_data) < local_ai_config.min_data_points: return False
        try:
            X, y_qty, y_qly, y_pers = [], [], [], []
            for dp in historical_data:
                X.append(self.extract_features(dp.get('stats',{}), dp.get('exec_summary',{}), dp.get('trend_data',{})).flatten())
                y_qty.append(dp.get('stats',{}).get('today_total',0))
                y_qly.append(dp.get('stats',{}).get('type_1_percent',0))
                scores = dp.get('scores',{'quantity':3,'quality':3,'stability':3})
                y_pers.append(1 if (scores['quantity']+scores['quality']+scores['stability'])/3 >= 4 else 0)
            
            X_scaled = self.scaler.fit_transform(np.array(X))
            self.quantity_model = RandomForestRegressor(n_estimators=50,random_state=42).fit(X_scaled, np.array(y_qty))
            self.quality_model = RandomForestRegressor(n_estimators=50,random_state=42).fit(X_scaled, np.array(y_qly))
            self.persona_classifier = RandomForestClassifier(n_estimators=50,random_state=42).fit(X_scaled, np.array(y_pers))
            self.is_trained = True
            self.save_models()
            return True
        except Exception as e:
            print(f"Training error: {e}")
            return False

    def save_models(self):
        try:
            with open(f"{local_ai_config.model_path}quantity_model.pkl", 'wb') as f: pickle.dump(self.quantity_model, f)
            with open(f"{local_ai_config.model_path}quality_model.pkl", 'wb') as f: pickle.dump(self.quality_model, f)
            with open(f"{local_ai_config.model_path}persona_classifier.pkl", 'wb') as f: pickle.dump(self.persona_classifier, f)
            with open(f"{local_ai_config.scaler_path}scaler.pkl", 'wb') as f: pickle.dump(self.scaler, f)
        except Exception as e: print(f"Model save error: {e}")

    def load_models(self):
        try:
            with open(f"{local_ai_config.model_path}quantity_model.pkl", 'rb') as f: self.quantity_model = pickle.load(f)
            with open(f"{local_ai_config.model_path}quality_model.pkl", 'rb') as f: self.quality_model = pickle.load(f)
            with open(f"{local_ai_config.model_path}persona_classifier.pkl", 'rb') as f: self.persona_classifier = pickle.load(f)
            with open(f"{local_ai_config.scaler_path}scaler.pkl", 'rb') as f: self.scaler = pickle.load(f)
            self.is_trained = True
            return True
        except FileNotFoundError:
            self.is_trained = False
            return False
        except Exception as e:
            print(f"Model load error: {e}")
            self.is_trained = False
            return False

local_ai_engine = LocalAIEngine()

# Try to load existing models at startup
try:
    local_ai_engine.load_models()
    print(f"AI models loaded successfully. Trained: {local_ai_engine.is_trained}")
except Exception as e:
    print(f"Failed to load AI models: {e}")

# --- HELPER FUNCTIONS ---
def _local_ai_anomaly_detection(stats, exec_summary) -> List[str]:
    anomalies = []
    try:
        today_total, avg_total = stats.get('today_total', 0), stats.get('avg_daily_tons', 0)
        
        # If no data today, return empty anomalies
        if today_total <= 0:
            return anomalies
            
        # 1. ตรวจสอบความผิดปกติของปริมาณรวม
        std_total = stats.get('std_daily_tons', avg_total * 0.2)
        if avg_total > 0 and std_total > 0:
            z_score = abs(today_total - avg_total) / std_total
            if z_score > 2.0:
                diff_pct = ((today_total - avg_total) / avg_total) * 100
                if diff_pct > 0:
                    anomalies.append(f"ปริมาณอ้อยวันนี้ ({format_quantity_with_unit(today_total, 'ตัน', 0)}) สูงกว่าค่าเฉลี่ย {format_percentage(diff_pct, 1, True)} อย่างมีนัยสำคัญ (Z-score: {z_score:.1f}) - อาจเกิดจาก: การเร่งรัดการขนส่ง, การตัดอ้อยเพิ่มเติม, หรือการปรับปรุงประสิทธิภาพ")
                else:
                    anomalies.append(f"ปริมาณอ้อยวันนี้ ({format_quantity_with_unit(today_total, 'ตัน', 0)}) ต่ำกว่าค่าเฉลี่ย {format_percentage(abs(diff_pct), 1, False)} อย่างมีนัยสำคัญ (Z-score: {z_score:.1f}) - อาจเกิดจาก: ปัญหาการขนส่ง, การหยุดเครื่องจักร, หรือสภาพอากาศไม่เอื้ออำนวย")
        
        # 2. ตรวจสอบความผิดปกติของคุณภาพอ้อย
        today_fresh, avg_fresh = stats.get('type_1_percent', 0), stats.get('avg_fresh_percent', 0)
        if avg_fresh > 0 and abs(today_fresh - avg_fresh) > 10:
            diff = today_fresh - avg_fresh
            if diff > 0:
                anomalies.append(f"คุณภาพอ้อยสดวันนี้ ({format_percentage(today_fresh, 1, False)}) สูงกว่าค่าเฉลี่ย {format_percentage(diff, 1, True)} - อาจเกิดจาก: การปรับปรุงการจัดการอ้อย, การตัดอ้อยสดเพิ่มขึ้น, หรือการควบคุมคุณภาพที่ดีขึ้น")
            else:
                anomalies.append(f"คุณภาพอ้อยสดวันนี้ ({format_percentage(today_fresh, 1, False)}) ต่ำกว่าค่าเฉลี่ย {format_percentage(abs(diff), 1, False)} - อาจเกิดจาก: การเพิ่มอ้อยไฟ, การจัดการอ้อยที่ไม่เหมาะสม, หรือการตัดอ้อยที่ล่าช้า")
        
        # 3. ตรวจสอบความผิดปกติของอัตราการรับอ้อยต่อชั่วโมง
        hours = exec_summary.get('hours_processed', 0)
        if hours > 0 and avg_total > 0:
            expected_hourly = avg_total / 24 
            if expected_hourly > 0:
                rate_ratio = (today_total / hours) / expected_hourly
                if rate_ratio > 1.5:
                    anomalies.append(f"อัตราการรับอ้อยต่อชั่วโมงสูงผิดปกติ ({rate_ratio:.1f} เท่าของค่าเฉลี่ย) - อาจเกิดจาก: การเร่งรัดการขนส่ง, การเพิ่มกำลังการผลิต, หรือการปรับปรุงประสิทธิภาพการทำงาน")
                elif rate_ratio < 0.5:
                    anomalies.append(f"อัตราการรับอ้อยต่อชั่วโมงต่ำผิดปกติ ({rate_ratio:.1f} เท่าของค่าเฉลี่ย) - อาจเกิดจาก: ปัญหาการขนส่ง, การหยุดเครื่องจักร, หรือการขาดแคลนแรงงาน")
        
        # 4. ตรวจสอบความผิดปกติของชั่วโมงเร่งด่วน
        peak_hour_tons = exec_summary.get('peak_hour_tons', 0)
        if hours > 0 and today_total > 0 and peak_hour_tons > 0:
            avg_hourly = today_total / hours
            if avg_hourly > 0:
                peak_ratio = peak_hour_tons / avg_hourly
                if peak_ratio > 2.5:
                    anomalies.append(f"ชั่วโมงเร่งด่วนมีปริมาณสูงผิดปกติ ({peak_ratio:.1f} เท่าของค่าเฉลี่ยต่อชั่วโมง) - อาจเกิดจาก: การกระจุกตัวของรถขนส่ง, การเร่งรัดการขนส่ง, หรือการวางแผนการขนส่งที่ไม่เหมาะสม")
        
        # 5. ตรวจสอบความผิดปกติของสัดส่วนราง A/B
        line_a_total = stats.get('line_a_total', 0)
        line_b_total = stats.get('line_b_total', 0)
        if line_a_total > 0 and line_b_total > 0:
            total_lines = line_a_total + line_b_total
            line_a_ratio = (line_a_total / total_lines) * 100
            if line_a_ratio > 70:
                anomalies.append(f"ราง A รับอ้อยมากผิดปกติ ({line_a_ratio:.1f}% ของทั้งหมด) - อาจเกิดจาก: ปัญหาราง B, การวางแผนการขนส่งที่ไม่สมดุล, หรือการปรับเปลี่ยนเส้นทางขนส่ง")
            elif line_a_ratio < 30:
                anomalies.append(f"ราง B รับอ้อยมากผิดปกติ ({100-line_a_ratio:.1f}% ของทั้งหมด) - อาจเกิดจาก: ปัญหาราง A, การวางแผนการขนส่งที่ไม่สมดุล, หรือการปรับเปลี่ยนเส้นทางขนส่ง")
        
        # 6. ตรวจสอบความผิดปกติของสัดส่วนอ้อยสด/อ้อยไฟ
        type_1_total = stats.get('type_1_total', 0)
        type_2_total = stats.get('type_2_total', 0)
        if type_1_total > 0 and type_2_total > 0:
            total_types = type_1_total + type_2_total
            fresh_ratio = (type_1_total / total_types) * 100
            if fresh_ratio > 95:
                anomalies.append(f"อ้อยสดมีสัดส่วนสูงผิดปกติ ({fresh_ratio:.1f}%) - อาจเกิดจาก: การตัดอ้อยสดเพิ่มขึ้น, การลดการเผาอ้อย, หรือการปรับปรุงการจัดการอ้อย")
            elif fresh_ratio < 50:
                anomalies.append(f"อ้อยไฟมีสัดส่วนสูงผิดปกติ ({100-fresh_ratio:.1f}%) - อาจเกิดจาก: การเพิ่มการเผาอ้อย, การตัดอ้อยที่ล่าช้า, หรือการจัดการอ้อยที่ไม่เหมาะสม")
                
    except Exception as e:
        print(f"Anomaly detection error: {e}")
    return anomalies

def _calculate_scores(stats, exec_summary):
    scores = {'quantity': 3, 'quality': 3, 'stability': 3}
    today_total, avg_total = stats.get('today_total', 0), stats.get('avg_daily_tons', 0)
    
    # If no data today, return neutral scores
    if today_total <= 0:
        return {'quantity': 0, 'quality': 0, 'stability': 0}
    
    if avg_total > 0:
        diff_pct = ((today_total - avg_total) / avg_total) * 100
        if diff_pct > 15: scores['quantity'] = 5
        elif diff_pct > 5: scores['quantity'] = 4
        elif diff_pct < -20: scores['quantity'] = 1
        elif diff_pct < -10: scores['quantity'] = 2
    
    today_fresh, avg_fresh = stats.get('type_1_percent', 0), stats.get('avg_fresh_percent', 0)
    if avg_fresh > 0:
        diff = today_fresh - avg_fresh
        if diff > 5: scores['quality'] = 5
        elif diff > 2: scores['quality'] = 4
        elif diff < -8: scores['quality'] = 1
        elif diff < -4: scores['quality'] = 2
        
    hours, peak_tons = exec_summary.get('hours_processed', 0), exec_summary.get('peak_hour_tons', 0)
    if hours > 0 and today_total > 0:
        avg_hourly = today_total / hours
        if avg_hourly > 0:
            ratio = peak_tons / avg_hourly
            if ratio < 1.3: scores['stability'] = 5
            elif ratio < 1.6: scores['stability'] = 4
            elif ratio < 2.5: scores['stability'] = 2
            elif ratio >= 2.5: scores['stability'] = 1
    return scores

def _select_persona_name(scores, trend_score, hours_processed):
    qty, qly, stb = scores['quantity'], scores['quality'], scores['stability']
    if 1 < hours_processed < 8 and (qty < 3 or qly < 3): return "WEAK_START"
    if qty <= 2 and qly <= 2: return "CRITICAL"
    if qty >= 4 and qly >= 4 and stb >= 4: return "EXCELLENT"
    if trend_score <= -2 and qty >= 3 and qly >= 3: return "CONCERNING_TREND"
    if trend_score >= 2 and qty >= 3 and qly >= 3: return "STABLE_RECOVERY"
    if qty >= 4 and qly <= 2: return "QUANTITY_PUSH"
    if qty <= 2 and qly >= 4: return "QUALITY_FOCUS"
    if qty >= 3 and stb <= 2: return "VOLATILE_PERFORMANCE"
    return "STEADY_PERFORMANCE"

def _get_temporal_context(selected_date: datetime) -> Optional[str]:
    if selected_date.month == 12 and selected_date.day in [29, 30]:
        return "PRE_HOLIDAY_PUSH"
    if selected_date.month == 1 and selected_date.day in [3, 4, 5]:
        return "POST_HOLIDAY_RESTART"
    return None

def _identify_operational_patterns(stats: Dict[str, Any], exec_summary: Dict[str, Any]) -> Optional[str]:
    """Analyzes intra-day data to identify specific operational patterns like clustering or gaps."""
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    hours = exec_summary.get('hours_processed', 0)
    peak_tons = exec_summary.get('peak_hour_tons', 0)

    if hours < 4 or today_total == 0:
        return None 

    avg_hourly_rate = today_total / hours
    
    if avg_hourly_rate > 0:
        peak_ratio = peak_tons / avg_hourly_rate
        if peak_ratio >= 2.5:
            return f"มีลักษณะ **'อ้อยกระจุกตัวอย่างหนาแน่น'** ในบางช่วงเวลา โดยชั่วโมงที่รับอ้อยสูงสุดนั้นมากกว่าค่าเฉลี่ยถึง {peak_ratio:.1f} เท่า"
        elif peak_ratio >= 1.8:
            return f"พบ **'การกระจุกตัวของอ้อย'** ในช่วงเวลาสั้นๆ ซึ่งมากกว่าปกติ"

    if avg_total > 0:
        pro_rated_target = avg_total * (hours / 24)
        if today_total < pro_rated_target * 0.7: 
             return "การรับอ้อยในวันนี้มีลักษณะ **'ขาดช่วงเป็นระยะ'** ทำให้การส่งมอบไม่ต่อเนื่องเท่าที่ควร"
             
    return None

def _advanced_pattern_recognition(stats: Dict[str, Any], exec_summary: Dict[str, Any], trend_data: Dict[str, Any], selected_date: Optional[datetime] = None) -> Dict[str, Any]:
    """Advanced AI-powered pattern recognition for operational intelligence."""
    patterns = {
        'seasonal_pattern': None,
        'weekly_pattern': None,
        'hourly_pattern': None,
        'quality_pattern': None,
        'anomaly_score': 0.0,
        'predictive_signals': []
    }
    
    try:
        # Use selected_date if provided, otherwise use current date
        if selected_date is None:
            selected_date = datetime.now()
        
        # Seasonal pattern detection based on selected date
        current_month = selected_date.month
        if current_month in [12, 1, 2]:
            patterns['seasonal_pattern'] = "peak_season"
        elif current_month in [3, 4]:
            patterns['seasonal_pattern'] = "late_season"
        elif current_month in [5, 6, 7, 8, 9, 10, 11]:
            patterns['seasonal_pattern'] = "off_season"
        
        # Weekly pattern detection based on selected date
        current_weekday = selected_date.weekday()
        if current_weekday in [0, 1]:  # Monday, Tuesday
            patterns['weekly_pattern'] = "week_start"
        elif current_weekday in [4, 5]:  # Friday, Saturday
            patterns['weekly_pattern'] = "week_end"
        
        # Hourly pattern analysis
        hours_processed = exec_summary.get('hours_processed', 0)
        today_total = stats.get('today_total', 0)
        if hours_processed > 0 and today_total > 0:
            current_hourly_rate = today_total / hours_processed
            avg_total = stats.get('avg_daily_tons', 0)
            if avg_total > 0:
                expected_hourly = avg_total / 24
                if expected_hourly > 0:
                    efficiency_ratio = current_hourly_rate / expected_hourly
                    if efficiency_ratio > 1.5:
                        patterns['hourly_pattern'] = "high_efficiency"
                    elif efficiency_ratio < 0.7:
                        patterns['hourly_pattern'] = "low_efficiency"
                    else:
                        patterns['hourly_pattern'] = "normal_efficiency"
        
        # Quality pattern analysis
        today_fresh = stats.get('type_1_percent', 0)
        avg_fresh = stats.get('avg_fresh_percent', 0)
        if avg_fresh > 0:
            quality_ratio = today_fresh / avg_fresh
            if quality_ratio > 1.1:
                patterns['quality_pattern'] = "improving"
            elif quality_ratio < 0.9:
                patterns['quality_pattern'] = "declining"
            else:
                patterns['quality_pattern'] = "stable"
        
        # Anomaly scoring
        anomaly_factors = []
        if avg_total > 0:
            volume_deviation = abs(today_total - avg_total) / avg_total
            if volume_deviation > 0.3:
                anomaly_factors.append(volume_deviation * 0.4)
        
        if avg_fresh > 0:
            quality_deviation = abs(today_fresh - avg_fresh) / avg_fresh
            if quality_deviation > 0.15:
                anomaly_factors.append(quality_deviation * 0.3)
        
        # Trend-based anomaly
        if trend_data.get('has_trend_data'):
            tons_trend = abs(trend_data.get('tons_trend_percent', 0))
            fresh_trend = abs(trend_data.get('fresh_trend_percent', 0))
            if tons_trend > 10:
                anomaly_factors.append(tons_trend * 0.01)
            if fresh_trend > 5:
                anomaly_factors.append(fresh_trend * 0.02)
        
        patterns['anomaly_score'] = min(1.0, sum(anomaly_factors))
        
        # Predictive signals
        if patterns['anomaly_score'] > 0.7:
            patterns['predictive_signals'].append("high_risk_alert")
        elif patterns['anomaly_score'] > 0.4:
            patterns['predictive_signals'].append("moderate_risk_alert")
        
        if patterns['seasonal_pattern'] == "late_season" and patterns['quality_pattern'] == "declining":
            patterns['predictive_signals'].append("seasonal_quality_decline")
        
        if patterns['weekly_pattern'] == "week_end" and patterns['hourly_pattern'] == "high_efficiency":
            patterns['predictive_signals'].append("weekend_push_pattern")
            
    except Exception as e:
        print(f"Pattern recognition error: {e}")
    
    return patterns

def _generate_guru_narrative(persona_name, stats, exec_summary, temporal_context, contextual_data=None, selected_date=None) -> str:
    if temporal_context == "PRE_HOLIDAY_PUSH":
        return "เข้าสู่ช่วงโค้งสุดท้ายก่อนหยุดยาวปีใหม่แล้วนะครับ ตอนนี้เรากำลังอยู่ในช่วงเร่งระบายอ้อยที่ค้างอยู่ให้หมด การที่ปริมาณไหลเข้ามาเยอะถือเป็นเรื่องดี แต่ก็ต้องคอยดูแลเรื่องคุณภาพและความปลอดภัยเป็นพิเศษด้วยครับ"
    if temporal_context == "POST_HOLIDAY_RESTART":
        return "กลับมาเดินเครื่องกันอีกครั้งหลังหยุดยาวนะครับ ช่วงวันแรกๆ แบบนี้เราจะเน้นการค่อยๆ เพิ่มปริมาณอ้อยเข้าหีบเพื่อตรวจสอบความพร้อมของเครื่องจักรและระบบต่างๆ ความเสถียรในช่วงนี้สำคัญที่สุดครับ"

    # Get time-based context
    if selected_date is None:
        selected_date = datetime.now()
    time_context = _get_time_based_context(selected_date)
    
    # Generate human-like greeting
    greeting = _generate_human_like_greeting(time_context, persona_name)
    
    persona = PERSONAS[persona_name]
    base_narrative = persona.get("narrative_template", "ภาพรวมการดำเนินงานวันนี้")

    # Add operational pattern analysis
    operational_pattern = _identify_operational_patterns(stats, exec_summary)
    if operational_pattern:
        base_narrative += f" รูปแบบการทำงานที่น่าสนใจในวันนี้คือ {operational_pattern} ครับ"

    # Add root cause suggestions
    if persona_name in ROOT_CAUSE_SUGGESTIONS:
        suggestion = random.choice(ROOT_CAUSE_SUGGESTIONS[persona_name])
        base_narrative += f" หนึ่งในสาเหตุที่เป็นไปได้คือ '{suggestion}' ซึ่งอาจต้องมีการตรวจสอบเพิ่มเติมครับ"
    
    # Add contextual insights
    contextual_insights = _generate_contextual_insights(stats, exec_summary, time_context)
    if contextual_insights:
        base_narrative += f" {contextual_insights[0]} ครับ"
    
    # Add emotional response
    emotional_response = _generate_emotional_response(persona_name, stats, time_context)
    if emotional_response:
        base_narrative += f" {emotional_response}"
    
    # Weather context removed
    
    # Combine greeting with narrative
    full_narrative = f"{greeting} {base_narrative}"
    
    return full_narrative

def _generate_ai_enhanced_narrative(persona_name, stats, exec_summary, temporal_context, patterns, efficiency_metrics, contextual_data=None, selected_date=None, analysis_mode: str = None) -> str:
    """AI-enhanced narrative generation with contextual intelligence and pattern recognition."""
    
    # Base narrative with human-like intelligence
    base_narrative = _generate_guru_narrative(persona_name, stats, exec_summary, temporal_context, contextual_data, selected_date)
    
    # Get time-based context
    if selected_date is None:
        selected_date = datetime.now()
    time_context = _get_time_based_context(selected_date)
    
    # AI enhancements
    enhancements = []
    
    # Pattern-based enhancements
    anomaly_score = patterns.get('anomaly_score', 0.0)
    if anomaly_score > 0.7:
        enhancements.append(f"ระบบ AI ตรวจพบความผิดปกติในระดับสูง (คะแนน: {anomaly_score:.2f}) ซึ่งบ่งชี้ถึงสถานการณ์ที่ต้องให้ความสนใจเป็นพิเศษ")
    elif anomaly_score > 0.4:
        enhancements.append(f"ระบบ AI ตรวจพบความผิดปกติในระดับปานกลาง (คะแนน: {anomaly_score:.2f}) ซึ่งควรติดตามอย่างใกล้ชิด")
    
    seasonal_pattern = patterns.get('seasonal_pattern')
    if seasonal_pattern == "peak_season":
        enhancements.append("เรากำลังอยู่ในช่วงฤดูหีบอ้อยที่คึกคักที่สุด ซึ่งเป็นโอกาสทองในการเพิ่มประสิทธิภาพการผลิต")
    elif seasonal_pattern == "late_season":
        enhancements.append("เข้าสู่ช่วงปลายฤดูหีบแล้ว ควรมีการวางแผนการจัดการอ้อยที่เหลืออย่างรอบคอบ")
    
    # Efficiency-based enhancements
    if 'volume_efficiency' in efficiency_metrics:
        vol_eff = efficiency_metrics['volume_efficiency']
        if vol_eff > 120:
            enhancements.append(f"ประสิทธิภาพการรับอ้อยสูงมาก ({format_percentage(vol_eff, 1, False)}) สะท้อนถึงการทำงานที่มีประสิทธิภาพของทีมงาน")
        elif vol_eff < 80:
            enhancements.append(f"ประสิทธิภาพการรับอ้อยต่ำกว่าเป้าหมาย ({format_percentage(vol_eff, 1, False)}) จำเป็นต้องมีการปรับปรุง")
    
    if 'quality_efficiency' in efficiency_metrics:
        qual_eff = efficiency_metrics['quality_efficiency']
        if qual_eff > 105:
            enhancements.append("คุณภาพอ้อยดีกว่าค่าเฉลี่ย ซึ่งจะส่งผลดีต่อค่า CCS")
        elif qual_eff < 95:
            enhancements.append("คุณภาพอ้อยต่ำกว่าเกณฑ์ ควรให้ความสำคัญกับการควบคุมคุณภาพมากขึ้น")
    
    # Predictive signals
    predictive_signals = patterns.get('predictive_signals', [])
    if "high_risk_alert" in predictive_signals:
        enhancements.append("ระบบ AI คาดการณ์ว่าอาจเกิดปัญหาสำคัญในอนาคตอันใกล้ ควรมีการเตรียมแผนรองรับ")
    elif "seasonal_quality_decline" in predictive_signals:
        enhancements.append("ระบบ AI ตรวจพบแนวโน้มคุณภาพที่ลดลงตามฤดูกาล ซึ่งเป็นรูปแบบที่พบได้ในช่วงปลายฤดู")
    
    # Time-based enhancements (skip immediate time-of-day advice for historical mode)
    if analysis_mode != 'historical':
        if time_context['time_specific_advice']:
            enhancements.extend(time_context['time_specific_advice'])
        # Proactive suggestions (only for current mode)
        proactive_suggestions = _generate_proactive_suggestions(stats, exec_summary, time_context)
        if proactive_suggestions:
            enhancements.extend(proactive_suggestions[:2])  # Limit to 2 suggestions
    
    # Combine narrative
    if enhancements:
        enhanced_narrative = base_narrative + " " + " ".join(enhancements) + "ครับ"
    else:
        enhanced_narrative = base_narrative

    # Historical mode framing
    if analysis_mode == 'historical':
        enhanced_narrative = (
            "สรุปย้อนหลัง: " + enhanced_narrative +
            " โฟกัสบทเรียนและแนวโน้มเพื่อปรับแผนในรอบถัดไปครับ"
        )
    
    return enhanced_narrative

def _generate_conversational_narrative_v3(persona_name: str, stats: dict, patterns: dict, efficiency_metrics: dict, analysis_mode: str = None) -> str:
    """
    V5 (ChatGPT-Style): สร้างเรื่องเล่าเชิงสนทนาที่มีสไตล์เหมือน ChatGPT
    มีความฉลาด, เป็นมิตร, และให้ข้อมูลเชิงลึกที่เข้าใจง่าย
    """
    persona = PERSONAS.get(persona_name, {})
    
    # --- ดึงข้อมูลสำคัญออกมาเพื่อใช้งานได้ง่ายขึ้น ---
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    today_fresh_pct = stats.get('type_1_percent', 0)
    avg_fresh_pct = stats.get('avg_fresh_percent', 0)
    
    # --- ประโยคเปิดการสนทนาแบบ ChatGPT ---
    openers = [
        "สวัสดีครับ! ผมได้วิเคราะห์ข้อมูลการดำเนินงานล่าสุดแล้ว และมีเรื่องน่าสนใจหลายประเด็นที่อยากแชร์ให้ฟัง",
        "วันนี้มีข้อมูลที่น่าสนใจมากครับ! ผมขอสรุปสถานการณ์ให้ฟังแบบเข้าใจง่าย",
        "ผมเพิ่งวิเคราะห์ข้อมูลเสร็จ และพบประเด็นสำคัญที่ควรให้ความสนใจครับ"
    ]
    narrative_parts = [random.choice(openers)]
    
    # --- สร้างเรื่องราวหลักตาม Persona แบบ ChatGPT ---
    if persona_name == "EXCELLENT":
        vol_eff = efficiency_metrics.get('volume_efficiency', 100)
        narrative_parts.append(f"\n**🎉 ผลงานยอดเยี่ยมมากครับ!**\n\nต้องบอกว่าวันนี้ทีมงานทำได้ดีมากเลยครับ! ปริมาณอ้อยที่รับเข้ามาสูงกว่าค่าเฉลี่ยถึง **{format_percentage(vol_eff - 100, 1, True)}** ซึ่งแสดงให้เห็นถึงการวางแผนและการจัดการที่มีประสิทธิภาพ")
        narrative_parts.append(f"**📊 ตัวเลขสำคัญ:**\n• ปริมาณรับเข้า: **{format_quantity_with_unit(today_total, 'ตัน', 0)}**\n• คุณภาพอ้อยสด: **{format_percentage(today_fresh_pct, 1, False)}**")
        if efficiency_metrics.get('peak_ratio', 1.0) < 1.5:
            narrative_parts.append("**💡 สิ่งที่ทำได้ดี:** การไหลเข้าของอ้อยที่สม่ำเสมอ ทำให้เครื่องจักรทำงานได้เต็มศักยภาพและประหยัดพลังงาน")

    elif persona_name == "CRITICAL":
        qty_diff_pct = ((today_total - avg_total) / avg_total * 100) if avg_total > 0 else 0
        qly_diff = today_fresh_pct - avg_fresh_pct
        anomaly_score = patterns.get('anomaly_score', 0.0)
        narrative_parts.append(f"\n**⚠️ สถานการณ์ที่ต้องให้ความสนใจ**\n\nผมต้องแจ้งให้ทราบว่าวันนี้มีประเด็นที่ต้องแก้ไขด่วนครับ")
        narrative_parts.append(f"**🔍 ปัญหาที่พบ:**\n• ปริมาณอ้อยต่ำกว่าค่าเฉลี่ย **{format_percentage(abs(qty_diff_pct), 1, False)}**\n• คุณภาพลดลง **{format_percentage(abs(qly_diff), 1, False)}**")
        narrative_parts.append(f"**🤖 การแจ้งเตือนจาก AI:** ระบบตรวจพบความผิดปกติในระดับสูง (คะแนน {anomaly_score:.2f}) ซึ่งบ่งชี้ว่าต้องมีการดำเนินการแก้ไขอย่างเร่งด่วน")

    elif persona_name == "QUANTITY_PUSH":
        qly_diff = today_fresh_pct - avg_fresh_pct
        narrative_parts.append(f"\n**⚖️ สมดุลปริมาณ-คุณภาพ**\n\nวันนี้ทำปริมาณได้ตามเป้าแล้วครับ แต่ต้องแลกมาด้วยคุณภาพที่ลดลงเล็กน้อย")
        narrative_parts.append(f"**📈 ข้อมูลสำคัญ:**\n• ปริมาณ: **{format_quantity_with_unit(today_total, 'ตัน', 0)}**\n• คุณภาพลดลง: **{format_percentage(abs(qly_diff), 1, False)}**")
        narrative_parts.append("**💭 ข้อสังเกต:** นี่เป็นเรื่องปกติในบางช่วง แต่ควรให้ความสำคัญกับสมดุลนี้เพื่อรักษา CCS ในระยะยาวครับ")

    elif persona_name == "QUALITY_FOCUS":
        qty_gap = avg_total - today_total
        narrative_parts.append(f"\n**🌟 คุณภาพดี แต่ปริมาณขาด**\n\nคุณภาพอ้อยวันนี้ดีมากครับ! แต่ปริมาณยังขาดเป้าหมายไปบ้าง")
        narrative_parts.append(f"**📊 ข้อมูล:**\n• คุณภาพ: **{format_percentage(today_fresh_pct, 1, False)}**\n• ขาดเป้า: **{format_num(qty_gap, 0)} ตัน**")
        narrative_parts.append("**🎯 โอกาสที่พลาด:** ยังไม่ได้ใช้กำลังการผลิตให้เต็มศักยภาพ ซึ่งอาจเป็นเพราะการวางแผนการขนส่งหรือการจัดการคิว")

    elif persona_name == "VOLATILE_PERFORMANCE":
        peak_ratio = efficiency_metrics.get('peak_ratio', 1.0)
        narrative_parts.append(f"\n**📈📉 ความผันผวนสูง**\n\nปริมาณรวมดูดีครับ แต่การรับอ้อยมีความผันผวนมาก")
        narrative_parts.append(f"**📊 ข้อมูล:**\n• ปริมาณรวม: **{format_quantity_with_unit(today_total, 'ตัน', 0)}**\n• Peak สูงสุด: **{peak_ratio:.1f}x** ปกติ")
        narrative_parts.append("**⚠️ ผลกระทบ:** ความผันผวนนี้กระทบการเดินเครื่องจักรและสิ้นเปลืองพลังงานมากกว่าปกติ")

    elif persona_name == "CONCERNING_TREND":
        narrative_parts.append(f"\n**👀 แนวโน้มที่น่าจับตา**\n\nผลงานวันนี้ยังอยู่ในเกณฑ์ครับ แต่มีสัญญาณที่ต้องระวัง")
        narrative_parts.append(f"**📊 ข้อมูลปัจจุบัน:**\n• ปริมาณ: **{format_quantity_with_unit(today_total, 'ตัน', 0)}**\n• คุณภาพ: **{format_percentage(today_fresh_pct, 1, False)}**")
        if efficiency_metrics.get('trend_direction') == 'ลง':
            narrative_parts.append("**🚨 สัญญาณเตือน:** ปริมาณเริ่มชะลอตัวลงอย่างต่อเนื่อง ต้องหาสาเหตุก่อนส่งผลกระทบใหญ่")
        elif efficiency_metrics.get('quality_trend') == 'ลง':
            narrative_parts.append("**🚨 สัญญาณเตือน:** คุณภาพเริ่มมีแนวโน้มลดลง ไม่ควรมองข้ามครับ")

    elif persona_name == "STABLE_RECOVERY":
        narrative_parts.append(f"\n**✅ กลับสู่ภาวะปกติ**\n\nเป็นข่าวดีครับ! สถานการณ์กลับมาดีแล้ว")
        narrative_parts.append(f"**📊 ข้อมูล:**\n• ปริมาณ: **{format_quantity_with_unit(today_total, 'ตัน', 0)}**\n• คุณภาพ: **{format_percentage(today_fresh_pct, 1, False)}**")
        narrative_parts.append("**🎯 สิ่งที่ต้องทำ:** รักษาความเสถียรนี้ไว้ และนำบทเรียนไปปรับปรุงเพื่อป้องกันปัญหาเกิดซ้ำครับ")

    elif persona_name == "WEAK_START":
        narrative_parts.append(f"\n**🌅 เริ่มต้นช้า แต่ยังมีโอกาส**\n\nการเริ่มต้นวันนี้อาจช้ากว่าเป้าหมายไปบ้าง แต่ยังไม่น่ากังวลครับ")
        narrative_parts.append(f"**📊 ข้อมูล:**\n• ปริมาณ: **{format_quantity_with_unit(today_total, 'ตัน', 0)}**\n• คุณภาพ: **{format_percentage(today_fresh_pct, 1, False)}**")
        narrative_parts.append("**🚀 โอกาส:** ยังมีเวลาเร่งการผลิตให้กลับมาตามเป้า และเตรียมพร้อมรับมือช่วงที่อ้อยหนาแน่น")

    else: # STEADY_PERFORMANCE
        narrative_parts.append(f"\n**📊 การดำเนินงานเสถียร**\n\nภาพรวมวันนี้การดำเนินงานมีเสถียรภาพและเป็นไปตามเกณฑ์มาตรฐานครับ")
        narrative_parts.append(f"**📈 ข้อมูลสำคัญ:**\n• ปริมาณรับเข้า: **{format_quantity_with_unit(today_total, 'ตัน', 0)}**\n• คุณภาพอ้อยสด: **{format_percentage(today_fresh_pct, 1, False)}**\n• การไหลเข้าของอ้อย: สม่ำเสมอดี")
        if patterns.get('quality_pattern') == 'declining':
            narrative_parts.append("**👀 ข้อสังเกต:** แนวโน้มคุณภาพเริ่มชะลอตัวลงเล็กน้อย ควรจับตาดูต่อไปครับ")

    # --- สรุปปิดท้ายแบบ ChatGPT ---
    if analysis_mode == 'historical':
        final_thoughts = [
            "\n**📚 สรุปการวิเคราะห์ย้อนหลัง**\n\nนี่คือการวิเคราะห์เหตุการณ์ในวันนั้นครับ ประเด็นเหล่านี้เป็นข้อสังเกตสำคัญจากข้อมูลในอดีตที่สามารถนำไปปรับปรุงการทำงานในอนาคตได้",
            "\n**🎓 บทเรียนจากอดีต**\n\nการวิเคราะห์ย้อนหลังนี้ช่วยให้เราเข้าใจรูปแบบและแนวโน้มที่เกิดขึ้น เพื่อนำไปปรับปรุงการทำงานในอนาคตครับ",
            "\n**🔍 มุมมองจากข้อมูลย้อนหลัง**\n\nข้อมูลในอดีตเป็นครูที่ดีเสมอครับ การวิเคราะห์นี้จะช่วยให้เราวางแผนได้ดีขึ้นในอนาคต"
        ]
        final_thought = random.choice(final_thoughts)
    else: # Current day
        final_thoughts = [
            "\n**💡 สรุปสถานการณ์ล่าสุด**\n\nนี่คือบทสรุปสถานการณ์ล่าสุดครับ หวังว่าข้อมูลนี้จะเป็นประโยชน์ในการวางแผนการทำงานต่อไป หากมีคำถามเพิ่มเติม สามารถสอบถามได้เสมอครับ",
            "\n**🚀 การวางแผนต่อไป**\n\nข้อมูลนี้จะช่วยให้เราวางแผนและปรับปรุงการทำงานในวันถัดไปได้อย่างมีประสิทธิภาพครับ ผมพร้อมให้คำแนะนำเพิ่มเติมเมื่อต้องการ",
            "\n**📋 สรุปและข้อเสนอแนะ**\n\nนี่คือการวิเคราะห์ที่ผมได้ทำขึ้นครับ หากต้องการข้อมูลเพิ่มเติมหรือมีคำถาม สามารถสอบถามได้เสมอ ผมยินดีช่วยเหลือครับ"
        ]
        final_thought = random.choice(final_thoughts)
    
    narrative_parts.append(final_thought)

    return "\n".join(narrative_parts)


def _generate_ai_predictions(stats, exec_summary, trend_data, selected_date=None) -> Optional[str]:
    """
    V4 (Enhanced): สร้างการคาดการณ์ AI ที่มีการจัดกลุ่มข้อมูลที่เข้าใจง่าย
    มีการจัดรูปแบบที่ชาญฉลาด การใช้สี และการแบ่งส่วนที่อ่านง่าย
    """
    # Check if models are trained
    if not local_ai_engine.is_trained:
        print("AI models are not trained yet")
        return "**โมเดล AI ยังไม่ได้ฝึกสอน**\n\nกรุณาฝึกโมเดลก่อนใช้งานการคาดการณ์ครับ"
    
    # Check if we have enough data for prediction
    today_qty = stats.get('today_total', 0)
    if today_qty <= 0:
        return "**ข้อมูลไม่เพียงพอ**\n\nยังไม่มีข้อมูลเพียงพอสำหรับการคาดการณ์ AI ในวันนี้ครับ"
    
    try:
        # ปรับการคาดการณ์ตามวันที่ที่เลือก
        if selected_date:
            today_date = datetime.now().date()
            selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
            
            if selected_date_only < today_date:
                # สำหรับข้อมูลย้อนหลัง ให้แสดงการวิเคราะห์แนวโน้มและรูปแบบที่เกิดขึ้นจริง
                pass  # ไม่ return ออกไป ให้ดำเนินการต่อเพื่อวิเคราะห์แนวโน้ม
        
        features = local_ai_engine.extract_features(stats, exec_summary, trend_data)
        features_scaled = local_ai_engine.scaler.transform(features)
        
        pred_qty, pred_qly = local_ai_engine.quantity_model.predict(features_scaled)[0], local_ai_engine.quality_model.predict(features_scaled)[0]
        today_qly = stats.get('type_1_percent', 0)
        
        # คำนวณการเปลี่ยนแปลงแบบสมเหตุสมผล
        qty_change = ((pred_qty - today_qty) / today_qty * 100) if today_qty > 0 else 0
        qly_change = pred_qly - today_qly

        # จำกัดการเปลี่ยนแปลงให้สมเหตุสมผล (ไม่เกิน ±25%)
        qty_change = max(-25, min(25, qty_change))
        
        # ตรวจสอบความสมเหตุสมผลของค่าที่คาดการณ์อย่างเข้มงวด
        prediction_adjusted = False
        
        # ปรับค่าคาดการณ์ให้สมเหตุสมผล
        if pred_qty < 0 or pred_qty > today_qty * 1.4 or pred_qty < today_qty * 0.6:
            pred_qty = today_qty  # ใช้ค่าปัจจุบันแทน
            prediction_adjusted = True
        
        # คำนวณการเปลี่ยนแปลงใหม่หลังจากปรับค่า
        qty_change = ((pred_qty - today_qty) / today_qty * 100) if today_qty > 0 else 0
        qty_change = max(-25, min(25, qty_change))  # จำกัดไม่เกิน ±25%
        
        if abs(qty_change) < 3:
            qty_text = "ปริมาณน่าจะยังคงทรงตัวในระดับใกล้เคียงเดิม"
        elif qty_change > 3:
            qty_text = f"ปริมาณมีแนวโน้มปรับตัวสูงขึ้นราว **{format_percentage(qty_change, 1, False)}**"
        else:
            qty_text = f"ปริมาณอาจปรับตัวลดลงประมาณ **{format_percentage(abs(qty_change), 1, False)}**"
        
        # ตรวจสอบและปรับคุณภาพ
        if pred_qly < 0 or pred_qly > 100:
            pred_qly = today_qly  # ใช้ค่าปัจจุบันแทน
            prediction_adjusted = True
        
        # คำนวณการเปลี่ยนแปลงคุณภาพใหม่
        qly_change = pred_qly - today_qly
        
        if abs(qly_change) < 1.5:
            qly_text = "และคุณภาพน่าจะใกล้เคียงกับวันนี้"
        elif qly_change > 1.5:
            qly_text = "และคุณภาพมีแนวโน้มดีขึ้น"
        else:
            qly_text = "แต่คุณภาพอาจจะลดลงเล็กน้อย"
        
        # สร้างโครงสร้างการคาดการณ์แบบใหม่
        prediction_parts = []
        
        # หัวข้อหลัก
        if selected_date:
            today_date = datetime.now().date()
            selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
            
            if selected_date_only < today_date:
                prediction_parts.append(f"**การวิเคราะห์ย้อนหลัง**\n\n**วันที่:** {selected_date_only.strftime('%d/%m/%Y')}")
                prediction_parts.append(f"**ผลการวิเคราะห์:** {qty_text} {qly_text}")
                prediction_parts.append("**หมายเหตุ:** นี่คือแนวโน้มที่เกิดขึ้นจริงในอดีต")
            elif selected_date_only == today_date:
                prediction_parts.append(f"**การคาดการณ์ AI**\n\n**วันที่:** วันนี้")
                prediction_parts.append(f"**คาดการณ์วันพรุ่งนี้:** {qty_text} {qly_text}")
            else:
                prediction_parts.append(f"**การคาดการณ์ AI**\n\n**วันที่:** {selected_date_only.strftime('%d/%m/%Y')}")
                prediction_parts.append(f"**คาดการณ์วันถัดไป:** {qty_text} {qly_text}")
        else:
            prediction_parts.append(f"**การคาดการณ์ AI**\n\n**วันที่:** วันนี้")
            prediction_parts.append(f"**คาดการณ์วันพรุ่งนี้:** {qty_text} {qly_text}")
        
        # ข้อมูลเชิงปริมาณ - แสดงข้อมูลพื้นฐานเสมอ
        prediction_parts.append(f"\n**ข้อมูลเชิงปริมาณ:**")
        if selected_date:
            today_date = datetime.now().date()
            selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
            
            if selected_date_only < today_date:
                prediction_parts.append(f"• ปริมาณในวันที่ {selected_date_only.strftime('%d/%m/%Y')}: **{format_quantity_with_unit(today_qty, 'ตัน', 0)}**")
                if abs(qty_change) >= 3 and not prediction_adjusted:
                    prediction_parts.append(f"• การเปลี่ยนแปลง: **{format_percentage(qty_change, 1, True)}** (คาดการณ์: {format_quantity_with_unit(pred_qty, 'ตัน', 0)})")
            elif selected_date_only == today_date:
                prediction_parts.append(f"• ปริมาณวันนี้: **{format_quantity_with_unit(today_qty, 'ตัน', 0)}**")
                if not prediction_adjusted:
                    prediction_parts.append(f"• คาดการณ์วันพรุ่งนี้: **{format_quantity_with_unit(pred_qty, 'ตัน', 0)}**")
            else:
                prediction_parts.append(f"• ปริมาณในวันที่ {selected_date_only.strftime('%d/%m/%Y')}: **{format_quantity_with_unit(today_qty, 'ตัน', 0)}**")
                if not prediction_adjusted:
                    prediction_parts.append(f"• คาดการณ์วันถัดไป: **{format_quantity_with_unit(pred_qty, 'ตัน', 0)}**")
        else:
            prediction_parts.append(f"• ปริมาณวันนี้: **{format_quantity_with_unit(today_qty, 'ตัน', 0)}**")
            if not prediction_adjusted:
                prediction_parts.append(f"• คาดการณ์วันพรุ่งนี้: **{format_quantity_with_unit(pred_qty, 'ตัน', 0)}**")
        
        # ข้อมูลคุณภาพ - แสดงข้อมูลพื้นฐานเสมอ
        prediction_parts.append(f"\n**ข้อมูลคุณภาพ:**")
        if selected_date:
            today_date = datetime.now().date()
            selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
            
            if selected_date_only < today_date:
                prediction_parts.append(f"• คุณภาพในวันที่ {selected_date_only.strftime('%d/%m/%Y')}: **{format_percentage(today_qly, 1, False)}**")
                if abs(qly_change) >= 1.5 and not prediction_adjusted:
                    prediction_parts.append(f"• การเปลี่ยนแปลง: **{format_percentage(qly_change, 1, True)}** (คาดการณ์: {format_percentage(pred_qly, 1, False)})")
            elif selected_date_only == today_date:
                prediction_parts.append(f"• คุณภาพวันนี้: **{format_percentage(today_qly, 1, False)}**")
                if not prediction_adjusted:
                    prediction_parts.append(f"• คาดการณ์วันพรุ่งนี้: **{format_percentage(pred_qly, 1, False)}**")
            else:
                prediction_parts.append(f"• คุณภาพในวันที่ {selected_date_only.strftime('%d/%m/%Y')}: **{format_percentage(today_qly, 1, False)}**")
                if not prediction_adjusted:
                    prediction_parts.append(f"• คาดการณ์วันถัดไป: **{format_percentage(pred_qly, 1, False)}**")
        else:
            prediction_parts.append(f"• คุณภาพวันนี้: **{format_percentage(today_qly, 1, False)}**")
            if not prediction_adjusted:
                prediction_parts.append(f"• คาดการณ์วันพรุ่งนี้: **{format_percentage(pred_qly, 1, False)}**")
        
        # ข้อมูลเพิ่มเติม
        additional_info = []
        
        # ระดับความเชื่อมั่น
        confidence_level = "สูง" if not prediction_adjusted else "ปานกลาง"
        additional_info.append(f"**ระดับความเชื่อมั่น:** {confidence_level}")
        
        # หมายเหตุการปรับ
        if prediction_adjusted:
            additional_info.append("**หมายเหตุ:** การคาดการณ์นี้ถูกปรับให้สมเหตุสมผลตามหลักการวิเคราะห์ทางสถิติ")
        
        # ข้อมูลบริบท
        if selected_date:
            today_date = datetime.now().date()
            selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
            
            # ฤดูกาล
            current_month = selected_date_only.month
            if current_month in [12, 1, 2]:
                season_info = "ฤดูหีบอ้อย (ธ.ค.-ก.พ.)"
            elif current_month in [3, 4]:
                season_info = "ฤดูหีบอ้อย (มี.ค.-เม.ย.)"
            else:
                season_info = "นอกฤดูหีบอ้อย"
            additional_info.append(f"**ฤดูกาล:** {season_info}")
            
            # วันในสัปดาห์
            weekday_names = ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์']
            weekday = selected_date_only.weekday()
            weekday_name = weekday_names[weekday]
            additional_info.append(f"**วัน:** {weekday_name}")
        
        # แนวโน้มย้อนหลัง
        if trend_data.get('has_trend_data'):
            tons_trend = trend_data.get('tons_trend_percent', 0)
            fresh_trend = trend_data.get('fresh_trend_percent', 0)
            
            if abs(tons_trend) > 5 or abs(fresh_trend) > 2:
                additional_info.append(f"**แนวโน้ม 2 สัปดาห์:** ปริมาณ {format_percentage(tons_trend, 1, True)}, คุณภาพ {format_percentage(fresh_trend, 1, True)}")
        
        # เพิ่มข้อมูลเพิ่มเติม
        if additional_info:
            prediction_parts.append(f"\n**ข้อมูลเพิ่มเติม:**")
            prediction_parts.extend([f"• {info}" for info in additional_info])
        
        return "\n".join(prediction_parts)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return f"**AI ไม่สามารถสร้างคำคาดการณ์ได้ในขณะนี้**\n\nเนื่องจาก: {str(e)}"
        
        # เพิ่มข้อมูลเชิงปริมาณที่ชัดเจน
        if abs(qty_change) >= 3:
            if selected_date:
                today_date = datetime.now().date()
                selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
                
                if selected_date_only < today_date:
                    prediction_details += f" (ปริมาณในวันที่ {selected_date_only.strftime('%d/%m/%Y')}: {format_quantity_with_unit(today_qty, 'ตัน', 0)}, แนวโน้ม: {format_quantity_with_unit(pred_qty, 'ตัน', 0)})"
                elif selected_date_only == today_date:
                    prediction_details += f" (ปริมาณวันนี้: {format_quantity_with_unit(today_qty, 'ตัน', 0)}, คาดการณ์: {format_quantity_with_unit(pred_qty, 'ตัน', 0)})"
                else:
                    prediction_details += f" (ปริมาณในวันที่ {selected_date_only.strftime('%d/%m/%Y')}: {format_quantity_with_unit(today_qty, 'ตัน', 0)}, คาดการณ์: {format_quantity_with_unit(pred_qty, 'ตัน', 0)})"
            else:
                prediction_details += f" (ปริมาณวันนี้: {format_quantity_with_unit(today_qty, 'ตัน', 0)}, คาดการณ์: {format_quantity_with_unit(pred_qty, 'ตัน', 0)})"
        
        # เพิ่มข้อมูลคุณภาพหากมีการเปลี่ยนแปลง
        if abs(qly_change) >= 1.5:
            if selected_date:
                today_date = datetime.now().date()
                selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
                
                if selected_date_only < today_date:
                    prediction_details += f" (คุณภาพในวันที่ {selected_date_only.strftime('%d/%m/%Y')}: {format_percentage(today_qly, 1, False)}, แนวโน้ม: {format_percentage(pred_qly, 1, False)})"
                elif selected_date_only == today_date:
                    prediction_details += f" (คุณภาพวันนี้: {format_percentage(today_qly, 1, False)}, คาดการณ์: {format_percentage(pred_qly, 1, False)})"
                else:
                    prediction_details += f" (คุณภาพในวันที่ {selected_date_only.strftime('%d/%m/%Y')}: {format_percentage(today_qly, 1, False)}, คาดการณ์: {format_percentage(pred_qly, 1, False)})"
            else:
                prediction_details += f" (คุณภาพวันนี้: {format_percentage(today_qly, 1, False)}, คาดการณ์: {format_percentage(pred_qly, 1, False)})"
        
        # เพิ่มข้อมูลการวินิจฉัยหากการคาดการณ์ถูกปรับ
        if prediction_adjusted:
            if selected_date:
                today_date = datetime.now().date()
                selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
                
                if selected_date_only < today_date:
                    prediction_details += " [หมายเหตุ: การวิเคราะห์นี้ถูกปรับให้สมเหตุสมผลตามหลักการวิเคราะห์ทางสถิติ สำหรับข้อมูลย้อนหลัง]"
                else:
                    prediction_details += " [หมายเหตุ: การคาดการณ์นี้ถูกปรับให้สมเหตุสมผลตามหลักการวิเคราะห์ทางสถิติ]"
            else:
                prediction_details += " [หมายเหตุ: การคาดการณ์นี้ถูกปรับให้สมเหตุสมผลตามหลักการวิเคราะห์ทางสถิติ]"
        
        # เพิ่มระดับความเชื่อมั่นของการคาดการณ์
        confidence_level = "สูง" if not prediction_adjusted else "ปานกลาง"
        prediction_details += f" (ระดับความเชื่อมั่น: {confidence_level})"
        
        # เพิ่มข้อมูลเพิ่มเติมสำหรับการคาดการณ์
        if selected_date:
            today_date = datetime.now().date()
            selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
            
            if selected_date_only < today_date:
                prediction_details += f" [ข้อมูลย้อนหลัง: วันที่ {selected_date_only.strftime('%d/%m/%Y')}]"
            elif selected_date_only > today_date:
                prediction_details += f" [ข้อมูลอนาคต: วันที่ {selected_date_only.strftime('%d/%m/%Y')}]"
        
        # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
        if trend_data.get('has_trend_data'):
            tons_trend = trend_data.get('tons_trend_percent', 0)
            fresh_trend = trend_data.get('fresh_trend_percent', 0)
            
            if abs(tons_trend) > 5 or abs(fresh_trend) > 2:
                if selected_date:
                    today_date = datetime.now().date()
                    selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
                    
                    if selected_date_only < today_date:
                        prediction_details += f" [แนวโน้มย้อนหลัง: ปริมาณ {format_percentage(tons_trend, 1, True)}, คุณภาพ {format_percentage(fresh_trend, 1, True)} ใน 2 สัปดาห์ก่อนวันที่ {selected_date_only.strftime('%d/%m/%Y')}]"
                    else:
                        prediction_details += f" [แนวโน้ม: ปริมาณ {format_percentage(tons_trend, 1, True)}, คุณภาพ {format_percentage(fresh_trend, 1, True)} ใน 2 สัปดาห์ที่ผ่านมา]"
                else:
                    prediction_details += f" [แนวโน้ม: ปริมาณ {format_percentage(tons_trend, 1, True)}, คุณภาพ {format_percentage(fresh_trend, 1, True)} ใน 2 สัปดาห์ที่ผ่านมา]"
        
        # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลเฉลี่ย
        if stats.get('has_comparison_data'):
            avg_daily_tons = stats.get('avg_daily_tons', 0)
            if avg_daily_tons > 0 and today_qty > 0:
                daily_performance = ((today_qty - avg_daily_tons) / avg_daily_tons * 100)
                if abs(daily_performance) > 10:
                    if selected_date:
                        today_date = datetime.now().date()
                        selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
                        
                        if selected_date_only < today_date:
                            prediction_details += f" [ประสิทธิภาพ: {format_percentage(daily_performance, 1, True)} เทียบกับค่าเฉลี่ยย้อนหลัง]"
                        else:
                            prediction_details += f" [ประสิทธิภาพ: {format_percentage(daily_performance, 1, True)} เทียบกับค่าเฉลี่ยย้อนหลัง]"
                    else:
                        prediction_details += f" [ประสิทธิภาพ: {format_percentage(daily_performance, 1, True)} เทียบกับค่าเฉลี่ยย้อนหลัง]"
        
        # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลฤดูกาล
        if selected_date:
            today_date = datetime.now().date()
            selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
            
            # ตรวจสอบฤดูกาล
            current_month = selected_date_only.month
            if current_month in [12, 1, 2]:
                season_info = "ฤดูหีบอ้อย (ธ.ค.-ก.พ.)"
            elif current_month in [3, 4]:
                season_info = "ฤดูหีบอ้อย (มี.ค.-เม.ย.)"
            else:
                season_info = "นอกฤดูหีบอ้อย"
            
            prediction_details += f" [ฤดูกาล: {season_info}]"
            
            # เพิ่มข้อมูลวันในสัปดาห์
            weekday_names = ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์']
            weekday = selected_date_only.weekday()
            weekday_name = weekday_names[weekday]
            prediction_details += f" [วัน: {weekday_name}]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if selected_date_only < today_date:
                days_diff = (today_date - selected_date_only).days
                if days_diff == 1:
                    prediction_details += " [ข้อมูลเมื่อวาน]"
                elif days_diff <= 7:
                    prediction_details += f" [ข้อมูล {days_diff} วันที่ผ่านมา]"
                elif days_diff <= 30:
                    prediction_details += f" [ข้อมูล {days_diff} วันที่ผ่านมา]"
                else:
                    prediction_details += f" [ข้อมูล {days_diff} วันที่ผ่านมา]"
            elif selected_date_only > today_date:
                days_diff = (selected_date_only - today_date).days
                if days_diff == 1:
                    prediction_details += " [ข้อมูลวันพรุ่งนี้]"
                elif days_diff <= 7:
                    prediction_details += f" [ข้อมูล {days_diff} วันข้างหน้า]"
                elif days_diff <= 30:
                    prediction_details += f" [ข้อมูล {days_diff} วันข้างหน้า]"
                else:
                    prediction_details += f" [ข้อมูล {days_diff} วันข้างหน้า]"
            else:
                prediction_details += " [ข้อมูลวันนี้]"
        
        # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
        if selected_date:
            today_date = datetime.now().date()
            selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
            
            # ตรวจสอบข้อมูลการเปรียบเทียบ
            if stats.get('has_comparison_data'):
                comparison_days = stats.get('comparison_period_days', 7)
                if comparison_days > 0:
                    if selected_date_only < today_date:
                        prediction_details += f" [ข้อมูลเปรียบเทียบ: {comparison_days} วันก่อนวันที่ {selected_date_only.strftime('%d/%m/%Y')}]"
                    else:
                        prediction_details += f" [ข้อมูลเปรียบเทียบ: {comparison_days} วันก่อนวันที่เลือก]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลฤดูกาล
            if stats.get('total_cane_all', 0) > 0:
                season_progress = (today_qty / stats.get('total_cane_all', 1)) * 100
                if season_progress > 0:
                    prediction_details += f" [ความคืบหน้าฤดูกาล: {format_percentage(season_progress, 1, False)}]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if stats.get('avg_daily_tons', 0) > 0:
                daily_variance = ((today_qty - stats.get('avg_daily_tons', 0)) / stats.get('avg_daily_tons', 1)) * 100
                if abs(daily_variance) > 15:
                    if daily_variance > 0:
                        prediction_details += f" [สูงกว่าค่าเฉลี่ย: {format_percentage(daily_variance, 1, False)}]"
                    else:
                        prediction_details += f" [ต่ำกว่าค่าเฉลี่ย: {format_percentage(abs(daily_variance), 1, False)}]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if stats.get('type_1_percent', 0) > 0:
                quality_variance = today_qly - stats.get('type_1_percent', 0)
                if abs(quality_variance) > 5:
                    if quality_variance > 0:
                        prediction_details += f" [คุณภาพดีขึ้น: {format_percentage(quality_variance, 1, False)}]"
                    else:
                        prediction_details += f" [คุณภาพลดลง: {format_percentage(abs(quality_variance), 1, False)}]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if stats.get('perf_index', 0) > 0:
                performance_score = stats.get('perf_index', 0)
                if performance_score >= 8:
                    prediction_details += " [ประสิทธิภาพ: ดีเยี่ยม]"
                elif performance_score >= 6:
                    prediction_details += " [ประสิทธิภาพ: ดี]"
                elif performance_score >= 4:
                    prediction_details += " [ประสิทธิภาพ: ปานกลาง]"
                else:
                    prediction_details += " [ประสิทธิภาพ: ต้องปรับปรุง]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if exec_summary.get('hours_processed', 0) > 0:
                hours_processed = exec_summary.get('hours_processed', 0)
                if hours_processed < 6:
                    prediction_details += " [ข้อมูลเบื้องต้น]"
                elif hours_processed < 12:
                    prediction_details += " [ข้อมูลครึ่งวัน]"
                else:
                    prediction_details += " [ข้อมูลเต็มวัน]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if trend_data.get('has_trend_data'):
                trend_strength = abs(trend_data.get('tons_trend_percent', 0)) + abs(trend_data.get('fresh_trend_percent', 0))
                if trend_strength > 15:
                    prediction_details += " [แนวโน้มชัดเจน]"
                elif trend_strength > 8:
                    prediction_details += " [แนวโน้มปานกลาง]"
                else:
                    prediction_details += " [แนวโน้มไม่ชัดเจน]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if selected_date_only < today_date:
                # สำหรับข้อมูลย้อนหลัง ให้เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลปัจจุบัน
                current_date = today_date
                days_since = (current_date - selected_date_only).days
                if days_since <= 7:
                    prediction_details += f" [ข้อมูลล่าสุด: {days_since} วันที่ผ่านมา]"
                elif days_since <= 30:
                    prediction_details += f" [ข้อมูลล่าสุด: {days_since} วันที่ผ่านมา]"
                else:
                    prediction_details += f" [ข้อมูลล่าสุด: {days_since} วันที่ผ่านมา]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if selected_date_only > today_date:
                # สำหรับข้อมูลในอนาคต ให้เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลปัจจุบัน
                current_date = today_date
                days_until = (selected_date_only - current_date).days
                if days_until <= 7:
                    prediction_details += f" [ข้อมูลในอนาคต: {days_until} วันข้างหน้า]"
                elif days_until <= 30:
                    prediction_details += f" [ข้อมูลในอนาคต: {days_until} วันข้างหน้า]"
                else:
                    prediction_details += f" [ข้อมูลในอนาคต: {days_until} วันข้างหน้า]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if selected_date_only == today_date:
                # สำหรับข้อมูลวันนี้ ให้เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลปัจจุบัน
                prediction_details += " [ข้อมูลปัจจุบัน]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if selected_date:
                # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
                if stats.get('has_comparison_data'):
                    comparison_days = stats.get('comparison_period_days', 7)
                    if comparison_days > 0:
                        if selected_date_only < today_date:
                            prediction_details += f" [ข้อมูลเปรียบเทียบ: {comparison_days} วันก่อนวันที่ {selected_date_only.strftime('%d/%m/%Y')}]"
                        else:
                            prediction_details += f" [ข้อมูลเปรียบเทียบ: {comparison_days} วันก่อนวันที่เลือก]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if selected_date:
                # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
                if stats.get('has_comparison_data'):
                    comparison_days = stats.get('comparison_period_days', 7)
                    if comparison_days > 0:
                        if selected_date_only < today_date:
                            prediction_details += f" [ข้อมูลเปรียบเทียบ: {comparison_days} วันก่อนวันที่ {selected_date_only.strftime('%d/%m/%Y')}]"
                        else:
                            prediction_details += f" [ข้อมูลเปรียบเทียบ: {comparison_days} วันก่อนวันที่เลือก]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if selected_date:
                # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
                if stats.get('has_comparison_data'):
                    comparison_days = stats.get('comparison_period_days', 7)
                    if comparison_days > 0:
                        if selected_date_only < today_date:
                            prediction_details += f" [ข้อมูลเปรียบเทียบ: {comparison_days} วันก่อนวันที่ {selected_date_only.strftime('%d/%m/%Y')}]"
                        else:
                            prediction_details += f" [ข้อมูลเปรียบเทียบ: {comparison_days} วันก่อนวันที่เลือก]"
            
            # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
            if selected_date:
                # เพิ่มข้อมูลการเปรียบเทียบกับข้อมูลย้อนหลัง
                if stats.get('has_comparison_data'):
                    comparison_days = stats.get('comparison_period_days', 7)
                    if comparison_days > 0:
                        if selected_date_only < today_date:
                            prediction_details += f" [ข้อมูลเปรียบเทียบ: {comparison_days} วันก่อนวันที่ {selected_date_only.strftime('%d/%m/%Y')}]"
                        else:
                            prediction_details += f" [ข้อมูลเปรียบเทียบ: {comparison_days} วันก่อนวันที่เลือก]"
        
        return prediction_details
    except Exception as e:
        print(f"Prediction error: {e}")
        return f"AI ไม่สามารถสร้างคำคาดการณ์ได้ในขณะนี้ เนื่องจาก: {str(e)}"

def _get_trend_context(trend_data):
    if not trend_data.get('has_trend_data'): return 0, ""
    tons_trend, fresh_trend = trend_data.get('tons_trend_percent', 0), trend_data.get('fresh_trend_percent', 0)
    score = (1 if tons_trend > 5 else -1 if tons_trend < -5 else 0) + (1 if fresh_trend > 2 else -1 if fresh_trend < -2 else 0)
    tons_icon, fresh_icon = ('text-success' if tons_trend >= 0 else 'text-danger'), ('text-success' if fresh_trend >= 0 else 'text-danger')
    return score, f"""<li class="mt-2 pt-2 border-top border-white border-opacity-10"><i class="bi bi-graph-up-arrow me-2"></i><strong>แนวโน้ม 2 สัปดาห์:</strong><ul class="list-unstyled mt-1 ps-3"><li><i class="bi {'bi-arrow-up-right-circle-fill' if tons_trend >= 0 else 'bi-arrow-down-right-circle-fill'} {tons_icon} me-1 small"></i>ปริมาณ: <span class="{tons_icon}">{format_percentage(tons_trend, 1, True)}</span></li><li><i class="bi {'bi-arrow-up-right-circle-fill' if fresh_trend >= 0 else 'bi-arrow-down-right-circle-fill'} {fresh_icon} me-1 small"></i>คุณภาพ: <span class="{fresh_icon}">{format_percentage(fresh_trend, 1, True)}</span></li></ul></li>"""

def _calculate_efficiency_metrics(stats, exec_summary, trend_data):
    metrics = {}
    
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    hours_processed = exec_summary.get('hours_processed', 0)
    
    # If no data today, return empty metrics
    if today_total <= 0:
        return {
            'volume_efficiency': 0,
            'volume_status': 'ไม่มีข้อมูล',
            'hourly_efficiency': 0,
            'hourly_status': 'ไม่มีข้อมูล',
            'quality_efficiency': 0,
            'quality_status': 'ไม่มีข้อมูล',
            'stability_status': 'ไม่มีข้อมูล'
        }
    
    if avg_total > 0:
        volume_efficiency = (today_total / avg_total) * 100
        metrics['volume_efficiency'] = volume_efficiency
        metrics['volume_status'] = 'สูง' if volume_efficiency > 110 else 'ปกติ' if volume_efficiency > 90 else 'ต่ำ'
    
    if hours_processed > 0 and avg_total > 0:
        expected_hourly = avg_total / 24
        actual_hourly = today_total / hours_processed
        hourly_efficiency = (actual_hourly / expected_hourly) * 100 if expected_hourly > 0 else 100
        metrics['hourly_efficiency'] = hourly_efficiency
        metrics['hourly_status'] = 'สูง' if hourly_efficiency > 120 else 'ปกติ' if hourly_efficiency > 80 else 'ต่ำ'
    
    today_fresh = stats.get('type_1_percent', 0)
    avg_fresh = stats.get('avg_fresh_percent', 0)
    if avg_fresh > 0:
        quality_efficiency = (today_fresh / avg_fresh) * 100
        metrics['quality_efficiency'] = quality_efficiency
        metrics['quality_status'] = 'สูง' if quality_efficiency > 105 else 'ปกติ' if quality_efficiency > 95 else 'ต่ำ'
    
    peak_hour_tons = exec_summary.get('peak_hour_tons', 0)
    if hours_processed > 0 and today_total > 0:
        avg_hourly = today_total / hours_processed
        if avg_hourly > 0:
            peak_ratio = peak_hour_tons / avg_hourly
            metrics['peak_ratio'] = peak_ratio
            metrics['stability_status'] = 'เสถียร' if peak_ratio < 1.5 else 'ปานกลาง' if peak_ratio < 2.0 else 'ผันผวน'
    
    if trend_data.get('has_trend_data'):
        tons_trend = trend_data.get('tons_trend_percent', 0)
        fresh_trend = trend_data.get('fresh_trend_percent', 0)
        metrics['trend_direction'] = 'ขึ้น' if tons_trend > 2 else 'ลง' if tons_trend < -2 else 'ทรงตัว'
        metrics['quality_trend'] = 'ขึ้น' if fresh_trend > 1 else 'ลง' if fresh_trend < -1 else 'ทรงตัว'
    
    return metrics

def _generate_real_time_alerts(stats, exec_summary, efficiency_metrics, trend_data):
    alerts = []
    
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    hours_processed = exec_summary.get('hours_processed', 0)
    
    if avg_total > 0:
        volume_ratio = today_total / avg_total
        if volume_ratio < 0.5 and hours_processed >= 6:
            alerts.append({
                'type': 'critical',
                'icon': 'bi-exclamation-triangle-fill',
                'color': 'text-danger',
                'title': 'ปริมาณอ้อยต่ำมาก',
                'message': f'ปริมาณอ้อยวันนี้ต่ำกว่าเป้าหมายมาก ({format_percentage(volume_ratio*100, 1, False)} ของค่าเฉลี่ย)',
                'action': 'ควรตรวจสอบปัญหาการขนส่งหรือการตัดอ้อยทันที'
            })
        elif volume_ratio < 0.7 and hours_processed >= 8:
            alerts.append({
                'type': 'warning',
                'icon': 'bi-exclamation-circle-fill',
                'color': 'text-warning',
                'title': 'ปริมาณอ้อยต่ำกว่าเป้าหมาย',
                'message': f'ปริมาณอ้อยวันนี้ต่ำกว่าเป้าหมาย ({format_percentage(volume_ratio*100, 1, False)} ของค่าเฉลี่ย)',
                'action': 'ควรประสานงานกับทีมขนส่งเพื่อเพิ่มปริมาณ'
            })
    
    today_fresh = stats.get('type_1_percent', 0)
    avg_fresh = stats.get('avg_fresh_percent', 0)
    
    if avg_fresh > 0 and today_fresh < avg_fresh * 0.8:
        alerts.append({
            'type': 'warning',
            'icon': 'bi-gem',
            'color': 'text-warning',
            'title': 'คุณภาพอ้อยลดลง',
            'message': f'คุณภาพอ้อยสดลดลงจากค่าเฉลี่ย ({today_fresh:.1f}% vs {avg_fresh:.1f}%)',
            'action': 'ควรตรวจสอบการจัดการอ้อยไฟและกระบวนการตรวจรับ'
        })
    
    if 'volume_efficiency' in efficiency_metrics:
        vol_eff = efficiency_metrics['volume_efficiency']
        if vol_eff < 70:
            alerts.append({
                'type': 'critical',
                'icon': 'bi-speedometer2',
                'color': 'text-danger',
                'title': 'ประสิทธิภาพการทำงานต่ำ',
                'message': f'ประสิทธิภาพการรับอ้อยต่ำมาก ({vol_eff:.1f}%)',
                'action': 'ควรวิเคราะห์หาสาเหตุและแก้ไขโดยเร่งด่วน'
            })
    
    if trend_data.get('has_trend_data'):
        tons_trend = trend_data.get('tons_trend_percent', 0)
        if tons_trend < -10:
            alerts.append({
                'type': 'warning',
                'icon': 'bi-graph-down-arrow',
                'color': 'text-warning',
                'title': 'แนวโน้มปริมาณลดลง',
                'message': f'แนวโน้มปริมาณอ้อยลดลงอย่างต่อเนื่อง ({tons_trend:.1f}%)',
                'action': 'ควรวิเคราะห์แนวโน้มและวางแผนแก้ไข'
            })
    
    if hours_processed <= 4 and today_total < avg_total * 0.2:
        alerts.append({
            'type': 'info',
            'icon': 'bi-clock-history',
            'color': 'text-info',
            'title': 'การเริ่มต้นวันช้า',
            'message': 'การเริ่มต้นรับอ้อยในวันนี้ช้ากว่าเป้าหมายที่ควรจะเป็น',
            'action': 'ควรเตรียมพร้อมสำหรับปริมาณที่อาจเพิ่มขึ้นอย่างรวดเร็วในช่วงบ่าย'
        })
    
    return alerts

def _generate_ai_enhanced_alerts(stats, exec_summary, efficiency_metrics, trend_data, patterns):
    """AI-enhanced alert system with pattern recognition and predictive alerts."""
    alerts = _generate_real_time_alerts(stats, exec_summary, efficiency_metrics, trend_data)
    
    # AI Pattern-based alerts
    anomaly_score = patterns.get('anomaly_score', 0.0)
    if anomaly_score > 0.8:
        alerts.append({
            'type': 'critical',
            'icon': 'bi-shield-exclamation',
            'color': 'text-danger',
            'title': 'AI ตรวจพบความผิดปกติสูง',
            'message': f'ระบบ AI ตรวจพบความผิดปกติในระดับสูงมาก (คะแนน: {anomaly_score:.2f})',
            'action': 'จำเป็นต้องมีการตรวจสอบและแก้ไขอย่างเร่งด่วน',
            'ai_generated': True
        })
    elif anomaly_score > 0.6:
        alerts.append({
            'type': 'warning',
            'icon': 'bi-shield-check',
            'color': 'text-warning',
            'title': 'AI ตรวจพบความผิดปกติ',
            'message': f'ระบบ AI ตรวจพบความผิดปกติในระดับปานกลาง (คะแนน: {anomaly_score:.2f})',
            'action': 'ควรติดตามสถานการณ์อย่างใกล้ชิด',
            'ai_generated': True
        })
    
    # Predictive alerts based on patterns
    predictive_signals = patterns.get('predictive_signals', [])
    if "high_risk_alert" in predictive_signals:
        alerts.append({
            'type': 'warning',
            'icon': 'bi-graph-up-arrow',
            'color': 'text-warning',
            'title': 'AI คาดการณ์ความเสี่ยงสูง',
            'message': 'ระบบ AI คาดการณ์ว่าอาจเกิดปัญหาสำคัญในอนาคตอันใกล้',
            'action': 'ควรเตรียมแผนการป้องกันและแก้ไขล่วงหน้า',
            'ai_generated': True
        })
    
    if "seasonal_quality_decline" in predictive_signals:
        alerts.append({
            'type': 'info',
            'icon': 'bi-calendar-event',
            'color': 'text-info',
            'title': 'AI ตรวจพบรูปแบบฤดูกาล',
            'message': 'ระบบ AI ตรวจพบแนวโน้มคุณภาพที่ลดลงตามฤดูกาล',
            'action': 'ควรมีการวางแผนการจัดการคุณภาพสำหรับช่วงปลายฤดู',
            'ai_generated': True
        })
    
    # Efficiency-based AI alerts
    if 'volume_efficiency' in efficiency_metrics:
        vol_eff = efficiency_metrics['volume_efficiency']
        if vol_eff > 130:
            alerts.append({
                'type': 'success',
                'icon': 'bi-award',
                'color': 'text-success',
                'title': 'AI ตรวจพบประสิทธิภาพยอดเยี่ยม',
                'message': f'ประสิทธิภาพการรับอ้อยสูงมาก ({vol_eff:.1f}%) สะท้อนถึงการทำงานที่มีประสิทธิภาพ',
                'action': 'ควรถอดบทเรียนความสำเร็จและนำไปประยุกต์ใช้',
                'ai_generated': True
            })
    
    # Seasonal pattern alerts
    seasonal_pattern = patterns.get('seasonal_pattern')
    if seasonal_pattern == "late_season":
        alerts.append({
            'type': 'info',
            'icon': 'bi-calendar-range',
            'color': 'text-info',
            'title': 'AI ตรวจพบช่วงปลายฤดู',
            'message': 'เข้าสู่ช่วงปลายฤดูหีบแล้ว ควรมีการวางแผนการจัดการอ้อยที่เหลือ',
            'action': 'ควรประเมินปริมาณอ้อยที่เหลือและวางแผนการขนส่ง',
            'ai_generated': True
        })
    
    return alerts

def _calculate_performance_benchmarks(stats, exec_summary, trend_data):
    benchmarks = {}
    
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    hours_processed = exec_summary.get('hours_processed', 0)
    
    if avg_total > 0:
        benchmarks['daily_target'] = avg_total
        benchmarks['daily_progress'] = (today_total / avg_total) * 100
        benchmarks['remaining_target'] = max(0, avg_total - today_total)
        
        if hours_processed > 0:
            current_hourly_rate = today_total / hours_processed
            target_hourly_rate = avg_total / 24
            benchmarks['hourly_target'] = target_hourly_rate
            benchmarks['current_hourly_rate'] = current_hourly_rate
            benchmarks['hourly_efficiency'] = (current_hourly_rate / target_hourly_rate) * 100 if target_hourly_rate > 0 else 0
    
    today_fresh = stats.get('type_1_percent', 0)
    avg_fresh = stats.get('avg_fresh_percent', 0)
    if avg_fresh > 0:
        benchmarks['quality_target'] = avg_fresh
        benchmarks['quality_current'] = today_fresh
        benchmarks['quality_gap'] = today_fresh - avg_fresh
    
    if trend_data.get('has_trend_data'):
        tons_trend = trend_data.get('tons_trend_percent', 0)
        benchmarks['trend_adjusted_target'] = avg_total * (1 + tons_trend / 100)
        benchmarks['trend_adjustment'] = tons_trend
    
    return benchmarks

def _generate_operational_recommendations(stats, exec_summary, efficiency_metrics, alerts, persona_name):
    recommendations = []
    persona = PERSONAS.get(persona_name, {})
    
    op_focus = persona.get("operational_focus", [])
    
    if "การเพิ่มปริมาณอ้อย" in op_focus or "การแก้ไขจุดที่งานติดขัด" in op_focus:
        recommendations.append({
            'priority': 'high',
            'category': 'ปริมาณ (Volume)',
            'title': 'เร่งรัดปริมาณอ้อยเข้าสู่เป้าหมาย',
            'description': 'ปริมาณอ้อยปัจจุบันยังต่ำกว่าเป้าหมาย ควรมีการประสานงานเพื่อเพิ่มอัตราการส่งมอบ',
            'actions': [
                'ตรวจสอบสถานะรถขนส่งที่ยังไม่เข้าโรงงาน',
                'สื่อสารกับทีมส่งเสริมเพื่อแก้ปัญหาหน้าไร่ (ถ้ามี)',
                'พิจารณาปรับแผนการจัดคิวรถเพื่อลดเวลารอ'
            ]
        })

    if "การปรับปรุงคุณภาพอ้อย" in op_focus:
        recommendations.append({
            'priority': 'high',
            'category': 'คุณภาพ (Quality)',
            'title': 'ยกระดับการควบคุมคุณภาพอ้อย',
            'description': 'สัดส่วนอ้อยสดลดลง ซึ่งอาจส่งผลต่อ CCS ควรมีมาตรการควบคุมที่เข้มข้นขึ้น',
            'actions': [
                'ทบทวนเกณฑ์การตรวจรับคุณภาพที่หน้าโรงงาน',
                'สื่อสารย้ำเรื่องการลดอ้อยไฟและสิ่งปนเปื้อนไปยังชาวไร่',
                'วิเคราะห์ข้อมูลอ้อยจากโซนต่างๆ เพื่อหาพื้นที่ที่คุณภาพต่ำ'
            ]
        })
        
    if "การสร้างความสม่ำเสมอในการรับอ้อย" in op_focus:
        recommendations.append({
            'priority': 'medium',
            'category': 'ความเสถียร (Stability)',
            'title': 'ลดความผันผวนในการรับอ้อย',
            'description': 'การรับอ้อยไม่สม่ำเสมอทำให้การบริหารจัดการหน้างานยาก และลดประสิทธิภาพเครื่องจักร',
            'actions': [
                'วิเคราะห์ข้อมูลรายชั่วโมงเพื่อหาช่วงเวลาที่เกิดปัญหา',
                'ทำงานร่วมกับทีมขนส่งเพื่อวางแผนการเข้าส่งอ้อยที่กระจายตัวมากขึ้น',
                'ปรับปรุงระบบการเรียกคิวรถให้มีความยืดหยุ่น'
            ]
        })

    if not recommendations:
         recommendations.append({
            'priority': 'low',
            'category': 'การพัฒนาต่อเนื่อง (Improvement)',
            'title': 'รักษามาตรฐานและมองหาโอกาสพัฒนา',
            'description': 'การดำเนินงานวันนี้เป็นไปอย่างราบรื่น ควรใช้โอกาสนี้ในการหาจุดพัฒนาเล็กๆ น้อยๆ',
            'actions': [
                'ประชุมทีมสั้นๆ เพื่อสรุปการทำงานและแลกเปลี่ยนความคิดเห็น',
                'ตรวจสอบรายงานและมองหาแนวโน้มที่น่าสนใจ',
                'วางแผนการทำงานสำหรับวันพรุ่งนี้ให้มีประสิทธิภาพเช่นเดิม'
            ]
        })

    return recommendations

def _generate_chatgpt_style_recommendations(stats, exec_summary, efficiency_metrics, alerts, persona_name, patterns, selected_date=None, analysis_mode: str = None) -> str:
    """
    Generate ChatGPT-style recommendations with human-like intelligence and empathy
    """
    # Get time-based context
    if selected_date is None:
        selected_date = datetime.now()
    time_context = _get_time_based_context(selected_date)
    
    # Extract key metrics
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    today_fresh_pct = stats.get('type_1_percent', 0)
    avg_fresh_pct = stats.get('avg_fresh_percent', 0)
    anomaly_score = patterns.get('anomaly_score', 0.0)
    
    # Start with empathetic opening
    recommendation_parts = []
    
    if analysis_mode == 'historical':
        recommendation_parts.append("**📚 ข้อเสนอแนะจากข้อมูลย้อนหลัง**\n\nจากการวิเคราะห์ข้อมูลในวันนั้น ผมมีข้อเสนอแนะดังนี้ครับ:")
    else:
        recommendation_parts.append("**💡 ข้อเสนอแนะและแนวทางปฏิบัติ**\n\nจากการวิเคราะห์ข้อมูลล่าสุด ผมมีข้อเสนอแนะดังนี้ครับ:")
    
    # Generate contextual recommendations based on persona and data
    if persona_name == "EXCELLENT":
        recommendation_parts.append("\n**🎉 สิ่งที่ทำได้ดีแล้ว:**")
        recommendation_parts.append("• การจัดการปริมาณและคุณภาพอยู่ในระดับยอดเยี่ยม")
        recommendation_parts.append("• การวางแผนการขนส่งมีประสิทธิภาพสูง")
        recommendation_parts.append("\n**🚀 แนวทางพัฒนาต่อ:**")
        recommendation_parts.append("• รักษามาตรฐานนี้ไว้และหาวิธีเพิ่มประสิทธิภาพให้มากขึ้น")
        recommendation_parts.append("• ศึกษาแนวทางที่ทำให้ได้ผลลัพธ์ดีเพื่อนำไปใช้ในอนาคต")
        recommendation_parts.append("• เตรียมแผนสำรองสำหรับช่วงที่อาจมีปัญหา")
        
    elif persona_name == "CRITICAL":
        recommendation_parts.append("\n**🚨 การดำเนินการเร่งด่วน:**")
        recommendation_parts.append("• เรียกประชุมทีมทันทีเพื่อวิเคราะห์สาเหตุหลัก")
        recommendation_parts.append("• ตรวจสอบระบบขนส่งและเครื่องจักรอย่างละเอียด")
        recommendation_parts.append("• ติดต่อผู้รับผิดชอบในแต่ละโซนเพื่อหาข้อมูลเพิ่มเติม")
        recommendation_parts.append("\n**🔧 แนวทางแก้ไขระยะสั้น:**")
        recommendation_parts.append("• ปรับแผนการขนส่งให้เหมาะสมกับสถานการณ์")
        recommendation_parts.append("• เพิ่มการตรวจสอบคุณภาพอ้อยที่เข้มงวดขึ้น")
        recommendation_parts.append("• จัดเตรียมกำลังคนและเครื่องจักรสำรอง")
        
    elif persona_name == "QUANTITY_PUSH":
        recommendation_parts.append("\n**⚖️ การปรับสมดุล:**")
        recommendation_parts.append("• วิเคราะห์สาเหตุที่คุณภาพลดลงเมื่อปริมาณเพิ่ม")
        recommendation_parts.append("• ปรับปรุงระบบการจัดการคิวรถให้มีประสิทธิภาพมากขึ้น")
        recommendation_parts.append("• เพิ่มการตรวจสอบคุณภาพระหว่างการขนส่ง")
        recommendation_parts.append("\n**📈 แนวทางปรับปรุง:**")
        recommendation_parts.append("• วางแผนการขนส่งให้กระจายตลอดวัน")
        recommendation_parts.append("• จัดอบรมทีมงานเรื่องการรักษาคุณภาพอ้อย")
        
    elif persona_name == "QUALITY_FOCUS":
        recommendation_parts.append("\n**🌟 การเพิ่มปริมาณ:**")
        recommendation_parts.append("• วิเคราะห์สาเหตุที่ปริมาณต่ำกว่าเป้า")
        recommendation_parts.append("• ตรวจสอบการขนส่งจากไร่ในแต่ละโซน")
        recommendation_parts.append("• ปรับปรุงระบบการจัดการคิวรถ")
        recommendation_parts.append("\n**🎯 แนวทางแก้ไข:**")
        recommendation_parts.append("• ติดต่อเกษตรกรเพื่อประสานการขนส่ง")
        recommendation_parts.append("• เพิ่มจำนวนรถขนส่งในช่วงที่จำเป็น")
        
    elif persona_name == "VOLATILE_PERFORMANCE":
        recommendation_parts.append("\n**📊 การลดความผันผวน:**")
        recommendation_parts.append("• วิเคราะห์สาเหตุของความผันผวนในการรับอ้อย")
        recommendation_parts.append("• ปรับปรุงระบบการจัดการคิวรถให้สม่ำเสมอ")
        recommendation_parts.append("• จัดทำแผนการขนส่งที่กระจายตลอดวัน")
        recommendation_parts.append("\n**⚡ แนวทางปรับปรุง:**")
        recommendation_parts.append("• สื่อสารกับเกษตรกรเรื่องเวลาการขนส่ง")
        recommendation_parts.append("• เพิ่มการตรวจสอบและควบคุมการไหลเข้าของอ้อย")
        
    else:  # Default case
        recommendation_parts.append("\n**📋 แนวทางทั่วไป:**")
        recommendation_parts.append("• รักษามาตรฐานการทำงานปัจจุบันไว้")
        recommendation_parts.append("• ติดตามข้อมูลอย่างต่อเนื่องเพื่อปรับปรุง")
        recommendation_parts.append("• เตรียมแผนสำรองสำหรับสถานการณ์ต่างๆ")
    
    # Add time-based recommendations for current mode
    if analysis_mode != 'historical':
        if time_context['time_of_day'] == 'morning':
            recommendation_parts.append("\n**🌅 คำแนะนำช่วงเช้า:**")
            recommendation_parts.append("• ตรวจสอบการเตรียมความพร้อมของเครื่องจักร")
            recommendation_parts.append("• ประสานงานกับทีมขนส่งเพื่อวางแผนวัน")
        elif time_context['time_of_day'] == 'afternoon':
            recommendation_parts.append("\n**☀️ คำแนะนำช่วงบ่าย:**")
            recommendation_parts.append("• ติดตามผลการดำเนินงานและปรับแผนตามจำเป็น")
            recommendation_parts.append("• เตรียมความพร้อมสำหรับช่วงเย็น")
        elif time_context['time_of_day'] == 'evening':
            recommendation_parts.append("\n**🌆 คำแนะนำช่วงเย็น:**")
            recommendation_parts.append("• สรุปผลการดำเนินงานและวางแผนวันถัดไป")
            recommendation_parts.append("• ตรวจสอบการบำรุงรักษาเครื่องจักร")
    
    # Add closing with empathy
    if analysis_mode == 'historical':
        recommendation_parts.append("\n**💭 สรุป:**")
        recommendation_parts.append("ข้อเสนอแนะเหล่านี้มาจากการวิเคราะห์ข้อมูลในอดีต หวังว่าจะเป็นประโยชน์ในการปรับปรุงการทำงานในอนาคตครับ")
    else:
        recommendation_parts.append("\n**🤝 สรุป:**")
        recommendation_parts.append("ข้อเสนอแนะเหล่านี้มาจากการวิเคราะห์ข้อมูลล่าสุด หากต้องการคำแนะนำเพิ่มเติมหรือมีคำถาม สามารถสอบถามได้เสมอครับ ผมยินดีช่วยเหลือ")
    
    return "\n".join(recommendation_parts)

def _generate_ai_enhanced_recommendations(stats, exec_summary, efficiency_metrics, alerts, persona_name, patterns, selected_date=None, analysis_mode: str = None):
    """AI-enhanced recommendation engine with contextual intelligence."""
    recommendations = []
    
    # Get time-based context
    if selected_date is None:
        selected_date = datetime.now()
    time_context = _get_time_based_context(selected_date)
    
    # Base recommendations from persona
    base_recs = _generate_operational_recommendations(stats, exec_summary, efficiency_metrics, alerts, persona_name)
    recommendations.extend(base_recs)
    
    # Adaptive recommendations based on time context (only for current mode)
    if analysis_mode != 'historical':
        adaptive_recs = _generate_adaptive_recommendations(persona_name, stats, time_context, efficiency_metrics)
        for rec in adaptive_recs:
            recommendations.append({
                'priority': 'medium',
                'category': 'การปรับตัวตามเวลา (Time-based Adaptation)',
                'title': 'คำแนะนำตามช่วงเวลา',
                'description': rec,
                'actions': [
                    'ปรับแผนการทำงานตามคำแนะนำ',
                    'ประสานงานกับทีมที่เกี่ยวข้อง',
                    'ติดตามผลการดำเนินการ'
                ],
                'ai_generated': True,
                'time_based': True
            })
    
    # AI-enhanced contextual recommendations
    anomaly_score = patterns.get('anomaly_score', 0.0)
    seasonal_pattern = patterns.get('seasonal_pattern')
    weekly_pattern = patterns.get('weekly_pattern')
    quality_pattern = patterns.get('quality_pattern')
    
    # High anomaly score recommendations
    if anomaly_score > 0.7:
        recommendations.append({
            'priority': 'critical',
            'category': 'การจัดการความเสี่ยง (Risk Management)',
            'title': 'การจัดการสถานการณ์วิกฤต',
            'description': f'ระบบตรวจพบความผิดปกติในระดับสูง (คะแนน: {anomaly_score:.2f}) จำเป็นต้องมีการดำเนินการอย่างเร่งด่วน',
            'actions': [
                'เรียกประชุมฉุกเฉินกับทีมที่เกี่ยวข้องทันที',
                'ตรวจสอบข้อมูลย้อนหลังเพื่อหาสาเหตุของความผิดปกติ',
                'เตรียมแผนการแก้ไขฉุกเฉินและแผนสำรอง',
                'สื่อสารสถานการณ์ไปยังผู้บริหารระดับสูง'
            ],
            'ai_generated': True
        })
    
    # Seasonal pattern recommendations
    if seasonal_pattern == "late_season":
        recommendations.append({
            'priority': 'medium',
            'category': 'การวางแผนฤดูกาล (Seasonal Planning)',
            'title': 'การจัดการช่วงปลายฤดูหีบ',
            'description': 'เข้าสู่ช่วงปลายฤดูหีบแล้ว ควรมีการวางแผนการจัดการอ้อยที่เหลือและเตรียมการปิดฤดู',
            'actions': [
                'ประเมินปริมาณอ้อยที่เหลือในแต่ละโซน',
                'วางแผนการขนส่งอ้อยที่เหลือให้เสร็จสิ้น',
                'เตรียมการบำรุงรักษาเครื่องจักรหลังปิดฤดู',
                'สรุปบทเรียนและวางแผนปรับปรุงสำหรับฤดูถัดไป'
            ],
            'ai_generated': True
        })
    
    # Quality pattern recommendations
    if quality_pattern == "declining":
        recommendations.append({
            'priority': 'high',
            'category': 'การควบคุมคุณภาพ (Quality Control)',
            'title': 'การแก้ไขปัญหาคุณภาพอ้อยที่ลดลง',
            'description': 'ระบบตรวจพบแนวโน้มคุณภาพอ้อยที่ลดลงอย่างต่อเนื่อง',
            'actions': [
                'วิเคราะห์สาเหตุการลดลงของคุณภาพอ้อย',
                'ตรวจสอบกระบวนการจัดการอ้อยไฟและสิ่งปนเปื้อน',
                'ปรับปรุงเกณฑ์การตรวจรับคุณภาพ',
                'จัดอบรมทีมงานเรื่องการควบคุมคุณภาพ'
            ],
            'ai_generated': True
        })
    
    # Weekly pattern recommendations
    if weekly_pattern == "week_end":
        recommendations.append({
            'priority': 'medium',
            'category': 'การวางแผนรายสัปดาห์ (Weekly Planning)',
            'title': 'การเตรียมความพร้อมสำหรับวันหยุดสุดสัปดาห์',
            'description': 'เข้าสู่ช่วงสุดสัปดาห์ ควรมีการวางแผนการทำงานที่เหมาะสม',
            'actions': [
                'ตรวจสอบปริมาณอ้อยที่คาดว่าจะเข้ามาในช่วงสุดสัปดาห์',
                'จัดเตรียมกำลังคนและเครื่องจักรให้เพียงพอ',
                'วางแผนการขนส่งอ้อยให้เสร็จสิ้นก่อนวันหยุด',
                'เตรียมแผนการทำงานสำหรับวันจันทร์'
            ],
            'ai_generated': True
        })
    
    # Efficiency-based recommendations
    if 'volume_efficiency' in efficiency_metrics:
        vol_eff = efficiency_metrics['volume_efficiency']
        if vol_eff < 80:
            recommendations.append({
                'priority': 'high',
                'category': 'การเพิ่มประสิทธิภาพ (Efficiency Improvement)',
                'title': 'การปรับปรุงประสิทธิภาพการรับอ้อย',
                'description': f'ประสิทธิภาพการรับอ้อยต่ำกว่าเป้าหมาย ({vol_eff:.1f}%) จำเป็นต้องมีการปรับปรุง',
                'actions': [
                    'วิเคราะห์จุดอ่อนในกระบวนการรับอ้อย',
                    'ปรับปรุงระบบการจัดการคิวรถ',
                    'เพิ่มประสิทธิภาพการขนส่งจากไร่',
                    'จัดอบรมทีมงานเพื่อเพิ่มทักษะการทำงาน'
                ],
                'ai_generated': True
            })
    
    # Post-process for historical mode: reframe to forward-looking improvements
    if analysis_mode == 'historical':
        reframed = []
        for rec in recommendations:
            new_rec = dict(rec)
            # downgrade urgency and reframe titles
            if new_rec.get('priority') == 'critical':
                new_rec['priority'] = 'medium'
            new_rec['title'] = f"[สำหรับรอบถัดไป] {new_rec.get('title','')}"
            new_rec['category'] = f"{new_rec.get('category','')} (ย้อนหลัง)".strip()
            # replace immediate actions with planning-oriented ones while preserving some
            preserved = new_rec.get('actions', [])[:1]
            new_rec['actions'] = preserved + [
                'ทบทวนสาเหตุหลัก (root cause) จากข้อมูลวันนี้',
                'อัปเดตแผน/ตาราง/ทรัพยากรสำหรับรอบถัดไป',
                'สื่อสารแนวทางปรับปรุงกับทีมที่เกี่ยวข้อง'
            ]
            reframed.append(new_rec)
        return reframed

    return recommendations

def _generate_operational_insights(stats, exec_summary, efficiency_metrics, persona_name):
    insights = []
    persona_insights = PERSONAS.get(persona_name, {}).get('efficiency_insights', [])
    
    insights.extend(persona_insights)
    
    if 'volume_efficiency' in efficiency_metrics:
        vol_eff = efficiency_metrics['volume_efficiency']
        if vol_eff > 110 and "ประสิทธิภาพโดยรวมสูงกว่าเป้าหมายชัดเจน" not in insights:
            insights.append("ปริมาณอ้อยเข้าหีบสูงกว่าเป้าหมาย สะท้อนถึงการขนส่งที่มีประสิทธิภาพ")
    
    if 'quality_efficiency' in efficiency_metrics:
        qual_eff = efficiency_metrics['quality_efficiency']
        if qual_eff < 95 and "คุณภาพอ้อยสดมีแนวโน้มลดลง" not in insights:
            insights.append("คุณภาพอ้อยสดต่ำกว่าเกณฑ์ ควรให้ความสำคัญกับการจัดการอ้อยไฟ")
    
    if 'peak_ratio' in efficiency_metrics:
        peak_ratio = efficiency_metrics['peak_ratio']
        if peak_ratio > 2.0 and "การไหลเข้าไม่สม่ำเสมอ" not in insights:
            insights.append("ความผันผวนในการรับอ้อยสูง อาจกระทบต่อการทำงานของเครื่องจักร")

    return list(dict.fromkeys(insights))

def _generate_ai_enhanced_insights(stats, exec_summary, efficiency_metrics, persona_name, patterns):
    """AI-enhanced operational insights with pattern recognition and predictive analysis."""
    # If no data today, return empty insights
    if stats.get('today_total', 0) <= 0:
        return []
    
    insights = _generate_operational_insights(stats, exec_summary, efficiency_metrics, persona_name)
    
    # AI Pattern-based insights
    anomaly_score = patterns.get('anomaly_score', 0.0)
    if anomaly_score > 0.6:
        insights.append(f"ระบบ AI ตรวจพบความผิดปกติในระดับสูง (คะแนน: {anomaly_score:.2f}) ซึ่งบ่งชี้ถึงสถานการณ์ที่ต้องให้ความสนใจเป็นพิเศษ")
    
    seasonal_pattern = patterns.get('seasonal_pattern')
    if seasonal_pattern == "peak_season":
        insights.append("เรากำลังอยู่ในช่วงฤดูหีบอ้อยที่คึกคักที่สุด ซึ่งเป็นโอกาสทองในการเพิ่มประสิทธิภาพการผลิต")
    elif seasonal_pattern == "late_season":
        insights.append("เข้าสู่ช่วงปลายฤดูหีบแล้ว ควรมีการวางแผนการจัดการอ้อยที่เหลืออย่างรอบคอบ")
    
    weekly_pattern = patterns.get('weekly_pattern')
    if weekly_pattern == "week_end":
        insights.append("เข้าสู่ช่วงสุดสัปดาห์ ซึ่งอาจส่งผลต่อรูปแบบการขนส่งและปริมาณอ้อยที่เข้ามา")
    
    quality_pattern = patterns.get('quality_pattern')
    if quality_pattern == "declining":
        insights.append("ระบบ AI ตรวจพบแนวโน้มคุณภาพอ้อยที่ลดลงอย่างต่อเนื่อง ซึ่งอาจส่งผลต่อค่า CCS")
    elif quality_pattern == "improving":
        insights.append("ระบบ AI ตรวจพบแนวโน้มคุณภาพอ้อยที่ดีขึ้น ซึ่งเป็นสัญญาณที่ดีสำหรับค่า CCS")
    
    # Predictive insights
    predictive_signals = patterns.get('predictive_signals', [])
    if "high_risk_alert" in predictive_signals:
        insights.append("ระบบ AI คาดการณ์ว่าอาจเกิดปัญหาสำคัญในอนาคตอันใกล้ ควรมีการเตรียมแผนรองรับ")
    elif "seasonal_quality_decline" in predictive_signals:
        insights.append("ระบบ AI ตรวจพบแนวโน้มคุณภาพที่ลดลงตามฤดูกาล ซึ่งเป็นรูปแบบที่พบได้ในช่วงปลายฤดู")
    elif "weekend_push_pattern" in predictive_signals:
        insights.append("ระบบ AI ตรวจพบรูปแบบการเร่งรัดในช่วงสุดสัปดาห์ ซึ่งอาจส่งผลต่อความเสถียรของการทำงาน")
    
    # Efficiency-based AI insights
    if 'volume_efficiency' in efficiency_metrics:
        vol_eff = efficiency_metrics['volume_efficiency']
        if vol_eff > 130:
            insights.append(f"ประสิทธิภาพการรับอ้อยสูงมาก ({vol_eff:.1f}%) สะท้อนถึงการทำงานที่มีประสิทธิภาพของทีมงาน")
        elif vol_eff < 70:
            insights.append(f"ประสิทธิภาพการรับอ้อยต่ำมาก ({vol_eff:.1f}%) จำเป็นต้องมีการปรับปรุงอย่างเร่งด่วน")
    
    if 'quality_efficiency' in efficiency_metrics:
        qual_eff = efficiency_metrics['quality_efficiency']
        if qual_eff > 110:
            insights.append("คุณภาพอ้อยดีกว่าค่าเฉลี่ยอย่างมีนัยสำคัญ ซึ่งจะส่งผลดีต่อค่า CCS")
        elif qual_eff < 85:
            insights.append("คุณภาพอ้อยต่ำกว่าเกณฑ์อย่างมาก ควรให้ความสำคัญกับการควบคุมคุณภาพเป็นพิเศษ")
    
    return list(dict.fromkeys(insights))

def _generate_predictive_insights(stats, exec_summary, trend_data, efficiency_metrics):
    predictions = []
    
    today_total = stats.get('today_total', 0)
    # If no data today, return empty predictions
    if today_total <= 0:
        return predictions
    
    hours_processed = exec_summary.get('hours_processed', 0)
    if 4 < hours_processed < 24:
        current_rate = today_total / hours_processed
        remaining_hours = 24 - hours_processed
        predicted_total = today_total + (current_rate * remaining_hours)
        predictions.append(f"คาดการณ์ปริมาณอ้อยเมื่อสิ้นสุดวันอาจอยู่ที่ประมาณ: {predicted_total:,.0f} ตัน")
    
    if 'quality_efficiency' in efficiency_metrics:
        qual_eff = efficiency_metrics['quality_efficiency']
        if qual_eff < 95:
            predictions.append("จากข้อมูลปัจจุบัน คุณภาพอ้อยในวันพรุ่งนี้อาจยังคงเป็นประเด็นที่ต้องจับตา")
    
    if trend_data.get('has_trend_data'):
        tons_trend = trend_data.get('tons_trend_percent', 0)
        if tons_trend < -5:
            predictions.append("แนวโน้มปริมาณที่ลดลงอาจส่งผลต่อเนื่องในระยะสั้น ควรมีแผนรองรับ")
    
    return predictions

def _generate_enhanced_findings_html(stats, trend_html, comparison_period_days, anomalies=None, efficiency_metrics=None, operational_insights=None, predictions=None, patterns=None):
    today_total, avg_total = stats.get('today_total', 0), stats.get('avg_daily_tons', 0)
    
    # If no data today, return empty findings
    if today_total <= 0:
        return ""
    
    tons_diff_pct = ((today_total - avg_total) / avg_total * 100) if avg_total > 0 else 0
    fresh_pct_diff = stats.get('type_1_percent', 0) - stats.get('avg_fresh_percent', 0)
    tons_icon, fresh_icon = ('text-success' if tons_diff_pct >= 0 else 'text-danger'), ('text-success' if fresh_pct_diff >= 0 else 'text-danger')
    
    findings_html = f"""<p class='mb-2'>เทียบกับค่าเฉลี่ยย้อนหลัง <strong>{comparison_period_days} วัน</strong>:</p>
    <ul class="list-unstyled mb-0">
        <li class="mb-2"><i class="bi {'bi-arrow-up-circle-fill' if tons_diff_pct >=0 else 'bi-arrow-down-circle-fill'} {tons_icon} me-2"></i>ปริมาณรวม: <span class="{tons_icon}">{tons_diff_pct:+.1f}%</span></li>
        <li><i class="bi {'bi-arrow-up-circle-fill' if fresh_pct_diff >=0 else 'bi-arrow-down-circle-fill'} {fresh_icon} me-2"></i>สัดส่วนอ้อยสด: <span class="{fresh_icon}">{fresh_pct_diff:+.1f}%</span></li>
    </ul>"""
    
    if efficiency_metrics:
        findings_html += f"""<li class="mt-3 pt-2 border-top border-white border-opacity-10">
            <i class="bi bi-speedometer2 me-2 text-info"></i><strong>ดัชนีชี้วัดประสิทธิภาพ:</strong>
            <ul class="list-unstyled mt-1 ps-3">"""
        
        if 'volume_efficiency' in efficiency_metrics:
            vol_status_color = 'text-success' if efficiency_metrics['volume_status'] == 'สูง' else 'text-warning' if efficiency_metrics['volume_status'] == 'ปกติ' else 'text-danger'
            findings_html += f"""<li><i class="bi bi-graph-up me-1 small"></i>ปริมาณ: <span class="{vol_status_color}">{efficiency_metrics['volume_efficiency']:.1f}% ({efficiency_metrics['volume_status']})</span></li>"""
        
        if 'quality_efficiency' in efficiency_metrics:
            qual_status_color = 'text-success' if efficiency_metrics['quality_status'] == 'สูง' else 'text-warning' if efficiency_metrics['quality_status'] == 'ปกติ' else 'text-danger'
            findings_html += f"""<li><i class="bi bi-gem me-1 small"></i>คุณภาพ: <span class="{qual_status_color}">{efficiency_metrics['quality_efficiency']:.1f}% ({efficiency_metrics['quality_status']})</span></li>"""
        
        if 'stability_status' in efficiency_metrics:
            stab_status_color = 'text-success' if efficiency_metrics['stability_status'] == 'เสถียร' else 'text-warning' if efficiency_metrics['stability_status'] == 'ปานกลาง' else 'text-danger'
            findings_html += f"""<li><i class="bi bi-activity me-1 small"></i>ความเสถียร: <span class="{stab_status_color}">{efficiency_metrics['stability_status']}</span></li>"""
        
        findings_html += "</ul></li>"
    
    # AI Pattern Analysis Section
    if patterns:
        findings_html += f"""<li class="mt-3 pt-2 border-top border-white border-opacity-10">
            <i class="bi bi-robot me-2 text-primary"></i><strong>การวิเคราะห์ AI:</strong>
            <ul class="list-unstyled mt-1 ps-3">"""
        
        anomaly_score = patterns.get('anomaly_score', 0.0)
        if anomaly_score > 0.6:
            anomaly_color = 'text-danger' if anomaly_score > 0.8 else 'text-warning'
            findings_html += f"""<li><i class="bi bi-shield-exclamation me-1 small {anomaly_color}"></i>ความผิดปกติ: <span class="{anomaly_color}">{anomaly_score:.2f} (สูง)</span></li>"""
        elif anomaly_score > 0.3:
            findings_html += f"""<li><i class="bi bi-shield-check me-1 small text-warning"></i>ความผิดปกติ: <span class="text-warning">{anomaly_score:.2f} (ปานกลาง)</span></li>"""
        else:
            findings_html += f"""<li><i class="bi bi-shield-check me-1 small text-success"></i>ความผิดปกติ: <span class="text-success">{anomaly_score:.2f} (ต่ำ)</span></li>"""
        
        seasonal_pattern = patterns.get('seasonal_pattern')
        if seasonal_pattern:
            season_text = {"peak_season": "ฤดูหีบสูงสุด", "late_season": "ปลายฤดูหีบ", "off_season": "นอกฤดูหีบ"}.get(seasonal_pattern, seasonal_pattern)
            findings_html += f"""<li><i class="bi bi-calendar-event me-1 small text-info"></i>รูปแบบฤดูกาล: <span class="text-info">{season_text}</span></li>"""
        
        quality_pattern = patterns.get('quality_pattern')
        if quality_pattern:
            qual_text = {"improving": "ดีขึ้น", "declining": "ลดลง", "stable": "เสถียร"}.get(quality_pattern, quality_pattern)
            qual_color = 'text-success' if quality_pattern == 'improving' else 'text-danger' if quality_pattern == 'declining' else 'text-info'
            findings_html += f"""<li><i class="bi bi-gem me-1 small {qual_color}"></i>แนวโน้มคุณภาพ: <span class="{qual_color}">{qual_text}</span></li>"""
        
        findings_html += "</ul></li>"
    
    if operational_insights:
        insights_list = ''.join([f'<li><i class="bi bi-lightbulb me-1 small text-info"></i>{insight}</li>' for insight in operational_insights[:3]])
        findings_html += f"""<li class="mt-2 pt-2 border-top border-white border-opacity-10">
            <i class="bi bi-gear-fill me-2 text-primary"></i><strong>ข้อมูลเชิงลึกปฏิบัติการ:</strong>
            <ul class="list-unstyled mt-1 ps-3">{insights_list}</ul></li>"""
    
    if predictions:
        predictions_list = ''.join([f'<li><i class="bi bi-crystal-ball me-1 small text-warning"></i>{prediction}</li>' for prediction in predictions[:2]])
        findings_html += f"""<li class="mt-2 pt-2 border-top border-white border-opacity-10">
            <i class="bi bi-graph-up-arrow me-2 text-warning"></i><strong>การคาดการณ์เชิงข้อมูล:</strong>
            <ul class="list-unstyled mt-1 ps-3">{predictions_list}</ul></li>"""
    
    findings_html += trend_html
    
    # Remove anomalies from findings_html since they will be displayed separately in the UI
    # if anomalies:
    #     anomaly_list = ''.join([f'<li><i class="bi bi-exclamation-circle me-1 small text-warning"></i>{anomaly}</li>' for anomaly in anomalies])
    #     findings_html += f"""<li class="mt-2 pt-2 border-top border-white border-opacity-10">
    #         <i class="bi bi-exclamation-triangle-fill me-2 text-warning"></i><strong>รายการที่ควรตรวจสอบ:</strong>
    #         <ul class="list-unstyled mt-1 ps-3">{anomaly_list}</ul></li>"""
    
    return findings_html

# --- MAIN ANALYSIS GENERATION FUNCTION ---
def generate_analysis(selected_date, statistics, executive_summary, trend_data={}, comparison_period_days=7, contextual_data: Optional[Dict[str, Any]] = None, analysis_mode: Optional[str] = None):
    try:
        if not isinstance(statistics, dict) or not isinstance(executive_summary, dict):
            raise ValueError("ข้อมูลสถิติ (statistics) หรือข้อมูลสรุป (executive_summary) ไม่ใช่ dictionary ที่ถูกต้อง")

        hours_processed = executive_summary.get('hours_processed', 0)
        exec_summary_data = {'latest_time': executive_summary.get('latest_volume_time', 'N/A'),'latest_tons': format_num(executive_summary.get('latest_volume_tons', 0)),'peak_time': executive_summary.get('peak_hour_time', 'N/A'),'peak_tons': format_num(executive_summary.get('peak_hour_tons', 0)),'forecast_total': format_num(executive_summary.get('forecasted_total', 0)),'forecast_hours': hours_processed, 'forecast_label': ""}
        if hours_processed > 0: exec_summary_data['forecast_label'] = "(แนวโน้มเบื้องต้น)" if hours_processed < 6 else "(ประเมินจากข้อมูลครึ่งวัน)" if hours_processed < 12 else "(คาดการณ์เต็มวัน)"

        temporal_context = _get_temporal_context(selected_date)
        # Determine analysis mode (current vs historical)
        try:
            today_date = datetime.now().date()
            selected_date_only = selected_date.date() if isinstance(selected_date, datetime) else selected_date
            computed_mode = 'historical' if selected_date_only < today_date else 'current'
        except Exception:
            computed_mode = 'current'
        if analysis_mode not in ['current', 'historical']:
            analysis_mode = computed_mode
        is_new_year_holiday = (selected_date.month == 12 and selected_date.day == 31) or (selected_date.month == 1 and selected_date.day in [1, 2])
        
        if is_new_year_holiday:
            return {"executive": exec_summary_data, "guru_analysis": {"headline": {"text": "หยุดทำการ (ช่วงปีใหม่)", "color": "text-info", "icon": "bi-calendar-check"}, "comment": "โรงงานหยุดทำการในช่วงเทศกาลปีใหม่ จะไม่มีข้อมูลการรับอ้อยในวันนี้ครับ", "recommendation": "สวัสดีปีใหม่ครับ ขอให้ทีมงานทุกท่านมีความสุข สุขภาพแข็งแรงตลอดปี", "findings_html": "", "scores": {"overall_score_display": "N/A", "overall_score_value": 0}, "ai_enhanced": False}}

        is_in_season = (selected_date.month >= 12 or selected_date.month <= 4)
        today_total = statistics.get('today_total', 0)
        
        # Check if there's no data for today
        if today_total <= 0:
            status, icon, comment = ("ไม่มีข้อมูล", "bi-exclamation-circle-fill", "ไม่พบข้อมูลการรับอ้อยในวันนี้")
            return {"executive": exec_summary_data, "guru_analysis": {
                "headline": {"text": status, "color": "text-warning", "icon": icon}, 
                "comment": comment, 
                "recommendation": "กรุณาตรวจสอบข้อมูลการรับอ้อยหรือรอให้มีข้อมูลเข้ามาในระบบ", 
                "findings_html": "", 
                "scores": {"overall_score_display": "N/A", "overall_score_value": 0}, 
                "ai_enhanced": False,
                "efficiency_metrics": {},
                "operational_insights": [],
                "predictive_insights": [],
                "anomalies": []
            }}
        
        # Check if not in season
        if not is_in_season:
            status, icon, comment = ("นอกฤดูหีบอ้อย", "bi-calendar-x", "ไม่มีการรับอ้อยในช่วงนี้ (ฤดูหีบปกติ: ธ.ค. - เม.ย.)")
            return {"executive": exec_summary_data, "guru_analysis": {
                "headline": {"text": status, "color": "text-info", "icon": icon}, 
                "comment": comment, 
                "recommendation": "", 
                "findings_html": "", 
                "scores": {"overall_score_display": "N/A", "overall_score_value": 0}, 
                "ai_enhanced": False,
                "efficiency_metrics": {},
                "operational_insights": [],
                "predictive_insights": [],
                "anomalies": []
            }}
        
        # Check if no comparison data
        if not statistics.get('has_comparison_data'):
            status, icon, comment = ("ข้อมูลไม่เพียงพอ", "bi-info-circle-fill", "ไม่พบข้อมูลย้อนหลังเพื่อใช้เปรียบเทียบ")
            return {"executive": exec_summary_data, "guru_analysis": {
                "headline": {"text": status, "color": "text-info", "icon": icon}, 
                "comment": comment, 
                "recommendation": "ระบบต้องการข้อมูลย้อนหลังเพื่อการวิเคราะห์ที่แม่นยำ", 
                "findings_html": "", 
                "scores": {"overall_score_display": "N/A", "overall_score_value": 0}, 
                "ai_enhanced": False,
                "efficiency_metrics": {},
                "operational_insights": [],
                "predictive_insights": [],
                "anomalies": []
            }}

        scores = _calculate_scores(statistics, executive_summary)
        trend_score, trend_html = _get_trend_context(trend_data)
        anomalies = _local_ai_anomaly_detection(statistics, executive_summary)
        
        persona_name = _select_persona_name(scores, trend_score, hours_processed)
        persona_profile = PERSONAS[persona_name]
        
        efficiency_metrics = _calculate_efficiency_metrics(statistics, executive_summary, trend_data)
        performance_benchmarks = _calculate_performance_benchmarks(statistics, executive_summary, trend_data)
        
        # Get time-based context for enhanced analysis
        time_context = _get_time_based_context(selected_date)
        
        # AI-Enhanced Analysis
        patterns = _advanced_pattern_recognition(statistics, executive_summary, trend_data, selected_date)
        operational_insights = _generate_ai_enhanced_insights(statistics, executive_summary, efficiency_metrics, persona_name, patterns)
        predictive_insights = _generate_predictive_insights(statistics, executive_summary, trend_data, efficiency_metrics)
        ai_enhanced_alerts = _generate_ai_enhanced_alerts(statistics, executive_summary, efficiency_metrics, trend_data, patterns)
        ai_enhanced_recommendations = _generate_ai_enhanced_recommendations(statistics, executive_summary, efficiency_metrics, ai_enhanced_alerts, persona_name, patterns, selected_date, analysis_mode)
        dynamic_comment = _generate_conversational_narrative_v3(persona_name, statistics, patterns, efficiency_metrics, analysis_mode)
        chatgpt_style_recommendations = _generate_chatgpt_style_recommendations(statistics, executive_summary, efficiency_metrics, ai_enhanced_alerts, persona_name, patterns, selected_date, analysis_mode)
        prediction_text = _generate_ai_predictions(statistics, executive_summary, trend_data, selected_date)
        
        # Enhanced practical insights and advice
        practical_insights = _generate_practical_insights(statistics, executive_summary, time_context, persona_name)
        actionable_advice = _generate_actionable_advice(statistics, executive_summary, time_context, persona_name)
        learning_points = _generate_learning_points(statistics, executive_summary, persona_name, patterns)
        
        # AI-Enhanced Scoring System
        base_weighted_score = (scores['quantity']*0.4) + (scores['quality']*0.4) + (scores['stability']*0.2)
        
        # Pattern-based score adjustments
        pattern_adjustment = 0.0
        anomaly_score = patterns.get('anomaly_score', 0.0)
        
        if anomaly_score > 0.7:
            pattern_adjustment -= 1.5  # High anomaly reduces score
        elif anomaly_score > 0.4:
            pattern_adjustment -= 0.5  # Moderate anomaly slightly reduces score
        elif anomaly_score < 0.2:
            pattern_adjustment += 0.3  # Low anomaly slightly improves score
        
        # Efficiency-based adjustments
        if 'volume_efficiency' in efficiency_metrics:
            vol_eff = efficiency_metrics['volume_efficiency']
            if vol_eff > 120:
                pattern_adjustment += 0.5
            elif vol_eff < 80:
                pattern_adjustment -= 0.5
        
        if 'quality_efficiency' in efficiency_metrics:
            qual_eff = efficiency_metrics['quality_efficiency']
            if qual_eff > 105:
                pattern_adjustment += 0.3
            elif qual_eff < 95:
                pattern_adjustment -= 0.3
        
        # Seasonal adjustments
        seasonal_pattern = patterns.get('seasonal_pattern')
        if seasonal_pattern == "peak_season":
            pattern_adjustment += 0.2  # Peak season bonus
        elif seasonal_pattern == "late_season":
            pattern_adjustment -= 0.1  # Late season slight penalty
        
        # Calculate final AI-enhanced score
        ai_enhanced_score = base_weighted_score + pattern_adjustment
        final_score_10_point = max(1.0, min(10.0, 2.0 + ((ai_enhanced_score - 1) / 4.0) * 8.0))
        
        # Get time-based context for enhanced recommendations
        time_context = _get_time_based_context(selected_date)
        
        # Use ChatGPT-style recommendations
        recommendation = chatgpt_style_recommendations
        
        # Add special context for holidays
        if temporal_context == "PRE_HOLIDAY_PUSH":
            recommendation = "**🎄 ข้อเสนอแนะช่วงก่อนหยุดยาว**\n\nเป้าหมายหลักคือการเคลียร์อ้อยให้หมดก่อนหยุดยาวครับ อยากให้ทุกทีมช่วยกันประสานงานอย่างใกล้ชิด และวางแผนการหยุดเครื่องจักรให้ปลอดภัย\n\n**📋 สิ่งที่ต้องทำ:**\n• ประสานงานกับเกษตรกรเพื่อเร่งการขนส่ง\n• วางแผนการหยุดเครื่องจักรอย่างปลอดภัย\n• เตรียมการบำรุงรักษาระหว่างหยุดยาว"
        elif temporal_context == "POST_HOLIDAY_RESTART":
            recommendation = "**🔄 ข้อเสนอแนะช่วงเริ่มต้นใหม่**\n\nช่วงนี้อยากให้เน้นการตรวจสอบความพร้อมของเครื่องจักรเป็นพิเศษ และค่อยๆ เพิ่มกำลังการผลิตอย่างระมัดระวัง เพื่อให้การเริ่มต้นใหม่เป็นไปอย่างราบรื่นครับ\n\n**🔧 สิ่งที่ต้องทำ:**\n• ตรวจสอบความพร้อมของเครื่องจักรทุกชิ้น\n• ค่อยๆ เพิ่มกำลังการผลิตอย่างระมัดระวัง\n• ติดตามผลการดำเนินงานอย่างใกล้ชิด"

        guru_analysis = {
            "headline": {"text": persona_profile.get("status"), "color": persona_profile.get("color"), "icon": persona_profile.get("icon")},
            "findings_html": _generate_enhanced_findings_html(statistics, trend_html, comparison_period_days, anomalies, efficiency_metrics, operational_insights, predictive_insights, patterns),
            "comment": dynamic_comment,
            "recommendation": recommendation,
            "prediction": prediction_text,
            "scores": {"overall_score_display": f"{format_num(final_score_10_point,1)} / 10", "overall_score_value": final_score_10_point},
            "ai_enhanced": True,
            "anomalies": anomalies,
            "efficiency_metrics": efficiency_metrics,
            "operational_insights": operational_insights,
            "predictive_insights": predictive_insights,
            "real_time_alerts": ai_enhanced_alerts,
            "performance_benchmarks": performance_benchmarks,
            "operational_recommendations": ai_enhanced_recommendations,
            "ai_patterns": patterns,
            "practical_insights": practical_insights,
            "actionable_advice": actionable_advice,
            "learning_points": learning_points,
            "time_context": time_context,
            "human_like_intelligence": True
        }
        return {"executive": exec_summary_data, "guru_analysis": guru_analysis, "mode": analysis_mode}

    except Exception as e:
        print(f"!!! ERROR IN generate_analysis: {e}")
        traceback.print_exc()
        return {
            "executive": {},
            "guru_analysis": {
                "headline": {"text": "เกิดข้อผิดพลาดภายใน", "color": "text-danger", "icon": "bi-exclamation-triangle-fill"},
                "comment": f"ระบบพบข้อผิดพลาดขณะประมวลผล: {e}",
                "recommendation": "กรุณาติดต่อผู้ดูแลระบบเพื่อตรวจสอบ Log",
                "findings_html": "", "scores": {"overall_score_display": "N/A", "overall_score_value": 0},
                "ai_enhanced": False
            }
        }

# --- AI Management Functions ---
def train_local_ai(historical_data: List[Dict[str, Any]]) -> bool:
    return local_ai_engine.train_models(historical_data)

def get_local_ai_status():
    return {"trained": local_ai_engine.is_trained, "model_path": local_ai_config.model_path, "min_data_points": local_ai_config.min_data_points}

def update_local_ai_config(model_path=None, min_data_points=None):
    if model_path: local_ai_config.model_path = model_path
    if min_data_points: local_ai_config.min_data_points = min_data_points
    local_ai_config.ensure_model_directory()
    return get_local_ai_status()

def test_local_ai():
    if not local_ai_engine.load_models(): return False, "ยังไม่ได้ฝึกสอนโมเดล"
    try:
        stats, exec_sum = {'today_total':1000, 'avg_daily_tons':950, 'type_1_percent':85, 'avg_fresh_percent':80}, {'hours_processed':12, 'peak_hour_tons':250}
        scores = _calculate_scores(stats, exec_sum)
        persona_name = _select_persona_name(scores, 0, exec_sum['hours_processed'])
        narrative = _generate_guru_narrative(persona_name, stats, exec_sum, None)
        return (True, "การทดสอบสำเร็จ") if narrative else (False, "การสร้าง Narrative ล้มเหลว")
    except Exception as e: return False, f"การทดสอบล้มเหลว: {e}"

def list_trained_ranges(model_path="ai_models/"):
    hist_file = os.path.join(model_path, "training_history.jsonl")
    if not os.path.exists(hist_file): return {"count": 0, "ranges": [], "years": []}
    ranges, years = [], set()
    try:
        with open(hist_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                r = rec.get("train_range", {})
                if r.get("start") and r.get("end"):
                    ranges.append({"start":r["start"], "end":r["end"], "trained_at":rec.get("trained_at")})
                    years.update([int(r["start"][:4]), int(r["end"][:4])])
    except Exception: return {"count": 0, "ranges": [], "years": []}
    return {"count": len(ranges), "ranges": sorted(ranges, key=lambda x: x["start"]), "years": sorted(list(years))}

# --- HUMAN-LIKE INTELLIGENCE FUNCTIONS ---
def _get_time_based_context(selected_date: datetime, current_hour: int = None) -> Dict[str, Any]:
    """Get time-based context for more human-like responses."""
    if current_hour is None:
        current_hour = datetime.now().hour
    
    context = {
        'time_of_day': None,
        'day_of_week': selected_date.strftime('%A'),
        'is_weekend': selected_date.weekday() >= 5,
        'is_holiday_season': False,
        'time_greeting': None,
        'urgency_level': 'normal',
        'time_specific_advice': []
    }
    
    # Time of day context
    if 6 <= current_hour < 12:
        context['time_of_day'] = 'morning'
        context['time_greeting'] = 'สวัสดีตอนเช้าครับ'
        context['urgency_level'] = 'moderate'
        context['time_specific_advice'].append('ช่วงเช้าเป็นเวลาที่สำคัญในการตั้งค่าการทำงานให้ถูกต้อง')
    elif 12 <= current_hour < 17:
        context['time_of_day'] = 'afternoon'
        context['time_greeting'] = 'สวัสดีตอนบ่ายครับ'
        context['urgency_level'] = 'high'
        context['time_specific_advice'].append('ช่วงบ่ายเป็นช่วงที่ควรเร่งรัดการทำงานให้เต็มที่')
    elif 17 <= current_hour < 22:
        context['time_of_day'] = 'evening'
        context['time_greeting'] = 'สวัสดีตอนเย็นครับ'
        context['urgency_level'] = 'moderate'
        context['time_specific_advice'].append('ช่วงเย็นควรเตรียมการสำหรับวันพรุ่งนี้')
    else:
        context['time_of_day'] = 'night'
        context['time_greeting'] = 'สวัสดีครับ'
        context['urgency_level'] = 'low'
        context['time_specific_advice'].append('ช่วงกลางคืนควรเน้นการรักษาความปลอดภัย')
    
    # Holiday season context
    if (selected_date.month == 12 and selected_date.day >= 25) or (selected_date.month == 1 and selected_date.day <= 5):
        context['is_holiday_season'] = True
        context['time_specific_advice'].append('ช่วงเทศกาลควรระมัดระวังเรื่องการจัดการแรงงานและตารางการทำงาน')
    
    # Weekend context
    if context['is_weekend']:
        context['time_specific_advice'].append('ช่วงสุดสัปดาห์อาจมีผลต่อรูปแบบการขนส่งและปริมาณอ้อย')
    
    return context

def _generate_human_like_greeting(time_context: Dict[str, Any], persona_name: str) -> str:
    """Generate human-like greeting combining time awareness with persona"""
    time_greeting = time_context.get('time_greeting', 'สวัสดีครับ')
    
    # Simplified persona-specific phrases
    persona_phrases = {
        "CRITICAL": "ผมต้องแจ้งให้ทราบว่าวันนี้มีสถานการณ์ที่ต้องให้ความสนใจเป็นพิเศษครับ",
        "EXCELLENT": "ผมมีข่าวดีมาบอกครับ วันนี้ผลงานของเรายอดเยี่ยมมาก",
        "WEAK_START": "ผมสังเกตเห็นบางอย่างที่น่าสนใจในวันนี้ครับ",
        "STEADY_PERFORMANCE": "วันนี้การดำเนินงานเป็นไปตามแผนครับ",
        "QUANTITY_PUSH": "วันนี้ปริมาณทำได้ดี แต่มีประเด็นที่ต้องเฝ้าระวังครับ",
        "QUALITY_FOCUS": "วันนี้คุณภาพอ้อยดีมาก แต่ปริมาณยังต้องเร่งขึ้นครับ",
        "VOLATILE_PERFORMANCE": "วันนี้ผลงานดี แต่มีความผันผวนครับ",
        "STABLE_RECOVERY": "เป็นสัญญาณที่ดีที่การดำเนินงานกลับมาสู่ภาวะปกติครับ",
        "CONCERNING_TREND": "วันนี้ผลงานยังอยู่ในเกณฑ์ แต่มีแนวโน้มที่ต้องจับตาครับ"
    }
    
    persona_phrase = persona_phrases.get(persona_name, "วันนี้มีข้อมูลที่น่าสนใจครับ")
    
    return f"{time_greeting} {persona_phrase}"

def _generate_contextual_insights(stats: Dict[str, Any], exec_summary: Dict[str, Any], time_context: Dict[str, Any]) -> List[str]:
    """Generate contextual insights based on current situation and time."""
    insights = []
    
    today_total = stats.get('today_total', 0)
    hours_processed = exec_summary.get('hours_processed', 0)
    
    # Time-based insights (simplified)
    if time_context['time_of_day'] == 'morning' and hours_processed < 4:
        if today_total < stats.get('avg_daily_tons', 0) * 0.2:
            insights.append("การเริ่มต้นวันนี้ค่อนข้างช้า ควรเร่งรัดการขนส่ง")
        else:
            insights.append("การเริ่มต้นวันนี้เป็นไปด้วยดี ควรรักษาโมเมนตัมนี้ไว้")
    
    elif time_context['time_of_day'] == 'afternoon' and hours_processed >= 8:
        expected_progress = (hours_processed / 24) * stats.get('avg_daily_tons', 0)
        if today_total < expected_progress * 0.8:
            insights.append("ในช่วงบ่ายนี้เรายังตามเป้าหมายอยู่ จำเป็นต้องเร่งรัดการทำงาน")
        elif today_total > expected_progress * 1.2:
            insights.append("ในช่วงบ่ายนี้เราทำได้ดีเกินเป้าหมาย ควรรักษามาตรฐานนี้ไว้")
    
    elif time_context['time_of_day'] == 'evening':
        if hours_processed >= 16:
            insights.append("ใกล้สิ้นสุดวันแล้ว ควรประเมินผลงานและเตรียมการสำหรับวันพรุ่งนี้")
    
    # Performance-based insights
    if today_total > stats.get('avg_daily_tons', 0) * 1.2:
        insights.append("ปริมาณอ้อยสูงกว่าเป้าหมาย สะท้อนถึงการขนส่งที่มีประสิทธิภาพ")
    
    if stats.get('type_1_percent', 0) < 80:
        insights.append("คุณภาพอ้อยต่ำกว่าเกณฑ์ ควรให้ความสำคัญกับการจัดการอ้อยไฟ")
    
    return insights

def _generate_adaptive_recommendations(persona_name: str, stats: Dict[str, Any], time_context: Dict[str, Any], efficiency_metrics: Dict[str, Any]) -> List[str]:
    """Generate adaptive recommendations based on current situation and time."""
    recommendations = []
    
    # Time-based recommendations (simplified)
    if time_context['time_of_day'] == 'morning':
        if persona_name in ["WEAK_START", "CRITICAL"]:
            recommendations.append("ในช่วงเช้านี้ควรมีการประชุมฉุกเฉินเพื่อแก้ไขปัญหาที่เกิดขึ้น")
        else:
            recommendations.append("ในช่วงเช้านี้ควรตั้งเป้าหมายและวางแผนการทำงานให้ชัดเจน")
    
    elif time_context['time_of_day'] == 'afternoon':
        if 'volume_efficiency' in efficiency_metrics and efficiency_metrics['volume_efficiency'] < 90:
            recommendations.append("ในช่วงบ่ายนี้ควรเร่งรัดการขนส่งและประสานงานกับทีมต่างๆ")
        else:
            recommendations.append("ในช่วงบ่ายนี้ควรรักษามาตรฐานการทำงานและมองหาโอกาสในการปรับปรุง")
    
    elif time_context['time_of_day'] == 'evening':
        recommendations.append("ในช่วงเย็นนี้ควรสรุปผลงานของวันและวางแผนการทำงานสำหรับวันพรุ่งนี้")
    
    # Performance-based recommendations
    if stats.get('today_total', 0) < stats.get('avg_daily_tons', 0) * 0.8:
        recommendations.append("ควรวิเคราะห์สาเหตุและวางแผนการแก้ไขสำหรับวันพรุ่งนี้")
    
    if stats.get('type_1_percent', 0) < 80:
        recommendations.append("ควรเพิ่มความสำคัญกับการควบคุมคุณภาพอ้อย")
    
    return recommendations

def _generate_emotional_response(persona_name: str, stats: Dict[str, Any], time_context: Dict[str, Any]) -> str:
    """Generate emotional response based on performance and persona."""
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    
    if persona_name == "EXCELLENT":
        if today_total > avg_total * 1.2:
            return "ผมรู้สึกภูมิใจมากที่เห็นทีมงานทำงานได้ยอดเยี่ยมครับ"
        else:
            return "ผมรู้สึกดีใจที่เห็นการทำงานที่เป็นไปตามมาตรฐานครับ"
    
    elif persona_name == "CRITICAL":
        if today_total < avg_total * 0.5:
            return "ผมรู้สึกกังวลกับสถานการณ์ในวันนี้ครับ แต่เชื่อว่าเราจะผ่านพ้นไปได้"
        else:
            return "ผมรู้สึกห่วงใยกับแนวโน้มที่เห็นครับ"
    
    elif persona_name == "WEAK_START":
        return "ผมรู้สึกมั่นใจว่าเรายังมีโอกาสในการปรับปรุงครับ"
    
    else:
        return "ผมรู้สึกว่าการทำงานของเรายังมีพื้นที่สำหรับการพัฒนาอีกมากครับ"

def _generate_proactive_suggestions(stats: Dict[str, Any], exec_summary: Dict[str, Any], time_context: Dict[str, Any]) -> List[str]:
    """Generate proactive suggestions for future improvement."""
    suggestions = []
    
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    hours_processed = exec_summary.get('hours_processed', 0)
    
    # Proactive suggestions based on current performance
    if today_total < avg_total * 0.8:
        suggestions.append("ควรวิเคราะห์สาเหตุและวางแผนการแก้ไขสำหรับวันพรุ่งนี้")
    
    if hours_processed >= 12 and today_total < avg_total * 0.9:
        suggestions.append("ควรเตรียมแผนการชดเชยปริมาณที่ขาดในวันพรุ่งนี้")
    
    if stats.get('type_1_percent', 0) < 80:
        suggestions.append("ควรปรับปรุงกระบวนการจัดการคุณภาพอ้อย")
    
    return suggestions

def _generate_human_like_recommendation(persona_name: str, stats: Dict[str, Any], time_context: Dict[str, Any], efficiency_metrics: Dict[str, Any]) -> str:
    """Generate human-like recommendation with emotional intelligence."""
    base_recommendation = PERSONAS.get(persona_name, {}).get("base_recommendation", "")
    
    # Add time-based context to recommendation
    time_enhancement = ""
    if time_context['time_of_day'] == 'morning':
        time_enhancement = "ในช่วงเช้านี้ "
    elif time_context['time_of_day'] == 'afternoon':
        time_enhancement = "ในช่วงบ่ายนี้ "
    elif time_context['time_of_day'] == 'evening':
        time_enhancement = "ในช่วงเย็นนี้ "
    
    # Add urgency based on performance
    urgency_enhancement = ""
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    
    if persona_name == "CRITICAL":
        if today_total < avg_total * 0.5:
            urgency_enhancement = "ผมขอเน้นย้ำว่า "
        else:
            urgency_enhancement = "ผมขอแนะนำให้ "
    elif persona_name == "EXCELLENT":
        urgency_enhancement = "ผมขอชื่นชมและแนะนำให้ "
    else:
        urgency_enhancement = "ผมขอแนะนำให้ "
    
    # Add efficiency-based enhancement
    efficiency_enhancement = ""
    if 'volume_efficiency' in efficiency_metrics:
        vol_eff = efficiency_metrics['volume_efficiency']
        if vol_eff < 80:
            efficiency_enhancement = "เนื่องจากประสิทธิภาพการทำงานยังต่ำกว่าเป้าหมาย "
        elif vol_eff > 120:
            efficiency_enhancement = "เนื่องจากประสิทธิภาพการทำงานดีเกินเป้าหมาย "
    
    # Combine all enhancements
    enhanced_recommendation = f"{urgency_enhancement}{time_enhancement}{efficiency_enhancement}{base_recommendation}"
    
    # Add emotional closing
    if persona_name == "CRITICAL":
        enhanced_recommendation += " เพราะผมเชื่อว่าเราจะผ่านพ้นสถานการณ์นี้ไปได้ด้วยความร่วมมือของทุกคนครับ"
    elif persona_name == "EXCELLENT":
        enhanced_recommendation += " เพื่อรักษามาตรฐานระดับสูงนี้ไว้ให้ได้ครับ"
    else:
        enhanced_recommendation += " เพื่อให้การทำงานของเราดีขึ้นไปอีกครับ"
    
    return enhanced_recommendation

def _generate_contextual_wisdom(persona_name: str, stats: Dict[str, Any], time_context: Dict[str, Any], patterns: Dict[str, Any]) -> str:
    """Generate contextual wisdom based on current situation."""
    wisdom = ""
    
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    
    # Wisdom based on persona and performance (simplified)
    if persona_name == "CRITICAL":
        if today_total < avg_total * 0.5:
            wisdom = "ในสถานการณ์ที่ท้าทาย การทำงานเป็นทีมและการสื่อสารที่ชัดเจนจะเป็นกุญแจสำคัญ"
        else:
            wisdom = "การแก้ไขปัญหาอย่างเป็นระบบจะนำไปสู่ผลลัพธ์ที่ดี"
    
    elif persona_name == "EXCELLENT":
        if today_total > avg_total * 1.2:
            wisdom = "ความสำเร็จในวันนี้เป็นผลมาจากการวางแผนที่ดีและการทำงานร่วมกัน"
        else:
            wisdom = "การรักษามาตรฐานระดับสูงไว้ได้เป็นเรื่องที่ควรภูมิใจ"
    
    elif persona_name == "WEAK_START":
        wisdom = "การเริ่มต้นที่ช้าไม่ใช่จุดจบ แต่เป็นโอกาสในการเรียนรู้และปรับปรุง"
    
    else:
        wisdom = "การพัฒนาอย่างต่อเนื่องและการปรับตัวตามสถานการณ์เป็นกุญแจสำคัญ"
    
    return wisdom

def _generate_encouraging_message(persona_name: str, stats: Dict[str, Any], time_context: Dict[str, Any]) -> str:
    """Generate encouraging message based on performance and time."""
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    
    if persona_name == "CRITICAL":
        if today_total < avg_total * 0.5:
            return "ผมเชื่อมั่นในความสามารถของทีมงานครับ เราจะผ่านพ้นสถานการณ์นี้ไปด้วยกัน"
        else:
            return "แม้จะมีความท้าทาย แต่ผมเห็นความพยายามของทุกคนครับ"
    
    elif persona_name == "EXCELLENT":
        if today_total > avg_total * 1.2:
            return "ผลงานที่ยอดเยี่ยมในวันนี้เป็นสิ่งที่ทุกคนควรภูมิใจครับ"
        else:
            return "การรักษามาตรฐานระดับสูงไว้ได้เป็นเรื่องที่น่าชื่นชมครับ"
    
    elif persona_name == "WEAK_START":
        return "เรายังมีโอกาสในการปรับปรุงครับ ผมเชื่อในความสามารถของทุกคน"
    
    else:
        return "การทำงานที่มั่นคงและต่อเนื่องเป็นรากฐานของความสำเร็จครับ"

def _generate_practical_insights(stats: Dict[str, Any], exec_summary: Dict[str, Any], time_context: Dict[str, Any], persona_name: str) -> List[str]:
    """Generate practical insights for operational improvement."""
    insights = []
    
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    hours_processed = exec_summary.get('hours_processed', 0)
    
    # Performance-based insights (simplified)
    if today_total > 0 and avg_total > 0:
        performance_ratio = today_total / avg_total
        
        if performance_ratio < 0.7:
            insights.append("การปรับปรุงประสิทธิภาพการขนส่งและการจัดการคิวรถจะช่วยเพิ่มปริมาณอ้อยได้")
        elif performance_ratio > 1.3:
            insights.append("การรักษาประสิทธิภาพระดับสูงนี้ไว้ได้จะส่งผลดีต่อการผลิตในระยะยาว")
    
    # Time-based practical insights (simplified)
    if time_context['time_of_day'] == 'morning' and hours_processed < 4:
        insights.append("การเตรียมความพร้อมในช่วงเช้าจะช่วยให้การทำงานตลอดทั้งวันเป็นไปอย่างราบรื่น")
    
    elif time_context['time_of_day'] == 'afternoon' and hours_processed >= 8:
        insights.append("ช่วงบ่ายเป็นโอกาสในการเร่งรัดการทำงานเพื่อให้บรรลุเป้าหมาย")
    
    elif time_context['time_of_day'] == 'evening':
        insights.append("การสรุปบทเรียนของวันนี้จะช่วยในการวางแผนการทำงานสำหรับวันพรุ่งนี้")
    
    # Operational insights based on persona (simplified)
    if persona_name == "VOLATILE_PERFORMANCE":
        insights.append("การปรับปรุงระบบการจัดการคิวรถจะช่วยลดความผันผวน")
    
    elif persona_name == "QUALITY_FOCUS":
        insights.append("การรักษามาตรฐานคุณภาพไว้ได้ในขณะที่เพิ่มปริมาณจะเป็นกุญแจสำคัญ")
    
    elif persona_name == "QUANTITY_PUSH":
        insights.append("การปรับปรุงกระบวนการตรวจรับคุณภาพจะช่วยยกระดับคุณภาพอ้อย")
    
    return insights

def _generate_actionable_advice(stats: Dict[str, Any], exec_summary: Dict[str, Any], time_context: Dict[str, Any], persona_name: str) -> List[str]:
    """Generate actionable advice for immediate implementation."""
    advice = []
    
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    hours_processed = exec_summary.get('hours_processed', 0)
    
    # Immediate actions based on current performance (simplified)
    if today_total < avg_total * 0.8:
        advice.append("ควรมีการประชุมฉุกเฉินกับทีมขนส่งเพื่อหาแนวทางเพิ่มปริมาณอ้อย")
        advice.append("ตรวจสอบสถานะรถขนส่งที่ยังไม่เข้าโรงงานและประสานงานเพื่อเร่งรัด")
    
    # Time-specific actionable advice (simplified)
    if time_context['time_of_day'] == 'morning':
        advice.append("ตั้งเป้าหมายและวางแผนการทำงานให้ชัดเจนสำหรับวันนี้")
        advice.append("ตรวจสอบความพร้อมของเครื่องจักรและระบบต่างๆ")
    
    elif time_context['time_of_day'] == 'afternoon':
        if today_total < avg_total * 0.6:
            advice.append("เร่งรัดการขนส่งและประสานงานกับทีมต่างๆ")
        else:
            advice.append("รักษามาตรฐานการทำงานและมองหาโอกาสในการปรับปรุง")
    
    elif time_context['time_of_day'] == 'evening':
        advice.append("สรุปผลงานของวันและวางแผนการทำงานสำหรับวันพรุ่งนี้")
    
    # Persona-specific advice (simplified)
    if persona_name == "CRITICAL":
        advice.append("เรียกประชุมทีมที่เกี่ยวข้องทันทีเพื่อวิเคราะห์สาเหตุและหาทางออก")
    
    elif persona_name == "EXCELLENT":
        advice.append("ถอดบทเรียนความสำเร็จและนำไปสร้างเป็นแนวทางการทำงานที่ดี")
    
    elif persona_name == "WEAK_START":
        advice.append("เตรียมความพร้อมสำหรับปริมาณอ้อยที่คาดว่าจะเพิ่มขึ้นในช่วงบ่ายและค่ำ")
    
    return advice

def _generate_learning_points(stats: Dict[str, Any], exec_summary: Dict[str, Any], persona_name: str, patterns: Dict[str, Any]) -> List[str]:
    """Generate learning points for continuous improvement."""
    learning_points = []
    
    today_total = stats.get('today_total', 0)
    avg_total = stats.get('avg_daily_tons', 0)
    anomaly_score = patterns.get('anomaly_score', 0.0)
    
    # Performance-based learning points (simplified)
    if today_total > 0 and avg_total > 0:
        performance_ratio = today_total / avg_total
        
        if performance_ratio < 0.7:
            learning_points.append("การวางแผนการขนส่งและการประสานงานที่ล้มเหลวเป็นบทเรียนสำคัญที่ต้องนำไปปรับปรุง")
        elif performance_ratio > 1.3:
            learning_points.append("ปัจจัยแห่งความสำเร็จในวันนี้ควรได้รับการบันทึกและนำไปประยุกต์ใช้")
    
    # Pattern-based learning points (simplified)
    if anomaly_score > 0.7:
        learning_points.append("การตรวจพบความผิดปกติแต่เนิ่นๆ และการแก้ไขอย่างรวดเร็วเป็นกุญแจสำคัญ")
    
    # Persona-specific learning points (simplified)
    if persona_name == "VOLATILE_PERFORMANCE":
        learning_points.append("การสร้างความสม่ำเสมอในการรับอ้อยเป็นความท้าทายที่ต้องแก้ไขอย่างเป็นระบบ")
    
    elif persona_name == "QUALITY_FOCUS":
        learning_points.append("การรักษามาตรฐานคุณภาพในขณะที่เพิ่มปริมาณเป็นทักษะที่ต้องพัฒนาอย่างต่อเนื่อง")
    
    elif persona_name == "STABLE_RECOVERY":
        learning_points.append("การแก้ไขปัญหาอย่างเป็นระบบและการทำงานร่วมกันเป็นปัจจัยสำคัญในการฟื้นตัว")
    
    return learning_points

def _generate_conversational_recommendations_v3(best_recommendation: Dict) -> str:
    """
    V4 (Enhanced): แปลงอ็อบเจกต์คำแนะนำให้เป็นโครงสร้างที่ชัดเจนและเข้าใจง่าย
    มีการจัดรูปแบบที่ชาญฉลาด การใช้สี และการแบ่งส่วนที่อ่านง่าย
    """
    if not best_recommendation or not best_recommendation.get('title'):
        return "**สถานการณ์ปกติ**\n\nตอนนี้ทุกอย่างดูเรียบร้อยดีครับ ควรรักษาระดับการทำงานนี้ไว้ และคอยมองหาจุดปรับปรุงเล็กๆ น้อยๆ ต่อไปครับ"

    title = best_recommendation.get('title', '').replace("โดยด่วน", "").replace("ทันที", "")
    actions = best_recommendation.get('actions', [])
    category = best_recommendation.get('category', '')
    priority = best_recommendation.get('priority', 'medium')
    
    # --- แผนที่ปัญหาและผลลัพธ์ที่ปรับปรุงแล้ว ---
    problem_map = {
        'ปริมาณ (Volume)': "ปัญหาปริมาณอ้อยไม่เข้าเป้า",
        'คุณภาพ (Quality)': "ประเด็นเรื่องคุณภาพอ้อย",
        'ความเสถียร (Stability)': "ความผันผวนในการรับอ้อย",
        'การจัดการความเสี่ยง (Risk Management)': "ความเสี่ยงที่ตรวจพบ",
        'การเพิ่มประสิทธิภาพ (Efficiency Improvement)': "โอกาสในการเพิ่มประสิทธิภาพ",
        'การพัฒนาต่อเนื่อง (Improvement)': "การพัฒนาการทำงาน"
    }
    problem_statement = problem_map.get(category, "สถานการณ์ปัจจุบัน")

    outcome_map = {
        'ปริมาณ (Volume)': "จะช่วยให้เรากลับเข้าสู่เป้าหมายการผลิตได้เร็วขึ้น",
        'คุณภาพ (Quality)': "จะช่วยรักษา CCS และเพิ่มมูลค่าผลผลิตของเรา",
        'ความเสถียร (Stability)': "จะทำให้การเดินเครื่องจักรมีเสถียรภาพและประสิทธิภาพสูงสุด",
        'การจัดการความเสี่ยง (Risk Management)': "จะช่วยให้เรารับมือกับสถานการณ์ที่ไม่คาดคิดได้ดีขึ้น",
        'การเพิ่มประสิทธิภาพ (Efficiency Improvement)': "จะช่วยลดต้นทุนและเพิ่มความสามารถในการแข่งขัน",
        'การพัฒนาต่อเนื่อง (Improvement)': "จะทำให้การทำงานโดยรวมราบรื่นและมีมาตรฐานสูงขึ้น"
    }
    expected_outcome = outcome_map.get(category, "จะทำให้การทำงานโดยรวมดีขึ้น")

    # --- สร้างโครงสร้างคำแนะนำแบบใหม่ ---
    rec_parts = []
    
    # หัวข้อหลัก
    rec_parts.append(f"**ข้อเสนอแนะหลัก**\n\n**{title}**")
    
    # บริบทปัญหา
    rec_parts.append(f"\n**บริบท:** {problem_statement}")
    
    # ขั้นตอนการดำเนินการ
    if actions:
        rec_parts.append(f"\n**ขั้นตอนการดำเนินการ:**")
        if len(actions) == 1:
            rec_parts.append(f"• {actions[0]}")
        elif len(actions) <= 3:
            for i, action in enumerate(actions, 1):
                rec_parts.append(f"• {action}")
        else:
            for i, action in enumerate(actions, 1):
                rec_parts.append(f"{i}. {action}")
    
    # ผลลัพธ์ที่คาดหวัง
    rec_parts.append(f"\n**ผลลัพธ์ที่คาดหวัง:**\n{expected_outcome}")
    
    # ระดับความสำคัญ
    priority_text = {
        'high': 'สูง (ควรดำเนินการทันที)',
        'medium': 'ปานกลาง (ควรดำเนินการในเร็วๆ นี้)',
        'low': 'ต่ำ (สามารถดำเนินการได้เมื่อมีเวลา)'
    }
    rec_parts.append(f"\n**ระดับความสำคัญ:** {priority_text.get(priority, 'ปานกลาง')}")

    return "\n".join(rec_parts)