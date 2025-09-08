"""
Local AI Configuration for Sugar Cane Analysis System
===================================================

This file contains configuration and setup instructions for the local AI-enhanced analysis system.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class LocalAISetup:
    """Local AI Setup and Configuration Helper"""
    
    def __init__(self):
        self.model_path = "ai_models/"
        self.scaler_path = "ai_models/"
        self.min_data_points = 10
        self.prediction_horizon = 7  # days
        self.confidence_threshold = 0.7
        self.quantity_model = None
        self.quality_model = None
        self.persona_classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def ensure_model_directory(self):
        """Ensure model directory exists"""
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.scaler_path, exist_ok=True)
    
    def extract_features(self, stats, exec_summary, trend_data):
        """Extract features for ML models"""
        features = []
        
        # Basic statistics
        features.extend([
            stats.get('today_total', 0),
            stats.get('avg_daily_tons', 0),
            stats.get('type_1_percent', 0),
            stats.get('avg_fresh_percent', 0),
            exec_summary.get('hours_processed', 0),
            exec_summary.get('peak_hour_tons', 0),
            trend_data.get('tons_trend_percent', 0),
            trend_data.get('fresh_trend_percent', 0)
        ])
        
        # Derived features
        today_total = stats.get('today_total', 0)
        avg_total = stats.get('avg_daily_tons', 0)
        today_fresh = stats.get('type_1_percent', 0)
        avg_fresh = stats.get('avg_fresh_percent', 0)
        
        # Performance ratios
        quantity_ratio = (today_total / avg_total) if avg_total > 0 else 1.0
        quality_ratio = (today_fresh / avg_fresh) if avg_fresh > 0 else 1.0
        
        # Volatility measures
        hours = exec_summary.get('hours_processed', 0)
        peak_tons = exec_summary.get('peak_hour_tons', 0)
        avg_hourly = (today_total / hours) if hours > 0 else 0
        peak_ratio = (peak_tons / avg_hourly) if avg_hourly > 0 else 1.0
        
        features.extend([
            quantity_ratio,
            quality_ratio,
            peak_ratio,
            abs(trend_data.get('tons_trend_percent', 0)),
            abs(trend_data.get('fresh_trend_percent', 0))
        ])
        
        # Time-based features
        from datetime import datetime
        current_month = datetime.now().month
        is_crushing_season = 1.0 if current_month in [12, 1, 2, 3, 4] else 0.0
        features.append(is_crushing_season)
        
        return np.array(features).reshape(1, -1)
    
    def train_models(self, historical_data):
        """Train local ML models with historical data"""
        if len(historical_data) < self.min_data_points:
            return False
            
        try:
            X = []
            y_quantity = []
            y_quality = []
            y_persona = []
            
            for data_point in historical_data:
                features = self.extract_features(
                    data_point.get('stats', {}),
                    data_point.get('exec_summary', {}),
                    data_point.get('trend_data', {})
                )
                X.append(features.flatten())
                
                # Target variables
                stats = data_point.get('stats', {})
                y_quantity.append(stats.get('today_total', 0))
                y_quality.append(stats.get('type_1_percent', 0))
                
                # Persona classification (simplified)
                scores = data_point.get('scores', {'quantity': 3, 'quality': 3, 'stability': 3})
                persona_score = (scores['quantity'] + scores['quality'] + scores['stability']) / 3
                y_persona.append(1 if persona_score >= 4 else 0)  # Binary: good/bad performance
            
            X = np.array(X)
            y_quantity = np.array(y_quantity)
            y_quality = np.array(y_quality)
            y_persona = np.array(y_persona)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.quantity_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.quality_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.persona_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            
            self.quantity_model.fit(X_scaled, y_quantity)
            self.quality_model.fit(X_scaled, y_quality)
            self.persona_classifier.fit(X_scaled, y_persona)
            
            self.is_trained = True
            
            # Save models
            self.save_models()
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def save_models(self):
        """Save trained models"""
        try:
            with open(f"{self.model_path}quantity_model.pkl", 'wb') as f:
                pickle.dump(self.quantity_model, f)
            with open(f"{self.model_path}quality_model.pkl", 'wb') as f:
                pickle.dump(self.quality_model, f)
            with open(f"{self.model_path}persona_classifier.pkl", 'wb') as f:
                pickle.dump(self.persona_classifier, f)
            with open(f"{self.scaler_path}scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            print(f"Model save error: {e}")
    
    def load_models(self):
        """Load trained models"""
        try:
            with open(f"{self.model_path}quantity_model.pkl", 'rb') as f:
                self.quantity_model = pickle.load(f)
            with open(f"{self.model_path}quality_model.pkl", 'rb') as f:
                self.quality_model = pickle.load(f)
            with open(f"{self.model_path}persona_classifier.pkl", 'rb') as f:
                self.persona_classifier = pickle.load(f)
            with open(f"{self.scaler_path}scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Model load error: {e}")
            return False
    
    def analyze_performance(self, stats, exec_summary, trend_data):
        """Analyze performance using local AI"""
        if not self.is_trained:
            return {"ai_insights": None, "ai_recommendations": None}
        
        try:
            features = self.extract_features(stats, exec_summary, trend_data)
            features_scaled = self.scaler.transform(features)
            
            # Get predictions
            predicted_quantity = self.quantity_model.predict(features_scaled)[0]
            predicted_quality = self.quality_model.predict(features_scaled)[0]
            performance_prob = self.persona_classifier.predict_proba(features_scaled)[0]
            
            # Generate insights based on predictions
            insights = []
            recommendations = []
            
            actual_quantity = stats.get('today_total', 0)
            actual_quality = stats.get('type_1_percent', 0)
            
            # Quantity analysis
            if actual_quantity > predicted_quantity * 1.1:
                insights.append("ปริมาณอ้อยวันนี้สูงกว่าที่ AI คาดการณ์ไว้ แสดงถึงประสิทธิภาพการจัดส่งที่ดี")
            elif actual_quantity < predicted_quantity * 0.9:
                insights.append("ปริมาณอ้อยวันนี้ต่ำกว่าที่ AI คาดการณ์ อาจมีปัญหาการจัดส่งหรือสภาพอากาศ")
                recommendations.append("ตรวจสอบปัญหาการขนส่งและสภาพอากาศในพื้นที่")
            
            # Quality analysis
            if actual_quality > predicted_quality + 5:
                insights.append("คุณภาพอ้อยสดดีกว่าที่ AI คาดการณ์ แสดงถึงการจัดการที่ดี")
            elif actual_quality < predicted_quality - 5:
                insights.append("คุณภาพอ้อยสดต่ำกว่าที่ AI คาดการณ์ ควรตรวจสอบการจัดการอ้อยไฟ")
                recommendations.append("ปรับปรุงการจัดการอ้อยไฟและเพิ่มการตรวจสอบคุณภาพ")
            
            # Performance probability
            good_performance_prob = performance_prob[1] if len(performance_prob) > 1 else 0.5
            if good_performance_prob > 0.8:
                insights.append("AI ประเมินว่าประสิทธิภาพวันนี้อยู่ในเกณฑ์ดีมาก")
            elif good_performance_prob < 0.3:
                insights.append("AI ประเมินว่าประสิทธิภาพวันนี้ต้องการการปรับปรุง")
                recommendations.append("วิเคราะห์หาสาเหตุและปรับปรุงกระบวนการทำงาน")
            
            return {
                "ai_insights": insights if insights else ["AI วิเคราะห์ว่าประสิทธิภาพอยู่ในเกณฑ์ปกติ"],
                "ai_recommendations": recommendations if recommendations else ["รักษามาตรฐานการทำงานปัจจุบัน"]
            }
            
        except Exception as e:
            print(f"Local AI analysis error: {e}")
            return {"ai_insights": None, "ai_recommendations": None}
    
    def predict_trends(self, stats, trend_data):
        """Predict future trends using local AI"""
        if not self.is_trained:
            return {"prediction": None, "confidence": None}
        
        try:
            # Create mock exec_summary for feature extraction
            mock_exec_summary = {
                'hours_processed': 12,
                'peak_hour_tons': stats.get('today_total', 0) / 12
            }
            
            features = self.extract_features(stats, mock_exec_summary, trend_data)
            features_scaled = self.scaler.transform(features)
            
            # Get predictions
            predicted_quantity = self.quantity_model.predict(features_scaled)[0]
            predicted_quality = self.quality_model.predict(features_scaled)[0]
            
            current_quantity = stats.get('today_total', 0)
            current_quality = stats.get('type_1_percent', 0)
            
            # Calculate trend predictions
            quantity_change = ((predicted_quantity - current_quantity) / current_quantity * 100) if current_quantity > 0 else 0
            quality_change = predicted_quality - current_quality
            
            # Determine confidence
            confidence = "medium"
            if abs(quantity_change) < 5 and abs(quality_change) < 2:
                confidence = "high"
            elif abs(quantity_change) > 15 or abs(quality_change) > 8:
                confidence = "low"
            
            # Generate prediction text
            if quantity_change > 5 and quality_change > 2:
                prediction = f"AI คาดการณ์ว่าปริมาณจะเพิ่มขึ้น {quantity_change:.1f}% และคุณภาพจะดีขึ้น {quality_change:.1f}% ในอีก 3-7 วัน"
            elif quantity_change < -5 or quality_change < -2:
                prediction = f"AI คาดการณ์ว่าปริมาณจะลดลง {abs(quantity_change):.1f}% และคุณภาพจะลดลง {abs(quality_change):.1f}% ในอีก 3-7 วัน"
            else:
                prediction = "AI คาดการณ์ว่าประสิทธิภาพจะคงที่ในอีก 3-7 วัน"
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "factors": [
                    f"แนวโน้มปริมาณปัจจุบัน: {trend_data.get('tons_trend_percent', 0):+.1f}%",
                    f"แนวโน้มคุณภาพปัจจุบัน: {trend_data.get('fresh_trend_percent', 0):+.1f}%"
                ]
            }
            
        except Exception as e:
            print(f"Local AI trend prediction error: {e}")
            return {"prediction": None, "confidence": None}

def print_setup_instructions():
    """Print setup instructions for local AI functionality"""
    print("🤖 Local AI-Enhanced Sugar Cane Analysis Setup")
    print("=" * 55)
    print()
    print("To enable local AI functionality, follow these steps:")
    print()
    print("1. Install Dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Train Local AI Models:")
    print("   - Collect historical data (minimum 10 data points)")
    print("   - Use train_local_ai() function to train models")
    print()
    print("3. Test Local AI:")
    print("   python -c \"from local_ai_config import LocalAISetup; ai = LocalAISetup(); print(ai.load_models())\"")
    print()
    print("✅ Local AI features will be automatically enabled once models are trained!")
    print()

if __name__ == "__main__":
    print_setup_instructions()
    
    # Test current setup
    ai_setup = LocalAISetup()
    ai_setup.ensure_model_directory()
    
    print("Local AI Status:")
    print(f"  Models Trained: {'✅' if ai_setup.load_models() else '❌'}")
    print(f"  Model Path: {ai_setup.model_path}")
    print(f"  Min Data Points: {ai_setup.min_data_points}")
    print(f"  Prediction Horizon: {ai_setup.prediction_horizon} days")
    print()
    
    if ai_setup.is_trained:
        print("✅ Local AI models loaded successfully")
    else:
        print("❌ Please train local AI models first")
