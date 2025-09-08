"""
Advanced Hourly Analysis Module
โมดูลการวิเคราะห์ข้อมูลรายชั่วโมงแบบ Advanced สำหรับระบบติดตามอ้อย
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Any
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedHourlyAnalyzer:
    """คลาสสำหรับการวิเคราะห์ข้อมูลรายชั่วโมงแบบ Advanced"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.performance_thresholds = {
            'high': 100,      # ตันต่อชั่วโมง
            'medium': 50,     # ตันต่อชั่วโมง
            'low': 20         # ตันต่อชั่วโมง
        }
    
    def _clean_nan_values(self, data):
        """แปลงค่า NaN เป็น None หรือค่าที่เหมาะสม"""
        if isinstance(data, dict):
            return {k: self._clean_nan_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_nan_values(item) for item in data]
        elif isinstance(data, (int, float)):
            if np.isnan(data):
                return 0
            elif np.isinf(data):
                return 0
            else:
                return data
        else:
            return data
        
    def analyze_hourly_performance(self, hourly_data: List[Dict]) -> Dict[str, Any]:
        """
        วิเคราะห์ประสิทธิภาพการทำงานรายชั่วโมง
        
        Args:
            hourly_data: ข้อมูลรายชั่วโมง
            
        Returns:
            ผลการวิเคราะห์ประสิทธิภาพ
        """
        if not hourly_data:
            return {'error': 'ไม่มีข้อมูลสำหรับการวิเคราะห์'}
        
        # แปลงข้อมูลเป็น DataFrame
        df = pd.DataFrame(hourly_data)
        
        # คำนวณประสิทธิภาพพื้นฐาน
        performance_metrics = self._calculate_performance_metrics(df)
        
        # วิเคราะห์แนวโน้ม
        trend_analysis = self._analyze_hourly_trends(df)
        
        # วิเคราะห์ความแปรปรวน
        variability_analysis = self._analyze_variability(df)
        
        # วิเคราะห์ช่วงเวลาที่มีประสิทธิภาพสูง
        peak_performance = self._identify_peak_performance_periods(df)
        
        # วิเคราะห์ความสัมพันธ์ระหว่างชั่วโมง
        correlation_analysis = self._analyze_hourly_correlations(df)
        
        # คำนวณดัชนีประสิทธิภาพรวม
        overall_performance_index = self._calculate_overall_performance_index(df)
        
        result = {
            'performance_metrics': performance_metrics,
            'trend_analysis': trend_analysis,
            'variability_analysis': variability_analysis,
            'peak_performance': peak_performance,
            'correlation_analysis': correlation_analysis,
            'overall_performance_index': overall_performance_index,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # ทำความสะอาดค่า NaN
        return self._clean_nan_values(result)
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """คำนวณเมตริกประสิทธิภาพพื้นฐาน"""
        total_tons = df['Total_Tons'].sum()
        total_trucks = df['Total_Count'].sum()
        active_hours = len(df[df['Total_Tons'] > 0])
        
        # คำนวณประสิทธิภาพเฉลี่ย
        avg_tons_per_hour = total_tons / 24 if total_tons > 0 else 0
        avg_tons_per_active_hour = total_tons / active_hours if active_hours > 0 else 0
        avg_tons_per_truck = total_tons / total_trucks if total_trucks > 0 else 0
        
        # คำนวณประสิทธิภาพสูงสุดและต่ำสุด
        max_hourly_tons = df['Total_Tons'].max()
        min_hourly_tons = df['Total_Tons'].min()
        
        # คำนวณประสิทธิภาพตามเกณฑ์
        high_performance_hours = len(df[df['Total_Tons'] >= self.performance_thresholds['high']])
        medium_performance_hours = len(df[(df['Total_Tons'] >= self.performance_thresholds['medium']) & 
                                        (df['Total_Tons'] < self.performance_thresholds['high'])])
        low_performance_hours = len(df[(df['Total_Tons'] >= self.performance_thresholds['low']) & 
                                     (df['Total_Tons'] < self.performance_thresholds['medium'])])
        no_performance_hours = len(df[df['Total_Tons'] < self.performance_thresholds['low']])
        
        return {
            'total_tons': total_tons,
            'total_trucks': total_trucks,
            'active_hours': active_hours,
            'avg_tons_per_hour': avg_tons_per_hour,
            'avg_tons_per_active_hour': avg_tons_per_active_hour,
            'avg_tons_per_truck': avg_tons_per_truck,
            'max_hourly_tons': max_hourly_tons,
            'min_hourly_tons': min_hourly_tons,
            'performance_distribution': {
                'high': high_performance_hours,
                'medium': medium_performance_hours,
                'low': low_performance_hours,
                'none': no_performance_hours
            },
            'efficiency_ratio': active_hours / 24 if active_hours > 0 else 0
        }
    
    def _analyze_hourly_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """วิเคราะห์แนวโน้มการทำงานรายชั่วโมง"""
        if len(df) < 3:
            return {'error': 'ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์แนวโน้ม'}
        
        # คำนวณแนวโน้มเชิงเส้น
        x = np.arange(len(df))
        y = df['Total_Tons'].values
        
        # ใช้ linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # วิเคราะห์แนวโน้ม
        if slope > 0.5:
            trend_direction = 'increasing'
            trend_strength = 'strong' if abs(r_value) > 0.7 else 'moderate'
        elif slope < -0.5:
            trend_direction = 'decreasing'
            trend_strength = 'strong' if abs(r_value) > 0.7 else 'moderate'
        else:
            trend_direction = 'stable'
            trend_strength = 'weak'
        
        # คำนวณการเปลี่ยนแปลงเปอร์เซ็นต์
        first_half_avg = df.iloc[:len(df)//2]['Total_Tons'].mean()
        second_half_avg = df.iloc[len(df)//2:]['Total_Tons'].mean()
        change_percentage = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'slope': slope if not np.isnan(slope) else 0,
            'correlation_coefficient': r_value if not np.isnan(r_value) else 0,
            'p_value': p_value if not np.isnan(p_value) else 0,
            'change_percentage': change_percentage if not np.isnan(change_percentage) else 0,
            'first_half_avg': first_half_avg if not np.isnan(first_half_avg) else 0,
            'second_half_avg': second_half_avg if not np.isnan(second_half_avg) else 0
        }
    
    def _analyze_variability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """วิเคราะห์ความแปรปรวนของการทำงาน"""
        if len(df) < 2:
            return {'error': 'ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์ความแปรปรวน'}
        
        tons_data = df['Total_Tons'].values
        
        # คำนวณสถิติความแปรปรวน
        mean_tons = np.mean(tons_data)
        std_tons = np.std(tons_data)
        variance = np.var(tons_data)
        cv = (std_tons / mean_tons * 100) if mean_tons > 0 else 0
        
        # คำนวณช่วงควอไทล์
        q25 = np.percentile(tons_data, 25)
        q50 = np.percentile(tons_data, 50)
        q75 = np.percentile(tons_data, 75)
        iqr = q75 - q25
        
        # วิเคราะห์ความเสถียร
        if cv < 20:
            stability_level = 'very_stable'
        elif cv < 40:
            stability_level = 'stable'
        elif cv < 60:
            stability_level = 'moderate'
        else:
            stability_level = 'unstable'
        
        # ตรวจสอบ outliers
        outliers = self._detect_outliers(tons_data)
        
        return {
            'mean': mean_tons if not np.isnan(mean_tons) else 0,
            'std_deviation': std_tons if not np.isnan(std_tons) else 0,
            'variance': variance if not np.isnan(variance) else 0,
            'coefficient_of_variation': cv if not np.isnan(cv) else 0,
            'quartiles': {
                'q25': q25 if not np.isnan(q25) else 0,
                'q50': q50 if not np.isnan(q50) else 0,
                'q75': q75 if not np.isnan(q75) else 0,
                'iqr': iqr if not np.isnan(iqr) else 0
            },
            'stability_level': stability_level,
            'outliers_count': len(outliers),
            'outliers_indices': outliers.tolist()
        }
    
    def _detect_outliers(self, data: np.ndarray) -> np.ndarray:
        """ตรวจสอบ outliers โดยใช้ IQR method"""
        if len(data) == 0:
            return np.array([])
            
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        iqr = q75 - q25
        
        # ตรวจสอบ NaN
        if np.isnan(q25) or np.isnan(q75) or np.isnan(iqr):
            return np.array([])
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        return outliers
    
    def _identify_peak_performance_periods(self, df: pd.DataFrame) -> Dict[str, Any]:
        """ระบุช่วงเวลาที่มีประสิทธิภาพสูง"""
        if len(df) < 3:
            return {'error': 'ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์ช่วงเวลาสูงสุด'}
        
        # คำนวณ moving average
        window_size = min(3, len(df))
        df['Moving_Avg'] = df['Total_Tons'].rolling(window=window_size, center=True).mean()
        
        # หาช่วงเวลาที่มีประสิทธิภาพสูง
        threshold = df['Total_Tons'].quantile(0.75)  # 75th percentile
        peak_periods = df[df['Total_Tons'] >= threshold]
        
        # วิเคราะห์การกระจายของช่วงเวลาสูงสุด
        peak_hours = peak_periods.index.tolist()
        
        # คำนวณความต่อเนื่องของช่วงเวลาสูงสุด
        consecutive_peaks = self._find_consecutive_periods(peak_hours)
        
        # วิเคราะห์ช่วงเวลาที่ดีที่สุด
        best_period = df.loc[df['Total_Tons'].idxmax()]
        
        return {
            'peak_threshold': threshold if not np.isnan(threshold) else 0,
            'peak_periods_count': len(peak_periods),
            'peak_hours': peak_hours,
            'consecutive_peaks': consecutive_peaks,
            'best_period': {
                'hour': best_period.name,
                'tons': best_period['Total_Tons'] if not np.isnan(best_period['Total_Tons']) else 0,
                'time_label': best_period.get('Time', 'N/A')
            },
            'peak_performance_ratio': len(peak_periods) / len(df) if len(df) > 0 else 0
        }
    
    def _find_consecutive_periods(self, peak_hours: List[int]) -> List[List[int]]:
        """หาช่วงเวลาต่อเนื่องที่มีประสิทธิภาพสูง"""
        if not peak_hours:
            return []
        
        consecutive_groups = []
        current_group = [peak_hours[0]]
        
        for i in range(1, len(peak_hours)):
            if peak_hours[i] == peak_hours[i-1] + 1:
                current_group.append(peak_hours[i])
            else:
                if len(current_group) > 1:
                    consecutive_groups.append(current_group)
                current_group = [peak_hours[i]]
        
        if len(current_group) > 1:
            consecutive_groups.append(current_group)
        
        return consecutive_groups
    
    def _analyze_hourly_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """วิเคราะห์ความสัมพันธ์ระหว่างชั่วโมง"""
        if len(df) < 4:
            return {'error': 'ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์ความสัมพันธ์'}
        
        # คำนวณความสัมพันธ์ระหว่างตัวแปรต่างๆ
        correlation_matrix = df[['A_Tons', 'B_Tons', 'Fresh_Tons', 'Burnt_Tons', 'Total_Tons']].corr()
        
        # วิเคราะห์ความสัมพันธ์ระหว่าง Line A และ B
        ab_correlation = correlation_matrix.loc['A_Tons', 'B_Tons']
        
        # วิเคราะห์ความสัมพันธ์ระหว่าง Fresh และ Burnt
        fresh_burnt_correlation = correlation_matrix.loc['Fresh_Tons', 'Burnt_Tons']
        
        # วิเคราะห์ความสัมพันธ์กับ Total
        a_total_correlation = correlation_matrix.loc['A_Tons', 'Total_Tons']
        b_total_correlation = correlation_matrix.loc['B_Tons', 'Total_Tons']
        
        # วิเคราะห์ความสมดุลระหว่าง Line A และ B
        balance_ratio = df['A_Tons'].sum() / df['B_Tons'].sum() if df['B_Tons'].sum() > 0 else 0
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'ab_correlation': ab_correlation if not np.isnan(ab_correlation) else 0,
            'fresh_burnt_correlation': fresh_burnt_correlation if not np.isnan(fresh_burnt_correlation) else 0,
            'a_total_correlation': a_total_correlation if not np.isnan(a_total_correlation) else 0,
            'b_total_correlation': b_total_correlation if not np.isnan(b_total_correlation) else 0,
            'balance_ratio': balance_ratio if not np.isnan(balance_ratio) else 0,
            'balance_assessment': 'balanced' if 0.8 <= balance_ratio <= 1.2 else 'unbalanced'
        }
    
    def _calculate_overall_performance_index(self, df: pd.DataFrame) -> Dict[str, Any]:
        """คำนวณดัชนีประสิทธิภาพรวม"""
        if len(df) == 0:
            return {'error': 'ไม่มีข้อมูลสำหรับการคำนวณดัชนี'}
        
        # คำนวณดัชนีย่อยต่างๆ
        efficiency_score = self._calculate_efficiency_score(df)
        consistency_score = self._calculate_consistency_score(df)
        productivity_score = self._calculate_productivity_score(df)
        balance_score = self._calculate_balance_score(df)
        
        # คำนวณดัชนีรวม (weighted average)
        weights = {
            'efficiency': 0.3,
            'consistency': 0.25,
            'productivity': 0.25,
            'balance': 0.2
        }
        
        overall_index = (
            efficiency_score * weights['efficiency'] +
            consistency_score * weights['consistency'] +
            productivity_score * weights['productivity'] +
            balance_score * weights['balance']
        )
        
        # กำหนดระดับประสิทธิภาพ
        if overall_index >= 80:
            performance_level = 'excellent'
        elif overall_index >= 70:
            performance_level = 'good'
        elif overall_index >= 60:
            performance_level = 'fair'
        elif overall_index >= 50:
            performance_level = 'poor'
        else:
            performance_level = 'very_poor'
        
        return {
            'overall_index': overall_index if not np.isnan(overall_index) else 0,
            'performance_level': performance_level,
            'component_scores': {
                'efficiency': efficiency_score if not np.isnan(efficiency_score) else 0,
                'consistency': consistency_score if not np.isnan(consistency_score) else 0,
                'productivity': productivity_score if not np.isnan(productivity_score) else 0,
                'balance': balance_score if not np.isnan(balance_score) else 0
            },
            'weights': weights
        }
    
    def _calculate_efficiency_score(self, df: pd.DataFrame) -> float:
        """คำนวณคะแนนประสิทธิภาพ"""
        active_hours = len(df[df['Total_Tons'] > 0])
        efficiency_ratio = active_hours / 24 if active_hours > 0 else 0
        score = min(100, efficiency_ratio * 100)
        return score if not np.isnan(score) else 0
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """คำนวณคะแนนความสม่ำเสมอ"""
        if len(df) < 2:
            return 0
        
        cv = (df['Total_Tons'].std() / df['Total_Tons'].mean() * 100) if df['Total_Tons'].mean() > 0 else 100
        # ยิ่ง CV ต่ำ ยิ่งสม่ำเสมอ
        consistency_score = max(0, 100 - cv)
        score = min(100, consistency_score)
        return score if not np.isnan(score) else 0
    
    def _calculate_productivity_score(self, df: pd.DataFrame) -> float:
        """คำนวณคะแนนผลิตภาพ"""
        total_tons = df['Total_Tons'].sum()
        total_trucks = df['Total_Count'].sum()
        
        if total_trucks == 0:
            return 0
        
        avg_tons_per_truck = total_tons / total_trucks
        # ใช้เกณฑ์ 10 ตันต่อคันเป็นเกณฑ์ดี
        productivity_score = min(100, (avg_tons_per_truck / 10) * 100)
        return productivity_score if not np.isnan(productivity_score) else 0
    
    def _calculate_balance_score(self, df: pd.DataFrame) -> float:
        """คำนวณคะแนนความสมดุล"""
        total_a = df['A_Tons'].sum()
        total_b = df['B_Tons'].sum()
        
        if total_a + total_b == 0:
            return 0
        
        balance_ratio = total_a / (total_a + total_b)
        # ยิ่งใกล้ 0.5 ยิ่งสมดุล
        balance_deviation = abs(balance_ratio - 0.5)
        balance_score = max(0, 100 - (balance_deviation * 200))
        return balance_score if not np.isnan(balance_score) else 0
    
    def generate_hourly_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """สร้างข้อมูลเชิงลึกจากการวิเคราะห์"""
        insights = []
        
        # ข้อมูลประสิทธิภาพ
        perf_metrics = analysis_result.get('performance_metrics', {})
        if perf_metrics:
            total_tons = perf_metrics.get('total_tons', 0)
            active_hours = perf_metrics.get('active_hours', 0)
            efficiency_ratio = perf_metrics.get('efficiency_ratio', 0)
            
            insights.append(f"วันนี้มีการทำงาน {active_hours} ชั่วโมง จาก 24 ชั่วโมง (ประสิทธิภาพ {efficiency_ratio:.1%})")
            
            if total_tons > 0:
                avg_tons_per_hour = perf_metrics.get('avg_tons_per_hour', 0)
                insights.append(f"ผลิตภาพเฉลี่ย {avg_tons_per_hour:.1f} ตันต่อชั่วโมง")
        
        # ข้อมูลแนวโน้ม
        trend = analysis_result.get('trend_analysis', {})
        if trend and 'trend_direction' in trend:
            direction = trend['trend_direction']
            strength = trend['trend_strength']
            change_pct = trend.get('change_percentage', 0)
            
            if direction == 'increasing':
                insights.append(f"แนวโน้มการผลิตเพิ่มขึ้น ({strength}) - เพิ่มขึ้น {change_pct:.1f}%")
            elif direction == 'decreasing':
                insights.append(f"แนวโน้มการผลิตลดลง ({strength}) - ลดลง {abs(change_pct):.1f}%")
            else:
                insights.append("แนวโน้มการผลิตคงที่")
        
        # ข้อมูลความเสถียร
        variability = analysis_result.get('variability_analysis', {})
        if variability and 'stability_level' in variability:
            stability = variability['stability_level']
            cv = variability.get('coefficient_of_variation', 0)
            
            if stability == 'very_stable':
                insights.append(f"การทำงานมีความเสถียรสูงมาก (CV: {cv:.1f}%)")
            elif stability == 'stable':
                insights.append(f"การทำงานมีความเสถียรดี (CV: {cv:.1f}%)")
            elif stability == 'unstable':
                insights.append(f"การทำงานมีความแปรปรวนสูง (CV: {cv:.1f}%)")
        
        # ข้อมูลช่วงเวลาสูงสุด
        peak = analysis_result.get('peak_performance', {})
        if peak and 'best_period' in peak:
            best_period = peak['best_period']
            insights.append(f"ช่วงเวลาที่มีประสิทธิภาพสูงสุด: {best_period.get('time_label', 'N/A')} ({best_period.get('tons', 0):.1f} ตัน)")
        
        # ข้อมูลดัชนีรวม
        overall = analysis_result.get('overall_performance_index', {})
        if overall and 'overall_index' in overall:
            index = overall['overall_index']
            level = overall['performance_level']
            insights.append(f"ดัชนีประสิทธิภาพรวม: {index:.1f}/100 ({level})")
        
        return insights
    
    def predict_next_hour_performance(self, hourly_data: List[Dict], current_hour: int) -> Dict[str, Any]:
        """พยากรณ์ประสิทธิภาพชั่วโมงถัดไป"""
        if len(hourly_data) < 3:
            return {'error': 'ข้อมูลไม่เพียงพอสำหรับการพยากรณ์'}
        
        # แปลงข้อมูลเป็น DataFrame
        df = pd.DataFrame(hourly_data)
        
        # ใช้ข้อมูล 3 ชั่วโมงล่าสุดสำหรับการพยากรณ์
        recent_data = df.tail(3)
        
        # คำนวณค่าเฉลี่ยเคลื่อนที่
        moving_avg = recent_data['Total_Tons'].mean()
        
        # คำนวณแนวโน้ม
        if len(recent_data) >= 2:
            trend = recent_data['Total_Tons'].iloc[-1] - recent_data['Total_Tons'].iloc[0]
            trend_per_hour = trend / len(recent_data)
        else:
            trend_per_hour = 0
        
        # พยากรณ์ชั่วโมงถัดไป
        predicted_tons = moving_avg + trend_per_hour
        
        # คำนวณช่วงความเชื่อมั่น
        std_dev = recent_data['Total_Tons'].std()
        confidence_interval = {
            'lower': max(0, predicted_tons - 1.96 * std_dev),
            'upper': predicted_tons + 1.96 * std_dev
        }
        
        # กำหนดระดับความเชื่อมั่น
        if std_dev < 10:
            confidence_level = 'high'
        elif std_dev < 20:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        result = {
            'predicted_tons': predicted_tons if not np.isnan(predicted_tons) else 0,
            'confidence_interval': {
                'lower': confidence_interval['lower'] if not np.isnan(confidence_interval['lower']) else 0,
                'upper': confidence_interval['upper'] if not np.isnan(confidence_interval['upper']) else 0
            },
            'confidence_level': confidence_level,
            'trend': trend_per_hour if not np.isnan(trend_per_hour) else 0,
            'moving_average': moving_avg if not np.isnan(moving_avg) else 0,
            'standard_deviation': std_dev if not np.isnan(std_dev) else 0,
            'prediction_hour': current_hour + 1
        }
        
        return self._clean_nan_values(result)
    
    def generate_hourly_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """สร้างคำแนะนำจากการวิเคราะห์"""
        recommendations = []
        
        # ตรวจสอบประสิทธิภาพ
        perf_metrics = analysis_result.get('performance_metrics', {})
        if perf_metrics:
            efficiency_ratio = perf_metrics.get('efficiency_ratio', 0)
            if efficiency_ratio < 0.5:
                recommendations.append({
                    'type': 'efficiency',
                    'priority': 'high',
                    'title': 'ปรับปรุงประสิทธิภาพการทำงาน',
                    'description': f'ประสิทธิภาพการทำงานอยู่ที่ {efficiency_ratio:.1%} ซึ่งต่ำกว่าเกณฑ์ ควรตรวจสอบสาเหตุที่ทำให้มีชั่วโมงว่าง'
                })
        
        # ตรวจสอบความเสถียร
        variability = analysis_result.get('variability_analysis', {})
        if variability and variability.get('stability_level') == 'unstable':
            recommendations.append({
                'type': 'stability',
                'priority': 'medium',
                'title': 'ปรับปรุงความเสถียรการผลิต',
                'description': 'การผลิตมีความแปรปรวนสูง ควรหาสาเหตุและปรับปรุงกระบวนการทำงาน'
            })
        
        # ตรวจสอบความสมดุล
        correlation = analysis_result.get('correlation_analysis', {})
        if correlation and correlation.get('balance_assessment') == 'unbalanced':
            balance_ratio = correlation.get('balance_ratio', 1)
            recommendations.append({
                'type': 'balance',
                'priority': 'medium',
                'title': 'ปรับสมดุลการผลิตระหว่าง Line A และ B',
                'description': f'อัตราส่วนการผลิตระหว่าง Line A:B = {balance_ratio:.2f}:1 ควรปรับให้สมดุลมากขึ้น'
            })
        
        # ตรวจสอบช่วงเวลาสูงสุด
        peak = analysis_result.get('peak_performance', {})
        if peak and peak.get('peak_periods_count', 0) < 6:
            recommendations.append({
                'type': 'peak_performance',
                'priority': 'low',
                'title': 'เพิ่มช่วงเวลาประสิทธิภาพสูง',
                'description': 'มีช่วงเวลาประสิทธิภาพสูงเพียงไม่กี่ชั่วโมง ควรหาวิธีเพิ่มประสิทธิภาพในชั่วโมงอื่นๆ'
            })
        
        return recommendations

# ฟังก์ชันช่วยสำหรับการใช้งาน
def analyze_hourly_data_advanced(hourly_data: List[Dict]) -> Dict[str, Any]:
    """
    ฟังก์ชันหลักสำหรับการวิเคราะห์ข้อมูลรายชั่วโมงแบบ Advanced
    
    Args:
        hourly_data: ข้อมูลรายชั่วโมง
        
    Returns:
        ผลการวิเคราะห์แบบ Advanced
    """
    analyzer = AdvancedHourlyAnalyzer()
    
    # ทำการวิเคราะห์
    analysis_result = analyzer.analyze_hourly_performance(hourly_data)
    
    # สร้างข้อมูลเชิงลึก
    insights = analyzer.generate_hourly_insights(analysis_result)
    
    # สร้างคำแนะนำ
    recommendations = analyzer.generate_hourly_recommendations(analysis_result)
    
    # รวมผลลัพธ์
    result = {
        'analysis': analysis_result,
        'insights': insights,
        'recommendations': recommendations,
        'generated_at': datetime.now().isoformat()
    }
    
    # ทำความสะอาดค่า NaN
    analyzer = AdvancedHourlyAnalyzer()
    return analyzer._clean_nan_values(result)

def predict_hourly_performance(hourly_data: List[Dict], current_hour: int) -> Dict[str, Any]:
    """
    ฟังก์ชันสำหรับพยากรณ์ประสิทธิภาพชั่วโมงถัดไป
    
    Args:
        hourly_data: ข้อมูลรายชั่วโมง
        current_hour: ชั่วโมงปัจจุบัน
        
    Returns:
        ผลการพยากรณ์
    """
    analyzer = AdvancedHourlyAnalyzer()
    result = analyzer.predict_next_hour_performance(hourly_data, current_hour)
    return analyzer._clean_nan_values(result)
