"""
AI Configuration for Sugar Cane Analysis System
==============================================

This file contains configuration and setup instructions for the AI-enhanced analysis system.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AISetup:
    """AI Setup and Configuration Helper"""
    
    @staticmethod
    def setup_environment():
        """Setup environment variables for AI functionality"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  OpenAI API Key not found!")
            print("Please set your OpenAI API key:")
            print("1. Get your API key from https://platform.openai.com/api-keys")
            print("2. Set environment variable: OPENAI_API_KEY=your_key_here")
            print("3. Or create a .env file with: OPENAI_API_KEY=your_key_here")
            return False
        return True
    
    @staticmethod
    def get_ai_status():
        """Get current AI configuration status"""
        api_key = os.getenv('OPENAI_API_KEY')
        return {
            "api_key_configured": bool(api_key),
            "api_key_length": len(api_key) if api_key else 0,
            "model": "gpt-4-turbo-preview",
            "max_tokens": 1000,
            "temperature": 0.7
        }
    
    @staticmethod
    def test_ai_connection():
        """Test AI connection and functionality"""
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": "Hello, this is a test message."}],
                max_tokens=10
            )
            return True, "AI connection successful"
        except Exception as e:
            return False, f"AI connection failed: {str(e)}"

# AI Configuration Constants
AI_CONFIG = {
    "DEFAULT_MODEL": "gpt-4-turbo-preview",
    "DEFAULT_MAX_TOKENS": 1000,
    "DEFAULT_TEMPERATURE": 0.7,
    "ANALYSIS_PROMPT_TEMPLATE": """
คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์ประสิทธิภาพการผลิตอ้อยในโรงงานน้ำตาล

ข้อมูลการผลิตวันนี้:
- ปริมาณอ้อยรวม: {today_total:,.0f} ตัน
- ค่าเฉลี่ยย้อนหลัง: {avg_daily_tons:,.0f} ตัน
- สัดส่วนอ้อยสด: {type_1_percent:.1f}%
- ค่าเฉลี่ยอ้อยสด: {avg_fresh_percent:.1f}%
- ชั่วโมงการทำงาน: {hours_processed} ชั่วโมง
- ชั่วโมงสูงสุด: {peak_hour_tons:,.0f} ตัน
- แนวโน้มปริมาณ 2 สัปดาห์: {tons_trend:+.1f}%
- แนวโน้มคุณภาพ 2 สัปดาห์: {fresh_trend:+.1f}%

กรุณาวิเคราะห์และให้:
1. ข้อมูลเชิงลึก (insights) 3-4 ข้อที่สำคัญ
2. คำแนะนำเชิงปฏิบัติ (actionable recommendations) 2-3 ข้อ

ตอบเป็น JSON format:
{{
    "insights": ["insight1", "insight2", "insight3"],
    "recommendations": ["rec1", "rec2"]
}}
""",
    "PERSONA_PROMPT_TEMPLATE": """
วิเคราะห์ข้อมูลการผลิตอ้อยและสร้าง persona ที่เหมาะสม:

คะแนนประสิทธิภาพ:
- ปริมาณ: {quantity_score}/5
- คุณภาพ: {quality_score}/5  
- เสถียรภาพ: {stability_score}/5

ข้อมูลสำคัญ:
- ปริมาณวันนี้: {today_total:,.0f} ตัน
- สัดส่วนอ้อยสด: {type_1_percent:.1f}%
- แนวโน้มปริมาณ: {tons_trend:+.1f}%
- แนวโน้มคุณภาพ: {fresh_trend:+.1f}%

สร้าง persona ใหม่ที่เหมาะสมที่สุด โดยตอบเป็น JSON:
{{
    "status": "ชื่อสถานการณ์",
    "color": "text-color-class",
    "icon": "bootstrap-icon-class",
    "comment": "คำอธิบายสถานการณ์",
    "recommendation": "คำแนะนำเชิงปฏิบัติ"
}}
""",
    "TREND_PREDICTION_TEMPLATE": """
ทำนายแนวโน้มการผลิตอ้อยในอีก 3-7 วันข้างหน้า:

ข้อมูลปัจจุบัน:
- ปริมาณเฉลี่ย: {avg_daily_tons:,.0f} ตัน
- แนวโน้มปริมาณ: {tons_trend:+.1f}%
- แนวโน้มคุณภาพ: {fresh_trend:+.1f}%
- ฤดูกาล: {season}

ตอบเป็น JSON:
{{
    "prediction": "คำทำนายแนวโน้ม",
    "confidence": "high/medium/low",
    "factors": ["ปัจจัย1", "ปัจจัย2"]
}}
"""
}

def print_setup_instructions():
    """Print setup instructions for AI functionality"""
    print("🤖 AI-Enhanced Sugar Cane Analysis Setup")
    print("=" * 50)
    print()
    print("To enable AI functionality, follow these steps:")
    print()
    print("1. Get OpenAI API Key:")
    print("   - Visit: https://platform.openai.com/api-keys")
    print("   - Create a new API key")
    print()
    print("2. Set Environment Variable:")
    print("   Option A - Set in system:")
    print("   export OPENAI_API_KEY=your_key_here")
    print()
    print("   Option B - Create .env file:")
    print("   echo 'OPENAI_API_KEY=your_key_here' > .env")
    print()
    print("3. Install Dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("4. Test AI Connection:")
    print("   python -c \"from ai_config import AISetup; print(AISetup.test_ai_connection())\"")
    print()
    print("✅ AI features will be automatically enabled once API key is configured!")
    print()

if __name__ == "__main__":
    print_setup_instructions()
    
    # Test current setup
    status = AISetup.get_ai_status()
    print("Current AI Status:")
    print(f"  API Key Configured: {'✅' if status['api_key_configured'] else '❌'}")
    print(f"  Model: {status['model']}")
    print(f"  Max Tokens: {status['max_tokens']}")
    print(f"  Temperature: {status['temperature']}")
    print()
    
    if status['api_key_configured']:
        success, message = AISetup.test_ai_connection()
        print(f"AI Connection Test: {'✅' if success else '❌'}")
        print(f"Message: {message}")
    else:
        print("❌ Please configure OpenAI API key first")
