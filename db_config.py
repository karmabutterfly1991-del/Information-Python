# db_config.py (เวอร์ชันที่แนะนำสำหรับ Cloud)
# วิธีนี้ปลอดภัยกว่า โดยอ่านข้อมูลจาก Environment Variables
import pyodbc
import os

def get_connection():
    """
    สร้างการเชื่อมต่อฐานข้อมูลโดยอ่านข้อมูลจาก Environment Variables
    ซึ่งเป็นวิธีที่ปลอดภัยสำหรับเซิร์ฟเวอร์ออนไลน์ (Cloud)
    """
    # อ่านค่าต่างๆ จาก Environment Variables ที่ตั้งค่าไว้บนเซิร์ฟเวอร์ (เช่น Render)
    server = os.getenv('49.231.150.136')
    database = os.getenv('dbPayment')
    uid = os.getenv('sa1')
    pwd = os.getenv('', '')  # ใช้ค่าว่างหากไม่มีการตั้งค่ารหัสผ่าน
    timeout = int(os.getenv('DB_TIMEOUT', 10)) # ใช้ค่าเริ่มต้น 10 วินาทีหากไม่มีการตั้งค่า
    driver = '{ODBC Driver 17 for SQL Server}'

    # ตรวจสอบว่าค่าที่จำเป็นถูกตั้งค่าไว้ครบถ้วนหรือไม่
    if not all([server, database, uid]):
        raise ValueError("กรุณาตั้งค่า Environment Variables (DB_SERVER, DB_DATABASE, DB_UID) บนเซิร์ฟเวอร์ Cloud ให้ครบถ้วน")

    # สร้าง Connection String จาก Environment Variables
    conn_str = (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={uid};"
        f"PWD={pwd};"
        "TrustServerCertificate=yes;"
    )
    
    # ส่งคืนการเชื่อมต่อ
    return pyodbc.connect(conn_str, timeout=timeout)
