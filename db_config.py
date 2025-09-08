# db_config.py
import pyodbc
import configparser
import os

def get_connection():
    config = configparser.ConfigParser()
    # ตรวจสอบว่าไฟล์ config.ini อยู่ใน path เดียวกันกับ script หรือไม่
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    if not os.path.exists(config_path):
        raise FileNotFoundError("ไม่พบไฟล์ config.ini กรุณาตรวจสอบว่าไฟล์อยู่ในตำแหน่งที่ถูกต้อง")

    config.read(config_path)
    db_config = config['DATABASE']

    conn_str = (
        f"DRIVER={db_config.get('DRIVER', '{ODBC Driver 17 for SQL Server}')};"
        f"SERVER={db_config.get('SERVER')};"
        f"DATABASE={db_config.get('DATABASE')};"
        f"UID={db_config.get('UID')};"
        f"PWD={db_config.get('PWD')};"
        "TrustServerCertificate=yes;"
    )
    timeout = db_config.getint('TIMEOUT', 10)
    # การจัดการข้อผิดพลาดจะถูกทำในส่วนที่เรียกใช้ get_connection()
    return pyodbc.connect(conn_str, timeout=timeout)