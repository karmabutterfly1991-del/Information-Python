# db_config.py
import pyodbc
import configparser
import os

def get_connection():
    # ตรวจสอบว่าอยู่ในสภาพแวดล้อมคลาวด์หรือไม่
    is_cloud = os.getenv('RENDER') or os.getenv('HEROKU') or os.getenv('CLOUD')
    
    if is_cloud:
        # ใช้ environment variables สำหรับคลาวด์
        server = os.getenv('DB_SERVER')
        database = os.getenv('DB_NAME')
        username = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        driver = os.getenv('DB_DRIVER', 'FreeTDS')
        
        if not all([server, database, username]):
            raise ValueError("Missing required database environment variables for cloud deployment")
        
        # ใช้ FreeTDS driver สำหรับคลาวด์
        conn_str = (
            f"DRIVER={driver};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password or ''};"
            "TDS_Version=8.0;"
            "TrustServerCertificate=yes;"
        )
        timeout = int(os.getenv('DB_TIMEOUT', '10'))
    else:
        # ใช้ config.ini สำหรับ local development
        config = configparser.ConfigParser()
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
    
    try:
        return pyodbc.connect(conn_str, timeout=timeout)
    except pyodbc.Error as e:
        print(f"Database connection error: {e}")
        raise