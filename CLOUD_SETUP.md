# คู่มือการ Deploy บนคลาวด์

## ไฟล์ที่เพิ่ม/แก้ไข

### 1. `db_config.py` - รองรับ Environment Variables
- ตรวจสอบสภาพแวดล้อม (local vs cloud) อัตโนมัติ
- ใช้ FreeTDS driver สำหรับคลาวด์
- รองรับ environment variables

### 2. `app.py` - ใช้ db_config.py
- ลบการเขียน database connection เอง
- import `get_connection` จาก `db_config.py`

### 3. `Procfile` - สำหรับ Heroku/Render
```
web: gunicorn app:app
```

### 4. `runtime.txt` - กำหนด Python version
```
python-3.9.18
```

### 5. `requirements.txt` - เพิ่ม gunicorn
- เพิ่ม `gunicorn>=20.1.0` สำหรับ production server

## การตั้งค่าบน Render.com

### Environment Variables ที่ต้องตั้งค่า:
```
DB_SERVER=your-sql-server-ip-or-domain
DB_NAME=dbPayment
DB_USER=sa1
DB_PASSWORD=your-password
DB_DRIVER=FreeTDS
DB_TIMEOUT=30
```

### การตั้งค่าบน Heroku:
```bash
heroku config:set DB_SERVER=your-sql-server-ip
heroku config:set DB_NAME=dbPayment
heroku config:set DB_USER=sa1
heroku config:set DB_PASSWORD=your-password
heroku config:set DB_DRIVER=FreeTDS
heroku config:set DB_TIMEOUT=30
```

## การตั้งค่า SQL Server

1. **เปิด Remote Connections:**
```sql
EXEC sp_configure 'remote access', 1;
RECONFIGURE;
```

2. **เปิด TCP/IP Protocol:**
- ไปที่ SQL Server Configuration Manager
- SQL Server Network Configuration > Protocols for MSSQLSERVER
- TCP/IP > Enable

3. **ตั้งค่า Firewall:**
- เปิด port 1433
- อนุญาตการเชื่อมต่อจาก Render.com/Heroku IP ranges

## การ Deploy

### Render.com:
1. เชื่อมต่อ GitHub repository
2. ตั้งค่า Environment Variables
3. Deploy

### Heroku:
```bash
git add .
git commit -m "Cloud deployment ready"
git push heroku main
```

## การทดสอบ

1. ตรวจสอบ logs ใน cloud dashboard
2. ทดสอบการเชื่อมต่อฐานข้อมูล
3. ทดสอบ web interface

## หมายเหตุ

- ระบบจะ detect สภาพแวดล้อมอัตโนมัติ
- ใช้ FreeTDS driver สำหรับคลาวด์
- Environment variables จะถูกใช้แทน config.ini เมื่อ deploy บนคลาวด์
