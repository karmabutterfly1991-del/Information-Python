#!/usr/bin/env bash
# exit on error
set -o errexit

# ==============================================================================
# สคริปต์นี้จะถูกรันในขั้นตอน Build บน Render เพื่อติดตั้งสิ่งที่จำเป็นทั้งหมด
# ==============================================================================

# 1. ติดตั้ง Dependencies พื้นฐานที่จำเป็นสำหรับ Microsoft ODBC Driver
echo "---> Installing ODBC Driver dependencies..."
apt-get update && apt-get install -y curl gnupg

# 2. เพิ่ม Repository ของ Microsoft และติดตั้ง Driver
echo "---> Adding Microsoft repository and installing ODBC Driver 17..."
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql17

# 3. ติดตั้งไลบรารี Python ทั้งหมดจาก requirements.txt
echo "---> Installing Python dependencies..."
pip install -r requirements.txt

echo "Build finished successfully!"
