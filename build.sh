#!/usr/bin/env bash

set -o errexit

echo "---> Installing ODBC Driver dependencies..."
apt-get update && apt-get install -y curl gnupg


echo "---> Adding Microsoft repository and installing ODBC Driver 17..."
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql17


echo "---> Installing Python dependencies..."
pip install -r requirements.txt

echo "Build finished successfully!"
