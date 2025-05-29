#!/bin/bash
path_to_file=$1
copy the .servcie file
sudo cp yolo.service /etc/systemd/system/
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
sudo apt update && sudo apt install -y libgl1

# reload daemon and restart the service
sudo systemctl daemon-reload
sudo systemctl restart yolo.service
sudo systemctl enable yolo.service
sudo systemctl start yolo.service

if ! systemctl is-active --quiet yolo.service; then
  echo "❌ yolo.service is not running."
  sudo systemctl status yolo.service --no-pager
  exit 1
fi

echo "S3_BUCKET_NAME=$s3_bucket_name" > $path_to_file/.env
# Check if the virtual environment exists
if [ ! -d "$path_to_file/.venv" ]; then  # Check if .venv is a directory
    python3 -m venv "$path_to_file/.venv"
    "$path_to_file/.venv/bin/pip" install -r "$path_to_file/torch-requirements.txt"
    "$path_to_file/.venv/bin/pip" install -r "$path_to_file/requirements.txt"
else
    echo "Virtual environment already exists."
fi