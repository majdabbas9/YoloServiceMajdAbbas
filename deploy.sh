#!/bin/bash
path_to_file=$1
s3_bucket_name_dev=$2
copy the .servcie file
sudo cp yolo.service /etc/systemd/system/

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

# Check if the virtual environment exists
echo "S3_BUCKET_NAME=$s3_bucket_name_dev" > $path_to_file/.env
if [ ! -d "$path_to_file/.venv" ]; then  # Check if .venv is a directory
    python3 -m venv "$path_to_file/.venv"
    "$path_to_file/.venv/bin/pip" install -r "$path_to_file/torch-requirements.txt"
    "$path_to_file/.venv/bin/pip" install -r "$path_to_file/requirements.txt"
else
    echo "Virtual environment already exists."
fi