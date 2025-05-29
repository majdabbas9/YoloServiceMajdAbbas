#!/bin/bash
path_to_file=$1
s3_bucket_name_dev=$2

# Define version and URLs
VERSION="0.127.0"
DEB_FILE="otelcol_${VERSION}_linux_amd64.deb"
URL="https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${VERSION}/${DEB_FILE}"
CONFIG_PATH="/etc/otelcol/config.yaml"

# Function to configure otelcol
configure_otelcol() {
    echo "Configuring otelcol to collect host metrics..."

    sudo tee "$CONFIG_PATH" > /dev/null <<EOF
receivers:
  hostmetrics:
    collection_interval: 15s
    scrapers:
      cpu:
      memory:
      disk:
      filesystem:
      load:
      network:
      processes:

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"

service:
  pipelines:
    metrics:
      receivers: [hostmetrics]
      exporters: [prometheus]
EOF

    echo "Restarting otelcol service..."
    sudo systemctl restart otelcol
    sudo systemctl start otelcol

    echo "otelcol configured and restarted."
}

# Main installation logic
if ! command -v otelcol &> /dev/null; then
    echo "otelcol not found. Installing version $VERSION..."

    # Update package list and install wget if needed
    sudo apt-get update
    sudo apt-get -y install wget

    # Download the .deb file
    wget "$URL"

    # Install the package
    sudo dpkg -i "$DEB_FILE"

    # Fix dependencies if needed
    sudo apt-get install -f -y

    # Cleanup
    rm "$DEB_FILE"
fi

# Final check and configure
if command -v otelcol &> /dev/null; then
    echo "otelcol is installed. Version: $(otelcol --version)"
    configure_otelcol
else
    echo "Failed to install otelcol."
fi

sudo cp yolo_dev.service /etc/systemd/system/
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
sudo apt update && sudo apt install -y libgl1

# reload daemon and restart the service
sudo systemctl daemon-reload
sudo systemctl restart yolo_dev.service
sudo systemctl enable yolo_dev.service
sudo systemctl start yolo_dev.service

if ! systemctl is-active --quiet yolo_dev.service; then
  echo "❌ yolo.service is not running."
  sudo systemctl status yolo_dev.service --no-pager
  exit 1
fi

echo "S3_BUCKET_NAME=$s3_bucket_name_dev" > $path_to_file/.env
# Check if the virtual environment exists
if [ ! -d "$path_to_file/.venv" ]; then  # Check if .venv is a directory
    python3 -m venv "$path_to_file/.venv"
    "$path_to_file/.venv/bin/pip" install -r "$path_to_file/torch-requirements.txt"
    "$path_to_file/.venv/bin/pip" install -r "$path_to_file/requirements.txt"
else
    echo "Virtual environment already exists."
fi