#!/bin/bash

# Target directory
TARGET_DIR="/Volumes/IISY/RWTH-PHOENIX-Weather-14T"

# Output file path
OUTPUT_FILE="$TARGET_DIR/phoenix-2014.v3.tar.gz"

# Download URL
URL="https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"

# Start download with curl
echo "Downloading to $OUTPUT_FILE..."
curl -L -o "$OUTPUT_FILE" "$URL"

# Verify download
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed."
fi
