#!/bin/bash

# Download the Maestro v3.0.0 zip file
echo "Downloading Maestro v3.0.0 dataset..."
curl -L -o maestro-v3.0.0.zip https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip

# Unzip the dataset
echo "Extracting dataset..."
unzip maestro-v3.0.0.zip

echo "Done."
