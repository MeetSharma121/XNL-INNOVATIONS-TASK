#!/bin/bash

# Exit on error
set -e

echo "Starting deployment process..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p models results logs

# Run the deployment script
echo "Running deployment script..."
python deploy.py

# Check if deployment was successful
if [ $? -eq 0 ]; then
    echo "Deployment completed successfully!"
    echo "Check deployment.log for detailed information"
    echo "Results are stored in the results directory"
    echo "Model versions are stored in the models directory"
else
    echo "Deployment failed! Check deployment.log for errors"
    exit 1
fi 