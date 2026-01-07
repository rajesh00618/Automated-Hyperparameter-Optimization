#!/bin/bash

# XGBoost Optuna Demo Deployment Script
# This script helps deploy the demo application

echo "🚀 XGBoost Optuna Optimization Demo Deployment"
echo "=============================================="

# Check if outputs exist
if [ ! -d "outputs" ]; then
    echo "❌ Outputs directory not found!"
    echo "Please run the optimization pipeline first:"
    echo "docker run -v \$(pwd)/outputs:/app/outputs optuna-mlflow-pipeline"
    exit 1
fi

# Check if results.json exists
if [ ! -f "outputs/results.json" ]; then
    echo "❌ results.json not found in outputs directory!"
    echo "Please run the optimization pipeline first."
    exit 1
fi

echo "✅ Optimization results found!"

# Build demo Docker image
echo "🏗️  Building demo Docker image..."
docker build -f Dockerfile.demo -t xgboost-demo .

if [ $? -eq 0 ]; then
    echo "✅ Demo image built successfully!"
    echo ""
    echo "🚀 To run the demo locally:"
    echo "docker run -p 8501:8501 xgboost-demo"
    echo ""
    echo "📱 Then open: http://localhost:8501"
    echo ""
    echo "🌐 For cloud deployment, push to your container registry:"
    echo "docker tag xgboost-demo your-registry/xgboost-demo"
    echo "docker push your-registry/xgboost-demo"
else
    echo "❌ Failed to build demo image"
    exit 1
fi