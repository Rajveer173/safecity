#!/bin/bash
echo "🚀 SafeCity MVP - Quick Start Script"
echo "====================================="

echo "📦 Installing requirements..."
pip install -r requirements.txt

echo "🧪 Running system test..."
python demo.py --test

echo "🚓 Starting SafeCity Dashboard..."
echo ""
echo "📱 Dashboard will open at: http://localhost:8501"
echo "💡 Click 'Load Sample Data' in sidebar to start demo"
echo ""
streamlit run dashboard/app.py