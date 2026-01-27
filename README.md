# 🚀 SafeCity MVP

A web-based system that **visualizes crime hotspots** and **predicts high-risk areas** to support smarter police patrol planning.

## 🎯 Project Overview

SafeCity uses machine learning to analyze historical crime data and provide actionable insights for law enforcement:

- **Crime Hotspot Detection** using DBSCAN clustering
- **Risk Prediction** using Random Forest classifier  
- **Patrol Priority Suggestions** with rule-based logic
- **Interactive Dashboard** with maps and risk tables

## 🛠️ Tech Stack

- **ML**: Python, Scikit-learn, Pandas, NumPy
- **Visualization**: Streamlit, Folium, Plotly
- **Data**: CSV processing, Geospatial analysis

## 📁 Project Structure

```
safecity/
├── data/                 # Crime datasets
├── src/                  # Core ML modules
├── models/               # Trained models
├── dashboard/            # Streamlit app
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run dashboard/app.py
```

## 🧠 ML Pipeline

```
Crime Data → Preprocessing → DBSCAN Hotspots → Feature Engineering → Random Forest → Risk Score → Patrol Priority → Dashboard
```

## 🏆 Key Features

- ✅ Interactive crime hotspot visualization
- ✅ Weekly risk prediction for zones
- ✅ Patrol priority recommendations
- ✅ Ethical AI with clear limitations
- ✅ Easy-to-understand interface

## ⚖️ Ethical Considerations

- No individual-level prediction
- No real-time surveillance
- Focus on resource allocation, not enforcement
- Transparent methodology

Built for hackathon demo - ready to scale responsibly! 🌟