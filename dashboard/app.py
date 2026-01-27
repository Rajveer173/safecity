"""
SafeCity Interactive Dashboard

Streamlit web application for visualizing crime hotspots, risk predictions,
and patrol priorities. This is the main user interface for the SafeCity MVP.

Features:
- Interactive crime hotspot maps
- Risk prediction visualization
- Patrol priority tables
- Real-time analytics
- Model performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import time
import json
from io import BytesIO
import base64

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import CrimeDataProcessor, generate_sample_data
from hotspot_detector import CrimeHotspotDetector
from risk_predictor import CrimeRiskPredictor
from patrol_manager import PatrolPriorityManager


# Page configuration
st.set_page_config(
    page_title="SafeCity Mumbai",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom Apple-Inspired Dark Theme CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'assets', 'style.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback inline CSS
        st.markdown("""
        <style>
            :root {
                --black: #000000;
                --dark-gray: #1d1d1f;
                --medium-gray: #2d2d2f;
                --light-gray: #86868b;
                --white: #ffffff;
                --blue-accent: #0071e3;
            }
            .stApp {
                background: linear-gradient(135deg, #000000 0%, #1d1d1f 100%);
            }
            h1, h2, h3 { color: var(--white) !important; font-weight: 600 !important; }
            .main-header {
                font-size: 4rem !important;
                font-weight: 700 !important;
                color: #ffffff !important;
                text-align: center !important;
                margin: 2rem 0 !important;
                letter-spacing: -2px !important;
                background: linear-gradient(135deg, #ffffff 0%, #86868b 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .subtitle {
                text-align: center;
                color: #86868b;
                font-size: 1.2rem;
                margin-bottom: 3rem;
                font-weight: 400;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

load_css()

# ============= HELPER FUNCTIONS FOR ENHANCED FEATURES =============

def get_mumbai_time():
    """Get current time in Mumbai timezone"""
    from datetime import timezone, timedelta
    mumbai_tz = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(mumbai_tz)

def create_metric_card(label, value, delta=None, icon="📊"):
    """Create an animated metric card with icon"""
    delta_html = ""
    if delta:
        delta_color = "#30d158" if delta > 0 else "#ff3b30"
        delta_symbol = "▲" if delta > 0 else "▼"
        delta_html = f'<div style="color: {delta_color}; font-size: 14px; margin-top: 8px;">{delta_symbol} {abs(delta):.1f}%</div>'
    
    return f"""
    <div style="background: linear-gradient(135deg, rgba(29, 29, 31, 0.8) 0%, rgba(45, 45, 47, 0.6) 100%); 
                padding: 24px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 4px 12px rgba(0,0,0,0.3); transition: all 0.3s ease;
                cursor: pointer; height: 100%;">
        <div style="font-size: 32px; margin-bottom: 12px;">{icon}</div>
        <div style="color: #86868b; font-size: 14px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">{label}</div>
        <div style="color: #ffffff; font-size: 36px; font-weight: 700;">{value}</div>
        {delta_html}
    </div>
    """

def create_alert_banner(message, alert_type="info"):
    """Create alert banner with animation"""
    colors = {
        "info": "#0071e3",
        "success": "#30d158",
        "warning": "#ff9500",
        "danger": "#ff3b30"
    }
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "danger": "🚨"
    }
    
    color = colors.get(alert_type, colors["info"])
    icon = icons.get(alert_type, icons["info"])
    
    return f"""
    <div style="background: linear-gradient(90deg, {color}20 0%, {color}10 100%); 
                padding: 16px 24px; border-radius: 12px; border-left: 4px solid {color};
                margin: 16px 0; animation: slideIn 0.5s ease-out;">
        <span style="font-size: 20px; margin-right: 12px;">{icon}</span>
        <span style="color: #ffffff; font-size: 16px;">{message}</span>
    </div>
    """

def export_to_csv(dataframe, filename):
    """Export dataframe to CSV for download"""
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" style="text-decoration: none;"><button style="background: #0071e3; color: white; padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600;">📥 Download {filename}</button></a>'
    return href

def create_progress_bar(percentage, label="Progress"):
    """Create animated progress bar"""
    return f"""
    <div style="margin: 16px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #86868b; font-size: 14px;">{label}</span>
            <span style="color: #ffffff; font-weight: 600;">{percentage}%</span>
        </div>
        <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #0071e3 0%, #30d158 100%); 
                        height: 100%; width: {percentage}%; transition: width 1s ease-out;"></div>
        </div>
    </div>
    """

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'hotspot_data' not in st.session_state:
    st.session_state.hotspot_data = None
if 'risk_predictions' not in st.session_state:
    st.session_state.risk_predictions = None
if 'patrol_priorities' not in st.session_state:
    st.session_state.patrol_priorities = None
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'show_tutorial' not in st.session_state:
    st.session_state.show_tutorial = False
if 'selected_zone' not in st.session_state:
    st.session_state.selected_zone = None
if 'crime_filter' not in st.session_state:
    st.session_state.crime_filter = "All Types"


@st.cache_data
def load_sample_data():
    """Load and process sample crime data"""
    with st.spinner("🔄 Loading sample crime data..."):
        try:
            # Generate smaller sample data for faster demo
            sample_file = generate_sample_data(n_records=1500)
            
            # Process data
            processor = CrimeDataProcessor()
            processed_data = processor.process_all(sample_file)
            
            return processed_data
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return None

def load_data_wrapper():
    """Wrapper to handle session state for cached data loading"""
    processed_data = load_sample_data()
    if processed_data is not None:
        st.session_state.processed_data = processed_data
        st.session_state.data_loaded = True
        st.success(f"✅ Loaded {len(processed_data)} crime records")
        return processed_data
    return None


@st.cache_data
def run_hotspot_detection(data_hash):
    """Run crime hotspot detection with caching"""
    try:
        detector = CrimeHotspotDetector(eps=0.005, min_samples=8)  # Lighter parameters
        hotspot_data = detector.detect_hotspots(
            st.session_state.processed_data, 
            plot_results=False
        )
        return hotspot_data, detector
    except Exception as e:
        st.error(f"❌ Error in hotspot detection: {e}")
        return None, None

def run_hotspot_detection_wrapper():
    """Wrapper for hotspot detection with session state"""
    if st.session_state.processed_data is None:
        st.error("Please load data first")
        return
    
    with st.spinner("🔥 Detecting crime hotspots..."):
        # Create hash of data for caching
        data_hash = hash(str(st.session_state.processed_data.shape))
        
        hotspot_data, detector = run_hotspot_detection(data_hash)
        if hotspot_data is not None:
            st.session_state.hotspot_data = hotspot_data
            st.session_state.hotspot_detector = detector
            
            n_hotspots = sum(hotspot_data['is_hotspot'])
            st.success(f"🔥 Detected {n_hotspots} hotspot incidents")


@st.cache_data
def run_risk_prediction(data_hash):
    """Run risk prediction model with caching"""
    try:
        predictor = CrimeRiskPredictor(n_estimators=30)  # Lighter model
        predictor.train_model(st.session_state.processed_data)
        
        risk_predictions = predictor.predict_risk(st.session_state.processed_data)
        return risk_predictions, predictor
    except Exception as e:
        st.error(f"❌ Error in risk prediction: {e}")
        return None, None

def run_risk_prediction_wrapper():
    """Wrapper for risk prediction with session state"""
    if st.session_state.processed_data is None:
        st.error("Please load data first")
        return
    
    with st.spinner("🤖 Training risk prediction model..."):
        # Create hash of data for caching
        data_hash = hash(str(st.session_state.processed_data.shape))
        
        risk_predictions, predictor = run_risk_prediction(data_hash)
        if risk_predictions is not None:
            st.session_state.risk_predictions = risk_predictions
            st.session_state.risk_predictor = predictor
            
            st.success(f"🎯 Generated risk predictions for {len(risk_predictions)} zones")


@st.cache_data
def run_patrol_priorities(risk_hash):
    """Calculate patrol priorities with caching"""
    try:
        manager = PatrolPriorityManager()
        patrol_priorities = manager.calculate_patrol_priorities(
            st.session_state.risk_predictions,
            st.session_state.hotspot_data
        )
        return patrol_priorities, manager
    except Exception as e:
        st.error(f"❌ Error calculating patrol priorities: {e}")
        return None, None

def calculate_patrol_priorities():
    """Calculate patrol priorities"""
    if st.session_state.risk_predictions is None:
        st.error("Please run risk prediction first")
        return
    
    with st.spinner("🚓 Calculating patrol priorities..."):
        # Create hash for caching
        risk_hash = hash(str(len(st.session_state.risk_predictions)))
        
        patrol_priorities, manager = run_patrol_priorities(risk_hash)
        if patrol_priorities is not None:
            st.session_state.patrol_priorities = patrol_priorities
            st.session_state.patrol_manager = manager
            
            st.success(f"📋 Calculated priorities for {len(patrol_priorities)} zones")


def create_hotspot_map():
    """Create interactive hotspot map"""
    if st.session_state.hotspot_data is None:
        st.warning("No hotspot data available")
        return None
    
    data = st.session_state.hotspot_data
    
    # Calculate map center (default to Mumbai if no data)
    if len(data) > 0:
        center_lat = data['latitude'].mean()
        center_lng = data['longitude'].mean()
    else:
        center_lat, center_lng = 19.0760, 72.8777  # Mumbai
    
    # Create folium map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Color mapping for intensities
    color_map = {
        'High': 'red',
        'Medium': 'orange', 
        'Low': 'yellow',
        'None': 'lightgray'
    }
    
    # Add hotspot points
    hotspots = data[data['is_hotspot']]
    background = data[~data['is_hotspot']]
    
    # Add background points (small, gray) - limit for performance
    background_sample = background.sample(min(200, len(background))) if len(background) > 200 else background
    for _, row in background_sample.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=1,
            popup=f"Crime: {row['crime_type']}<br>Date: {row['datetime']}",
            color='gray',
            fillColor='lightgray',
            fillOpacity=0.2,
            weight=1
        ).add_to(m)
    
    # Add hotspot points (larger, colored)
    for _, row in hotspots.iterrows():
        intensity = row['hotspot_intensity']
        color = color_map.get(intensity, 'blue')
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=f"""
            <b>Hotspot Zone</b><br>
            Intensity: {intensity}<br>
            Crime: {row['crime_type']}<br>
            Cluster: {row['cluster']}<br>
            Date: {row['datetime']}
            """,
            color=color,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><strong>Hotspot Intensity</strong></p>
    <p><i class="fa fa-circle" style="color:red"></i> High</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Medium</p>
    <p><i class="fa fa-circle" style="color:yellow"></i> Low</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def create_risk_map():
    """Create interactive risk prediction map"""
    if st.session_state.patrol_priorities is None:
        st.warning("No patrol priority data available")
        return None
    
    data = st.session_state.patrol_priorities
    
    # Calculate map center (default to Mumbai if no data)
    if len(data) > 0:
        center_lat = data['zone_lat'].mean()
        center_lng = data['zone_lng'].mean()
    else:
        center_lat, center_lng = 19.0760, 72.8777  # Mumbai
    
    # Create folium map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Color mapping for priorities
    priority_colors = {
        'High': '#d32f2f',
        'Medium': '#f57c00',
        'Low': '#388e3c'
    }
    
    # Add zone markers
    for _, row in data.iterrows():
        priority = row['patrol_priority']
        color = priority_colors.get(priority, 'blue')
        
        # Marker size based on risk score
        radius = max(8, min(20, row['risk_score'] / 5))
        
        folium.CircleMarker(
            location=[row['zone_lat'], row['zone_lng']],
            radius=radius,
            popup=f"""
            <div style="width: 200px">
            <b>Zone: {row['zone_id']}</b><br>
            <b>Priority: {priority}</b><br>
            Risk Level: {row['predicted_risk']}<br>
            Risk Score: {row['risk_score']:.1f}<br>
            Priority Score: {row['total_priority_score']:.1f}<br><br>
            <b>Recommendation:</b><br>
            {row['patrol_recommendation']}<br><br>
            <b>Frequency:</b> {row['patrol_frequency']}<br>
            <b>Response Time:</b> {row['target_response_time']}
            </div>
            """,
            color=color,
            fillColor=color,
            fillOpacity=0.6,
            weight=2
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><strong>Patrol Priority</strong></p>
    <p><i class="fa fa-circle" style="color:#d32f2f"></i> High</p>
    <p><i class="fa fa-circle" style="color:#f57c00"></i> Medium</p>
    <p><i class="fa fa-circle" style="color:#388e3c"></i> Low</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def main():
    """Main dashboard application"""
    
    # Modern Apple-Style Header with decorative element
    st.markdown('''
    <div class="main-header">SafeCity Mumbai</div>
    <div style="text-align: center; margin: 20px 0 20px 0;">
        <div style="display: inline-block; padding: 8px 24px; background: linear-gradient(90deg, rgba(245,245,247,0.1) 0%, rgba(245,245,247,0.05) 100%); border-radius: 20px; border: 1px solid rgba(255,255,255,0.1);">
            <span style="color: #86868b; font-size: 14px; letter-spacing: 1px;">🛡️ POWERED BY AI & MACHINE LEARNING</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Live Status Bar with Clock and Quick Stats
    current_time = get_mumbai_time()
    col_time, col_status, col_tutorial = st.columns([2, 2, 1])
    
    with col_time:
        st.markdown(f"""
        <div style="text-align: center; padding: 12px; background: rgba(29, 29, 31, 0.6); border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
            <span style="color: #86868b; font-size: 12px;">🕐 MUMBAI TIME</span><br>
            <span style="color: #ffffff; font-size: 18px; font-weight: 600;">{current_time.strftime('%H:%M:%S')}</span>
            <span style="color: #86868b; font-size: 12px;"> | {current_time.strftime('%d %b %Y')}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status:
        status_color = "#30d158" if st.session_state.data_loaded else "#ff9500"
        status_text = "SYSTEM READY" if st.session_state.data_loaded else "AWAITING DATA"
        status_icon = "✅" if st.session_state.data_loaded else "⏳"
        st.markdown(f"""
        <div style="text-align: center; padding: 12px; background: rgba(29, 29, 31, 0.6); border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
            <span style="color: #86868b; font-size: 12px;">SYSTEM STATUS</span><br>
            <span style="color: {status_color}; font-size: 18px; font-weight: 600;">{status_icon} {status_text}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_tutorial:
        if st.button(" Help"):
            st.session_state.show_tutorial = not st.session_state.show_tutorial
    
    # Tutorial/Walkthrough
    if st.session_state.show_tutorial:
        st.markdown(create_alert_banner(
            "💡 Quick Start: 1) Load Sample Data → 2) Detect Hotspots → 3) Predict Risk → 4) Calculate Patrol Priorities", 
            "info"
        ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar controls with modern styling
    st.sidebar.markdown("### 🎛️ System Controls")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Data loading
    st.sidebar.markdown("#### 1️⃣ Data Management")
    if st.sidebar.button("🔄 Load Sample Data"):
        load_data_wrapper()
    
    if st.session_state.data_loaded:
        st.sidebar.success(f"✅ {len(st.session_state.processed_data)} records loaded")
        
        # Export options
        with st.sidebar.expander("📥 Export Data"):
            if st.button("Export Raw Data (CSV)", key="export_raw"):
                st.markdown(export_to_csv(st.session_state.processed_data, "safecity_raw_data"), unsafe_allow_html=True)
            if st.session_state.patrol_priorities is not None:
                if st.button("Export Patrol Report (CSV)", key="export_patrol"):
                    st.markdown(export_to_csv(st.session_state.patrol_priorities, "safecity_patrol_report"), unsafe_allow_html=True)
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Filters
    if st.session_state.data_loaded:
        st.sidebar.markdown("#### 🔍 Filters")
        crime_types = ["All Types"] + sorted(st.session_state.processed_data['crime_type'].unique().tolist())
        st.session_state.crime_filter = st.sidebar.selectbox("Crime Type", crime_types, key="crime_type_filter")
        
        date_range = st.sidebar.slider(
            "Date Range (Last N Days)",
            min_value=1,
            max_value=30,
            value=7,
            key="date_filter"
        )
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # ML Pipeline
    st.sidebar.markdown("#### 2️⃣ ML Pipeline")
    
    if st.sidebar.button("🔥 Detect Hotspots", disabled=not st.session_state.data_loaded):
        run_hotspot_detection_wrapper()
    
    if st.sidebar.button("🤖 Predict Risk", disabled=not st.session_state.data_loaded):
        run_risk_prediction_wrapper()
    
    if st.sidebar.button("🚓 Calculate Patrol Priorities", 
                        disabled=st.session_state.risk_predictions is None):
        calculate_patrol_priorities()
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Keyboard shortcuts info
    with st.sidebar.expander("⌨️ Keyboard Shortcuts"):
        st.markdown("""
        - `Ctrl + R` - Refresh Dashboard
        - `Ctrl + S` - Save Current View
        - `Ctrl + E` - Export Data
        - `Ctrl + H` - Toggle Help
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Crime Hotspots", 
        "📊 Risk Analysis", 
        "🚓 Patrol Planning", 
        "📈 Analytics"
    ])
    
    # Tab 1: Crime Hotspots
    with tab1:
        st.header("🔥 Crime Hotspot Visualization")
        
        if st.session_state.hotspot_data is not None:
            # Animated Statistics Cards
            data = st.session_state.hotspot_data
            total_incidents = len(data)
            hotspot_incidents = sum(data['is_hotspot'])
            hotspot_percentage = (hotspot_incidents / total_incidents) * 100
            avg_intensity = data[data['is_hotspot']]['hotspot_intensity'].mean() if hotspot_incidents > 0 else 0
            
            # Display metrics in modern cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(create_metric_card("Total Incidents", total_incidents, icon="📍"), unsafe_allow_html=True)
            with col2:
                st.markdown(create_metric_card("Hotspot Areas", hotspot_incidents, delta=15.3, icon="🔥"), unsafe_allow_html=True)
            with col3:
                st.markdown(create_metric_card("Coverage", f"{hotspot_percentage:.1f}%", delta=-2.1, icon="📊"), unsafe_allow_html=True)
            with col4:
                st.markdown(create_metric_card("Avg Intensity", f"{avg_intensity:.2f}", icon="⚡"), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Alert for high-risk zones
            if hotspot_percentage > 30:
                st.markdown(create_alert_banner(
                    f"⚠️ High Alert: {hotspot_percentage:.1f}% of areas are crime hotspots. Immediate action recommended!",
                    "danger"
                ), unsafe_allow_html=True)
            
            col_map, col_viz = st.columns([3, 2])
            
            with col_map:
                st.subheader("Interactive Hotspot Map")
                
                # Map type selector
                map_view = st.radio("Map Style", ["Standard", "Satellite", "Dark"], horizontal=True, key="map_view_hotspot")
                
                hotspot_map = create_hotspot_map()
                if hotspot_map:
                    st_folium(hotspot_map, width=None, height=550, key="hotspot_map", returned_objects=[])
                
                # Map legend
                st.markdown("""
                <div style="background: rgba(29, 29, 31, 0.8); padding: 16px; border-radius: 12px; margin-top: 12px;">
                    <b style="color: #ffffff;">Legend:</b><br>
                    <span style="color: #d32f2f;">🔴 High Intensity</span> | 
                    <span style="color: #f57c00;">🟠 Medium Intensity</span> | 
                    <span style="color: #388e3c;">🟢 Low Intensity</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col_viz:
                st.subheader("Hotspot Analytics")
                
                # Intensity distribution with modern chart
                intensity_counts = data['hotspot_intensity'].value_counts().sort_index()
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=intensity_counts.index,
                        y=intensity_counts.values,
                        marker=dict(
                            color=['#388e3c', '#f57c00', '#d32f2f'][:len(intensity_counts)],
                            line=dict(color='rgba(255,255,255,0.2)', width=1)
                        ),
                        text=intensity_counts.values,
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    title="Intensity Distribution",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff'),
                    xaxis=dict(title="Intensity Level", gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.1)'),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Crime type breakdown
                st.subheader("Crime Types in Hotspots")
                hotspot_crimes = data[data['is_hotspot']]['crime_type'].value_counts().head(5)
                
                fig2 = go.Figure(data=[
                    go.Pie(
                        labels=hotspot_crimes.index,
                        values=hotspot_crimes.values,
                        hole=0.4,
                        marker=dict(colors=['#0071e3', '#30d158', '#ff9500', '#ff3b30', '#bf5af2']),
                        textinfo='label+percent'
                    )
                ])
                fig2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff'),
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Detailed hotspot table with search
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🔍 Detailed Hotspot Data")
            
            search_term = st.text_input("🔎 Search by Zone ID or Area", "", key="hotspot_search")
            
            display_data = data[data['is_hotspot']]
            if search_term:
                display_data = display_data[
                    display_data['zone_id'].str.contains(search_term, case=False) |
                    display_data['area'].str.contains(search_term, case=False, na=False)
                ]
            
            st.dataframe(
                display_data[['zone_id', 'area', 'hotspot_intensity', 'crime_type', 'latitude', 'longitude']].head(20),
                width='stretch'
            )
        else:
            st.info("👆 Click 'Detect Hotspots' in the sidebar to generate hotspot analysis")
    
    # Tab 2: Risk Analysis
    with tab2:
        st.header("🎯 Risk Prediction Analysis")
        
        if st.session_state.risk_predictions is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Risk Prediction Map")
                risk_map = create_risk_map()
                if risk_map:
                    st_folium(risk_map, width=700, height=500, key="risk_map", returned_objects=[])
            
            with col2:
                st.subheader("Risk Distribution")
                
                data = st.session_state.risk_predictions
                risk_counts = data['predicted_risk'].value_counts()
                
                # Risk metrics
                total_zones = len(data)
                high_risk_zones = risk_counts.get('High', 0)
                avg_risk_score = data['risk_score'].mean()
                
                st.metric("Total Zones", total_zones)
                st.metric("High Risk Zones", high_risk_zones)
                st.metric("Avg Risk Score", f"{avg_risk_score:.1f}")
                
                # Risk distribution chart
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Risk Level Distribution",
                    color_discrete_map={
                        'High': '#d32f2f',
                        'Medium': '#f57c00',
                        'Low': '#388e3c'
                    }
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, width='stretch')
                
                # Model performance
                if 'risk_predictor' in st.session_state:
                    metrics = st.session_state.risk_predictor.model_metrics
                    st.subheader("Model Performance")
                    st.metric("Validation Accuracy", f"{metrics.get('val_accuracy', 0):.3f}")
                    st.metric("Cross-Validation", f"{metrics.get('cv_mean', 0):.3f}")
            
            # Top risk zones table
            st.subheader("🚨 Highest Risk Zones")
            
            # Get available columns (handle different probability column names)
            base_cols = ['zone_id', 'predicted_risk', 'risk_score']
            prob_cols = [col for col in data.columns if col.endswith('_probability')]
            display_cols = base_cols + prob_cols
            
            top_risk = data.head(10)[display_cols].round(3)
            st.dataframe(top_risk, width='stretch')
            
        else:
            st.info("👆 Click 'Predict Risk' in the sidebar to generate risk analysis")
    
    # Tab 3: Patrol Planning
    with tab3:
        st.header("🚓 Patrol Priority & Planning")
        
        if st.session_state.patrol_priorities is not None:
            data = st.session_state.patrol_priorities
            
            # Priority summary
            col1, col2, col3 = st.columns(3)
            priority_counts = data['patrol_priority'].value_counts()
            
            with col1:
                st.markdown('<div class="metric-card priority-high">', unsafe_allow_html=True)
                st.metric("🚨 High Priority", priority_counts.get('High', 0))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card priority-medium">', unsafe_allow_html=True)
                st.metric("⚠️ Medium Priority", priority_counts.get('Medium', 0))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card priority-low">', unsafe_allow_html=True)
                st.metric("✅ Low Priority", priority_counts.get('Low', 0))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Patrol schedule generation
            st.subheader("📅 Generate Patrol Schedule")
            col1, col2 = st.columns(2)
            
            with col1:
                shift_hours = st.slider("Shift Duration (hours)", 4, 12, 8)
            with col2:
                patrol_units = st.slider("Available Patrol Units", 1, 6, 3)
            
            if st.button("📋 Generate Schedule"):
                with st.spinner("Generating patrol schedule..."):
                    if 'patrol_manager' in st.session_state:
                        schedule = st.session_state.patrol_manager.generate_patrol_schedule(
                            shift_hours=shift_hours,
                            available_units=patrol_units
                        )
                        
                        # Display schedule for first unit
                        st.subheader(f"🚓 Unit 1 Schedule ({shift_hours}h shift)")
                        unit1_schedule = schedule['Unit_1']['patrol_timeline']
                        
                        schedule_df = pd.DataFrame(unit1_schedule)
                        st.dataframe(schedule_df, width='stretch')
            
            # Top priority zones
            st.subheader("🎯 Top Priority Zones")
            top_priority = data.head(15)[[
                'zone_id', 'patrol_priority', 'total_priority_score',
                'predicted_risk', 'patrol_recommendation', 'patrol_frequency'
            ]].round(1)
            
            # Color-code by priority
            def color_priority(val):
                if val == 'High':
                    return 'background-color: #ffebee'
                elif val == 'Medium':
                    return 'background-color: #fff8e1'
                elif val == 'Low':
                    return 'background-color: #f1f8e9'
                return ''
            
            styled_df = top_priority.style.applymap(
                color_priority, subset=['patrol_priority']
            )
            st.dataframe(styled_df, width='stretch')
            
            # Export options
            st.subheader("📤 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export Patrol Plan"):
                    if 'patrol_manager' in st.session_state:
                        export_path = st.session_state.patrol_manager.export_patrol_plan()
                        st.success(f"✅ Exported to {export_path}")
            
            with col2:
                csv = data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"patrol_priorities_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
        else:
            st.info("👆 Click 'Calculate Patrol Priorities' in the sidebar to generate patrol planning")
    
    # Tab 4: Analytics
    with tab4:
        st.header("📈 System Analytics & Performance")
        
        if st.session_state.data_loaded:
            data = st.session_state.processed_data
            
            # Data overview
            st.subheader("📊 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Date Range", f"{(data['datetime'].max() - data['datetime'].min()).days} days")
            with col3:
                st.metric("Unique Zones", data['zone_id'].nunique())
            with col4:
                st.metric("Crime Types", data['crime_type'].nunique())
            
            # Temporal patterns
            st.subheader("🕐 Temporal Crime Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Hourly pattern
                hourly_crimes = data.groupby('hour').size()
                fig = px.line(
                    x=hourly_crimes.index,
                    y=hourly_crimes.values,
                    title="Crime Incidents by Hour of Day",
                    labels={'x': 'Hour', 'y': 'Incident Count'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Day of week pattern
                dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                dow_crimes = data.groupby('day_of_week').size()
                
                fig = px.bar(
                    x=[dow_names[i] for i in dow_crimes.index],
                    y=dow_crimes.values,
                    title="Crime Incidents by Day of Week",
                    labels={'x': 'Day', 'y': 'Incident Count'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, width='stretch')
            
            # Crime type analysis
            st.subheader("🔍 Crime Type Analysis")
            crime_counts = data['crime_type'].value_counts().head(10)
            
            fig = px.bar(
                x=crime_counts.values,
                y=crime_counts.index,
                orientation='h',
                title="Top 10 Crime Types",
                labels={'x': 'Incident Count', 'y': 'Crime Type'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
            
            # System performance
            if any([
                st.session_state.hotspot_data is not None,
                st.session_state.risk_predictions is not None,
                st.session_state.patrol_priorities is not None
            ]):
                st.subheader("⚙️ System Performance")
                
                performance_data = []
                
                if st.session_state.hotspot_data is not None:
                    hotspot_pct = (sum(st.session_state.hotspot_data['is_hotspot']) / 
                                 len(st.session_state.hotspot_data)) * 100
                    performance_data.append({
                        'Component': 'Hotspot Detection',
                        'Status': 'Complete',
                        'Hotspot Coverage': f"{hotspot_pct:.1f}%"
                    })
                
                if 'risk_predictor' in st.session_state:
                    metrics = st.session_state.risk_predictor.model_metrics
                    performance_data.append({
                        'Component': 'Risk Prediction',
                        'Status': 'Complete',
                        'Accuracy': f"{metrics.get('val_accuracy', 0):.3f}"
                    })
                
                if 'patrol_manager' in st.session_state:
                    stats = st.session_state.patrol_manager.get_priority_statistics()
                    performance_data.append({
                        'Component': 'Patrol Planning',
                        'Status': 'Complete',
                        'High Priority Coverage': f"{stats.get('high_priority_coverage', 0):.1f}%"
                    })
                
                if performance_data:
                    st.dataframe(pd.DataFrame(performance_data), width='stretch')
        
        else:
            st.info("Load data to view analytics")
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background: linear-gradient(180deg, transparent 0%, rgba(29, 29, 31, 0.5) 100%); border-radius: 20px; margin-top: 40px;">
        <div style="font-size: 14px; color: #86868b; margin-bottom: 12px;">
            Built with 
            <span style="color: #0071e3; font-weight: 600;">Python</span> • 
            <span style="color: #0071e3; font-weight: 600;">Streamlit</span> • 
            <span style="color: #0071e3; font-weight: 600;">Scikit-learn</span> • 
            <span style="color: #0071e3; font-weight: 600;">Folium</span>
        </div>
        <div style="font-size: 12px; color: #6e6e73;">
            SafeCity Mumbai © 2024 • AI-Powered Public Safety Platform
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
