"""
SafeCity Fast Demo Script

Quick demonstration version with optimized parameters for faster execution.
Perfect for live demos and rapid testing.

Usage:
    python fast_demo.py
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append('src')

from data_processor import CrimeDataProcessor, generate_sample_data
from hotspot_detector import CrimeHotspotDetector
from risk_predictor import CrimeRiskPredictor
from patrol_manager import PatrolPriorityManager


def run_fast_demo():
    """Run optimized SafeCity demo for speed"""
    
    print("🚀 SafeCity Fast Demo - Optimized for Speed")
    print("=" * 50)
    
    # Step 1: Generate smaller dataset
    print("📊 Generating sample data (1000 records)...")
    sample_file = generate_sample_data(n_records=1000, output_path="data/fast_demo_data.csv")
    
    # Step 2: Process data
    print("🧹 Processing data...")
    processor = CrimeDataProcessor()
    processed_data = processor.process_all(sample_file, grid_size=0.02)  # Larger grid = fewer zones
    
    print(f"✅ Processed {len(processed_data)} records into {processed_data['zone_id'].nunique()} zones")
    
    # Step 3: Fast hotspot detection
    print("🔥 Detecting hotspots (optimized)...")
    detector = CrimeHotspotDetector(eps=0.01, min_samples=5)  # More relaxed parameters
    hotspot_data = detector.detect_hotspots(processed_data, plot_results=False)
    
    n_hotspots = sum(hotspot_data['is_hotspot'])
    print(f"✅ Found {n_hotspots} hotspot incidents")
    
    # Step 4: Fast risk prediction
    print("🤖 Training lightweight risk model...")
    predictor = CrimeRiskPredictor(n_estimators=20)  # Fewer trees = faster
    predictor.train_model(processed_data)
    
    risk_predictions = predictor.predict_risk(processed_data)
    print(f"✅ Generated predictions for {len(risk_predictions)} zones")
    
    # Step 5: Patrol priorities
    print("🚓 Calculating patrol priorities...")
    manager = PatrolPriorityManager()
    patrol_priorities = manager.calculate_patrol_priorities(risk_predictions, hotspot_data)
    
    # Export quick results
    patrol_priorities.to_csv("data/fast_patrol_plan.csv", index=False)
    
    print("\n🎉 Fast Demo Complete!")
    print("=" * 30)
    print(f"📊 Results:")
    print(f"  • Total incidents: {len(processed_data)}")
    print(f"  • Hotspot incidents: {n_hotspots}")
    print(f"  • Zones analyzed: {len(risk_predictions)}")
    print(f"  • High priority zones: {len(patrol_priorities[patrol_priorities['patrol_priority'] == 'High'])}")
    
    print(f"\n📁 Files created:")
    print(f"  • data/fast_demo_data.csv")
    print(f"  • data/fast_patrol_plan.csv")
    
    return patrol_priorities


if __name__ == "__main__":
    start_time = datetime.now()
    results = run_fast_demo()
    end_time = datetime.now()
    
    execution_time = (end_time - start_time).total_seconds()
    print(f"\n⚡ Execution time: {execution_time:.1f} seconds")
    print("🎯 Ready for instant demo!")