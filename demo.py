"""
SafeCity MVP Demo Script

End-to-end testing and demonstration of the complete SafeCity system.
This script runs all components together to ensure everything works correctly.

Usage:
    python demo.py
    
This will:
1. Generate sample crime data
2. Run the complete ML pipeline
3. Generate all outputs and visualizations
4. Provide instructions for running the dashboard
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src directory to path
sys.path.append('src')

from data_processor import CrimeDataProcessor, generate_sample_data
from hotspot_detector import CrimeHotspotDetector
from risk_predictor import CrimeRiskPredictor
from patrol_manager import PatrolPriorityManager


def print_banner():
    """Print SafeCity banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🚓 SAFECITY MVP DEMO 🚓                   ║
    ║                                                              ║
    ║           AI-Powered Crime Analysis & Patrol Planning        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("🎯 MVP GOAL: Visualize crime hotspots and predict high-risk areas")
    print("🧠 ML MODELS: DBSCAN hotspot detection + Random Forest risk prediction")
    print("🚓 OUTPUT: Smart patrol priority recommendations")
    print("=" * 64)


def run_complete_demo():
    """Run complete SafeCity demo pipeline"""
    
    print_banner()
    
    # Step 1: Data Generation and Processing
    print("\n📊 STEP 1: DATA PROCESSING")
    print("=" * 40)
    
    try:
        # Generate sample crime data
        print("🔄 Generating sample crime data...")
        sample_file = generate_sample_data(n_records=5000)
        print(f"✅ Generated sample data: {sample_file}")
        
        # Process the data
        print("\n🧹 Processing crime data...")
        processor = CrimeDataProcessor()
        processed_data = processor.process_all(sample_file)
        
        # Save processed data
        processed_file = processor.save_processed_data("data/demo_processed_data.csv")
        
        print(f"✅ Data processing complete:")
        print(f"  📁 Raw data: {len(processed_data)} records")
        print(f"  📅 Date range: {processed_data['datetime'].min().date()} to {processed_data['datetime'].max().date()}")
        print(f"  🗺️ Zones created: {processed_data['zone_id'].nunique()}")
        print(f"  🔍 Crime types: {processed_data['crime_type'].nunique()}")
        
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return False
    
    # Step 2: Hotspot Detection
    print("\n🔥 STEP 2: CRIME HOTSPOT DETECTION")
    print("=" * 40)
    
    try:
        print("🔍 Running DBSCAN hotspot detection...")
        detector = CrimeHotspotDetector(eps=0.005, min_samples=10)  # Better parameters
        hotspot_data = detector.detect_hotspots(processed_data, auto_eps=False, plot_results=False)
        
        # Export hotspots
        geojson_file = detector.export_hotspot_geojson("data/demo_hotspots.geojson")
        
        # Get hotspot summary
        summary = detector.get_hotspot_summary()
        
        n_hotspots = sum(hotspot_data['is_hotspot'])
        hotspot_pct = (n_hotspots / len(hotspot_data)) * 100
        
        print(f"✅ Hotspot detection complete:")
        print(f"  🔥 Hotspot incidents: {n_hotspots} ({hotspot_pct:.1f}%)")
        print(f"  🎯 Clusters found: {len(summary)}")
        print(f"  📄 Exported to: {geojson_file}")
        
        if len(summary) > 0:
            print(f"\n📋 Top 3 Hotspot Clusters:")
            for i, row in summary.head(3).iterrows():
                print(f"  {i+1}. Cluster {row['Cluster ID']}: {row['Incidents']} incidents, "
                      f"Primary: {row['Primary Crime']}")
        
    except Exception as e:
        print(f"❌ Error in hotspot detection: {e}")
        return False
    
    # Step 3: Risk Prediction
    print("\n🤖 STEP 3: RISK PREDICTION MODEL")
    print("=" * 40)
    
    try:
        print("🧠 Training Random Forest risk prediction model...")
        predictor = CrimeRiskPredictor(n_estimators=100)
        
        # Train model
        metrics = predictor.train_model(processed_data)
        
        # Make predictions
        print("🔮 Generating risk predictions...")
        risk_predictions = predictor.predict_risk(processed_data)
        
        # Save model and predictions
        model_file = predictor.save_model("models/demo_risk_model.joblib")
        risk_predictions.to_csv("data/demo_risk_predictions.csv", index=False)
        
        print(f"✅ Risk prediction complete:")
        print(f"  🎯 Model accuracy: {metrics['val_accuracy']:.3f}")
        print(f"  🔄 Cross-validation: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        print(f"  📊 Zones analyzed: {len(risk_predictions)}")
        print(f"  💾 Model saved: {model_file}")
        
        # Risk distribution
        risk_dist = risk_predictions['predicted_risk'].value_counts()
        print(f"\n📈 Risk Distribution:")
        for risk_level in ['High', 'Medium', 'Low']:
            count = risk_dist.get(risk_level, 0)
            pct = (count / len(risk_predictions)) * 100
            print(f"  {risk_level:6}: {count:3} zones ({pct:5.1f}%)")
        
    except Exception as e:
        print(f"❌ Error in risk prediction: {e}")
        return False
    
    # Step 4: Patrol Priority Assignment
    print("\n🚓 STEP 4: PATROL PRIORITY ASSIGNMENT")
    print("=" * 40)
    
    try:
        print("📋 Calculating patrol priorities...")
        manager = PatrolPriorityManager()
        
        # Calculate priorities
        patrol_priorities = manager.calculate_patrol_priorities(
            risk_predictions, 
            hotspot_data
        )
        
        # Generate patrol schedule
        print("📅 Generating patrol schedule...")
        schedule = manager.generate_patrol_schedule(
            shift_hours=8,
            available_units=3
        )
        
        # Export results
        patrol_file = manager.export_patrol_plan("data/demo_patrol_plan.csv")
        
        # Get statistics
        stats = manager.get_priority_statistics()
        
        print(f"✅ Patrol planning complete:")
        print(f"  📋 Zones prioritized: {stats['total_zones']}")
        print(f"  🚨 High priority: {stats['priority_distribution'].get('High', 0)} zones")
        print(f"  📊 Coverage: {stats['high_priority_coverage']:.1f}% high priority")
        print(f"  📄 Exported to: {patrol_file}")
        
        print(f"\n🎯 Top 5 Priority Zones:")
        top_zones = patrol_priorities.head(5)
        for i, row in top_zones.iterrows():
            print(f"  {i+1}. {row['zone_id']}: {row['patrol_priority']} priority "
                  f"(Score: {row['total_priority_score']:.1f})")
        
    except Exception as e:
        print(f"❌ Error in patrol planning: {e}")
        return False
    
    # Step 5: Summary and Instructions
    print("\n🎉 DEMO COMPLETION SUMMARY")
    print("=" * 40)
    
    print("✅ All components successfully executed!")
    print("\n📁 Generated Files:")
    files = [
        "data/sample_crime_data.csv",
        "data/demo_processed_data.csv", 
        "data/demo_hotspots.geojson",
        "data/demo_risk_predictions.csv",
        "data/demo_patrol_plan.csv",
        "models/demo_risk_model.joblib"
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"  📄 {file}")
    
    print("\n🚀 Next Steps - Run the Dashboard:")
    print("  1. Install requirements: pip install -r requirements.txt")
    print("  2. Run dashboard: streamlit run dashboard/app.py")
    print("  3. Open browser: http://localhost:8501")
    print("  4. Click 'Load Sample Data' and run the ML pipeline")
    
    print("\n🏆 HACKATHON DEMO READY!")
    print("=" * 40)
    print("🎯 Key Demo Points:")
    print("  • Real-time crime hotspot visualization")
    print("  • AI-powered risk prediction for zones")
    print("  • Smart patrol priority recommendations")
    print("  • Interactive maps and analytics dashboard")
    print("  • Explainable AI with clear justifications")
    
    return True


def run_quick_test():
    """Run a quick test to verify all imports and basic functionality"""
    print("\n🧪 QUICK SYSTEM TEST")
    print("=" * 20)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from data_processor import CrimeDataProcessor
        from hotspot_detector import CrimeHotspotDetector  
        from risk_predictor import CrimeRiskPredictor
        from patrol_manager import PatrolPriorityManager
        print("✅ All modules imported successfully")
        
        # Test data generation
        print("📊 Testing data generation...")
        sample_file = generate_sample_data(n_records=100, output_path="data/test_data.csv")
        print("✅ Sample data generation works")
        
        # Test basic processing
        print("🧹 Testing data processing...")
        processor = CrimeDataProcessor()
        test_data = processor.process_all(sample_file)
        print(f"✅ Data processing works ({len(test_data)} records)")
        
        # Cleanup
        if os.path.exists("data/test_data.csv"):
            os.remove("data/test_data.csv")
        
        print("🎉 System test passed!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 SafeCity MVP Demo Starting...")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if this is a quick test
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = run_quick_test()
    else:
        success = run_complete_demo()
    
    if success:
        print("\n🌟 Demo completed successfully! Ready for hackathon presentation! 🌟")
    else:
        print("\n❌ Demo encountered errors. Please check the logs above.")
        sys.exit(1)