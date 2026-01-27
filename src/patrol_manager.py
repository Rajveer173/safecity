"""
SafeCity Patrol Priority Module

Implements rule-based system to sort zones by risk score and assign patrol priorities.
Converts AI predictions into actionable patrol recommendations.

Key features:
- Rule-based patrol priority assignment
- Resource allocation optimization
- Patrol route suggestions
- Priority level justification
- Performance tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json


class PatrolPriorityManager:
    """Manages patrol priority assignment and resource allocation"""
    
    def __init__(self):
        self.priority_rules = {
            'High': {'weight': 3, 'patrol_frequency': 'Every 2-3 hours', 'response_time': '< 10 minutes'},
            'Medium': {'weight': 2, 'patrol_frequency': 'Every 4-6 hours', 'response_time': '< 20 minutes'},
            'Low': {'weight': 1, 'patrol_frequency': 'Daily check', 'response_time': '< 45 minutes'}
        }
        self.patrol_assignments = None
        
    def calculate_patrol_priorities(self, risk_predictions: pd.DataFrame,
                                 hotspot_data: Optional[pd.DataFrame] = None,
                                 resource_constraints: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate patrol priorities based on risk predictions and hotspots
        
        Args:
            risk_predictions: Zone risk predictions from risk predictor
            hotspot_data: Optional hotspot detection results
            resource_constraints: Optional patrol resource limits
            
        Returns:
            DataFrame with patrol priority assignments
        """
        print("🚓 Calculating patrol priorities...")
        
        # Start with risk predictions
        patrol_data = risk_predictions.copy()
        
        # Initialize priority scoring
        patrol_data['base_priority_score'] = 0
        patrol_data['hotspot_bonus'] = 0
        patrol_data['temporal_bonus'] = 0
        patrol_data['total_priority_score'] = 0
        
        # Base priority score from risk predictions
        risk_weights = {'High': 100, 'Medium': 60, 'Low': 30}
        patrol_data['base_priority_score'] = patrol_data['predicted_risk'].map(risk_weights)
        
        # Add risk score (confidence) to base priority
        patrol_data['base_priority_score'] += patrol_data['risk_score'] * 0.5
        
        # Hotspot bonus (if hotspot data available)
        if hotspot_data is not None:
            print("🔥 Incorporating hotspot data...")
            
            # Merge with hotspot information
            hotspot_zones = hotspot_data[hotspot_data['is_hotspot']].groupby('zone_id').agg({
                'hotspot_intensity': lambda x: x.mode().iloc[0] if len(x) > 0 else 'None',
                'crime_type': 'count'
            }).reset_index()
            hotspot_zones.columns = ['zone_id', 'hotspot_intensity', 'hotspot_crime_count']
            
            patrol_data = patrol_data.merge(hotspot_zones, on='zone_id', how='left')
            patrol_data['hotspot_intensity'] = patrol_data['hotspot_intensity'].fillna('None')
            patrol_data['hotspot_crime_count'] = patrol_data['hotspot_crime_count'].fillna(0)
            
            # Calculate hotspot bonus
            intensity_bonus = {'High': 40, 'Medium': 25, 'Low': 15, 'None': 0}
            patrol_data['hotspot_bonus'] = (
                patrol_data['hotspot_intensity'].map(intensity_bonus) +
                patrol_data['hotspot_crime_count'] * 2
            )
        
        # Temporal bonus (current time considerations)
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        # Higher priority during peak crime hours (evening/night)
        if 18 <= current_hour <= 23 or 0 <= current_hour <= 3:
            temporal_multiplier = 1.3
        elif 6 <= current_hour <= 17:
            temporal_multiplier = 1.0
        else:
            temporal_multiplier = 0.8
        
        # Weekend adjustment
        weekend_multiplier = 1.2 if current_day >= 5 else 1.0
        
        patrol_data['temporal_bonus'] = (patrol_data['base_priority_score'] * 
                                       (temporal_multiplier - 1) * weekend_multiplier * 20)
        
        # Calculate total priority score
        patrol_data['total_priority_score'] = (
            patrol_data['base_priority_score'] + 
            patrol_data['hotspot_bonus'] + 
            patrol_data['temporal_bonus']
        )
        
        # Assign patrol priority levels
        patrol_data = self._assign_priority_levels(patrol_data, resource_constraints)
        
        # Add patrol recommendations
        patrol_data = self._add_patrol_recommendations(patrol_data)
        
        # Sort by priority
        patrol_data = patrol_data.sort_values('total_priority_score', ascending=False).reset_index(drop=True)
        
        self.patrol_assignments = patrol_data
        
        print(f"✅ Patrol priorities calculated for {len(patrol_data)} zones")
        self._print_priority_summary(patrol_data)
        
        return patrol_data
    
    def _assign_priority_levels(self, data: pd.DataFrame, 
                              resource_constraints: Optional[Dict] = None) -> pd.DataFrame:
        """
        Assign patrol priority levels based on scores and resource constraints
        
        Args:
            data: DataFrame with priority scores
            resource_constraints: Optional resource limits
            
        Returns:
            DataFrame with assigned priority levels
        """
        data_copy = data.copy()
        
        if resource_constraints is None:
            # Default resource allocation: 20% High, 30% Medium, 50% Low
            resource_constraints = {
                'high_patrol_zones': 0.20,
                'medium_patrol_zones': 0.30,
                'low_patrol_zones': 0.50
            }
        
        n_zones = len(data_copy)
        n_high = max(1, int(n_zones * resource_constraints['high_patrol_zones']))
        n_medium = max(1, int(n_zones * resource_constraints['medium_patrol_zones']))
        
        # Sort by priority score and assign levels
        sorted_indices = data_copy['total_priority_score'].argsort()[::-1]
        
        patrol_priority = ['Low'] * n_zones
        
        # Assign High priority to top zones
        for i in range(min(n_high, n_zones)):
            patrol_priority[sorted_indices[i]] = 'High'
        
        # Assign Medium priority to next zones
        for i in range(n_high, min(n_high + n_medium, n_zones)):
            patrol_priority[sorted_indices[i]] = 'Medium'
        
        data_copy['patrol_priority'] = patrol_priority
        
        return data_copy
    
    def _add_patrol_recommendations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add specific patrol recommendations for each zone"""
        data_copy = data.copy()
        
        recommendations = []
        justifications = []
        
        for _, row in data_copy.iterrows():
            priority = row['patrol_priority']
            risk_level = row['predicted_risk']
            score = row['total_priority_score']
            
            # Generate recommendation text
            rec_parts = []
            
            # Base recommendation
            if priority == 'High':
                rec_parts.append("Immediate patrol deployment recommended")
            elif priority == 'Medium':
                rec_parts.append("Regular patrol monitoring suggested")
            else:
                rec_parts.append("Standard patrol schedule adequate")
            
            # Add specific recommendations based on features
            if 'hotspot_intensity' in row and row['hotspot_intensity'] in ['High', 'Medium']:
                rec_parts.append(f"Active hotspot zone ({row['hotspot_intensity'].lower()} intensity)")
            
            if risk_level == 'High':
                rec_parts.append("High crime risk predicted")
            
            if row.get('hotspot_crime_count', 0) > 5:
                rec_parts.append(f"Recent hotspot activity ({int(row['hotspot_crime_count'])} incidents)")
            
            recommendation = "; ".join(rec_parts)
            recommendations.append(recommendation)
            
            # Generate justification
            justification_parts = []
            
            if risk_level != 'Low':
                justification_parts.append(f"Risk level: {risk_level}")
            
            if score > 100:
                justification_parts.append(f"High priority score ({score:.1f})")
            
            if row.get('hotspot_bonus', 0) > 0:
                justification_parts.append("Located in crime hotspot")
            
            if row.get('temporal_bonus', 0) > 0:
                justification_parts.append("Peak crime time factor")
            
            justification = "; ".join(justification_parts) if justification_parts else "Standard risk assessment"
            justifications.append(justification)
        
        data_copy['patrol_recommendation'] = recommendations
        data_copy['priority_justification'] = justifications
        
        # Add patrol frequency and response time
        frequency_map = {p: self.priority_rules[p]['patrol_frequency'] for p in self.priority_rules}
        response_map = {p: self.priority_rules[p]['response_time'] for p in self.priority_rules}
        
        data_copy['patrol_frequency'] = data_copy['patrol_priority'].map(frequency_map)
        data_copy['target_response_time'] = data_copy['patrol_priority'].map(response_map)
        
        return data_copy
    
    def _print_priority_summary(self, data: pd.DataFrame) -> None:
        """Print summary of patrol priority assignments"""
        priority_counts = data['patrol_priority'].value_counts()
        
        print("\n📊 Patrol Priority Summary:")
        for priority in ['High', 'Medium', 'Low']:
            count = priority_counts.get(priority, 0)
            percentage = (count / len(data)) * 100
            print(f"  {priority:6} Priority: {count:3} zones ({percentage:5.1f}%)")
        
        print(f"\n🎯 Top 5 Priority Zones:")
        top_zones = data.head().round(2)
        for _, zone in top_zones.iterrows():
            print(f"  Zone {zone['zone_id']}: {zone['patrol_priority']} priority "
                  f"(Score: {zone['total_priority_score']:.1f}, Risk: {zone['predicted_risk']})")
    
    def generate_patrol_schedule(self, shift_hours: int = 8,
                               available_units: int = 3) -> Dict:
        """
        Generate optimal patrol schedule for available units
        
        Args:
            shift_hours: Duration of patrol shift
            available_units: Number of patrol units available
            
        Returns:
            Patrol schedule with zone assignments
        """
        if self.patrol_assignments is None:
            raise ValueError("No patrol assignments available. Run calculate_patrol_priorities() first.")
        
        print(f"📅 Generating patrol schedule for {available_units} units ({shift_hours}h shift)")
        
        # Filter zones by priority
        high_priority = self.patrol_assignments[self.patrol_assignments['patrol_priority'] == 'High']
        medium_priority = self.patrol_assignments[self.patrol_assignments['patrol_priority'] == 'Medium']
        low_priority = self.patrol_assignments[self.patrol_assignments['patrol_priority'] == 'Low']
        
        # Calculate time allocation per priority level
        time_allocation = {
            'High': 0.6 * shift_hours,    # 60% of time on high priority
            'Medium': 0.3 * shift_hours,   # 30% on medium priority
            'Low': 0.1 * shift_hours       # 10% on low priority
        }
        
        # Distribute zones among available units
        schedule = {}
        
        for unit_id in range(1, available_units + 1):
            unit_schedule = {
                'unit_id': unit_id,
                'high_priority_zones': [],
                'medium_priority_zones': [],
                'low_priority_zones': [],
                'estimated_routes': {},
                'time_allocation': time_allocation.copy()
            }
            
            # Distribute high priority zones
            unit_high_zones = high_priority[unit_id-1::available_units]['zone_id'].tolist()
            unit_schedule['high_priority_zones'] = unit_high_zones
            
            # Distribute medium priority zones  
            unit_medium_zones = medium_priority[unit_id-1::available_units]['zone_id'].tolist()
            unit_schedule['medium_priority_zones'] = unit_medium_zones
            
            # Distribute low priority zones
            unit_low_zones = low_priority[unit_id-1::available_units]['zone_id'].tolist()
            unit_schedule['low_priority_zones'] = unit_low_zones
            
            # Generate patrol times
            current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
            
            patrol_timeline = []
            
            # High priority zones (first half of shift)
            high_time_per_zone = time_allocation['High'] / max(len(unit_high_zones), 1)
            for zone in unit_high_zones:
                patrol_timeline.append({
                    'zone_id': zone,
                    'priority': 'High',
                    'start_time': current_time.strftime('%H:%M'),
                    'duration_minutes': int(high_time_per_zone * 60),
                    'patrol_type': 'Active patrol'
                })
                current_time += timedelta(hours=high_time_per_zone)
            
            # Medium priority zones (middle of shift)
            medium_time_per_zone = time_allocation['Medium'] / max(len(unit_medium_zones), 1)
            for zone in unit_medium_zones:
                patrol_timeline.append({
                    'zone_id': zone,
                    'priority': 'Medium',
                    'start_time': current_time.strftime('%H:%M'),
                    'duration_minutes': int(medium_time_per_zone * 60),
                    'patrol_type': 'Regular check'
                })
                current_time += timedelta(hours=medium_time_per_zone)
            
            # Low priority zones (end of shift)
            low_time_per_zone = time_allocation['Low'] / max(len(unit_low_zones), 1)
            for zone in unit_low_zones:
                patrol_timeline.append({
                    'zone_id': zone,
                    'priority': 'Low',
                    'start_time': current_time.strftime('%H:%M'),
                    'duration_minutes': int(low_time_per_zone * 60),
                    'patrol_type': 'Drive-by check'
                })
                current_time += timedelta(hours=low_time_per_zone)
            
            unit_schedule['patrol_timeline'] = patrol_timeline
            schedule[f'Unit_{unit_id}'] = unit_schedule
        
        print("✅ Patrol schedule generated")
        return schedule
    
    def export_patrol_plan(self, output_path: str = None) -> str:
        """Export patrol assignments and schedule"""
        if self.patrol_assignments is None:
            raise ValueError("No patrol assignments to export.")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/patrol_plan_{timestamp}.csv"
        
        # Prepare export data
        export_data = self.patrol_assignments[[
            'zone_id', 'zone_lat', 'zone_lng', 'predicted_risk', 'risk_score',
            'patrol_priority', 'total_priority_score', 'patrol_recommendation',
            'priority_justification', 'patrol_frequency', 'target_response_time'
        ]].copy()
        
        export_data.to_csv(output_path, index=False)
        print(f"📄 Exported patrol plan to {output_path}")
        
        return output_path
    
    def get_priority_statistics(self) -> Dict:
        """Get statistics about patrol priority assignments"""
        if self.patrol_assignments is None:
            return {}
        
        data = self.patrol_assignments
        
        stats = {
            'total_zones': len(data),
            'priority_distribution': data['patrol_priority'].value_counts().to_dict(),
            'risk_distribution': data['predicted_risk'].value_counts().to_dict(),
            'average_priority_score': data['total_priority_score'].mean(),
            'highest_priority_zone': {
                'zone_id': data.iloc[0]['zone_id'],
                'score': data.iloc[0]['total_priority_score'],
                'risk': data.iloc[0]['predicted_risk']
            }
        }
        
        # Calculate coverage metrics
        high_coverage = (stats['priority_distribution'].get('High', 0) / stats['total_zones']) * 100
        stats['high_priority_coverage'] = round(high_coverage, 1)
        
        return stats


def demo_patrol_priorities():
    """Demo function to show patrol priority system"""
    print("🚀 SafeCity Patrol Priority Demo")
    
    from data_processor import CrimeDataProcessor, generate_sample_data
    from risk_predictor import CrimeRiskPredictor
    from hotspot_detector import CrimeHotspotDetector
    
    # Generate and process sample data
    sample_file = generate_sample_data(n_records=5000)
    processor = CrimeDataProcessor()
    processed_data = processor.process_all(sample_file)
    
    # Get risk predictions
    print("\n🤖 Training risk prediction model...")
    predictor = CrimeRiskPredictor(n_estimators=30)
    predictor.train_model(processed_data)
    risk_predictions = predictor.predict_risk(processed_data)
    
    # Get hotspot data
    print("\n🔥 Detecting crime hotspots...")
    detector = CrimeHotspotDetector()
    hotspot_data = detector.detect_hotspots(processed_data)
    
    # Calculate patrol priorities
    patrol_manager = PatrolPriorityManager()
    patrol_priorities = patrol_manager.calculate_patrol_priorities(
        risk_predictions, hotspot_data
    )
    
    # Generate patrol schedule
    schedule = patrol_manager.generate_patrol_schedule(
        shift_hours=8, available_units=3
    )
    
    # Show results
    print("\n📊 Top 10 Priority Zones:")
    top_zones = patrol_priorities.head(10)[[
        'zone_id', 'patrol_priority', 'total_priority_score', 
        'predicted_risk', 'patrol_recommendation'
    ]]
    print(top_zones.to_string(index=False))
    
    print("\n📅 Unit 1 Schedule Sample:")
    unit1_timeline = schedule['Unit_1']['patrol_timeline'][:5]
    for patrol in unit1_timeline:
        print(f"  {patrol['start_time']}: {patrol['zone_id']} "
              f"({patrol['priority']} - {patrol['duration_minutes']}min)")
    
    # Export results
    patrol_file = patrol_manager.export_patrol_plan()
    
    # Get statistics
    stats = patrol_manager.get_priority_statistics()
    print(f"\n📈 Patrol Statistics:")
    print(f"  Total zones: {stats['total_zones']}")
    print(f"  High priority coverage: {stats['high_priority_coverage']}%")
    print(f"  Average priority score: {stats['average_priority_score']:.1f}")
    
    return patrol_priorities, schedule


if __name__ == "__main__":
    demo_patrol_priorities()