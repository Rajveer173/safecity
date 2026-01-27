"""
SafeCity Data Preprocessing Module

Handles crime data ingestion, cleaning, and preprocessing for ML pipeline.
Designed for CSV files with crime incident data.

Expected CSV columns:
- latitude: Latitude coordinate
- longitude: Longitude coordinate  
- crime_type: Type of crime (e.g., 'Theft', 'Assault', 'Burglary')
- datetime: Date and time of incident
- area: Area/zone identifier
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class CrimeDataProcessor:
    """Processes and cleans crime data for SafeCity ML pipeline"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load crime data from CSV file"""
        try:
            self.data = pd.read_csv(filepath)
            print(f"✅ Loaded {len(self.data)} records from {filepath}")
            return self.data
        except Exception as e:
            raise ValueError(f"❌ Error loading data: {e}")
    
    def validate_columns(self, required_cols: List[str] = None) -> bool:
        """Validate that required columns exist"""
        if required_cols is None:
            required_cols = ['latitude', 'longitude', 'crime_type', 'datetime', 'area']
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing required columns: {missing_cols}")
        
        print(f"✅ All required columns present: {required_cols}")
        return True
    
    def clean_coordinates(self) -> pd.DataFrame:
        """Clean and validate latitude/longitude coordinates"""
        # Remove rows with missing coordinates
        initial_count = len(self.data)
        self.data = self.data.dropna(subset=['latitude', 'longitude'])
        
        # Validate coordinate ranges
        self.data = self.data[
            (self.data['latitude'].between(-90, 90)) & 
            (self.data['longitude'].between(-180, 180))
        ]
        
        final_count = len(self.data)
        removed = initial_count - final_count
        
        if removed > 0:
            print(f"🧹 Removed {removed} records with invalid coordinates")
        
        return self.data
    
    def parse_datetime(self, datetime_col: str = 'datetime', 
                      datetime_format: str = None) -> pd.DataFrame:
        """Parse datetime column and extract time features"""
        try:
            # Try to parse datetime automatically
            if datetime_format:
                self.data['datetime'] = pd.to_datetime(self.data[datetime_col], format=datetime_format)
            else:
                self.data['datetime'] = pd.to_datetime(self.data[datetime_col])
            
            # Extract time features
            self.data['year'] = self.data['datetime'].dt.year
            self.data['month'] = self.data['datetime'].dt.month
            self.data['day'] = self.data['datetime'].dt.day
            self.data['hour'] = self.data['datetime'].dt.hour
            self.data['day_of_week'] = self.data['datetime'].dt.dayofweek  # 0=Monday
            self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6])  # Saturday, Sunday
            
            print("✅ Parsed datetime and extracted time features")
            return self.data
            
        except Exception as e:
            raise ValueError(f"❌ Error parsing datetime: {e}")
    
    def standardize_crime_types(self) -> pd.DataFrame:
        """Standardize and clean crime type categories"""
        # Convert to uppercase and strip whitespace
        self.data['crime_type'] = self.data['crime_type'].str.upper().str.strip()
        
        # Basic crime type mapping for common variations
        crime_mapping = {
            'BURGLARY': ['BREAK AND ENTER', 'BREAKING AND ENTERING', 'B&E'],
            'THEFT': ['LARCENY', 'STEALING', 'SHOPLIFTING'],
            'ASSAULT': ['BATTERY', 'VIOLENT CRIME'],
            'ROBBERY': ['ARMED ROBBERY', 'MUGGING'],
            'VANDALISM': ['PROPERTY DAMAGE', 'GRAFFITI'],
            'DRUG': ['DRUG OFFENSE', 'NARCOTICS', 'SUBSTANCE'],
            'VEHICLE': ['AUTO THEFT', 'VEHICLE THEFT', 'CAR THEFT']
        }
        
        # Apply mapping
        for standard_type, variations in crime_mapping.items():
            mask = self.data['crime_type'].isin(variations)
            self.data.loc[mask, 'crime_type'] = standard_type
        
        # Show crime type distribution
        print("📊 Crime type distribution:")
        print(self.data['crime_type'].value_counts().head(10))
        
        return self.data
    
    def remove_outliers(self, method: str = 'iqr') -> pd.DataFrame:
        """Remove spatial outliers based on coordinate distribution"""
        initial_count = len(self.data)
        
        if method == 'iqr':
            # Use IQR method for latitude and longitude
            for coord in ['latitude', 'longitude']:
                Q1 = self.data[coord].quantile(0.25)
                Q3 = self.data[coord].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.data = self.data[
                    (self.data[coord] >= lower_bound) & 
                    (self.data[coord] <= upper_bound)
                ]
        
        final_count = len(self.data)
        removed = initial_count - final_count
        
        if removed > 0:
            print(f"🧹 Removed {removed} spatial outliers using {method} method")
        
        return self.data
    
    def create_zones(self, grid_size: float = 0.01) -> pd.DataFrame:
        """Create grid-based zones for spatial analysis"""
        # Create grid zones based on lat/lng
        self.data['lat_zone'] = (self.data['latitude'] / grid_size).round() * grid_size
        self.data['lng_zone'] = (self.data['longitude'] / grid_size).round() * grid_size
        self.data['zone_id'] = (
            self.data['lat_zone'].astype(str) + "_" + 
            self.data['lng_zone'].astype(str)
        )
        
        print(f"🗺️ Created {self.data['zone_id'].nunique()} spatial zones with {grid_size} degree grid")
        return self.data
    
    def process_all(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Run complete preprocessing pipeline"""
        print("🔄 Starting SafeCity data preprocessing...")
        
        # Load and validate data
        self.load_data(filepath)
        self.validate_columns()
        
        # Clean data
        self.clean_coordinates()
        self.parse_datetime(**kwargs.get('datetime_kwargs', {}))
        self.standardize_crime_types()
        self.remove_outliers()
        self.create_zones(kwargs.get('grid_size', 0.01))
        
        # Store processed data
        self.processed_data = self.data.copy()
        
        print(f"✅ Preprocessing complete! Final dataset: {len(self.processed_data)} records")
        print(f"📍 Date range: {self.processed_data['datetime'].min()} to {self.processed_data['datetime'].max()}")
        
        return self.processed_data
    
    def get_recent_data(self, days: int = 365) -> pd.DataFrame:
        """Get data from the last N days for training"""
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process_all() first.")
        
        cutoff_date = self.processed_data['datetime'].max() - timedelta(days=days)
        recent_data = self.processed_data[self.processed_data['datetime'] >= cutoff_date]
        
        print(f"📅 Retrieved {len(recent_data)} records from last {days} days")
        return recent_data
    
    def save_processed_data(self, output_path: str = None) -> str:
        """Save processed data to CSV"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/processed_crime_data_{timestamp}.csv"
        
        self.processed_data.to_csv(output_path, index=False)
        print(f"💾 Saved processed data to {output_path}")
        return output_path


def generate_sample_data(n_records: int = 5000, 
                        output_path: str = "data/sample_crime_data.csv") -> str:
    """Generate sample crime data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Define city bounds (Mumbai, India coordinates)
    lat_center, lng_center = 19.0760, 72.8777  # Mumbai coordinates
    lat_range, lng_range = 0.15, 0.15
    
    # Crime types with realistic distributions
    crime_types = ['THEFT', 'ASSAULT', 'BURGLARY', 'VANDALISM', 'ROBBERY', 
                   'DRUG', 'VEHICLE', 'FRAUD', 'DOMESTIC', 'OTHER']
    crime_weights = [0.25, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.05, 0.05, 0.05]
    
    # Areas/zones (Mumbai areas)
    areas = ['Andheri', 'Bandra', 'Borivali', 'Churchgate', 'Dadar', 'Goregaon', 
             'Juhu', 'Kurla', 'Malad', 'Marine Lines', 'Matunga', 'Mulund', 
             'Powai', 'Santa Cruz', 'Vile Parle', 'Worli', 'Colaba', 'Fort', 
             'Lower Parel', 'Vikhroli']
    
    # Generate data
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for _ in range(n_records):
        # Random coordinates with some clustering (hotspots)
        if np.random.random() < 0.3:  # 30% chance of hotspot
            # Create hotspots
            hotspot_lat = lat_center + np.random.choice([-0.02, 0.02, 0.03, -0.03])
            hotspot_lng = lng_center + np.random.choice([-0.03, 0.03, 0.04, -0.04])
            lat = hotspot_lat + np.random.normal(0, 0.005)
            lng = hotspot_lng + np.random.normal(0, 0.005)
        else:
            lat = lat_center + np.random.uniform(-lat_range/2, lat_range/2)
            lng = lng_center + np.random.uniform(-lng_range/2, lng_range/2)
        
        # Random datetime in the past year
        random_days = np.random.randint(0, 365)
        random_hours = np.random.randint(0, 24)
        random_minutes = np.random.randint(0, 60)
        
        crime_datetime = start_date + timedelta(
            days=random_days, 
            hours=random_hours, 
            minutes=random_minutes
        )
        
        # Crime type based on weights
        crime_type = np.random.choice(crime_types, p=crime_weights)
        
        # Area
        area = np.random.choice(areas)
        
        data.append({
            'latitude': lat,
            'longitude': lng,
            'crime_type': crime_type,
            'datetime': crime_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'area': area
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"🎯 Generated {n_records} sample crime records")
    print(f"💾 Saved to {output_path}")
    print(f"📊 Crime types: {df['crime_type'].value_counts().to_dict()}")
    
    return output_path


if __name__ == "__main__":
    # Demo usage
    print("🚀 SafeCity Data Preprocessing Demo")
    
    # Generate sample data
    sample_file = generate_sample_data()
    
    # Process the data
    processor = CrimeDataProcessor()
    processed_data = processor.process_all(sample_file)
    
    print("\n📋 Processed Data Summary:")
    print(f"Shape: {processed_data.shape}")
    print(f"Columns: {list(processed_data.columns)}")
    print(f"Zones created: {processed_data['zone_id'].nunique()}")