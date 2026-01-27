"""
SafeCity Crime Hotspot Detection Module

Uses DBSCAN clustering to identify crime hotspots and density zones.
This is the core visualization component that creates the "wow" moment on the map.

Key features:
- DBSCAN clustering for hotspot identification
- Density-based zone classification
- Geographic cluster analysis
- Visualization-ready outputs
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class CrimeHotspotDetector:
    """Detects crime hotspots using DBSCAN clustering"""
    
    def __init__(self, eps: float = 0.005, min_samples: int = 10):
        """
        Initialize hotspot detector
        
        Args:
            eps: Maximum distance between samples in a cluster (in degrees)
            min_samples: Minimum samples in a neighborhood for a core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = None
        self.scaler = StandardScaler()
        self.hotspot_data = None
        self.cluster_stats = None
        
    def prepare_features(self, data: pd.DataFrame, 
                        include_temporal: bool = True,
                        include_crime_type: bool = True) -> np.ndarray:
        """
        Prepare features for clustering
        
        Args:
            data: Processed crime data
            include_temporal: Include time-based features
            include_crime_type: Include crime type frequency features
            
        Returns:
            Feature matrix for clustering
        """
        features = []
        feature_names = []
        
        # Core spatial features (always included)
        features.extend([data['latitude'].values, data['longitude'].values])
        feature_names.extend(['latitude', 'longitude'])
        
        if include_temporal:
            # Time-based features
            features.extend([
                data['hour'].values,
                data['day_of_week'].values,
                data['is_weekend'].astype(int).values
            ])
            feature_names.extend(['hour', 'day_of_week', 'is_weekend'])
        
        if include_crime_type:
            # Crime type encoding (simple frequency-based)
            crime_freq = data.groupby('zone_id')['crime_type'].count().to_dict()
            data['crime_frequency'] = data['zone_id'].map(crime_freq).fillna(1)
            features.append(data['crime_frequency'].values)
            feature_names.append('crime_frequency')
        
        feature_matrix = np.column_stack(features)
        self.feature_names = feature_names
        
        print(f"🔧 Prepared {feature_matrix.shape[1]} features: {feature_names}")
        return feature_matrix
    
    def find_optimal_eps(self, features: np.ndarray, k: int = 4) -> float:
        """
        Find optimal eps parameter using k-distance graph
        
        Args:
            features: Feature matrix
            k: Number of nearest neighbors
            
        Returns:
            Suggested eps value
        """
        # Standardize features for distance calculation
        features_scaled = self.scaler.fit_transform(features)
        
        # Calculate k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(features_scaled)
        distances, indices = neighbors_fit.kneighbors(features_scaled)
        
        # Sort distances to k-th nearest neighbor
        k_distances = distances[:, k-1]
        k_distances = np.sort(k_distances)
        
        # Find the "knee" point (optimal eps)
        # Simple method: point of maximum curvature
        diff = np.diff(k_distances)
        diff2 = np.diff(diff)
        knee_point = np.argmax(diff2) + 1
        suggested_eps = k_distances[knee_point]
        
        print(f"📊 Suggested eps value: {suggested_eps:.4f}")
        return suggested_eps
    
    def detect_hotspots(self, data: pd.DataFrame, 
                       auto_eps: bool = True,
                       plot_results: bool = False) -> pd.DataFrame:
        """
        Detect crime hotspots using DBSCAN
        
        Args:
            data: Processed crime data
            auto_eps: Automatically find optimal eps value
            plot_results: Generate visualization plots
            
        Returns:
            Data with cluster labels and hotspot classification
        """
        print("🔍 Detecting crime hotspots with DBSCAN...")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Auto-tune eps if requested
        if auto_eps:
            optimal_eps = self.find_optimal_eps(features)
            # Use the optimal value but adjust for geographic coordinates
            self.eps = min(optimal_eps * 0.001, 0.01)  # Scale down more and cap at 0.01
            print(f"🎯 Using auto-tuned eps: {self.eps:.6f}")
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply DBSCAN
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = self.dbscan.fit_predict(features_scaled)
        
        # Add cluster information to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        data_with_clusters['is_hotspot'] = cluster_labels != -1  # -1 is noise
        
        # Calculate cluster statistics
        self._calculate_cluster_stats(data_with_clusters)
        
        # Classify hotspot intensity
        data_with_clusters = self._classify_hotspot_intensity(data_with_clusters)
        
        self.hotspot_data = data_with_clusters
        
        # Print results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"✅ Hotspot detection complete!")
        print(f"🔥 Found {n_clusters} hotspot clusters")
        print(f"🌫️ {n_noise} noise points ({n_noise/len(data)*100:.1f}%)")
        print(f"📍 {sum(data_with_clusters['is_hotspot'])} incidents in hotspots")
        
        if plot_results:
            self.plot_hotspots()
        
        return data_with_clusters
    
    def _calculate_cluster_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate statistics for each cluster"""
        cluster_stats = {}
        
        for cluster_id in data['cluster'].unique():
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_data = data[data['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'center_lat': cluster_data['latitude'].mean(),
                'center_lng': cluster_data['longitude'].mean(),
                'lat_std': cluster_data['latitude'].std(),
                'lng_std': cluster_data['longitude'].std(),
                'top_crime_type': cluster_data['crime_type'].mode().iloc[0],
                'crime_diversity': cluster_data['crime_type'].nunique(),
                'peak_hour': cluster_data['hour'].mode().iloc[0],
                'weekend_pct': cluster_data['is_weekend'].mean() * 100
            }
            
            # Calculate density (incidents per unit area)
            area = np.pi * stats['lat_std'] * stats['lng_std']  # Approximate ellipse area
            stats['density'] = stats['size'] / max(area, 0.001)  # Avoid division by zero
            
            cluster_stats[cluster_id] = stats
        
        self.cluster_stats = cluster_stats
        return cluster_stats
    
    def _classify_hotspot_intensity(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify hotspots by intensity (High/Medium/Low)"""
        data_copy = data.copy()
        
        # Initialize intensity as 'None' for non-hotspots
        data_copy['hotspot_intensity'] = 'None'
        
        if self.cluster_stats:
            # Get cluster sizes for intensity classification
            cluster_sizes = [stats['size'] for stats in self.cluster_stats.values()]
            
            if len(cluster_sizes) > 0:
                # Use quantiles to classify intensity
                high_threshold = np.percentile(cluster_sizes, 75)
                medium_threshold = np.percentile(cluster_sizes, 50)
                
                for cluster_id, stats in self.cluster_stats.items():
                    mask = data_copy['cluster'] == cluster_id
                    
                    if stats['size'] >= high_threshold:
                        intensity = 'High'
                    elif stats['size'] >= medium_threshold:
                        intensity = 'Medium' 
                    else:
                        intensity = 'Low'
                    
                    data_copy.loc[mask, 'hotspot_intensity'] = intensity
        
        return data_copy
    
    def get_hotspot_summary(self) -> pd.DataFrame:
        """Get summary of detected hotspots"""
        if not self.cluster_stats:
            # Return empty DataFrame silently if no hotspots detected
            return pd.DataFrame()
        
        summary_list = []
        for stats in self.cluster_stats.values():
            summary_list.append({
                'Cluster ID': stats['cluster_id'],
                'Incidents': stats['size'],
                'Center Latitude': round(stats['center_lat'], 6),
                'Center Longitude': round(stats['center_lng'], 6),
                'Primary Crime': stats['top_crime_type'],
                'Crime Diversity': stats['crime_diversity'],
                'Peak Hour': stats['peak_hour'],
                'Weekend %': round(stats['weekend_pct'], 1),
                'Density': round(stats['density'], 2)
            })
        
        summary_df = pd.DataFrame(summary_list)
        summary_df = summary_df.sort_values('Incidents', ascending=False)
        
        return summary_df
    
    def plot_hotspots(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Create visualization plots for hotspots"""
        if self.hotspot_data is None:
            print("❌ No hotspot data available. Run detect_hotspots() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('🔥 SafeCity Crime Hotspot Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Spatial distribution of hotspots
        ax1 = axes[0, 0]
        hotspots = self.hotspot_data[self.hotspot_data['is_hotspot']]
        noise = self.hotspot_data[~self.hotspot_data['is_hotspot']]
        
        # Plot noise points
        ax1.scatter(noise['longitude'], noise['latitude'], 
                   c='lightgray', alpha=0.3, s=1, label='Background')
        
        # Plot hotspots with different colors per cluster
        if len(hotspots) > 0:
            unique_clusters = hotspots['cluster'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster in enumerate(unique_clusters):
                cluster_data = hotspots[hotspots['cluster'] == cluster]
                ax1.scatter(cluster_data['longitude'], cluster_data['latitude'],
                           c=[colors[i]], s=10, alpha=0.7, 
                           label=f'Hotspot {cluster}')
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Crime Hotspots by Location')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Hotspot intensity distribution
        ax2 = axes[0, 1]
        intensity_counts = self.hotspot_data['hotspot_intensity'].value_counts()
        colors_intensity = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow', 'None': 'lightgray'}
        
        bars = ax2.bar(intensity_counts.index, intensity_counts.values,
                      color=[colors_intensity.get(x, 'blue') for x in intensity_counts.index])
        ax2.set_title('Hotspot Intensity Distribution')
        ax2.set_ylabel('Number of Incidents')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Plot 3: Crime types in hotspots vs background
        ax3 = axes[1, 0]
        hotspot_crimes = hotspots['crime_type'].value_counts().head(8) if len(hotspots) > 0 else pd.Series()
        background_crimes = noise['crime_type'].value_counts().head(8) if len(noise) > 0 else pd.Series()
        
        if len(hotspot_crimes) > 0 or len(background_crimes) > 0:
            crime_comparison = pd.DataFrame({
                'Hotspots': hotspot_crimes,
                'Background': background_crimes
            }).fillna(0)
            
            crime_comparison.plot(kind='bar', ax=ax3, rot=45)
            ax3.set_title('Crime Types: Hotspots vs Background')
            ax3.set_ylabel('Incident Count')
            ax3.legend()
        
        # Plot 4: Temporal patterns in hotspots
        ax4 = axes[1, 1]
        if len(hotspots) > 0:
            hourly_hotspots = hotspots.groupby('hour').size()
            hourly_background = noise.groupby('hour').size()
            
            ax4.plot(hourly_hotspots.index, hourly_hotspots.values, 
                    'r-o', label='Hotspots', linewidth=2)
            ax4.plot(hourly_background.index, hourly_background.values, 
                    'b-o', label='Background', alpha=0.7)
            
            ax4.set_title('Temporal Patterns by Hour')
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('Incident Count')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_hotspot_geojson(self, output_path: str = None) -> str:
        """Export hotspots as GeoJSON for web mapping"""
        if self.hotspot_data is None:
            raise ValueError("No hotspot data available. Run detect_hotspots() first.")
        
        if output_path is None:
            output_path = "data/hotspots.geojson"
        
        # This is a simplified version - in a full implementation,
        # you'd use geopandas to create proper GeoJSON
        hotspots = self.hotspot_data[self.hotspot_data['is_hotspot']]
        
        geojson_features = []
        for _, row in hotspots.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['longitude'], row['latitude']]
                },
                "properties": {
                    "cluster": int(row['cluster']),
                    "intensity": row['hotspot_intensity'],
                    "crime_type": row['crime_type'],
                    "datetime": str(row['datetime']),
                    "zone_id": row['zone_id']
                }
            }
            geojson_features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": geojson_features
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"📄 Exported {len(geojson_features)} hotspot points to {output_path}")
        return output_path


def demo_hotspot_detection():
    """Demo function to show hotspot detection in action"""
    print("🚀 SafeCity Hotspot Detection Demo")
    
    # This would typically use real processed crime data
    # For demo, we'll create some sample clustered data
    from data_processor import CrimeDataProcessor, generate_sample_data
    
    # Generate sample data
    sample_file = generate_sample_data(n_records=3000)
    
    # Process data
    processor = CrimeDataProcessor()
    processed_data = processor.process_all(sample_file)
    
    # Detect hotspots
    detector = CrimeHotspotDetector(eps=0.003, min_samples=15)
    hotspot_data = detector.detect_hotspots(processed_data, plot_results=True)
    
    # Show summary
    print("\n📊 Hotspot Summary:")
    summary = detector.get_hotspot_summary()
    print(summary.to_string(index=False))
    
    # Export for web mapping
    detector.export_hotspot_geojson()
    
    return hotspot_data


if __name__ == "__main__":
    demo_hotspot_detection()