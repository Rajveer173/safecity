"""
SafeCity Risk Prediction Module

Uses Random Forest classifier to predict High/Medium/Low risk zones for the next week.
This is the core AI prediction component that supports patrol planning.

Key features:
- Random Forest classification for risk prediction
- Feature engineering for temporal and spatial patterns
- Zone-based risk assessment
- Weekly prediction horizon
- Model evaluation and validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')


class CrimeRiskPredictor:
    """Predicts crime risk levels for zones using Random Forest"""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize risk predictor
        
        Args:
            n_estimators: Number of trees in Random Forest
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'  # Handle imbalanced classes
        )
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        self.model_metrics = {}
        
    def engineer_features(self, data: pd.DataFrame, 
                         prediction_week: Optional[datetime] = None) -> pd.DataFrame:
        """
        Engineer features for risk prediction
        
        Args:
            data: Processed crime data with zones
            prediction_week: Week to predict for (default: next week)
            
        Returns:
            Feature matrix with zone-level aggregated features
        """
        print("🔧 Engineering features for risk prediction...")
        
        if prediction_week is None:
            prediction_week = data['datetime'].max() + timedelta(days=7)
        
        # Set prediction week bounds
        week_start = prediction_week - timedelta(days=prediction_week.weekday())
        week_end = week_start + timedelta(days=6)
        
        print(f"📅 Engineering features for week: {week_start.date()} to {week_end.date()}")
        
        # Group by zone for feature aggregation
        zone_features = []
        
        for zone_id in data['zone_id'].unique():
            zone_data = data[data['zone_id'] == zone_id]
            
            if len(zone_data) == 0:
                continue
            
            # Basic zone info
            features = {
                'zone_id': zone_id,
                'zone_lat': zone_data['lat_zone'].iloc[0],
                'zone_lng': zone_data['lng_zone'].iloc[0]
            }
            
            # Historical crime features (last 4 weeks before prediction week)
            cutoff_date = week_start - timedelta(days=1)  # Day before prediction week
            historical_data = zone_data[zone_data['datetime'] <= cutoff_date]
            
            # Time windows for feature engineering
            last_week = historical_data[historical_data['datetime'] > cutoff_date - timedelta(days=7)]
            last_2weeks = historical_data[historical_data['datetime'] > cutoff_date - timedelta(days=14)]
            last_4weeks = historical_data[historical_data['datetime'] > cutoff_date - timedelta(days=28)]
            
            # Crime count features
            features.update({
                'crime_count_1week': len(last_week),
                'crime_count_2weeks': len(last_2weeks),
                'crime_count_4weeks': len(last_4weeks),
                'crime_count_total': len(historical_data)
            })
            
            # Crime rate features (crimes per day)
            features.update({
                'crime_rate_1week': len(last_week) / 7,
                'crime_rate_2weeks': len(last_2weeks) / 14,
                'crime_rate_4weeks': len(last_4weeks) / 28
            })
            
            # Crime type diversity
            features.update({
                'crime_types_1week': last_week['crime_type'].nunique() if len(last_week) > 0 else 0,
                'crime_types_4weeks': last_4weeks['crime_type'].nunique() if len(last_4weeks) > 0 else 0
            })
            
            # Temporal patterns
            if len(last_4weeks) > 0:
                mode_vals = last_4weeks['hour'].mode()
                peak_hour = mode_vals.iloc[0] if not mode_vals.empty else 12
                features.update({
                    'avg_hour': last_4weeks['hour'].mean(),
                    'weekend_ratio': last_4weeks['is_weekend'].mean(),
                    'peak_hour': peak_hour
                })
            else:
                features.update({
                    'avg_hour': 12,
                    'weekend_ratio': 0.2,
                    'peak_hour': 12
                })
            
            # Crime type frequencies
            crime_type_counts = last_4weeks['crime_type'].value_counts() if len(last_4weeks) > 0 else pd.Series()
            
            # Top crime types (fill missing with 0)
            for crime_type in ['THEFT', 'ASSAULT', 'BURGLARY', 'VANDALISM', 'ROBBERY']:
                features[f'{crime_type.lower()}_count'] = crime_type_counts.get(crime_type, 0)
            
            # Trend features (comparing recent vs older periods)
            recent_rate = len(last_week) / 7
            older_rate = len(last_4weeks[last_4weeks['datetime'] <= cutoff_date - timedelta(days=7)]) / 21
            features['trend_ratio'] = recent_rate / max(older_rate, 0.1)  # Avoid division by zero
            
            # Day of week patterns
            if len(last_4weeks) > 0:
                dow_counts = last_4weeks['day_of_week'].value_counts()
                for dow in range(7):
                    features[f'dow_{dow}_count'] = dow_counts.get(dow, 0)
            else:
                for dow in range(7):
                    features[f'dow_{dow}_count'] = 0
            
            # Hotspot features (if available)
            if 'is_hotspot' in zone_data.columns:
                hotspot_data = last_4weeks[last_4weeks['is_hotspot']] if len(last_4weeks) > 0 else pd.DataFrame()
                features.update({
                    'hotspot_crimes': len(hotspot_data),
                    'hotspot_ratio': len(hotspot_data) / max(len(last_4weeks), 1)
                })
            else:
                features.update({
                    'hotspot_crimes': 0,
                    'hotspot_ratio': 0
                })
            
            zone_features.append(features)
        
        feature_df = pd.DataFrame(zone_features)
        
        # Fill any remaining NaN values
        feature_df = feature_df.fillna(0)
        
        print(f"✅ Engineered {len(feature_df.columns)-1} features for {len(feature_df)} zones")
        
        return feature_df
    
    def create_risk_labels(self, features: pd.DataFrame) -> pd.Series:
        """
        Create risk labels based on historical crime patterns
        
        Args:
            features: Feature DataFrame with zone-level aggregated data
            
        Returns:
            Risk labels (High/Medium/Low)
        """
        # Use recent crime rate as basis for risk classification
        # Extract the column and convert to Series explicitly
        if isinstance(features, pd.DataFrame):
            crime_rates = features['crime_rate_1week'].values
        else:
            crime_rates = features.values
        
        # Ensure numeric and remove NaN
        crime_rates = pd.Series(crime_rates).fillna(0).astype(float)
        
        # Define thresholds using quantiles
        # Check if we have any non-zero crime rates
        positive_rates = crime_rates[crime_rates > 0]
        if len(positive_rates) > 0:
            high_threshold = float(np.percentile(positive_rates, 75))
            medium_threshold = float(np.percentile(positive_rates, 50))
        else:
            # Fallback thresholds
            high_threshold = 2.0  # 2+ crimes per week
            medium_threshold = 0.5  # 0.5+ crimes per week
        
        # Create risk labels using vectorized operations
        risk_labels = pd.Series(['Low'] * len(crime_rates))
        risk_labels[crime_rates >= medium_threshold] = 'Medium'
        risk_labels[crime_rates >= high_threshold] = 'High'
        risk_labels.name = 'risk_level'
        
        # Print distribution
        print("📊 Risk label distribution:")
        print(risk_labels.value_counts())
        
        return risk_labels
    
    def prepare_training_data(self, data: pd.DataFrame, 
                            weeks_back: int = 8) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data using historical windows
        
        Args:
            data: Processed crime data
            weeks_back: Number of weeks to use for training data
            
        Returns:
            Features and labels for training
        """
        print(f"📚 Preparing training data using {weeks_back} weeks of history...")
        
        # Get the latest date and work backwards
        max_date = data['datetime'].max()
        
        all_features = []
        all_labels = []
        
        # Create training samples for each week in the lookback period
        for week_offset in range(1, weeks_back + 1):
            prediction_week = max_date - timedelta(weeks=week_offset)
            
            # Use data up to the week before prediction week for features
            training_cutoff = prediction_week - timedelta(days=7)
            training_data = data[data['datetime'] <= training_cutoff]
            
            if len(training_data) < 100:  # Skip if insufficient data
                continue
            
            # Engineer features for this week
            week_features = self.engineer_features(training_data, prediction_week)
            
            # Get actual crime data for the prediction week to create labels
            week_start = prediction_week - timedelta(days=prediction_week.weekday())
            week_end = week_start + timedelta(days=6)
            actual_week_data = data[
                (data['datetime'] >= week_start) & (data['datetime'] <= week_end)
            ]
            
            # Calculate actual crime rates for this week
            actual_rates = []
            for zone_id in week_features['zone_id']:
                zone_crimes = actual_week_data[actual_week_data['zone_id'] == zone_id]
                actual_rate = len(zone_crimes) / 7  # Crimes per day
                actual_rates.append(actual_rate)
            
            # Create risk labels based on actual crime rates
            # Use a temporary column for labeling (don't overwrite existing crime_rate_1week)
            label_df = week_features[['zone_id']].copy()
            label_df['crime_rate_1week'] = actual_rates
            week_labels = self.create_risk_labels(label_df)
            
            all_features.append(week_features)
            all_labels.extend(week_labels.tolist())
        
        # Combine all training data
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_labels = pd.Series(all_labels, name='risk_level')
            
            print(f"✅ Created training dataset: {len(combined_features)} samples")
            print(f"📊 Label distribution: {combined_labels.value_counts().to_dict()}")
            
            return combined_features, combined_labels
        else:
            raise ValueError("Insufficient data to create training samples")
    
    def train_model(self, data: pd.DataFrame) -> Dict:
        """
        Train the Random Forest risk prediction model
        
        Args:
            data: Processed crime data
            
        Returns:
            Training metrics and model performance
        """
        print("🤖 Training Random Forest risk prediction model...")
        
        # Prepare training data
        features_df, labels = self.prepare_training_data(data)
        
        # Separate zone_id and coordinates from features
        zone_info = features_df[['zone_id', 'zone_lat', 'zone_lng']]
        feature_columns = [col for col in features_df.columns 
                          if col not in ['zone_id', 'zone_lat', 'zone_lng']]
        X = features_df[feature_columns]
        
        self.feature_names = feature_columns
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Split training and validation data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        
        # Predictions for detailed evaluation
        val_pred = self.model.predict(X_val)
        val_pred_labels = self.label_encoder.inverse_transform(val_pred)
        val_true_labels = self.label_encoder.inverse_transform(y_val)
        
        # Store metrics
        self.model_metrics = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples': len(X),
            'n_features': len(feature_columns)
        }
        
        # Print results
        print(f"✅ Model training complete!")
        print(f"📊 Training accuracy: {train_score:.3f}")
        print(f"📊 Validation accuracy: {val_score:.3f}")
        print(f"📊 Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(val_true_labels, val_pred_labels))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Top 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return self.model_metrics
    
    def predict_risk(self, data: pd.DataFrame, 
                    prediction_week: Optional[datetime] = None) -> pd.DataFrame:
        """
        Predict risk levels for zones
        
        Args:
            data: Processed crime data
            prediction_week: Week to predict for (default: next week)
            
        Returns:
            DataFrame with zone risk predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("🔮 Predicting zone risk levels...")
        
        # Engineer features for prediction
        features_df = self.engineer_features(data, prediction_week)
        
        # Prepare features for prediction
        zone_info = features_df[['zone_id', 'zone_lat', 'zone_lng']]
        X = features_df[self.feature_names]
        
        # Make predictions
        risk_probabilities = self.model.predict_proba(X)
        risk_predictions = self.model.predict(X)
        risk_labels = self.label_encoder.inverse_transform(risk_predictions)
        
        # Create results DataFrame
        results = zone_info.copy()
        results['predicted_risk'] = risk_labels
        results['risk_score'] = np.max(risk_probabilities, axis=1) * 100  # Max probability as score
        
        # Add individual class probabilities
        risk_classes = self.label_encoder.classes_
        for i, risk_class in enumerate(risk_classes):
            results[f'{risk_class.lower()}_probability'] = risk_probabilities[:, i]
        
        # Sort by risk score
        results = results.sort_values('risk_score', ascending=False).reset_index(drop=True)
        
        print(f"✅ Risk predictions complete for {len(results)} zones")
        print(f"📊 Risk distribution: {pd.Series(risk_labels).value_counts().to_dict()}")
        
        return results
    
    def plot_model_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Create visualizations for model analysis"""
        if not self.is_trained:
            print("❌ Model not trained. Call train_model() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('🤖 SafeCity Risk Prediction Model Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Feature importance
        ax1 = axes[0, 0]
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)
        
        ax1.barh(range(len(feature_importance)), feature_importance['importance'])
        ax1.set_yticks(range(len(feature_importance)))
        ax1.set_yticklabels(feature_importance['feature'])
        ax1.set_title('Top 15 Feature Importance')
        ax1.set_xlabel('Importance')
        
        # Plot 2: Model metrics
        ax2 = axes[0, 1]
        metrics = ['Train Acc', 'Val Acc', 'CV Mean']
        values = [
            self.model_metrics['train_accuracy'],
            self.model_metrics['val_accuracy'],
            self.model_metrics['cv_mean']
        ]
        
        bars = ax2.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Model Performance Metrics')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 3: Cross-validation scores
        ax3 = axes[1, 0]
        # This would show CV fold scores if we stored them
        ax3.text(0.5, 0.5, f"Cross-Validation\nMean: {self.model_metrics['cv_mean']:.3f}\nStd: {self.model_metrics['cv_std']:.3f}", 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax3.set_title('Cross-Validation Results')
        ax3.axis('off')
        
        # Plot 4: Model info
        ax4 = axes[1, 1]
        info_text = f"""Model Information
        
Algorithm: Random Forest
Trees: {self.model.n_estimators}
Features: {self.model_metrics['n_features']}
Training Samples: {self.model_metrics['n_samples']}
Classes: {len(self.label_encoder.classes_)}

Risk Levels: {', '.join(self.label_encoder.classes_)}"""
        
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        ax4.set_title('Model Configuration')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path: str = "models/risk_prediction_model.joblib") -> str:
        """Save trained model and preprocessors"""
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'metrics': self.model_metrics,
            'trained_date': datetime.now()
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Saved model to {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model and preprocessors"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.model_metrics = model_data['metrics']
            self.is_trained = True
            
            print(f"✅ Loaded model from {model_path}")
            print(f"📅 Model trained on: {model_data['trained_date']}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


def demo_risk_prediction():
    """Demo function to show risk prediction in action"""
    print("🚀 SafeCity Risk Prediction Demo")
    
    from data_processor import CrimeDataProcessor, generate_sample_data
    
    # Generate sample data with more history for training
    sample_file = generate_sample_data(n_records=8000)
    
    # Process data
    processor = CrimeDataProcessor()
    processed_data = processor.process_all(sample_file)
    
    # Train risk prediction model
    predictor = CrimeRiskPredictor(n_estimators=50)  # Smaller for demo speed
    metrics = predictor.train_model(processed_data)
    
    # Make predictions
    predictions = predictor.predict_risk(processed_data)
    
    # Show results
    print("\n📊 Risk Predictions Summary:")
    print(predictions.head(10).to_string(index=False))
    
    # Plot analysis
    predictor.plot_model_analysis()
    
    # Save model
    model_path = predictor.save_model()
    
    return predictions, model_path


if __name__ == "__main__":
    demo_risk_prediction()