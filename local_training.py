#!/usr/bin/env python3
"""
Local XGBoost Training Pipeline
Trains disease risk prediction model locally (for testing before SageMaker)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
from datetime import datetime

class LocalDiseaseRiskTrainer:
    """Local training pipeline for disease risk prediction"""
    
    def __init__(self):
        """Initialize trainer"""
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            'lat_center', 'lon_center', 'year', 'month', 'day_of_year',
            'mean_temp_c', 'total_precip_mm', 'mean_rel_humidity_pct', 'flood_recent'
        ]
        
    def prepare_data(self, csv_path="dataset.csv"):
        """Prepare and split data for training"""
        print("Loading and preparing data...")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        print(f"Dataset shape: {df.shape}")
        
        # Convert time to features
        df['time'] = pd.to_datetime(df['time'])
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        
        # Target variables
        target_columns = {
            'dengue': 'dengue_risk_0_1',
            'malaria': 'malaria_risk_0_1', 
            'lyme': 'lyme_risk_0_1'
        }
        
        # Prepare data for each disease
        self.datasets = {}
        
        for disease, target_col in target_columns.items():
            print(f"\nPreparing {disease} dataset...")
            
            # Filter out rows with missing target values
            disease_df = df[df[target_col].notna()].copy()
            
            if len(disease_df) == 0:
                print(f"No data for {disease}, skipping...")
                continue
            
            # Features and target
            X = disease_df[self.feature_names].values
            y = disease_df[target_col].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.datasets[disease] = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler
            }
            
            print(f"{disease}: Train={len(X_train)}, Test={len(X_test)}")
    
    def train_models(self):
        """Train XGBoost models for each disease"""
        print("\nTraining XGBoost models...")
        
        for disease, data in self.datasets.items():
            print(f"\nTraining {disease} model...")
            
            # Create XGBoost model
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # Train model
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            
            # Calculate metrics
            mse = mean_squared_error(data['y_test'], y_pred)
            r2 = r2_score(data['y_test'], y_pred)
            
            print(f"{disease} - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # Store model and scaler
            self.models[disease] = model
            self.scalers[disease] = data['scaler']
    
    def save_models(self):
        """Save trained models and scalers"""
        print("\nSaving models...")
        
        os.makedirs("models", exist_ok=True)
        
        for disease, model in self.models.items():
            # Save model
            model_path = f"models/{disease}_model.joblib"
            joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = f"models/{disease}_scaler.joblib"
            joblib.dump(self.scalers[disease], scaler_path)
            
            print(f"Saved {disease} model to: {model_path}")
    
    def generate_global_predictions(self):
        """Generate predictions for global grid"""
        print("\nGenerating global predictions...")
        
        # Create global grid
        lats = np.arange(-90, 90, 0.5)
        lons = np.arange(-180, 180, 0.5)
        
        # Current time features
        current_time = datetime.now()
        year = current_time.year
        month = current_time.month
        day_of_year = current_time.timetuple().tm_yday
        
        predictions = []
        
        print(f"Generating predictions for {len(lats)} x {len(lons)} = {len(lats) * len(lons)} grid points...")
        
        for i, lat in enumerate(lats):
            if i % 10 == 0:  # Progress indicator
                print(f"Processing latitude {lat} ({i+1}/{len(lats)})")
            
            for lon in lons:
                # Sample climate features (replace with real data)
                temp = 20 - abs(lat) * 0.5 + np.random.normal(0, 5)
                precip = max(0, 50 + np.random.normal(0, 20))
                humidity = 60 + np.random.normal(0, 15)
                flood = 0  # No recent floods
                
                # Create feature vector
                features = np.array([[
                    lat, lon, year, month, day_of_year,
                    temp, precip, humidity, flood
                ]])
                
                # Get predictions from all models
                row_predictions = {
                    'lat': lat,
                    'lon': lon,
                    'year': year,
                    'month': month
                }
                
                for disease, model in self.models.items():
                    try:
                        # Scale features
                        scaler = self.scalers[disease]
                        features_scaled = scaler.transform(features)
                        
                        # Make prediction
                        pred = model.predict(features_scaled)
                        row_predictions[f'{disease}_risk'] = float(pred[0])
                    except Exception as e:
                        print(f"Error predicting {disease}: {e}")
                        row_predictions[f'{disease}_risk'] = 0.0
                
                predictions.append(row_predictions)
        
        # Save predictions
        predictions_df = pd.DataFrame(predictions)
        predictions_file = "global_risk_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        
        print(f"\nGlobal predictions saved to: {predictions_file}")
        print(f"Predictions shape: {predictions_df.shape}")
        
        # Show sample predictions
        print("\nSample predictions:")
        print(predictions_df.head())
        
        return predictions_df
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("Starting Local Disease Risk Training Pipeline...")
        
        # Step 1: Prepare data
        self.prepare_data()
        
        # Step 2: Train models
        self.train_models()
        
        # Step 3: Save models
        self.save_models()
        
        # Step 4: Generate predictions
        predictions = self.generate_global_predictions()
        
        print("\nPipeline completed successfully!")
        return predictions

def main():
    """Main function to run the training pipeline"""
    trainer = LocalDiseaseRiskTrainer()
    predictions = trainer.run_full_pipeline()
    return predictions

if __name__ == "__main__":
    predictions = main()
