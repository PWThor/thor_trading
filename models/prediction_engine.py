import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional
import glob

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.postgres_connector import PostgresConnector
from features.feature_generator_wrapper import create_feature_generator


class PredictionEngine:
    """
    Engine for generating predictions from trained models and storing them in the database.
    """
    
    def __init__(
        self,
        db_connector: PostgresConnector,
        feature_generator: any,  # Changed from specific type to any
        model_dir: str = None,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize the PredictionEngine.
        
        Args:
            db_connector: Database connector for retrieving data and storing predictions
            feature_generator: Feature generator for creating features
            model_dir: Directory with trained models
            confidence_threshold: Threshold for considering predictions actionable
        """
        self.db = db_connector
        self.feature_generator = feature_generator
        self.confidence_threshold = confidence_threshold
        
        if model_dir is None:
            self.model_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'saved_models'
            )
        else:
            self.model_dir = model_dir
            
        self.regression_model = None
        self.classification_model = None
        self.model_metadata = None
    
    def load_latest_models(self, horizon: int = 1) -> Tuple[bool, str]:
        """
        Load the latest regression and classification models for a specific prediction horizon.
        
        Args:
            horizon: Prediction horizon in days
            
        Returns:
            Tuple of (success_flag, message)
        """
        # Find the latest regression model
        regression_pattern = os.path.join(self.model_dir, f'price_prediction_{horizon}d_*.pkl')
        regression_models = glob.glob(regression_pattern)
        
        if not regression_models:
            return False, f"No regression models found with pattern: {regression_pattern}"
        
        latest_regression = max(regression_models, key=os.path.getctime)
        
        # Find the latest classification model
        classification_pattern = os.path.join(self.model_dir, f'direction_prediction_{horizon}d_*.pkl')
        classification_models = glob.glob(classification_pattern)
        
        if not classification_models:
            return False, f"No classification models found with pattern: {classification_pattern}"
        
        latest_classification = max(classification_models, key=os.path.getctime)
        
        # Load the models
        try:
            with open(latest_regression, 'rb') as f:
                regression_results = pickle.load(f)
                self.regression_model = regression_results['model']
                self.reg_features = regression_results['features']
                
            with open(latest_classification, 'rb') as f:
                classification_results = pickle.load(f)
                self.classification_model = classification_results['model']
                self.cls_features = classification_results['features']
                
            # Set model metadata
            self.model_metadata = {
                'regression_path': latest_regression,
                'classification_path': latest_classification,
                'regression_training_date': regression_results['training_date'],
                'classification_training_date': classification_results['training_date'],
                'regression_rmse': regression_results['rmse'],
                'classification_f1': classification_results['f1'],
                'prediction_horizon': horizon
            }
            
            return True, f"Loaded models: {os.path.basename(latest_regression)}, {os.path.basename(latest_classification)}"
            
        except Exception as e:
            return False, f"Error loading models: {str(e)}"
    
    def load_specific_models(self, regression_path: str, classification_path: str) -> Tuple[bool, str]:
        """
        Load specific regression and classification models by path.
        
        Args:
            regression_path: Path to regression model
            classification_path: Path to classification model
            
        Returns:
            Tuple of (success_flag, message)
        """
        try:
            with open(regression_path, 'rb') as f:
                regression_results = pickle.load(f)
                self.regression_model = regression_results['model']
                self.reg_features = regression_results['features']
                
            with open(classification_path, 'rb') as f:
                classification_results = pickle.load(f)
                self.classification_model = classification_results['model']
                self.cls_features = classification_results['features']
                
            # Extract horizon from filename
            import re
            horizon_match = re.search(r'prediction_(\d+)d_', regression_path)
            if horizon_match:
                horizon = int(horizon_match.group(1))
            else:
                horizon = 1
                
            # Set model metadata
            self.model_metadata = {
                'regression_path': regression_path,
                'classification_path': classification_path,
                'regression_training_date': regression_results['training_date'],
                'classification_training_date': classification_results['training_date'],
                'regression_rmse': regression_results['rmse'],
                'classification_f1': classification_results['f1'],
                'prediction_horizon': horizon
            }
            
            return True, f"Loaded models: {os.path.basename(regression_path)}, {os.path.basename(classification_path)}"
            
        except Exception as e:
            return False, f"Error loading models: {str(e)}"
    
    def prepare_features(self, days_window: int = 60) -> pd.DataFrame:
        """
        Prepare feature data for prediction using recent market data.
        
        Args:
            days_window: Number of days of historical data to use
            
        Returns:
            DataFrame with features ready for prediction
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_window)
        
        # Fetch raw market data
        market_data = self.db.get_market_data_range('CL-HO-SPREAD', start_date, end_date)
        
        # Get weather data
        weather_data = self.db.get_weather_data_range(start_date, end_date)
        
        # Get EIA data
        eia_data = self.db.get_eia_data_range(start_date, end_date)
        
        # Get COT data
        cot_data = self.db.get_cot_data_range(start_date, end_date)
        
        # Merge all data sources
        merged_data = market_data.copy()
        
        # Merge with other data sources if they exist and are not empty
        if weather_data is not None and not weather_data.empty:
            merged_data = pd.merge(
                merged_data, 
                weather_data, 
                left_index=True, 
                right_index=True, 
                how='left'
            )
            
        if eia_data is not None and not eia_data.empty:
            merged_data = pd.merge(
                merged_data, 
                eia_data, 
                left_index=True, 
                right_index=True, 
                how='left'
            )
            
        if cot_data is not None and not cot_data.empty:
            merged_data = pd.merge(
                merged_data, 
                cot_data, 
                left_index=True, 
                right_index=True, 
                how='left'
            )
        
        # Generate features
        feature_data = self.feature_generator.generate_features(merged_data)
        
        # Forward fill missing values
        feature_data = feature_data.ffill()
        
        return feature_data
    
    def generate_predictions(self, feature_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate predictions using loaded models.
        
        Args:
            feature_data: Optional feature data for prediction. If None, will fetch recent data.
            
        Returns:
            Dictionary with predictions and metadata
        """
        if self.regression_model is None or self.classification_model is None:
            raise ValueError("Models not loaded. Call load_latest_models() first.")
            
        # Prepare features if not provided
        if feature_data is None:
            feature_data = self.prepare_features()
            
        # Get the latest data point
        latest_data = feature_data.iloc[-1]
        
        # Extract features for regression model
        reg_features_data = pd.DataFrame([latest_data[self.reg_features]])
        
        # Extract features for classification model
        cls_features_data = pd.DataFrame([latest_data[self.cls_features]])
        
        # Generate predictions
        price_change_pred = float(self.regression_model.predict(reg_features_data)[0])
        direction_prob = float(self.classification_model.predict_proba(cls_features_data)[0, 1])
        direction_pred = 1 if direction_prob > 0.5 else 0
        
        # Calculate confidence score
        confidence = direction_prob if direction_pred == 1 else (1 - direction_prob)
        
        # Determine if prediction is actionable
        is_actionable = confidence > self.confidence_threshold
        
        # Current price
        current_price = float(latest_data['close'])
        
        # Predicted price
        predicted_price = current_price + price_change_pred
        
        # Create prediction object
        prediction = {
            'timestamp': datetime.now(),
            'prediction_date': latest_data.name,
            'target_date': latest_data.name + timedelta(days=self.model_metadata['prediction_horizon']),
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change': price_change_pred,
            'direction_probability': direction_prob,
            'predicted_direction': direction_pred,
            'confidence': confidence,
            'is_actionable': is_actionable,
            'model_metadata': self.model_metadata
        }
        
        return prediction
    
    def save_prediction(self, prediction: Dict) -> bool:
        """
        Save prediction to database.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            Success flag
        """
        try:
            if hasattr(self.db, 'save_prediction'):
                self.db.save_prediction(prediction)
            return True
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            return False
    
    def predict_and_save(self) -> Dict:
        """
        Generate and save predictions in one step.
        
        Returns:
            Prediction dictionary
        """
        prediction = self.generate_predictions()
        self.save_prediction(prediction)
        return prediction


if __name__ == "__main__":
    # Example usage
    db = PostgresConnector()
    feature_gen = create_feature_generator()
    
    engine = PredictionEngine(
        db_connector=db,
        feature_generator=feature_gen,
        confidence_threshold=0.65
    )
    
    # Load latest models
    success, message = engine.load_latest_models(horizon=1)
    
    if success:
        # Generate and save prediction
        prediction = engine.predict_and_save()
        
        # Print prediction
        print(f"Prediction for {prediction['target_date']}:")
        print(f"Current price: {prediction['current_price']}")
        print(f"Predicted price: {prediction['predicted_price']} (change: {prediction['predicted_change']})")
        print(f"Direction: {'UP' if prediction['predicted_direction'] == 1 else 'DOWN'} with {prediction['confidence']:.2f} confidence")
        print(f"Actionable: {prediction['is_actionable']}")
    else:
        print(f"Error: {message}")