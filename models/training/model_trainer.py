import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from typing import Dict, List, Tuple, Union, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from connectors.postgres_connector import PostgresConnector
from features.feature_generator_wrapper import create_feature_generator


class ModelTrainer:
    def _prepare_data_for_ml(self, X):
        """Prepare data for machine learning by handling problematic data types."""
        # Make a copy to avoid modifying original
        X = X.copy()
        
        # Check data types that XGBoost can handle
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        object_cols = X.select_dtypes(include=['object']).columns
        
        # Log data types for debugging
        print(f"Data types before cleaning for ML: {X.dtypes.value_counts().to_dict()}")
        print(f"Found {len(datetime_cols)} datetime columns and {len(object_cols)} object columns")
        
        # Drop datetime columns (they can't be used directly by models)
        if len(datetime_cols) > 0:
            print(f"Dropping datetime columns: {list(datetime_cols)}")
            X = X.drop(columns=datetime_cols)
        
        # Drop object columns unless they're categorical 
        if len(object_cols) > 0:
            print(f"Dropping object columns: {list(object_cols)}")
            X = X.drop(columns=object_cols)
        
        # Final check for any remaining problematic columns
        remaining_non_numeric = X.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool', 'category']).columns
        if len(remaining_non_numeric) > 0:
            print(f"Still have non-numeric columns: {list(remaining_non_numeric)}")
            X = X.drop(columns=remaining_non_numeric)
            
        return X

    """
    Model training pipeline for crack spread prediction and trading signal generation.
    Handles feature preparation, model training, evaluation, and persistence.
    """
    
    def __init__(
        self,
        db_connector: PostgresConnector,
        feature_generator: any,  # Changed from FeatureGenerator to any
        model_output_dir: str = None,
        prediction_horizon: int = 1,
        test_size: int = 60,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the ModelTrainer.
        
        Args:
            db_connector: Database connector for retrieving data
            feature_generator: Feature generator for creating features
            model_output_dir: Directory to save trained models
            prediction_horizon: Days ahead to predict (default 1 day)
            test_size: Number of days to use for testing
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.db = db_connector
        self.feature_generator = feature_generator
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        if model_output_dir is None:
            self.model_output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'saved_models'
            )
        else:
            self.model_output_dir = model_output_dir
            
        os.makedirs(self.model_output_dir, exist_ok=True)
        
        self.regression_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        self.classification_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'scale_pos_weight': [1, 3, 5]
        }
    
    def prepare_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for model training by fetching market data and generating features.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Tuple containing market data and feature dataframes
        """
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
        
        # Create target variables for prediction horizon
        feature_data[f'price_change_{self.prediction_horizon}d'] = feature_data['close'].shift(-self.prediction_horizon) - feature_data['close']
        feature_data[f'direction_{self.prediction_horizon}d'] = (feature_data[f'price_change_{self.prediction_horizon}d'] > 0).astype(int)
        
        # Forward fill missing values and drop rows with remaining NaNs
        feature_data = feature_data.ffill()
        feature_data = feature_data.dropna()
        
        return merged_data, feature_data
    
    def train_regression_model(
        self, 
        feature_data: pd.DataFrame,
        target_col: str = None,
        features_to_exclude: List[str] = None
    ) -> Dict:
        """
        Train a regression model for price prediction.
        
        Args:
            feature_data: DataFrame with features and target variables
            target_col: Target column name (default: price_change_{prediction_horizon}d)
            features_to_exclude: List of features to exclude from training
            
        Returns:
            Dictionary with model and training results
        """
        if target_col is None:
            target_col = f'price_change_{self.prediction_horizon}d'
            
        if features_to_exclude is None:
            features_to_exclude = ['open', 'high', 'low', 'close', 'volume', 
                                 f'price_change_{self.prediction_horizon}d', 
                                 f'direction_{self.prediction_horizon}d']
        
        # Prepare features and target
        X = feature_data.drop(columns=features_to_exclude)
        y = feature_data[target_col]
        
        # Clean data for ML compatibility
        X = self._prepare_data_for_ml(X)

        # Train/test split using recent data for testing
        split_idx = len(X) - self.test_size
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Initialize XGBoost regressor
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_state
        )
        
        # Hyperparameter tuning with time series cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.regression_param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create results dictionary
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'rmse': rmse,
            'feature_importance': feature_importance,
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'features': list(X.columns),
            'target': target_col
        }
        
        return results
    
    def train_classification_model(
        self, 
        feature_data: pd.DataFrame,
        target_col: str = None,
        features_to_exclude: List[str] = None
    ) -> Dict:
        """
        Train a classification model for direction prediction.
        
        Args:
            feature_data: DataFrame with features and target variables
            target_col: Target column name (default: direction_{prediction_horizon}d)
            features_to_exclude: List of features to exclude from training
            
        Returns:
            Dictionary with model and training results
        """
        if target_col is None:
            target_col = f'direction_{self.prediction_horizon}d'
            
        if features_to_exclude is None:
            features_to_exclude = ['open', 'high', 'low', 'close', 'volume', 
                                 f'price_change_{self.prediction_horizon}d', 
                                 f'direction_{self.prediction_horizon}d']
        
        # Prepare features and target
        X = feature_data.drop(columns=features_to_exclude)
        y = feature_data[target_col]
        
        # Clean data for ML compatibility
        X = self._prepare_data_for_ml(X)
        
        # Train/test split using recent data for testing
        split_idx = len(X) - self.test_size
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Initialize XGBoost classifier
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=self.random_state
        )
        
        # Hyperparameter tuning with time series cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.classification_param_grid,
            cv=tscv,
            scoring='f1',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create results dictionary
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'feature_importance': feature_importance,
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'features': list(X.columns),
            'target': target_col
        }
        
        return results
    
    def save_model(self, results: Dict, model_type: str):
        """
        Save model and training results to disk.
        
        Args:
            results: Dictionary with model and training results
            model_type: Type of model ('regression' or 'classification')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create model filename
        if model_type == 'regression':
            model_name = f'price_prediction_{self.prediction_horizon}d_{timestamp}.pkl'
        else:
            model_name = f'direction_prediction_{self.prediction_horizon}d_{timestamp}.pkl'
        
        # Create full path
        model_path = os.path.join(self.model_output_dir, model_name)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(results, f)
            
        # Log to database
        model_metadata = {
            'model_path': model_path,
            'model_type': model_type,
            'prediction_horizon': self.prediction_horizon,
            'training_date': results['training_date'],
            'performance': results['rmse'] if model_type == 'regression' else results['f1'],
            'feature_count': len(results['features']),
            'top_features': ','.join(results['feature_importance']['feature'].iloc[:10].tolist()),
            'parameters': str(results['best_params'])
        }
        
        # Save metadata to database if method exists
        if hasattr(self.db, 'save_model_metadata'):
            self.db.save_model_metadata(model_metadata)
        
        return model_path
    
    def train_and_save_models(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, str]:
        """
        Complete training pipeline: prepare data, train models, save to disk.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Dictionary with paths to saved models
        """
        # Prepare data
        _, feature_data = self.prepare_data(start_date, end_date)
        
        # Train regression model
        regression_results = self.train_regression_model(feature_data)
        regression_path = self.save_model(regression_results, 'regression')
        
        # Train classification model
        classification_results = self.train_classification_model(feature_data)
        classification_path = self.save_model(classification_results, 'classification')
        
        return {
            'regression_model': regression_path,
            'classification_model': classification_path
        }


if __name__ == "__main__":
    # Example usage
    from datetime import datetime, timedelta
    
    # Initialize connectors
    db = PostgresConnector()
    feature_gen = create_feature_generator()
    
    # Initialize model trainer
    trainer = ModelTrainer(
        db_connector=db,
        feature_generator=feature_gen,
        prediction_horizon=1,
        test_size=60,
        cv_folds=5
    )
    
    # Train models using 2 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Train and save models
    model_paths = trainer.train_and_save_models(start_date, end_date)
    
    print(f"Models saved to: {model_paths}")