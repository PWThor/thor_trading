#!/usr/bin/env python
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.postgres_connector import PostgresConnector
from features.feature_generator_wrapper import create_feature_generator
from models.training.model_trainer import ModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """
    Optimizer for ML models using hyperparameter tuning and feature selection.
    """
    
    def __init__(
        self,
        db_connector: PostgresConnector,
        start_date: datetime,
        end_date: datetime,
        prediction_horizon: int = 1,
        output_dir: str = None,
        cv_folds: int = 5,
        max_evals: int = 50
    ):
        """
        Initialize the model optimizer.
        
        Args:
            db_connector: Database connector
            start_date: Start date for optimization
            end_date: End date for optimization
            prediction_horizon: Days ahead to predict
            output_dir: Directory for optimization output
            cv_folds: Number of cross-validation folds
            max_evals: Maximum number of evaluations for hyperparameter tuning
        """
        self.db = db_connector
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_horizon = prediction_horizon
        self.cv_folds = cv_folds
        self.max_evals = max_evals
        
        # Initialize feature generator
        self.feature_generator = create_feature_generator()
        
        # Set up output directory
        if output_dir is None:
            self.output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'optimized_models',
                f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data structures
        self.feature_data = None
        self.feature_importances = None
        
        # Initialize hyperparameter search space
        self.regression_space = {
            'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500, 1000]),
            'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8]),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'min_child_weight': hp.choice('min_child_weight', [1, 3, 5, 7])
        }
        
        self.classification_space = {
            'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500, 1000]),
            'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8]),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'min_child_weight': hp.choice('min_child_weight', [1, 3, 5, 7]),
            'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 5)
        }
    
    def load_data(self) -> bool:
        """
        Load data for optimization.
        
        Returns:
            Success flag
        """
        logger.info(f"Loading data from {self.start_date} to {self.end_date}")
        
        try:
            # Initialize model trainer for data preparation
            trainer = ModelTrainer(
                db_connector=self.db,
                feature_generator=self.feature_generator,
                prediction_horizon=self.prediction_horizon
            )
            
            # Prepare data
            _, self.feature_data = trainer.prepare_data(self.start_date, self.end_date)
            
            if self.feature_data is None or self.feature_data.empty:
                logger.error("Failed to load data")
                return False
                
            logger.info(f"Loaded {len(self.feature_data)} rows and {len(self.feature_data.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def optimize_regression_model(self) -> Dict:
        """
        Optimize regression model using hyperparameter tuning.
        
        Returns:
            Dictionary with best model and hyperparameters
        """
        logger.info("Optimizing regression model")
        
        # Features to exclude from training
        features_to_exclude = ['open', 'high', 'low', 'close', 'volume', 
                             f'price_change_{self.prediction_horizon}d', 
                             f'direction_{self.prediction_horizon}d']
        
        # Prepare features and target
        X = self.feature_data.drop(columns=features_to_exclude, errors='ignore')
        y = self.feature_data[f'price_change_{self.prediction_horizon}d']
        
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Define the objective function for hyperopt
        def objective(params):
            # Create XGBoost regressor with given parameters
            model = xgb.XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                min_child_weight=params['min_child_weight'],
                objective='reg:squarederror',
                random_state=42
            )
            
            # Perform cross-validation
            cv_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                
                cv_scores.append(rmse)
            
            # Average RMSE across folds
            avg_rmse = np.mean(cv_scores)
            
            return {'loss': avg_rmse, 'status': STATUS_OK}
        
        # Run hyperparameter optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=self.regression_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            verbose=1
        )
        
        # Get best parameters
        param_values = [
            [100, 200, 300, 500, 1000],  # n_estimators
            [3, 4, 5, 6, 7, 8],  # max_depth
            [1, 3, 5, 7]  # min_child_weight
        ]
        
        best_params = {
            'n_estimators': param_values[0][best['n_estimators']],
            'max_depth': param_values[1][best['max_depth']],
            'learning_rate': best['learning_rate'],
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'min_child_weight': param_values[2][best['min_child_weight']],
        }
        
        logger.info(f"Best regression parameters: {best_params}")
        
        # Train final model with best parameters
        best_model = xgb.XGBRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            min_child_weight=best_params['min_child_weight'],
            objective='reg:squarederror',
            random_state=42
        )
        
        best_model.fit(X, y)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        self.feature_importances = feature_importance
        
        # Create results dictionary
        results = {
            'model': best_model,
            'best_params': best_params,
            'feature_importance': feature_importance,
            'training_date': datetime.now(),
            'features': list(X.columns)
        }
        
        return results
    
    def optimize_classification_model(self) -> Dict:
        """
        Optimize classification model using hyperparameter tuning.
        
        Returns:
            Dictionary with best model and hyperparameters
        """
        logger.info("Optimizing classification model")
        
        # Features to exclude from training
        features_to_exclude = ['open', 'high', 'low', 'close', 'volume', 
                             f'price_change_{self.prediction_horizon}d', 
                             f'direction_{self.prediction_horizon}d']
        
        # Prepare features and target
        X = self.feature_data.drop(columns=features_to_exclude, errors='ignore')
        y = self.feature_data[f'direction_{self.prediction_horizon}d']
        
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Define the objective function for hyperopt
        def objective(params):
            # Create XGBoost classifier with given parameters
            model = xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                min_child_weight=params['min_child_weight'],
                scale_pos_weight=params['scale_pos_weight'],
                objective='binary:logistic',
                random_state=42
            )
            
            # Perform cross-validation
            cv_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate F1 score
                from sklearn.metrics import f1_score
                f1 = f1_score(y_test, y_pred)
                
                cv_scores.append(f1)
            
            # Average F1 score across folds
            avg_f1 = np.mean(cv_scores)
            
            return {'loss': -avg_f1, 'status': STATUS_OK}  # Negative because we want to maximize F1
        
        # Run hyperparameter optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=self.classification_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            verbose=1
        )
        
        # Get best parameters
        param_values = [
            [100, 200, 300, 500, 1000],  # n_estimators
            [3, 4, 5, 6, 7, 8],  # max_depth
            [1, 3, 5, 7]  # min_child_weight
        ]
        
        best_params = {
            'n_estimators': param_values[0][best['n_estimators']],
            'max_depth': param_values[1][best['max_depth']],
            'learning_rate': best['learning_rate'],
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'min_child_weight': param_values[2][best['min_child_weight']],
            'scale_pos_weight': best['scale_pos_weight']
        }
        
        logger.info(f"Best classification parameters: {best_params}")
        
        # Train final model with best parameters
        best_model = xgb.XGBClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            min_child_weight=best_params['min_child_weight'],
            scale_pos_weight=best_params['scale_pos_weight'],
            objective='binary:logistic',
            random_state=42
        )
        
        best_model.fit(X, y)
        
        # Get feature importance if not already calculated
        if self.feature_importances is None:
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importances = feature_importance
        
        # Create results dictionary
        results = {
            'model': best_model,
            'best_params': best_params,
            'feature_importance': self.feature_importances,
            'training_date': datetime.now(),
            'features': list(X.columns)
        }
        
        return results
    
    def select_optimal_features(self, feature_importance: pd.DataFrame, threshold: float = 0.9) -> List[str]:
        """
        Select optimal features based on feature importance.
        
        Args:
            feature_importance: Feature importance DataFrame
            threshold: Cumulative importance threshold
            
        Returns:
            List of selected features
        """
        # Sort features by importance
        sorted_features = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative importance
        sorted_features['cumulative_importance'] = sorted_features['importance'].cumsum() / sorted_features['importance'].sum()
        
        # Select features up to threshold
        selected_features = sorted_features[sorted_features['cumulative_importance'] <= threshold]
        
        logger.info(f"Selected {len(selected_features)} features (threshold: {threshold})")
        
        return selected_features['feature'].tolist()
    
    def train_models_with_selected_features(self, selected_features: List[str]) -> Tuple[Dict, Dict]:
        """
        Train models with selected features.
        
        Args:
            selected_features: List of selected features
            
        Returns:
            Tuple with regression and classification results
        """
        logger.info(f"Training models with {len(selected_features)} selected features")
        
        # Features to exclude from training
        features_to_exclude = ['open', 'high', 'low', 'close', 'volume', 
                             f'price_change_{self.prediction_horizon}d', 
                             f'direction_{self.prediction_horizon}d']
        
        # Prepare features and targets
        X = self.feature_data[selected_features]
        y_reg = self.feature_data[f'price_change_{self.prediction_horizon}d']
        y_cls = self.feature_data[f'direction_{self.prediction_horizon}d']
        
        # Train regression model
        reg_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            objective='reg:squarederror',
            random_state=42
        )
        
        reg_model.fit(X, y_reg)
        
        # Train classification model
        cls_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            scale_pos_weight=1,
            objective='binary:logistic',
            random_state=42
        )
        
        cls_model.fit(X, y_cls)
        
        # Create results dictionaries
        reg_results = {
            'model': reg_model,
            'selected_features': selected_features,
            'training_date': datetime.now(),
            'features': selected_features
        }
        
        cls_results = {
            'model': cls_model,
            'selected_features': selected_features,
            'training_date': datetime.now(),
            'features': selected_features
        }
        
        return reg_results, cls_results
    
    def save_models(self, regression_results: Dict, classification_results: Dict) -> Tuple[str, str]:
        """
        Save optimized models to disk.
        
        Args:
            regression_results: Regression model results
            classification_results: Classification model results
            
        Returns:
            Tuple with paths to saved models
        """
        logger.info("Saving optimized models")
        
        # Create timestamps
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save regression model
        reg_filename = f"optimized_regression_{self.prediction_horizon}d_{timestamp}.pkl"
        reg_path = os.path.join(self.output_dir, reg_filename)
        
        with open(reg_path, 'wb') as f:
            pickle.dump(regression_results, f)
        
        # Save classification model
        cls_filename = f"optimized_classification_{self.prediction_horizon}d_{timestamp}.pkl"
        cls_path = os.path.join(self.output_dir, cls_filename)
        
        with open(cls_path, 'wb') as f:
            pickle.dump(classification_results, f)
        
        # Save feature importance plot
        if self.feature_importances is not None:
            plt.figure(figsize=(12, 8))
            
            # Get top 20 features
            top_features = self.feature_importances.head(20)
            
            # Create horizontal bar chart
            ax = top_features.plot(kind='barh', x='feature', y='importance', legend=False)
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title('Top 20 Features by Importance')
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        
        # Save optimization metadata
        metadata = {
            'regression_model': reg_filename,
            'classification_model': cls_filename,
            'prediction_horizon': self.prediction_horizon,
            'optimization_date': timestamp,
            'feature_count': len(regression_results['features']),
            'data_range': {
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d')
            }
        }
        
        with open(os.path.join(self.output_dir, 'optimization_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {self.output_dir}")
        
        return reg_path, cls_path
    
    def run_optimization(self) -> bool:
        """
        Run full optimization pipeline.
        
        Returns:
            Success flag
        """
        # Load data
        if not self.load_data():
            return False
        
        # Optimize regression model
        regression_results = self.optimize_regression_model()
        
        # Optimize classification model
        classification_results = self.optimize_classification_model()
        
        # Select optimal features
        selected_features = self.select_optimal_features(self.feature_importances)
        
        # Train models with selected features
        reg_results, cls_results = self.train_models_with_selected_features(selected_features)
        
        # Save optimized models
        reg_path, cls_path = self.save_models(reg_results, cls_results)
        
        return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimize ML models for trading')
    
    # Date range
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--years', type=int, default=5, help='Number of years of data to use')
    
    # Optimization parameters
    parser.add_argument('--horizon', type=int, default=1, help='Prediction horizon in days')
    parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--evals', type=int, default=50, help='Maximum number of evaluations')
    
    # Output
    parser.add_argument('--output-dir', type=str, help='Directory for optimization output')
    
    return parser.parse_args()

def main():
    """Main entry point for model optimization."""
    args = parse_args()
    
    # Get date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    if args.end:
        try:
            end_date = datetime.strptime(args.end, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid end date format: {args.end}. Use YYYY-MM-DD.")
            return 1
    
    if args.start:
        try:
            start_date = datetime.strptime(args.start, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid start date format: {args.start}. Use YYYY-MM-DD.")
            return 1
    else:
        start_date = end_date - timedelta(days=365 * args.years)
    
    logger.info(f"Optimizing models from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Initialize database connector
    db = PostgresConnector()
    
    # Initialize model optimizer
    optimizer = ModelOptimizer(
        db_connector=db,
        start_date=start_date,
        end_date=end_date,
        prediction_horizon=args.horizon,
        output_dir=args.output_dir,
        cv_folds=args.folds,
        max_evals=args.evals
    )
    
    # Run optimization
    success = optimizer.run_optimization()
    
    if not success:
        logger.error("Optimization failed")
        return 1
    
    logger.info("Optimization complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())