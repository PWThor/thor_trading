#!/usr/bin/env python
# Script to fix model_trainer.py and enable proper data type handling
import sys
import os

# First, backup the original file
def backup_original_file():
    """Create a backup of the original model_trainer.py file."""
    model_trainer_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'models', 'training', 'model_trainer.py'
    )
    backup_path = model_trainer_path + '.bak'
    
    try:
        with open(model_trainer_path, 'r') as f:
            original_content = f.read()
            
        with open(backup_path, 'w') as f:
            f.write(original_content)
            
        print(f"Created backup at {backup_path}")
        return True
    except Exception as e:
        print(f"Error backing up file: {str(e)}")
        return False
        
def patch_model_trainer():
    """Add data type handling to the model_trainer.py file."""
    model_trainer_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'models', 'training', 'model_trainer.py'
    )
    
    # Read the file
    try:
        with open(model_trainer_path, 'r') as f:
            content = f.readlines()
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return False
    
    # Find positions to insert our new method
    prepare_data_index = None
    train_regression_index1 = None
    train_regression_index2 = None
    train_classification_index1 = None
    train_classification_index2 = None
    
    for i, line in enumerate(content):
        if "class ModelTrainer:" in line:
            prepare_data_index = i + 1
        elif "def train_regression_model(" in line:
            train_regression_index1 = i
        elif "X = feature_data.drop(columns=features_to_exclude)" in line and train_regression_index2 is None:
            train_regression_index2 = i
        elif "def train_classification_model(" in line:
            train_classification_index1 = i
        elif "X = feature_data.drop(columns=features_to_exclude)" in line and train_classification_index2 is None and train_regression_index2 is not None:
            train_classification_index2 = i
    
    # Insert our new method after ModelTrainer class
    new_method = [
        "    def _prepare_data_for_ml(self, X):\n",
        "        \"\"\"Prepare data for machine learning by handling problematic data types.\"\"\"\n",
        "        # Make a copy to avoid modifying original\n",
        "        X = X.copy()\n",
        "        \n",
        "        # Check data types that XGBoost can handle\n",
        "        datetime_cols = X.select_dtypes(include=['datetime64']).columns\n",
        "        object_cols = X.select_dtypes(include=['object']).columns\n",
        "        \n",
        "        # Log data types for debugging\n",
        "        print(f\"Data types before cleaning for ML: {X.dtypes.value_counts().to_dict()}\")\n",
        "        print(f\"Found {len(datetime_cols)} datetime columns and {len(object_cols)} object columns\")\n",
        "        \n",
        "        # Drop datetime columns (they can't be used directly by models)\n",
        "        if len(datetime_cols) > 0:\n",
        "            print(f\"Dropping datetime columns: {list(datetime_cols)}\")\n",
        "            X = X.drop(columns=datetime_cols)\n",
        "        \n",
        "        # Drop object columns unless they're categorical \n",
        "        if len(object_cols) > 0:\n",
        "            print(f\"Dropping object columns: {list(object_cols)}\")\n",
        "            X = X.drop(columns=object_cols)\n",
        "        \n",
        "        # Final check for any remaining problematic columns\n",
        "        remaining_non_numeric = X.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool', 'category']).columns\n",
        "        if len(remaining_non_numeric) > 0:\n",
        "            print(f\"Still have non-numeric columns: {list(remaining_non_numeric)}\")\n",
        "            X = X.drop(columns=remaining_non_numeric)\n",
        "            \n",
        "        return X\n",
        "\n"
    ]
    
    # Add call to prepare_data_for_ml in train_regression_model
    data_prep_call = [
        "        # Clean data for ML compatibility\n",
        "        X = self._prepare_data_for_ml(X)\n",
        "\n"
    ]
    
    # Apply the patches
    if prepare_data_index and train_regression_index2 and train_classification_index2:
        # Insert the new method
        content[prepare_data_index:prepare_data_index] = new_method
        
        # Adjust indices because we've added lines
        train_regression_index2 += len(new_method)
        train_classification_index2 += len(new_method)
        
        # Add data prep calls
        content[train_regression_index2 + 3:train_regression_index2 + 3] = data_prep_call
        content[train_classification_index2 + 3:train_classification_index2 + 3] = data_prep_call
        
        # Write the modified file
        try:
            with open(model_trainer_path, 'w') as f:
                f.writelines(content)
            print(f"Successfully patched {model_trainer_path}")
            return True
        except Exception as e:
            print(f"Error writing file: {str(e)}")
            return False
    else:
        print("Could not find all required positions in the file")
        return False

if __name__ == "__main__":
    print("Fixing model_trainer.py to handle data types in ML properly...")
    
    # Backup the original file
    if backup_original_file():
        # Patch the model_trainer.py file
        if patch_model_trainer():
            print("\nFix complete! Now you can run the backtest with:")
            print("cd backtesting")
            print("python run_backtest.py --symbols CL --train-days 90 --test-days 30 --start 2022-01-01 --end 2023-12-31")
        else:
            print("\nFailed to patch model_trainer.py")
    else:
        print("\nFailed to backup model_trainer.py")