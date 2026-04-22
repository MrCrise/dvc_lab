import sys
import os
import pandas as pd
import yaml
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.getcwd())
from src.loggers import get_logger


def load_config(config_path):
    with open(config_path) as conf_file:
        return yaml.safe_load(conf_file)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def test_model(config_path):
    logger = get_logger('TEST_MODEL')
    config = load_config(config_path)

    test_path = config['data_split']['testset_path']
    model_path = config['train']['model_path']
    power_path = config['train']['power_path']

    logger.info("Loading test data and model...")
    df_test = pd.read_csv(test_path)
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(power_path, "rb") as f:
        power_trans = pickle.load(f)

    target_col = 'charges'
    
    X_test = df_test.drop(columns=[target_col]).values
    y_test_real = df_test[target_col].values.reshape(-1, 1)

    logger.info("Running prediction...")
    y_pred_scaled = model.predict(X_test)

    y_pred_real = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1))

    rmse, mae, r2 = eval_metrics(y_test_real, y_pred_real)
    
    logger.info(f"Final Test Metrics: R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

    os.makedirs('dvclive', exist_ok=True)
    with open('dvclive/metrics.json', 'w') as f:
        import json
        json.dump({"r2": r2, "rmse": rmse, "mae": mae}, f)

if __name__ == "__main__":
    test_model("./src/config.yaml")
