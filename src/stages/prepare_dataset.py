
from pandas._config import config
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
import sys
import os
sys.path.append(os.getcwd())

from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def clear_data(path2data):
    df = pd.read_csv(path2data)
    df = df.drop_duplicates().dropna()
    
    df = df[df['bmi'] < 55] # Убираем экстремальный ИМТ.

    cat_columns = ['sex', 'smoker', 'region']
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])
    return df

def scale_frame(frame):
    df = frame.copy()
    X,y = df.drop(columns = ['Price(euro)']), df['Price(euro)']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1,1))
    return X_scale, Y_scale, power_trans

def featurize(dframe, config) -> None:
    """Генерация новых признаков."""

    logger = get_logger('FEATURIZE')
    logger.info('Create features for Insurance')

    dframe['is_obese'] = (dframe['bmi'] > 30).astype(int)
    dframe['age_smoker'] = dframe['age'] * dframe['smoker']
    
    features_path = config['featurize']['features_path']
    dframe.to_csv(features_path, index=False)


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    df_prep = clear_data(config['data_load']['dataset_csv'])
    df_new_featur = featurize(df_prep, config)
    