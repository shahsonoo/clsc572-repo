import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from my_lipofp_package import generate_morgan_fingerprints, generate_maccs_keys

SMILES_COL="smiles"
TARGET_COL="exp"

def load_dataset(path="data/Lipophilicity.csv"):
    df=pd.read_csv(path)
    return df[SMILES_COL].values, df[TARGET_COL].values.astype(float)

def train_mlp(X_train,X_test,y_train,y_test):
    scaler=StandardScaler()
    y_train_scaled=scaler.fit_transform(y_train.reshape(-1,1)).ravel()
    model=MLPRegressor(hidden_layer_sizes=(128,64),activation="relu",max_iter=500,random_state=42)
    model.fit(X_train,y_train_scaled)
    preds_scaled=model.predict(X_test)
    preds=scaler.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
    return mean_squared_error(y_test,preds,squared=False)

def main():
    smiles,y = load_dataset()
    smiles_train,smiles_test,y_train,y_test = train_test_split(smiles,y,test_size=0.2,random_state=42)
    X_morgan_train=generate_morgan_fingerprints(smiles_train)
    X_morgan_test=generate_morgan_fingerprints(smiles_test)
    X_maccs_train=generate_maccs_keys(smiles_train)
    X_maccs_test=generate_maccs_keys(smiles_test)
    rmse_morgan=train_mlp(X_morgan_train,X_morgan_test,y_train,y_test)
    rmse_maccs=train_mlp(X_maccs_train,X_maccs_test,y_train,y_test)
    print("Morgan RMSE:",rmse_morgan)
    print("MACCS RMSE:",rmse_maccs)
    print("Conda env:",os.getenv("CONDA_DEFAULT_ENV","Unknown"))

if __name__=="__main__":
    main()
