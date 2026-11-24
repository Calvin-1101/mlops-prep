import argparse 
import random 
from pathlib import Path 
import numpy as np 
import pandas as pd 
import joblib 
import os 

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow 
import mlflow.sklearn
from mlflow.models import infer_signature


TARGET = "MedHouseVal" #median house value in units of $100k 
ARTIFACT_DIR = Path('artifacts')

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "california-housing-v1"

def parse_args() -> argparse.Namespace:
    """
    argparse defines what arguments can be added to the command line execution
    these can be called as args.test_size, args.n_estimators, etc 
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-size", type=float, default = 0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=None)
    return p.parse_args()


#set seeds 
def set_seeds(seed: int) -> None: 
    random.seed(seed)
    np.random.seed(seed)

#load data
def load_data() -> pd.DataFrame:
    return fetch_california_housing(as_frame=True).frame


def split(df: pd.DataFrame, test_size: float, seed: int): 
    #perform train_test_split on df 

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(X,y,test_size = test_size, random_state=seed)


def build_models(args: argparse.Namespace) -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=args.n_estimators,
                        max_depth=args.max_depth,
                        random_state=args.seed,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }





def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


    df = load_data()
    X_train, X_test, y_train, y_test = split(df, args.test_size, args.seed)
    print(f"train={len(X_train)} rows, test={len(X_test)} rows")

    best_name, best_model, best_rmse = None, None, float("inf")

    for name, model in build_models(args).items():
        with mlflow.start_run(run_name=name):        
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, preds)
            print(f"{name:<20} RMSE = {rmse:.4f}")

            #track what params made this run 
            mlflow.log_params(vars(args))
            mlflow.log_param("model_type",name)

            #how this run performed 
            mlflow.log_metric('rmse',rmse)

            #log the model itself -> the signature records the expected input and output schema 
            mlflow.sklearn.log_model(
                model,
                name='model',
                signature = infer_signature(X_test,preds),
                input_example=X_test.head(3),
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            print(f"train={len(X_train)} rows, test={len(X_test)} rows")
            #no more checking which is the best model, because we log every run in mlflow
            #this makes the model selection outside of the training script 
    #         if rmse < best_rmse:
    #             best_name, best_model, best_rmse = name, model, rmse


    #no longer using joblib, mlflow does this job 
    # # Local save for now; from Day 3 the MLflow Model Registry replaces this.
    # ARTIFACT_DIR.mkdir(exist_ok=True)
    # joblib.dump(best_model, ARTIFACT_DIR / "model.joblib")
    # print(f"best: {best_name} (RMSE {best_rmse:.4f}) -> {ARTIFACT_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()