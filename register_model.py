
"""
this script is to find the best run and assign the "champion" alias to it 
"""
import os

import mlflow
from mlflow import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "california-housing-v1"
MODEL_NAME = "california-housing"
ALIAS = "champion"

def main() -> None: 
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient() 

    #find the finished run with lowest RMSE

    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string = "attribute.status = 'FINISHED'",
        order_by = ["metrics.rmse ASC"],
        max_results=1,
    )

    if runs.empty:
        raise SystemExit("No finished runs found. Run train.py first")

    best = runs.iloc[0]
    run_id, rmse = best["run_id"], best["metrics.rmse"]
    print(f"best run: {run_id} rmse = {rmse:4f} model = {best['params.model_type']}")

    #register the model -> store the model in the model registry, the registry is a list of models that are candidates for deployment 
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)

    #tags are key-value label assigned to a registered model version 
    #first arg is "which model", second arg is "which version", third and fourth arg is the key and value
    #this tag adds the rmse metric to the model. the metric lives on the run, not on the model itself
    #so someone browsing the models can see the rmse metric, rather than having to click through to the runs of each model 
    #tags is not only restricted to metrics, can be stuff like ["validation_status" "passed"], ["dataset", "24-11-2026"]
    client.set_model_version_tag(MODEL_NAME, mv.version, "rmse", f"{rmse:.4f}")

    #point the alias (champion) at this version 
    client.set_registered_model_alias(MODEL_NAME,ALIAS,mv.version)
    print(f"registered {MODEL_NAME} v{mv.version} as @{ALIAS}")


if __name__ == "__main__":
    main()

    


