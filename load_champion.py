"""
load the champion model by alias and predict on a few rows 
"""

import os 
import mlflow 
from sklearn.datasets import fetch_california_housing

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5000"))

model = mlflow.sklearn.load_model("models:/california-housing@champion")
X = fetch_california_housing(as_frame=True).frame.drop(columns=["MedHouseVal"]).head(3)


print(type(model).__name__)
print(model.predict(X))  # in $100k units