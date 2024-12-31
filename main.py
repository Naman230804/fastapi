from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import base64
from io import BytesIO
import time
import pydotplus
from six import StringIO

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    task_type: str = Form(...),  # "classification" or "regression"
    target_column: str = Form(""),
    model_type: str = Form("random_forest"),  # "random_forest" or "decision_tree"
):
    try:
        # Load dataset
        df = pd.read_csv(file.file)
        if target_column not in df.columns:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"Target column '{target_column}' not found in the dataset.",
                },
                status_code=400,
            )

        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        metrics = {}
        start_time = time.time()

        # Select model
        if model_type.lower() == "random_forest":
            if task_type.lower() == "classification":
                model = RandomForestClassifier(random_state=42)
            elif task_type.lower() == "regression":
                model = RandomForestRegressor(random_state=42)
            else:
                return JSONResponse(
                    content={
                        "status": "error",
                        "message": f"Invalid task_type '{task_type}'. Use 'classification' or 'regression'.",
                    },
                    status_code=400,
                )
        elif model_type.lower() == "decision_tree":
            if task_type.lower() == "classification":
                model = DecisionTreeClassifier(random_state=42)
            elif task_type.lower() == "regression":
                model = DecisionTreeRegressor(random_state=42)
            else:
                return JSONResponse(
                    content={
                        "status": "error",
                        "message": f"Invalid task_type '{task_type}'. Use 'classification' or 'regression'.",
                    },
                    status_code=400,
                )
        else:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"Invalid model_type '{model_type}'. Use 'random_forest' or 'decision_tree'.",
                },
                status_code=400,
            )

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        if task_type.lower() == "classification":
            cm = confusion_matrix(y_test, y_pred)
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["precision"] = precision_score(y_test, y_pred, average="weighted")
            metrics["recall"] = recall_score(y_test, y_pred, average="weighted")
            metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")
            metrics["classification_report"] = classification_report(y_test, y_pred)

            # Confusion matrix plot
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            metrics["confusion_matrix_plot"] = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()

            # Decision tree plot (if applicable)
            if model_type.lower() == "decision_tree":
                dot_data = StringIO()
                export_graphviz(
                    model,
                    out_file=dot_data,
                    feature_names=X.columns,
                    class_names=[str(cls) for cls in model.classes_],  # Convert to strings
                    filled=True,
                    rounded=True,
                    special_characters=True,
                )
                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                buffer = BytesIO()
                graph.write_png(buffer)
                buffer.seek(0)
                metrics["decision_tree_plot"] = base64.b64encode(buffer.read()).decode("utf-8")
                buffer.close()

        elif task_type.lower() == "regression":
            metrics["mean_absolute_error"] = mean_absolute_error(y_test, y_pred)
            metrics["mean_squared_error"] = mean_squared_error(y_test, y_pred)
            metrics["r2_score"] = r2_score(y_test, y_pred)

        # Capture training time
        metrics["training_time"] = time.time() - start_time

        # Include feature importances
        if hasattr(model, "feature_importances_"):
            metrics["feature_importances"] = model.feature_importances_.tolist()
        elif hasattr(model, "coef_"):
            metrics["coefficients"] = model.coef_.tolist()
            metrics["intercept"] = model.intercept_

        return JSONResponse(
            content={
                "status": "success",
                "message": "Model trained successfully.",
                "metrics": metrics,
            }
        )

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
