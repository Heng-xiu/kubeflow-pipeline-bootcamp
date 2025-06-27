# iris_lgbm_pipeline.py
from kfp import dsl
from kfp.dsl import component, Output, Artifact, Dataset, Model
from kfp import compiler

# ------------- Component 1: 讀取 CSV -------------
@component(
    base_image="python:3.10",             # 也可換成公司內部鏡像
    packages_to_install=[
        "pandas==2.2.2",
        "pyarrow==15.0.2"          # ⬅️ 新增這行
    ],
)
def load_data(iris_csv_path: str, raw_data: Output[Dataset]):
    import pandas as pd
    df = pd.read_csv(iris_csv_path)
    df.to_parquet(raw_data.path)

# ------------- Component 2: 前處理 -------------
@component(
    base_image="python:3.10",
    packages_to_install=[
        "pandas==2.2.2",
        "scikit-learn==1.5.0",
        "pyarrow==15.0.2"          # 👈 加上
    ]
)
def preprocess(raw_data: Dataset, processed_data: Output[Dataset]):
    import pandas as pd, pyarrow.parquet as pq, os
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(raw_data.path).dropna()
    df["label"] = df["species"].astype("category").cat.codes
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 儲存為 Parquet，利於下游 component 讀取
    import pyarrow.parquet as pq, pyarrow as pa
    
    os.makedirs(processed_data.path, exist_ok=True)
    X_train.to_parquet(f"{processed_data.path}/X_train.parquet", index=False)
    X_test.to_parquet(f"{processed_data.path}/X_test.parquet", index=False)
    y_train.to_frame("label").to_parquet(f"{processed_data.path}/y_train.parquet", index=False)
    y_test.to_frame("label").to_parquet(f"{processed_data.path}/y_test.parquet", index=False)

# ------------- Component 3: 訓練模型 -------------
@component(
    base_image="python:3.10",
    packages_to_install=[
        "lightgbm==4.3.0",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "pyarrow==15.0.2",
        "mlflow==2.13.0",           # ← 新增
        "boto3","tenacity"
    ]
)
def train_model(processed_data: Dataset,
                model_artifact: Output[Model],
                num_boost_round: int = 500,
                learning_rate: float = 0.05):
    import pandas as pd, json, numpy as np, lightgbm as lgb, pyarrow.parquet as pq, io
    import mlflow, os

    

    # ---------- MLflow 設定 ----------
    # ✅ MinIO S3 憑證（視環境調整）
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR-SECRET"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-minio.kubeflow:9000"
    tracking_uri: str = "http://mlflow-server.kubeflow.svc.cluster.local:5000"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iris-lightgbm")

    # ---------- 讀取資料 ----------
    table = pq.read_table(processed_data.path)
    dict_data = {col: table[col][0].as_py() for col in table.column_names}

    X_train = pd.read_parquet(f"{processed_data.path}/X_train.parquet")
    X_test  = pd.read_parquet(f"{processed_data.path}/X_test.parquet")
    y_train = pd.read_parquet(f"{processed_data.path}/y_train.parquet")["label"]
    y_test  = pd.read_parquet(f"{processed_data.path}/y_test.parquet")["label"]

    with mlflow.start_run(run_name="train"):
        # Log 參數
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_boost_round", num_boost_round)

        # ---------- 訓練 ----------
        train_ds = lgb.Dataset(X_train, label=y_train)
        valid_ds = lgb.Dataset(X_test,  label=y_test, reference=train_ds)

        booster = lgb.train(
            params=dict(objective="multiclass",
                        num_class=3,
                        metric="multi_logloss",
                        learning_rate=learning_rate),
            train_set=train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[valid_ds],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=20)
            ]
        )

        # Log 1 個簡單指標 (最佳迭代數對應的 multi_logloss)
        best_iter = booster.best_iteration
        best_loss = booster.best_score["valid_0"]["multi_logloss"]
        mlflow.log_metric("best_iter", best_iter)
        mlflow.log_metric("valid_multi_logloss", best_loss)

        # 保存模型到 artifact + MLflow
        booster.save_model(model_artifact.path)
        mlflow.lightgbm.log_model(booster, artifact_path="model")

# ------------- Component 4: 評估 -------------
@component(
    base_image="python:3.10",
    packages_to_install=[
        "lightgbm==4.3.0",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.5.0",
        "pyarrow==15.0.2",
        "mlflow==2.13.0",           # ← 新增
        "boto3","tenacity"
    ]
)
def evaluate(processed_data: Dataset, model_artifact: Model, metrics: Output[Artifact]):
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    import pyarrow.parquet as pq
    import mlflow, os
    from sklearn.metrics import accuracy_score, classification_report

    # ---------- MLflow ----------
    # ✅ MinIO S3 憑證（視環境調整）
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR-SECRET"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-minio.kubeflow:9000"
    tracking_uri: str = "http://mlflow-server.kubeflow.svc.cluster.local:5000"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iris-lightgbm")

    # ---------- 讀資料 ----------
    X_test = pd.read_parquet(f"{processed_data.path}/X_test.parquet")
    y_test = pd.read_parquet(f"{processed_data.path}/y_test.parquet")["label"]

    booster = lgb.Booster(model_file=model_artifact.path)
    y_pred = booster.predict(X_test, num_iteration=booster.best_iteration)
    y_pred_label = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred_label)
    report = classification_report(y_test, y_pred_label, digits=4)

    # ---------- MLflow Log ----------
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_text(report, artifact_file="classification_report.txt")

    # 同時寫回 KFP metrics artifact
    with open(metrics.path, "w") as f:
        f.write(f"accuracy: {acc:.4f}\n\n{report}")

# ------------- 定義 Pipeline -------------
@dsl.pipeline(
    name="iris-lightgbm-pipeline",
    description="Train LightGBM on Iris dataset with Kubeflow Pipelines"
)
def iris_lgbm_pipeline(iris_csv_path: str = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"):
    raw = load_data(iris_csv_path=iris_csv_path)
    processed = preprocess(raw_data=raw.outputs["raw_data"])
    model = train_model(processed_data=processed.outputs["processed_data"])
    evaluate(processed_data=processed.outputs["processed_data"], model_artifact=model.outputs["model_artifact"])

# ------------- 編譯 -------------
if __name__ == "__main__":
    filename = "iris_lgbm_pipeline_v3.yaml"
    compiler.Compiler().compile(
        pipeline_func=iris_lgbm_pipeline,
        package_path=filename
    )
    print(f"✅ 已輸出 {filename}，可上傳至 Kubeflow UI 或用 KFP Client 執行。")
