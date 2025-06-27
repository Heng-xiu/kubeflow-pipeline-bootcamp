# iris_lgbm_pipeline.py
from kfp import dsl
from kfp.dsl import component, Output, Artifact, Dataset, Model
from kfp import compiler

# ------------- Component 1: 讀取 CSV -------------
@component(
    base_image="python:3.10",             # 也可換成公司內部鏡像
    packages_to_install=[
        "pandas==2.2.2",
    ],
)
def load_data(iris_csv_path: str, raw_data: Output[Dataset]):
    import pandas as pd
    df = pd.read_csv(iris_csv_path)
    raw_data.path = "/tmp/iris_raw.parquet"
    df.to_parquet(raw_data.path)

# ------------- Component 2: 前處理 -------------
@component(
    base_image="python:3.10",
    packages_to_install=["pandas==2.2.2","scikit-learn==1.5.0"]
)
def preprocess(raw_data: Dataset, processed_data: Output[Dataset]):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(raw_data.path).dropna()
    df["label"] = df["Species"].astype("category").cat.codes
    X = df[["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 儲存為 Parquet，利於下游 component 讀取
    import pyarrow.parquet as pq, pyarrow as pa
    processed_data.path = "/tmp/iris_processed.parquet"
    pq.write_table(
        pa.Table.from_pydict({
            "X_train": [X_train.to_json()],
            "X_test": [X_test.to_json()],
            "y_train": [y_train.to_json()],
            "y_test": [y_test.to_json()],
        }),
        processed_data.path
    )

# ------------- Component 3: 訓練模型 -------------
@component(
    base_image="python:3.10",
    packages_to_install=["lightgbm==4.3.0","pandas==2.2.2","numpy==1.26.4","pyarrow"]
)
def train_model(processed_data: Dataset,
                model_artifact: Output[Model],
                num_boost_round: int = 500,
                learning_rate: float = 0.05):
    import pandas as pd, json, numpy as np, lightgbm as lgb, pyarrow.parquet as pq
    table = pq.read_table(processed_data.path)
    dict_data = {col: table[col][0].as_py() for col in table.column_names}

    X_train = pd.read_json(dict_data["X_train"])
    y_train = pd.read_json(dict_data["y_train"]).iloc[:,0]
    X_test  = pd.read_json(dict_data["X_test"])
    y_test  = pd.read_json(dict_data["y_test"]).iloc[:,0]

    train_ds = lgb.Dataset(X_train, label=y_train)
    valid_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)

    params = dict(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        learning_rate=learning_rate
    )

    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[valid_ds],
        early_stopping_rounds=50
    )
    # 輸出二進位 model.txt
    model_artifact.path = "/tmp/iris_lgbm.txt"
    booster.save_model(model_artifact.path)

# ------------- Component 4: 評估 -------------
@component(
    base_image="python:3.10",
    packages_to_install=["lightgbm==4.3.0","pandas==2.2.2","numpy==1.26.4","scikit-learn==1.5.0","pyarrow"]
)
def evaluate(processed_data: Dataset, model_artifact: Model, metrics: Output[Artifact]):
    import pandas as pd, json, numpy as np, lightgbm as lgb, pyarrow.parquet as pq
    from sklearn.metrics import accuracy_score, classification_report

    table = pq.read_table(processed_data.path)
    dict_data = {col: table[col][0].as_py() for col in table.column_names}
    X_test  = pd.read_json(dict_data["X_test"])
    y_test  = pd.read_json(dict_data["y_test"]).iloc[:,0]

    booster = lgb.Booster(model_file=model_artifact.path)
    y_pred = booster.predict(X_test, num_iteration=booster.best_iteration)
    y_pred_label = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred_label)
    report = classification_report(y_test, y_pred_label)

    # 將結果寫至 metrics.txt
    metrics.path = "/tmp/metrics.txt"
    with open(metrics.path, "w") as f:
        f.write(f"accuracy: {acc}\n\n{report}")

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
    filename = "iris_lgbm_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=iris_lgbm_pipeline,
        package_path=filename
    )
    print(f"✅ 已輸出 {filename}，可上傳至 Kubeflow UI 或用 KFP Client 執行。")
