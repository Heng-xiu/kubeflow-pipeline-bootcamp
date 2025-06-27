# Kubeflow Pipeline Bootcamp：從 Notebook 到工程實作


哈囉！歡迎來到這次的教學專案。  
本系列將帶你一步步將熟悉的 Notebook 教學程式碼，轉換成 **工程等級的 Kubeflow Pipelines 工作流程**。  

這份教學是基於簡單的 Iris 資料集製作，從入門的 `.ipynb` 到完整可部署的 `.yaml` pipeline，逐步拆解過程。

## 👨‍💻 作者與社群

如果您對 MLOps、Kubeflow、MLFlow、生成式 AI 或 Agent 系統有興趣，也歡迎與我聯繫或追蹤我的社群媒體：

- **GitHub**：[Heng-xiu](https://github.com/Heng-xiu)
- **Hugging Face**：[Heng666](https://huggingface.co/Heng666)
- **部落格**：[我的 Medium](https://r23456999.medium.com/)
- **LinkedIn**：[hengshiousheu](https://www.linkedin.com/in/heng-shiou-sheu-85321b70)

<div align="center">
  <a href="https://ko-fi.com/hengshiousheu"><img src="https://github.com/unslothai/unsloth/raw/main/images/Kofi button.png" width="145"></a>
</div>

---

## 目錄結構說明
```
├── 0\_iris.csv               # Iris 資料集 CSV 檔
├── 0\_tutorial.ipynb         # 初學者用的 Iris 教學 Notebook
├── 1\_0\_iris\_lgbm\_pipeline.py # 初步轉成 Kubeflow Pipeline 格式的程式碼
├── 1\_1\_iris\_lgbm\_pipeline.py # 可直接執行的 Pipeline 版本
├── 1\_2\_iris\_lgbm\_pipeline.py # 添加 MLflow 追蹤功能的 Pipeline
└── 2\_編譯管道.ipynb          # Pipeline 編譯成 YAML 檔的 Notebook
```

## 🤔 適合對象
- 熟悉 Jupyter Notebook，但不熟 Kubeflow
- 希望學習如何將實驗碼模組化、管線化
- 想導入 MLflow 作為模型追蹤工具
- 想理解如何從研究程式碼進入 MLOps 世界

## 📘 課程簡報
👉 請參考 [本教學簡報連結](https://gamma.app/docs/MLOps-jmkrewn1yuepb76) 以獲取理論說明與投影片導引。

## 🧭 教學路線與目標
| 步驟     | 內容                          | 目的                                |
| ------ | --------------------------- | --------------------------------- |
| Step 0 | 先從 `0_tutorial.ipynb`          | 熟悉 Iris 資料及 LightGBM 基本訓練流程。                        |
| Step 1 | 透過 `1_0_iris_lgbm_pipeline.py` | 學習如何將訓練程式碼封裝成 Kubeflow Pipeline 的元件。                    |
| Step 2 | 使用 `1_1_iris_lgbm_pipeline.py` | 執行整合後的 Pipeline，了解整體執行架構。                   |
| Step 3 | 探索 `1_2_iris_lgbm_pipeline.py` | 加入 MLflow 進行訓練過程監控與紀錄。                  |
| Step 4 | 最後使用 `2_編譯管道.ipynb`              | 將 Pipeline 編譯成 YAML 檔，準備部署至 Kubeflow 平台。 |


## 環境需求

- Python 3.8+
- Kubeflow Pipelines SDK (kfp) 安裝
- MLflow（用於 `1_2_iris_lgbm_pipeline.py`）
- LightGBM

安裝套件指令：
```bash
pip install -r requirements.txt
````

或直接安裝：

```bash
pip install kfp mlflow lightgbm pandas scikit-learn
```
---

## 使用說明

- 執行 `1_1_iris_lgbm_pipeline.py`：
```bash
python 1_1_iris_lgbm_pipeline.py
````

* 使用 Notebook `2_編譯管道.ipynb` 進行 Pipeline 編譯：
  開啟 Notebook，依序執行各段程式碼，即可產生 `pipeline.yaml`。

* 將編譯好的 YAML 檔部署至 Kubeflow 平台進行執行。
```

## 參考資源

- [Kubeflow Pipelines 官方文件](https://www.kubeflow.org/docs/components/pipelines/)
- [MLflow 官方網站](https://mlflow.org/)
- 教學簡報連結：[你的簡報網址]


---