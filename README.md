# TensorRoot

## 中文說明

一個使用 **NumPy** 從零實作的多層感知器（MLP），訓練於 **MNIST** 手寫數字資料集，並在訓練後視覺化模型預測結果。

### 專案特色

- 純 NumPy 前向傳播與反向傳播（不依賴深度學習框架）
- 支援多個隱藏層（目前預設 `hidden_sizes=(256, 128)`）
- 使用 mini-batch 訓練
- 包含 L2 正則化、learning rate decay、early stopping
- 顯示訓練/驗證的 loss 與 accuracy
- 訓練結束後可視化樣本預測結果與信心分數

### 專案結構

```text
TensorRoot/
├─ network.py
└─ README.md
```

### 環境需求

- Python 3.10+（建議 3.11）
- 套件：
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

### 安裝

在專案根目錄執行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scikit-learn matplotlib
```

### 執行

```powershell
python network.py
```

執行流程大致如下：

1. 從 OpenML 下載 MNIST（`fetch_openml`）
2. 取前 20,000 筆資料，做正規化（`/255.0`）與打散
3. 切分訓練集與驗證集
4. 進行多輪訓練並在每輪輸出訓練/驗證指標
5. 依 `val_loss` 保存最佳參數，觸發 early stopping 時提前結束
6. 顯示 10 張驗證集樣本的預測與信心分數

### 主要可調參數（`network.py`）

你可以在主程式區塊調整以下變數：

- `epochs = 100`：最大訓練輪數
- `learning_rate = 0.1`：初始學習率
- `batch_size = 128`：mini-batch 大小
- `l2 = 1e-4`：L2 正則化強度
- `lr_decay = 0.95`：每輪學習率衰減倍率
- `patience = 5`：early stopping 容忍輪數

### 預期輸出

訓練時會看到類似：

```text
Epoch 1/100 | train_acc=... train_loss=... | val_acc=... val_loss=... | lr=...
```

若驗證損失連續多輪未改善，會顯示：

```text
Early stopping triggered.
```

最後會輸出最佳驗證準確率，並跳出影像視窗顯示預測結果。

### 常見問題

#### 1) `AttributeError: 'DataFrame' object has no attribute 'reshape'`

請確認 `fetch_openml` 有設定：

```python
fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
```

`as_frame=False` 可避免回傳 `DataFrame`，改用 NumPy 陣列。

#### 2) `ValueError: cannot reshape array ... into shape (28,28)`

通常是把多張圖片當成單張處理。單張 MNIST 應為長度 `784`，才可 `reshape(28, 28)`。

#### 3) 視窗一張一張跳出很慢

目前使用 `plt.show()` 逐張顯示。可依需求改成子圖（subplot）一次顯示多張。

#### 4) 第一次下載資料很慢或失敗

OpenML 下載速度受網路影響；可稍後重試，或先確認網路/代理設定。

### 後續建議

- 加入測試集（test split）與混淆矩陣
- 新增模型儲存/載入功能
- 把訓練參數改成 CLI 參數（例如 `argparse`）
- 使用 `requirements.txt` 固定版本以提升可重現性

## License
MIT License

## English Guide

An MLP implemented from scratch with **NumPy**, trained on the **MNIST** handwritten digit dataset, with prediction visualization after training.

### Features

- Pure NumPy forward/backward propagation (no deep learning framework dependency)
- Supports multiple hidden layers (default: `hidden_sizes=(256, 128)`)
- Uses mini-batch training
- Includes L2 regularization, learning rate decay, and early stopping
- Reports training/validation loss and accuracy
- Visualizes sample predictions and confidence scores after training

### Project Structure

```text
TensorRoot/
├─ network.py
└─ README.md
```

### Requirements

- Python 3.10+ (3.11 recommended)
- Packages:
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

### Installation

Run the following commands in the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scikit-learn matplotlib
```

### Run

```powershell
python network.py
```

Execution flow:

1. Download MNIST from OpenML (`fetch_openml`)
2. Use the first 20,000 samples, normalize (`/255.0`), and shuffle
3. Split into training and validation sets
4. Train for multiple epochs and print train/validation metrics each epoch
5. Save best parameters by `val_loss` and stop early when patience is exceeded
6. Display predictions and confidence for 10 validation samples

### Main Tunable Parameters (`network.py`)

You can tune these variables in the main script block:

- `epochs = 100`: Maximum training epochs
- `learning_rate = 0.1`: Initial learning rate
- `batch_size = 128`: Mini-batch size
- `l2 = 1e-4`: L2 regularization strength
- `lr_decay = 0.95`: Learning-rate decay factor per epoch
- `patience = 5`: Early stopping patience

### Expected Output

During training, you will see logs like:

```text
Epoch 1/100 | train_acc=... train_loss=... | val_acc=... val_loss=... | lr=...
```

If validation loss does not improve for several epochs, you will see:

```text
Early stopping triggered.
```

Finally, the script prints the best validation accuracy and opens windows to show prediction results.

### FAQ

#### 1) `AttributeError: 'DataFrame' object has no attribute 'reshape'`

Make sure `fetch_openml` is called with:

```python
fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
```

`as_frame=False` ensures NumPy arrays are returned instead of a `DataFrame`.

#### 2) `ValueError: cannot reshape array ... into shape (28,28)`

This usually means multiple images were treated as one. A single MNIST image should be length `784` before `reshape(28, 28)`.

#### 3) Plot windows pop up slowly one by one

The current code uses `plt.show()` per image. You can switch to subplots to display multiple images in one figure.

#### 4) First-time dataset download is slow or fails

OpenML speed depends on network conditions. Retry later or verify network/proxy settings.

### Next Steps

- Add a test split and confusion matrix
- Add model save/load support
- Move training hyperparameters to CLI options (for example, `argparse`)
- Pin dependency versions in `requirements.txt` for better reproducibility

## License
MIT License
