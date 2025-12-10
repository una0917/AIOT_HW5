# AI vs Human 文章偵測器（AI Detector）

## 📋 專案概述

這是一個簡單而強大的 AI vs Human 文章分類工具，可以幫助用戶快速判斷一段文本是否由人工智能生成。該工具基於教學文檔中的高階理論（Perplexity、Burstiness、Stylometry 等），並使用 Streamlit 提供友好的用戶界面。

## 📋 DEMO連結

https://aiothw5-dheplqnhzzzegeetnsmffq.streamlit.app

## 🎯 功能特性

### 核心功能
- ✅ **即時文本分析**：輸入文本立即顯示 AI% / Human% 判定結果
- ✅ **多維度特徵分析**：包含 17 個精心設計的統計特徵
- ✅ **可視化展示**：提供概率分佈、句長分析、特徵值對比等視覺化圖表
- ✅ **詳細統計量**：展示每個特徵的具體數值，便於深入理解

### 核心特徵（基於教學）

#### 1. **句子節奏（Burstiness）**
- 計算句子長度的標準差與平均值的比率
- AI 文本通常具有較低的 Burstiness（句子長度穩定）
- Human 文本通常具有較高的 Burstiness（長短句交錯）

#### 2. **詞彙多樣性（TTR - Type-Token Ratio）**
- 不重複詞彙數 / 總詞彙數
- Human 寫作通常詞彙多樣性更高

#### 3. **Stylometry（文風統計）**
- **Lexical**：功能詞比例、詞長、詞頻分布
- **Syntactic**：標點符號比例、句式重複度
- **Emotion & Noise**：常見連接詞、問號/驚嘆號比例

#### 4. **Zipf's Law（長尾分布）**
- 計算罕見詞（出現 1 次）的比例
- AI 文本通常長尾詞比例較低（更傾向使用常見詞）
- Human 文本通常具有更長的尾巴

#### 5. **詞頻熵（Entropy）**
- 衡量詞彙使用的多樣性
- Human 文本通常具有更高的熵值

## 🚀 快速開始

### 環境需求
- Python 3.8 或更高版本
- Windows / macOS / Linux

### 安裝步驟

#### 方法 1：使用 batch 腳本（Windows）
```bash
cd c:\Users\Una\AIOT\AIOT_HW5
run.bat
```

#### 方法 2：手動安裝
```bash
# 1. 進入專案目錄
cd c:\Users\Una\AIOT\AIOT_HW5

# 2. 安裝相依套件
pip install -r requirements.txt

# 3. 啟動應用
streamlit run ai_detector.py
```

應用將在瀏覽器中自動打開（通常為 http://localhost:8501）

## 📖 使用方法

### 基本操作
1. **輸入文本**：在文本區域粘貼或輸入要分析的內容（至少 50 個字）
2. **點擊分析**：按下「🔍 立即分析」按鈕
3. **查看結果**：
   - 🤖 AI 概率：文本為 AI 生成的概率
   - 👤 Human 概率：文本為人工撰寫的概率

### 側邊欄選項
- ☑️ **顯示詳細特徵**：展示所有 17 個特徵的具體數值
- ☑️ **顯示可視化圖表**：展示特徵值對比和句長分布直方圖

## 📊 輸出解讀

### 判定結論

| AI 概率 | 結論 | 說明 |
|--------|------|------|
| > 70% | 🚨 很可能為 AI 生成 | 具有明顯的 AI 文本特徵 |
| 50-70% | ⚡ 混合特徵 | 可能為 AI 生成或經過大幅修改 |
| < 50% | ✅ 很可能為 Human 撰寫 | 具有明顯的人工撰寫特徵 |

### 特徵解讀

**AI 文本的典型特徵：**
- 低 Burstiness（句子長度穩定）
- 較高的常見連接詞比例（「因此」、「值得注意」等）
- 較低的 Zipf 長尾詞比例
- 標點使用規則化

**Human 文本的典型特徵：**
- 高 Burstiness（長短句交錯）
- 較高的詞彙多樣性（TTR）
- 較高的詞頻熵
- 存在錯字、感嘆詞、問號等自然語言噪音

## 🔬 技術架構

### 模型選擇
```
文本輸入 → 特徵提取 → 特徵標準化 → Logistic Regression → AI 概率
                         ↓
                    17 維特徵向量
```

### 使用的技術棧
- **NLP**：NLTK（分詞、斷句）
- **統計分析**：NumPy、Pandas
- **機器學習**：scikit-learn（LogisticRegression）
- **UI 框架**：Streamlit
- **數據可視化**：Matplotlib、Seaborn

## ⚠️ 重要限制和注意事項

1. **非決定性工具**：
   - 該工具提供的是「訊號與機率」，而非「絕對判定」
   - 不應作為學術倫理調查的唯一證據

2. **語言限制**：
   - 主要針對中文優化，可能不適用於其他語言
   - 對於極短文本（< 50 詞）準確性較低

3. **對抗性文本**：
   - AI 文本經過人工修改後可能被誤判
   - AI 故意注入錯字可能規避偵測

4. **Domain 差異**：
   - 科學論文、社群貼文、程式碼等不同語域可能需要重新調整模型

## 🛠️ 自訂和擴展

### 修改特徵權重
在 `AIDetectorModel.train_sample_model()` 中調整合成訓練數據的均值和標準差：

```python
# AI 文本特徵（示例）
feature_dict = {
    'burstiness': np.random.normal(0.25, 0.1),  # 調整這裡
    # ... 其他特徵
}
```

### 添加新特徵
1. 在 `AIDetectorFeatureExtractor.extract_features()` 中添加計算邏輯
2. 更新 `feature_names` 列表
3. 確保在訓練模型時考慮新特徵

### 使用自己的訓練數據
```python
# 修改 train_sample_model() 方法，使用真實數據集
X, y = load_your_dataset()  # 自己的數據
X_scaled = self.scaler.fit_transform(X)
self.model.fit(X_scaled, y)
```

## 📈 性能表現

### 在合成數據上的表現
- **準確率（Accuracy）**：~85%
- **在隨機測試樣本上**：能有效區分 AI 和 Human 特徵

### 實際使用建議
- 對於明顯的 AI 文本（> 80%）有較高置信度
- 對於邊界情況（40-60%）建議結合人工審查

## 📚 參考資料

該項目基於以下教學內容：
- Perplexity 與困惑度的定義和應用
- Burstiness（句子節奏）與人類寫作的關係
- Stylometry（文風統計）多層次特徵提取
- Zipf's Law 與自然語言的長尾分布
- XAI 可解釋性與模型透明度

## 🤝 貢獻

歡迎提出改進建議或提交問題！

## 📄 許可證

此項目為教學用途，可自由使用和修改。

---

**最後更新**：2025年12月10日  
**作者**：AI Detector Team  
**版本**：1.0.0
