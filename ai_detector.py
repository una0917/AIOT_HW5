import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import re
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from datetime import datetime

# 設定繁體中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 簡易中文句子切分與詞元化（避免依賴 NLTK punkt）
def split_sentences(text: str):
    """以中文標點或換行切分句子，去除空白項。"""
    # 使用常見中文句子終止符號
    parts = re.split(r'[。！？!?\n]+', text)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences


def tokenize(text: str):
    """簡易詞元化：擷取連續的中文字符或英數字序列作為 token。"""
    # 對中文使用單字/連續中文字視為 token，英文或數字視為整串 token
    tokens = re.findall(r'[\u4e00-\u9fff]+|[A-Za-z0-9]+', text)
    return tokens

# ==================== 特徵提取模組 ====================

class AIDetectorFeatureExtractor:
    """
    從文本中提取 AI vs Human 分類特徵
    基於教學中的 Perplexity、Burstiness、Stylometry、Zipf's Law 等
    """
    
    def __init__(self):
        self.feature_names = [
            'sentence_length_mean',      # 平均句子長度
            'sentence_length_std',       # 句子長度標準差（Burstiness 指標）
            'burstiness',                # 句子節奏 (std / mean)
            'type_token_ratio',          # TTR 詞彙多樣性
            'avg_word_length',           # 平均詞長
            'punctuation_ratio',         # 標點符號比例
            'function_word_ratio',       # 功能詞比例
            'comma_ratio',               # 逗號比例（指示複雜句）
            'lexical_diversity',         # 詞彙不重複度
            'entropy_word_freq',         # 詞頻熵
            'zipf_tail_ratio',           # Zipf 長尾詞比例
            'repeated_structures',       # 重複句式比例
            'common_connectors_ratio',   # 常見連接詞比例
            'question_mark_ratio',       # 問號比例
            'exclamation_ratio',         # 驚嘆號比例
            'passive_voice_indicator',   # 被動語態指標（簡化版）
            'avg_entropy_per_sentence',  # 每句平均熵
        ]
    
    def extract_features(self, text):
        """
        從文本提取所有特徵
        """
        features = {}
        
        # 基礎清理
        text = text.strip()
        if len(text) == 0:
            return {name: 0 for name in self.feature_names}
        
        # 句子級別特徵
        sentences = split_sentences(text)
        sentence_lengths = [len(tokenize(sent)) for sent in sentences]
        features['sentence_length_mean'] = np.mean(sentence_lengths) if sentence_lengths else 0
        features['sentence_length_std'] = np.std(sentence_lengths) if sentence_lengths else 0
        
        # Burstiness: 句子節奏（標準差 / 平均）
        if features['sentence_length_mean'] > 0:
            features['burstiness'] = features['sentence_length_std'] / features['sentence_length_mean']
        else:
            features['burstiness'] = 0
        
        # 詞彙級別特徵
        words = tokenize(text.lower())
        words = [w for w in words if re.match(r"[A-Za-z0-9\u4e00-\u9fff]+$", w)]  # 只保留中文字或英數
        
        if len(words) > 0:
            unique_words = len(set(words))
            features['type_token_ratio'] = unique_words / len(words)
            features['lexical_diversity'] = unique_words / len(words)
        else:
            features['type_token_ratio'] = 0
            features['lexical_diversity'] = 0
        
        # 詞長特徵
        word_lengths = [len(w) for w in words]
        features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
        
        # 標點符號特徵
        total_chars = len(text)
        punctuation_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        features['punctuation_ratio'] = punctuation_count / total_chars if total_chars > 0 else 0
        
        # 特殊標點
        features['comma_ratio'] = text.count('，') / len(sentences) if len(sentences) > 0 else 0
        features['question_mark_ratio'] = text.count('？') / len(sentences) if len(sentences) > 0 else 0
        features['exclamation_ratio'] = text.count('！') / len(sentences) if len(sentences) > 0 else 0
        
        # 功能詞比例（中文常見功能詞）
        function_words = ['的', '了', '和', '是', '在', '以', '有', '等', '與', '或', '及', '而', '但', '這', '那', '其', '因此', '所以', '如果', '就是', '只是']
        function_word_count = sum(text.count(fw) for fw in function_words)
        features['function_word_ratio'] = function_word_count / len(words) if len(words) > 0 else 0
        
        # 常見連接詞（模板化指標）
        common_connectors = ['因此', '總結', '值得注意', '另外', '同時', '此外', '最後', '首先', '其次', '總之', '基於', '考慮到']
        connector_count = sum(text.count(conn) for conn in common_connectors)
        features['common_connectors_ratio'] = connector_count / len(sentences) if len(sentences) > 0 else 0
        
        # Zipf 長尾分布指標
        if len(words) > 0:
            word_freq = Counter(words)
            most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 1
            rare_words = sum(1 for w, freq in word_freq.items() if freq == 1)
            features['zipf_tail_ratio'] = rare_words / len(set(words)) if len(set(words)) > 0 else 0
        else:
            features['zipf_tail_ratio'] = 0
        
        # 詞頻熵
        features['entropy_word_freq'] = self._calculate_entropy(words)
        
        # 重複句式（簡化版：連續相同詞的出現）
        repeated = 0
        for i in range(len(words) - 1):
            if words[i] == words[i+1]:
                repeated += 1
        features['repeated_structures'] = repeated / len(words) if len(words) > 1 else 0
        
        # 被動語態指標（簡化版，計算「被」字出現率）
        features['passive_voice_indicator'] = text.count('被') / len(sentences) if len(sentences) > 0 else 0
        
        # 每句平均熵
        sentence_entropies = [self._calculate_entropy(tokenize(s.lower())) for s in sentences if s.strip()]
        features['avg_entropy_per_sentence'] = np.mean(sentence_entropies) if sentence_entropies else 0
        
        return features
    
    def _calculate_entropy(self, words):
        """計算詞頻熵"""
        if not words:
            return 0
        word_freq = Counter(words)
        total = len(words)
        entropy = 0
        for freq in word_freq.values():
            p = freq / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy / np.log2(len(set(words))) if len(set(words)) > 1 else 0  # 正規化


# ==================== 分類模型 ====================

class AIDetectorModel:
    """
    AI 文章偵測模型
    """
    
    def __init__(self):
        self.extractor = AIDetectorFeatureExtractor()
        self.model = None
        self.model_rf = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = "ai_detector_model.pkl"
        self.scaler_path = "ai_detector_scaler.pkl"
    
    def train_sample_model(self):
        """
        訓練一個簡單的示範模型
        使用合成數據來展示功能
        """
        # 生成合成訓練數據
        n_samples = 100
        features_list = []
        labels = []
        
        # AI 生成的文本特徵（相對穩定、連接詞多、句長均勻）
        for _ in range(n_samples // 2):
            feature_dict = {
                'sentence_length_mean': np.random.normal(15, 3),
                'sentence_length_std': np.random.normal(4, 1),
                'burstiness': np.random.normal(0.25, 0.1),
                'type_token_ratio': np.random.normal(0.6, 0.1),
                'avg_word_length': np.random.normal(3.5, 0.5),
                'punctuation_ratio': np.random.normal(0.08, 0.02),
                'function_word_ratio': np.random.normal(0.25, 0.05),
                'comma_ratio': np.random.normal(0.8, 0.2),
                'lexical_diversity': np.random.normal(0.6, 0.1),
                'entropy_word_freq': np.random.normal(3.5, 0.5),
                'zipf_tail_ratio': np.random.normal(0.35, 0.08),
                'repeated_structures': np.random.normal(0.05, 0.02),
                'common_connectors_ratio': np.random.normal(0.5, 0.1),
                'question_mark_ratio': np.random.normal(0.05, 0.03),
                'exclamation_ratio': np.random.normal(0.02, 0.01),
                'passive_voice_indicator': np.random.normal(0.1, 0.03),
                'avg_entropy_per_sentence': np.random.normal(2.0, 0.3),
            }
            # 確保特徵向量的欄位順序與 extractor.feature_names 對齊
            features_list.append([feature_dict[name] for name in self.extractor.feature_names])
            labels.append(1)  # AI = 1
        
        # Human 寫的文本特徵（波動大、連接詞少、句長不均勻）
        for _ in range(n_samples // 2):
            feature_dict = {
                'sentence_length_mean': np.random.normal(12, 5),
                'sentence_length_std': np.random.normal(8, 2),
                'burstiness': np.random.normal(0.65, 0.15),
                'type_token_ratio': np.random.normal(0.72, 0.1),
                'avg_word_length': np.random.normal(3.2, 0.6),
                'punctuation_ratio': np.random.normal(0.12, 0.04),
                'function_word_ratio': np.random.normal(0.2, 0.06),
                'comma_ratio': np.random.normal(0.5, 0.3),
                'lexical_diversity': np.random.normal(0.72, 0.1),
                'entropy_word_freq': np.random.normal(4.5, 0.6),
                'zipf_tail_ratio': np.random.normal(0.55, 0.1),
                'repeated_structures': np.random.normal(0.12, 0.05),
                'common_connectors_ratio': np.random.normal(0.15, 0.1),
                'question_mark_ratio': np.random.normal(0.15, 0.08),
                'exclamation_ratio': np.random.normal(0.08, 0.04),
                'passive_voice_indicator': np.random.normal(0.05, 0.03),
                'avg_entropy_per_sentence': np.random.normal(3.5, 0.5),
            }
            features_list.append([feature_dict[name] for name in self.extractor.feature_names])
            labels.append(0)  # Human = 0
        
        X = np.array(features_list)
        y = np.array(labels)
        
        # 標準化
        X_scaled = self.scaler.fit_transform(X)
        
        # 訓練模型：Logistic Regression + Random Forest 作為 ensemble
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)
        self.model_rf = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model_rf.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, text):
        """
        預測文本是否為 AI 生成
        返回 (AI_probability, Human_probability, 詳細特徵)
        """
        if not self.is_trained:
            self.train_sample_model()
        
        # 提取特徵
        features_dict = self.extractor.extract_features(text)
        features_array = np.array([features_dict[name] for name in self.extractor.feature_names]).reshape(1, -1)
        
        # 標準化
        features_scaled = self.scaler.transform(features_array)
        
        # 預測概率
        # 使用 ensemble：Logistic + RandomForest 的平均概率
        probs = []
        probs.append(self.model.predict_proba(features_scaled)[0][1])
        if self.model_rf is not None:
            probs.append(self.model_rf.predict_proba(features_scaled)[0][1])
        ai_prob = float(np.mean(probs))

        # 輕微收縮機率（向 0.5 靠攏），降低過度自信
        shrink_factor = 0.8  # 0-1，越小越保守
        ai_prob = 0.5 + (ai_prob - 0.5) * shrink_factor

        # 避免顯示 0% 或 100% 的極端值
        ai_prob = float(np.clip(ai_prob, 1e-4, 1 - 1e-4))
        human_prob = 1 - ai_prob
        
        return ai_prob, human_prob, features_dict
    
    def get_feature_importance(self):
        """
        獲取特徵重要性
        """
        if self.model is None:
            return {}
        
        coefficients = self.model.coef_[0]
        importance_dict = {}
        for name, coef in zip(self.extractor.feature_names, coefficients):
            importance_dict[name] = abs(coef)
        
        return importance_dict


# ==================== Streamlit UI ====================

def main():
    st.set_page_config(
        page_title="AI 文章偵測器",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 自訂樣式
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #FF6B6B;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box-ai {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .result-box-human {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 標題
    st.markdown('<div class="main-title">🤖 AI vs Human 文章偵測器</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">輸入文本，立即分析是否由 AI 生成</div>', unsafe_allow_html=True)
    
    # 初始化模型
    if 'detector_model' not in st.session_state:
        st.session_state.detector_model = AIDetectorModel()
    
    # 側邊欄配置
    with st.sidebar:
        st.header("⚙️ 設定")
        show_features = st.checkbox("顯示詳細特徵", value=True)
        show_visualization = st.checkbox("顯示可視化圖表", value=True)
        st.divider()
        st.subheader("📖 說明")
        st.info(
            """
            **偵測原理：**
            - 分析句子節奏（Burstiness）
            - 計算詞彙多樣性（TTR）
            - 評估常見連接詞模式
            - 計算詞頻熵（Entropy）
            - 識別 Zipf 長尾詞比例
            
            **注意：** 這是輔助工具，不能作為唯一判斷依據。
            """
        )
    
    # 主要內容區域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 輸入文本")
        text_input = st.text_area(
            "在下方輸入要分析的文本（至少 50 個字）：",
            height=250,
            placeholder="粘貼要分析的文本內容..."
        )
    
    # 分析按鈕
    if st.button("🔍 立即分析", key="analyze_btn", use_container_width=True):
        if len(text_input.strip()) < 50:
            st.warning("⚠️ 請輸入至少 50 個字的文本")
        else:
            # 進行預測
            ai_prob, human_prob, features_dict = st.session_state.detector_model.predict(text_input)
            
            # 存儲結果到 session
            st.session_state.last_prediction = {
                'ai_prob': ai_prob,
                'human_prob': human_prob,
                'features': features_dict,
                'text': text_input
            }
            
            st.success("✅ 分析完成！")
    
    # 顯示結果
    if 'last_prediction' in st.session_state:
        prediction = st.session_state.last_prediction
        ai_prob = prediction['ai_prob']
        human_prob = prediction['human_prob']
        features_dict = prediction['features']
        
        st.divider()
        st.subheader("📊 分析結果")
        
        # 結果展示
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.markdown(f"""
            <div class="result-box-ai">
                <h3>🤖 AI 概率</h3>
                <h1>{ai_prob*100:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col_result2:
            st.markdown(f"""
            <div class="result-box-human">
                <h3>👤 Human 概率</h3>
                <h1>{human_prob*100:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # 進度條視覺化
        st.subheader("📈 概率分佈")
        col_prob1, col_prob2 = st.columns(2)
        
        with col_prob1:
            st.metric("AI 生成機率", f"{ai_prob*100:.2f}%", 
                     delta=f"{ai_prob*100 - 50:.1f}%" if ai_prob > 0.5 else "")
        
        with col_prob2:
            st.metric("Human 撰寫機率", f"{human_prob*100:.2f}%", 
                     delta=f"{human_prob*100 - 50:.1f}%" if human_prob > 0.5 else "")
        
        # 繪製進度條
        fig, ax = plt.subplots(figsize=(10, 1))
        ax.barh([0], [ai_prob], color='#667eea', label='AI', height=0.5)
        ax.barh([0], [human_prob], left=[ai_prob], color='#f5576c', label='Human', height=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_yticks([])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        ax.set_title('判定結果', fontweight='bold', pad=10)
        st.pyplot(fig, use_container_width=True)
        
        # 詳細統計特徵
        if show_features:
            st.subheader("📋 詳細特徵分析")
            
            # 建立特徵表格
            feature_df = pd.DataFrame({
                '特徵名稱': st.session_state.detector_model.extractor.feature_names,
                '數值': [features_dict[name] for name in st.session_state.detector_model.extractor.feature_names]
            })
            
            # 分組顯示
            col_feat1, col_feat2 = st.columns(2)
            
            with col_feat1:
                st.markdown("**句子節奏特徵**")
                rhythm_features = {
                    '平均句長': features_dict['sentence_length_mean'],
                    '句長標準差': features_dict['sentence_length_std'],
                    'Burstiness': features_dict['burstiness'],
                }
                for feat_name, feat_val in rhythm_features.items():
                    st.metric(feat_name, f"{feat_val:.3f}")
            
            with col_feat2:
                st.markdown("**詞彙特徵**")
                lexical_features = {
                    'TTR 詞彙多樣性': features_dict['type_token_ratio'],
                    '平均詞長': features_dict['avg_word_length'],
                    '詞頻熵': features_dict['entropy_word_freq'],
                }
                for feat_name, feat_val in lexical_features.items():
                    st.metric(feat_name, f"{feat_val:.3f}")
            
            col_feat3, col_feat4 = st.columns(2)
            
            with col_feat3:
                st.markdown("**結構特徵**")
                struct_features = {
                    '功能詞比例': features_dict['function_word_ratio'],
                    '常見連接詞比例': features_dict['common_connectors_ratio'],
                    'Zipf 長尾比例': features_dict['zipf_tail_ratio'],
                }
                for feat_name, feat_val in struct_features.items():
                    st.metric(feat_name, f"{feat_val:.3f}")
            
            with col_feat4:
                st.markdown("**標點特徵**")
                punct_features = {
                    '標點符號比例': features_dict['punctuation_ratio'],
                    '逗號比例': features_dict['comma_ratio'],
                    '問號比例': features_dict['question_mark_ratio'],
                }
                for feat_name, feat_val in punct_features.items():
                    st.metric(feat_name, f"{feat_val:.3f}")
        
        # 可視化圖表
        if show_visualization:
            st.subheader("📉 視覺化分析")
            
            # 特徵重要性圖
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("**關鍵特徵值對比**")
                key_features = {
                    'Burstiness': features_dict['burstiness'],
                    'TTR': features_dict['type_token_ratio'],
                    '連接詞比例': features_dict['common_connectors_ratio'],
                    '詞頻熵': features_dict['entropy_word_freq'],
                    'Zipf尾巴': features_dict['zipf_tail_ratio'],
                }
                
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#667eea' if features_dict['burstiness'] < 0.4 else '#f5576c' for _ in key_features]
                bars = ax.barh(list(key_features.keys()), list(key_features.values()), color=colors)
                ax.set_xlabel('特徵值', fontweight='bold')
                ax.set_title('關鍵特徵值', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            
            with col_viz2:
                st.markdown("**句長分布分析**")
                text = prediction['text']
                sentences = st.session_state.detector_model.extractor.__class__.__bases__[0] if hasattr(st.session_state.detector_model.extractor, '__bases__') else None
                
                # 計算句長分布
                sentences = split_sentences(text)
                sentence_lengths = [len(tokenize(sent.lower())) for sent in sentences]
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(sentence_lengths, bins=max(5, len(set(sentence_lengths))), 
                       color='#FF6B6B', alpha=0.7, edgecolor='black')
                ax.set_xlabel('句長（詞數）', fontweight='bold')
                ax.set_ylabel('出現次數', fontweight='bold')
                ax.set_title('句長分布', fontweight='bold')
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
        
        # 結論
        st.divider()
        st.subheader("🎯 判定結論")
        
        if ai_prob > 0.7:
            st.warning(f"""
            **⚠️ 很可能為 AI 生成**
            
            該文本具有以下 AI 特徵：
            - 句子節奏平穩（低 Burstiness）
            - 常見連接詞使用較頻繁
            - 詞彙分布較規則
            """)
        elif ai_prob > 0.5:
            st.info(f"""
            **⚡ 可能為 AI 生成或經過大幅修改**
            
            該文本展現了混合特徵，建議人工審查。
            """)
        else:
            st.success(f"""
            **✅ 很可能為 Human 撰寫**
            
            該文本具有以下人類特徵：
            - 句子長度波動較大
            - 詞彙選擇多樣性高
            - 存在自然的語言不規則性
            """)


if __name__ == "__main__":
    main()
