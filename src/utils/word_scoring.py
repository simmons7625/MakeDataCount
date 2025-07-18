"""
単語スコアリング関数
trainデータセットのラベル間TF-IDFから算出したスコアを使用
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import json

# NLTK データの準備
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class WordScorer:
    """単語スコアリングクラス"""
    
    def __init__(self, dataset_path: str = "./dataset"):
        self.dataset_path = Path(dataset_path)
        self.train_labels = None
        self.label_tfidf_scores = {}
        self.label_vocabularies = {}
        self.label_word_counts = {}
        self.tfidf_vectorizer = None
        self.is_trained = False
        
    def load_train_data(self) -> pd.DataFrame:
        """訓練データを読み込み"""
        train_labels_path = self.dataset_path / "train_labels.csv"
        if not train_labels_path.exists():
            raise FileNotFoundError(f"Train labels file not found: {train_labels_path}")
        
        self.train_labels = pd.read_csv(train_labels_path)
        print(f"Loaded {len(self.train_labels)} training samples")
        return self.train_labels
    
    def preprocess_text(self, text: str) -> List[str]:
        """テキストの前処理（02ノートブックと同じ処理）"""
        if not text:
            return []
        
        # 小文字化
        text = text.lower()
        
        # 数字、特殊文字を除去
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # トークン化
        tokens = word_tokenize(text)
        
        # ストップワード除去
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # レンマ化
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """PDFからテキストを抽出"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text("text", flags=fitz.TEXTFLAGS_TEXT)
                    if text.strip():
                        text_parts.append(text)
                except Exception as page_error:
                    print(f"Warning: Error processing page {page_num} in {pdf_path}: {page_error}")
                    continue
            
            doc.close()
            return ' '.join(text_parts)
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return ""
    
    def build_label_texts(self) -> Dict[str, List[str]]:
        """ラベル別のテキストを構築"""
        if self.train_labels is None:
            self.load_train_data()
        
        pdf_dir = self.dataset_path / "train" / "PDF"
        
        label_texts = {
            'Primary': [],
            'Secondary': [],
            'Missing': []
        }
        
        processed_count = 0
        for _, row in self.train_labels.iterrows():
            article_id = row['article_id']
            label = row['type']
            
            pdf_path = pdf_dir / f"{article_id}.pdf"
            
            if pdf_path.exists():
                text = self.extract_text_from_pdf(pdf_path)
                if text:
                    label_texts[label].append(text)
                    processed_count += 1
        
        print(f"Processed {processed_count} articles")
        for label, texts in label_texts.items():
            print(f"  {label}: {len(texts)} articles")
        
        return label_texts
    
    def build_word_frequency_analysis(self, label_texts: Dict[str, List[str]]) -> Dict[str, Counter]:
        """ラベル別の単語頻度分析"""
        label_word_counts = {}
        
        for label, texts in label_texts.items():
            print(f"Processing {label} articles...")
            
            all_words = []
            for text in texts:
                words = self.preprocess_text(text)
                all_words.extend(words)
            
            word_counts = Counter(all_words)
            label_word_counts[label] = word_counts
            self.label_vocabularies[label] = set(all_words)
            
            print(f"  Total words: {len(all_words):,}")
            print(f"  Unique words: {len(word_counts):,}")
        
        self.label_word_counts = label_word_counts
        return label_word_counts
    
    def build_tfidf_scores(self, label_texts: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """TF-IDFスコアを計算"""
        # 各ラベルの全テキストを結合
        label_documents = {}
        for label, texts in label_texts.items():
            if texts:
                # 前処理済みのテキストを結合
                processed_texts = [' '.join(self.preprocess_text(text)) for text in texts]
                label_documents[label] = ' '.join(processed_texts)
        
        if not label_documents:
            print("No documents to process")
            return {}
        
        # TF-IDFベクトライザーを作成
        documents = list(label_documents.values())
        labels = list(label_documents.keys())
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # 各ラベルの特徴的な単語を抽出
        label_tfidf_scores = {}
        for i, label in enumerate(labels):
            tfidf_scores = tfidf_matrix[i].toarray()[0]
            word_scores = dict(zip(feature_names, tfidf_scores))
            label_tfidf_scores[label] = word_scores
            
            # 上位10単語を表示
            top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n{label} ラベルの特徴的な単語 (TF-IDF):")
            for word, score in top_words:
                print(f"  {word}: {score:.4f}")
        
        self.label_tfidf_scores = label_tfidf_scores
        return label_tfidf_scores
    
    def train_scorer(self) -> None:
        """スコアラーを訓練"""
        print("Training word scorer...")
        
        # ラベル別テキストを構築
        label_texts = self.build_label_texts()
        
        # 単語頻度分析
        self.build_word_frequency_analysis(label_texts)
        
        # TF-IDFスコア計算
        self.build_tfidf_scores(label_texts)
        
        self.is_trained = True
        print("Word scorer training completed!")
    
    def get_word_scores(self, words: List[str]) -> Dict[str, Dict[str, float]]:
        """単語リストに対するラベル別スコアを取得"""
        if not self.is_trained:
            raise ValueError("Scorer must be trained first. Call train_scorer()")
        
        word_scores = {}
        for word in words:
            word_scores[word] = {}
            for label in ['Primary', 'Secondary', 'Missing']:
                score = self.label_tfidf_scores.get(label, {}).get(word, 0.0)
                word_scores[word][label] = score
        
        return word_scores
    
    def score_text(self, text: str) -> Dict[str, float]:
        """テキストをスコアリング"""
        if not self.is_trained:
            raise ValueError("Scorer must be trained first. Call train_scorer()")
        
        # テキストを前処理
        words = self.preprocess_text(text)
        
        # 各ラベルでのスコア計算
        label_scores = {'Primary': 0.0, 'Secondary': 0.0, 'Missing': 0.0}
        
        for word in words:
            for label in label_scores.keys():
                score = self.label_tfidf_scores.get(label, {}).get(word, 0.0)
                label_scores[label] += score
        
        # 単語数で正規化
        if len(words) > 0:
            for label in label_scores.keys():
                label_scores[label] /= len(words)
        
        return label_scores
    
    def get_characteristic_words(self, label: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """ラベルの特徴的な単語を取得"""
        if not self.is_trained:
            raise ValueError("Scorer must be trained first. Call train_scorer()")
        
        if label not in self.label_tfidf_scores:
            raise ValueError(f"Unknown label: {label}")
        
        word_scores = self.label_tfidf_scores[label]
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return top_words
    
    def compare_labels(self, word: str) -> Dict[str, float]:
        """特定の単語のラベル間スコア比較"""
        if not self.is_trained:
            raise ValueError("Scorer must be trained first. Call train_scorer()")
        
        scores = {}
        for label in ['Primary', 'Secondary', 'Missing']:
            scores[label] = self.label_tfidf_scores.get(label, {}).get(word, 0.0)
        
        return scores
    
    def save_scorer(self, save_path: Path) -> None:
        """スコアラーを保存"""
        if not self.is_trained:
            raise ValueError("Scorer must be trained first. Call train_scorer()")
        
        save_data = {
            'label_tfidf_scores': self.label_tfidf_scores,
            'label_vocabularies': {k: list(v) for k, v in self.label_vocabularies.items()},
            'label_word_counts': {k: dict(v) for k, v in self.label_word_counts.items()},
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'is_trained': self.is_trained
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Word scorer saved to {save_path}")
    
    def load_scorer(self, load_path: Path) -> None:
        """スコアラーを読み込み"""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.label_tfidf_scores = save_data['label_tfidf_scores']
        self.label_vocabularies = {k: set(v) for k, v in save_data['label_vocabularies'].items()}
        self.label_word_counts = {k: Counter(v) for k, v in save_data['label_word_counts'].items()}
        self.tfidf_vectorizer = save_data['tfidf_vectorizer']
        self.is_trained = save_data['is_trained']
        
        print(f"Word scorer loaded from {load_path}")
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        stats = {
            "status": "trained",
            "labels": list(self.label_tfidf_scores.keys()),
            "vocabulary_sizes": {
                label: len(vocab) for label, vocab in self.label_vocabularies.items()
            },
            "total_words": {
                label: sum(counts.values()) for label, counts in self.label_word_counts.items()
            },
            "unique_words": {
                label: len(counts) for label, counts in self.label_word_counts.items()
            }
        }
        
        return stats


def create_word_scorer(dataset_path: str = "./dataset") -> WordScorer:
    """単語スコアラーを作成・訓練"""
    scorer = WordScorer(dataset_path)
    scorer.train_scorer()
    return scorer


def score_document_chunks(chunks: List[Dict], scorer: WordScorer) -> List[Dict]:
    """文書チャンクにスコアを付与"""
    scored_chunks = []
    
    for chunk in chunks:
        text = chunk.get('text', '')
        scores = scorer.score_text(text)
        
        # スコアを追加
        scored_chunk = chunk.copy()
        scored_chunk['label_scores'] = scores
        scored_chunk['predicted_label'] = max(scores.items(), key=lambda x: x[1])[0]
        scored_chunk['confidence'] = max(scores.values())
        
        scored_chunks.append(scored_chunk)
    
    return scored_chunks


def get_dataset_insights(scorer: WordScorer) -> Dict:
    """データセットの洞察を取得"""
    if not scorer.is_trained:
        raise ValueError("Scorer must be trained first")
    
    insights = {
        "label_statistics": scorer.get_statistics(),
        "characteristic_words": {},
        "label_overlaps": {}
    }
    
    # 各ラベルの特徴的な単語
    for label in ['Primary', 'Secondary', 'Missing']:
        insights["characteristic_words"][label] = scorer.get_characteristic_words(label, 15)
    
    # ラベル間の単語重複分析
    all_words = set()
    for vocab in scorer.label_vocabularies.values():
        all_words.update(vocab)
    
    label_pairs = [('Primary', 'Secondary'), ('Primary', 'Missing'), ('Secondary', 'Missing')]
    for label1, label2 in label_pairs:
        vocab1 = scorer.label_vocabularies[label1]
        vocab2 = scorer.label_vocabularies[label2]
        overlap = len(vocab1 & vocab2)
        union = len(vocab1 | vocab2)
        jaccard = overlap / union if union > 0 else 0
        
        insights["label_overlaps"][f"{label1}_vs_{label2}"] = {
            "overlap": overlap,
            "jaccard_similarity": jaccard
        }
    
    return insights