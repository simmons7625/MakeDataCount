"""
PDFとXMLからテキストを抽出・前処理するためのユーティリティ関数
"""

import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from lxml import etree
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 必要なNLTKデータをダウンロード
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


def extract_text_from_pdf(pdf_path: Path) -> str:
    """PDFファイルからテキストを抽出"""
    try:
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


def extract_text_from_xml(xml_path: Path) -> Dict[str, str]:
    """XMLファイルから構造化テキストを抽出"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # TEI XML構造を解析
        extracted_data = {
            'title': '',
            'authors': [],
            'abstract': '',
            'keywords': [],
            'full_text': '',
            'references': []
        }
        
        # 名前空間を考慮したパス
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        # タイトル抽出
        title_elem = root.find('.//tei:titleStmt/tei:title[@level="a"]', ns)
        if title_elem is not None and title_elem.text:
            extracted_data['title'] = title_elem.text.strip()
        
        # 抽象抽出
        abstract_elem = root.find('.//tei:abstract', ns)
        if abstract_elem is not None:
            abstract_text = ' '.join(abstract_elem.itertext()).strip()
            extracted_data['abstract'] = abstract_text
        
        # 本文抽出
        body_elem = root.find('.//tei:body', ns)
        if body_elem is not None:
            body_text = ' '.join(body_elem.itertext()).strip()
            extracted_data['full_text'] = body_text
        
        # 参考文献抽出
        biblio_elems = root.findall('.//tei:biblStruct', ns)
        for biblio in biblio_elems:
            ref_text = ' '.join(biblio.itertext()).strip()
            if ref_text:
                extracted_data['references'].append(ref_text)
        
        return extracted_data
    
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return {'title': '', 'authors': [], 'abstract': '', 'keywords': [], 'full_text': '', 'references': []}


def preprocess_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> List[str]:
    """テキストの前処理"""
    if not text:
        return []
    
    # 小文字化
    text = text.lower()
    
    # 特殊文字・数字を除去（DOI等は保持）
    text = re.sub(r'[^\w\s\.\-/]', ' ', text)
    
    # 複数の空白を単一に
    text = re.sub(r'\s+', ' ', text)
    
    # トークン化
    tokens = word_tokenize(text)
    
    # 短すぎる単語を除去
    tokens = [token for token in tokens if len(token) > 2]
    
    # ストップワード除去
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # レンマ化
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


def clean_text(text: str) -> str:
    """テキストの基本的なクリーニング"""
    if not text:
        return ""
    
    # 改行を空白に
    text = text.replace('\n', ' ')
    
    # 複数の空白を単一に
    text = re.sub(r'\s+', ' ', text)
    
    # 前後の空白を削除
    text = text.strip()
    
    return text


def split_into_sentences(text: str) -> List[str]:
    """テキストを文単位で分割"""
    if not text:
        return []
    
    sentences = sent_tokenize(text)
    return [sent.strip() for sent in sentences if sent.strip()]


def get_context_around_match(text: str, match_start: int, match_end: int, context_size: int = 100) -> str:
    """マッチした部分の周辺コンテキストを取得"""
    start_pos = max(0, match_start - context_size)
    end_pos = min(len(text), match_end + context_size)
    
    context = text[start_pos:end_pos]
    
    # 前後に省略記号を追加
    if start_pos > 0:
        context = "..." + context
    if end_pos < len(text):
        context = context + "..."
    
    return context.strip()