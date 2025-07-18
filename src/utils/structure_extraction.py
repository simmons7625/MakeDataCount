"""
論文構造の抽出・構造化データ生成のためのユーティリティ関数
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from .text_extraction import clean_text, split_into_sentences, get_context_around_match


# セクションヘッダーのパターン
SECTION_PATTERNS = {
    'abstract': r'(?i)^(abstract|summary)\s*$',
    'introduction': r'(?i)^(\d+\.?\s*)?introduction\s*$',
    'methods': r'(?i)^(\d+\.?\s*)?(methods?|methodology|experimental|materials?\s+and\s+methods?)\s*$',
    'results': r'(?i)^(\d+\.?\s*)?(results?|findings)\s*$',
    'discussion': r'(?i)^(\d+\.?\s*)?(discussion|analysis)\s*$',
    'conclusion': r'(?i)^(\d+\.?\s*)?(conclusion|conclusions?)\s*$',
    'references': r'(?i)^(\d+\.?\s*)?(references?|bibliography)\s*$'
}

# データセット言及のパターン
DATASET_PATTERNS = [
    r'(?i)data(?:set|base)?\s+(?:from|at|available|deposited|stored)',
    r'(?i)repository\s+(?:at|URL|link|available)',
    r'(?i)doi\.org/[\w\./\-]+',
    r'(?i)https?://[\w\./\-]+',
    r'(?i)supplementary\s+(?:data|material|information)',
    r'(?i)archive(?:d|s)?\s+(?:at|in)',
    r'(?i)publicly\s+available',
    r'(?i)data\s+(?:is|are)\s+available',
    r'(?i)github\.com/[\w\./\-]+',
    r'(?i)figshare\.com/[\w\./\-]+',
    r'(?i)zenodo\.org/[\w\./\-]+',
    r'(?i)dryad\.org/[\w\./\-]+',
    r'(?i)ncbi\.nlm\.nih\.gov/[\w\./\-]+',
    r'(?i)genbank',
    r'(?i)accession\s+number',
    r'(?i)bioproject'
]


def extract_paper_structure(text: str) -> Dict:
    """PDFテキストから論文構造を抽出してJSONライクな構造で返す"""
    if not text:
        return {}
    
    # テキストを行単位で分割
    lines = text.split('\n')
    
    structure = {
        "doi": "",
        "metadata": {
            "title": "",
            "authors": [],
            "abstract": "",
            "keywords": [],
            "publication_info": {
                "journal": "",
                "year": "",
                "volume": "",
                "issue": "",
                "pages": ""
            }
        },
        "content": {
            "sections": {
                "introduction": "",
                "methods": "",
                "results": "",
                "discussion": "",
                "conclusion": ""
            },
            "full_text": text
        },
        "references": [],
        "datasets": [],
        "urls": [],
        "quality_score": 0
    }
    
    # セクションを特定
    sections = identify_sections(lines)
    
    # 各セクションのテキストを抽出
    for section_name, (start_line, end_line) in sections.items():
        if start_line is not None and end_line is not None:
            section_text = '\n'.join(lines[start_line:end_line])
            if section_name in structure["content"]["sections"]:
                structure["content"]["sections"][section_name] = clean_text(section_text)
    
    # メタデータを抽出
    structure["metadata"]["title"] = extract_title(text)
    structure["metadata"]["authors"] = extract_authors(text)
    structure["metadata"]["abstract"] = extract_abstract(text)
    structure["metadata"]["keywords"] = extract_keywords(text)
    
    # データセット言及を抽出
    structure["datasets"] = extract_dataset_mentions(text)
    
    # DOI参照を抽出
    structure["references"] = extract_doi_references(text)
    
    # URLを抽出
    structure["urls"] = extract_urls(text)
    
    # 品質スコアを計算
    structure["quality_score"] = validate_structure(structure)
    
    return structure


def identify_sections(lines: List[str]) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    """テキスト行からセクションを特定"""
    sections = {}
    current_section = None
    section_starts = {}
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # セクションヘッダーかチェック
        for section_name, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, line_clean):
                if current_section is not None:
                    sections[current_section] = (section_starts[current_section], i)
                current_section = section_name
                section_starts[section_name] = i + 1
                break
    
    # 最後のセクションを処理
    if current_section is not None:
        sections[current_section] = (section_starts[current_section], len(lines))
    
    return sections


def extract_title(text: str) -> str:
    """論文タイトルを抽出"""
    lines = text.split('\n')[:20]  # 最初の20行から探索
    
    for line in lines:
        line = line.strip()
        if len(line) > 20 and len(line) < 200:  # 適切な長さ
            # 大文字が多い、または各単語が大文字から始まる
            if line.isupper() or re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', line):
                return line
    
    return ""


def extract_authors(text: str) -> List[str]:
    """著者名を抽出してリストで返す"""
    lines = text.split('\n')[:50]  # 最初の50行から探索
    
    author_patterns = [
        r'(?i)^author[s]?[:\s]',
        r'(?i)^by\s+',
        r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*',
        r'^[A-Z]\.\s*[A-Z][a-z]+(?:\s*,\s*[A-Z]\.\s*[A-Z][a-z]+)*'
    ]
    
    for line in lines:
        line = line.strip()
        for pattern in author_patterns:
            if re.match(pattern, line):
                # カンマで分割して著者リストを作成
                authors = [author.strip() for author in line.split(',')]
                return authors
    
    return []


def extract_abstract(text: str) -> str:
    """アブストラクトを抽出"""
    lines = text.split('\n')
    
    abstract_started = False
    abstract_lines = []
    
    for line in lines:
        line = line.strip()
        
        # アブストラクトの開始を検出
        if re.match(r'(?i)^(abstract|summary)\s*$', line):
            abstract_started = True
            continue
        
        # アブストラクトが開始されている場合
        if abstract_started:
            # 次のセクションの開始を検出したら終了
            if re.match(r'(?i)^(\d+\.?\s*)?(introduction|keywords?|background)', line):
                break
            
            if line:
                abstract_lines.append(line)
    
    return ' '.join(abstract_lines)


def extract_keywords(text: str) -> List[str]:
    """キーワードを抽出"""
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if re.match(r'(?i)^keywords?[:\s]', line):
            # "Keywords:" を除去
            keywords_text = re.sub(r'(?i)^keywords?[:\s]*', '', line)
            # セミコロンまたはカンマで分割
            keywords = [kw.strip() for kw in re.split(r'[;,]', keywords_text) if kw.strip()]
            return keywords
    
    return []


def extract_dataset_mentions(text: str) -> List[Dict]:
    """データセット言及を抽出"""
    mentions = []
    
    for pattern in DATASET_PATTERNS:
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            context = get_context_around_match(text, match.start(), match.end(), 150)
            mentions.append({
                "text": match.group(),
                "context": context,
                "position": match.start(),
                "pattern": pattern,
                "type": "dataset_mention"
            })
    
    return mentions


def extract_doi_references(text: str) -> List[str]:
    """DOI参照を抽出"""
    doi_pattern = r'(?i)(?:doi:?\s*)?10\.\d{4,}/[^\s\]]+[^\s\].,]'
    matches = re.findall(doi_pattern, text)
    
    # 重複を除去し、クリーニング
    unique_dois = []
    for doi in matches:
        clean_doi = doi.strip().lower()
        if clean_doi.startswith('doi:'):
            clean_doi = clean_doi[4:].strip()
        if clean_doi not in unique_dois:
            unique_dois.append(clean_doi)
    
    return unique_dois


def extract_urls(text: str) -> List[str]:
    """URLを抽出"""
    url_pattern = r'https?://[^\s\]]+[^\s\].,]'
    matches = re.findall(url_pattern, text)
    
    # 重複を除去
    unique_urls = list(set(matches))
    
    return unique_urls


def create_structured_json(doi: str, structure: Dict) -> Dict:
    """構造化データをJSON形式で生成"""
    
    json_structure = {
        "doi": doi,
        "metadata": {
            "title": structure.get("metadata", {}).get("title", ""),
            "authors": structure.get("metadata", {}).get("authors", []),
            "abstract": structure.get("metadata", {}).get("abstract", ""),
            "keywords": structure.get("metadata", {}).get("keywords", []),
            "publication_info": structure.get("metadata", {}).get("publication_info", {})
        },
        "content": {
            "sections": structure.get("content", {}).get("sections", {}),
            "chunks": hierarchical_text_split(structure.get("content", {}).get("full_text", ""))
        },
        "references": structure.get("references", []),
        "datasets": structure.get("datasets", []),
        "urls": structure.get("urls", []),
        "quality_score": structure.get("quality_score", 0),
        "extraction_metadata": {
            "total_length": len(structure.get("content", {}).get("full_text", "")),
            "num_sections": len([s for s in structure.get("content", {}).get("sections", {}).values() if s]),
            "num_chunks": len(structure.get("content", {}).get("chunks", [])),
            "has_abstract": bool(structure.get("metadata", {}).get("abstract", "")),
            "has_methods": bool(structure.get("content", {}).get("sections", {}).get("methods", "")),
            "has_results": bool(structure.get("content", {}).get("sections", {}).get("results", ""))
        }
    }
    
    return json_structure


def hierarchical_text_split(text: str, max_chunk_size: int = 512) -> List[Dict]:
    """階層的にテキストを分割"""
    chunks = []
    
    # 1. 論文構造を抽出
    structure = extract_paper_structure(text)
    
    # 2. セクション単位で分割
    sections = ['introduction', 'methods', 'results', 'discussion', 'conclusion']
    
    for section_name in sections:
        section_text = structure.get("content", {}).get("sections", {}).get(section_name, "")
        if not section_text:
            continue
        
        if len(section_text) <= max_chunk_size:
            chunks.append({
                "id": f"{section_name}_full",
                "type": "section",
                "section": section_name,
                "text": section_text,
                "size": len(section_text),
                "metadata": {
                    "chunk_index": len(chunks),
                    "section_name": section_name
                }
            })
        else:
            # 3. 段落単位で分割
            paragraphs = section_text.split('\n\n')
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if not para:
                    continue
                
                if len(para) <= max_chunk_size:
                    chunks.append({
                        "id": f"{section_name}_para_{i}",
                        "type": "paragraph",
                        "section": section_name,
                        "text": para,
                        "size": len(para),
                        "metadata": {
                            "chunk_index": len(chunks),
                            "section_name": section_name,
                            "paragraph_index": i
                        }
                    })
                else:
                    # 4. 文単位で分割
                    sentences = split_into_sentences(para)
                    current_chunk = ""
                    sentence_indices = []
                    
                    for j, sent in enumerate(sentences):
                        if len(current_chunk + sent) <= max_chunk_size:
                            current_chunk += sent + " "
                            sentence_indices.append(j)
                        else:
                            if current_chunk:
                                chunks.append({
                                    "id": f"{section_name}_para_{i}_sent_{min(sentence_indices)}-{max(sentence_indices)}",
                                    "type": "sentence_group",
                                    "section": section_name,
                                    "text": current_chunk.strip(),
                                    "size": len(current_chunk.strip()),
                                    "metadata": {
                                        "chunk_index": len(chunks),
                                        "section_name": section_name,
                                        "paragraph_index": i,
                                        "sentence_indices": sentence_indices
                                    }
                                })
                            current_chunk = sent + " "
                            sentence_indices = [j]
                    
                    if current_chunk:
                        chunks.append({
                            "id": f"{section_name}_para_{i}_sent_{min(sentence_indices)}-{max(sentence_indices)}",
                            "type": "sentence_group",
                            "section": section_name,
                            "text": current_chunk.strip(),
                            "size": len(current_chunk.strip()),
                            "metadata": {
                                "chunk_index": len(chunks),
                                "section_name": section_name,
                                "paragraph_index": i,
                                "sentence_indices": sentence_indices
                            }
                        })
    
    return chunks


def validate_structure(structure: Dict) -> int:
    """抽出された構造の品質をチェック"""
    quality_score = 0
    
    # メタデータの品質チェック
    metadata = structure.get("metadata", {})
    if metadata.get("title"): 
        quality_score += 20
    if metadata.get("abstract"): 
        quality_score += 30
    if metadata.get("authors"): 
        quality_score += 10
    
    # セクションの品質チェック
    sections = structure.get("content", {}).get("sections", {})
    if sections.get("methods"): 
        quality_score += 20
    if sections.get("results"): 
        quality_score += 20
    
    # 長さの妥当性チェック
    if metadata.get("title") and len(metadata["title"]) > 10:
        quality_score += 5
    if metadata.get("abstract") and len(metadata["abstract"]) > 100:
        quality_score += 10
    if structure.get("content", {}).get("full_text") and len(structure["content"]["full_text"]) > 1000:
        quality_score += 5
    
    return min(quality_score, 100)


def save_structured_data(structure: Dict, output_path: Path):
    """構造化データをJSONファイルに保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)


def load_structured_data(input_path: Path) -> Dict:
    """JSONファイルから構造化データを読み込み"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)