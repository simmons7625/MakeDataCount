# Make Data Count: データ参照の発見

Kaggleコンペティション: https://www.kaggle.com/competitions/make-data-count-finding-data-references

## コンペティション概要
このコンペティションでは、論文内での科学データの使用を識別し、それらがどのように言及されているかを分類することが求められます。目標は以下を行う機械学習モデルの開発です：
- 科学文献内でのデータセットの言及を識別
- 論文とデータ参照の関係をコンテキスト化

## セットアップ手順
1. 依存関係のインストール: `pip install -r requirements.txt`
2. Kaggle API認証情報を `~/.kaggle/kaggle.json` に設置
3. データのダウンロード: `kaggle competitions download -c make-data-count-finding-data-references`

## プロジェクト構造
```
├── data/           # コンペティションデータ
├── notebooks/      # 分析・モデリングノートブック
├── src/           # ソースコード
└── submissions/   # 提出ファイル
```

## データ概要
- `train/` - 学習用論文データ (1.8GB)
- `test/` - テスト用論文データ
- `train_labels.csv` - 正解ラベル
- `sample_submission.csv` - 提出形式サンプル

## 使用技術
- Python 3.8+
- pandas, numpy, scikit-learn
- transformers, torch (自然言語処理)
- spacy, nltk (テキスト前処理)
- BeautifulSoup4, lxml (XML/HTML解析)