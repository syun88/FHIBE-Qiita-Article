# FHIBE Qiita Article Series

FHIBE (Fairness in Human Imaging Benchmark Evaluation) を題材にした **Qiita Advent Calendar 2025** 連載用リポジトリです。  
Day 1〜25 までの執筆ネタを整理しつつ、データセット分析コードや記事下書きを管理します。

## 連載テーマ（抜粋）
- Day 1: FHIBE データセットとは？　記事未公開
- Day 2: データ構造とメタデータの完全解剖　記事未公開
- Day 3: 利用規約・倫理実務　記事未公開
- …
- Day 25: 総括・コード公開　記事未公開


## リポジトリ構成

```
FHIBE-Qiita-Article/
├── README.md                # このファイル
├── day2/
│   └── main.py              # Day 2 記事用レポート生成スクリプト
└── day3/
    └── main.py              # Day 3 以降の下書き（未実装）
```

## 必要環境
- Python 3.10 以降（標準ライブラリのみで動作）
- FHIBE データセット展開済みディレクトリ  
  例: `/Volumes/data/SONY_AI/fhibe.20250716.u.gT5_rFTA_downsampled_public`
- 依存パッケージ: [rich](https://github.com/Textualize/rich), [tabulate](https://github.com/astanin/python-tabulate)  
  インストール例: `python3 -m pip install -r requirements.txt`

## Day2 スクリプトの使い方

Day 2 のテーマ「データ構造完全解剖」を支援するため、`day2/main.py` が FHIBE データセットの構造やメタデータ統計を Markdown 形式で出力します。

```bash
# 依存パッケージのインストール（初回のみ）
python3 -m pip install -r requirements.txt

# 直接パスを渡す場合
python3 day2/main.py \
  --dataset-root /Volumes/data/SONY_AI/fhibe.20250716.u.gT5_rFTA_downsampled_public \
  --max-samples 3 \
  --top-n 5

# 環境変数を使う場合
export FHIBE_DATA_DIR=/Volumes/data/SONY_AI/fhibe.20250716.u.gT5_rFTA_downsampled_public
python3 day2/main.py

# Markdownテーブルを併せて表示（Qiita貼り付け用）
python3 day2/main.py --markdown
```

主な出力内容:
1. ディレクトリ一覧（raw / processed / results 等）
2. `fhibe_downsampled.csv` に基づく統計（件数、解像度、年齢帯、カメラ機種、Pronoun・Ancestry 等）
3. `annotator_demographics.csv` と `QAannotator_demographics.csv` の属性統計

`rich` によるカラー付きテーブルと `tabulate` による GitHub Flavored Markdown (オプション `--markdown`) を併用することで、ターミナル表示と Qiita 貼り付けの両方が整った形で得られます。

### Day2 CLI 引数
- `--dataset-root`: FHIBE データセットを展開したルートディレクトリ。未指定時は `FHIBE_DATA_DIR` を参照します。
- `--focus-dirs`: ディレクトリ概要に載せたい相対パスのリスト。指定がなければ `data/processed` などが含まれる既定リストを使用。
- `--max-samples`: 各ディレクトリで表示するサンプルファイル／フォルダの件数。
- `--top-n`: カメラ機種や属性ラベルで表示する上位頻度の件数。

## 今後の予定
- Day 3 以降のトピックに応じたスクリプトやノートブックを順次追加
- 公平性評価パイプライン構築メモ
- 記事テンプレートの整備

質問や改善案があれば issue / PR で連絡してください。
