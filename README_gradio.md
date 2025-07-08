# GraphRAG Anthropic LlamaIndex - Gradio Web App

GradioベースのシンプルなWebアプリケーションで、GraphRAG Anthropic LlamaIndex CLIツールを操作できます。

## 🚀 特徴

- **シンプルなUI**: Gradioによる直感的なインターフェース
- **設定管理**: config.yamlファイルの読み込み
- **ドキュメント処理**: ファイルをインデックスに追加
- **検索機能**: 追加されたドキュメントからの検索
- **進行状況表示**: 処理の進行状況をリアルタイム表示
- **日本語対応**: 完全日本語インターフェース

## 📦 インストール

Gradioは既にインストール済みです：

```bash
# 確認
pdm list | grep gradio
```

## 🎯 使用方法

### 1. アプリケーションの起動

```bash
python gradio_app.py
```

ブラウザで `http://localhost:7860` にアクセスしてください。

### 2. 基本的な使用手順

1. **設定タブ**
   - 設定ファイルのパス（デフォルト: `config.yaml`）を入力
   - 「設定を読み込み」ボタンをクリック

2. **ドキュメント追加タブ**
   - 入力ディレクトリ（ドキュメントの場所）を指定
   - 出力ディレクトリ（処理結果の保存先）を指定
   - 「ドキュメントを追加」ボタンをクリック

3. **検索タブ**
   - 検索クエリを入力
   - 対象インデックス（main/entity/community/both）を選択
   - 「検索実行」ボタンをクリック

## ⚙️ 設定ファイル例

```yaml
anthropic:
  api_key: "your-anthropic-api-key"
  model: "claude-3-opus-20240229"

input_dir: "./data"
output_dir: "./graphrag_output"

embedding_model:
  name: "intfloat/multilingual-e5-small"

chunking:
  chunk_size: 1024
  chunk_overlap: 20

ignore_patterns:
  - "*.tmp"
  - ".git/*"
  - "__pycache__/*"
```

## 📁 ディレクトリ構造

```
project/
├── config.yaml           # 設定ファイル
├── gradio_app.py         # Gradioアプリケーション
├── data/                 # 入力ドキュメント
├── graphrag_output/      # 処理結果・インデックス
└── src/                  # GraphRAGライブラリ
```

## 🛠️ 開発者向け

### カスタマイズ

```python
# ポート変更
interface.launch(server_port=8080)

# 外部アクセス許可
interface.launch(share=True)

# 認証追加
interface.launch(auth=("username", "password"))
```

### ログレベル設定

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔧 トラブルシューティング

### よくある問題

1. **設定読み込みエラー**
   - config.yamlファイルの存在確認
   - YAML形式の正確性確認
   - Anthropic APIキーの設定確認

2. **ドキュメント追加エラー**
   - 入力ディレクトリの存在確認
   - ファイルアクセス権限確認
   - ディスク容量確認

3. **検索エラー**
   - インデックスファイルの存在確認
   - 出力ディレクトリの確認
   - APIキーの有効性確認

### デバッグ方法

```bash
# ログ出力付きで実行
python gradio_app.py 2>&1 | tee app.log
```

## 📋 CLIとの比較

| 機能 | CLI | Gradio Web App |
|------|-----|----------------|
| 設定管理 | コマンドライン引数 | Webフォーム |
| ドキュメント追加 | `add`コマンド | ボタンクリック |
| 検索 | `search`コマンド | フォーム入力 |
| 進行状況 | ターミナル出力 | プログレスバー |
| 結果表示 | テキスト出力 | 整形されたWebUI |

Gradio版は技術的な知識がなくても簡単に使用できる一方、CLI版はスクリプト化や自動化に適しています。