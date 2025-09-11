# GraphRAG Anthropic-LlamaIndex 設定ファイル実装仕様書

## 概要

本プロジェクトは、Anthropic Claude APIまたはAWS Bedrockを使用してGraphRAGシステムを構築するためのLlamaIndexベースの実装です。設定ファイルはYAML形式で記述され、シンプルかつ柔軟な構成管理を提供します。

## アーキテクチャ

### 設定ファイル構造

```
graphrag-anthropic-llamaindex/
├── config/              # 設定ファイルディレクトリ
│   ├── config.yaml      # メイン設定ファイル（ユーザー作成）
│   ├── config.example.yaml  # 設定ファイルテンプレート
│   └── config_bedrock.yaml  # Bedrock用設定サンプル
└── src/graphrag_anthropic_llamaindex/
    └── config_manager.py  # 設定読み込みモジュール
```

### 設定管理の特徴

1. **シンプルな実装**: Pydanticやdataclassを使用せず、純粋なYAML辞書として管理
2. **環境変数サポート**: APIキーは環境変数から取得（セキュリティ重視）
3. **プロバイダー切り替え**: AnthropicとBedrockを簡単に切り替え可能
4. **デフォルト値**: 各設定項目にデフォルト値を提供
5. **バリデーション**: 実行時に必要な値の存在確認

## 設定ファイル形式

### 基本構造（config/config.yaml）

```yaml
# LLMプロバイダー選択
llm_provider: "anthropic"  # or "bedrock"

# Anthropic設定
anthropic:
  model: "claude-3-opus-20240620"
  # api_key は環境変数 ANTHROPIC_API_KEY から取得

# AWS Bedrock設定
bedrock:
  model: "anthropic.claude-3-sonnet-20240229-v1:0"
  region: "us-east-1"

# エンベディングモデル設定
embedding_model:
  name: "intfloat/multilingual-e5-small"

# チャンキング設定
chunking:
  chunk_size: 1024
  chunk_overlap: 20

# 入出力ディレクトリ
input_dir: "./data"
output_dir: "./graphrag_output"

# ファイル無視パターン
ignore_patterns:
  - "*.tmp"
  - "*.log"
  - ".git/*"

# ベクトルストア設定
# 全てのベクトルストアは単一のLanceDBデータベースに統合され、
# ストアタイプごとに異なるテーブル（ハードコーディング）を使用:
# - メイン: "vectors" テーブル
# - エンティティ: "entities_vectors" テーブル
# - コミュニティ: "community_vectors" テーブル
vector_store:
  type: "lancedb"
  lancedb:
    uri: "./graphrag_output/lancedb"  # 全ストア共通の統合データベース

# コミュニティ検出設定
community_detection:
  max_cluster_size: 10
  use_lcc: true
  seed: 42

# グローバル検索設定
global_search:
  max_context_tokens: 8000
  include_community_weight: true
  response_type: "multiple paragraphs"
  include_key_points: false
  min_community_rank: 0
  max_concurrent: 5
  shuffle_data: true
  random_state: 42

# エンティティ抽出設定
entity_extraction:
  enabled: true
  max_entities_per_chunk: 10
```

## 実装詳細

### 1. 設定読み込み（config_manager.py）

```python
import yaml
import os

def load_config(config_path="config/config.yaml"):
    """Loads the configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Please copy 'config/config.example.yaml' to 'config/config.yaml' and set your API key.")
        return None
```

#### 特徴
- **シンプルな実装**: `yaml.safe_load()`で辞書として読み込み
- **エラーハンドリング**: ファイルが見つからない場合に親切なメッセージ
- **返り値**: 辞書形式の設定データまたはNone

### 2. 設定の使用（main.py）

#### LLMプロバイダーの設定

```python
# プロバイダー選択
llm_provider = config.get("llm_provider", "anthropic")

if llm_provider == "bedrock":
    # AWS Bedrock設定
    bedrock_config = config.get("bedrock", {})
    model_name = bedrock_config.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
    region_name = bedrock_config.get("region", "us-east-1")
else:
    # Anthropic直接設定
    anthropic_config = config.get("anthropic", {})
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found")
        sys.exit(1)
```

#### デフォルト値の適用

```python
# 各設定項目でget()メソッドを使用してデフォルト値を提供
input_dir = config.get("input_dir", "./data")
output_dir = config.get("output_dir", "./graphrag_output")

chunking_config = config.get("chunking", {})
chunk_size = chunking_config.get("chunk_size", 1024)
chunk_overlap = chunking_config.get("chunk_overlap", 20)
```

### 3. ベクトルストア設定の読み込み

```python
def get_vector_store(config, store_type="main"):
    """設定からベクトルストアを取得"""
    if store_type == "main":
        store_config = config.get("vector_store", {})
    elif store_type == "community":
        store_config = config.get("community_vector_store", {})
    elif store_type == "entity":
        store_config = config.get("entity_vector_store", {})
    
    # LanceDB設定の取得
    if store_config.get("type") == "lancedb":
        lancedb_config = store_config.get("lancedb", {})
        uri = lancedb_config.get("uri")
        table_name = lancedb_config.get("table_name")
        # ベクトルストアの初期化...
```

## バリデーション

### 実行時バリデーション

1. **必須項目の確認**
   - APIキー（Anthropic使用時）
   - モデル名
   - 入出力ディレクトリ

2. **型チェック**
   - 数値型: chunk_size, chunk_overlap, max_cluster_size
   - ブール型: use_lcc, enabled, shuffle_data
   - 文字列型: model, uri, table_name
   - リスト型: ignore_patterns

3. **値の範囲チェック**
   - chunk_size > 0
   - chunk_overlap >= 0
   - max_cluster_size > 0
   - min_community_rank >= 0

### エラーハンドリング

```python
# APIキーの確認
if llm_provider == "anthropic":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found")
        sys.exit(1)

# 設定ファイルの存在確認
config = load_config(args.config)
if not config:
    return  # エラーメッセージは load_config() 内で表示
```

## 環境変数

### サポートされる環境変数

| 環境変数 | 説明 | 必須 |
|---------|------|------|
| `ANTHROPIC_API_KEY` | Anthropic API キー | Anthropic使用時は必須 |
| `AWS_ACCESS_KEY_ID` | AWS アクセスキー | Bedrock使用時（オプション） |
| `AWS_SECRET_ACCESS_KEY` | AWS シークレットキー | Bedrock使用時（オプション） |
| `AWS_SESSION_TOKEN` | AWS セッショントークン | オプション |

### 環境変数の読み込み

```python
from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv()

# 環境変数の取得
api_key = os.environ.get("ANTHROPIC_API_KEY")
```

## 設定の優先順位

1. **コマンドライン引数** （最優先）
   - `--config` で指定されたファイル
   - `--mode`, `--response-type` などの個別オプション

2. **設定ファイル**
   - config/config.yaml の値

3. **環境変数**
   - APIキーなどの機密情報

4. **デフォルト値** （最低優先）
   - コード内で定義されたデフォルト値

## 設定の拡張性

### 新しい設定項目の追加方法

1. **config/config.example.yaml に追加**
```yaml
new_feature:
  enabled: false
  parameter1: "default_value"
  parameter2: 100
```

2. **main.py で読み込み**
```python
new_feature_config = config.get("new_feature", {})
enabled = new_feature_config.get("enabled", False)
param1 = new_feature_config.get("parameter1", "default_value")
```

3. **バリデーション追加**
```python
if enabled and not param1:
    raise ValueError("new_feature.parameter1 is required when enabled")
```

## セキュリティ考慮事項

1. **APIキーの管理**
   - 設定ファイルにAPIキーを直接記載しない
   - 環境変数または.envファイルを使用
   - .gitignoreで.envとconfig/config.yamlを除外

2. **ファイルパスの検証**
   - 相対パスを使用
   - パストラバーサル攻撃の防止

3. **設定ファイルの権限**
   - config/config.yamlは適切な権限（600）で保護
   - 本番環境では環境変数を推奨

## ベストプラクティス

1. **設定ファイルの管理**
   - config/config.example.yamlを常に最新に保つ
   - 本番用と開発用で別の設定ファイルを使用
   - 環境ごとの設定を分離

2. **デフォルト値の設計**
   - 合理的なデフォルト値を提供
   - 必須項目は明確にエラー表示
   - 型の一貫性を保つ

3. **ドキュメント**
   - 各設定項目にコメントを付与
   - 設定例を提供
   - 変更履歴を記録

## トラブルシューティング

### よくある問題と解決方法

1. **設定ファイルが見つからない**
   - config/config.example.yamlをconfig/config.yamlにコピー
   - --configオプションでパスを指定

2. **APIキーエラー**
   - 環境変数ANTHROPIC_API_KEYを設定
   - .envファイルを作成して記載

3. **型エラー**
   - YAMLの構文を確認
   - インデントが正しいか確認
   - 文字列は引用符で囲む

## 今後の改善案

1. **設定の検証強化**
   - Pydanticを使用した型安全な設定管理
   - JSONSchemaによる設定ファイル検証

2. **設定の階層化**
   - 基本設定と詳細設定の分離
   - プロファイル機能の追加

3. **動的設定更新**
   - 実行中の設定変更対応
   - ホットリロード機能

4. **設定のマイグレーション**
   - バージョン管理
   - 自動マイグレーション機能