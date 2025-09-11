# MS GraphRAG 設定ファイル仕様書

## 概要

MS GraphRAGは、YAMLまたはJSON形式の設定ファイル（`settings.yml`または`settings.json`）を使用して、グラフベースのRAG（Retrieval Augmented Generation）システムを構成します。環境変数による動的な値の設定にも対応しており、`${ENV_VAR}`形式で参照できます。

## ファイル構造

### 基本構成

```yaml
# settings.yml
models:            # 言語モデル設定
input:             # データ入力設定
chunks:            # テキストチャンキング設定
output:            # 出力ストレージ設定
cache:             # キャッシュ設定
vector_store:      # ベクトルストア設定
reporting:         # レポート設定
workflows:         # ワークフロー設定
extract_graph:     # グラフ抽出設定
community_reports: # コミュニティレポート設定
embeddings:        # エンベディング設定
search:            # 検索設定
```

## 主要設定セクション

### 1. models（言語モデル設定）

言語モデルとエンベディングモデルの設定を定義します。複数のモデルを定義し、用途に応じて使い分けることができます。

```yaml
models:
  default_chat_model:    # チャット/生成用モデル
    type: ${GRAPHRAG_LLM_TYPE}  # openai_chat | azure_openai_chat | mock_chat
    api_key: ${GRAPHRAG_API_KEY}
    api_base: ${GRAPHRAG_API_BASE}  # Azure OpenAIの場合
    api_version: ${GRAPHRAG_API_VERSION}  # Azure OpenAIの場合
    deployment_name: ${GRAPHRAG_LLM_DEPLOYMENT_NAME}  # Azure OpenAIの場合
    model: ${GRAPHRAG_LLM_MODEL}  # gpt-4-turbo-preview等
    auth_type: api_key  # api_key | azure_managed_identity
    model_supports_json: true  # JSONモード対応
    tokens_per_minute: ${GRAPHRAG_LLM_TPM}  # レート制限
    requests_per_minute: ${GRAPHRAG_LLM_RPM}  # レート制限
    concurrent_requests: 50  # 並行リクエスト数
    async_mode: threaded  # threaded | asyncio
    temperature: 0.0  # 生成パラメータ
    top_p: 1.0
    max_tokens: 4000
    
  default_embedding_model:  # エンベディング用モデル
    type: ${GRAPHRAG_EMBEDDING_TYPE}  # openai_embedding | azure_openai_embedding
    api_key: ${GRAPHRAG_API_KEY}
    model: ${GRAPHRAG_EMBEDDING_MODEL}  # text-embedding-3-small等
    # その他パラメータは同上
```

#### 必須パラメータ
- `type`: モデルタイプ
- `model`: モデル名
- `api_key`: 認証キー（azure_managed_identity使用時は不要）

#### オプションパラメータ
- レート制限: `tokens_per_minute`, `requests_per_minute`
- エラーハンドリング: `max_retries`, `retry_strategy`
- 生成パラメータ: `temperature`, `top_p`, `max_tokens`
- 並行処理: `concurrent_requests`, `async_mode`

### 2. input（入力設定）

データソースと入力形式を設定します。

```yaml
input:
  type: file  # file | blob | cosmosdb
  file_type: text  # text | csv | json
  encoding: utf-8
  file_pattern: ".*\\.txt$"  # ファイル名パターン（正規表現）
  base_dir: input  # ファイル入力時のベースディレクトリ
  
  # CSVファイルの場合
  text_column: content  # テキストデータのカラム名
  title_column: title  # タイトルカラム名
  
  # Azure Blob Storage使用時
  storage:
    type: blob
    connection_string: ${BLOB_CONNECTION_STRING}
    container_name: mycontainer
    base_dir: input
```

### 3. chunks（チャンキング設定）

テキストを処理単位に分割する設定です。

```yaml
chunks:
  size: 1200  # 最大チャンクサイズ（トークン数）
  overlap: 100  # チャンク間のオーバーラップ（トークン数）
  strategy: tokens  # tokens | sentences
  encoding_model: cl100k_base  # トークンカウント用モデル
  group_by_columns: ["id"]  # ドキュメントグループ化のカラム
  prepend_metadata: false  # メタデータの先頭付加
  chunk_size_includes_metadata: false  # メタデータをサイズに含めるか
```

### 4. vector_store（ベクトルストア設定）

ベクトルデータベースの設定を定義します。複数のストアを定義可能です。

```yaml
vector_store:
  default_vector_store:
    type: lancedb  # lancedb | azure_ai_search
    
    # LanceDB使用時
    db_uri: ./lancedb  # データベースパス
    container_name: my_collection
    overwrite: true  # 既存データの上書き
    
    # Azure AI Search使用時
    # type: azure_ai_search
    # url: ${AZURE_AI_SEARCH_URL_ENDPOINT}
    # api_key: ${AZURE_AI_SEARCH_API_KEY}
    # container_name: my_index
```

### 5. output（出力設定）

処理結果の保存先を設定します。

```yaml
output:
  type: file  # file | memory | blob | cosmosdb
  base_dir: output  # ファイル出力時のベースディレクトリ
  
  # Azure Blob Storage使用時
  # type: blob
  # connection_string: ${BLOB_CONNECTION_STRING}
  # container_name: output-container
  # base_dir: output
```

### 6. cache（キャッシュ設定）

処理中のキャッシュ保存場所を設定します。

```yaml
cache:
  type: file  # file | memory | blob | none
  base_dir: cache
  
  # Azure Blob Storage使用時
  # type: blob
  # connection_string: ${BLOB_CONNECTION_STRING}
  # container_name: cache-container
  # base_dir: cache
```

### 7. reporting（レポート設定）

実行ログとレポートの出力設定です。

```yaml
reporting:
  type: file  # file | console | blob
  base_dir: reports
  
  # Azure Blob Storage使用時
  # type: blob
  # connection_string: ${BLOB_CONNECTION_STRING}
  # container_name: reports-container
  # base_dir: reports
```

### 8. extract_graph（グラフ抽出設定）

エンティティと関係の抽出に関する設定です。

```yaml
extract_graph:
  enabled: true
  prompt: prompts/entity_extraction.txt  # カスタムプロンプトファイル
  entity_types: ["person", "organization", "location", "event"]
  max_entities: 10  # 抽出するエンティティの最大数
  strategy: default  # 抽出戦略
  model_id: default_chat_model  # 使用するモデルID
```

### 9. community_reports（コミュニティレポート設定）

コミュニティ検出とレポート生成の設定です。

```yaml
community_reports:
  enabled: true
  prompt: prompts/community_report.txt  # カスタムプロンプトファイル
  max_length: 2000  # レポートの最大長
  max_input_length: 8000  # 入力の最大長
  model_id: default_chat_model  # 使用するモデルID
```

### 10. extract_claims（クレーム抽出設定）

主張やクレームの抽出設定です。

```yaml
extract_claims:
  enabled: false  # デフォルトは無効
  prompt: prompts/claim_extraction.txt
  max_claims: 10  # 抽出するクレームの最大数
  model_id: default_chat_model
```

### 11. snapshots（スナップショット設定）

処理結果のスナップショット保存設定です。

```yaml
snapshots:
  embeddings: true  # エンベディングのスナップショット保存
  entities: true  # エンティティのスナップショット保存
  relationships: true  # 関係のスナップショット保存
```

### 12. 検索設定

各種検索メソッドの設定です。

#### local_search（ローカル検索）

```yaml
local_search:
  text_unit_prop: 0.9  # テキストユニットの重み
  community_prop: 0.1  # コミュニティの重み
  top_k_entities: 10  # 取得するエンティティ数
  top_k_relationships: 10  # 取得する関係数
  max_data_tokens: 12000  # 最大データトークン数
  temperature: 0.0  # 生成温度
  top_p: 1.0
  max_tokens: 2000
  model_id: default_chat_model
```

#### global_search（グローバル検索）

```yaml
global_search:
  data_max_tokens: 12000  # データの最大トークン数
  reduce_max_tokens: 2000  # リデュース時の最大トークン数
  reduce_temperature: 0.0  # リデュース時の温度
  concurrency: 32  # 並行度
  model_id: default_chat_model
```

#### drift_search（DRIFT検索）

```yaml
drift_search:
  n_depth: 3  # 探索深度
  drift_k_followups: 20  # フォローアップ数
  primer_folds: 5  # プライマーフォールド数
  primer_llm_max_tokens: 12000  # プライマーLLMの最大トークン数
  local_search_text_unit_prop: 0.9  # ローカル検索のテキストユニット重み
  local_search_community_prop: 0.1  # ローカル検索のコミュニティ重み
  local_search_top_k_mapped_entities: 10
  local_search_top_k_relationships: 10
  local_search_max_data_tokens: 12000
  local_search_temperature: 0.0
  model_id: default_chat_model
  embedding_model_id: default_embedding_model
```

### 13. workflows（ワークフロー設定）

実行するワークフローのリストです。

```yaml
workflows:
  - create_base_text_units
  - create_base_extracted_entities
  - create_summarized_entities
  - create_base_entity_graph
  - create_final_entities
  - create_final_relationships
  - create_final_communities
  - create_final_community_reports
  - create_final_text_embeddings
  - create_final_entity_embeddings
```

## 環境変数

設定ファイルでは、`${ENV_VAR}`形式で環境変数を参照できます。主な環境変数：

- `GRAPHRAG_API_KEY`: APIキー
- `GRAPHRAG_LLM_TYPE`: LLMのタイプ（openai_chat, azure_openai_chat等）
- `GRAPHRAG_LLM_MODEL`: LLMモデル名
- `GRAPHRAG_LLM_DEPLOYMENT_NAME`: Azure OpenAIのデプロイメント名
- `GRAPHRAG_API_BASE`: APIベースURL
- `GRAPHRAG_API_VERSION`: APIバージョン
- `GRAPHRAG_LLM_TPM`: トークン/分のレート制限
- `GRAPHRAG_LLM_RPM`: リクエスト/分のレート制限
- `GRAPHRAG_EMBEDDING_TYPE`: エンベディングタイプ
- `GRAPHRAG_EMBEDDING_MODEL`: エンベディングモデル名
- `GRAPHRAG_EMBEDDING_DEPLOYMENT_NAME`: Azure OpenAIエンベディングのデプロイメント名

## デフォルト値

主要なデフォルト値：

- チャンクサイズ: 1200トークン
- チャンクオーバーラップ: 100トークン
- 並行リクエスト数: 50
- キャッシュタイプ: file
- 出力ディレクトリ: output
- キャッシュディレクトリ: cache
- レポートディレクトリ: reports
- エンコーディング: utf-8
- ファイルタイプ: text

## 初期化と実行

### 1. 初期化

```bash
graphrag init --root ./project_dir
```

これにより、`./project_dir`に以下のファイルが作成されます：
- `.env`: 環境変数定義ファイル
- `settings.yaml`: 設定ファイル

### 2. 設定の編集

必要に応じて`settings.yaml`を編集し、環境変数を`.env`ファイルに設定します。

### 3. インデックス作成

```bash
graphrag index --root ./project_dir
```

### 4. クエリ実行

```bash
# グローバル検索
graphrag query --root ./project_dir --method global --query "質問内容"

# ローカル検索
graphrag query --root ./project_dir --method local --query "質問内容"
```

## 設定ファイルの検証

設定ファイルは、Pydanticモデルによって検証されます。無効な設定がある場合、起動時にエラーが発生します。

## 注意事項

1. **必須設定**: `default_chat_model`と`default_embedding_model`は必須です
2. **パスの解決**: 相対パスは`root_dir`からの相対パスとして解決されます
3. **Azure OpenAI**: Azure OpenAI使用時は、`api_base`、`api_version`、`deployment_name`の設定が必要です
4. **認証**: Azure Managed Identity使用時は、`auth_type: azure_managed_identity`を設定し、`az login`でログインが必要です
5. **ベクトルストア**: LanceDB使用時は`db_uri`が必須、Azure AI Search使用時は`url`と`api_key`が必須です