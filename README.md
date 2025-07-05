# GraphRAG with LlamaIndex and Anthropic

このプロジェクトは、`ms-graphrag` の主要な機能を LlamaIndex と Anthropic Claude を使用して再現し、拡張することを目的としています。

## 機能一覧

*   **ドキュメント処理:**
    *   多様なファイル形式（TXT, CSV, PDF, DOCX, PPTX, HTML, EMLなど）からのテキスト抽出をサポート。
    *   `unstructured` ライブラリによる高度なファイル解析。
    *   CSVファイルは1行ごとに個別のノードとして処理。
*   **重複排除:** ファイル内容のハッシュ値に基づき、一度処理したファイルの重複追加を防止。
*   **テキストチャンキング:** ドキュメントをチャンクに分割し、ベクトル検索の粒度を調整（チャンクサイズとオーバーラップを設定可能）。
*   **エンティティ・リレーションシップ抽出:** LLM（Anthropic Claude）を使用してテキストチャンクからエンティティ（人、組織、場所、概念など）とその関連性を抽出。
*   **データ永続化:** 抽出されたエンティティ、リレーションシップ、処理済みファイル情報、コミュニティ、コミュニティ要約をParquet形式で保存。
*   **コミュニティ検出:** 抽出されたリレーションシップからグラフを構築し、Leiden法を用いてコミュニティを検出。
*   **コミュニティ要約:** 検出されたコミュニティの内容をLLMで要約し、キーエンティティを抽出。
*   **ベクトル検索:**
    *   メインテキストチャンク、抽出されたエンティティ、コミュニティ要約のそれぞれに対してベクトルインデックスを作成。
    *   LanceDB をベクトルストアとして利用（設定で切り替え可能）。
    *   HuggingFace の `intfloat/multilingual-e5-small` を埋め込みモデルとして使用。
*   **CLIインターフェース:** ドキュメントの追加 (`add`) と検索 (`search`) をコマンドラインから実行可能。
*   **柔軟な設定:** YAMLファイルを通じて、APIキー、モデル名、入出力ディレクトリ、チャンキング設定、コミュニティ検出パラメータなどを詳細に設定可能。

## プロジェクト構造

```
graphrag-anthropic-llamaindex/
├── config.example.yaml       # 設定ファイルのテンプレート
├── pyproject.toml            # Poetry のプロジェクト設定と依存関係
├── README.md                 # このファイル
├── data/                     # 処理対象のドキュメントを配置するディレクトリ (input_dir)
│   ├── sample_document.txt
│   ├── sample_data.csv
│   └── ...
├── graphrag_output/          # 処理結果が保存されるルートディレクトリ (output_dir)
│   ├── processed_files.parquet
│   ├── entities.parquet
│   ├── relationships.parquet
│   ├── communities.parquet
│   ├── community_summaries.parquet
│   ├── lancedb_main/         # メインテキストチャンクのLanceDBベクトルストア
│   ├── lancedb_entities/     # エンティティのLanceDBベクトルストア
│   ├── lancedb_communities/  # コミュニティ要約のLanceDBベクトルストア
│   └── storage/              # デフォルトのLlamaIndexストレージ（LanceDBを使用しない場合）
├── scripts/
│   └── demo.sh               # 機能紹介デモスクリプト
└── src/
    └── graphrag_anthropic_llamaindex/
        ├── __init__.py
        ├── main.py           # CLIエントリポイント
        ├── config_manager.py # 設定ファイルの読み込み
        ├── db_manager.py     # Parquetデータベースの読み書き
        ├── vector_store_manager.py # ベクトルストアとインデックスの管理
        ├── graph_operations.py # グラフ操作とコミュニティ検出
        ├── llm_utils.py      # LLMからのJSON出力パースとプロンプトテンプレート
        ├── document_processor.py # ドキュメントの追加処理
        └── search_processor.py # 検索処理
```

## インストール

1.  **リポジトリのクローン:**
    ```bash
    git clone <リポジトリのURL>
    cd graphrag-anthropic-llamaindex
    ```

2.  **Poetry のインストール (未導入の場合):**
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3.  **依存関係のインストール:**
    ```bash
    poetry install
    ```

## 設定

`config.example.yaml` を `config.yaml` としてコピーし、必要な設定を更新してください。

```bash
cp config.example.yaml config.yaml
```

`config.yaml` の主要な設定項目:

*   `anthropic.api_key`: Anthropic APIキーを設定してください。
*   `anthropic.model`: 使用するClaudeモデル名（例: `claude-3-opus-20240229`）。
*   `embedding_model.name`: 埋め込みモデル名（デフォルト: `intfloat/multilingual-e5-small`）。
*   `input_dir`: 処理対象のドキュメントが置かれるディレクトリ（デフォルト: `./data`）。
*   `output_dir`: 処理結果（Parquetファイル、LanceDBデータ、LlamaIndexストレージ）が保存されるルートディレクトリ（デフォルト: `./graphrag_output`）。
*   `vector_store`, `entity_vector_store`, `community_vector_store`: LanceDB の設定。`uri` は `output_dir` からの相対パスで指定します（例: `lancedb_main` は `./graphrag_output/lancedb_main` に保存されます）。`table_name` はテーブル名です。

## 使用方法

### 1. ドキュメントの追加と処理

`input_dir` で指定されたディレクトリ（デフォルト: `data/`）に処理したいドキュメント（TXT, CSV, PDF, DOCX, PPTX, HTML, EMLなど）を配置します。

```bash
poetry run python src/graphrag_anthropic_llamaindex/main.py --config config.yaml add
```

*   このコマンドは、`input_dir` 内の新しいドキュメントを読み込み、テキスト抽出、チャンキング、エンティティ・リレーションシップ抽出、コミュニティ検出、コミュニティ要約を行います。
*   処理済みファイル情報、抽出されたデータは `output_dir` で指定されたディレクトリにParquetファイルとして保存されます。
*   メインテキスト、エンティティ、コミュニティ要約のベクトルインデックスが更新されます。
*   初回実行時や新しいドキュメントが追加された場合、LLMの呼び出しや埋め込みモデルのダウンロードにより時間がかかることがあります。

### 2. 検索

以下のコマンドで、異なるインデックスを対象に検索を実行できます。

```bash
bash
poetry run python src/graphrag_anthropic_llamaindex/main.py --config config.yaml search "あなたのクエリ" --target-index <インデックスタイプ>
```

`<インデックスタイプ>` には以下のいずれかを指定します。

*   `main`: メインテキストチャンクのインデックスを検索します。
*   `entity`: 抽出されたエンティティのインデックスを検索します。
*   `community`: コミュニティ要約のインデックスを検索します。
*   `both` (デフォルト): `main`, `entity`, `community` のすべてのインデックスを検索します。

**例:**

*   **メインテキストインデックスの検索:**
    ```bash
    poetry run python src/graphrag_anthropic_llamaindex/main.py --config config.yaml search "AIの最新動向について教えてください"
    ```
*   **エンティティインデックスの検索:**
    ```bash
    poetry run python src/graphrag_anthropic_llamaindex/main.py --config config.yaml search "Acme CorpのCEOは誰ですか？" --target-index entity
    ```
*   **コミュニティ要約インデックスの検索:**
    ```bash
    poetry run python src/graphrag_anthropic_llamaindex/main.py --config config.yaml search "サステナブルテクノロジーに関するコミュニティはありますか？" --target-index community
    ```
*   **すべてのインデックスを検索:**
    ```bash
    poetry run python src/graphrag_anthropic_llamaindex/main.py --config config.yaml search "気候変動に関する主要な研究機関は？"
    ```

## デモスクリプトの実行

`scripts/demo.sh` を実行すると、一連のドキュメント追加と検索のデモンストレーションを自動で行います。

```bash
./scripts/demo.sh
```

**注意:** 初回実行時は、依存関係のインストール、LLMの呼び出し、埋め込みモデルのダウンロードなどにより、時間がかかる場合があります。また、Anthropic APIキーが正しく設定されていることを確認してください。
