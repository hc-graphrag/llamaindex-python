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
*   **データ永続化:** SQLデータとベクトルインデックスを、同じ組み込みAlopexDBへ保存。外部DBサーバーは不要。
*   **コミュニティ検出:** 抽出されたリレーションシップからグラフを構築し、Leiden法を用いてコミュニティを検出。
*   **コミュニティ要約:** 検出されたコミュニティの内容をLLMで要約し、キーエンティティを抽出。
*   **ベクトル検索:**
    *   メインテキストチャンク、抽出されたエンティティ、コミュニティ要約のそれぞれに対してベクトルインデックスを作成。
    *   AlopexDBの名前付きHNSWインデックスを利用。
    *   HuggingFace の `intfloat/multilingual-e5-small` を埋め込みモデルとして使用。
*   **CLIインターフェース:** ドキュメントの追加 (`add`) と検索 (`search`) をコマンドラインから実行可能。
*   **柔軟な設定:** YAMLファイルを通じて、APIキー、モデル名、入出力ディレクトリ、チャンキング設定、コミュニティ検出パラメータなどを詳細に設定可能。

## プロジェクト構造

```
graphrag-anthropic-llamaindex/
├── config/                   # 設定ファイルディレクトリ
│   ├── config.example.yaml   # 設定ファイルのテンプレート
│   └── config_bedrock.yaml   # AWS Bedrock用設定例
├── pyproject.toml            # プロジェクト設定と依存関係 (PDM)
├── README.md                 # このファイル
├── data/                     # 処理対象のドキュメントを配置するディレクトリ (input_dir)
│   ├── sample_document.txt
│   ├── sample_data.csv
│   └── ...
├── graphrag_output/          # 処理結果が保存されるルートディレクトリ (output_dir)
│   └── alopexdb/             # SQLデータとmain/entity/community HNSWインデックス
├── scripts/
│   └── demo.sh               # 機能紹介デモスクリプト
└── src/
    └── graphrag_anthropic_llamaindex/
        ├── __init__.py
        ├── main.py           # CLIエントリポイント
        ├── config_manager.py # 設定ファイルの読み込み
        ├── alopex_db.py      # 組み込みAlopexDBの接続とスキーマ
        ├── db_manager.py     # AlopexDB SQLデータの読み書き
        ├── alopex_vector_store.py # LlamaIndex用HNSWアダプター
        ├── vector_store_manager.py # ベクトルストアとインデックスの管理
        ├── graph_operations.py # グラフ操作とコミュニティ検出
        ├── llm_utils.py      # LLMからのJSON出力パースとプロンプトテンプレート
        ├── document_processor.py # ドキュメントの追加処理
```

## インストール

1.  **リポジトリのクローン:**
    ```bash
    git clone <リポジトリのURL>
    cd graphrag-anthropic-llamaindex
    ```

2.  **PDM のインストール (未導入の場合):**
    ```bash
    curl -sSL https://pdm-project.org/install-pdm.py | python3 -
    ```
    Python は 3.11 以上 3.13 未満が必要です (`.tool-versions` は 3.11.11)。

3.  **依存関係のインストール:**
    ```bash
    pdm install
    ```

## 設定

### 1. LLM プロバイダと環境変数の設定

`config/config.yaml` の `llm_provider` で LLM を選択します。

*   `claude_code` (デフォルト): ログイン済みの `claude` CLI のサブスクリプションを claude-agent-sdk 経由で利用します。API キーは不要です(個人のローカル利用のみ)。
*   `anthropic`: `ANTHROPIC_API_KEY` を `.env` または環境変数で指定します(YAML には書けません)。
*   `bedrock`: AWS Bedrock (`BedrockConverse`) を利用します。

`anthropic` を使う場合は、`.env.example` を `.env` としてコピーし、APIキーを設定してください。

```bash
cp .env.example .env
```

`.env` ファイルを編集して、Anthropic APIキーを設定:

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### 2. 設定ファイルの作成

`config/config.example.yaml` を `config/config.yaml` としてコピーし、必要な設定を更新してください。

```bash
cp config/config.example.yaml config/config.yaml
```

`config/config.yaml` の主要な設定項目:
*   `anthropic.model`: 使用するClaudeモデル名（例: `claude-3-opus-20240229`）。
*   `embedding_model.name`: 埋め込みモデル名（デフォルト: `intfloat/multilingual-e5-small`）。
*   `input_dir`: 処理対象のドキュメントが置かれるディレクトリ（デフォルト: `./data`）。
*   `output_dir`: 組み込みAlopexDBが保存されるルートディレクトリ（デフォルト: `./graphrag_output`）。DB本体はその直下の `alopexdb/` に作られます。

## 使用方法

### 1. ドキュメントの追加と処理

`input_dir` で指定されたディレクトリ（デフォルト: `data/`）に処理したいドキュメント（TXT, CSV, PDF, DOCX, PPTX, HTML, EMLなど）を配置します。

```bash
pdm run python -m graphrag_anthropic_llamaindex.main --config config/config.yaml add
```

*   このコマンドは、`input_dir` 内の新しいドキュメントを読み込み、テキスト抽出、チャンキング、エンティティ・リレーションシップ抽出、コミュニティ検出、コミュニティ要約を行います。
*   処理済みファイル情報、抽出データ、3種類のベクトルインデックスは同じ組み込みAlopexDBへ保存されます。
*   メインテキスト、エンティティ、コミュニティ要約のベクトルインデックスが更新されます。
*   初回実行時や新しいドキュメントが追加された場合、LLMの呼び出しや埋め込みモデルのダウンロードにより時間がかかることがあります。

### サーバー構成

サーバー構成は、AlopexDBとLlamaIndex APIを別コンテナで起動し、Docker Composeの内部ネットワークで接続します。クライアントはLlamaIndex APIだけを呼び出します。

```text
クライアント → http://127.0.0.1:8000 → LlamaIndex API → http://alopexdb:8080 → AlopexDB
```

```bash
make server-build
make server-up
make server-test
```

独立クライアントの使用例:

```bash
pdm run python -m graphrag_anthropic_llamaindex.server_client health
pdm run python -m graphrag_anthropic_llamaindex.server_client add ./data/example.txt
pdm run python -m graphrag_anthropic_llamaindex.server_client search "検索内容" --mode global
```


### 2. 検索

以下のコマンドで、異なるインデックスを対象に検索を実行できます。

```bash
bash
pdm run python -m graphrag_anthropic_llamaindex.main --config config/config.yaml search "あなたのクエリ" --mode <local|global|drift|auto>
```

`--mode` には以下のいずれかを指定します(`config` の `search.mode` が指定されている場合はそちらが優先されます)。

*   `local`: エンティティ中心の詳細な検索。
*   `global`: コミュニティ要約の map-reduce による包括的な検索(デフォルト)。
*   `drift`: DRIFT 検索(global プライマー → 追従 local 検索)。
*   `auto`: クエリから自動でモードを選択。

**例:**

*   **デフォルト(global)での検索:**
    ```bash
    pdm run python -m graphrag_anthropic_llamaindex.main --config config/config.yaml search "AIの最新動向について教えてください"
    ```
*   **local モードでの検索:**
    ```bash
    pdm run python -m graphrag_anthropic_llamaindex.main --config config/config.yaml search "Acme CorpのCEOは誰ですか？" --mode local
    ```
*   **global モードでの検索:**
    ```bash
    pdm run python -m graphrag_anthropic_llamaindex.main --config config/config.yaml search "サステナブルテクノロジーに関するコミュニティはありますか？" --mode global
    ```
*   **モード自動選択(--mode auto)での検索:**
    ```bash
    pdm run python -m graphrag_anthropic_llamaindex.main --config config/config.yaml search "気候変動に関する主要な研究機関は？" --mode auto
    ```

## デモスクリプトの実行

`scripts/demo.sh` を実行すると、一連のドキュメント追加と検索のデモンストレーションを自動で行います。

```bash
./scripts/demo.sh
```

**注意:** 初回実行時は、依存関係のインストール、LLMの呼び出し、埋め込みモデルのダウンロードなどにより、時間がかかる場合があります。また、Anthropic APIキーが`.env`ファイルまたは環境変数に正しく設定されていることを確認してください。
