# Docker環境でのGraphRAG起動方法

このディレクトリにはGraphRAG Anthropic LlamaIndex WebアプリをDockerコンテナで実行するための設定が含まれています。

## 📁 ディレクトリ構造

```
docker/
├── Dockerfile              # コンテナイメージ定義
├── docker-compose.yml      # コンテナオーケストレーション
├── .env.example            # 環境変数のサンプル
└── README.md               # このファイル
```

## 🚀 起動方法

### 1. 環境変数の設定

```bash
# .envファイルを作成
cp docker/.env.example docker/.env

# APIキーを設定
vim docker/.env
```

### 2. 必要なディレクトリの準備

```bash
# プロジェクトルートで実行
mkdir -p data graphrag_output

# サンプルドキュメントを配置
echo "サンプルテキスト" > data/sample.txt
```

### 3. 設定ファイルの準備

```bash
# config.yaml を作成（プロジェクトルートに配置）。`make setup`（`make up` が内部で実行）でも作られます。
cp docker/config.docker.yaml config.yaml
```

コンテナは非 root ユーザー `app` で動きます。bind mount（`graphrag_output/` など）に書かれるファイルがホストのユーザー所有になるよう、`make build` / `make up` はホストの UID/GID をビルド引数（`APP_UID` / `APP_GID`）として渡します。`make setup` が先に `data/` と `graphrag_output/` を作るため、これらがホストのユーザー所有になります（Docker に作らせると root 所有になります）。

`llm_provider` は `anthropic`（`docker/.env` の `ANTHROPIC_API_KEY`）、`bedrock`、または `claude_code`（`docker/.env` の `CLAUDE_CODE_OAUTH_TOKEN`、個人利用のみ）を使えます。`claude_code` の手順と注意点は [docs/guides/docker-claude-subscription.md](../docs/guides/docker-claude-subscription.md) を参照してください。

### 4. アプリケーションの起動

#### Makefileを使用（推奨）

```bash
# ヘルプを表示
make help

# 初期セットアップ + 起動
make up

# ログを確認
make logs

# ブラウザでアクセス
open http://localhost:7860
```

#### Docker Composeを直接使用

```bash
# コンテナをビルド・起動
docker-compose -f docker/docker-compose.yml up -d

# ログを確認
docker-compose -f docker/docker-compose.yml logs -f

# ブラウザでアクセス
open http://localhost:7860
```

## 📂 ボリュームマウント

以下のローカルディレクトリがコンテナにマウントされます：

| ローカルパス | コンテナパス | 用途 | モード |
|-------------|-------------|------|-------|
| `./data/` | `/app/data/` | 入力ドキュメント | 読み取り専用 |
| `./graphrag_output/` | `/app/graphrag_output/` | 処理結果・インデックス | 読み書き |
| `./config.yaml` | `/app/config/config.yaml` | 設定ファイル（Gradio UIの既定パス `config/config.yaml`） | 読み取り専用 |

## 🛠️ 管理コマンド

### Makefileコマンド（推奨）

```bash
# 利用可能なコマンド一覧
make help

# 基本操作
make up          # 起動
make down        # 停止
make restart     # 再起動
make status      # 状態確認
make logs        # ログ表示

# 開発用
make dev         # フォアグラウンド起動
make shell       # コンテナ内シェル
make test        # 動作確認

# メンテナンス
make clean       # クリーンアップ
make backup-data # データバックアップ
```

### Docker Composeコマンド

```bash
# コンテナの状態確認
docker-compose -f docker/docker-compose.yml ps

# コンテナに入る
docker-compose -f docker/docker-compose.yml exec graphrag-app bash

# コンテナを停止
docker-compose -f docker/docker-compose.yml down

# コンテナを再起動
docker-compose -f docker/docker-compose.yml restart

# イメージを再ビルド
docker-compose -f docker/docker-compose.yml build --no-cache
```

## 🔧 トラブルシューティング

### よくある問題

1. **APIキーエラー**
   ```bash
   # .envファイルを確認
   cat docker/.env
   ```

2. **ポート競合**
   ```bash
   # 別のポートを使用
   sed -i 's/7860:7860/8080:7860/' docker/docker-compose.yml
   ```

3. **ファイルアクセス権限**
   ```bash
   # ディレクトリの権限を確認
   ls -la data/ graphrag_output/
   
   # 必要に応じて権限を変更
   chmod 755 data/ graphrag_output/
   ```

### ログの確認

```bash
# アプリケーションログ
docker-compose -f docker/docker-compose.yml logs graphrag-app

# リアルタイムログ
docker-compose -f docker/docker-compose.yml logs -f graphrag-app
```

### デバッグモード

```bash
# デバッグ用にコンテナに入る
docker-compose -f docker/docker-compose.yml exec graphrag-app bash

# 手動でアプリを起動
python gradio_app.py
```

## 🐳 Docker環境の利点

- **環境の統一**: ローカル環境に依存しない実行環境
- **簡単な配布**: Docker環境があれば誰でも実行可能
- **隔離性**: ホストシステムから独立した実行環境
- **スケーラビリティ**: 複数インスタンスの起動が容易

## 📋 注意事項

- `data/`ディレクトリは読み取り専用でマウントされます
- 処理結果は`graphrag_output/`に保存されます
- APIキーは環境変数で安全に管理されます
- コンテナは自動的にヘルスチェックを実行します