# GraphRAG Anthropic LlamaIndex Makefile
# Docker環境での開発・運用を簡単にするためのコマンド集

.PHONY: help build up down restart logs status clean setup dev test

# デフォルトターゲット
help: ## ヘルプを表示
	@echo "GraphRAG Anthropic LlamaIndex - 利用可能なコマンド:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# Docker Compose設定
COMPOSE_FILE = docker/docker-compose.yml
SERVICE_NAME = graphrag-app

# 環境構築
setup: ## 初期環境をセットアップ
	@echo "🔧 初期環境をセットアップ中..."
	@mkdir -p data graphrag_output
	@if [ ! -f docker/.env ]; then \
		cp docker/.env.example docker/.env; \
		echo "📝 docker/.env ファイルを作成しました。APIキーを設定してください。"; \
	fi
	@if [ ! -f config.yaml ]; then \
		echo "📄 config.yaml サンプルを作成中..."; \
		cat > config.yaml << 'EOF'; \
anthropic:; \
  api_key: "your-api-key-will-be-set-by-env"; \
  model: "claude-3-opus-20240229"; \
; \
input_dir: "/app/data"; \
output_dir: "/app/graphrag_output"; \
; \
embedding_model:; \
  name: "intfloat/multilingual-e5-small"; \
; \
chunking:; \
  chunk_size: 1024; \
  chunk_overlap: 20; \
; \
ignore_patterns:; \
  - "*.tmp"; \
  - ".git/*"; \
  - "__pycache__/*"; \
EOF; \
		echo "📄 config.yaml ファイルを作成しました。"; \
	fi
	@echo "✅ セットアップ完了！"

# Docker操作
build: ## Dockerイメージをビルド
	@echo "🏗️  Dockerイメージをビルド中..."
	@docker-compose -f $(COMPOSE_FILE) build

up: setup ## コンテナを起動（デタッチモード）
	@echo "🚀 GraphRAG アプリを起動中..."
	@docker-compose -f $(COMPOSE_FILE) up -d
	@echo "✅ アプリが起動しました: http://localhost:7860"

down: ## コンテナを停止・削除
	@echo "🛑 コンテナを停止中..."
	@docker-compose -f $(COMPOSE_FILE) down

restart: ## コンテナを再起動
	@echo "🔄 コンテナを再起動中..."
	@docker-compose -f $(COMPOSE_FILE) restart

# ログ・ステータス確認
logs: ## リアルタイムログを表示
	@echo "📋 ログを表示中... (Ctrl+C で終了)"
	@docker-compose -f $(COMPOSE_FILE) logs -f

logs-tail: ## 最新のログを表示（最後の50行）
	@docker-compose -f $(COMPOSE_FILE) logs --tail=50

status: ## コンテナの状態を確認
	@echo "📊 コンテナ状態:"
	@docker-compose -f $(COMPOSE_FILE) ps
	@echo ""
	@echo "🔍 ヘルスチェック:"
	@docker ps --filter "name=$(SERVICE_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 開発用
dev: ## 開発モードで起動（フォアグラウンド）
	@echo "💻 開発モードで起動中..."
	@docker-compose -f $(COMPOSE_FILE) up

shell: ## コンテナ内でシェルを起動
	@echo "🐚 コンテナ内シェルに接続中..."
	@docker-compose -f $(COMPOSE_FILE) exec $(SERVICE_NAME) bash

# テスト・検証
test: ## アプリの動作確認
	@echo "🧪 アプリの動作確認中..."
	@if curl -f -s http://localhost:7860/ > /dev/null; then \
		echo "✅ アプリは正常に動作しています"; \
	else \
		echo "❌ アプリにアクセスできません"; \
		exit 1; \
	fi

health: ## ヘルスチェック実行
	@echo "🏥 ヘルスチェック実行中..."
	@docker-compose -f $(COMPOSE_FILE) exec $(SERVICE_NAME) curl -f http://localhost:7860/ || echo "❌ ヘルスチェック失敗"

# クリーンアップ
clean: ## コンテナ・イメージ・ボリュームを削除
	@echo "🧹 クリーンアップ中..."
	@docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans
	@docker system prune -f
	@echo "✅ クリーンアップ完了"

clean-all: ## 全てのDocker関連リソースを削除（注意）
	@echo "⚠️  警告: 全てのDockerリソースを削除します"
	@read -p "続行しますか？ (y/N): " confirm && [ "$$confirm" = "y" ]
	@docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans
	@docker system prune -a -f --volumes
	@echo "✅ 全クリーンアップ完了"

# データ管理
backup-data: ## データディレクトリをバックアップ
	@echo "💾 データをバックアップ中..."
	@mkdir -p backups
	@tar -czf backups/graphrag-data-$(shell date +%Y%m%d-%H%M%S).tar.gz data/ graphrag_output/ config.yaml
	@echo "✅ バックアップ完了: backups/"

restore-data: ## データディレクトリを復元（バックアップファイルを指定）
	@echo "📂 利用可能なバックアップ:"
	@ls -la backups/*.tar.gz 2>/dev/null || echo "バックアップファイルが見つかりません"
	@echo "使用方法: make restore-data BACKUP=backups/graphrag-data-YYYYMMDD-HHMMSS.tar.gz"

# 情報表示
info: ## システム情報を表示
	@echo "📋 GraphRAG Anthropic LlamaIndex システム情報"
	@echo "================================================"
	@echo "🐳 Docker バージョン: $(shell docker --version)"
	@echo "🐙 Docker Compose バージョン: $(shell docker-compose --version)"
	@echo "📁 プロジェクトディレクトリ: $(shell pwd)"
	@echo "📄 設定ファイル: $(COMPOSE_FILE)"
	@echo "🌐 アプリURL: http://localhost:7860"
	@echo "================================================"

# 便利なエイリアス
start: up ## コンテナを起動（upのエイリアス）
stop: down ## コンテナを停止（downのエイリアス）
rebuild: clean build up ## 完全リビルド

# 本番環境用
prod-up: ## 本番環境用起動（ログレベル警告以上）
	@echo "🏭 本番環境モードで起動中..."
	@COMPOSE_FILE=$(COMPOSE_FILE) docker-compose up -d --remove-orphans
	@echo "✅ 本番環境で起動完了"