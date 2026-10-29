# GraphRAG Anthropic LlamaIndex Makefile
# Docker環境での開発・運用を簡単にするためのコマンド集

.PHONY: help build up down restart logs status clean setup dev test claude-login claude-status server-build server-up server-down server-logs server-test

# デフォルトターゲット
help: ## ヘルプを表示
	@echo "GraphRAG Anthropic LlamaIndex - 利用可能なコマンド:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# Docker Compose設定
COMPOSE_FILE = docker/docker-compose.yml

# コンテナ内ユーザーの UID/GID をホストに合わせる (bind mount 上のファイルがホストのユーザー所有になる)。
# root (UID 0) はコンテナ内の app ユーザーにできないため 1000 に置き換える。
export HOST_UID := $(shell u=$$(id -u); [ "$$u" = 0 ] && echo 1000 || echo $$u)
export HOST_GID := $(shell g=$$(id -g); [ "$$g" = 0 ] && echo 1000 || echo $$g)
SERVICE_NAME = graphrag-app

# 環境構築
setup: ## 初期環境をセットアップ
	@echo "🔧 初期環境をセットアップ中..."
	@mkdir -p data graphrag_output claude_home
	@if [ ! -f docker/.env ]; then \
		cp docker/.env.example docker/.env; \
		echo "📝 docker/.env ファイルを作成しました。APIキーを設定してください。"; \
	fi
	@if [ ! -f config.yaml ]; then \
		cp docker/config.docker.yaml config.yaml; \
		echo "📄 config.yaml ファイルを作成しました（docker/config.docker.yaml から）。"; \
	fi
	@echo "✅ セットアップ完了！"

# Docker操作
build: ## Dockerイメージをビルド
	@echo "🏗️  Dockerイメージをビルド中..."
	@docker compose -f $(COMPOSE_FILE) build

up: setup ## コンテナを起動（デタッチモード）
	@echo "🚀 GraphRAG アプリを起動中..."
	@docker compose -f $(COMPOSE_FILE) up -d
	@echo "✅ アプリが起動しました: http://localhost:7860"

down: ## コンテナを停止・削除
	@echo "🛑 コンテナを停止中..."
	@docker compose -f $(COMPOSE_FILE) down

restart: ## コンテナを再起動
	@echo "🔄 コンテナを再起動中..."
	@docker compose -f $(COMPOSE_FILE) restart

# ログ・ステータス確認
logs: ## リアルタイムログを表示
	@echo "📋 ログを表示中... (Ctrl+C で終了)"
	@docker compose -f $(COMPOSE_FILE) logs -f

logs-tail: ## 最新のログを表示（最後の50行）
	@docker compose -f $(COMPOSE_FILE) logs --tail=50

status: ## コンテナの状態を確認
	@echo "📊 コンテナ状態:"
	@docker compose -f $(COMPOSE_FILE) ps
	@echo ""
	@echo "🔍 ヘルスチェック:"
	@docker ps --filter "label=com.docker.compose.service=$(SERVICE_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 開発用
dev: ## 開発モードで起動（フォアグラウンド）
	@echo "💻 開発モードで起動中..."
	@docker compose -f $(COMPOSE_FILE) up

shell: ## コンテナ内でシェルを起動
	@echo "🐚 コンテナ内シェルに接続中..."
	@docker compose -f $(COMPOSE_FILE) exec $(SERVICE_NAME) bash

claude-login: ## コンテナ内で Claude にログイン (本人が一度だけ。claude_home/ に保存)
	@docker compose -f $(COMPOSE_FILE) exec $(SERVICE_NAME) claude auth login

claude-status: ## コンテナ内の Claude ログイン状態を表示 (未ログインは終了コード 1)
	@docker compose -f $(COMPOSE_FILE) exec -T $(SERVICE_NAME) claude auth status

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
	@docker compose -f $(COMPOSE_FILE) exec $(SERVICE_NAME) curl -fsS -o /dev/null http://localhost:7860/ && echo "✅ ヘルスチェック成功" || echo "❌ ヘルスチェック失敗"

server-build: ## AlopexDB + LlamaIndexサーバーをビルド
	@docker compose -f $(COMPOSE_FILE) --profile server build alopexdb graphrag-server

server-up: ## AlopexDB + LlamaIndexサーバーを起動
	@docker compose -f $(COMPOSE_FILE) --profile server up -d alopexdb graphrag-server

server-down: ## AlopexDB + LlamaIndexサーバーを停止
	@docker compose -f $(COMPOSE_FILE) --profile server stop graphrag-server alopexdb

server-logs: ## AlopexDB + LlamaIndexサーバーのログを表示
	@docker compose -f $(COMPOSE_FILE) --profile server logs -f alopexdb graphrag-server

server-test: ## LlamaIndex APIからAlopexDBへの接続を確認
	@pdm run python -m graphrag_anthropic_llamaindex.server_client health
	@docker compose -f $(COMPOSE_FILE) --profile server exec -T graphrag-server pdm run python -m graphrag_anthropic_llamaindex.server_smoke

# クリーンアップ
clean: ## コンテナ・イメージ・ボリュームを削除
	@echo "🧹 クリーンアップ中..."
	@docker compose -f $(COMPOSE_FILE) down -v --remove-orphans
	@docker system prune -f
	@echo "✅ クリーンアップ完了"

clean-all: ## 全てのDocker関連リソースを削除（注意）
	@echo "⚠️  警告: 全てのDockerリソースを削除します"
	@read -p "続行しますか？ (y/N): " confirm && [ "$$confirm" = "y" ]
	@docker compose -f $(COMPOSE_FILE) down -v --remove-orphans
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
	@echo "🐙 Docker Compose バージョン: $(shell docker compose version)"
	@echo "📁 プロジェクトディレクトリ: $(shell pwd)"
	@echo "📄 設定ファイル: $(COMPOSE_FILE)"
	@echo "🌐 アプリURL: http://localhost:7860"
	@echo "================================================"

# AWS認証
aws-login: ## .envファイルを使用してAWSにログイン
	@echo "🔐 AWSにログイン中..."
	@if [ -f .env ]; then \
		set -a && . $(PWD)/.env && set +a; \
		if [ -n "$$AWS_PROFILE_NAME" ]; then \
			echo "📋 AWS Profile: $$AWS_PROFILE_NAME を使用"; \
			aws configure list --profile $$AWS_PROFILE_NAME; \
			if ! aws sts get-caller-identity --profile $$AWS_PROFILE_NAME >/dev/null 2>&1; then \
				echo "⚠️  トークンが期限切れです。SSOに再ログインします..."; \
				aws sso login --profile $$AWS_PROFILE_NAME; \
				aws sts get-caller-identity --profile $$AWS_PROFILE_NAME; \
			else \
				aws sts get-caller-identity --profile $$AWS_PROFILE_NAME; \
			fi; \
		elif [ -n "$$AWS_ACCESS_KEY_ID" ] && [ -n "$$AWS_SECRET_ACCESS_KEY" ]; then \
			echo "🔑 AWS Access Key を使用"; \
			aws sts get-caller-identity; \
		else \
			echo "❌ AWS認証情報が設定されていません"; \
			echo "    .envファイルで以下のいずれかを設定してください:"; \
			echo "    - AWS_PROFILE_NAME (推奨)"; \
			echo "    - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY"; \
			exit 1; \
		fi; \
	else \
		echo "⚠️  .envファイルが見つかりません"; \
		echo "    cp .env.example .env"; \
		exit 1; \
	fi

# デモ実行
demo: ## DRIFT検索デモを実行（.envファイルを使用）
	@echo "🚀 DRIFT検索デモを開始中..."
	@if [ -f .env ]; then \
		set -a && . $(PWD)/.env && set +a && ./scripts/demo_dataset/run_demo.sh; \
	else \
		echo "⚠️  .envファイルが見つかりません。.env.exampleを参考に作成してください。"; \
		echo "    cp .env.example .env"; \
		echo "    # .envファイルを編集してAPIキーを設定"; \
		exit 1; \
	fi

# 便利なエイリアス
start: up ## コンテナを起動（upのエイリアス）
stop: down ## コンテナを停止（downのエイリアス）
rebuild: clean build up ## 完全リビルド

# 本番環境用
prod-up: ## 本番環境用起動（ログレベル警告以上）
	@echo "🏭 本番環境モードで起動中..."
	@COMPOSE_FILE=$(COMPOSE_FILE) docker compose up -d --remove-orphans
	@echo "✅ 本番環境で起動完了"
