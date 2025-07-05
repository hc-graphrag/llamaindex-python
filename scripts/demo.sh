#!/bin/bash

# デモスクリプト：GraphRAG with LlamaIndex and Anthropic の機能紹介

echo "==================================================="
echo "GraphRAG with LlamaIndex and Anthropic デモスクリプト"
echo "==================================================="

# 1. 環境のセットアップ
echo "1. 依存関係のインストール (poetry install)..."
poetry install
if [ $? -ne 0 ]; then
    echo "エラー: poetry install に失敗しました。環境を確認してください。"
    exit 1
fi
echo "インストール完了。"

# 2. データの追加と処理
echo -e "\n2. ドキュメントの追加と処理 (エンティティ/リレーションシップ抽出、コミュニティ検出/要約)..."
echo "   (初回実行時はLLM呼び出しとモデルダウンロードに時間がかかります)"
poetry run python src/graphrag_anthropic_llamaindex/main.py --config config.yaml add
if [ $? -ne 0 ]; then
    echo "エラー: ドキュメントの追加処理に失敗しました。"
    exit 1
fi
echo "ドキュメント処理完了。"

# 3. 検索機能のデモンストレーション

echo -e "\n3. 検索機能のデモンストレーション:"

# 3.1. メインテキストインデックスの検索
echo -e "\n--- 3.1. メインテキストインデックスの検索 ---"
echo "クエリ: \"AIの最新動向について教えてください\""
poetry run python src/graphrag_anthropic_llamaindex/main.py --config config.yaml search "AIの最新動向について教えてください" --target-index main

# 3.2. エンティティインデックスの検索
echo -e "\n--- 3.2. エンティティインデックスの検索 ---"
echo "クエリ: \"Acme CorpのCEOは誰ですか？\""
poetry run python src/graphrag-anthropic-llamaindex/main.py --config config.yaml search "Acme CorpのCEOは誰ですか？" --target-index entity

# 3.3. コミュニティ要約インデックスの検索
echo -e "\n--- 3.3. コミュニティ要約インデックスの検索 ---"
echo "クエリ: \"サステナブルテクノロジーに関するコミュニティはありますか？\""
poetry run python src/graphrag-anthropic_llamaindex/main.py --config config.yaml search "サステナブルテクノロジーに関するコミュニティはありますか？" --target-index community

# 3.4. 全てのインデックスを検索 (デフォルト)
echo -e "\n--- 3.4. 全てのインデックスを検索 (デフォルト) ---"
echo "クエリ: \"気候変動に関する主要な研究機関は？\""
poetry run python src/graphrag-anthropic-llamaindex/main.py --config config.yaml search "気候変動に関する主要な研究機関は？" --target-index both

echo -e "\n==================================================="
echo "デモスクリプト完了。"
echo "==================================================="