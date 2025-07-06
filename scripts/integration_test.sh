#!/bin/bash

# 統合テストスクリプト：GraphRAG with LlamaIndex and Anthropic

echo "==================================================="
echo "GraphRAG 統合テストスクリプト"
echo "==================================================="

CONFIG_FILE="config.yaml"
OUTPUT_DIR=$(grep -E "^output_dir:" $CONFIG_FILE | awk '{print $2}' | tr -d '"' | tr -d '/.')

# 1. 既存の出力データをクリーンアップ
echo "1. 既存の出力データ ($OUTPUT_DIR) をクリーンアップ..."
rm -rf "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "エラー: 出力ディレクトリのクリーンアップに失敗しました。"
    exit 1
fi
echo "クリーンアップ完了。"

# 2. データの追加と処理
echo -e "\n2. ドキュメントの追加と処理 (add コマンド実行)..."
poetry run python src/graphrag_anthropic_llamaindex/main.py --config "$CONFIG_FILE" add
if [ $? -ne 0 ]; then
    echo "エラー: ドキュメントの追加処理に失敗しました。"
    exit 1
fi
echo "ドキュメント追加処理完了。"

# 3. 検索機能のテスト

echo -e "\n3. 検索機能のテスト:"
TEST_FAILURES=0

# 3.1. メインテキストインデックスの検索テスト
echo -e "\n--- 3.1. メインテキストインデックス検索テスト ---"
QUERY_MAIN="AIの最新動向について教えてください"
echo "クエリ: \"$QUERY_MAIN\""
poetry run python src/graphrag_anthropic_llamaindex/main.py --config "$CONFIG_FILE" search "$QUERY_MAIN" --target-index main
if [ $? -ne 0 ]; then
    echo "テスト失敗: メインテキストインデックス検索"
    TEST_FAILURES=$((TEST_FAILURES + 1))
else
    echo "テスト成功: メインテキストインデックス検索"
fi

# 3.2. エンティティインデックスの検索テスト
echo -e "\n--- 3.2. エンティティインデックス検索テスト ---"
QUERY_ENTITY="Acme CorpのCEOは誰ですか？"
echo "クエリ: \"$QUERY_ENTITY\""
poetry run python src/graphrag_anthropic_llamaindex/main.py --config "$CONFIG_FILE" search "$QUERY_ENTITY" --target-index entity
if [ $? -ne 0 ]; then
    echo "テスト失敗: エンティティインデックス検索"
    TEST_FAILURES=$((TEST_FAILURES + 1))
else
    echo "テスト成功: エンティティインデックス検索"
fi

# 3.3. コミュニティ要約インデックスの検索テスト
echo -e "\n--- 3.3. コミュニティ要約インデックス検索テスト ---"
QUERY_COMMUNITY="サステナブルテクノロジーに関するコミュニティはありますか？"
echo "クエリ: \"$QUERY_COMMUNITY\""
poetry run python src/graphrag_anthropic_llamaindex/main.py --config "$CONFIG_FILE" search "$QUERY_COMMUNITY" --target-index community
if [ $? -ne 0 ]; then
    echo "テスト失敗: コミュニティ要約インデックス検索"
    TEST_FAILURES=$((TEST_FAILURES + 1))
else
    echo "テスト成功: コミュニティ要約インデックス検索"
fi

# 3.4. 全てのインデックス検索テスト (デフォルト)
echo -e "\n--- 3.4. 全てのインデックス検索テスト ---"
QUERY_BOTH="気候変動に関する主要な研究機関は？"
echo "クエリ: \"$QUERY_BOTH\""
poetry run python src/graphrag_anthropic_llamaindex/main.py --config "$CONFIG_FILE" search "$QUERY_BOTH" --target-index both
if [ $? -ne 0 ]; then
    echo "テスト失敗: 全てのインデックス検索"
    TEST_FAILURES=$((TEST_FAILURES + 1))
else
    echo "テスト成功: 全てのインデックス検索"
fi

echo -e "\n==================================================="
if [ $TEST_FAILURES -eq 0 ]; then
    echo "統合テスト完了: 全てのテストが成功しました！"
    exit 0
else
    echo "統合テスト完了: $TEST_FAILURES 個のテストが失敗しました。"
    exit 1
fi
echo "==================================================="
