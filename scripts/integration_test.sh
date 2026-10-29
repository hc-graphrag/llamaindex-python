#!/bin/bash

# 統合テストスクリプト：GraphRAG with LlamaIndex and Anthropic

echo "==================================================="
echo "GraphRAG 統合テストスクリプト"
echo "==================================================="

CONFIG_FILE="config/config.yaml"
# output_dir の値から前後の引用符だけを除去する（スラッシュ/ドットは保持）
OUTPUT_DIR=$(grep -E "^output_dir:" "$CONFIG_FILE" | head -n1 | sed -E 's/^output_dir:[[:space:]]*//; s/^"(.*)"$/\1/; s/^'\''(.*)'\''$/\1/')

if [ -z "$OUTPUT_DIR" ] || [ "$OUTPUT_DIR" = "/" ] || [ "$OUTPUT_DIR" = "." ]; then
    echo "エラー: output_dir を設定ファイルから取得できませんでした。処理を中止します。"
    exit 1
fi

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
pdm run python -m graphrag_anthropic_llamaindex.main --config "$CONFIG_FILE" add
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
pdm run python -m graphrag_anthropic_llamaindex.main --config "$CONFIG_FILE" search "$QUERY_MAIN" --mode local
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
pdm run python -m graphrag_anthropic_llamaindex.main --config "$CONFIG_FILE" search "$QUERY_ENTITY" --mode local
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
pdm run python -m graphrag_anthropic_llamaindex.main --config "$CONFIG_FILE" search "$QUERY_COMMUNITY" --mode global
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
pdm run python -m graphrag_anthropic_llamaindex.main --config "$CONFIG_FILE" search "$QUERY_BOTH" --mode auto
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
