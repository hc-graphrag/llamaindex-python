# Requirements Document

## Introduction

GLOBAL Search機能は、MS-GraphRAGのMap-Reduceパターンを使用して、コミュニティレベルの要約情報を並列処理し、包括的な回答を生成します。

## Requirements

### Requirement 1: Map-Reduce検索パターンの実装

**User Story:** GLOBAL検索として、コミュニティサマリーを並列処理して統合結果を返したい

#### Acceptance Criteria

1. WHEN グローバル検索を実行 THEN システム SHALL コミュニティレポートをバッチに分割してMap-Reduce処理する
2. IF Map処理 THEN システム SHALL 各バッチを並列でLLMに処理させ、key pointsとスコアを抽出する
3. WHEN Reduce処理 THEN システム SHALL スコア順にソートされたMap結果を統合して最終回答を生成する
4. IF max_context_tokens制限あり THEN システム SHALL トークン制限内でバッチを切り分ける

### Requirement 2: 検索モードの統一

**User Story:** CLIから--mode引数で検索タイプを選択したい

#### Acceptance Criteria

1. WHEN --mode local THEN システム SHALL エンティティベース検索を実行
2. WHEN --mode global THEN システム SHALL Map-Reduce検索を実行
3. WHEN --mode drift THEN システム SHALL DRIFT検索を実行
4. IF --mode未指定 THEN システム SHALL デフォルトでglobalを使用

### Requirement 3: コミュニティ重み付けの必須化

**User Story:** コミュニティの重要度による結果のランク付けを必須にしたい

#### Acceptance Criteria

1. WHEN global検索実行 THEN システム SHALL コミュニティ重み付け（occurrence）を確認する
2. IF エンティティ情報がない、または重み付け未設定 THEN システム SHALL エラーを表示して終了する
3. WHEN 重み付け設定済み THEN システム SHALL occurrence値に基づいて重要度順に結果を処理する
4. IF normalize_community_weight=True THEN システム SHALL 重み値を正規化する

### Requirement 4: コミュニティランクによる階層制御

**User Story:** コミュニティランクを使用して検索対象の階層を制御したい

#### Acceptance Criteria

1. WHEN min_community_rank指定あり THEN システム SHALL rank >= min_community_rankのコミュニティのみを検索
2. IF min_community_rank未指定 THEN システム SHALL デフォルト値0（全階層）を使用
3. WHEN rank値が大きい THEN システム SHALL より上位（抽象的）な階層のコミュニティを対象とする

### Requirement 5: レスポンスタイプと出力形式の設定

**User Story:** LLMの回答詳細度とCLI出力形式を独立して制御したい

#### Acceptance Criteria

1. WHEN --response-type指定あり THEN システム SHALL LLMに指定タイプ（"multiple paragraphs", "single paragraph", "list"等）で生成させる
2. IF --response-type未指定 THEN システム SHALL "multiple paragraphs"をデフォルトとする
3. WHEN --output-format指定あり THEN システム SHALL CLI出力を指定形式（markdown/json）で表示する
4. IF --output-format未指定 THEN システム SHALL markdownをデフォルトとする
5. WHEN --output-format=json THEN システム SHALL 構造化されたJSON形式で結果とメタデータを出力する

### Requirement 6: 完全なトレーサビリティ

**User Story:** 検索結果の全ての情報源を追跡可能にしたい

#### Acceptance Criteria

1. WHEN 結果生成 THEN システム SHALL レポートID、文書ID、チャンクID、エンティティIDを保持する
2. IF トレーサビリティ要求 THEN システム SHALL 完全な参照チェーンを提供する

## Non-Functional Requirements

### 実装方針
- 既存search_processorは必要に応じて廃止・置き換え可能
- LlamaIndexとの統合は適切な調査後に最適な方法を選択
- プロジェクトは開発中のため後方互換性は不要

### アーキテクチャ
- Map-Reduceパターンの明確な実装
- モジュール化された設計
- LlamaIndex統合の適切な実装