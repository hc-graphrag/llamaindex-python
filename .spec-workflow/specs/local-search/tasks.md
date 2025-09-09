# タスク一覧

## Phase 1: データモデル実装

- [x] 1. データモデルの作成 - src/graphrag_anthropic_llamaindex/local_search/models.py
  - File: src/graphrag_anthropic_llamaindex/local_search/models.py
  - Entity, Relationship, TextUnit, ContextResultの各データクラスを実装
  - 型ヒントを完備し、基本的なバリデーションを含める
  - Purpose: ローカル検索の基本データ構造を確立
  - _Requirements: 1_
  - _Prompt: Role: Python Developer | Task: dataclassを使用してEntity, Relationship, TextUnit, ContextResultの4つのデータモデルを実装。各フィールドに適切な型ヒントを追加 | Restrictions: 複雑なバリデーションは不要、シンプルな構造を維持 | Success: 全てのデータクラスが正しく定義され、型チェックがパスする_

## Phase 2: コンテキスト構築

- [ ] 2. EntityMapperの実装 - src/graphrag_anthropic_llamaindex/local_search/entity_mapper.py
  - File: src/graphrag_anthropic_llamaindex/local_search/entity_mapper.py
  - map_query_to_entitiesメソッドを実装
  - ベクトル検索を使用してクエリから関連エンティティを取得
  - Purpose: クエリとエンティティのマッピング機能を提供
  - _Leverage: src/graphrag_anthropic_llamaindex/vector_store_manager.py_
  - _Requirements: 2_
  - _Prompt: Role: Backend Developer | Task: ベクトルストアを使用してクエリから関連エンティティを検索するEntityMapperクラスを実装。similarity_searchを使用してtop_k個のエンティティを返す | Restrictions: エラー時は空のリストを返す、複雑なランキングは不要 | Success: クエリを受け取って関連エンティティのリストを返すことができる_

- [ ] 3. LocalContextBuilderの基本実装 - src/graphrag_anthropic_llamaindex/local_search/context_builder.py
  - File: src/graphrag_anthropic_llamaindex/local_search/context_builder.py
  - build_contextメソッドの基本実装
  - エンティティと関係性からシンプルなテキストコンテキストを生成
  - Purpose: 検索コンテキストの構築
  - _Requirements: 3, 5_
  - _Prompt: Role: Python Developer | Task: エンティティと関係性のリストを受け取り、LLMプロンプト用のテキストコンテキストを生成するLocalContextBuilderクラスを実装。トークン制限を考慮 | Restrictions: 複雑なフォーマットは不要、読みやすいテキスト形式で十分 | Success: エンティティと関係性からコンテキストテキストを生成できる_

## Phase 3: 検索実装

- [ ] 4. プロンプトテンプレートの作成 - src/graphrag_anthropic_llamaindex/local_search/prompts.py
  - File: src/graphrag_anthropic_llamaindex/local_search/prompts.py
  - LOCAL_SEARCH_PROMPTテンプレートを定義
  - コンテキストとクエリを組み合わせるシンプルなテンプレート
  - Purpose: LLMへのプロンプト生成
  - _Requirements: 6_
  - _Prompt: Role: Prompt Engineer | Task: ローカル検索用のプロンプトテンプレートを作成。コンテキストデータとユーザークエリを組み合わせて回答を生成するための指示を含める | Restrictions: 過度に複雑な指示は避ける、データの引用形式は後で追加 | Success: コンテキストとクエリからLLMが適切な回答を生成できるプロンプト_

- [ ] 5. LocalSearchRetrieverの実装 - src/graphrag_anthropic_llamaindex/local_search/retriever.py
  - File: src/graphrag_anthropic_llamaindex/local_search/retriever.py
  - BaseRetrieverを継承してLocalSearchRetrieverクラスを実装
  - _retrieveメソッドでEntityMapper、LocalContextBuilder、LLMを連携
  - Purpose: ローカル検索のメインエントリーポイント
  - _Leverage: llama_index.core.base.base_retriever.BaseRetriever_
  - _Requirements: 6, 7_
  - _Prompt: Role: Backend Developer | Task: BaseRetrieverを継承してLocalSearchRetrieverを実装。EntityMapperでエンティティを取得、LocalContextBuilderでコンテキストを構築、LLMで回答生成の流れを実装 | Restrictions: エラー処理はシンプルに、ストリーミングは不要 | Success: クエリを受け取って検索結果を返すことができる_

## Phase 4: 統合

- [ ] 6. SearchModeRouterへの統合 - src/graphrag_anthropic_llamaindex/global_search/router.py (修正)
  - File: src/graphrag_anthropic_llamaindex/global_search/router.py
  - _execute_local_searchメソッドを実装
  - LocalSearchRetrieverのインスタンス化と呼び出し
  - Purpose: 既存のルーティングシステムとの統合
  - _Requirements: 7_
  - _Prompt: Role: Integration Developer | Task: SearchModeRouterクラスの_execute_local_searchメソッドを実装。LocalSearchRetrieverをインスタンス化して検索を実行 | Restrictions: 既存のグローバル検索を壊さない、エラー時は空の結果を返す | Success: ローカルモードが選択された時に正しく動作する_

- [ ] 7. 初期化処理の追加 - src/graphrag_anthropic_llamaindex/local_search/__init__.py
  - File: src/graphrag_anthropic_llamaindex/local_search/__init__.py
  - モジュールのエクスポート設定
  - 必要なクラスを公開
  - Purpose: モジュールの適切な公開インターフェース
  - _Requirements: All_
  - _Prompt: Role: Python Developer | Task: local_searchモジュールの__init__.pyを作成。LocalSearchRetriever、Entity、Relationship等の主要クラスをエクスポート | Restrictions: 内部実装の詳細は公開しない | Success: from local_search import LocalSearchRetrieverが動作する_

## Phase 5: データ連携

- [ ] 8. エンティティデータのロード機能 - src/graphrag_anthropic_llamaindex/local_search/data_loader.py
  - File: src/graphrag_anthropic_llamaindex/local_search/data_loader.py
  - Parquetファイルからエンティティと関係性を読み込む
  - DataFrameをデータモデルに変換
  - Purpose: 既存のデータストレージとの連携
  - _Leverage: src/graphrag_anthropic_llamaindex/db_manager.py_
  - _Requirements: 1_
  - _Prompt: Role: Data Engineer | Task: db_managerのload_entities_db、load_relationships_dbを使用してParquetファイルからデータを読み込み、EntityとRelationshipモデルに変換する関数を実装 | Restrictions: データが存在しない場合は空のリストを返す | Success: Parquetファイルからデータモデルへの変換が正しく動作する_

## Phase 6: テストと検証

- [ ] 9. 基本的な動作テスト - tests/test_local_search.py
  - File: tests/test_local_search.py
  - エンティティマッピングのテスト
  - コンテキスト構築のテスト
  - エンドツーエンドの検索テスト
  - Purpose: 基本機能の動作確認
  - _Requirements: All_
  - _Prompt: Role: QA Engineer | Task: local_search機能の基本的なテストを作成。モックデータを使用してEntityMapper、LocalContextBuilder、LocalSearchRetrieverの動作を確認 | Restrictions: 外部依存はモック化、基本的なケースのみテスト | Success: 全ての基本機能が正しく動作することを確認_

- [ ] 10. CLIでの動作確認 - main.pyの確認
  - File: src/graphrag_anthropic_llamaindex/main.py
  - --mode localオプションで実際に動作することを確認
  - 必要に応じて微調整
  - Purpose: エンドユーザーインターフェースの動作確認
  - _Requirements: 7_
  - _Prompt: Role: DevOps Engineer | Task: main.pyでlocal検索モードが正しく動作することを確認。必要に応じてSearchModeRouterの初期化を調整 | Restrictions: 既存の機能を壊さない | Success: python -m graphrag_anthropic_llamaindex search --mode local "クエリ"が動作する_