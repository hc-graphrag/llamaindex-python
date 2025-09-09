# タスクドキュメント

## フェーズ1: コアインフラストラクチャ

- [x] 1. global_searchモジュール構造の作成
  - ファイル: src/graphrag_anthropic_llamaindex/global_search/__init__.py
  - グローバル検索機能用の新しいモジュールディレクトリを作成
  - パッケージ初期化を追加
  - 目的: GLOBAL検索のモジュール構造を確立
  - _要件: Design-Architecture_

- [x] 2. データモデルの定義
  - ファイル: src/graphrag_anthropic_llamaindex/global_search/models.py
  - MapResult, KeyPoint, GlobalSearchResult, TraceabilityInfoデータクラスを定義
  - 出力形式サポート用のJSONシリアライゼーションメソッドを追加
  - 目的: 型安全なデータ構造の確立
  - _要件: 1, 5, 6_

- [x] 3. プロンプトテンプレートの作成
  - ファイル: src/graphrag_anthropic_llamaindex/global_search/prompts.py
  - MAP_SYSTEM_PROMPTとREDUCE_SYSTEM_PROMPTを実装
  - response_typeパラメータサポートを追加
  - 目的: Map-Reduce処理用のLLMプロンプトを定義
  - _活用: MS-GraphRAGプロンプトパターン_
  - _要件: 1, 5_

## フェーズ2: コンテキスト管理

- [x] 4. CommunityContextBuilderの実装
  - ファイル: src/graphrag_anthropic_llamaindex/global_search/context_builder.py
  - バッチ作成用のbuild_contextメソッドを実装
  - 必須重み付けチェック付きのapply_community_weightsを追加
  - min_community_rankフィルタリングを実装
  - 目的: コミュニティレポートのコンテキストとバッチ処理を管理
  - _活用: vector_store_manager.py, LanceDB統合_
  - _要件: 1, 3, 4_

- [x] 5. コミュニティ重み付け検証の追加
  - ファイル: src/graphrag_anthropic_llamaindex/global_search/context_builder.py
  - __init__内で重み付け検証を実装
  - 重み付けが設定されていない場合はエラーを発生
  - normalize_community_weightサポートを追加
  - 目的: コミュニティ重み付けの必須化を強制
  - _要件: 3_

## フェーズ3: Map-Reduce処理

- [x] 6. MapProcessorの実装
  - ファイル: src/graphrag_anthropic_llamaindex/global_search/map_processor.py
  - 並列LLM呼び出し用のprocess_batchを実装
  - LLMレスポンス解析用のextract_key_pointsを追加
  - 同時実行制御用のasyncio.Semaphoreを実装
  - 目的: 並列Map処理を実行
  - _活用: 既存のLLMプロバイダー統合_
  - _要件: 1_

- [x] 7. ReduceProcessorの実装
  - ファイル: src/graphrag_anthropic_llamaindex/global_search/reduce_processor.py
  - Map結果を結合するreduceメソッドを実装
  - スコアベースのソートロジックを追加
  - markdown/json変換用のformat_outputを実装
  - 目的: Map結果を最終回答に結合
  - _活用: 既存のLLMプロバイダー統合_
  - _要件: 1, 5_

## フェーズ4: LlamaIndex統合

- [x] 8. GlobalSearchRetrieverの作成
  - ファイル: src/graphrag_anthropic_llamaindex/global_search/retriever.py
  - BaseRetrieverを継承
  - retrieveメソッドを実装
  - aretrieve非同期メソッドを実装
  - 目的: LlamaIndexのretrieverパターンと統合
  - _活用: llama_index.core.base.base_retriever.BaseRetriever_
  - _要件: 1_

- [x] 9. トレーサビリティ追跡の追加
  - ファイル: src/graphrag_anthropic_llamaindex/global_search/retriever.py
  - report_ids, document_ids, chunk_ids, entity_idsを追跡
  - TraceabilityInfoオブジェクトを構築
  - メタデータ付きのNodeWithScoreを返す
  - 目的: 完全なソーストレーサビリティを提供
  - _要件: 6_

## フェーズ5: 検索モードルーティング

- [x] 10. SearchModeRouterの作成
  - ファイル: src/graphrag_anthropic_llamaindex/global_search/router.py
  - モード選択用のrouteメソッドを実装
  - local, global, driftモードをサポート
  - デフォルトをglobalモードに設定
  - 目的: モード選択付きの統一検索インターフェース
  - _要件: 2_

- [x] 11. CLIインターフェースの更新
  - ファイル: src/graphrag_anthropic_llamaindex/main.py
  - --target-indexを--mode引数に置き換え
  - --response-type引数を追加
  - --output-format引数（markdown/json）を追加
  - --min-community-rank引数を追加
  - 目的: 新しい検索インターフェース用にCLIを更新
  - _要件: 2, 4, 5_

## フェーズ6: 設定管理

- [x] 12. 設定スキーマの更新
  - ファイル: src/graphrag_anthropic_llamaindex/config_manager.py
  - global_search設定セクションを追加
  - コミュニティ重み付け検証を追加
  - デフォルトresponse_type設定を追加
  - 目的: グローバル検索パラメータを設定
  - _活用: 既存のConfigManager_
  - _要件: 3, 5_

- [x] 13. サンプル設定の作成
  - ファイル: config.yaml（既存を更新）
  - 例付きのglobal_searchセクションを追加
  - コミュニティ重み付け要件をドキュメント化
  - min_community_rankの例を追加
  - 目的: 設定テンプレートを提供
  - _要件: 3, 4_

## フェーズ7: テスト

- [x] 14. CommunityContextBuilderの単体テスト
  - ファイル: tests/test_context_builder.py
  - バッチ作成ロジックをテスト
  - 重み付け検証とエラーハンドリングをテスト
  - ランクフィルタリングをテスト
  - 目的: コンテキスト構築の信頼性を確保
  - _要件: 3, 4_

- [x] 15. Map-Reduceプロセッサの単体テスト
  - ファイル: tests/test_map_reduce.py
  - MapProcessorのキーポイント抽出をテスト
  - ReduceProcessorのスコアソートをテスト
  - 出力フォーマッティングをテスト
  - 目的: 処理の信頼性を確保
  - _要件: 1, 5_

- [x] 16. GlobalSearchRetrieverの統合テスト
  - ファイル: tests/test_global_search_integration.py
  - LlamaIndex統合をテスト
  - 非同期/同期検索をテスト
  - トレーサビリティ情報をテスト
  - 目的: LlamaIndex互換性を確保
  - _要件: 1, 6_

- [x] 17. エンドツーエンドCLIテスト
  - ファイル: tests/test_cli_e2e.py
  - --mode global実行をテスト
  - 出力形式オプションをテスト
  - エラーシナリオをテスト
  - 目的: 完全な機能性を確保
  - _要件: All_

## フェーズ8: ドキュメント

- [x] 18. APIドキュメント
  - ファイル: docs/api/global_search.md
  - GlobalSearchRetriever APIをドキュメント化
  - 設定オプションをドキュメント化
  - 使用例を追加
  - 目的: 開発者の使用を可能にする
  - _要件: All_

- [x] 19. 使用ガイド
  - ファイル: docs/guides/global_search_usage.md
  - GLOBAL検索の使い方をドキュメント化
  - CLIコマンド例を提供
  - 設定例を含める
  - 目的: ユーザーガイドを提供
  - _要件: All_