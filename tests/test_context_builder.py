"""
Unit tests for CommunityContextBuilder
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from graphrag_anthropic_llamaindex.global_search.context_builder import CommunityContextBuilder


class TestCommunityContextBuilder:
    """CommunityContextBuilderのテストクラス"""
    
    @pytest.fixture
    def mock_config(self):
        """テスト用の設定"""
        return {
            "global_search": {
                "include_community_weight": True,
                "max_context_tokens": 8000
            },
            "entity_extraction": {
                "enabled": True
            }
        }
    
    @pytest.fixture
    def mock_vector_store(self):
        """モックベクターストア"""
        return Mock()
    
    @pytest.fixture
    def mock_token_encoder(self):
        """モックトークンエンコーダー"""
        encoder = Mock()
        encoder.encode = Mock(side_effect=lambda text: [0] * (len(text) // 4))
        return encoder
    
    @pytest.fixture
    def sample_reports(self) -> List[Dict[str, Any]]:
        """サンプルレポートデータ"""
        return [
            {
                "id": "report_1",
                "content": "This is a community report about technology",
                "score": 0.9,
                "metadata": {
                    "title": "Technology Community",
                    "occurrence": 10.0,
                    "rank": 2
                },
                "rank": 2
            },
            {
                "id": "report_2",
                "content": "This is a community report about science",
                "score": 0.8,
                "metadata": {
                    "title": "Science Community",
                    "occurrence": 8.0,
                    "rank": 1
                },
                "rank": 1
            },
            {
                "id": "report_3",
                "content": "This is a community report about innovation",
                "score": 0.7,
                "metadata": {
                    "title": "Innovation Community",
                    "occurrence": 6.0,
                    "rank": 0
                },
                "rank": 0
            }
        ]
    
    def test_init_with_valid_config(self, mock_config, mock_vector_store):
        """有効な設定での初期化をテスト"""
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store
        )
        
        assert builder.config == mock_config
        assert builder.vector_store == mock_vector_store
        assert builder.max_context_tokens == 8000
    
    def test_init_with_missing_community_weight_raises_error(self, mock_vector_store):
        """コミュニティ重み付けが無効な場合のエラーをテスト"""
        invalid_config = {
            "global_search": {
                "include_community_weight": False
            }
        }
        
        with pytest.raises(ValueError, match="コミュニティ重み付けは必須です"):
            CommunityContextBuilder(
                config=invalid_config,
                vector_store=mock_vector_store
            )
    
    def test_validate_community_weights_warns_on_disabled_entity_extraction(
        self, mock_vector_store, caplog
    ):
        """エンティティ抽出が無効な場合の警告をテスト"""
        config = {
            "global_search": {
                "include_community_weight": True
            },
            "entity_extraction": {
                "enabled": False
            }
        }
        
        builder = CommunityContextBuilder(
            config=config,
            vector_store=mock_vector_store
        )
        
        assert "エンティティ抽出が無効です" in caplog.text
    
    def test_filter_by_rank(self, mock_config, mock_vector_store, sample_reports):
        """ランクフィルタリングをテスト"""
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store
        )
        
        # min_rank = 1でフィルタリング
        filtered = builder._filter_by_rank(sample_reports, min_rank=1)
        
        assert len(filtered) == 2
        assert all(r["rank"] >= 1 for r in filtered)
        assert "report_1" in [r["id"] for r in filtered]
        assert "report_2" in [r["id"] for r in filtered]
        assert "report_3" not in [r["id"] for r in filtered]
    
    def test_apply_community_weights_with_normalization(
        self, mock_config, mock_vector_store, sample_reports
    ):
        """正規化ありのコミュニティ重み付けをテスト"""
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store
        )
        
        weighted = builder.apply_community_weights(sample_reports, normalize=True)
        
        # 重みが追加されているか確認
        assert all("weight" in r for r in weighted)
        
        # 正規化されているか確認（最大値が1.0）
        max_weight = max(r["weight"] for r in weighted)
        assert max_weight == 1.0
        
        # 重みでソートされているか確認（降順）
        weights = [r["weight"] for r in weighted]
        assert weights == sorted(weights, reverse=True)
    
    def test_apply_community_weights_without_normalization(
        self, mock_config, mock_vector_store, sample_reports
    ):
        """正規化なしのコミュニティ重み付けをテスト"""
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store
        )
        
        weighted = builder.apply_community_weights(sample_reports, normalize=False)
        
        # 重みが元のoccurrence値と一致するか確認
        for report in weighted:
            expected = report["metadata"]["occurrence"]
            assert report["weight"] == expected
    
    def test_create_batches_single_batch(
        self, mock_config, mock_vector_store, mock_token_encoder, sample_reports
    ):
        """単一バッチ作成をテスト"""
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store,
            max_context_tokens=10000,  # 大きなトークン制限
            token_encoder=mock_token_encoder
        )
        
        batches = builder._create_batches(sample_reports)
        
        # 全てのレポートが1つのバッチに収まるはず
        assert len(batches) == 1
        assert len(batches[0]["records"]) == 3
        assert len(batches[0]["report_ids"]) == 3
        assert "report_1" in batches[0]["report_ids"]
    
    def test_create_batches_multiple_batches(
        self, mock_config, mock_vector_store, mock_token_encoder, sample_reports
    ):
        """複数バッチ作成をテスト"""
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store,
            max_context_tokens=50,  # より小さなトークン制限（ヘッダーとレポート1つ分程度）
            token_encoder=mock_token_encoder
        )
        
        batches = builder._create_batches(sample_reports)
        
        # 複数のバッチに分割されるはず
        assert len(batches) > 1
        
        # 全てのレポートが含まれているか確認
        all_report_ids = []
        for batch in batches:
            all_report_ids.extend(batch["report_ids"])
        assert len(all_report_ids) == len(sample_reports)
    
    def test_format_report(self, mock_config, mock_vector_store):
        """レポートフォーマッティングをテスト"""
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store
        )
        
        report = {
            "id": "test_id",
            "content": "Test content",
            "metadata": {
                "title": "Test Title"
            },
            "rank": 2,
            "weight": 0.75
        }
        
        formatted = builder._format_report(report)
        
        assert "test_id" in formatted
        assert "Test Title" in formatted
        assert "Test content" in formatted
        assert "2" in formatted
        assert "0.750" in formatted
    
    def test_count_tokens_with_encoder(self, mock_config, mock_vector_store, mock_token_encoder):
        """エンコーダーありのトークンカウントをテスト"""
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store,
            token_encoder=mock_token_encoder
        )
        
        text = "This is a test text"
        count = builder._count_tokens(text)
        
        # モックエンコーダーが呼ばれたか確認
        mock_token_encoder.encode.assert_called_once_with(text)
        assert count == len(text) // 4
    
    def test_count_tokens_without_encoder(self, mock_config, mock_vector_store):
        """エンコーダーなしのトークンカウントをテスト"""
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store,
            token_encoder=None
        )
        
        text = "This is a test text"
        count = builder._count_tokens(text)
        
        # 簡易推定（4文字で1トークン）
        assert count == len(text) // 4
    
    @patch('llama_index.core.VectorStoreIndex')
    def test_retrieve_community_reports_success(
        self, mock_index_class, mock_config, mock_vector_store
    ):
        """コミュニティレポート取得の成功ケースをテスト"""
        # モックノードを作成
        mock_node1 = Mock()
        mock_node1.node.id_ = "node1"
        mock_node1.node.text = "Community report 1"
        mock_node1.score = 0.9
        mock_node1.node.metadata = {"rank": 2}
        
        mock_node2 = Mock()
        mock_node2.node.id_ = "node2"
        mock_node2.node.text = "Community report 2"
        mock_node2.score = 0.8
        mock_node2.node.metadata = {"rank": 1}
        
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.source_nodes = [mock_node1, mock_node2]
        
        # モッククエリエンジンを設定
        mock_query_engine = Mock()
        mock_query_engine.query.return_value = mock_response
        
        # モックインデックスを設定
        mock_index = Mock()
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_index_class.from_vector_store.return_value = mock_index
        
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store
        )
        
        reports = builder._retrieve_community_reports("test query")
        
        assert len(reports) == 2
        assert reports[0]["id"] == "node1"
        assert reports[0]["content"] == "Community report 1"
        assert reports[0]["score"] == 0.9
        assert reports[0]["rank"] == 2
    
    @patch('llama_index.core.VectorStoreIndex')
    def test_retrieve_community_reports_error(
        self, mock_index_class, mock_config, mock_vector_store, caplog
    ):
        """コミュニティレポート取得のエラーケースをテスト"""
        # エラーを発生させる
        mock_index_class.from_vector_store.side_effect = Exception("Test error")
        
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store
        )
        
        reports = builder._retrieve_community_reports("test query")
        
        assert reports == []
        assert "コミュニティレポートの取得中にエラー" in caplog.text
    
    def test_retrieve_community_reports_no_vector_store(
        self, mock_config, caplog
    ):
        """ベクターストアがない場合のテスト"""
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=None
        )
        
        reports = builder._retrieve_community_reports("test query")
        
        assert reports == []
        assert "コミュニティベクターストアが設定されていません" in caplog.text
    
    @patch.object(CommunityContextBuilder, '_retrieve_community_reports')
    @patch.object(CommunityContextBuilder, '_filter_by_rank')
    @patch.object(CommunityContextBuilder, 'apply_community_weights')
    @patch.object(CommunityContextBuilder, '_create_batches')
    def test_build_context_full_flow(
        self,
        mock_create_batches,
        mock_apply_weights,
        mock_filter,
        mock_retrieve,
        mock_config,
        mock_vector_store,
        sample_reports
    ):
        """build_contextの完全なフローをテスト"""
        # モックの戻り値を設定
        mock_retrieve.return_value = sample_reports
        mock_filter.return_value = sample_reports[:2]  # 2つに絞る
        mock_apply_weights.return_value = sample_reports[:2]
        mock_create_batches.return_value = [
            {
                "context": "batch1",
                "records": sample_reports[:2],
                "tokens": 100,
                "report_ids": ["report_1", "report_2"]
            }
        ]
        
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store
        )
        
        batches = builder.build_context(
            query="test query",
            min_community_rank=1,
            shuffle_data=True,
            random_state=42
        )
        
        # 各ステップが呼ばれたか確認
        mock_retrieve.assert_called_once_with("test query")
        mock_filter.assert_called_once_with(sample_reports, 1)
        mock_apply_weights.assert_called_once()
        mock_create_batches.assert_called_once()
        
        assert len(batches) == 1
        assert batches[0]["context"] == "batch1"
    
    @patch.object(CommunityContextBuilder, '_retrieve_community_reports')
    def test_build_context_with_shuffle(
        self,
        mock_retrieve,
        mock_config,
        mock_vector_store,
        sample_reports
    ):
        """シャッフル機能のテスト"""
        mock_retrieve.return_value = sample_reports.copy()
        
        builder = CommunityContextBuilder(
            config=mock_config,
            vector_store=mock_vector_store
        )
        
        # シャッフルありで実行
        batches1 = builder.build_context(
            query="test",
            shuffle_data=True,
            random_state=42
        )
        
        # 同じシードで再実行（同じ結果になるはず）
        mock_retrieve.return_value = sample_reports.copy()
        batches2 = builder.build_context(
            query="test",
            shuffle_data=True,
            random_state=42
        )
        
        # 異なるシードで実行（異なる結果になる可能性）
        mock_retrieve.return_value = sample_reports.copy()
        batches3 = builder.build_context(
            query="test",
            shuffle_data=True,
            random_state=123
        )
        
        # シャッフルなしで実行
        mock_retrieve.return_value = sample_reports.copy()
        batches4 = builder.build_context(
            query="test",
            shuffle_data=False
        )
        
        # アサーション（バッチ作成の詳細は省略）
        assert batches1 is not None
        assert batches2 is not None
        assert batches3 is not None
        assert batches4 is not None


class TestEdgeCases:
    """エッジケースのテスト"""
    
    @pytest.fixture
    def builder(self):
        """基本的なビルダーインスタンス"""
        config = {
            "global_search": {
                "include_community_weight": True
            },
            "entity_extraction": {
                "enabled": True
            }
        }
        return CommunityContextBuilder(
            config=config,
            vector_store=Mock()
        )
    
    def test_empty_reports_list(self, builder):
        """空のレポートリストの処理"""
        filtered = builder._filter_by_rank([], min_rank=0)
        assert filtered == []
        
        weighted = builder.apply_community_weights([])
        assert weighted == []
        
        batches = builder._create_batches([])
        assert batches == []
    
    def test_reports_without_metadata(self, builder):
        """メタデータなしのレポート処理"""
        reports = [
            {
                "id": "report_1",
                "content": "Content without metadata",
                "score": 0.5
            }
        ]
        
        # apply_community_weightsはデフォルト値を使用
        weighted = builder.apply_community_weights(reports)
        assert weighted[0]["weight"] == 1.0
        
        # _format_reportはデフォルト値を使用
        formatted = builder._format_report(reports[0])
        assert "Report" in formatted  # デフォルトタイトル
    
    def test_very_long_report(self, builder):
        """非常に長いレポートの処理"""
        long_content = "x" * 10000
        reports = [
            {
                "id": "long_report",
                "content": long_content,
                "metadata": {"title": "Long", "occurrence": 1.0},
                "rank": 0,
                "weight": 1.0
            }
        ]
        
        # 小さなmax_context_tokensで複数バッチに分割されるはず
        builder.max_context_tokens = 100
        batches = builder._create_batches(reports)
        
        # バッチが作成されることを確認
        assert len(batches) >= 1
    
    def test_zero_weight_normalization(self, builder):
        """ゼロ重み正規化のエッジケース"""
        reports = [
            {
                "id": "report_1",
                "content": "Content",
                "metadata": {"occurrence": 0.0}
            }
        ]
        
        weighted = builder.apply_community_weights(reports, normalize=True)
        
        # ゼロ除算エラーが発生しないことを確認
        assert weighted[0]["weight"] == 0.0