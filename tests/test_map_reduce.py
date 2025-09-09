"""
Unit tests for MapProcessor and ReduceProcessor
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import List, Dict, Any

from graphrag_anthropic_llamaindex.global_search.map_processor import MapProcessor
from graphrag_anthropic_llamaindex.global_search.reduce_processor import ReduceProcessor
from graphrag_anthropic_llamaindex.global_search.models import (
    MapResult, KeyPoint, GlobalSearchResult, TraceabilityInfo
)


@pytest.fixture(autouse=True)
def mock_llm_settings():
    """自動的にSettingsをモック"""
    with patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings') as mock_map_settings, \
         patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Settings') as mock_reduce_settings, \
         patch('graphrag_anthropic_llamaindex.global_search.map_processor.Anthropic') as mock_map_anthropic, \
         patch('graphrag_anthropic_llamaindex.global_search.map_processor.Bedrock') as mock_map_bedrock, \
         patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Anthropic') as mock_reduce_anthropic, \
         patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Bedrock') as mock_reduce_bedrock:
        
        # Settingsのllmを常にNoneに設定
        mock_map_settings.llm = None
        mock_reduce_settings.llm = None
        
        # モックLLMを返すように設定
        mock_llm = Mock()
        mock_llm.chat = Mock(return_value="Mocked response")
        
        mock_map_anthropic.return_value = mock_llm
        mock_map_bedrock.return_value = mock_llm
        mock_reduce_anthropic.return_value = mock_llm
        mock_reduce_bedrock.return_value = mock_llm
        
        yield {
            'map_settings': mock_map_settings,
            'reduce_settings': mock_reduce_settings,
            'map_anthropic': mock_map_anthropic,
            'map_bedrock': mock_map_bedrock,
            'reduce_anthropic': mock_reduce_anthropic,
            'reduce_bedrock': mock_reduce_bedrock,
            'mock_llm': mock_llm
        }


class TestMapProcessor:
    """MapProcessorのテストクラス"""
    
    @pytest.fixture
    def mock_llm_config(self):
        """テスト用のLLM設定"""
        return {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229"
        }
    
    @pytest.fixture
    def mock_llm(self):
        """モックLLM"""
        llm = Mock()
        llm.chat = Mock(return_value="Test response")
        return llm
    
    @pytest.fixture
    def sample_batch(self):
        """サンプルバッチデータ"""
        return {
            "context": "Report context about technology",
            "records": [
                {
                    "id": "report_1",
                    "content": "Technology report",
                    "metadata": {
                        "document_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "entity_ids": ["entity_1", "entity_2"]
                    }
                }
            ],
            "tokens": 100,
            "report_ids": ["report_1"]
        }
    
    @pytest.fixture
    def sample_batches(self):
        """複数のサンプルバッチ"""
        return [
            {
                "context": "Report 1 context",
                "records": [{"id": "r1", "content": "Content 1"}],
                "tokens": 100,
                "report_ids": ["r1"]
            },
            {
                "context": "Report 2 context",
                "records": [{"id": "r2", "content": "Content 2"}],
                "tokens": 150,
                "report_ids": ["r2"]
            }
        ]
    
    def test_init(self, mock_llm_config):
        """MapProcessorの初期化をテスト"""
        with patch.object(MapProcessor, '_get_or_create_llm') as mock_get_llm:
            mock_get_llm.return_value = Mock()
            
            processor = MapProcessor(
                llm_config=mock_llm_config,
                max_concurrent=3,
                response_type="multiple paragraphs"
            )
            
            assert processor.max_concurrent == 3
            assert processor.response_type == "multiple paragraphs"
            assert processor.semaphore._value == 3
            mock_get_llm.assert_called_once_with(mock_llm_config)
    
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    def test_get_or_create_llm_with_existing(self, mock_settings, mock_llm_config, mock_llm):
        """既存のLLMがある場合のテスト"""
        mock_settings.llm = mock_llm
        
        processor = MapProcessor(llm_config=mock_llm_config)
        
        assert processor.llm == mock_llm
    
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Anthropic')
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    def test_get_or_create_llm_anthropic(self, mock_settings, mock_anthropic_class, mock_llm_config):
        """Anthropic LLMの作成をテスト"""
        mock_settings.llm = None
        mock_anthropic = Mock()
        mock_anthropic_class.return_value = mock_anthropic
        
        processor = MapProcessor(llm_config=mock_llm_config)
        
        assert processor.llm == mock_anthropic
        mock_anthropic_class.assert_called_once_with(
            model="claude-3-opus-20240229"
        )
    
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Bedrock')
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    def test_get_or_create_llm_bedrock(self, mock_settings, mock_bedrock_class):
        """Bedrock LLMの作成をテスト"""
        mock_settings.llm = None
        mock_bedrock = Mock()
        mock_bedrock_class.return_value = mock_bedrock
        
        llm_config = {
            "provider": "bedrock",
            "model": "anthropic.claude-3-sonnet",
            "region": "us-west-2"
        }
        
        processor = MapProcessor(llm_config=llm_config)
        
        assert processor.llm == mock_bedrock
        mock_bedrock_class.assert_called_once_with(
            model="anthropic.claude-3-sonnet",
            region_name="us-west-2"
        )
    
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Anthropic')
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    def test_extract_key_points_json_format(self, mock_settings, mock_anthropic_class, mock_llm_config):
        """JSON形式のレスポンスからのキーポイント抽出をテスト"""
        mock_settings.llm = None
        mock_anthropic_class.return_value = Mock()
        processor = MapProcessor(llm_config=mock_llm_config)
        
        llm_response = '''
        Here are the key points:
        ```json
        {
            "key_points": [
                {
                    "description": "First key point",
                    "score": 90,
                    "report_ids": ["r1", "r2"]
                },
                {
                    "description": "Second key point",
                    "score": 80,
                    "report_ids": ["r3"]
                }
            ]
        }
        ```
        '''
        
        report_ids = ["r1", "r2", "r3"]
        records = []
        
        key_points = processor.extract_key_points(llm_response, report_ids, records)
        
        assert len(key_points) == 2
        assert key_points[0].description == "First key point"
        assert key_points[0].score == 90
        assert key_points[1].description == "Second key point"
        assert key_points[1].score == 80
    
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    def test_extract_key_points_json_list_format(self, mock_settings, mock_llm_config):
        """JSONリスト形式のレスポンスからの抽出をテスト"""
        mock_settings.llm = None
        processor = MapProcessor(llm_config=mock_llm_config)
        
        llm_response = '''
        ```json
        [
            {
                "description": "Point 1",
                "score": 85
            },
            {
                "description": "Point 2",
                "score": 75
            }
        ]
        ```
        '''
        
        report_ids = ["r1"]
        records = []
        
        key_points = processor.extract_key_points(llm_response, report_ids, records)
        
        assert len(key_points) == 2
        assert key_points[0].description == "Point 1"
        assert key_points[0].score == 85
    
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    def test_extract_key_points_text_format(self, mock_settings, mock_llm_config):
        """テキスト形式のレスポンスからの抽出をテスト"""
        mock_settings.llm = None
        processor = MapProcessor(llm_config=mock_llm_config)
        
        llm_response = '''
        This is the first important paragraph with significant information.
        
        This is the second paragraph with additional details.
        
        - Bullet point one with key insight
        - Bullet point two with another insight
        '''
        
        report_ids = ["r1", "r2"]
        records = []
        
        key_points = processor.extract_key_points(llm_response, report_ids, records)
        
        assert len(key_points) > 0
        assert any("first important paragraph" in kp.description for kp in key_points)
        assert any("Bullet point" in kp.description for kp in key_points)
    
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    def test_extract_from_text_with_bullets(self, mock_settings, mock_llm_config):
        """箇条書きを含むテキストからの抽出をテスト"""
        mock_settings.llm = None
        processor = MapProcessor(llm_config=mock_llm_config)
        
        text = '''
        - First bullet point with enough content to be included
        - Second bullet point with significant information
        • Third bullet point using different style
        1. Numbered item with important details
        '''
        
        report_ids = ["r1"]
        records = []
        
        key_points = processor._extract_from_text(text, report_ids, records)
        
        assert len(key_points) == 4
        assert "First bullet point" in key_points[0].description
        assert "Second bullet point" in key_points[1].description
        assert "Third bullet point" in key_points[2].description
        assert "Numbered item" in key_points[3].description
    
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    def test_extract_metadata(self, mock_settings, mock_llm_config):
        """メタデータ抽出をテスト"""
        mock_settings.llm = None
        processor = MapProcessor(llm_config=mock_llm_config)
        
        records = [
            {
                "id": "r1",
                "metadata": {
                    "document_id": "doc1",
                    "chunk_id": "chunk1",
                    "entity_ids": ["e1", "e2"]
                }
            },
            {
                "id": "r2",
                "metadata": {
                    "document_id": "doc2",
                    "chunk_id": "chunk2",
                    "entity_ids": ["e3"]
                }
            }
        ]
        
        metadata = processor._extract_metadata(records, ["r1", "r2"])
        
        assert "doc1" in metadata["document_ids"]
        assert "doc2" in metadata["document_ids"]
        assert "chunk1" in metadata["chunk_ids"]
        assert "e1" in metadata["entity_ids"]
        assert "e3" in metadata["entity_ids"]
        assert len(metadata["entity_ids"]) == 3  # 重複削除後
    
    @pytest.mark.asyncio
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    async def test_process_single_batch_success(self, mock_settings, mock_llm_config, sample_batch):
        """単一バッチ処理の成功ケースをテスト"""
        mock_settings.llm = None
        processor = MapProcessor(llm_config=mock_llm_config)
        
        # モックLLM応答を設定
        with patch.object(processor, '_call_llm_async') as mock_call:
            mock_call.return_value = '''
            ```json
            {
                "key_points": [
                    {"description": "Test point", "score": 85}
                ]
            }
            ```
            '''
            
            result = await processor._process_single_batch(sample_batch, "test query", 0)
            
            assert isinstance(result, MapResult)
            assert result.batch_id == 0
            assert len(result.key_points) == 1
            assert result.key_points[0].description == "Test point"
            assert result.context_tokens == 100
    
    @pytest.mark.asyncio
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    async def test_process_single_batch_error(self, mock_settings, mock_llm_config, sample_batch):
        """単一バッチ処理のエラーケースをテスト"""
        mock_settings.llm = None
        processor = MapProcessor(llm_config=mock_llm_config)
        
        with patch.object(processor, '_call_llm_async') as mock_call:
            mock_call.side_effect = Exception("LLM error")
            
            with pytest.raises(Exception, match="LLM error"):
                await processor._process_single_batch(sample_batch, "test query", 0)
    
    @pytest.mark.asyncio
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    async def test_process_batch_parallel(self, mock_settings, mock_llm_config, sample_batches):
        """並列バッチ処理をテスト"""
        mock_settings.llm = None
        processor = MapProcessor(llm_config=mock_llm_config, max_concurrent=2)
        
        with patch.object(processor, '_process_single_batch') as mock_process:
            # 異なる結果を返すように設定
            async def mock_result(batch, query, batch_id):
                return MapResult(
                    batch_id=batch_id,
                    key_points=[
                        KeyPoint(
                            description=f"Point from batch {batch_id}",
                            score=80 + batch_id,
                            report_ids=batch["report_ids"]
                        )
                    ],
                    context_tokens=batch["tokens"],
                    processing_time=0.1
                )
            
            mock_process.side_effect = mock_result
            
            results = await processor.process_batch(sample_batches, "test query")
            
            assert len(results) == 2
            assert results[0].batch_id == 0
            assert results[1].batch_id == 1
            assert "Point from batch 0" in results[0].key_points[0].description
            assert "Point from batch 1" in results[1].key_points[0].description
    
    @pytest.mark.asyncio
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    async def test_process_batch_with_exception(self, mock_settings, mock_llm_config, sample_batches):
        """例外を含む並列処理をテスト"""
        mock_settings.llm = None
        processor = MapProcessor(llm_config=mock_llm_config)
        
        with patch.object(processor, '_process_single_batch') as mock_process:
            # 最初のバッチは成功、2番目は失敗
            async def mock_result(batch, query, batch_id):
                if batch_id == 0:
                    return MapResult(
                        batch_id=0,
                        key_points=[KeyPoint("Success", 80, ["r1"])],
                        context_tokens=100,
                        processing_time=0.1
                    )
                else:
                    raise Exception("Batch processing error")
            
            mock_process.side_effect = mock_result
            
            results = await processor.process_batch(sample_batches, "test query")
            
            assert len(results) == 2
            assert len(results[0].key_points) == 1
            assert len(results[1].key_points) == 0  # エラーの場合は空の結果
    
    @pytest.mark.asyncio
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    async def test_call_llm_async(self, mock_settings, mock_llm_config, mock_llm):
        """非同期LLM呼び出しをテスト"""
        mock_settings.llm = None
        processor = MapProcessor(llm_config=mock_llm_config)
        processor.llm = mock_llm
        mock_llm.chat.return_value = "LLM response"
        
        response = await processor._call_llm_async("system", "user")
        
        assert response == "LLM response"
        mock_llm.chat.assert_called_once()


class TestReduceProcessor:
    """ReduceProcessorのテストクラス"""
    
    @pytest.fixture
    def mock_llm_config(self):
        """テスト用のLLM設定"""
        return {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229"
        }
    
    @pytest.fixture
    def mock_llm(self):
        """モックLLM"""
        llm = Mock()
        llm.chat = Mock(return_value="Final synthesized response")
        return llm
    
    @pytest.fixture
    def sample_map_results(self):
        """サンプルMap結果"""
        return [
            MapResult(
                batch_id=0,
                key_points=[
                    KeyPoint(
                        description="First key point",
                        score=90,
                        report_ids=["r1"],
                        source_metadata={
                            "document_ids": ["doc1"],
                            "chunk_ids": ["chunk1"],
                            "entity_ids": ["e1"]
                        }
                    ),
                    KeyPoint(
                        description="Second key point",
                        score=70,
                        report_ids=["r2"]
                    )
                ],
                context_tokens=100,
                processing_time=0.5
            ),
            MapResult(
                batch_id=1,
                key_points=[
                    KeyPoint(
                        description="Third key point",
                        score=80,
                        report_ids=["r3"],
                        source_metadata={
                            "document_ids": ["doc2"],
                            "chunk_ids": ["chunk2"],
                            "entity_ids": ["e2", "e3"]
                        }
                    )
                ],
                context_tokens=150,
                processing_time=0.6
            )
        ]
    
    def test_init(self, mock_llm_config):
        """ReduceProcessorの初期化をテスト"""
        with patch.object(ReduceProcessor, '_get_or_create_llm') as mock_get_llm:
            mock_get_llm.return_value = Mock()
            
            processor = ReduceProcessor(
                llm_config=mock_llm_config,
                response_type="multiple paragraphs"
            )
            
            assert processor.response_type == "multiple paragraphs"
            mock_get_llm.assert_called_once_with(mock_llm_config)
    
    @patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Settings')
    def test_reduce_success(self, mock_settings, mock_llm_config, mock_llm, sample_map_results):
        """Reduce処理の成功ケースをテスト"""
        mock_settings.llm = None
        processor = ReduceProcessor(llm_config=mock_llm_config)
        processor.llm = mock_llm
        
        result = processor.reduce(
            map_results=sample_map_results,
            query="test query",
            processing_time=1.5,
            output_format="markdown"
        )
        
        assert isinstance(result, GlobalSearchResult)
        assert result.response == "Final synthesized response"
        assert result.response_type == "multiple paragraphs"
        assert result.total_tokens == 250  # 100 + 150
        assert result.processing_time == 1.5
        
        # LLMが呼び出されたか確認
        mock_llm.chat.assert_called_once()
    
    @patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Settings')
    def test_reduce_with_llm_error(self, mock_settings, mock_llm_config, mock_llm, sample_map_results):
        """LLMエラー時のフォールバックをテスト"""
        mock_settings.llm = None
        processor = ReduceProcessor(llm_config=mock_llm_config)
        processor.llm = mock_llm
        mock_llm.chat.side_effect = Exception("LLM error")
        
        result = processor.reduce(
            map_results=sample_map_results,
            query="test query",
            processing_time=1.5
        )
        
        # フォールバック応答が生成されるか確認
        assert isinstance(result, GlobalSearchResult)
        assert "test query" in result.response
        assert "自動生成されたもの" in result.response
    
    @patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Settings')
    def test_build_reduce_context(self, mock_settings, mock_llm_config):
        """Reduceコンテキスト構築をテスト"""
        mock_settings.llm = None
        processor = ReduceProcessor(llm_config=mock_llm_config)
        
        key_points = [
            KeyPoint("Point 1", 90, ["r1", "r2", "r3", "r4"]),
            KeyPoint("Point 2", 80, ["r5"])
        ]
        
        context = processor._build_reduce_context(key_points)
        
        assert "-----Key Points-----" in context
        assert "score|description|report_ids" in context
        assert "90|Point 1|r1;r2;r3" in context  # 最初の3つのIDのみ
        assert "80|Point 2|r5" in context
    
    @patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Settings')
    def test_build_traceability(self, mock_settings, mock_llm_config):
        """トレーサビリティ情報構築をテスト"""
        mock_settings.llm = None
        processor = ReduceProcessor(llm_config=mock_llm_config)
        
        key_points = [
            KeyPoint(
                "Point 1", 90, ["r1"],
                source_metadata={
                    "document_ids": ["doc1", "doc2"],
                    "chunk_ids": ["chunk1"],
                    "entity_ids": ["e1"]
                }
            ),
            KeyPoint(
                "Point 2", 80, ["r2"],
                source_metadata={
                    "document_ids": ["doc2", "doc3"],
                    "chunk_ids": ["chunk2"],
                    "entity_ids": ["e1", "e2"]
                }
            )
        ]
        
        traceability = processor._build_traceability(key_points)
        
        assert isinstance(traceability, TraceabilityInfo)
        assert "r1" in traceability.report_ids
        assert "r2" in traceability.report_ids
        assert "doc1" in traceability.document_ids
        assert "doc3" in traceability.document_ids
        assert "chunk1" in traceability.chunk_ids
        assert "e1" in traceability.entity_ids
        assert "e2" in traceability.entity_ids
    
    @patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Settings')
    def test_create_fallback_response(self, mock_settings, mock_llm_config):
        """フォールバック応答作成をテスト"""
        mock_settings.llm = None
        processor = ReduceProcessor(llm_config=mock_llm_config)
        
        key_points = [
            KeyPoint(f"Point {i}", 100 - i * 10, [f"r{i}"])
            for i in range(15)
        ]
        
        response = processor._create_fallback_response(key_points, "test query")
        
        assert "test query" in response
        assert "1. Point 0" in response
        assert "10. Point 9" in response
        assert "11. Point 10" not in response  # 最大10個まで
        assert "自動生成されたもの" in response
    
    @patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Settings')
    def test_format_output(self, mock_settings, mock_llm_config, sample_map_results):
        """出力フォーマッティングをテスト"""
        mock_settings.llm = None
        processor = ReduceProcessor(llm_config=mock_llm_config)
        
        result = GlobalSearchResult(
            response="Test response",
            response_type="multiple paragraphs",
            map_results=sample_map_results,
            traceability=TraceabilityInfo(
                report_ids=["r1"],
                document_ids=["doc1"],
                chunk_ids=["chunk1"],
                entity_ids=["e1"]
            ),
            total_tokens=250,
            processing_time=1.5
        )
        
        # Markdown形式
        markdown_output = processor.format_output(result, "markdown")
        assert isinstance(markdown_output, str)
        assert "Test response" in markdown_output
        assert "GLOBAL Search Result" in markdown_output
        
        # JSON形式
        json_output = processor.format_output(result, "json")
        assert isinstance(json_output, dict)
        assert json_output["response"] == "Test response"
    
    @patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Settings')
    def test_key_point_sorting(self, mock_settings, mock_llm_config, mock_llm):
        """キーポイントのスコアソートをテスト"""
        mock_settings.llm = None
        processor = ReduceProcessor(llm_config=mock_llm_config)
        processor.llm = mock_llm
        
        # スコアがバラバラのMap結果を作成
        map_results = [
            MapResult(
                batch_id=0,
                key_points=[
                    KeyPoint("Low score", 50, ["r1"]),
                    KeyPoint("High score", 95, ["r2"]),
                    KeyPoint("Medium score", 75, ["r3"])
                ],
                context_tokens=100,
                processing_time=0.5
            )
        ]
        
        # reduce内でキーポイントがソートされることを確認
        with patch.object(processor, '_build_reduce_context') as mock_build:
            processor.reduce(map_results, "test", 1.0)
            
            # _build_reduce_contextに渡されたキーポイントを確認
            called_key_points = mock_build.call_args[0][0]
            
            # スコアが降順でソートされているか確認
            scores = [kp.score for kp in called_key_points]
            assert scores == [95, 75, 50]
    
    @patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Settings')
    def test_top_key_points_limit(self, mock_settings, mock_llm_config, mock_llm):
        """上位20個のキーポイント制限をテスト"""
        mock_settings.llm = None
        processor = ReduceProcessor(llm_config=mock_llm_config)
        processor.llm = mock_llm
        
        # 30個のキーポイントを持つMap結果を作成
        map_results = [
            MapResult(
                batch_id=0,
                key_points=[
                    KeyPoint(f"Point {i}", 100 - i, [f"r{i}"])
                    for i in range(30)
                ],
                context_tokens=100,
                processing_time=0.5
            )
        ]
        
        with patch.object(processor, '_build_reduce_context') as mock_build:
            processor.reduce(map_results, "test", 1.0)
            
            # 上位20個のみが渡されることを確認
            called_key_points = mock_build.call_args[0][0]
            assert len(called_key_points) == 20
            
            # 最高スコアと最低スコアを確認
            assert called_key_points[0].score == 100
            assert called_key_points[-1].score == 81


class TestIntegration:
    """MapとReduceの統合テスト"""
    
    @pytest.mark.asyncio
    @patch('graphrag_anthropic_llamaindex.global_search.map_processor.Settings')
    @patch('graphrag_anthropic_llamaindex.global_search.reduce_processor.Settings')
    async def test_map_reduce_pipeline(self, mock_reduce_settings, mock_map_settings):
        """Map-Reduceパイプライン全体をテスト"""
        mock_map_settings.llm = None
        mock_reduce_settings.llm = None
        # Map処理
        map_processor = MapProcessor(
            llm_config={"provider": "anthropic"},
            max_concurrent=2
        )
        
        batches = [
            {
                "context": "Technology context",
                "records": [{"id": "r1"}],
                "tokens": 100,
                "report_ids": ["r1"]
            },
            {
                "context": "Science context",
                "records": [{"id": "r2"}],
                "tokens": 150,
                "report_ids": ["r2"]
            }
        ]
        
        # モックLLM応答を設定
        with patch.object(map_processor, '_call_llm_async') as mock_map_llm:
            mock_map_llm.return_value = '''
            ```json
            {
                "key_points": [
                    {"description": "Key insight", "score": 85}
                ]
            }
            ```
            '''
            
            map_results = await map_processor.process_batch(batches, "test query")
        
        assert len(map_results) == 2
        
        # Reduce処理
        reduce_processor = ReduceProcessor(
            llm_config={"provider": "anthropic"}
        )
        
        with patch.object(reduce_processor.llm, 'chat') as mock_reduce_llm:
            mock_reduce_llm.return_value = "Final comprehensive answer"
            
            final_result = reduce_processor.reduce(
                map_results=map_results,
                query="test query",
                processing_time=2.0
            )
        
        assert isinstance(final_result, GlobalSearchResult)
        assert final_result.response == "Final comprehensive answer"
        assert final_result.total_tokens == 250
        assert len(final_result.map_results) == 2