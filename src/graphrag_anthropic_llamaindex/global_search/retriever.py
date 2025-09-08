"""
GlobalSearchRetriever - LlamaIndexと統合されたGLOBAL検索のRetriever実装
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from .models import GlobalSearchResult
from .context_builder import CommunityContextBuilder
from .map_processor import MapProcessor
from .reduce_processor import ReduceProcessor

logger = logging.getLogger(__name__)


class GlobalSearchRetriever(BaseRetriever):
    """GLOBAL検索を実行するLlamaIndexリトリーバー"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        vector_store=None,
        response_type: str = "multiple paragraphs",
        min_community_rank: int = 0,
        max_concurrent: int = 5,
        shuffle_data: bool = True,
        random_state: int = 42,
        output_format: str = "markdown"
    ):
        """
        Args:
            config: 設定辞書
            vector_store: コミュニティベクターストア（Noneの場合は設定から作成）
            response_type: レスポンスタイプ
            min_community_rank: 最小コミュニティランク
            max_concurrent: 最大同時実行数
            shuffle_data: データをシャッフルするか
            random_state: ランダムシード
            output_format: 出力形式（"markdown" または "json"）
        """
        super().__init__()
        self.config = config
        self.response_type = response_type
        self.min_community_rank = min_community_rank
        self.output_format = output_format
        
        # コンポーネントを初期化
        self.context_builder = CommunityContextBuilder(
            config=config,
            vector_store=vector_store,
            max_context_tokens=config.get("global_search", {}).get("max_context_tokens", 8000)
        )
        
        llm_config = config.get("llm", {})
        self.map_processor = MapProcessor(
            llm_config=llm_config,
            max_concurrent=max_concurrent,
            response_type=response_type
        )
        
        self.reduce_processor = ReduceProcessor(
            llm_config=llm_config,
            response_type=response_type
        )
        
        self.shuffle_data = shuffle_data
        self.random_state = random_state
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        同期的に検索を実行
        
        Args:
            query_bundle: クエリバンドル
        
        Returns:
            検索結果のNodeWithScoreリスト
        """
        # 非同期メソッドを同期的に実行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._aretrieve(query_bundle))
        finally:
            loop.close()
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        非同期的に検索を実行
        
        Args:
            query_bundle: クエリバンドル
        
        Returns:
            検索結果のNodeWithScoreリスト
        """
        start_time = time.time()
        query = query_bundle.query_str
        
        logger.info(f"GLOBAL検索を開始: {query}")
        
        try:
            # 1. コンテキストを構築してバッチに分割
            batches = self.context_builder.build_context(
                query=query,
                min_community_rank=self.min_community_rank,
                shuffle_data=self.shuffle_data,
                random_state=self.random_state
            )
            
            if not batches:
                logger.warning("コンテキストバッチが空です")
                return []
            
            logger.info(f"{len(batches)} バッチを処理します")
            
            # 2. Map処理を並列実行
            map_results = await self.map_processor.process_batch(batches, query)
            
            # 3. Reduce処理で最終回答を生成
            processing_time = time.time() - start_time
            global_result = self.reduce_processor.reduce(
                map_results=map_results,
                query=query,
                processing_time=processing_time,
                output_format=self.output_format
            )
            
            # 4. NodeWithScoreに変換
            nodes = self._create_nodes(global_result)
            
            logger.info(f"GLOBAL検索完了: {processing_time:.2f}秒")
            
            return nodes
            
        except Exception as e:
            logger.error(f"GLOBAL検索中にエラー: {e}")
            raise
    
    def _create_nodes(self, result: GlobalSearchResult) -> List[NodeWithScore]:
        """
        GlobalSearchResultをNodeWithScoreのリストに変換
        
        Args:
            result: GlobalSearchResult
        
        Returns:
            NodeWithScoreのリスト
        """
        nodes = []
        
        # メインの応答ノード
        main_node = TextNode(
            text=result.response,
            metadata={
                "response_type": result.response_type,
                "total_tokens": result.total_tokens,
                "processing_time": result.processing_time,
                "traceability": result.traceability.to_dict(),
                "search_type": "global"
            }
        )
        nodes.append(NodeWithScore(node=main_node, score=1.0))
        
        # 各キーポイントをノードとして追加（オプション）
        if self.output_format == "json" or self.config.get("global_search", {}).get("include_key_points", False):
            for map_result in result.map_results:
                for kp in map_result.key_points:
                    kp_node = TextNode(
                        text=kp.description,
                        metadata={
                            "type": "key_point",
                            "score": kp.score,
                            "report_ids": kp.report_ids,
                            "source_metadata": kp.source_metadata,
                            "batch_id": map_result.batch_id
                        }
                    )
                    # スコアを0-1の範囲に正規化
                    normalized_score = kp.score / 100.0
                    nodes.append(NodeWithScore(node=kp_node, score=normalized_score))
        
        return nodes
    
    def retrieve_with_traceability(self, query: str) -> GlobalSearchResult:
        """
        トレーサビリティ情報付きで検索を実行
        
        Args:
            query: 検索クエリ
        
        Returns:
            GlobalSearchResult（完全なトレーサビリティ情報を含む）
        """
        query_bundle = QueryBundle(query_str=query)
        
        # 非同期メソッドを同期的に実行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            start_time = time.time()
            
            # バッチを構築
            batches = self.context_builder.build_context(
                query=query,
                min_community_rank=self.min_community_rank,
                shuffle_data=self.shuffle_data,
                random_state=self.random_state
            )
            
            # Map処理
            map_results = loop.run_until_complete(
                self.map_processor.process_batch(batches, query)
            )
            
            # Reduce処理
            processing_time = time.time() - start_time
            result = self.reduce_processor.reduce(
                map_results=map_results,
                query=query,
                processing_time=processing_time,
                output_format=self.output_format
            )
            
            return result
            
        finally:
            loop.close()