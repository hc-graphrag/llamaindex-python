"""
SearchModeRouter - 検索モードのルーティングを管理
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from .retriever import GlobalSearchRetriever
# Local検索は後で実装（search_processorの統合）

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """検索モード"""
    LOCAL = "local"
    GLOBAL = "global"
    DRIFT = "drift"
    AUTO = "auto"


class SearchModeRouter(BaseRetriever):
    """検索モードに基づいてクエリをルーティングする統一インターフェース"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        mode: SearchMode = SearchMode.GLOBAL,
        vector_store_main=None,
        vector_store_entity=None,
        vector_store_community=None,
        **kwargs
    ):
        """
        Args:
            config: 設定辞書
            mode: 検索モード
            vector_store_main: メインベクターストア（Local検索用）
            vector_store_entity: エンティティベクターストア（Local検索用）
            vector_store_community: コミュニティベクターストア（Global検索用）
            **kwargs: 追加のRetriever設定
        """
        super().__init__()
        self.config = config
        self.mode = mode if isinstance(mode, SearchMode) else SearchMode(mode)
        
        # Local検索用のRetriever（今後実装）
        self.local_retriever = None
        # TODO: search_processorとの統合を実装
        # if vector_store_main is not None:
        #     try:
        #         self.local_retriever = LocalSearchRetriever(
        #             config=config,
        #             vector_store=vector_store_main,
        #             entity_vector_store=vector_store_entity
        #         )
        #     except Exception as e:
        #         logger.warning(f"Local検索の初期化に失敗: {e}")
        
        # Global検索用のGlobalSearchRetriever
        self.global_retriever = None
        if vector_store_community is not None or mode == SearchMode.GLOBAL:
            try:
                global_config = {**config, **kwargs}
                self.global_retriever = GlobalSearchRetriever(
                    config=global_config,
                    vector_store=vector_store_community,
                    response_type=kwargs.get("response_type", "multiple paragraphs"),
                    min_community_rank=kwargs.get("min_community_rank", 0),
                    output_format=kwargs.get("output_format", "markdown")
                )
            except Exception as e:
                logger.warning(f"Global検索の初期化に失敗: {e}")
        
        # DRIFT検索（将来の実装用）
        self.drift_retriever = None
    
    def route(self, query: str, mode: Optional[SearchMode] = None) -> SearchMode:
        """
        クエリをルーティングして適切な検索モードを決定
        
        Args:
            query: 検索クエリ
            mode: 明示的に指定された検索モード（Noneの場合は自動選択）
        
        Returns:
            選択された検索モード
        """
        if mode is not None:
            return mode
        
        if self.mode == SearchMode.AUTO:
            # 自動モード選択のロジック
            return self._auto_select_mode(query)
        
        return self.mode
    
    def _auto_select_mode(self, query: str) -> SearchMode:
        """
        クエリに基づいて自動的にモードを選択
        
        Args:
            query: 検索クエリ
        
        Returns:
            選択された検索モード
        """
        # クエリの特性を分析
        query_lower = query.lower()
        
        # キーワードベースの簡単な選択ロジック
        global_keywords = ["全体", "概要", "サマリー", "要約", "まとめ", 
                          "overall", "summary", "overview", "general"]
        local_keywords = ["詳細", "具体的", "特定", "detail", "specific", 
                         "particular", "exact"]
        
        # Global検索のキーワードが含まれる場合
        if any(kw in query_lower for kw in global_keywords):
            if self.global_retriever is not None:
                return SearchMode.GLOBAL
        
        # Local検索のキーワードが含まれる場合
        if any(kw in query_lower for kw in local_keywords):
            if self.local_retriever is not None:
                return SearchMode.LOCAL
        
        # デフォルトはGlobal（利用可能な場合）
        if self.global_retriever is not None:
            return SearchMode.GLOBAL
        elif self.local_retriever is not None:
            return SearchMode.LOCAL
        else:
            logger.warning("利用可能なRetrieverがありません")
            return SearchMode.GLOBAL
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        同期的に検索を実行
        
        Args:
            query_bundle: クエリバンドル
        
        Returns:
            検索結果のNodeWithScoreリスト
        """
        query = query_bundle.query_str
        selected_mode = self.route(query)
        
        logger.info(f"検索モード: {selected_mode.value}")
        
        if selected_mode == SearchMode.LOCAL:
            if self.local_retriever is None:
                logger.error("Local検索が初期化されていません")
                return []
            return self._execute_local_search(query_bundle)
        
        elif selected_mode == SearchMode.GLOBAL:
            if self.global_retriever is None:
                logger.error("Global検索が初期化されていません")
                return []
            return self.global_retriever._retrieve(query_bundle)
        
        elif selected_mode == SearchMode.DRIFT:
            if self.drift_retriever is None:
                logger.warning("DRIFT検索はまだ実装されていません。Global検索にフォールバック")
                if self.global_retriever is not None:
                    return self.global_retriever._retrieve(query_bundle)
                return []
            return self.drift_retriever._retrieve(query_bundle)
        
        else:
            logger.error(f"不明な検索モード: {selected_mode}")
            return []
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        非同期的に検索を実行
        
        Args:
            query_bundle: クエリバンドル
        
        Returns:
            検索結果のNodeWithScoreリスト
        """
        query = query_bundle.query_str
        selected_mode = self.route(query)
        
        logger.info(f"検索モード（非同期）: {selected_mode.value}")
        
        if selected_mode == SearchMode.LOCAL:
            if self.local_retriever is None:
                logger.error("Local検索が初期化されていません")
                return []
            # LocalのQueryProcessorは非同期をサポートしていない可能性があるため
            # 同期メソッドを呼び出す
            return self._execute_local_search(query_bundle)
        
        elif selected_mode == SearchMode.GLOBAL:
            if self.global_retriever is None:
                logger.error("Global検索が初期化されていません")
                return []
            return await self.global_retriever._aretrieve(query_bundle)
        
        elif selected_mode == SearchMode.DRIFT:
            if self.drift_retriever is None:
                logger.warning("DRIFT検索はまだ実装されていません。Global検索にフォールバック")
                if self.global_retriever is not None:
                    return await self.global_retriever._aretrieve(query_bundle)
                return []
            return await self.drift_retriever._aretrieve(query_bundle)
        
        else:
            logger.error(f"不明な検索モード: {selected_mode}")
            return []
    
    def _execute_local_search(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Local検索を実行（現在は未実装）"""
        logger.warning("Local検索は現在実装中です。Global検索を使用してください。")
        return []
    
    def get_available_modes(self) -> List[SearchMode]:
        """利用可能な検索モードのリストを返す"""
        modes = []
        
        if self.local_retriever is not None:
            modes.append(SearchMode.LOCAL)
        
        if self.global_retriever is not None:
            modes.append(SearchMode.GLOBAL)
        
        if self.drift_retriever is not None:
            modes.append(SearchMode.DRIFT)
        
        return modes