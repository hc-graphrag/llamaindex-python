"""
コミュニティレポートのコンテキスト構築とバッチ処理
"""

import logging
from typing import List, Dict, Any, Optional
import random
import pandas as pd

from ..vector_store_manager import get_vector_store, get_index

logger = logging.getLogger(__name__)


class CommunityContextBuilder:
    """コミュニティレポートのコンテキストを構築しバッチに分割する"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        vector_store=None,
        max_context_tokens: int = 8000,
        token_encoder=None
    ):
        """
        Args:
            config: 設定辞書
            vector_store: ベクターストアインスタンス（Noneの場合は設定から作成）
            max_context_tokens: バッチあたりの最大トークン数
            token_encoder: トークンエンコーダー
        """
        self.config = config
        self.max_context_tokens = max_context_tokens
        self.token_encoder = token_encoder
        
        # ベクターストアの初期化
        if vector_store is None:
            self.vector_store = get_vector_store(config, store_type="community")
        else:
            self.vector_store = vector_store
            
        # コミュニティ重み付けの検証（必須）
        self._validate_community_weights()
        
    def _validate_community_weights(self):
        """コミュニティ重み付けが設定されているか検証"""
        global_search_config = self.config.get("global_search", {})
        
        # 重み付けが有効になっているか確認
        if not global_search_config.get("include_community_weight", True):
            raise ValueError(
                "コミュニティ重み付けは必須です。"
                "config.yamlで 'global_search.include_community_weight: true' を設定してください。"
            )
        
        # エンティティ情報が利用可能か確認（重み付けに必要）
        if not self.config.get("entity_extraction", {}).get("enabled", False):
            logger.warning(
                "エンティティ抽出が無効です。"
                "コミュニティ重み付けには 'entity_extraction.enabled: true' が推奨されます。"
            )
    
    def build_context(
        self,
        query: str,
        min_community_rank: int = 0,
        shuffle_data: bool = True,
        random_state: int = 42
    ) -> List[Dict[str, Any]]:
        """
        クエリに基づいてコンテキストをバッチに分割して構築
        
        Args:
            query: 検索クエリ
            min_community_rank: 最小コミュニティランク（0が最下層）
            shuffle_data: データをシャッフルするか
            random_state: ランダムシード
        
        Returns:
            バッチのリスト、各バッチは以下を含む辞書：
            - context: コンテキストテキスト
            - records: レコードのDataFrame
            - tokens: トークン数
        """
        # コミュニティサマリーインデックスから関連情報を取得
        community_reports = self._retrieve_community_reports(query)
        
        # min_community_rankでフィルタリング
        filtered_reports = self._filter_by_rank(community_reports, min_community_rank)
        
        # コミュニティ重み付けを適用
        weighted_reports = self.apply_community_weights(filtered_reports)
        
        # データをシャッフル
        if shuffle_data:
            random.seed(random_state)
            random.shuffle(weighted_reports)
        
        # バッチに分割
        batches = self._create_batches(weighted_reports)
        
        return batches
    
    def _retrieve_community_reports(self, query: str) -> List[Dict[str, Any]]:
        """コミュニティレポートを取得"""
        # ベクターストアから検索
        if self.vector_store is None:
            logger.warning("コミュニティベクターストアが設定されていません")
            return []
        
        # LlamaIndexのベクターストアから検索
        try:
            from llama_index.core import VectorStoreIndex
            
            # インデックスを作成または取得
            index = VectorStoreIndex.from_vector_store(self.vector_store)
            
            # クエリエンジンを作成
            query_engine = index.as_query_engine(
                similarity_top_k=50  # 上位50件のコミュニティレポートを取得
            )
            
            # 検索実行
            response = query_engine.query(query)
            
            # レスポンスから情報を抽出
            reports = []
            for node in response.source_nodes:
                report = {
                    "id": node.node.id_,
                    "content": node.node.text,
                    "score": node.score,
                    "metadata": node.node.metadata or {},
                    "rank": node.node.metadata.get("rank", 0) if node.node.metadata else 0
                }
                reports.append(report)
            
            return reports
            
        except Exception as e:
            logger.error(f"コミュニティレポートの取得中にエラー: {e}")
            return []
    
    def _filter_by_rank(
        self,
        reports: List[Dict[str, Any]],
        min_rank: int
    ) -> List[Dict[str, Any]]:
        """ランクでフィルタリング"""
        filtered = []
        for report in reports:
            if report.get("rank", 0) >= min_rank:
                filtered.append(report)
        
        logger.info(f"ランク >= {min_rank} でフィルタリング: {len(reports)} -> {len(filtered)} レポート")
        return filtered
    
    def apply_community_weights(
        self,
        reports: List[Dict[str, Any]],
        normalize: bool = True
    ) -> List[Dict[str, Any]]:
        """
        コミュニティ重み付けを適用
        
        Args:
            reports: コミュニティレポートのリスト
            normalize: 重みを正規化するか
        
        Returns:
            重み付けされたレポートのリスト
        """
        if not reports:
            return []
        
        # occurrence値を取得または計算
        for report in reports:
            # メタデータからoccurrenceを取得
            occurrence = report.get("metadata", {}).get("occurrence", 1.0)
            report["weight"] = occurrence
        
        # 正規化
        if normalize:
            max_weight = max(r["weight"] for r in reports)
            if max_weight > 0:
                for report in reports:
                    report["weight"] = report["weight"] / max_weight
        
        # 重みでソート（降順）
        reports.sort(key=lambda x: x["weight"], reverse=True)
        
        return reports
    
    def _create_batches(
        self,
        reports: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """レポートをバッチに分割"""
        batches = []
        current_batch = {
            "context": "",
            "records": [],
            "tokens": 0,
            "report_ids": []
        }
        
        header = "-----Reports-----\nid|title|content|rank|weight\n"
        current_batch["context"] = header
        current_batch["tokens"] = self._count_tokens(header)
        
        for report in reports:
            # レポートのテキストを作成
            report_text = self._format_report(report)
            report_tokens = self._count_tokens(report_text)
            
            # バッチに収まるか確認
            if current_batch["tokens"] + report_tokens > self.max_context_tokens:
                # 現在のバッチを保存
                if current_batch["records"]:
                    batches.append(current_batch)
                
                # 新しいバッチを開始
                current_batch = {
                    "context": header,
                    "records": [],
                    "tokens": self._count_tokens(header),
                    "report_ids": []
                }
            
            # レポートを現在のバッチに追加
            current_batch["context"] += report_text
            current_batch["records"].append(report)
            current_batch["tokens"] += report_tokens
            current_batch["report_ids"].append(report["id"])
        
        # 最後のバッチを追加
        if current_batch["records"]:
            batches.append(current_batch)
        
        logger.info(f"{len(reports)} レポートを {len(batches)} バッチに分割")
        return batches
    
    def _format_report(self, report: Dict[str, Any]) -> str:
        """レポートをテキスト形式にフォーマット"""
        title = report.get("metadata", {}).get("title", "Report")
        content = report.get("content", "")
        rank = report.get("rank", 0)
        weight = report.get("weight", 1.0)
        
        # CSV形式の行を作成
        row = f"{report['id']}|{title}|{content}|{rank}|{weight:.3f}\n"
        return row
    
    def _count_tokens(self, text: str) -> int:
        """テキストのトークン数をカウント"""
        if self.token_encoder is None:
            # 簡易的な推定（4文字で1トークン）
            return len(text) // 4
        
        try:
            return len(self.token_encoder.encode(text))
        except Exception:
            return len(text) // 4