"""
Reduce処理の実装 - Map結果の統合と最終回答生成
"""

import logging
from typing import List, Dict, Any, Optional
import json

from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.bedrock import Bedrock

from .models import MapResult, KeyPoint, GlobalSearchResult, TraceabilityInfo
from .prompts import REDUCE_SYSTEM_PROMPT, REDUCE_USER_PROMPT

logger = logging.getLogger(__name__)


class ReduceProcessor:
    """Reduce処理を実行してMap結果を統合し最終回答を生成"""
    
    def __init__(
        self,
        llm_config: Dict[str, Any],
        response_type: str = "multiple paragraphs",
        max_response_length: int = 2000
    ):
        """
        Args:
            llm_config: LLM設定辞書
            response_type: レスポンスタイプ
            max_response_length: 応答の最大語数
        """
        # LLMを初期化または既存のSettingsから取得
        self.llm = self._get_or_create_llm(llm_config)
        self.response_type = response_type
        self.max_response_length = max_response_length
    
    def _get_or_create_llm(self, llm_config: Dict[str, Any]):
        """LLMインスタンスを取得または作成"""
        # 既にSettingsにLLMが設定されていればそれを使用
        if Settings.llm is not None:
            return Settings.llm
        
        # 設定からLLMを作成
        provider = llm_config.get("provider", "anthropic")
        if provider == "bedrock":
            return Bedrock(
                model=llm_config.get("model", "anthropic.claude-3-sonnet-20240229-v1:0"),
                region_name=llm_config.get("region", "us-east-1")
            )
        else:
            return Anthropic(
                model=llm_config.get("model", "claude-3-opus-20240229")
            )
    
    def reduce(
        self,
        map_results: List[MapResult],
        query: str,
        processing_time: float,
        output_format: str = "markdown"
    ) -> GlobalSearchResult:
        """
        Map結果を統合して最終回答を生成
        
        Args:
            map_results: MapResultのリスト
            query: ユーザークエリ
            processing_time: 全体の処理時間
            output_format: 出力形式（"markdown" または "json"）
        
        Returns:
            GlobalSearchResult
        """
        # すべてのキーポイントを収集
        all_key_points = []
        for map_result in map_results:
            all_key_points.extend(map_result.key_points)
        
        # スコアでソート（降順）
        all_key_points.sort(key=lambda x: x.score, reverse=True)
        
        # 上位のキーポイントを選択（最大20個程度）
        top_key_points = all_key_points[:20]
        
        # Reduce用のコンテキストを構築
        context = self._build_reduce_context(top_key_points)
        
        # システムプロンプトを構築
        system_prompt = REDUCE_SYSTEM_PROMPT.format(
            max_length=self.max_response_length,
            response_type=self.response_type,
            report_data=context
        )
        
        # ユーザープロンプトを構築
        user_prompt = REDUCE_USER_PROMPT.format(
            report_data=context,
            query=query
        )
        
        # LLMを呼び出して最終回答を生成
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.llm.chat(messages)
            final_response = str(response)
        except Exception as e:
            logger.error(f"Reduce処理中のLLM呼び出しエラー: {e}")
            final_response = self._create_fallback_response(top_key_points, query)
        
        # トレーサビリティ情報を構築
        traceability = self._build_traceability(all_key_points)
        
        # トークン数を計算
        total_tokens = sum(mr.context_tokens for mr in map_results)
        
        # 結果を構築
        result = GlobalSearchResult(
            response=final_response,
            response_type=self.response_type,
            map_results=map_results,
            traceability=traceability,
            total_tokens=total_tokens,
            processing_time=processing_time
        )
        
        return result
    
    def _build_reduce_context(self, key_points: List[KeyPoint]) -> str:
        """Reduce用のコンテキストを構築"""
        context_lines = []
        context_lines.append("-----Key Points-----")
        context_lines.append("score|description|report_ids")
        context_lines.append("")
        
        for kp in key_points:
            # CSV形式の行を作成
            report_ids_str = ";".join(kp.report_ids[:3])  # 最初の3つのIDのみ
            line = f"{kp.score}|{kp.description}|{report_ids_str}"
            context_lines.append(line)
        
        return "\n".join(context_lines)
    
    def _build_traceability(self, key_points: List[KeyPoint]) -> TraceabilityInfo:
        """トレーサビリティ情報を構築"""
        all_report_ids = set()
        all_document_ids = set()
        all_chunk_ids = set()
        all_entity_ids = set()
        
        for kp in key_points:
            # レポートID
            all_report_ids.update(kp.report_ids)
            
            # ソースメタデータから情報を収集
            metadata = kp.source_metadata
            if metadata:
                all_document_ids.update(metadata.get("document_ids", []))
                all_chunk_ids.update(metadata.get("chunk_ids", []))
                all_entity_ids.update(metadata.get("entity_ids", []))
        
        return TraceabilityInfo(
            report_ids=list(all_report_ids),
            document_ids=list(all_document_ids),
            chunk_ids=list(all_chunk_ids),
            entity_ids=list(all_entity_ids)
        )
    
    def _create_fallback_response(self, key_points: List[KeyPoint], query: str) -> str:
        """LLM呼び出しが失敗した場合のフォールバック応答を作成"""
        response_lines = []
        response_lines.append(f"クエリ「{query}」に関する情報:")
        response_lines.append("")
        
        # 重要度順にキーポイントを列挙
        for i, kp in enumerate(key_points[:10], 1):
            response_lines.append(f"{i}. {kp.description}")
        
        response_lines.append("")
        response_lines.append("（注：この応答は自動生成されたものです）")
        
        return "\n".join(response_lines)
    
    def format_output(
        self,
        result: GlobalSearchResult,
        output_format: str = "markdown"
    ) -> Any:
        """
        結果を指定された形式でフォーマット
        
        Args:
            result: GlobalSearchResult
            output_format: "markdown" または "json"
        
        Returns:
            フォーマットされた出力
        """
        return result.format_output(output_format)