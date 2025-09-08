"""
Map処理の実装 - 並列LLM呼び出しとキーポイント抽出
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import json
import re

from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.bedrock import Bedrock

from .models import MapResult, KeyPoint
from .prompts import MAP_SYSTEM_PROMPT, MAP_USER_PROMPT

logger = logging.getLogger(__name__)


class MapProcessor:
    """Map処理を実行してコミュニティレポートからキーポイントを抽出"""
    
    def __init__(
        self,
        llm_config: Dict[str, Any],
        max_concurrent: int = 5,
        response_type: str = "multiple paragraphs"
    ):
        """
        Args:
            llm_config: LLM設定辞書
            max_concurrent: 最大同時実行数
            response_type: レスポンスタイプ
        """
        # LLMを初期化または既存のSettingsから取得
        self.llm = self._get_or_create_llm(llm_config)
        self.max_concurrent = max_concurrent
        self.response_type = response_type
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
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
        
    async def process_batch(
        self,
        batches: List[Dict[str, Any]],
        query: str
    ) -> List[MapResult]:
        """
        バッチを並列処理してMap結果を生成
        
        Args:
            batches: コンテキストバッチのリスト
            query: ユーザークエリ
        
        Returns:
            MapResultのリスト
        """
        # 並列タスクを作成
        tasks = []
        for i, batch in enumerate(batches):
            task = self._process_single_batch(batch, query, i)
            tasks.append(task)
        
        # 並列実行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # エラーチェックと結果の収集
        map_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"バッチ {i} の処理中にエラー: {result}")
                # 空の結果を返す
                map_results.append(MapResult(
                    batch_id=i,
                    key_points=[],
                    context_tokens=batches[i]["tokens"],
                    processing_time=0.0
                ))
            else:
                map_results.append(result)
        
        return map_results
    
    async def _process_single_batch(
        self,
        batch: Dict[str, Any],
        query: str,
        batch_id: int
    ) -> MapResult:
        """単一バッチを処理"""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # システムプロンプトを構築
                system_prompt = MAP_SYSTEM_PROMPT.format(
                    response_type=self.response_type
                )
                
                # ユーザープロンプトを構築
                user_prompt = MAP_USER_PROMPT.format(
                    context=batch["context"],
                    query=query
                )
                
                # LLM呼び出し
                response = await self._call_llm_async(system_prompt, user_prompt)
                
                # キーポイントを抽出
                key_points = self.extract_key_points(
                    response,
                    batch["report_ids"],
                    batch["records"]
                )
                
                processing_time = time.time() - start_time
                
                return MapResult(
                    batch_id=batch_id,
                    key_points=key_points,
                    context_tokens=batch["tokens"],
                    processing_time=processing_time
                )
                
            except Exception as e:
                logger.error(f"バッチ {batch_id} の処理中にエラー: {e}")
                raise
    
    async def _call_llm_async(self, system_prompt: str, user_prompt: str) -> str:
        """非同期でLLMを呼び出し"""
        # LlamaIndexのLLMは同期的なので、run_in_executorを使用
        loop = asyncio.get_event_loop()
        
        def _sync_call():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.llm.chat(messages)
            return str(response)
        
        return await loop.run_in_executor(None, _sync_call)
    
    def extract_key_points(
        self,
        llm_response: str,
        report_ids: List[str],
        records: List[Dict[str, Any]]
    ) -> List[KeyPoint]:
        """
        LLMレスポンスからキーポイントを抽出
        
        Args:
            llm_response: LLMの応答テキスト
            report_ids: レポートIDのリスト
            records: レコードのリスト
        
        Returns:
            KeyPointのリスト
        """
        key_points = []
        
        # JSON形式のレスポンスを試みる
        try:
            # JSONブロックを探す
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                
                if isinstance(data, dict) and "key_points" in data:
                    points = data["key_points"]
                elif isinstance(data, list):
                    points = data
                else:
                    points = []
                
                for point in points:
                    if isinstance(point, dict):
                        key_point = KeyPoint(
                            description=point.get("description", ""),
                            score=point.get("score", 50),
                            report_ids=point.get("report_ids", report_ids[:3]),  # 上位3つのレポートIDを使用
                            source_metadata=self._extract_metadata(records, point.get("report_ids", []))
                        )
                        key_points.append(key_point)
            else:
                # JSON形式でない場合は、段落ごとに処理
                key_points = self._extract_from_text(llm_response, report_ids, records)
                
        except json.JSONDecodeError:
            # JSONパースに失敗した場合は、テキスト形式として処理
            key_points = self._extract_from_text(llm_response, report_ids, records)
        
        return key_points
    
    def _extract_from_text(
        self,
        text: str,
        report_ids: List[str],
        records: List[Dict[str, Any]]
    ) -> List[KeyPoint]:
        """テキスト形式のレスポンスからキーポイントを抽出"""
        key_points = []
        
        # 段落またはリストアイテムごとに分割
        paragraphs = re.split(r'\n\n+', text.strip())
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) > 20:  # 短すぎる段落は無視
                # スコアを推定（最初の方が重要と仮定）
                score = max(100 - (i * 10), 50)
                
                # 箇条書きを処理
                if paragraph.startswith(('- ', '* ', '• ', '1. ', '2. ', '3. ')):
                    # 箇条書きの各項目を処理
                    items = re.findall(r'[-*•\d.]\s+(.+)', paragraph)
                    for item in items:
                        if len(item.strip()) > 20:
                            key_point = KeyPoint(
                                description=item.strip(),
                                score=score,
                                report_ids=report_ids[:3],
                                source_metadata=self._extract_metadata(records, report_ids[:3])
                            )
                            key_points.append(key_point)
                else:
                    # 通常の段落
                    key_point = KeyPoint(
                        description=paragraph.strip(),
                        score=score,
                        report_ids=report_ids[:3],
                        source_metadata=self._extract_metadata(records, report_ids[:3])
                    )
                    key_points.append(key_point)
        
        return key_points
    
    def _extract_metadata(
        self,
        records: List[Dict[str, Any]],
        report_ids: List[str]
    ) -> Dict[str, Any]:
        """レコードからメタデータを抽出"""
        metadata = {
            "document_ids": [],
            "chunk_ids": [],
            "entity_ids": []
        }
        
        for record in records:
            if record.get("id") in report_ids:
                record_metadata = record.get("metadata", {})
                
                # ドキュメントID
                if "document_id" in record_metadata:
                    metadata["document_ids"].append(record_metadata["document_id"])
                
                # チャンクID
                if "chunk_id" in record_metadata:
                    metadata["chunk_ids"].append(record_metadata["chunk_id"])
                
                # エンティティID
                if "entity_ids" in record_metadata:
                    metadata["entity_ids"].extend(record_metadata["entity_ids"])
        
        # 重複を削除
        metadata["document_ids"] = list(set(metadata["document_ids"]))
        metadata["chunk_ids"] = list(set(metadata["chunk_ids"]))
        metadata["entity_ids"] = list(set(metadata["entity_ids"]))
        
        return metadata