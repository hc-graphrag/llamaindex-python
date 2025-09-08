"""
GLOBAL検索用のデータモデル定義
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json


@dataclass
class KeyPoint:
    """Map処理で抽出されるキーポイント"""
    description: str
    score: int  # 0-100の重要度スコア
    report_ids: List[str]
    source_metadata: Dict[str, Any]  # document_id, chunk_id, entity_idsなど
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)
    
    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class MapResult:
    """Map処理の結果"""
    batch_id: int
    key_points: List[KeyPoint]
    context_tokens: int
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "batch_id": self.batch_id,
            "key_points": [kp.to_dict() for kp in self.key_points],
            "context_tokens": self.context_tokens,
            "processing_time": self.processing_time
        }
    
    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class TraceabilityInfo:
    """トレーサビリティ情報"""
    report_ids: List[str]
    document_ids: List[str]
    chunk_ids: List[str]
    entity_ids: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)
    
    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class GlobalSearchResult:
    """GLOBAL検索の最終結果"""
    response: str
    response_type: str
    map_results: List[MapResult]
    traceability: TraceabilityInfo
    total_tokens: int
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "response": self.response,
            "response_type": self.response_type,
            "map_results": [mr.to_dict() for mr in self.map_results],
            "traceability": self.traceability.to_dict(),
            "total_tokens": self.total_tokens,
            "processing_time": self.processing_time
        }
    
    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def format_output(self, output_format: str = "markdown") -> Any:
        """指定された形式で出力をフォーマット
        
        Args:
            output_format: "markdown" または "json"
        
        Returns:
            フォーマットされた出力
        """
        if output_format == "json":
            return self.to_dict()
        elif output_format == "markdown":
            # マークダウン形式で出力
            output = f"# GLOBAL Search Result\n\n"
            output += f"## Response\n\n{self.response}\n\n"
            output += f"## Metadata\n\n"
            output += f"- Response Type: {self.response_type}\n"
            output += f"- Total Tokens: {self.total_tokens}\n"
            output += f"- Processing Time: {self.processing_time:.2f}s\n"
            output += f"- Total Key Points: {sum(len(mr.key_points) for mr in self.map_results)}\n"
            output += f"\n## Traceability\n\n"
            output += f"- Report IDs: {len(self.traceability.report_ids)}\n"
            output += f"- Document IDs: {len(self.traceability.document_ids)}\n"
            output += f"- Chunk IDs: {len(self.traceability.chunk_ids)}\n"
            output += f"- Entity IDs: {len(self.traceability.entity_ids)}\n"
            return output
        else:
            raise ValueError(f"Unsupported output format: {output_format}")