# 設計書

## 概要

DRIFT Search（Dynamic Retrieval with Interactive Filtering and Transformations）は、ローカル検索とグローバル検索の長所を組み合わせた高度な検索メカニズムです。この設計は、MS GraphRAGの実装を参考にしながら、LlamaIndexフレームワークとの統合を考慮したアーキテクチャを提供します。

## 技術アーキテクチャ

### システムコンポーネント

```
┌─────────────────────────────────────────────────────┐
│                  DRIFT Search Engine                  │
├───────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────────┐    │
│  │  Query Processor │───▶│  Context Builder     │    │
│  └─────────────────┘    └──────────────────────┘    │
│           │                       │                   │
│           ▼                       ▼                   │
│  ┌─────────────────┐    ┌──────────────────────┐    │
│  │  Local Search    │    │  Global Search       │    │
│  └─────────────────┘    └──────────────────────┘    │
│           │                       │                   │
│           ▼                       ▼                   │
│  ┌─────────────────────────────────────────────┐    │
│  │         Response Generator                   │    │
│  └─────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Vector Stores    │
                  ├─────────────────┤
                  │ • Main Index     │
                  │ • Entity Index   │
                  │ • Community Index│
                  └─────────────────┘
```

### クラス設計

#### 1. DriftSearchEngine

```python
class DriftSearchEngine:
    """DRIFT検索のメインエンジン"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        vector_stores: Dict[str, VectorStore],
        llm: Optional[Any] = None
    ):
        """
        Args:
            config: DRIFT検索設定
            vector_stores: ベクターストアのマッピング
            llm: 使用するLLMインスタンス
        """
        self.config = config
        self.vector_stores = vector_stores
        self.llm = llm or Settings.llm
        self.local_searcher = LocalSearcher(vector_stores)
        self.global_searcher = GlobalSearcher(vector_stores)
        self.context_builder = ContextBuilder()
        self.response_generator = ResponseGenerator(llm)
    
    async def search(
        self,
        query: str,
        streaming: bool = False,
        include_context: bool = True
    ) -> Union[str, AsyncGenerator[str, None], Tuple[str, Dict]]:
        """DRIFT検索の実行"""
        pass
```

#### 2. LocalSearcher

```python
class LocalSearcher:
    """ローカルエンティティ検索"""
    
    def __init__(self, vector_stores: Dict[str, VectorStore]):
        self.entity_store = vector_stores.get("entity")
        self.main_store = vector_stores.get("main")
    
    async def search_entities(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Entity]:
        """関連エンティティの検索"""
        pass
    
    async def expand_context(
        self,
        entities: List[Entity],
        max_hops: int = 2
    ) -> List[Entity]:
        """エンティティ関係の展開"""
        pass
```

#### 3. GlobalSearcher

```python
class GlobalSearcher:
    """グローバルコミュニティ検索"""
    
    def __init__(self, vector_stores: Dict[str, VectorStore]):
        self.community_store = vector_stores.get("community")
    
    async def search_communities(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Community]:
        """関連コミュニティの検索"""
        pass
    
    async def get_community_summaries(
        self,
        communities: List[Community]
    ) -> List[str]:
        """コミュニティサマリの取得"""
        pass
```

#### 4. ContextBuilder

```python
class ContextBuilder:
    """コンテキスト構築"""
    
    def build_search_context(
        self,
        query: str,
        local_results: List[Entity],
        global_results: List[Community]
    ) -> SearchContext:
        """検索コンテキストの構築"""
        pass
    
    def prioritize_context(
        self,
        context: SearchContext,
        max_tokens: int = 8000
    ) -> SearchContext:
        """コンテキストの優先順位付けとトリミング"""
        pass
```

#### 5. ResponseGenerator

```python
class ResponseGenerator:
    """レスポンス生成"""
    
    def __init__(self, llm: Any):
        self.llm = llm
        self.prompt_builder = PromptBuilder()
    
    async def generate_response(
        self,
        context: SearchContext,
        streaming: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """最終レスポンスの生成"""
        pass
    
    async def stream_response(
        self,
        context: SearchContext
    ) -> AsyncGenerator[str, None]:
        """ストリーミングレスポンスの生成"""
        pass
```

### データモデル

#### Entity
```python
@dataclass
class Entity:
    id: str
    name: str
    type: str
    description: str
    attributes: Dict[str, Any]
    relationships: List[Relationship]
    embedding: Optional[np.ndarray]
```

#### Community
```python
@dataclass
class Community:
    id: str
    title: str
    summary: str
    entities: List[str]
    level: int
    embedding: Optional[np.ndarray]
```

#### SearchContext
```python
@dataclass
class SearchContext:
    query: str
    entities: List[Entity]
    communities: List[Community]
    text_units: List[TextUnit]
    metadata: Dict[str, Any]
```

## API設計

### REST API エンドポイント

```yaml
/api/drift/search:
  post:
    summary: DRIFT検索の実行
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              query:
                type: string
                description: 検索クエリ
              streaming:
                type: boolean
                default: false
                description: ストリーミングレスポンスの有効化
              include_context:
                type: boolean
                default: true
                description: コンテキストデータの包含
              parameters:
                type: object
                properties:
                  local_search_limit:
                    type: integer
                    default: 10
                  global_search_limit:
                    type: integer
                    default: 5
                  max_tokens:
                    type: integer
                    default: 8000
    responses:
      200:
        description: 検索結果
        content:
          application/json:
            schema:
              type: object
              properties:
                response:
                  type: string
                context:
                  type: object
```

### Python API

```python
# 基本的な使用法
drift_search = DriftSearchEngine(config, vector_stores)
response = await drift_search.search("What are the main themes?")

# ストリーミング
async for chunk in await drift_search.search(query, streaming=True):
    print(chunk, end="")

# コンテキスト付き
response, context = await drift_search.search(
    query,
    include_context=True
)
```

## 実装戦略

### フェーズ1: 基本実装
1. DriftSearchEngineクラスの基本構造
2. LocalSearcherの実装
3. GlobalSearcherの実装
4. 単純なResponseGeneratorの実装

### フェーズ2: 高度な機能
1. ContextBuilderの実装
2. エンティティ関係の展開
3. コミュニティ階層の処理
4. プロンプトテンプレートシステム

### フェーズ3: 最適化
1. ストリーミングレスポンス
2. 非同期処理の最適化
3. キャッシング戦略
4. バッチ処理

### フェーズ4: 統合
1. CLI統合
2. Web UI統合
3. API エンドポイント
4. 設定管理

## 設定仕様

```yaml
drift_search:
  enabled: true
  
  local_search:
    entity_top_k: 10
    relationship_depth: 2
    include_text_units: true
    text_unit_top_k: 5
  
  global_search:
    community_top_k: 5
    include_summaries: true
    max_summary_length: 500
  
  context:
    max_tokens: 8000
    prioritization_strategy: "relevance"  # relevance, recency, mixed
    include_metadata: true
  
  response:
    temperature: 0.7
    max_tokens: 2000
    streaming_enabled: true
    chunk_size: 50
  
  prompts:
    local_search_prompt: "prompts/drift_local_search.txt"
    global_search_prompt: "prompts/drift_global_search.txt"
    final_response_prompt: "prompts/drift_final_response.txt"
```

## パフォーマンス最適化

### インデックス戦略
- エンティティ埋め込みの事前計算
- コミュニティサマリの事前生成
- 関係グラフのインメモリキャッシュ

### 並列処理
- ローカル検索とグローバル検索の並列実行
- 複数エンティティの並列埋め込み検索
- バッチ処理による効率化

### キャッシング
- クエリ結果のキャッシング（TTL: 1時間）
- エンティティ関係の展開結果のキャッシング
- LLMレスポンスのキャッシング（類似クエリ）

## エラーハンドリング

### 段階的劣化
```python
try:
    # プライマリ検索
    results = await primary_search()
except VectorStoreError:
    # フォールバック検索
    results = await fallback_search()
except Exception as e:
    # 基本検索
    results = await basic_text_search()
```

### エラータイプ
- `VectorStoreError`: ベクターストアアクセスエラー
- `LLMError`: LLM呼び出しエラー
- `TimeoutError`: タイムアウトエラー
- `ValidationError`: 入力検証エラー

## テスト戦略

### ユニットテスト
- 各コンポーネントの独立テスト
- モックを使用した統合テスト
- エッジケースのテスト

### 統合テスト
- エンドツーエンドの検索フロー
- ストリーミングレスポンステスト
- 並列処理テスト

### パフォーマンステスト
- レスポンス時間の測定
- メモリ使用量の監視
- 同時リクエストの処理

## セキュリティ考慮事項

### 入力検証
- SQLインジェクション対策
- プロンプトインジェクション対策
- サイズ制限の実装

### データサニタイゼーション
- レスポンスのHTMLエスケープ
- 個人情報のマスキング
- URLとパスの検証

### アクセス制御
- APIキーによる認証
- レート制限の実装
- 監査ログの記録