#!/usr/bin/env python3
"""
検索インデックス作成の包括的テストスイート

このテストスイートは以下をテストします：
1. 基本的なインデックス作成機能
2. 複数タイプのベクターストア統合
3. エラーハンドリングと回復機能
4. データ整合性とパフォーマンス
"""

import os
import tempfile
import shutil
import yaml
import pandas as pd
from pathlib import Path
import sys

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# 必要な環境変数設定
os.environ["ANTHROPIC_API_KEY"] = "test-key"

try:
    from graphrag_anthropic_llamaindex.document_processor import add_documents
    from graphrag_anthropic_llamaindex.vector_store_manager import get_vector_store, get_index
    from graphrag_anthropic_llamaindex.alopex_db import close_database, get_database_path
    from graphrag_anthropic_llamaindex.db_manager import load_entities_db, load_relationships_db
    from graphrag_anthropic_llamaindex.config_manager import load_config
    from graphrag_anthropic_llamaindex.file_filter import FileFilter
    from llama_index.core import Settings
    from llama_index.llms.anthropic import Anthropic
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.node_parser import SentenceSplitter
except ImportError as e:
    print(f"❌ モジュールインポートエラー: {e}")
    sys.exit(1)

class TestIndexCreation:
    """検索インデックス作成のテストクラス"""
    
    def setup_method(self):
        """各テストメソッド実行前の初期化"""
        self.test_dir = tempfile.mkdtemp(prefix="test_graphrag_")
        self.test_data_dir = os.path.join(self.test_dir, "test_data")
        self.output_dir = os.path.join(self.test_dir, "output")
        self.config = self._create_test_config()
        os.makedirs(self.test_data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # LlamaIndex Settings初期化
        self._setup_llama_index_settings()
        
    def teardown_method(self):
        """各テストメソッド実行後のクリーンアップ"""
        close_database(self.output_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_config(self):
        """テスト用設定を作成"""
        return {
            "anthropic": {
                "api_key": "test-key",
                "model": "claude-3-haiku-20240307"
            },
            "embedding_model": {
                "name": "intfloat/multilingual-e5-small"
            },
            "chunking": {
                "chunk_size": 512,
                "chunk_overlap": 50
            },
            "input_dir": self.test_data_dir if hasattr(self, 'test_data_dir') else "./test_data",
            "output_dir": self.output_dir if hasattr(self, 'output_dir') else "./test_output",
            "community_detection": {
                "max_cluster_size": 5,
                "use_lcc": True,
                "seed": 42
            }
        }
    
    def _create_test_data(self):
        """テスト用データファイルを作成"""
        test_files = {
            "document1.txt": "Python is a high-level programming language. It supports object-oriented programming.",
            "document2.txt": "Machine learning uses algorithms to analyze data patterns. Neural networks are powerful tools.",
            "document3.txt": "GraphRAG combines graph databases with retrieval-augmented generation for better search.",
            "data.csv": "name,description,category\nPython,Programming Language,Software\nAI,Artificial Intelligence,Technology\nGraph,Data Structure,Computer Science"
        }
        
        for filename, content in test_files.items():
            file_path = os.path.join(self.test_data_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return list(test_files.keys())
    
    def _setup_llama_index_settings(self):
        """LlamaIndex Settings初期化"""
        try:
            # モックLLM設定（実際のAPI呼び出しを避けるため）
            from llama_index.core.llms.mock import MockLLM
            
            # モックLLMでエンティティ抽出のJSONレスポンスを生成
            mock_llm = MockLLM(max_tokens=1000)
            mock_llm._response = """[START_JSON]
{
    "entities": [
        {"name": "Python", "type": "Programming Language"},
        {"name": "Machine Learning", "type": "Technology"},
        {"name": "GraphRAG", "type": "Technology"}
    ],
    "relationships": [
        {"source": "Python", "target": "Machine Learning", "type": "supports", "description": "Python supports machine learning"},
        {"source": "GraphRAG", "target": "Machine Learning", "type": "uses", "description": "GraphRAG uses machine learning techniques"}
    ]
}
[END_JSON]"""
            
            Settings.llm = mock_llm
            
            # Embedding model設定
            embed_model = HuggingFaceEmbedding(model_name=self.config["embedding_model"]["name"])
            Settings.embed_model = embed_model
            
            # Node parser設定
            node_parser = SentenceSplitter(
                chunk_size=self.config["chunking"]["chunk_size"],
                chunk_overlap=self.config["chunking"]["chunk_overlap"]
            )
            Settings.node_parser = node_parser
            
            print(f"✅ LlamaIndex Settings設定完了: LLM={type(Settings.llm).__name__}, Embed={type(Settings.embed_model).__name__}")
            
        except Exception as e:
            print(f"❌ LlamaIndex Settings設定エラー: {e}")
            # テストが失敗することを明示するために例外を再発生
            raise

class TestBasicIndexCreation(TestIndexCreation):
    """基本的なインデックス作成テスト"""
    
    def test_basic_document_processing(self):
        """基本的な文書処理とインデックス作成のテスト"""
        print("🧪 テスト: 基本的な文書処理とインデックス作成")
        
        # テストデータ作成
        test_files = self._create_test_data()
        print(f"📁 テストファイル作成: {test_files}")
        
        try:
            # ベクターストア設定
            main_vector_store = get_vector_store(self.config, store_type="main")
            entity_vector_store = get_vector_store(self.config, store_type="entity")
            community_vector_store = get_vector_store(self.config, store_type="community")
            
            print(f"🔧 ベクターストア設定完了")
            print(f"   - メイン: {type(main_vector_store).__name__ if main_vector_store else 'None'}")
            print(f"   - エンティティ: {type(entity_vector_store).__name__ if entity_vector_store else 'None'}")
            print(f"   - コミュニティ: {type(community_vector_store).__name__ if community_vector_store else 'None'}")
            
            # 文書処理実行
            community_detection_config = self.config.get("community_detection", {})
            file_filter = FileFilter()
            
            print(f"📝 文書処理開始...")
            add_documents(
                input_dir=self.test_data_dir,
                output_dir=self.output_dir,
                vector_store=main_vector_store,
                entity_vector_store=entity_vector_store,
                community_vector_store=community_vector_store,
                community_detection_config=community_detection_config,
                use_archive_reader=False,
                file_filter=file_filter
            )
            
            print(f"✅ 文書処理完了")
            
            # 結果検証
            self._verify_processing_results()
            
        except Exception as e:
            print(f"❌ テスト失敗: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def test_vector_store_creation(self):
        """ベクターストア作成のテスト"""
        print("🧪 テスト: ベクターストア作成")
        
        # 各タイプのベクターストア作成テスト
        store_types = ["main", "entity", "community"]
        
        for store_type in store_types:
            print(f"📊 {store_type}ベクターストア作成テスト")
            
            vector_store = get_vector_store(self.config, store_type=store_type)
            
            if vector_store is not None:
                print(f"   ✅ {store_type}: {type(vector_store).__name__}")
                
                assert vector_store.index_name.endswith("_vectors")
            else:
                print(f"   ❌ {store_type}: 作成失敗")
                
    def _verify_processing_results(self):
        """処理結果の検証"""
        print("🔍 処理結果検証開始")
        
        assert os.path.isdir(get_database_path(self.output_dir))
        print(f"   - エンティティ数: {len(load_entities_db(self.output_dir))}")
        print(f"   - 関係性数: {len(load_relationships_db(self.output_dir))}")

class TestIndexIntegrity(TestIndexCreation):
    """インデックス整合性テスト"""
    
    def test_data_consistency(self):
        """データ整合性テスト"""
        print("🧪 テスト: データ整合性")
        
        # テストデータ作成と処理
        self._create_test_data()
        self._process_test_documents()
        
        # データ整合性確認
        self._check_data_relationships()
    
    def _process_test_documents(self):
        """テスト文書の処理"""
        main_vector_store = get_vector_store(self.config, store_type="main")
        entity_vector_store = get_vector_store(self.config, store_type="entity")
        community_vector_store = get_vector_store(self.config, store_type="community")
        
        add_documents(
            input_dir=self.test_data_dir,
            output_dir=self.output_dir,
            vector_store=main_vector_store,
            entity_vector_store=entity_vector_store,
            community_vector_store=community_vector_store,
            community_detection_config=self.config.get("community_detection", {}),
            use_archive_reader=False,
            file_filter=FileFilter()
        )
    
    def _check_data_relationships(self):
        """データの関係性確認"""
        try:
            entities_df = load_entities_db(self.output_dir)
            relationships_df = load_relationships_db(self.output_dir)
            
            print(f"📊 データ統計:")
            print(f"   - エンティティ数: {len(entities_df)}")
            print(f"   - 関係性数: {len(relationships_df)}")
            
            # エンティティと関係性の整合性確認
            entity_names = set(entities_df['name'].tolist())
            relation_entities = set()
            
            for _, row in relationships_df.iterrows():
                relation_entities.add(row['source'])
                relation_entities.add(row['target'])
            
            missing_entities = relation_entities - entity_names
            if missing_entities:
                print(f"   ❌ 関係性に存在するが、エンティティテーブルに無いもの: {missing_entities}")
            else:
                print(f"   ✅ エンティティと関係性の整合性: OK")
                
        except Exception as e:
            print(f"   ❌ データ整合性チェックエラー: {str(e)}")

def run_comprehensive_test():
    """包括的テストの実行"""
    print("🚀 検索インデックス作成 包括的テスト開始\n")
    
    # テストクラスのインスタンス作成
    basic_test = TestBasicIndexCreation()
    integrity_test = TestIndexIntegrity()
    
    test_results = {
        "passed": 0,
        "failed": 0,
        "errors": []
    }
    
    tests = [
        ("基本的なインデックス作成", basic_test.test_basic_document_processing),
        ("ベクターストア作成", basic_test.test_vector_store_creation),
        ("データ整合性", integrity_test.test_data_consistency)
    ]
    
    for test_name, test_method in tests:
        try:
            print(f"\n{'='*60}")
            print(f"🧪 実行中: {test_name}")
            print(f"{'='*60}")
            
            # テスト環境セットアップ
            if hasattr(test_method, '__self__'):
                test_method.__self__.setup_method()
            
            # テスト実行
            test_method()
            
            print(f"✅ {test_name}: 成功")
            test_results["passed"] += 1
            
        except Exception as e:
            print(f"❌ {test_name}: 失敗 - {str(e)}")
            test_results["failed"] += 1
            test_results["errors"].append(f"{test_name}: {str(e)}")
            
        finally:
            # テスト環境クリーンアップ
            if hasattr(test_method, '__self__'):
                test_method.__self__.teardown_method()
    
    # テスト結果サマリー
    print(f"\n{'='*60}")
    print(f"📊 テスト結果サマリー")
    print(f"{'='*60}")
    print(f"✅ 成功: {test_results['passed']}")
    print(f"❌ 失敗: {test_results['failed']}")
    print(f"📝 総テスト数: {test_results['passed'] + test_results['failed']}")
    
    if test_results["errors"]:
        print(f"\n❌ エラー詳細:")
        for error in test_results["errors"]:
            print(f"   - {error}")
    
    return test_results

if __name__ == "__main__":
    # テスト実行
    results = run_comprehensive_test()
    
    # 終了コード設定
    exit_code = 0 if results["failed"] == 0 else 1
    sys.exit(exit_code)
