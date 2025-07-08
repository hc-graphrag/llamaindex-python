import os
import asyncio
import logging
from typing import Optional, Tuple
import gradio as gr

from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

from src.graphrag_anthropic_llamaindex.config_manager import load_config
from src.graphrag_anthropic_llamaindex.vector_store_manager import get_vector_store
from src.graphrag_anthropic_llamaindex.document_processor import add_documents
from src.graphrag_anthropic_llamaindex.search_processor import search_index
from src.graphrag_anthropic_llamaindex.file_filter import FileFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGApp:
    def __init__(self):
        self.config = None
        self.llm_params = {}
        self.vector_stores = {}
        self.file_filter = None
        self.is_initialized = False
    
    def initialize_config(self, config_path: str = "config.yaml") -> str:
        try:
            self.config = load_config(config_path)
            if not self.config:
                return "❌ 設定ファイルの読み込みに失敗しました"
            
            # Initialize Anthropic settings
            anthropic_config = self.config.get("anthropic", {})
            if not anthropic_config.get("api_key"):
                return "❌ Anthropic APIキーが設定されていません"
                
            os.environ["ANTHROPIC_API_KEY"] = anthropic_config["api_key"]
            model_name = anthropic_config.get("model", "claude-3-opus-20240229")
            api_base_url = anthropic_config.get("api_base_url")
            
            self.llm_params = {"model": model_name}
            if api_base_url:
                self.llm_params["api_base_url"] = api_base_url
            
            # Initialize vector stores
            self.vector_stores = {
                "main": get_vector_store(self.config, store_type="main"),
                "entity": get_vector_store(self.config, store_type="entity"),
                "community": get_vector_store(self.config, store_type="community")
            }
            
            # Configure embedding model
            embedding_config = self.config.get("embedding_model", {})
            embed_model_name = embedding_config.get("name", "intfloat/multilingual-e5-small")
            embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
            
            # Configure chunking
            chunking_config = self.config.get("chunking", {})
            chunk_size = chunking_config.get("chunk_size", 1024)
            chunk_overlap = chunking_config.get("chunk_overlap", 20)
            node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Configure Settings
            llm = Anthropic(**self.llm_params)
            Settings.llm = llm
            Settings.embed_model = embed_model
            Settings.node_parser = node_parser
            
            # Initialize file filter
            ignore_patterns = self.config.get("ignore_patterns", [])
            self.file_filter = FileFilter(ignore_patterns)
            
            self.is_initialized = True
            logger.info("Configuration initialized successfully")
            return "✅ 設定が正常に読み込まれました"
            
        except Exception as e:
            error_msg = f"❌ 設定の初期化に失敗しました: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def add_documents_sync(self, input_dir: str, output_dir: str, progress=gr.Progress()) -> str:
        if not self.is_initialized:
            return "❌ 設定が初期化されていません。まず設定を読み込んでください。"
        
        if not input_dir or not output_dir:
            return "❌ 入力ディレクトリと出力ディレクトリを指定してください。"
        
        try:
            progress(0, desc="ドキュメント処理を開始...")
            logger.info(f"Adding documents: input_dir={input_dir}, output_dir={output_dir}")
            
            community_detection_config = self.config.get("community_detection", {})
            
            progress(0.5, desc="ドキュメントを処理中...")
            add_documents(
                input_dir,
                output_dir,
                self.vector_stores["main"],
                self.vector_stores["entity"],
                self.vector_stores["community"],
                community_detection_config,
                True,  # use_archive_reader
                self.file_filter
            )
            
            progress(1.0, desc="完了")
            result = f"✅ ドキュメントが正常に追加されました\n入力: {input_dir}\n出力: {output_dir}"
            logger.info("Documents added successfully")
            return result
            
        except Exception as e:
            error_msg = f"❌ ドキュメント追加に失敗しました: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def search_chat(self, message: str, history: list, search_method: str, output_dir: str, progress=gr.Progress()) -> tuple:
        if not self.is_initialized:
            error_msg = "❌ 設定が初期化されていません。まず設定タブで設定を読み込んでください。"
            history.append([message, error_msg])
            return "", history
        
        if not message.strip():
            error_msg = "❌ 検索クエリを入力してください。"
            history.append([message, error_msg])
            return "", history
        
        if not output_dir:
            error_msg = "❌ 出力ディレクトリを指定してください。"
            history.append([message, error_msg])
            return "", history
        
        try:
            progress(0, desc="検索を開始...")
            logger.info(f"Searching: query='{message}', search_method={search_method}")
            
            progress(0.5, desc="検索中...")
            result = search_index(
                message,
                output_dir,
                self.llm_params,
                self.vector_stores["main"],
                self.vector_stores["entity"],
                self.vector_stores["community"],
                search_method
            )
            
            progress(1.0, desc="完了")
            logger.info("Search completed successfully")
            
            # Format the response for chat - Anthropic Claude style
            response = f"🔍 **検索方法**: {self._get_search_method_name(search_method)}\n\n{result}"
            history.append([message, response])
            
            return "", history
            
        except Exception as e:
            error_msg = f"❌ 検索に失敗しました: {str(e)}"
            logger.error(error_msg)
            history.append([message, error_msg])
            return "", history
    
    def _get_search_method_name(self, search_method: str) -> str:
        method_names = {
            "both": "統合検索（エンティティ + コミュニティ）",
            "main": "メイン検索（基本的な意味検索）",
            "entity": "エンティティ検索（固有名詞・概念中心）",
            "community": "コミュニティ検索（関連性・グループ中心）"
        }
        return method_names.get(search_method, search_method)

# Global app instance
app = GraphRAGApp()

def create_interface():
    with gr.Blocks(title="GraphRAG Anthropic LlamaIndex", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔗 GraphRAG Anthropic LlamaIndex Web App")
        gr.Markdown("CLIツールのためのシンプルなWebインターフェース")
        
        with gr.Tab("⚙️ 設定"):
            gr.Markdown("### 設定ファイルの読み込み")
            config_path = gr.Textbox(
                label="設定ファイルパス",
                value="config.yaml",
                placeholder="config.yamlファイルのパスを入力"
            )
            config_btn = gr.Button("設定を読み込み", variant="primary")
            config_status = gr.Textbox(
                label="設定状況",
                interactive=False,
                lines=3
            )
            
            config_btn.click(
                fn=app.initialize_config,
                inputs=[config_path],
                outputs=[config_status]
            )
        
        with gr.Tab("📄 ドキュメント追加"):
            gr.Markdown("### ドキュメントをインデックスに追加")
            
            with gr.Row():
                input_dir = gr.Textbox(
                    label="入力ディレクトリ",
                    value="./data",
                    placeholder="ドキュメントが格納されているディレクトリ"
                )
                output_dir = gr.Textbox(
                    label="出力ディレクトリ",
                    value="./graphrag_output",
                    placeholder="処理結果の出力先ディレクトリ"
                )
            
            add_btn = gr.Button("ドキュメントを追加", variant="primary")
            add_result = gr.Textbox(
                label="処理結果",
                interactive=False,
                lines=5
            )
            
            add_btn.click(
                fn=app.add_documents_sync,
                inputs=[input_dir, output_dir],
                outputs=[add_result]
            )
        
        with gr.Tab("🔍 検索"):
            gr.Markdown("### 💬 チャット形式で検索")
            
            with gr.Row():
                with gr.Column(scale=3):
                    search_output_dir = gr.Textbox(
                        label="出力ディレクトリ",
                        value="./graphrag_output",
                        placeholder="インデックスが格納されているディレクトリ"
                    )
                with gr.Column(scale=2):
                    search_method = gr.Dropdown(
                        label="検索方法",
                        choices=[
                            ("統合検索（推奨）", "both"),
                            ("メイン検索", "main"), 
                            ("エンティティ検索", "entity"),
                            ("コミュニティ検索", "community")
                        ],
                        value="both",
                        info="検索の方法を選択してください"
                    )
            
            # Chat interface
            chatbot = gr.Chatbot(
                value=[],
                label="検索チャット",
                show_label=True,
                height=400,
                show_copy_button=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="メッセージ",
                    placeholder="質問や検索したい内容を入力してください...",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("送信", variant="primary", scale=1)
            
            # Clear chat button
            clear_btn = gr.Button("チャット履歴をクリア", variant="secondary")
            
            # Event handlers
            def send_message(message, history, method, output_dir):
                return app.search_chat(message, history, method, output_dir)
            
            def clear_chat():
                return []
            
            # Send message on button click or Enter
            send_btn.click(
                fn=send_message,
                inputs=[msg, chatbot, search_method, search_output_dir],
                outputs=[msg, chatbot]
            )
            
            msg.submit(
                fn=send_message,
                inputs=[msg, chatbot, search_method, search_output_dir],
                outputs=[msg, chatbot]
            )
            
            # Clear chat
            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot]
            )
        
        with gr.Tab("ℹ️ 使用方法"):
            gr.Markdown("""
            ## 📖 使用手順
            
            1. **設定タブ**: まず設定ファイル（config.yaml）を読み込んでください
            2. **ドキュメント追加タブ**: ドキュメントをインデックスに追加します
            3. **検索タブ**: チャット形式で追加されたドキュメントから検索を実行します
            
            ## 🔍 検索方法の説明
            
            - **統合検索（推奨）**: エンティティとコミュニティ検索を組み合わせた総合的な検索
            - **メイン検索**: 基本的な意味検索（単語の類似性ベース）
            - **エンティティ検索**: 固有名詞や重要な概念に焦点を当てた検索
            - **コミュニティ検索**: 関連性の高いトピックグループから検索
            
            ## 📁 ディレクトリ構造
            
            ```
            ./data/              # 入力ドキュメント
            ./graphrag_output/   # 処理結果・インデックス
            config.yaml          # 設定ファイル
            ```
            
            ## ⚙️ 設定ファイル例
            
            ```yaml
            anthropic:
              api_key: "your-api-key"
              model: "claude-3-opus-20240229"
            
            input_dir: "./data"
            output_dir: "./graphrag_output"
            
            embedding_model:
              name: "intfloat/multilingual-e5-small"
            
            chunking:
              chunk_size: 1024
              chunk_overlap: 20
            
            ignore_patterns:
              - "*.tmp"
              - ".git/*"
            ```
            
            ## 💬 チャット検索の使い方
            
            1. 検索方法を選択（通常は「統合検索」がおすすめ）
            2. チャット欄に自然な日本語で質問を入力
            3. Enterキーまたは「送信」ボタンで検索実行
            4. 結果がチャット形式で表示されます
            5. 続けて追加の質問も可能です
            """)
    
    return interface

def main():
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()