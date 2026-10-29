import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import gradio as gr
from dotenv import load_dotenv

from llama_index.llms.anthropic import Anthropic
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import QueryBundle

from src.graphrag_anthropic_llamaindex.claude_code_llm import ClaudeCodeLLM
from src.graphrag_anthropic_llamaindex.config_manager import load_config
from src.graphrag_anthropic_llamaindex.vector_store_manager import get_vector_store
from src.graphrag_anthropic_llamaindex.global_search import SearchModeRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGApp:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.config = None
        self.llm_params = {}
        self.vector_stores = {}
        self.is_initialized = False
        self.llm_provider = None
    
    def initialize_config(self, config_path: str = "config/config.yaml") -> str:
        try:
            # UI からの任意パス読み込みを作業ディレクトリ配下に限定する
            # (/etc/passwd やマウント済み認証ファイルの読み出しを防ぐ)
            base_dir = Path(os.environ.get("GRAPHRAG_CONFIG_BASE", os.getcwd())).resolve()
            requested = Path(config_path).resolve()
            if requested != base_dir and base_dir not in requested.parents:
                logger.warning("許可範囲外の設定ファイルパスが指定されました: %s", config_path)
                return "❌ 設定ファイルパスは作業ディレクトリ配下を指定してください"
            self.config = load_config(str(requested))
            if not self.config:
                return "❌ 設定ファイルの読み込みに失敗しました"
            
            # LLMプロバイダーの設定を取得
            self.llm_provider = self.config.get("llm_provider", "anthropic")
            
            if self.llm_provider == "claude_code":
                # Claude Code (claude CLI) のサブスクリプション認証 - APIキー不要
                model_name = self.config.get("claude_code", {}).get("model", "sonnet")
                self.llm_params = {"model": model_name}
            elif self.llm_provider == "bedrock":
                # AWS Bedrock設定 - ANTHROPIC_API_KEYは不要
                bedrock_config = self.config.get("bedrock", {})
                model_name = bedrock_config.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
                region_name = bedrock_config.get("region", "us-east-1")
                aws_access_key_id = bedrock_config.get("aws_access_key_id")
                aws_secret_access_key = bedrock_config.get("aws_secret_access_key")
                aws_session_token = bedrock_config.get("aws_session_token")
                
                # Bedrock用のLLMパラメータ
                self.llm_params = {
                    "model": model_name,
                    "region_name": region_name,
                }
                if aws_access_key_id:
                    self.llm_params["aws_access_key_id"] = aws_access_key_id
                if aws_secret_access_key:
                    self.llm_params["aws_secret_access_key"] = aws_secret_access_key
                if aws_session_token:
                    self.llm_params["aws_session_token"] = aws_session_token
            else:
                # Anthropic直接設定 - API_KEYが必要
                anthropic_config = self.config.get("anthropic", {})
                # 環境変数からのみ取得
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    return "❌ ANTHROPIC_API_KEY が環境変数に設定されていません。\nexport ANTHROPIC_API_KEY='your-api-key' で設定するか、\nconfig/config.yaml で llm_provider: 'claude_code' (Claudeサブスクリプション) または 'bedrock' を設定してください。"
                    
                model_name = anthropic_config.get("model", "claude-sonnet-5-5")
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
            
            # Configure Settings based on provider
            if self.llm_provider == "claude_code":
                llm = ClaudeCodeLLM(**self.llm_params)
                Settings.llm = llm
                logger.info(f"Using Claude Code (subscription login) with model: {model_name}")
            elif self.llm_provider == "bedrock":
                llm = BedrockConverse(**self.llm_params)
                Settings.llm = llm
                logger.info(f"Using AWS Bedrock with model: {model_name}")
            else:
                llm = Anthropic(**self.llm_params)
                Settings.llm = llm
                logger.info(f"Using Anthropic API with model: {model_name}")
            
            Settings.embed_model = embed_model
            Settings.node_parser = node_parser
            
            self.is_initialized = True
            logger.info("Configuration initialized successfully")
            return f"✅ 設定が正常に読み込まれました\nLLMプロバイダー: {self.llm_provider}\nモデル: {model_name}"
            
        except Exception as e:
            error_msg = f"❌ 設定の初期化に失敗しました: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    
    def search_chat(self, message: str, history: list, search_mode: str, response_type: str, output_format: str, min_community_rank: int, progress=gr.Progress()) -> tuple:
        if not self.is_initialized:
            error_msg = "❌ 設定が初期化されていません。まず設定タブで設定を読み込んでください。"
            history.extend(self._chat_turn(message, error_msg))
            return "", history
        
        if not message.strip():
            error_msg = "❌ 検索クエリを入力してください。"
            history.extend(self._chat_turn(message, error_msg))
            return "", history
        
        try:
            progress(0, desc="検索を開始...")
            logger.info(f"Searching: query='{message}', search_mode={search_mode}, response_type={response_type}")
            
            progress(0.5, desc="検索中...")
            
            # Use the new SearchModeRouter for unified search interface
            router = SearchModeRouter(
                config=self.config,
                mode=search_mode,
                vector_store_main=self.vector_stores.get("main"),
                vector_store_entity=self.vector_stores.get("entity"),
                vector_store_community=self.vector_stores.get("community"),
                response_type=response_type,
                min_community_rank=min_community_rank,
                output_format=output_format
            )
            
            # Execute search
            query_bundle = QueryBundle(query_str=message)
            results = router._retrieve(query_bundle)
            
            progress(1.0, desc="完了")
            logger.info("Search completed successfully")
            
            # Format results
            if results:
                if output_format == "json":
                    # JSON形式で結果を整形
                    result_data = []
                    for i, node_with_score in enumerate(results):
                        result_data.append({
                            "score": node_with_score.score,
                            "text": node_with_score.node.text,
                            "metadata": node_with_score.node.metadata
                        })
                    response = f"🔍 **検索モード**: {self._get_search_mode_name(search_mode)}\n\n```json\n{json.dumps(result_data, ensure_ascii=False, indent=2)}\n```"
                else:
                    # Markdown形式で結果を整形
                    main_result = results[0].node.text if results else "結果が見つかりませんでした。"
                    response = f"🔍 **検索モード**: {self._get_search_mode_name(search_mode)}\n\n{main_result}"
                    
                    # 追加のキーポイントがある場合
                    if len(results) > 1:
                        response += "\n\n### 📌 キーポイント\n"
                        for i, node_with_score in enumerate(results[1:], 1):
                            response += f"\n**ポイント {i}** (スコア: {node_with_score.score:.2f})\n{node_with_score.node.text}\n"
            else:
                response = f"🔍 **検索モード**: {self._get_search_mode_name(search_mode)}\n\n結果が見つかりませんでした。"
            
            history.extend(self._chat_turn(message, response))
            return "", history
            
        except Exception as e:
            error_msg = f"❌ 検索に失敗しました: {str(e)}"
            logger.error(error_msg, exc_info=True)
            history.extend(self._chat_turn(message, error_msg))
            return "", history
    
    @staticmethod
    def _chat_turn(message: str, response: str) -> list:
        """Chatbot (messages 形式) に追加する1往復分のメッセージ。"""
        return [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]

    def _get_search_mode_name(self, search_mode: str) -> str:
        mode_names = {
            "local": "ローカル検索（詳細・高精度）",
            "global": "グローバル検索（包括的・要約）",
            "drift": "DRIFT検索（次世代・実験的）",
            "auto": "自動選択（クエリに最適な方法を自動選択）"
        }
        return mode_names.get(search_mode, search_mode)

# Global app instance
app = GraphRAGApp()

def create_interface():
    with gr.Blocks(title="GraphRAG Anthropic LlamaIndex") as interface:
        gr.Markdown("# 🔗 GraphRAG Anthropic LlamaIndex Web App")
        gr.Markdown("CLIで作成されたインデックスを検索するためのWebインターフェース")
        
        with gr.Tab("⚙️ 設定"):
            gr.Markdown("### 設定ファイルの読み込み")
            config_path = gr.Textbox(
                label="設定ファイルパス",
                value="config/config.yaml",
                placeholder="config/config.yamlファイルのパスを入力"
            )
            config_btn = gr.Button("設定を読み込み", variant="primary")
            config_status = gr.Textbox(
                label="設定状況",
                interactive=False,
                lines=5
            )
            
            config_btn.click(
                fn=app.initialize_config,
                inputs=[config_path],
                outputs=[config_status]
            )
        
        with gr.Tab("🔍 検索"):
            gr.Markdown("### 💬 チャット形式で検索")
            
            with gr.Row():
                with gr.Column(scale=2):
                    search_mode = gr.Dropdown(
                        label="検索モード",
                        choices=[
                            ("グローバル検索（包括的）", "global"),
                            ("ローカル検索（詳細）", "local"), 
                            ("DRIFT検索（実験的）", "drift"),
                            ("自動選択", "auto")
                        ],
                        value="global",
                        info="検索モードを選択してください"
                    )
                with gr.Column(scale=2):
                    response_type = gr.Dropdown(
                        label="回答タイプ",
                        choices=[
                            ("複数段落", "multiple paragraphs"),
                            ("単一段落", "single paragraph"),
                            ("リスト形式", "list"),
                            ("要点のみ", "key points")
                        ],
                        value="multiple paragraphs",
                        info="回答の形式を選択"
                    )
            
            with gr.Row():
                with gr.Column(scale=2):
                    output_format = gr.Radio(
                        label="出力形式",
                        choices=[("Markdown", "markdown"), ("JSON", "json")],
                        value="markdown",
                        info="結果の表示形式"
                    )
                with gr.Column(scale=2):
                    min_community_rank = gr.Slider(
                        label="最小コミュニティランク",
                        minimum=0,
                        maximum=5,
                        value=0,
                        step=1,
                        info="0 = すべてのレベル"
                    )
            
            # Chat interface
            chatbot = gr.Chatbot(
                value=[],
                label="検索チャット",
                show_label=True,
                height=400,
                buttons=["copy"]
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
            def send_message(message, history, mode, response_type, output_format, min_rank):
                return app.search_chat(message, history, mode, response_type, output_format, min_rank)
            
            def clear_chat():
                return []
            
            # Send message on button click or Enter
            send_btn.click(
                fn=send_message,
                inputs=[msg, chatbot, search_mode, response_type, output_format, min_community_rank],
                outputs=[msg, chatbot]
            )
            
            msg.submit(
                fn=send_message,
                inputs=[msg, chatbot, search_mode, response_type, output_format, min_community_rank],
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
            
            ### 1️⃣ インデックスの作成（CLIで実行）
            ```bash
            # インデックスを作成
            python -m graphrag_anthropic_llamaindex add
            ```
            
            ### 2️⃣ Webアプリで検索
            1. **設定タブ**: 設定ファイル（config.yaml）を読み込む
            2. **検索タブ**: チャット形式でインデックスから検索を実行
            
            ## 🔍 検索モードの説明
            
            - **グローバル検索**: 包括的な要約と洞察を提供（推奨）
            - **ローカル検索**: エンティティに基づいた詳細な検索
            - **DRIFT検索**: 次世代の実験的な検索方法
            - **自動選択**: クエリに基づいて最適な方法を自動選択
            
            ## 📝 回答タイプ
            
            - **複数段落**: 詳細な説明を含む包括的な回答
            - **単一段落**: 簡潔にまとめられた回答
            - **リスト形式**: 箇条書きで整理された回答
            - **要点のみ**: 重要なポイントのみを抽出
            
            ## ⚙️ 設定ファイル例
            
            ```yaml
            # LLMプロバイダー設定
            llm_provider: "anthropic"  # または "bedrock"
            
            # Anthropic API設定
            anthropic:
              model: "claude-3-opus-20240229"
              # api_keyは環境変数ANTHROPIC_API_KEYから読み込み
            
            # または AWS Bedrock設定
            bedrock:
              model: "anthropic.claude-3-sonnet-20240229-v1:0"
              region: "us-east-1"
            
            # 入出力ディレクトリ
            input_dir: "./data"
            output_dir: "./graphrag_output"
            
            # 埋め込みモデル
            embedding_model:
              name: "intfloat/multilingual-e5-small"
            
            # チャンキング設定
            chunking:
              chunk_size: 1024
              chunk_overlap: 20
            ```
            
            ## 💬 チャット検索の使い方
            
            1. 検索モードを選択（通常は「グローバル検索」がおすすめ）
            2. 回答タイプと出力形式を選択
            3. チャット欄に自然な日本語で質問を入力
            4. Enterキーまたは「送信」ボタンで検索実行
            5. 結果がチャット形式で表示されます
            6. 続けて追加の質問も可能です
            
            ## 🔧 トラブルシューティング
            
            - **API Key エラー**: 環境変数 `ANTHROPIC_API_KEY` を設定するか、AWS Bedrockを使用
            - **インデックスが見つからない**: CLIで `python -m graphrag_anthropic_llamaindex add` を実行
            - **検索結果が空**: 検索モードを変更して再試行
            """)
    
    return interface

def _parse_gradio_auth():
    """GRADIO_AUTH="user:password" を (user, password) に変換。未設定なら None。"""
    raw = os.environ.get("GRADIO_AUTH", "").strip()
    if not raw or ":" not in raw:
        return None
    user, _, password = raw.partition(":")
    return (user, password)

def main():
    interface = create_interface()
    # 既定は 127.0.0.1。外部公開する場合は環境変数で明示し、認証も設定する。
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    auth = _parse_gradio_auth()
    if server_name not in ("127.0.0.1", "localhost") and auth is None:
        logger.warning(
            "Gradio を %s で公開しますが GRADIO_AUTH が未設定です。"
            "ホスト側ポートを 127.0.0.1 に限定するか GRADIO_AUTH を設定してください。",
            server_name,
        )
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,
        auth=auth,
        theme=gr.themes.Soft()
    )

if __name__ == "__main__":
    main()