"""Response generator for DRIFT Search."""

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from graphrag_anthropic_llamaindex.llm_utils import require_configured_llm

from .context_builder import ContextBuilder
from .models import SearchContext

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses for DRIFT search."""
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize response generator.
        
        Args:
            llm: LLM instance to use
            config: Response generation configuration
        """
        self.llm = llm or require_configured_llm("DRIFT応答生成にはLLMが必要です。llmを渡すか、Settings.llmを設定してください。")
        self.config = config or {}
        
        # Configuration
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2000)
        self.streaming_enabled = self.config.get("streaming_enabled", True)
        self.chunk_size = self.config.get("chunk_size", 50)
        
        # Context builder for formatting
        self.context_builder = ContextBuilder()
        
        # Default prompts
        self.system_prompt = self._get_system_prompt()
        self.user_prompt_template = self._get_user_prompt_template()
        
        logger.info(f"ResponseGenerator initialized with max_tokens={self.max_tokens}")
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for DRIFT search."""
        return """You are an AI assistant that uses DRIFT (Dynamic Retrieval with Interactive Filtering and Transformations) search to answer questions.

You are provided with context that includes:
1. Local entities - specific objects, concepts and their relationships
2. Global communities - groups of related entities with summarized themes  
3. Text units - relevant excerpts from source documents

Your task:
- Synthesize information from all sources to provide a comprehensive answer
- Prioritize specific facts from local entities
- Use global communities for context and generalizations
- Reference text units to support your claims
- **IMPORTANT: Always respond in the same language as the user's question**
- If the question is in Japanese, respond in Japanese
- If the question is in English, respond in English
- Provide accurate, relevant, and well-structured responses"""
    
    def _get_user_prompt_template(self) -> str:
        """Get user prompt template."""
        return """Question: {query}

Context for answering:
{context}

Please provide a detailed and accurate answer based on the provided context. Remember to respond in the same language as the question."""
    
    async def generate_response(
        self,
        context: SearchContext,
    ) -> str:
        """
        Generate response from search context.
        
        Args:
            context: Search context
            
        Returns:
            Generated response
        """
        try:
            # Format context for prompt
            formatted_context = self.context_builder.format_context_for_prompt(context)
            
            # Build messages using proper LlamaIndex ChatMessage format
            from llama_index.core.llms import ChatMessage, MessageRole
            
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                ChatMessage(
                    role=MessageRole.USER,
                    content=self.user_prompt_template.format(
                        query=context.query,
                        context=formatted_context,
                    ),
                ),
            ]
            
            # Generate response
            response = await self.llm.achat(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract text from response
            if hasattr(response, "message"):
                response_text = response.message.content
            elif hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
            
            logger.info(f"Generated response of {len(response_text)} characters")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"エラーが発生しました: {str(e)}"
    
    async def stream_response(
        self,
        context: SearchContext,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response generation.
        
        Args:
            context: Search context
            
        Yields:
            Response chunks
        """
        if not self.streaming_enabled:
            # Fall back to non-streaming
            response = await self.generate_response(context)
            yield response
            return
        
        try:
            # Format context for prompt
            formatted_context = self.context_builder.format_context_for_prompt(context)
            
            # Build messages using proper LlamaIndex ChatMessage format
            from llama_index.core.llms import ChatMessage, MessageRole
            
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                ChatMessage(
                    role=MessageRole.USER,
                    content=self.user_prompt_template.format(
                        query=context.query,
                        context=formatted_context,
                    ),
                ),
            ]
            
            # Stream response
            stream = await self.llm.astream_chat(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            buffer = ""
            async for chunk in stream:
                # Extract text from chunk
                if hasattr(chunk, "delta"):
                    chunk_text = chunk.delta
                elif hasattr(chunk, "content"):
                    chunk_text = chunk.content
                else:
                    chunk_text = str(chunk)
                
                buffer += chunk_text
                
                # Yield when buffer reaches chunk size
                while len(buffer) >= self.chunk_size:
                    yield buffer[:self.chunk_size]
                    buffer = buffer[self.chunk_size:]
            
            # Yield remaining buffer
            if buffer:
                yield buffer
                
        except Exception as e:
            logger.error(f"Error streaming response: {e}", exc_info=True)
            yield f"ストリーミング生成でエラーが発生しました: {str(e)}"
    
    def create_summary_response(
        self,
        context: SearchContext,
    ) -> str:
        """
        Create a summary response without LLM.
        
        Args:
            context: Search context
            
        Returns:
            Summary response
        """
        key_info = self.context_builder.extract_key_information(context)
        
        response_parts = [
            f"DRIFT search results for query: '{context.query}'",
            "",
        ]
        
        # Add main entities
        if key_info["main_entities"]:
            response_parts.append("**Main Entities:**")
            for entity in key_info["main_entities"]:
                response_parts.append(
                    f"- {entity['name']} ({entity['type']}): {entity['description']}"
                )
            response_parts.append("")
        
        # Add main communities
        if key_info["main_communities"]:
            response_parts.append("**Knowledge Communities:**")
            for community in key_info["main_communities"]:
                response_parts.append(
                    f"- {community['title']} ({community['size']} entities): {community['summary']}"
                )
            response_parts.append("")
        
        # Add summary points
        if key_info["summary_points"]:
            response_parts.append("**Summary:**")
            for point in key_info["summary_points"]:
                response_parts.append(f"- {point}")
        
        return "\n".join(response_parts)
    
    def validate_response(
        self,
        response: str,
        context: SearchContext,
    ) -> bool:
        """
        Validate response quality.
        
        Args:
            response: Generated response
            context: Search context used
            
        Returns:
            True if response is valid
        """
        # Check response length
        if len(response) < 50:
            logger.warning("Response too short")
            return False
        
        # Check if response addresses the query
        query_terms = context.query.lower().split()
        response_lower = response.lower()
        
        matching_terms = sum(1 for term in query_terms if term in response_lower)
        if matching_terms < len(query_terms) * 0.3:
            logger.warning("Response doesn't address query terms")
            return False
        
        return True