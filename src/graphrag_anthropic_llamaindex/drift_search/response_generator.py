"""Response generator for DRIFT Search."""

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from llama_index.core import Settings

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
        self.llm = llm or Settings.llm
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
        return """Вы - помощник для ответов на вопросы, использующий DRIFT (Dynamic Retrieval with Interactive Filtering and Transformations) поиск.

Вам предоставлен контекст, включающий:
1. Локальные сущности - конкретные объекты, концепции и их отношения
2. Глобальные сообщества - группы связанных сущностей с обобщенными темами
3. Текстовые фрагменты - релевантные отрывки из исходных документов

Ваша задача:
- Синтезировать информацию из всех источников для предоставления исчерпывающего ответа
- Приоритизировать конкретные факты из локальных сущностей
- Использовать глобальные сообщества для контекста и обобщений
- Ссылаться на текстовые фрагменты для подтверждения утверждений
- Отвечать на русском языке, если вопрос на русском"""
    
    def _get_user_prompt_template(self) -> str:
        """Get user prompt template."""
        return """Вопрос: {query}

Контекст для ответа:
{context}

Пожалуйста, предоставьте подробный и точный ответ на основе предоставленного контекста."""
    
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
            
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt_template.format(
                        query=context.query,
                        context=formatted_context,
                    ),
                },
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
            return f"Ошибка при генерации ответа: {str(e)}"
    
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
            
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt_template.format(
                        query=context.query,
                        context=formatted_context,
                    ),
                },
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
            yield f"Ошибка при потоковой генерации: {str(e)}"
    
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
            f"Результаты DRIFT поиска для запроса: '{context.query}'",
            "",
        ]
        
        # Add main entities
        if key_info["main_entities"]:
            response_parts.append("**Основные сущности:**")
            for entity in key_info["main_entities"]:
                response_parts.append(
                    f"- {entity['name']} ({entity['type']}): {entity['description']}"
                )
            response_parts.append("")
        
        # Add main communities
        if key_info["main_communities"]:
            response_parts.append("**Сообщества знаний:**")
            for community in key_info["main_communities"]:
                response_parts.append(
                    f"- {community['title']} ({community['size']} сущностей): {community['summary']}"
                )
            response_parts.append("")
        
        # Add summary points
        if key_info["summary_points"]:
            response_parts.append("**Сводка:**")
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