"""Prompt templates for local search."""

LOCAL_SEARCH_PROMPT = """You are a helpful assistant that answers questions based on the provided context information from a knowledge graph.

## Context Information:
{context}

## User Question:
{query}

## Instructions:
1. Answer the question based ONLY on the information provided in the context above.
2. Be specific and cite relevant entities and relationships when possible.
3. If the context doesn't contain enough information to answer the question, say so clearly.
4. Keep your answer concise and focused on the question asked.

Answer:"""

LOCAL_SEARCH_WITH_CITATIONS_PROMPT = """You are a helpful assistant that answers questions based on the provided context information from a knowledge graph.

## Context Information:
{context}

## User Question:
{query}

## Instructions:
1. Answer the question based ONLY on the information provided in the context above.
2. Include inline citations to specific entities and relationships using [Entity: name] or [Relationship: source->target] format.
3. If the context doesn't contain enough information to answer the question completely, acknowledge what you can answer and what information is missing.
4. Structure your answer clearly with proper formatting.
5. Be precise and avoid speculation beyond what the context provides.

Answer:"""

LOCAL_SEARCH_ANALYTICAL_PROMPT = """You are an analytical assistant that provides detailed insights based on knowledge graph data.

## Context Information:
{context}

## User Question:
{query}

## Analysis Guidelines:
1. Provide a comprehensive analysis based on the entities and relationships in the context.
2. Identify patterns, connections, and insights that may not be immediately obvious.
3. Organize your response with clear sections if the answer is complex.
4. Highlight key entities and their relationships.
5. If relevant, suggest what additional information might be helpful but is not present in the context.
6. Use bullet points or numbered lists for clarity when appropriate.

Analytical Response:"""

LOCAL_SEARCH_SUMMARY_PROMPT = """You are a summarization assistant that creates concise summaries from knowledge graph information.

## Context Information:
{context}

## User Request:
{query}

## Summary Instructions:
1. Create a brief, informative summary that directly addresses the user's request.
2. Focus on the most relevant and important information from the context.
3. Use clear, simple language.
4. Maximum 3-4 sentences unless more detail is specifically requested.
5. Mention key entities but avoid overwhelming detail.

Summary:"""

def get_local_search_prompt(style: str = "default") -> str:
    """
    Get the appropriate prompt template based on the requested style.
    
    Args:
        style: The prompt style - "default", "citations", "analytical", or "summary"
        
    Returns:
        The prompt template string
    """
    prompts = {
        "default": LOCAL_SEARCH_PROMPT,
        "citations": LOCAL_SEARCH_WITH_CITATIONS_PROMPT,
        "analytical": LOCAL_SEARCH_ANALYTICAL_PROMPT,
        "summary": LOCAL_SEARCH_SUMMARY_PROMPT
    }
    
    return prompts.get(style, LOCAL_SEARCH_PROMPT)