import json
from llama_index.core import Settings

def _stitch_responses(s1, s2):
    if not s1:
        return s2
    if not s2:
        return s1
    # Match up to 200 characters from the end of s1 with the start of s2
    search_window = 200
    max_overlap = 0
    for k in range(min(len(s1), len(s2), search_window), 0, -1):
        if s1.endswith(s2[:k]):
            max_overlap = k
            break
    return s1 + s2[max_overlap:]

def _get_full_llm_response_with_continuation(original_prompt, max_continuation_attempts=5):
    """
    Handles LLM calls with continuation logic for token limits.
    It repeatedly calls the LLM with a continuation prompt that includes the original prompt
    and the already generated partial response, stitching the parts together while removing overlaps.
    Includes a safeguard for infinite loops with max_continuation_attempts.
    """
    
    full_response_text = ""
    attempts = 0
    json_parse_successful = False

    while attempts < max_continuation_attempts and not json_parse_successful:
        attempts += 1
        current_prompt = original_prompt
        if attempts > 1:
            # Construct the continuation prompt: original prompt + current full response + continuation instruction
            current_prompt = (
                f"{original_prompt}\n\n"
                f"これまでの応答はトークン制限により途中で終了しました。続きを生成してください。\n"
                f"これまでの応答:\n```\n{full_response_text}\n```\n"
                f"続きを生成してください。"
            )
        
        response = Settings.llm.complete(current_prompt)
        next_part = response.text
        
        full_response_text = _stitch_responses(full_response_text, next_part)
        
        # Check if the current full_response_text is valid JSON
        parsed_json = parse_llm_json_output(full_response_text)
        json_parse_successful = (parsed_json is not None)
        
        # Also check for explicit truncation reason from LLM
        is_explicitly_truncated = response.raw.get("stop_reason") == "max_tokens"
        
        # If JSON parsing failed AND it's not explicitly truncated, it means the LLM stopped for another reason
        # or produced malformed JSON. We continue if it's explicitly truncated OR if JSON parsing failed.
        if not json_parse_successful and not is_explicitly_truncated:
            # If JSON parsing failed but LLM didn't say max_tokens, it might be a malformed JSON issue
            # that continuation won't fix, or a different stop reason. For now, we'll assume it needs continuation.
            # However, to prevent infinite loops on consistently malformed JSON, max_attempts is crucial.
            pass # The loop condition already handles attempts < max_continuation_attempts
        
        # The loop continues if JSON is not successful OR if it was explicitly truncated
        # and we haven't reached max attempts.
        # The condition `not json_parse_successful` is the primary driver for continuation now.

    if not json_parse_successful:
        print(f"Warning: JSON parsing failed after {attempts} attempts. Response might be incomplete or malformed.")
    elif attempts >= max_continuation_attempts:
        print(f"Warning: Max continuation attempts ({max_continuation_attempts}) reached. Response might be incomplete.")

    return full_response_text

def parse_llm_json_output(json_string):
    try:
        # Look for [START_JSON] and [END_JSON] tags
        start_tag = "[START_JSON]"
        end_tag = "[END_JSON]"
        
        start_idx = json_string.find(start_tag)
        end_idx = json_string.rfind(end_tag)

        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_string = json_string[start_idx + len(start_tag) : end_idx].strip()
        else:
            # Fallback to old logic if tags are not found
            if json_string.strip().startswith("```json") and json_string.strip().endswith("```"):
                json_string = json_string.strip()[7:-3].strip()
            elif json_string.strip().startswith("```") and json_string.strip().endswith("```"):
                json_string = json_string.strip()[3:-3].strip()
            
            start_idx = json_string.find('{')
            end_idx = json_string.rfind('}')
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_string = json_string[start_idx:end_idx+1]
        
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        return None

extraction_prompt_template = """
Extract entities and relationships from the following text.
Entities should have a 'name' and 'type'.
Relationships should have 'source', 'target', 'type', and 'description'.
Output the result as a JSON object with two keys: 'entities' (list of entity objects) and 'relationships' (list of relationship objects).
IMPORTANT: Only output the JSON object, enclosed within [START_JSON] and [END_JSON] tags. Do not include any other text or markdown formatting outside these tags.

Example JSON format:
[START_JSON]
{{
    "entities": [
        {{"name": "Alice", "type": "Person"}},
        {{"name": "Microsoft", "type": "Organization"}}
    ],
    "relationships": [
        {{"source": "Alice", "target": "Microsoft", "type": "works_for", "description": "Alice works for Microsoft"}}
    ]
}}
[END_JSON]

Text: {text}
"""

summary_prompt_template = """
Summarize the following text, focusing on key entities and their relationships.
Provide a concise summary and a list of key entities mentioned.
Output the result as a JSON object with three keys: 'community_id' (integer), 'summary' (string), and 'key_entities' (list of strings).
IMPORTANT: Only output the JSON object, enclosed within [START_JSON] and [END_JSON] tags. Do not include any other text or markdown formatting outside these tags.

Example JSON format:
[START_JSON]
{{
    "community_id": 123,
    "summary": "This community discusses...",
    "key_entities": ["Entity A", "Entity B"]
}}
[END_JSON]

Text: {text}
"""
