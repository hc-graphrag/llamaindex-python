import json
from llama_index.core import Settings


def get_configured_llm():
    """明示的に設定された LLM を返す。未設定なら None。

    `Settings.llm` を直接読むと、未設定時に LlamaIndex が既定の OpenAI LLM を解決しようとして
    OPENAI_API_KEY 不足で例外になる。設定有無の判定には本関数を使う。
    """
    return Settings._llm


def require_configured_llm(message):
    """設定済みの LLM を返す。未設定なら message で ValueError を送出する。"""
    llm = get_configured_llm()
    if llm is None:
        raise ValueError(message)
    return llm

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
    
    # LLM設定の確認
    llm = require_configured_llm("LLMが設定されていません。Settings.llmを設定してください。")
    
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
                f"Previous response was truncated due to token limits. Please continue and complete the JSON output.\n"
                f"Previous response:\n```\n{full_response_text}\n```\n"
                f"Please continue and complete the JSON with proper [START_JSON] and [END_JSON] tags."
            )
        
        try:
            response = llm.complete(current_prompt)
        except Exception as e:
            print(f"LLM呼び出しエラー (試行 {attempts}/{max_continuation_attempts}): {e}")
            if attempts == max_continuation_attempts:
                raise e
            continue
        next_part = response.text
        
        full_response_text = _stitch_responses(full_response_text, next_part)
        
        # Check if the current full_response_text is valid JSON
        parsed_json = parse_llm_json_output(full_response_text)
        json_parse_successful = (parsed_json is not None)
        
        # Also check for explicit truncation reason from LLM
        is_explicitly_truncated = False
        if hasattr(response, 'raw') and response.raw:
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
    """
    Parse JSON from LLM output with multiple fallback strategies.
    """
    import re

    if not json_string or not json_string.strip():
        return None

    # Debug: Print first 200 chars to understand the format
    print(f"DEBUG: Parsing JSON response (first 200 chars): {repr(json_string[:200])}")

    original_string = json_string

    try:
        # Strategy 1: Look for [START_JSON] and [END_JSON] tags
        start_tag = "[START_JSON]"
        end_tag = "[END_JSON]"

        start_idx = json_string.find(start_tag)
        end_idx = json_string.rfind(end_tag)

        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_string = json_string[start_idx + len(start_tag) : end_idx].strip()
            try:
                return json.loads(json_string)
            except json.JSONDecodeError:
                pass  # Try other strategies

        # Strategy 2: Look for code blocks
        json_string = original_string
        if json_string.strip().startswith("```json") and json_string.strip().endswith("```"):
            json_string = json_string.strip()[7:-3].strip()
            try:
                return json.loads(json_string)
            except json.JSONDecodeError:
                pass
        elif json_string.strip().startswith("```") and json_string.strip().endswith("```"):
            json_string = json_string.strip()[3:-3].strip()
            try:
                return json.loads(json_string)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Extract JSON object with balanced braces
        json_string = original_string
        start_idx = json_string.find('{')
        if start_idx != -1:
            # Find matching closing brace
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(json_string)):
                if json_string[i] == '{':
                    brace_count += 1
                elif json_string[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break

            if end_idx != -1:
                json_string = json_string[start_idx:end_idx+1]
                try:
                    return json.loads(json_string)
                except json.JSONDecodeError:
                    pass

        # Strategy 4: Use regex to find JSON-like structures
        json_string = original_string
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, json_string, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Strategy 5: Try to fix common JSON issues
        json_string = original_string

        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            "Here's the JSON:", "Here is the JSON:", "JSON:", "Response:",
            "The JSON output is:", "Output:", "Result:"
        ]
        for prefix in prefixes_to_remove:
            if json_string.strip().lower().startswith(prefix.lower()):
                json_string = json_string[len(prefix):].strip()
                break

        # Extract content between first { and last }
        start_idx = json_string.find('{')
        end_idx = json_string.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_string = json_string[start_idx:end_idx+1]

            # Try to fix common issues
            json_string = json_string.replace('\n', ' ')  # Remove newlines
            json_string = re.sub(r',\s*}', '}', json_string)  # Remove trailing commas
            json_string = re.sub(r',\s*]', ']', json_string)  # Remove trailing commas in arrays

            try:
                return json.loads(json_string)
            except json.JSONDecodeError:
                pass

        print(f"DEBUG: All JSON parsing strategies failed for: {repr(original_string[:500])}")
        return None

    except Exception as e:
        print(f"DEBUG: Unexpected error in JSON parsing: {e}")
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
以下のテキストを要約し、主要なエンティティとその関係に焦点を当ててください。
簡潔な要約と言及された主要エンティティのリストを提供してください。
結果は、'community_id'（整数）、'summary'（文字列）、'key_entities'（文字列のリスト）の3つのキーを持つJSONオブジェクトとして出力してください。

重要: [START_JSON]と[END_JSON]タグで囲まれたJSONオブジェクトのみを出力してください。これらのタグの外側には他のテキストやマークダウン形式を含めないでください。

JSONフォーマットの例:
[START_JSON]
{{
    "community_id": 123,
    "summary": "このコミュニティでは...",
    "key_entities": ["エンティティA", "エンティティB"]
}}
[END_JSON]

テキスト: {text}

必ず上記の形式に従い、[START_JSON]と[END_JSON]タグで囲んでJSONのみを出力してください。説明や追加のコメントは一切含めないでください。
"""
