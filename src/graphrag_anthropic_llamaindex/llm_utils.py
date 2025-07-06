import json

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
