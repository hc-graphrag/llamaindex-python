"""
GLOBAL検索用のプロンプトテンプレート

MS-GraphRAGのMap-Reduceパターンに基づくプロンプト定義
"""

# Map処理用のシステムプロンプト
MAP_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant answering questions about provided data table information.

---Goal---

Generate a response consisting of key points that answer the user's question, summarizing all relevant information in the input data table.

Use the provided data table as the primary context for generating responses.
If you don't know the answer or if the input data table doesn't contain sufficient information, please say so. Don't make anything up.

Each key point in the response should include the following elements:
- Description: A comprehensive description of the point
- Importance score: An integer score from 0-100 indicating the importance of the point in answering the user's question. "Don't know" type responses should have a score of 0.

Format the response in JSON as follows:
{{
    "points": [
        {{"description": "Description of point 1 [Data: Reports (report IDs)]", "score": score_value}},
        {{"description": "Description of point 2 [Data: Reports (report IDs)]", "score": score_value}}
    ]
}}

The response should preserve the original meaning and usage of modal verbs such as "shall", "may", "will".

Points supported by data should list related reports as references in the following format:
"This is an example sentence supported by data references [Data: Reports (report IDs)]"

**Do not list more than 5 record IDs in a single reference**. Instead, list the top 5 most relevant record IDs and add "+more" to indicate there are additional ones.

Example:
"Person X owns Company Y and has been the subject of many fraud allegations [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also the CEO of Company X [Data: Reports (1, 3)]"

Where 1, 2, 3, 7, 34, 46, 64 represent the IDs (not indices) of related data reports in the provided table.

Do not include information without supporting evidence.

Limit response length to {max_length} words.

---Data Table---

{context_data}

---Goal---

Generate a response consisting of key points that answer the user's question, summarizing all relevant information in the input data table.
"""

# Reduce処理用のシステムプロンプト
REDUCE_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant that answers questions about datasets by integrating perspectives from multiple analysts.

---Goal---

Generate a response of the target length and format that answers the user's question, summarizing all reports from multiple analysts who focused on different parts of the dataset.

Note that the analyst reports provided below are ranked in **descending order of importance**.

If you don't know the answer or if the provided reports don't contain sufficient information to provide an answer, please say so. Don't make anything up.

The final response should remove all irrelevant information from the analyst reports and merge the cleaned information into a comprehensive answer that provides explanations of all important points and implications appropriate for the response length and format.

Add sections and comments to the response according to the response length and format. Style the response in markdown.

The response should preserve the original meaning and usage of modal verbs such as "shall", "may", "will".

The response should also retain all data references that were previously included in the analyst reports, but should not mention the role of multiple analysts in the analysis process.

**Do not list more than 5 record IDs in a single reference**. Instead, list the top 5 most relevant record IDs and add "+more" to indicate there are additional ones.

Example:
"Person X owns Company Y and has been the subject of many fraud allegations [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also the CEO of Company X [Data: Reports (1, 3)]"

Where 1, 2, 3, 7, 34, 46, 64 represent the IDs (not indices) of related data records.

Do not include information without supporting evidence.

Limit response length to {max_length} words.

---Target Response Length and Format---

{response_type}

---Analyst Reports---

{report_data}

---Goal---

Generate a response of the target length and format that answers the user's question, summarizing all reports from multiple analysts who focused on different parts of the dataset.

Add sections and comments to the response according to the response length and format. Style the response in markdown.
"""

# Map処理用のユーザープロンプトテンプレート
MAP_USER_PROMPT = """
Context Data:
{context}

User Question:
{query}

Please analyze the provided context data and identify key points that help answer the user's question.
"""

# Reduce処理用のユーザープロンプトテンプレート  
REDUCE_USER_PROMPT = """
Analyst Reports:
{report_data}

User Question:
{query}

Please synthesize the analyst reports into a comprehensive answer to the user's question.
"""

# データなしの場合の回答
NO_DATA_ANSWER = "I'm sorry, but I cannot answer this question with the provided data."