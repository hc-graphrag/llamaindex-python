# Docker コンテナで Claude サブスクリプションを使う

調査日: 2026-10-24。出典は末尾。`llm_provider: "claude_code"` をコンテナで動かす方法と、その制約をまとめる。

## 結論

- コンテナ内での `claude login` や `~/.claude` のコピーは不要。**`claude setup-token` で発行した OAuth トークンを環境変数 `CLAUDE_CODE_OAUTH_TOKEN` で渡す**のが公式の方法。
- `claude` CLI は `claude-agent-sdk` の wheel に同梱されている (`claude_agent_sdk/_bundled/claude`)。Dockerfile への CLI の追加インストールは不要。
- トークンはモデル呼び出し専用で、有効期間は 1 年。Pro / Max / Team / Enterprise のプランが必要。
- **個人利用に限る。** 他人に使わせるサービスや共有環境への組み込みは規約上認められない。

## 手順

1. ホストで `claude setup-token` を実行し、ブラウザで承認してトークンを控える (保存はされない)。
2. `docker/.env` に `CLAUDE_CODE_OAUTH_TOKEN=<トークン>` を書く。`ANTHROPIC_API_KEY` は書かない。
3. `config.yaml` (`docker/config.docker.yaml` のコピー) の `llm_provider` を `"claude_code"` にする。
4. `make up`。

compose は `environment:` に素の `- CLAUDE_CODE_OAUTH_TOKEN` / `- ANTHROPIC_API_KEY` を列挙し、ホスト側で設定されているものだけを渡す。`NAME=${NAME}` と書くと未設定でも空文字がコンテナへ入り、認証の優先順位を乱すため使わない。`docker/.env.example` でも認証情報の行はコメントアウトのままにしてある。

## 認証の優先順位と落とし穴

claude CLI は次の順に認証情報を選ぶ。上にあるものが勝つ。

1. クラウドプロバイダ (`CLAUDE_CODE_USE_BEDROCK` 等)
2. `ANTHROPIC_AUTH_TOKEN`
3. `ANTHROPIC_API_KEY` (非対話モードでは設定されていれば常に使われる)
4. `apiKeyHelper`
5. **`CLAUDE_CODE_OAUTH_TOKEN`**
6. Anthropic プロファイル
7. `/login` のサブスクリプション認証

`ANTHROPIC_API_KEY` が環境に残っていると、サブスクリプションではなく API キーの従量課金で動く。`ClaudeCodeLLM` は該当の環境変数を検出すると警告をログに出す。

## 実機で確認できたこと (python:3.11-slim + `claude-agent-sdk==0.2.172`)

- 同梱 CLI は slim イメージでそのまま動く (`2.1.308`)。
- 認証なし: `Not logged in · Please run /login`。
- 不正なトークン: `Failed to authenticate. API Error: 401 OAuth access token is invalid.`

有効なトークンでの生成成功は、`claude setup-token` がブラウザ承認を要するため未検証。

## 規約上の制約

Anthropic の Legal and compliance ページによる。

- OAuth 認証はサブスクリプション契約者本人の通常利用のためのもの。
- Agent SDK を使って製品やサービスを作る開発者は API キー (Console) かクラウドプロバイダを使う。Claude.ai ログインの提供、Free / Pro / Max の認証情報でのユーザー代理リクエスト、Claude.ai 認証情報の収集・保管・仲介は認められない。
- 一方、利用者本人が自分のサブスクリプションで未改変の Claude Code にサインインすることは妨げない。

したがって、本人が自分のトークンを自分のコンテナに渡す使い方は範囲内と解釈できる。複数人で共有するトークンや、他者向けの Gradio UI の公開は範囲外。判断に迷う用途は Anthropic に確認すること。

## 出典

- Authentication — https://code.claude.com/docs/en/authentication
- Legal and compliance — https://code.claude.com/docs/en/legal-and-compliance
