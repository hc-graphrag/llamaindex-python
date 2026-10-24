# Docker コンテナで Claude サブスクリプションを使う

調査日: 2026-10-24。出典は末尾。`llm_provider: "claude_code"` をコンテナで動かす方法と、その制約をまとめる。

## 結論

認証の渡し方は 2 通り。**どちらか一方だけ**使う。

| | A. `setup-token` (環境変数) | B. コンテナ内ログイン (永続ディレクトリ) |
|---|---|---|
| 認証情報 | 1 年有効の OAuth トークン (`docker/.env`) | `claude_home/` に保存されるログイン状態 (自動更新) |
| 初回操作 | ホストで `claude setup-token` | `make claude-login` (コンテナ内でブラウザ承認) |
| 複数コンテナ・並行実行 | 問題なし | 同じ `claude_home/` を共有する並行実行は避ける |
| 向く用途 | 常時起動の Gradio UI | ayanomics の Codex runner と同じ運用 |

- `claude` CLI は `claude-agent-sdk` の wheel に同梱されている (`claude_agent_sdk/_bundled/claude`)。追加インストールは不要。Dockerfile で PATH に通してある。
- トークンはモデル呼び出し専用。Pro / Max / Team / Enterprise のプランが必要。
- **個人利用に限る。** 他人に使わせるサービスや共有環境への組み込みは規約上認められない。

## 手順 A: `setup-token`

1. ホストの**ターミナルで直接** `claude setup-token` を実行し、ブラウザで承認してトークンを控える (保存はされない)。バックグラウンド実行や TTY の無い環境では完了しない。
2. `docker/.env` に `CLAUDE_CODE_OAUTH_TOKEN=<トークン>` を書く。`ANTHROPIC_API_KEY` は書かない。
3. `config.yaml` (`docker/config.docker.yaml` のコピー) の `llm_provider` を `"claude_code"` にする。
4. `make up`。

compose は `environment:` に素の `- CLAUDE_CODE_OAUTH_TOKEN` / `- ANTHROPIC_API_KEY` を列挙し、ホスト側で設定されているものだけを渡す。`NAME=${NAME}` と書くと未設定でも空文字がコンテナへ入り、認証の優先順位を乱すため使わない。`docker/.env.example` でも認証情報の行はコメントアウトのままにしてある。

## 手順 B: コンテナ内でログイン (ayanomics の Codex runner と同型)

ayanomics の `docker/codex-actions-runner/` は、`CODEX_HOME` を bind mount で永続化し、本人がコンテナ内で `codex login --device-auth` を一度だけ実行する。同じ構造を Claude に写したもの。

| ayanomics (Codex) | このリポジトリ (Claude) |
|---|---|
| `CODEX_HOME` を `data/codex/` に bind mount | `CLAUDE_CONFIG_DIR=/home/app/.claude` を `claude_home/` に bind mount |
| `make login` (`codex login --device-auth`) | `make claude-login` (`claude auth login`) |
| `make auth-status` (`codex login status`) | `make claude-status` (`claude auth status`。未ログインは終了コード 1) |

1. `make setup` (`claude_home/` を作る。Docker に作らせると root 所有になる)。
2. `docker/.env` に `CLAUDE_CODE_OAUTH_TOKEN` を**書かない**。環境変数のトークンはログイン状態より優先される。
3. `make up` でコンテナを起動し、`make claude-login`。表示された URL をブラウザで開いて承認し、必要なら表示されたコードを端末に貼る。
4. `make claude-status` で `loggedIn: true` を確認する。

運用ルール (ayanomics の不変条件に準拠):

- `claude_home/` の中身を別の環境へコピー・共有しない。環境ごとに本人が再ログインする。`claude_home/` は Git 管理外。
- 同じ `claude_home/` を使うコンテナ・ジョブを同時に走らせない (トークン更新が競合しうる)。
- `--dangerously-skip-permissions` を使うコンテナや、信頼できない入力を扱う用途では、`claude_home/` の認証情報が持ち出される恐れがある。公式の dev container 文書も同じ警告をしている。

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

## 実機で確認できたこと

2026-10-24、WSL2 上の Docker 29.4.1、`claude-agent-sdk` 同梱 CLI 2.1.308。

- **手順 A**: イメージをビルドして起動し、ヘルスチェックが `healthy`、UI が HTTP 200。`CLAUDE_CODE_OAUTH_TOKEN` だけがコンテナに渡り、`ANTHROPIC_API_KEY` は未設定。コンテナ内の `ClaudeCodeLLM(model="sonnet").complete(...)` が実際に応答を返した。
- **手順 B の土台**: `claude` が PATH にあり、`make claude-status` が動く。トークンの環境変数を外すと `loggedIn: false` / `authMethod: none` / 終了コード 1。`.claude.json` はホストの `claude_home/` にホストのユーザー所有で書かれ、コンテナを作り直しても残る。
- 認証なし: `Not logged in · Please run /login`。不正なトークン: `Failed to authenticate. API Error: 401 OAuth access token is invalid.`

**未検証**: `make claude-login` によるブラウザ承認の完了と、ログイン後の生成。ブラウザ承認が本人操作を要するため。

### WSL2 でのビルド

既定のブリッジネットワークでは、`docker compose build` が `pdm install` の PyPI ダウンロードで止まることがある (51 分待っても進まなかった)。`--network host` を付けると数分で通る。compose の `build` には `--network` オプションが無いので、compose が使うイメージ名 (`docker-graphrag-app`) を指定して直接ビルドする。

```
docker build --network host -f docker/Dockerfile \
  --build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g) \
  -t docker-graphrag-app .
make up    # 既存のイメージを使って起動する
```

## 規約上の制約

Anthropic の Legal and compliance ページによる。

- OAuth 認証はサブスクリプション契約者本人の通常利用のためのもの。
- Agent SDK を使って製品やサービスを作る開発者は API キー (Console) かクラウドプロバイダを使う。Claude.ai ログインの提供、Free / Pro / Max の認証情報でのユーザー代理リクエスト、Claude.ai 認証情報の収集・保管・仲介は認められない。
- 一方、利用者本人が自分のサブスクリプションで未改変の Claude Code にサインインすることは妨げない。

したがって、本人が自分のトークンを自分のコンテナに渡す使い方は範囲内と解釈できる。複数人で共有するトークンや、他者向けの Gradio UI の公開は範囲外。判断に迷う用途は Anthropic に確認すること。

## 出典

- Authentication — https://code.claude.com/docs/en/authentication
- Development containers — https://code.claude.com/docs/en/devcontainer
- Legal and compliance — https://code.claude.com/docs/en/legal-and-compliance
