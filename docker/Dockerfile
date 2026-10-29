# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PDM
RUN pip install pdm

# 非 root ユーザーで動かす。bind mount に書かれるファイルがホストのユーザー所有になるよう、
# UID/GID はビルド引数でホストに合わせる (Makefile が `id -u` / `id -g` を渡す)
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid ${APP_GID} app && useradd --create-home --uid ${APP_UID} --gid ${APP_GID} app

# Create directories for mounted volumes (名前付きボリューム /app/logs はここの所有者を引き継ぐ)
RUN mkdir -p /app/data /app/graphrag_output /app/config /app/logs && chown -R app:app /app
USER app

# Copy project files
COPY --chown=app:app pyproject.toml pdm.lock* README.md ./

# Install dependencies using PDM
RUN pdm install --prod --no-self

COPY --chown=app:app src/ ./src/
COPY --chown=app:app gradio_app.py ./

# Expose port
EXPOSE 7860

# Set environment variables
ENV PYTHONPATH=/app/src:/app
# claude_code プロバイダ: ログイン状態 (~/.claude と .claude.json) をここへ置き、bind mount で永続化する
ENV CLAUDE_CONFIG_DIR=/home/app/.claude
# claude-agent-sdk の wheel に同梱された CLI を PATH に通す (`claude auth login` 用)
ENV PATH=/app/.venv/lib/python3.11/site-packages/claude_agent_sdk/_bundled:$PATH
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run the Gradio app
CMD ["pdm", "run", "python", "gradio_app.py"]
