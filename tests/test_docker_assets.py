"""Docker 関連ファイル (Makefile / compose / Dockerfile / コンテナ用設定) の整合性テスト。

イメージのビルド・起動そのものは別途実機で検証する (ネットワークと時間が必要なため)。
ここでは、過去に壊れていた静的な食い違いの再発を防ぐ。
"""

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from graphrag_anthropic_llamaindex.config_manager import load_config

ROOT = Path(__file__).resolve().parent.parent
DOCKER = ROOT / "docker"
CONTAINER_CONFIG = DOCKER / "config.docker.yaml"


def test_makefile_uses_compose_v2_only():
    """`docker-compose` (v1) は廃止済みで、多くの環境に存在しない。"""
    makefile = (ROOT / "Makefile").read_text()
    # ファイル名 docker-compose.yml は v2 でも有効なので、コマンドとしての呼び出しだけを禁止する
    assert not re.search(r"\bdocker-compose\b(?!\.ya?ml)", makefile)


def test_makefile_compose_file_points_to_an_existing_file():
    makefile = (ROOT / "Makefile").read_text()
    compose_file = re.search(r"^COMPOSE_FILE\s*=\s*(\S+)", makefile, re.M).group(1)
    assert (ROOT / compose_file).is_file()


@pytest.mark.skipif(shutil.which("make") is None, reason="make が無い")
def test_make_setup_creates_valid_config_and_env(tmp_path):
    (tmp_path / "docker").mkdir()
    shutil.copy(ROOT / "Makefile", tmp_path / "Makefile")
    shutil.copy(DOCKER / ".env.example", tmp_path / "docker" / ".env.example")
    shutil.copy(CONTAINER_CONFIG, tmp_path / "docker" / CONTAINER_CONFIG.name)

    result = subprocess.run(["make", "setup"], cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "docker" / ".env").exists()
    assert (tmp_path / "data").is_dir() and (tmp_path / "graphrag_output").is_dir()
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == yaml.safe_load(CONTAINER_CONFIG.read_text())


def test_make_setup_does_not_overwrite_existing_config(tmp_path):
    (tmp_path / "docker").mkdir()
    shutil.copy(ROOT / "Makefile", tmp_path / "Makefile")
    shutil.copy(DOCKER / ".env.example", tmp_path / "docker" / ".env.example")
    shutil.copy(CONTAINER_CONFIG, tmp_path / "docker" / CONTAINER_CONFIG.name)
    (tmp_path / "config.yaml").write_text("custom: true\n")

    subprocess.run(["make", "setup"], cwd=tmp_path, capture_output=True, text=True, check=True)

    assert (tmp_path / "config.yaml").read_text() == "custom: true\n"


def test_container_config_uses_container_paths_and_a_usable_provider():
    config = load_config(str(CONTAINER_CONFIG))

    assert config["input_dir"] == "/app/data"
    assert config["output_dir"] == "/app/graphrag_output"
    # claude CLI のログインはコンテナ内に無いので claude_code は使えない
    assert config["llm_provider"] in {"anthropic", "bedrock"}
    # ベクトルDBはマウントされた出力ディレクトリ配下に置く (コンテナ再作成で消えない)
    assert config["vector_store"]["lancedb"]["uri"].startswith(config["output_dir"] + "/")


def test_compose_mounts_config_where_the_ui_looks_for_it():
    compose = yaml.safe_load((DOCKER / "docker-compose.yml").read_text())
    volumes = compose["services"]["graphrag-app"]["volumes"]
    targets = {v.split(":")[1] for v in volumes if isinstance(v, str) and v.startswith("../")}

    gradio_source = (ROOT / "gradio_app.py").read_text()
    ui_default = re.search(r'label="設定ファイルパス",\s*value="([^"]+)"', gradio_source).group(1)

    assert f"/app/{ui_default}" in targets


def test_compose_has_no_obsolete_version_key():
    compose = yaml.safe_load((DOCKER / "docker-compose.yml").read_text())
    assert "version" not in compose


def test_healthcheck_command_exists_in_image():
    compose = yaml.safe_load((DOCKER / "docker-compose.yml").read_text())
    health_cmd = compose["services"]["graphrag-app"]["healthcheck"]["test"][1]
    apt_block = re.search(r"apt-get install -y(.*?)&&", (DOCKER / "Dockerfile").read_text(), re.S).group(1)

    assert health_cmd in apt_block.split()


def test_dockerfile_copies_everything_the_project_build_needs():
    """pdm install はプロジェクト自体のビルドで pyproject の readme を要求する。"""
    readme = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["readme"]
    copy_lines = [l for l in (DOCKER / "Dockerfile").read_text().splitlines() if l.startswith("COPY")]

    assert any(readme in l.split() for l in copy_lines), f"Dockerfile が {readme} を COPY していない"


def _dockerfile_instructions():
    return [line.strip() for line in (DOCKER / "Dockerfile").read_text().splitlines() if line.strip() and not line.startswith("#")]


def test_container_does_not_run_as_root():
    """root で動くと bind mount に root 所有のファイルが作られ、ホストのユーザーが消せなくなる。"""
    instructions = _dockerfile_instructions()
    users = [i.split()[1] for i in instructions if i.upper().startswith("USER ")]

    assert users, "USER 指定が無く、root で動いてしまう"
    assert users[-1] not in {"root", "0"}
    last_user = max(n for n, i in enumerate(instructions) if i.upper().startswith("USER "))
    cmd = next(n for n, i in enumerate(instructions) if i.upper().startswith("CMD "))
    assert last_user < cmd


def test_image_user_can_match_the_host_user():
    """bind mount に書くファイルをホストのユーザー所有にするため、UID/GID をビルド引数で合わせられる。"""
    dockerfile = (DOCKER / "Dockerfile").read_text()
    assert re.search(r"^ARG APP_UID=1000$", dockerfile, re.M) and re.search(r"^ARG APP_GID=1000$", dockerfile, re.M)

    compose = yaml.safe_load((DOCKER / "docker-compose.yml").read_text())
    args = compose["services"]["graphrag-app"]["build"]["args"]
    assert args["APP_UID"] == "${HOST_UID:-1000}" and args["APP_GID"] == "${HOST_GID:-1000}"

    makefile = (ROOT / "Makefile").read_text()
    assert re.search(r"^export HOST_UID\s*:?=.*id -u", makefile, re.M)
    assert re.search(r"^export HOST_GID\s*:?=.*id -g", makefile, re.M)


def test_mount_points_are_owned_by_the_app_user():
    dockerfile = (DOCKER / "Dockerfile").read_text()
    assert re.search(r"chown -R app:app .*?/app", dockerfile)
    assert "/app/logs" in dockerfile  # 名前付きボリュームの初期所有者はイメージ側の同名ディレクトリから引き継がれる


def test_compose_passes_credentials_only_when_set():
    """`NAME=${NAME}` は未設定でも空文字をコンテナへ渡し、claude CLI の認証優先順位を乱す。素の `- NAME` で透過する。"""
    compose = yaml.safe_load((DOCKER / "docker-compose.yml").read_text())
    env = compose["services"]["graphrag-app"]["environment"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" in env and "ANTHROPIC_API_KEY" in env


def test_env_example_documents_the_subscription_token():
    example = (DOCKER / ".env.example").read_text()
    assert "CLAUDE_CODE_OAUTH_TOKEN" in example
    # 既定ではどの認証も有効にしない (コメントアウトのまま)
    assert not re.search(r"^CLAUDE_CODE_OAUTH_TOKEN=", example, re.M)


def test_env_example_has_no_active_empty_credential():
    """`make setup` が docker/.env にコピーするため、有効な行があると空の認証情報がコンテナへ渡る。"""
    example = (DOCKER / ".env.example").read_text()
    assert not re.search(r"^(ANTHROPIC_API_KEY|CLAUDE_CODE_OAUTH_TOKEN)=", example, re.M)
