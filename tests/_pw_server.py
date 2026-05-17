"""Subprocess entrypoint for the Playwright live test.

Seeds a tmp data_dir, bootstraps the admin key (writing the cleartext
to ``bootstrap.key`` next to the config), and starts uvicorn. The
Playwright harness reads the key file and drives the UI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    data_dir = Path(sys.argv[1])
    port = int(sys.argv[2])
    cfg_path = data_dir / "config.toml"
    # Write with explicit LF — tomlkit refuses CRLF in some configurations.
    cfg_path.write_bytes(
        f"""
[server]
bind = "127.0.0.1"
port = {port}
llama_server_binary = "true"
llama_server_port = {port + 1}
data_dir = "{data_dir.as_posix()}"

[defaults]
model = ""

[default_args.llama]
temp = 0.7

[default_args.mlx]
temp = 0.6
""".encode("utf-8")
    )

    # Seed two fake models so the page shows both engines.
    models = data_dir / "models"
    gguf = models / "demo-org" / "demo-model.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    mlx_dir = models / "mlx-demo"
    mlx_dir.mkdir(parents=True, exist_ok=True)
    (mlx_dir / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (mlx_dir / "model.safetensors").write_bytes(b"")

    from llamanager.app import create_app

    app = create_app(cfg_path, print_bootstrap=False)
    am = app.state.auth
    boot = am.get_origin_by_name("bootstrap")
    assert boot is not None
    key = am.rotate_key(boot.id)
    (data_dir / "bootstrap.key").write_text(key, encoding="utf-8")

    import uvicorn
    os.environ.setdefault("LLAMANAGER_NO_BANNER", "1")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
