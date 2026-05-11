# llamanager

Queue AI work against a local model and reach it from anywhere on your network.

llamanager wraps `llama-server` (from llama.cpp) so a single GPU can serve a phone, a laptop, a CI job, and whatever else you point at it, without those clients trampling each other. Requests land on a per-origin priority queue, the loaded model swaps on demand, and the whole thing is reachable through one OpenAI-compatible endpoint.

## why this exists

`llama-server` is great at running a model. It is not great at being shared. Two processes hitting it directly will collide on context, you cannot swap models without restarting, and there is no auth. If you want your home workstation to act as a small private inference service for everything you own, you need a queue, a supervisor, a key per client, and a way to swap models without dropping in-flight work. That is what this is.

## what you get

- OpenAI-compatible `/v1/*` proxy, so any existing client library works
- per-origin priority queue with cancellation, so a long batch job cannot block your editor
- model lifecycle: start, stop, restart, hot-swap by model alias on a per-request header
- crash supervisor with a 3-in-5-minutes restart cap, so a broken GGUF cannot melt the box
- Hugging Face GGUF puller with disk-space checks
- HTMX web UI that is usable from a phone over Tailscale
- bearer-token auth, one key per origin, hashed with argon2id at rest

Full design notes are in [`llamanager-spec.md`](llamanager-spec.md).

## platforms

| os      | status         | auto-start                                                                              | `llama-server` binary                          |
|---------|----------------|-----------------------------------------------------------------------------------------|------------------------------------------------|
| macos   | primary target | `llamanager install-launchd`                                                            | Homebrew `llama.cpp` (Metal)                   |
| linux   | supported      | `llamanager install-systemd`                                                            | distro / source / ROCm / Vulkan                |
| windows | supported      | `install-windows-service` (real service, pywin32) or `install-windows` (Task Scheduler) | `llama-server.exe` (Vulkan or ROCm via HIP SDK)|

Python 3.11 or newer is required. The daemon uses `tomllib` and recent asyncio features.

## prerequisites

llamanager requires `llama-server` — the inference engine from [llama.cpp](https://github.com/ggerganov/llama.cpp) — to be installed separately. llamanager manages and proxies it; it does not bundle the binary.

| os      | easiest install                                    |
|---------|----------------------------------------------------|
| macOS   | `brew install llama.cpp`                           |
| Linux   | auto-install via the llamanager UI (CPU build), or download a CUDA/ROCm/Vulkan build from [releases](https://github.com/ggerganov/llama.cpp/releases) |
| Windows | auto-install via the llamanager UI (AVX2 CPU build), or download a CUDA build from [releases](https://github.com/ggerganov/llama.cpp/releases) |

After installing llamanager, open **http://localhost:7200/ui/setup** to verify detection, set the binary path if needed, or trigger an automatic install directly from the UI.

If `llama-server` is already installed but not on `PATH`, you can point llamanager at it by setting `llama_server_binary` in `~/.llamanager/config.toml`, or using the path field on the Setup page.

## install

```bash
python3 -m venv .venv                      # python 3.11 or newer required
source .venv/bin/activate                  # macos / linux
# .\.venv\Scripts\Activate.ps1             # windows powershell
# .\.venv\Scripts\activate.bat             # windows cmd

pip install -e '.[dev]'
pytest
```

After install, `llamanager` is on `PATH` inside the venv:

- macos / linux: `.venv/bin/llamanager`
- windows: `.venv\Scripts\llamanager.exe`

You can also call it without activating the venv:

```bash
.venv/bin/llamanager serve                 # macos / linux
.venv\Scripts\llamanager.exe serve         # windows
```

## first run

```bash
llamanager init-config                     # writes ~/.llamanager/config.toml
llamanager serve                           # foreground, ctrl-c to stop
```

The first launch prints a bootstrap admin key to stdout:

```
==============================================================================
  llamanager BOOTSTRAP ADMIN KEY (shown ONCE — won't be displayed again)
  lm_2n8RkFf...
  Use it to create real origins via /admin/origins, then revoke 'bootstrap'.
==============================================================================
```

Copy it now. Only the argon2id hash is stored. If you lose the key before creating a second admin origin, your only recovery is to delete `~/.llamanager/state.db` and start over.

Default ports:

| component    | port | bound to                                        |
|--------------|------|-------------------------------------------------|
| llamanager   | 7200 | `127.0.0.1` (loopback only by default)          |
| llama-server | 7201 | `127.0.0.1` (loopback only, never exposed)      |

Open the web UI at <http://localhost:7200/ui/login> and paste the bootstrap key.

> **Exposing on a LAN or Tailnet.** llamanager binds to loopback by default because bearer tokens travel in cleartext. To listen elsewhere, set `[server].bind = "0.0.0.0"` (or a specific IP) in `config.toml` and put a TLS-terminating proxy in front of it (Caddy, nginx, or `tailscale serve`). Do not put plain HTTP on a hostile network.

## quick dev uninstall & reinstall commands (macOS)

```bash
  deactivate
  rm -rf .venv
  python3.14 -m venv .venv
  source .venv/bin/activate
  pip install -e '.[dev]'
  llamanager serve
```

## pulling a model

Either through the web UI (Models → "Pull a model") or the API:

```bash
curl -X POST http://localhost:7200/admin/models/pull \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"source": "hf://unsloth/Qwen3.5-4B-GGUF", "files": ["Q4_K_M.gguf"]}'
```

The response returns a `download_id`. Poll `GET /admin/downloads/{id}` for progress.

## listing available models

Query the OpenAI-compatible models endpoint to see what is on disk:

```bash
curl http://localhost:7200/v1/models \
  -H "Authorization: Bearer $ORIGIN_KEY"
```

Or use the admin endpoint for more detail (size, source, sha256):

```bash
curl http://localhost:7200/admin/models \
  -H "Authorization: Bearer $ADMIN_KEY"
```

## sending an inference request

Any OpenAI-compatible client works. With `curl`:

```bash
curl -N http://localhost:7200/v1/chat/completions \
  -H "Authorization: Bearer $ORIGIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b",
    "stream": true,
    "messages": [{"role": "user", "content": "Hi!"}]
  }'
```

### requesting a specific model

By default, requests use whatever model is currently loaded (or the default profile if nothing is running yet). To request a different model, add the `X-Llamanager-Model` header. llamanager will hot-swap automatically if needed.

**By model ID** (the path relative to the models directory):

```bash
curl -N http://localhost:7200/v1/chat/completions \
  -H "Authorization: Bearer $ORIGIN_KEY" \
  -H "X-Llamanager-Model: unsloth/Qwen3.5-4B-GGUF/Q4_K_M.gguf" \
  -H "Content-Type: application/json" \
  -d '{"model": "any", "messages": [{"role": "user", "content": "Hi!"}]}'
```

**By profile name** (uses the profile's model, mmproj, and args):

```bash
curl -N http://localhost:7200/v1/chat/completions \
  -H "Authorization: Bearer $ORIGIN_KEY" \
  -H "X-Llamanager-Model: profile:qwen35-4b-vision" \
  -H "Content-Type: application/json" \
  -d '{"model": "any", "messages": [{"role": "user", "content": "Hi!"}]}'
```

Each origin's `allowed_models` setting restricts which models it can request. Set to `*` to allow any model, `default` to allow only the default, or a comma-separated list of specific model IDs.

While a request is queued or a model swap is happening, llamanager emits SSE comment lines (`: status=swapping_model`, `: keepalive`) every 10 seconds so client connections do not time out.

## chat in the browser

llamanager includes a built-in chat interface at <http://localhost:7200/chat>. Any user with a valid origin API key can use it. No admin access required.

Features:

- streaming responses with real-time token display
- chat history stored in the browser (localStorage), with multiple conversations
- system prompt configuration
- profile and model selection per conversation
- markdown rendering toggle
- dark/light theme

The admin panel also has a chat page at `/ui/chat` that uses the admin session key automatically.

## auto-start at boot or login

### macos, launchd

```bash
llamanager install-launchd
launchctl load -w ~/Library/LaunchAgents/com.llamanager.plist
```

To unload:

```bash
launchctl unload -w ~/Library/LaunchAgents/com.llamanager.plist
```

### linux, user systemd unit

```bash
llamanager install-systemd
systemctl --user daemon-reload
systemctl --user enable --now llamanager.service
journalctl --user -u llamanager.service -f
```

For a system-wide service, copy the generated unit file to `/etc/systemd/system/` and use `sudo systemctl ...`.

### windows

Two options. Pick a real Windows service if you want it always-on with no logon required, or a Task Scheduler entry if you want fewer dependencies.

#### option A: real Windows service (recommended for headless boxes)

Needs the `pywin32` extra and an elevated PowerShell or cmd.

```powershell
pip install -e ".[windows-service]"

# from an elevated shell:
llamanager install-windows-service
```

The service registers as `llamanager` ("llamanager — manager and proxy for llama-server"), startup `auto`. Logs land in `%USERPROFILE%\.llamanager\logs\llamanager.log`. Lifecycle events also show up in the Windows Event Viewer under Windows Logs → Application.

```powershell
sc query llamanager
python -m llamanager.win_service start
python -m llamanager.win_service stop
python -m llamanager.win_service restart
llamanager remove-windows-service
```

To run as a domain account instead of `LocalSystem`:

```powershell
llamanager install-windows-service --username "DOMAIN\svc-llamanager" --password "..."
```

> **Important:** before installing the service for the first time, run `llamanager serve` once interactively so the bootstrap admin key prints to a console you can copy from. The service has no stdout. If it generates the bootstrap key, the key is gone.

#### option B: Task Scheduler (no extra deps, logon-triggered)

```powershell
llamanager install-windows
schtasks /Create /XML "$env:USERPROFILE\.llamanager\llamanager.task.xml" /TN llamanager
schtasks /Run /TN llamanager
```

The task is logon-triggered (closer to the macOS LaunchAgent than to a system service), retries 5 times on failure with a 1 minute interval, and runs at LeastPrivilege as the current user. It is the right pick for desktop or single-user setups where pulling in Visual Studio Build Tools for `pywin32` is overkill.

```powershell
schtasks /Delete /TN llamanager /F
```

## cli

Daemon and installer commands:

```
llamanager serve [--host ...] [--port ...] [--log-level info]
llamanager init-config [--path PATH]
llamanager status                    # prints last persisted runtime.json
llamanager install-launchd [--label com.llamanager] [--port 7200] [--binary PATH]
llamanager install-systemd [--unit llamanager.service] [--port 7200] [--binary PATH]
llamanager install-windows [--task llamanager] [--port 7200] [--binary PATH]
llamanager install-windows-service [--startup auto|manual|delayed|disabled]
                                   [--username USER] [--password PASS] [--no-start]
llamanager remove-windows-service
llamanager --config /path/to/config.toml <subcommand>
```

### admin verbs (drive a running daemon)

These talk to a running `llamanager serve` over `/admin/*`, so an agent or
shell script can manage models, queue, and origins without a custom client.
They print JSON to stdout (pipe through `jq` if you want to slice the result).

Auth resolution order:

1. `--admin-key` flag
2. `$LLAMANAGER_ADMIN_KEY`
3. `admin_key` under a `[cli]` section in `config.toml`

Base URL resolution order:

1. `--url` flag
2. `$LLAMANAGER_URL`
3. derived from config (`http://<bind>:<port>`, with `0.0.0.0` rewritten to
   `127.0.0.1` since the CLI usually runs on the same host)

```
llamanager server status                    # full daemon snapshot
llamanager server start --profile P         # start llama-server
llamanager server stop
llamanager server restart [--profile P | --model M]
llamanager server swap --profile P [--arg key=value ...]

llamanager models list
llamanager models pull <hf://user/repo> [--file Q4_K_M.gguf ...]
llamanager models delete <repo/file.gguf> [--force]

llamanager downloads list
llamanager downloads get <id>
llamanager downloads cancel <id>

llamanager queue list
llamanager queue cancel <request_id>
llamanager queue pause | resume

llamanager origins list
llamanager origins create <name> [--priority N] [--allowed M ...] [--admin]
llamanager origins delete <id>
llamanager origins rotate-key <id>

llamanager events [--limit 200]
llamanager disk
llamanager reload
llamanager logs [--source llama-server|llamanager] [--tail 200]
```

Example: an agent that wants to swap models before a long batch and revert
after can do this without writing any HTTP code:

```bash
export LLAMANAGER_ADMIN_KEY=lm_...
llamanager server swap --profile qwen35-4b-vision
llamanager models list | jq '.[].model_id'
llamanager queue list
llamanager server swap --profile qwen35-4b-default
```

## configuration

Lives at `~/.llamanager/config.toml` (windows: `%USERPROFILE%\.llamanager\config.toml`). Hot-reload with `POST /admin/reload` or `SIGHUP` (POSIX only). Full schema is in spec §7.

A minimal example:

```toml
[server]
bind = "0.0.0.0"
port = 7200
llama_server_binary = "llama-server"   # or "C:\\path\\to\\llama-server.exe"
llama_server_port = 7201
data_dir = "~/.llamanager"

[defaults]
model = "unsloth/Qwen3.5-4B-GGUF/Q4_K_M.gguf"
profile = "qwen35-4b-default"

[profiles.qwen35-4b-default]
model = "unsloth/Qwen3.5-4B-GGUF/Q4_K_M.gguf"
args = { ctx-size = 16384, temp = 0.7, alias = "qwen3.5-4b" }
```

## api

| path        | purpose                                 | auth               |
|-------------|-----------------------------------------|--------------------|
| `/v1/*`     | OpenAI-compatible inference (queued)    | bearer (any key)   |
| `/admin/*`  | control plane (lifecycle, models, etc.) | bearer (admin key) |
| `/ui/*`     | web UI                                  | cookie session     |
| `/health`   | liveness check                          | none               |

Full endpoint list is in spec §5.

## filesystem layout

```
~/.llamanager/
├── config.toml             static config (you edit this)
├── state.db                sqlite: origins, requests, downloads, events
├── runtime.json            current run state (atomic writes)
├── .session-secret         signed-cookie secret (mode 0600)
├── models/                 downloaded GGUFs
├── logs/
│   ├── llamanager.log      llamanager's own log (rotated 5 × 50 MB)
│   └── llama-server.log    captured stdout/stderr of the child
└── llamanager.task.xml     windows: generated by `install-windows` (if used)
```

On windows, `~` resolves to `%USERPROFILE%`, e.g. `C:\Users\<you>\.llamanager`.

## troubleshooting

**`llama-server: command not found` on launch.** The binary is not on `PATH`. Either add it, or set `[server].llama_server_binary` in `config.toml` to the absolute path.

**Port 7200 or 7201 already in use.** Change `[server].port` (llamanager) or `[server].llama_server_port` (the upstream `llama-server`) in `config.toml`. Make sure the same port shows up under any matching `args = { ... port = ... }` profile entry.

**Lost the bootstrap key.** No recovery, argon2id is one-way. Stop the daemon, delete `~/.llamanager/state.db`, restart, and a new bootstrap key prints. Existing origins and keys go with it, so this is only safe if you have not created real origins yet.

**Web UI says "invalid admin key" but the key is admin.** The verification cache is per-process. If you just rotated the key, restart the daemon or wait 5 minutes for the cache TTL.

**Crash loop, `state: crashed` and 503s.** The 3-in-5 restart policy gave up. Check `~/.llamanager/logs/llama-server.log`, fix the cause, then `POST /admin/server/start` (or use the UI).

**Windows: argon2-cffi fails to install.** Python 3.11+ ships prebuilt wheels for argon2-cffi on Windows. If you hit a build error, upgrade pip (`python -m pip install -U pip`) and retry. As a last resort, install Microsoft's [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
