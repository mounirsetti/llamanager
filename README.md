<!-- markdownlint-disable MD033 MD041 MD060 -->
<!-- MD033: the banner below uses <picture>/<img> for theme-aware logos
     and badge images, which Markdown can't express natively.
     MD041: the file's first visible line is that banner, not an h1.
     MD060: some tables mix narrow rows (short commands) with very wide
     ones (long install URLs); aligning every pipe to the longest row
     would force excessive padding for no readability gain. -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg">
    <img alt="llamanager" src="assets/logo.svg" width="340">
  </picture>
</p>

<p align="center">
  <strong>One open-source server to install, serve, and manage local LLMs and diffusion models.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache 2.0 License"></a>
  <img src="https://img.shields.io/badge/version-0.3.994-green.svg" alt="Version 0.3.994">
  <img src="https://img.shields.io/badge/python-3.11+-3776ab.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg" alt="Platforms">
</p>

<p align="center">
  Created by <a href="https://github.com/mounirsetti">Mounir Ould Setti</a> at <a href="https://soulthread.group">SoulThread Technologies</a>
</p>

---

llamanager runs on a single host you already own. It installs the inference engines, downloads model weights from Hugging Face, supervises the processes, queues requests with per-origin priorities, and exposes everything behind one OpenAI-compatible (and Anthropic-compatible) endpoint. Text and image families share the same dashboard, the same queue, and the same auth.

The text side wraps `llama-server` (from llama.cpp) plus `mlx-lm` on Apple Silicon. The image side runs three diffusion stacks: HiDream-O1-Image, FLUX 2 via `sd.cpp`, and Z-Image (Tongyi-MAI), with Z-Anime supported as a Z-Image fine-tune. New engines plug in as small adapter modules.

<p align="center">
  <img src="assets/screenshot-llm-models.png" alt="llamanager LLM models page — installed models with sizes and per-model profiles, the loaded model highlighted, and the sticky text + image model picker across the top" width="900">
</p>
<p align="center">
  <sub>The <strong>LLM models</strong> page: every installed model with its size and engine, expandable per-model profiles, and the sticky model picker for both the text and image families up top.</sub>
</p>

> **New in 0.3.9 · beta.** **Multi-slot LLM** lets you keep multiple language models warm at the same time, each on its own port. Routing is by model id — no swap penalty when the requested model is already loaded. Off by default; enable from the **Slots (beta)** page or `llamanager slots enable`. See [Multi-slot LLM (beta)](#multi-slot-llm-beta).

## Table of contents

- [What it is](#what-it-is)
- [Supported engines](#supported-engines)
- [Platforms and GPUs](#platforms-and-gpus)
- [Install](#install)
- [Quick start](#quick-start)
- [The model picker top bar](#the-model-picker-top-bar)
- [LLM models](#llm-models)
- [Multi-slot LLM (beta)](#multi-slot-llm-beta)
- [Diffusion models](#diffusion-models)
  - [Install dependencies](#install-dependencies)
  - [Download models](#download-models)
  - [Reference images (editing, composition, img2img)](#reference-images-editing-composition-img2img)
  - [Sharing the GPU with the text engine](#sharing-the-gpu-with-the-text-engine)
- [Calling the API](#calling-the-api)
  - [Anthropic-compatible API](#anthropic-compatible-api)
  - [Reasoning / thinking control](#reasoning--thinking-control)
- [Chat in the browser](#chat-in-the-browser)
- [Auto-start at boot or login](#auto-start-at-boot-or-login)
- [Uninstall](#uninstall)
- [CLI](#cli)
- [Configuration](#configuration)
- [Filesystem layout](#filesystem-layout)
- [Troubleshooting](#troubleshooting)
- [Releasing a new version](#releasing-a-new-version)
- [Credits](#credits)
- [License](#license)

---

## What it is

`llama-server` and the various diffusion CLIs are good at running one model at a time. None of them are good at being shared. Two clients hitting `llama-server` directly will collide on context, you can't swap models without restarting, image stacks need their own Python environments, and there's no auth on either side. llamanager fills that gap on a single machine: a queue, a supervisor, a key per client, a one-stop UI for installing dependencies and pulling weights, and a coexistence policy so a diffusion request doesn't blow up the LLM you have loaded.

What you get out of the box:

- **OpenAI-compatible `/v1/*` proxy** for chat, completions, and image generations. Any existing OpenAI client library works.
- **Anthropic-compatible `/anthropic/v1/*` proxy** for the Messages API, token counting, and model listing. Point the official `anthropic` SDK at llamanager via `base_url` and it just works — including streaming, tool use, and base64 image inputs.
- **Per-engine install flow** in the web UI. Click `Install dependencies` for HiDream or Z-Image and llamanager creates a Python venv under `~/.llamanager/venvs/<engine>/` and pip-installs the right packages. The disk footprint and live install log are visible on the page.
- **Model download manager** that pulls whole HF repos (or a single subfolder, useful for monster repos like SeeSee21/Z-Anime where only the `diffusers/` subtree is needed). Progress streams to the UI every 2 s.
- **Per-origin priority queue** with cancellation that propagates all the way to the running subprocess. A long batch can't block your editor; cancelling a queued or in-flight image task actually stops the work.
- **Family-aware concurrency.** Text and image are mutually exclusive by default (they fight over the same GPU), but the policy is one toggle to switch when you have the VRAM headroom.
- **Hot-swap by header.** Requests can name a specific model via `X-Llamanager-Model`; the queue swaps the loaded model in-flight without dropping in-flight work.
- **Multi-slot LLM (beta, since 0.3.9).** Opt-in mode that keeps several models warm in parallel slots, each on its own port. Requests route by model id — no swap penalty. Mutually exclusive with the host-wide "exclusive mode" sweep; managed from the **Slots** page or the `llamanager slots` CLI. See [Multi-slot LLM (beta)](#multi-slot-llm-beta).
- **Crash supervisor** with a 3-in-5-minutes restart cap, applied to text servers and image engines alike.
- **Sticky model picker** at the top of every page: pick the loaded LLM, save default LLM and default diffusion model, with a live "loaded" indicator.
- **HTMX web UI** that works from a phone over Tailscale or any LAN.
- **Bearer-token auth.** One key per origin, argon2id-hashed at rest. Each origin has an operator on/off switch — a disabled origin still authenticates but is refused at the submission paths with a 403, so you can park a client without rotating or deleting its key.
- **Model-loading lock.** A single Settings toggle freezes the engine on whatever's currently loaded: requests the loaded model can serve still run, but any request that would start the engine, swap models, or change profile is rejected instead. Protects a warm cache or a debugging session from being yanked out from under you.

Full design notes are in [`llamanager-spec.md`](llamanager-spec.md).

## Supported engines

| family    | engine                       | inference path                          | install flow                                |
|-----------|------------------------------|-----------------------------------------|---------------------------------------------|
| text      | `llama` (llama.cpp)          | persistent HTTP server                  | auto-install from the LLM engines page      |
| text      | `mlx`                        | persistent HTTP server (Apple Silicon)  | manual: `pip install mlx-lm`                |
| diffusion | `hidream` (HiDream-O1-Image) | one-shot Python subprocess              | auto-install venv + deps                    |
| diffusion | `flux2` (FLUX 2 via sd.cpp)  | one-shot `sd-cli` binary                | manual: download from sd.cpp releases       |
| diffusion | `z_image` (Z-Image / Z-Anime)| one-shot Python subprocess (diffusers)  | auto-install venv + deps                    |

Adding a new engine is a single Python module in [`llamanager/engines/`](llamanager/engines/) plus three lines of registration. The existing five live there as references.

## Platforms and GPUs

| os      | status         | auto-start                                                                              |
|---------|----------------|-----------------------------------------------------------------------------------------|
| macOS   | primary target | `llamanager install-launchd`                                                            |
| Linux   | supported      | `llamanager install-systemd`                                                            |
| Windows | supported      | `install-windows-service` (real service via pywin32) or `install-windows` (Task Scheduler) |

Python 3.11 or newer is required.

The dashboard auto-detects your GPU and reports VRAM. RAM and VRAM are summed into one capacity estimate that the model-size suggestions use. On Apple Silicon the two are the same unified pool, so they're counted once.

| GPU vendor       | how llamanager reads it                              | notes                                                  |
|------------------|------------------------------------------------------|--------------------------------------------------------|
| Apple Silicon    | `system_profiler`                                    | Unified memory                                         |
| NVIDIA           | `nvidia-smi`                                         | Reports total + free VRAM live                         |
| AMD              | `rocm-smi` on Linux, Windows registry + PDH counters | Linux: ROCm runtime required. Windows: works via Vulkan/HIP without rocm-smi |
| Intel Arc / DC   | `xpu-smi` from [XPU Manager](https://github.com/intel/xpumanager) | oneAPI runtime required                                |

If none of those tools are available, llamanager falls back to system RAM for capacity estimates and shows "No compatible GPU detected" on the dashboard. CPU inference still works; it's just slower.

On Windows specifically: AMD users running llama.cpp's Vulkan build no longer need `rocm-smi`. llamanager reads the adapter name and VRAM from the driver's registry entries and live VRAM usage from the `\GPU Adapter Memory(*)\Dedicated Usage` performance counter. Per-process VRAM (which process is holding what) is queried from `\GPU Process Memory(*)\Dedicated Usage` and shown alongside per-process RAM on the dashboard.

After installing llamanager, open <http://localhost:7200/ui/setup> to verify detection. The LLM engines page lets you install `llama-server` directly without leaving the browser, or point at an existing binary.

### Alternative LLM engines

llamanager supports installing compatible `llama.cpp` forks side-by-side with the default engine. Switching does not remove the original install. Both binaries live under `~/.llamanager/bin/` and the active one is selected from the LLM engines page.

| engine | what it adds | install location |
|--------|-------------|-----------------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) (default) | Official CPU/Metal/CUDA build | `~/.llamanager/bin/llama-server` |
| [Atomic TurboQuant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) | TurboQuant compression (2–4 bit KV cache, up to 6.4x compression vs FP16) and Gemma 4 MTP speculative decoding | `~/.llamanager/bin/atomic/llama-server` |

When using Atomic TurboQuant, add its flags to a profile's args:

```toml
[models."your-model.gguf".profiles.turbo]
args = { ctx-size = 16384, ctk = "turbo3", ctv = "turbo3", fa = true }
```

For MTP speculative decoding with Gemma 4:

```toml
[models."gemma4-target.gguf".profiles.mtp]
args = { ctx-size = 16384, mtp-head = "gemma4-assistant.gguf", spec-type = "mtp" }
```

### Updating engines

Every managed engine can be updated from the UI and the CLI, and each one has a
per-engine **Auto-update when idle** switch.

- **Manual update + check.** On the **LLM engines** page (`/ui/setup`) each
  installed variant has *Check for updates* → *Update to vX*. On the
  **Diffusion engines** page, *Reinstall dependencies* re-runs the pip install
  (= an update). llamanager itself has *Check for updates* / *Update* on
  `/ui/about`. The CLI mirrors all of it:

  ```bash
  llamanager setup check-updates                 # all installed LLM variants: installed vs latest
  llamanager setup check-updates --variant llama.cpp-cuda
  llamanager setup install-llama-server --backend cuda    # update the variant to latest
  llamanager diffusion versions z_image          # diffusion: installed vs target (+ has_update)
  llamanager diffusion install z_image           # reinstall/update diffusion deps
  llamanager update --check                      # llamanager itself: is a newer release out?
  llamanager update                              # update llamanager + restart
  ```
- **See versions + install any version (downgrade).** Each installed variant
  card has a *Versions…* button that lists the builds available upstream
  (GitHub releases for llama.cpp/Atomic, PyPI for MLX) — filtered to ones that
  ship a binary for your backend/platform — and lets you install (or roll back
  to) a specific one. Diffusion cards have the same for the `diffusers` version.
  From the CLI:

  ```bash
  llamanager setup engine-versions llama.cpp-cuda            # list installable builds
  llamanager setup install-llama-server --backend cuda --version b1234   # install/roll back to a tag
  llamanager diffusion versions z_image                      # list diffusers versions
  llamanager diffusion install z_image --diffusers-version 0.37.1   # pin a diffusers version
  llamanager diffusion install z_image --reset-diffusers     # back to the tested pin
  ```

  Picking a `diffusers` version **overrides the tested pin** for that engine
  (an untested version can break image generation). The override is persisted
  and becomes the auto-update *target*, so a deliberate downgrade isn't silently
  re-bumped; *Reset to pin* (or `--reset-diffusers`) clears it.
- **Auto-update when idle.** Flip the switch next to any engine and llamanager
  checks upstream on a fixed cadence and, once the service has been idle (no
  in-flight or pending requests) for the configured window, applies the update
  automatically. If the engine being updated is the **active** one with a model
  loaded, llamanager unloads it, swaps the binary (required on Windows, where a
  running `.exe` is locked), then reloads the model.

Engine keys are a llama variant id (`llama.cpp-cuda`, `atomic-vulkan`,
`mlx-apple-silicon`), a diffusion engine name (`hidream`, `z_image`), or
`llamanager` for the service's own self-update. Configure it under
`[auto_update]` (see [Configuration](#configuration)) or:

```bash
llamanager setup auto-update list                       # show all switches + cadence
llamanager setup auto-update llama.cpp-cuda on          # enable one engine
llamanager setup auto-update z_image off
llamanager setup auto-update llamanager on              # auto-update llamanager itself
llamanager setup auto-update settings --idle-seconds 300 --check-interval-seconds 21600
```

How "an update exists" is decided per engine:

- **llama.cpp / Atomic** compare the installed build against the latest GitHub
  release tag; **MLX** against the latest PyPI version.
- **llamanager** compares against the latest GitHub release tag (editable
  installs are skipped — update the checkout with `git pull` yourself).
- **Diffusion engines** (`hidream`, `z_image`) track the `diffusers` release
  llamanager pins and is tested against (currently `0.38.0`, the first release
  that ships `ZImagePipeline`). Auto-update fires only when the engine's
  installed `diffusers` is *older* than that pin — which happens when you
  update llamanager to a build that bumped the pin. It never chases git `main`
  or jumps ahead of the tested release, and only fires for an already-installed
  engine. (`flux2` has no auto-install path, so no switch.)

## Install

Clone the repo, or grab a zip from the [releases page](https://github.com/mounirsetti/Llamanager/releases) if you don't have git.

```bash
git clone https://github.com/mounirsetti/Llamanager.git
cd Llamanager
```

Pin to a tagged release:

```bash
git clone --branch "v0.3.994" --depth 1 https://github.com/mounirsetti/Llamanager.git
cd Llamanager
```

Update an existing checkout:

```bash
git pull origin main
```

Install the package and run the tests:

```bash
python3 -m venv .venv                      # python 3.11 or newer required
source .venv/bin/activate                  # macos / linux
# .\.venv\Scripts\Activate.ps1             # windows powershell
# .\.venv\Scripts\activate.bat             # windows cmd

pip install -e '.[dev]'
pytest
```

`llamanager` is on `PATH` inside the venv:

- macOS / Linux: `.venv/bin/llamanager`
- Windows: `.venv\Scripts\llamanager.exe`

You can call it without activating the venv:

```bash
.venv/bin/llamanager serve                 # macos / linux
.venv\Scripts\llamanager.exe serve         # windows
```

### Getting `git` on your system

| os      | install command                                                                                                 |
|---------|-----------------------------------------------------------------------------------------------------------------|
| macOS   | `xcode-select --install` (ships git), or `brew install git`                                                     |
| Linux   | Debian/Ubuntu: `sudo apt install git` · Fedora: `sudo dnf install git` · Arch: `sudo pacman -S git`             |
| Windows | `winget install --id Git.Git -e`, or download from [git-scm.com](https://git-scm.com/download/win) |

On Windows, run the clone command from Git Bash, PowerShell, or Windows Terminal. The installer adds `git` to `PATH` by default.

## Quick start

One guided command gets you running — it writes config, creates your admin key, and starts the service + tray icon immediately (no restart):

```bash
llamanager init
```

`init` does two steps, then launches:

1. **Config** — writes `~/.llamanager/config.toml` if it's missing.
2. **Admin key** — creates the bootstrap admin key and prints it (also saved to `~/.llamanager/bootstrap-key.txt`, mode 0600). Capturing it here is reliable even if you later run headless — the old "the service swallowed the key" trap is gone.

It then starts the service (if not already running) and the tray/menu-bar icon, so the icon appears right away. Pass `--no-launch` to skip that and just write config + key.

```text
==============================================================================
  BOOTSTRAP ADMIN KEY (shown once — also saved to ~/.llamanager/bootstrap-key.txt)
  lm_2n8RkFf...
  Paste it at /ui/login, create a real origin, then revoke 'bootstrap'.
==============================================================================
```

Then open <http://localhost:7200/ui/login> and paste the key. **Run-at-startup, engine install, and model downloads all live in the UI:** choose how it starts from the tray icon (right-click → **Autorun at startup**) or the Setup page (see [Auto-start](#auto-start-at-boot-or-login)); the dashboard shows a checklist (install engine → `/ui/setup`, download a model → `/ui/models`, create an API key, set autorun) for whatever's still left.

<details>
<summary>Manual equivalent (if you'd rather do the steps yourself)</summary>

```bash
llamanager init-config                     # writes ~/.llamanager/config.toml
llamanager serve                           # foreground, ctrl-c to stop (prints the key)
```

Only the argon2id hash of the key is stored. If you lose the key before creating a second admin origin, the only recovery is deleting `~/.llamanager/state.db` and starting over.
</details>

Default ports:

| component    | port | bound to                                        |
|--------------|------|-------------------------------------------------|
| llamanager   | 7200 | `127.0.0.1` (loopback only by default)          |
| llama-server | 7201 | `127.0.0.1` (loopback only, never exposed)      |

Open the web UI at <http://localhost:7200/ui/login> and paste the bootstrap key.

### Reaching it from other devices (Tailscale, LAN, etc.)

By default llamanager binds to `127.0.0.1`. To reach it from a phone, another laptop, or any device on your Tailscale network, set this in `~/.llamanager/config.toml`:

```toml
[server]
bind = "0.0.0.0"
```

Restart and access at `http://<your-ip>:7200/ui/`.

## The model picker top bar

Every page in the admin UI has a sticky model-picker strip at the top, designed like a mixer-channel: status sigil, model selector, profile selector, then an action cluster.

The LLM lane has a pulsing dot when a model is loaded, the model and profile dropdowns, a primary `Load` button, an `Unload` icon, and a star toggle that saves the current selection as the default LLM. Clicking the filled star clears the default; clicking the hollow one sets it.

When [Multi-slot LLM (beta)](#multi-slot-llm-beta) is enabled, a scrollable strip appears below the LLM lane showing one row per slot with its currently-loaded model. The strip caps at ~3 visible rows then scrolls; the whole strip is clickable and jumps to the Slots page. The **Load** button on the LLM models page becomes a dropdown listing every slot (`Load into slot N`, or `slot N · <model>` when already loaded).

The diffusion lane has the same shape with one action: a star toggle that saves the selected model + profile as the default diffusion model. There's no `Load` for diffusion because the engines are one-shot per request; "default" means "use this when `/v1/images/generations` omits the model field".

Setting both defaults is the fastest way to use llamanager from a client that doesn't know about your specific model IDs: send a request without a model and the right engine picks it up.

## LLM models

The LLM models page (formerly just "Models") manages text-family models for the `llama` and `mlx` engines. Pull, organise, set the default, edit per-model profiles.

Either from the UI (LLM models → "Find a language model" or "Pull a language model") or the API:

```bash
curl -X POST http://localhost:7200/admin/models/pull \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"source": "hf://bartowski/Llama-3.2-1B-Instruct-GGUF", "files": ["Llama-3.2-1B-Instruct-Q4_K_M.gguf"]}'
```

The response returns a `download_id`. Poll `GET /admin/downloads/{id}` for progress.

To see what's on disk:

```bash
curl http://localhost:7200/v1/models \
  -H "Authorization: Bearer $ORIGIN_KEY"
```

Or the admin endpoint, which adds size / source / sha256:

```bash
curl http://localhost:7200/admin/models \
  -H "Authorization: Bearer $ADMIN_KEY"
```

llamanager auto-creates a default profile when you pull a model. Add more from the LLM models page or in `config.toml` (see [Configuration](#configuration)).

## Multi-slot LLM (beta)

> **Status:** beta as of **0.3.9**. The wire format and config keys are
> considered stable; the UI dashboard and crash-recovery semantics may
> still evolve. Off by default; opt in from the **Slots** page or
> `llamanager slots enable`.

### What it solves

Single-instance llamanager loads one LLM at a time. Every cross-model
request pays a swap (stop, free VRAM, restart, warm) — fine when one
model fits your workflow, painful when you alternate between, say, a
fast 7B for autocomplete and a larger 30B for code review.

Multi-slot keeps several `llama-server` processes alive in parallel,
each on its own port, each holding a different model. Requests are
routed by `model` id — when the model is already warm in some slot,
there's no swap and no warm-up cost. When it isn't, the request is
rejected with a clear error (no auto-eviction in v1 — the operator
owns slot assignments).

### Quick start

1. Open **Slots (beta)** in the left nav.
2. Flip the master switch ON. Exclusive mode is force-disabled if it
   was on — the two features are mutually exclusive (see "Caveats").
3. Click **Add slot** to allocate slot 1 (port 7202). Repeat up to
   `multi_slot_max` (default 4).
4. For each slot, pick a model from the dropdown and click **Load**.
5. Send chat requests with `X-Llamanager-Model: <model id>` (or with
   `model` in the JSON body). The dispatcher routes to whichever slot
   holds that model.

From the CLI:

```bash
llamanager slots enable
llamanager slots add                       # → {"id": 1, ...}
llamanager slots load 1 --model org/repo-7B-GGUF/Q4_K_M.gguf --profile fast
llamanager slots add                       # → {"id": 2, ...}
llamanager slots load 2 --model org/repo-30B-GGUF/Q4_K_M.gguf --profile balanced
llamanager slots list                      # full dashboard JSON
```

Slot 0 is always the legacy single-instance slot — it can never be
removed. Slots 1..N are added at runtime and persist in
`~/.llamanager/slots.json`, so they survive service restarts.

### Routing rules

| Scenario | What happens |
|----------|--------------|
| Request names a model loaded in some slot | Routes to that slot. No swap. |
| Request names a model loaded with a different profile | Swaps the profile **within that slot**. Other slots untouched. |
| Request names a model not in any slot | Rejected with `failed` status and a clear "model X is not loaded in any slot — load it via /ui/slots or `llamanager slots load`" message. |
| Request omits the model entirely | Rejected. Multi-slot requires the request to name a model — there's no ambiguous "default". |

The `X-Llamanager-Slot` header is **not** part of v1. Routing is
strictly by model id; if you load the same model into two slots
(e.g. one with a long-context profile and one with a fast profile),
only the first match is used. Distinguish them by giving each one a
unique model id alias or by using `X-Llamanager-Profile` against the
unique-named profile.

### Topbar + Load button

When multi-slot is on:

- A scrollable strip in the top bar shows one row per slot with its
  loaded model. The strip caps at ~3 rows visible; beyond that it
  scrolls vertically. Click the strip to jump to the Slots page.
- The **Load** button on the LLM models page becomes a dropdown
  listing every slot: `Load into slot N` for empty slots,
  `slot N · <current model>` for ones already holding something
  (clicking swaps to your new pick).

### Diffusion coexistence

By default, dispatching an image task while LLM slots are loaded will
unload **all** slots, run the image, then restart them — the same
"yield_to_image" behaviour as the legacy single slot, fanned out
across the pool.

If you have the VRAM headroom and want LLM slots to stay loaded
across image tasks, enable **"Allow diffusion alongside LLM slots"**
on the Slots page (or `llamanager slots coex on`). With this on:

```text
[image task arrives]
  → LLM slots stay loaded
  → image worker spawns on whatever VRAM is left
  → potential VRAM OOM is on you
```

This is independent of the legacy `[coexistence].allow_concurrent`
flag, which governs single-slot mutual exclusion. With multi-slot
on, only the new toggle applies.

### Persistence

Slot 0 keeps writing to `~/.llamanager/runtime.json` (legacy shape,
unchanged). Slots 1..N are described in a sibling
`~/.llamanager/slots.json`:

```json
{
  "version": 1,
  "slots": [
    {"id": 1, "port": 7202, "model_id": "org/repo-7B.gguf",  "profile": "fast",     "args": {}},
    {"id": 2, "port": 7203, "model_id": "org/repo-30B.gguf", "profile": "balanced", "args": {}}
  ]
}
```

On service start, llamanager re-creates each slot and (re)starts any
that have a `model_id`. On disable (toggling the master switch off),
slots 1..N are stopped but their entries are kept in `slots.json` so
you can re-enable and find your layout intact.

### Caveats

- **No auto-eviction.** Requests for a model not in any slot fail
  fast. This is deliberate — automatic LRU eviction would silently
  drop a model you wanted kept warm. Plan your slot assignments
  explicitly.
- **No live VRAM admission.** The dashboard shows the **combined
  on-disk file size** of loaded models as a VRAM-pressure proxy; the
  service won't refuse a slot load that "looks" too big. On the
  AMD AI PRO R9700 (32 GB) for example, two 13B-Q4 models fit
  comfortably; two 30B-Q4 models do not. Treat the combined-size
  number as your guardrail.
- **Mutex with exclusive mode.** When slots is on, `exclusive_mode`
  is force-disabled and the controls on the Launch page are greyed
  out with a note. The reverse also holds: trying to enable
  exclusive mode while slots is on returns HTTP 409. The two
  features address opposite problems (exclusive: "kill everyone
  else's GPU process"; slots: "run multiple of our own GPU
  processes") and are mutually exclusive by design.
- **Single-shared `--parallel`.** Within a single slot, llama-server
  still allocates one decoding slot per request unless you pass
  `--parallel N` via the profile args. Multi-slot is about parallel
  *models*, not parallel requests against one model.
- **`/admin/server/swap`** and the autolaunch path target slot 0
  only — they predate slots and we keep that contract for the
  legacy off-path. Use `llamanager slots load / swap` for slots
  1..N.
- **`/v1/models` and the Anthropic `/models` facade** union every
  loaded slot's model so SDK clients discover them all.
- **Crash supervisor is per-slot.** A flapping slot 2 enters its own
  exponential cooldown window without affecting slot 0's recovery.
- **Drain on disable / remove.** When you disable slots or remove a
  slot, in-flight requests on the affected slot are aborted after a
  short grace window (v1 takes the hammer route per spec).

### Configuration

```toml
[server]
multi_slot_enabled    = true     # master switch (write via /admin/slots/enable or the UI)
multi_slot_base_port  = 7201     # first port for the pool; slot 0 uses this
multi_slot_max        = 4        # hard cap on slot count

[coexistence]
allow_diffusion_with_slots = false   # when true AND slots is on,
                                     # image tasks leave LLM slots loaded
```

Toggling `multi_slot_enabled` via SIGHUP hot-reload is **not** safe —
slot lifecycles need explicit start/stop choreography. Always use
`POST /admin/slots/enable`, `llamanager slots enable|disable`, or
the UI toggle, which orchestrate the transition correctly.

## Diffusion models

Two pages cover the image side. The **Diffusion engines** page (`/ui/setup-diffusion`) is the one-stop shop for setup: per-engine setup cards, dependency installer, model downloader with progress, and the coexistence policy at the bottom. The **Diffusion models** page (`/ui/diffusion-models`) is where you manage what's actually installed: a catalog of known-good models joined against what's on disk, an Activate button to pick the dashboard/API default, and the per-image-model profile editor (CRUD + clone + materialize built-in defaults). The catalog rows for not-yet-installed entries link back to the engines page with the canonical HF repo pre-suggested.

Both pages also have CLI counterparts under `llamanager diffusion` — see [CLI](#cli).

Three engines are wired today:

| engine | layout | typical disk | typical VRAM | reference model |
|--------|--------|--------------|--------------|-----------------|
| Z-Image (Tongyi-MAI) | Diffusers pipeline (`model_index.json` + `transformer/`, `text_encoder/`, `vae/`) | ~20 GB | ~14 GB at bf16 | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) |
| HiDream-O1-Image | tokenizer + safetensors shards | ~33 GB | ~16 GB | [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) |
| FLUX 2 (sd.cpp) | flux*.gguf + ae.safetensors + text-encoder GGUF | varies by quant | ~12-27 GB | community GGUF re-hosts |

Z-Image's adapter also handles fine-tunes that ship the same Diffusers layout, including [SeeSee21/Z-Anime](https://huggingface.co/SeeSee21/Z-Anime). Z-Anime's full repo is 203 GB, so the download form lets you specify a `diffusers/` subfolder and pull only the runnable variant (~12-20 GB).

### Install dependencies

Each engine card on the Diffusion engines page has an `Install dependencies` button when an auto-install plan exists. Clicking it creates a fresh Python venv under `~/.llamanager/venvs/<engine>/`, runs `pip install` for the engine's package set, and writes the resulting interpreter path back to your config so the engine is ready to use immediately.

| engine | what gets installed | rough size |
|--------|---------------------|------------|
| Z-Image | torch, transformers, accelerate, huggingface_hub, safetensors, Pillow, sentencepiece, `diffusers==0.38.0` (first release shipping `ZImagePipeline`) | ~8.5 GB |
| HiDream | GPU-aware. On AMD: official ROCm wheels (torch+rocm7.2.1, torchvision, triton) from `repo.radeon.com` + pinned `transformers==4.57.1`, `accelerate==1.13.0`, `diffusers==0.38.0`, etc. On NVIDIA/CPU: generic CUDA/CPU torch + the same HF pins. | ~7.5–9 GB |
| FLUX 2 | (no auto-install — see below) | — |

The installer streams pip's stdout into the page, so you can watch it work and cancel mid-flight. Failures surface inline with the last 200 KB of log.

FLUX 2 uses the `sd-cli` binary from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), which ships per-backend builds (Vulkan, CUDA, ROCm) that are too platform-specific to auto-install reliably. Download the matching release zip, extract, and point the `sd-cli executable` field on the engine card at the binary.

HiDream auto-detects the GPU family (probes `/dev/kfd` for AMD ROCm, `nvidia-smi` for NVIDIA) and installs the matching wheel set. On AMD the card also offers a one-checkbox patch that flips `use_flash_attn: True` to `False` in `<hidream_repo>/models/pipeline.py` — required because the upstream pipeline hardcodes flash-attn, which isn't available on AMD ROCm. The patch keeps a `.bak`. If `<hidream_repo>` isn't set when you click install, the patch step is skipped with a note; set the path and re-run. On AMD, the installer also surfaces a warning if the llamanager process isn't a member of the `render` group, since HIP needs `/dev/kfd` access.

Z-Image's auto-install still picks a generic `torch` wheel — on AMD ROCm or Apple Silicon you should build that venv yourself with the right vendor wheels and point the engine at it.

### Download models

Each engine card has a Models section with two inputs: an HF repo (`org/name`) and an optional subfolder. The Download button kicks off a `huggingface_hub.snapshot_download` in the background, with a live progress bar that compares bytes-on-disk against an up-front size estimate from the HF API.

For Z-Image, the recommended downloads are:

```
Tongyi-MAI/Z-Image                                      # the base model, ~20 GB
SeeSee21/Z-Anime          subfolder: diffusers          # the anime fine-tune, ~12-20 GB
```

For HiDream, point at `HiDream-ai/HiDream-O1-Image` and leave the subfolder blank.

For FLUX 2, the canonical fp16 weights live at `black-forest-labs/FLUX.2-dev`; for runnable GGUF quants search Hugging Face for community re-hosts.

Downloads land in `~/.llamanager/models/<repo>/...` (or `<repo>/<subfolder>/...` for subfolder pulls). llamanager auto-detects the layout, registers the model with the right engine, and seeds default profiles on first detection (`z-image-fast`, `z-image-quality`; `hidream-dev`, `hidream-full`; `flux2-fast`, `flux2-quality`).

### Generating from the API

OpenAI-compatible:

```bash
curl -X POST http://localhost:7200/v1/images/generations \
  -H "Authorization: Bearer $ORIGIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Tongyi-MAI/Z-Image",
    "prompt": "A still life of three pears on a blue table",
    "size": "1280x1280",
    "n": 1,
    "response_format": "b64_json"
  }'
```

Omit `model` to use the saved default diffusion model from the top bar. Streaming (`"stream": true`) emits `: step=N/M` SSE comments while generating, then a final `data:` event with the result.

### Reference images (editing, composition, img2img)

`/v1/images/generations` accepts one or more reference images alongside the prompt. The engines interpret them differently:

| engine | refs | what it does |
|---|---|---|
| HiDream-O1-Image | 1 | Editing. With `--model_type dev` the flash scheduler is replaced with flow-match, shift drops to 1.0, and the ref is treated as the image being edited. Set `keep_original_aspect: true` to preserve the ref's aspect ratio and bypass the 2048-bucket snap. |
| HiDream-O1-Image | 2–8 | Composition / multi-subject. Optionally steer layout with `layout_bboxes`. |
| FLUX 2 / sd.cpp  | exactly 1 | img2img. Forwarded as sd-cli's `--init-img` with `--strength` controlling how much of the init image is preserved (`0.0` = exact copy, `1.0` = full re-generation). |
| FLUX 2 / sd.cpp  | 2+ | Rejected with 400. sd-cli's init-image path is single-slot. |
| Z-Image          | any | Currently ignored. The base Z-Image pipeline doesn't accept refs. |

Request fields (alongside the usual `prompt`, `model`, `size`, `n`, `seed`, `profile`, `response_format`, `stream`):

| field | type | meaning |
|---|---|---|
| `image` | base64 string or `data:image/...;base64,...` URL | Single reference image. Shorthand for `images: [image]`. |
| `images` | array of base64 strings / data URLs | Up to 8 reference images (HiDream); exactly 1 (Flux2). |
| `keep_original_aspect` | bool | HiDream only. With exactly one ref, resize it to max 2048 on the long side and use those dimensions for the output. |
| `layout_bboxes` | string (JSON) | HiDream only. Forwarded to `--layout_bboxes`, e.g. `"[[0.1,0.4,0.2,0.6]]"` (relative `x1,x2,y1,y2` per box). |
| `strength` | float in `[0, 1]` | Flux2 only. sd-cli img2img denoise strength. Default `0.75`. |

Reference bytes are decoded server-side and sniffed against PNG / JPEG / WebP magic bytes (a request 400s if the bytes don't match), capped at 20 MiB per image and 8 images per request, and staged to `~/.llamanager/refs/<request_id>/` for the duration of the run. A `finally` block deletes the staging directory regardless of how the run terminates (success, engine crash, queue cancel, timeout). Refs are not copied into the gallery, only the sidecar JSON next to each output preserves the prompt-and-ref provenance.

Example, HiDream editing:

```bash
REF_B64=$(base64 -w0 ./input.png)     # macOS: base64 -i ./input.png | tr -d '\n'

curl -X POST http://localhost:7200/v1/images/generations \
  -H "Authorization: Bearer $ORIGIN_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"HiDream-O1-Image\",
    \"profile\": \"hidream-dev\",
    \"prompt\": \"Replace the puppy with a small calico kitten, keep the wooden sign and garden background identical.\",
    \"image\": \"data:image/png;base64,$REF_B64\",
    \"keep_original_aspect\": true,
    \"seed\": 11,
    \"response_format\": \"url\"
  }"
```

Example, Flux2 img2img:

```bash
curl -X POST http://localhost:7200/v1/images/generations \
  -H "Authorization: Bearer $ORIGIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flux2-dev",
    "profile": "flux2-fast",
    "prompt": "Same composition, oil-painting style, heavy brush texture.",
    "image": "data:image/png;base64,'"$REF_B64"'",
    "strength": 0.55,
    "size": "1024x1024",
    "response_format": "url"
  }'
```

### Generating from the UI

Open <http://localhost:7200/ui/images>. Three-pane workspace: an on-disk gallery on the left (every PNG under `~/.llamanager/images/YYYY-MM-DD/<origin>/`, newest first, lazy-loaded), the selected image plus its sidecar metadata in the center, and a schema-driven composer on the right.

The composer auto-renders the right fields per engine by walking each adapter's `profile_schema()` — the same mechanism the Diffusion models page uses for profile editing. Pick a model, optionally pick one of its profiles to pre-fill the override placeholders, then override any individual field per-request. Generation streams back via SSE so the `<progress>` bar advances step-by-step rather than just toggling between "Queued" and "Done". When the request finishes, the new image is auto-prepended to the gallery and selected.

Selected images surface their full sidecar (model, profile, seed, size, steps, guidance, duration) plus `Reuse prompt` / `Reuse seed` shortcuts that push the value back into the composer. Generated PNGs and a sidecar JSON live under `~/.llamanager/images/YYYY-MM-DD/<origin>/`. The gallery is size-capped (`[image].max_disk_gb = 10` by default, oldest-first GC).

There's also a public sibling page at `/images` for non-admin API-key holders. Same three-pane layout; instead of the admin session cookie it accepts a bearer key (pasted on a login screen, stored in localStorage) and scopes the gallery to that origin's own directory.

### Sharing the GPU with the text engine

A single GPU usually can't hold a large LLM and a diffusion model at the same time. When an image request lands while a text engine is running, llamanager by default:

1. Snapshots the current text spec (model + profile + args)
2. Stops the text server to free VRAM
3. Runs the image task to completion
4. Restarts the text server from the snapshot

This is the single-slot invariant. The dashboard "Now serving" hero shows one engine at a time, the same way LLM swaps work. The queue enforces it: when `allow_concurrent` is off, the text and image families are mutually exclusive at the dispatcher level, so no image starts while a text request is in flight and vice versa.

Two toggles on the Diffusion engines page change the behaviour:

- **Restart text engine after image completes** (default on). Turn off to stay in image-only mode after a generation.
- **Allow concurrent text + image** (default off). Lets both families run in parallel up to their per-family caps (`max_concurrent` for text, 1 for image). Risks VRAM OOM on cards with less than ~48 GB; enable only when you know both fit.

```toml
[coexistence]
unload_text_on_arrival = true
restart_text_after_image = true
allow_concurrent = false
```

When the [Multi-slot LLM (beta)](#multi-slot-llm-beta) feature is on,
the legacy `allow_concurrent` toggle no longer applies; the multi-slot
equivalent is `[coexistence].allow_diffusion_with_slots` (also
toggleable from the Slots page) which leaves every LLM slot loaded
when an image task arrives.

Cancellation: cancelling a queued image request removes it before it starts. Cancelling an in-flight one terminates the subprocess (SIGTERM, escalating to SIGKILL after 5 s) so the GPU is freed promptly.

## Calling the API

Any OpenAI-compatible client works. With `curl`:

```bash
curl -N http://localhost:7200/v1/chat/completions \
  -H "Authorization: Bearer $ORIGIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "stream": true,
    "messages": [{"role": "user", "content": "Hi!"}]
  }'
```

To pick a model, name it the standard OpenAI way — the **`model`** field in the request body, set to a model ID (the path relative to the models directory). When the body names a model llamanager knows (a configured model or one currently loaded), the queue routes to it and hot-swaps if needed; its default profile is used. So stock clients like the Continue VS Code extension or the `openai` SDK work with no extra configuration.

A request that names a model llamanager can't honour — an uninstalled id, a generic `"gpt-4"` / `"any"` placeholder, or one the origin isn't permitted to use — **falls back to the default model** rather than erroring. When that happens the response carries an `x-llamanager-model-fallback` header explaining why, and the reason is logged. Model choice never returns a 403 on the text endpoints; it degrades to the default. (The image endpoints still 403, since they require an explicit model and have no default to fall back to.)

The **`X-Llamanager-Model`** header takes precedence over the body field, and **`X-Llamanager-Profile`** selects a non-default profile. Use the headers when you need to override the body, pin a specific profile, or route to an on-disk model that has no config entry:

```bash
curl -N http://localhost:7200/v1/chat/completions \
  -H "Authorization: Bearer $ORIGIN_KEY" \
  -H "X-Llamanager-Model: bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
  -H "X-Llamanager-Profile: balanced" \
  -H "Content-Type: application/json" \
  -d '{"model": "any", "messages": [{"role": "user", "content": "Hi!"}]}'
```

Each origin's `allowed_models` setting restricts which models it can request. New origins **allow all models (`*`) by default**; edit an origin on the Origins page (or `llamanager origins update`) to restrict it to a picked set of installed models. The legacy `default` token (default model only) is still honoured. A request for a model outside the list doesn't error — it falls back to the default model, as above.

While a request is queued or a model swap is happening, llamanager emits SSE comment lines (`: status=swapping_model`, `: keepalive`) every 10 seconds so client connections don't time out.

Two operator switches can refuse a request outright (distinct from the soft model-fallback above):

- **Disabled origin.** If an admin has flipped the origin's Status to *disabled* on the Origins page (`/ui/origins`), every submission from that key returns **403** (`origin '<name>' is disabled …`). The key still authenticates — this is a deliberate park-the-client switch, not a revocation.
- **Model-loading lock.** When the lock is on (Settings → *Lock model loading*), a request the loaded model can serve runs normally, but one that would start the engine, swap models, or change profile resolves to a `failed` status with a `model loading is locked: …` error telling you to load the model manually or turn the lock off. It takes effect immediately and persists across restarts (`[queue].lock_model_loading`).

### Anthropic-compatible API

llamanager also speaks the Anthropic Messages API, mounted under `/anthropic/v1/` so it sits beside the OpenAI surface without colliding on the `/v1/models` envelope. The official `anthropic` SDK works out of the box — just point `base_url` at llamanager:

```python
from anthropic import Anthropic

client = Anthropic(api_key="lm_...", base_url="http://localhost:7200/anthropic")
msg = client.messages.create(
    model="any",                     # any string; the loaded model is used
    max_tokens=512,
    messages=[{"role": "user", "content": "Hi!"}],
)
print(msg.content[0].text)
```

With `curl`:

```bash
curl http://localhost:7200/anthropic/v1/messages \
  -H "x-api-key: $ORIGIN_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "any",
    "max_tokens": 512,
    "messages": [{"role": "user", "content": "Hi!"}]
  }'
```

Endpoints:

| method | path                                | purpose                                                     |
|--------|-------------------------------------|-------------------------------------------------------------|
| POST   | `/anthropic/v1/messages`            | Messages API (non-streaming and SSE streaming).             |
| POST   | `/anthropic/v1/messages/count_tokens` | Count input tokens via the loaded engine's tokenizer.       |
| GET    | `/anthropic/v1/models`              | List visible models in Anthropic's envelope shape.          |

Auth accepts either the Anthropic SDK's `x-api-key` header or the existing `Authorization: Bearer` header — your llamanager origin key works in both slots. The `anthropic-version` header is accepted but the value is ignored (llamanager has no version-gated behavior).

Supported features:

- **Streaming.** Full `message_start` / `content_block_start` / `content_block_delta` / `content_block_stop` / `message_delta` / `message_stop` event sequence. Tool-call arguments stream as `input_json_delta` partials.
- **Tool use.** Anthropic `tools` definitions, `tool_choice` (`auto` / `any` / `none` / `tool`), assistant `tool_use` blocks, and user `tool_result` blocks all translate to and from OpenAI tool calls. Server-side tools (`web_search_20250305`, `computer_*`, etc.) are rejected with a 400 — bring your own.
- **Vision.** `image` content blocks with `source.type` of `base64` or `url` are forwarded to the multimodal model.
- **System prompts** as either a string or a list of `{"type":"text"}` blocks.
- **Sampling controls** — `temperature`, `top_p`, `top_k`, `stop_sequences`, and the required `max_tokens`.
- **Model selection** — same as the OpenAI side: the body's `model` field routes when it names a known, permitted model (it's also echoed back), the `X-Llamanager-Model` / `X-Llamanager-Profile` headers override it, and an unknown or not-permitted model falls back to the default rather than erroring. The `X-Llamanager-Thinking` header also works here.

Not implemented: `document` content blocks (PDFs), Anthropic server-side tools, prompt caching markers, and the Batches API.

### Reasoning / thinking control

Thinking-capable models (Qwen3, GLM, MiniCPM, …) emit a `<think>…</think>` preamble that wastes leading tokens when you don't want it. llamanager lets you turn it off two ways:

- **As a profile default** — set `thinking = "off"` on a profile (UI: LLM models → edit profile → *Reasoning / thinking* dropdown; or in `config.toml`). Every `/v1/chat/completions` call routed through that profile gets `chat_template_kwargs.enable_thinking = false` merged into the body. You stop having to send the field on every request.
- **As a per-request override** — send the `X-Llamanager-Thinking: on|off` header. It wins over the profile default *and* over any `chat_template_kwargs.enable_thinking` the caller put in the body. Use it as the escape hatch when one client needs the opposite of the profile default.

```bash
# force thinking off for this one request, regardless of profile
curl -N http://localhost:7200/v1/chat/completions \
  -H "Authorization: Bearer $ORIGIN_KEY" \
  -H "X-Llamanager-Thinking: off" \
  -H "Content-Type: application/json" \
  -d '{"model": "any", "messages": [{"role": "user", "content": "Hi!"}]}'
```

Precedence (highest wins): `X-Llamanager-Thinking` header → caller's own `chat_template_kwargs.enable_thinking` in the body → profile's `thinking` setting → upstream/template default. The merge only fires on `/v1/chat/completions` (the bare `/v1/completions` endpoint doesn't render the chat template that consumes the kwarg). The kwarg is silently ignored by templates that don't reference it, so setting it on a non-reasoning model is harmless.

**Reasoning budget — bound thinking without disabling it.** When you *want* thinking but not a runaway loop, set a **Reasoning budget** on the profile (UI: LLM models → edit profile → *Basic* tab; or `reasoning_budget = N` in `config.toml`). It maps to llama-server's `--reasoning-budget`: once the model has spent N thinking tokens, llama.cpp forces the end-of-thinking tag so it stops reasoning and produces its answer. This prevents the failure where a model thinks until it exhausts `max_tokens` and returns *empty content* (a reasoning model can otherwise "finish" with nothing for the client to use). Blank = unbounded; `0` = no thinking. The profile editor suggests a value from the model's **measured decode throughput** — roughly 20s of thinking at the speed that model actually runs on your hardware — so the number isn't a guess. llama-engine models only.

## Chat in the browser

A built-in chat UI lives at <http://localhost:7200/chat>. Any user with a valid origin API key can use it; no admin access required.

The UI has streaming token display, multiple conversations stored in localStorage, configurable system prompt, per-conversation profile and model selection, a markdown toggle, and dark/light theme. The admin panel has its own chat page at `/ui/chat` that reuses the admin session.

**Image input (vision).** When the currently-selected profile has an `mmproj` configured (i.e. it's vision-capable), a paperclip button appears next to the textarea. Click it (or drop files) to attach PNG/JPEG/WEBP/GIF images up to 10 MB each; thumbnails show above the input. On send, the message goes out as an OpenAI multimodal `content` array (`[{type:"text",text:"..."}, {type:"image_url",image_url:{url:"data:image/png;base64,..."}}]`) that llama-server with `--mmproj` consumes directly. Switching to a profile without `mmproj` hides the paperclip and clears any pending attachments. The public `/chat` page does the same — the bearer's `allowed_models` decides which profiles even appear in the dropdown.

## Auto-start at boot or login

One command, four modes, all three platforms:

```bash
llamanager autostart --mode tray+service     # recommended
```

| mode | what it does | runs before login? | tray icon? |
|------|--------------|:------:|:------:|
| `tray+service` | always-on service **+** a tray/menu-bar icon to control it | yes¹ | yes |
| `boot-service` | always-on service, headless | yes¹ | no |
| `login-tray`   | service + tray, only while you're logged in | no | yes |
| `off`          | tear everything down | — | — |

¹ Linux uses `loginctl enable-linger`; Windows uses a real service. On **macOS**, before-login start needs a root LaunchDaemon — add `--pre-login` (you'll be prompted for `sudo`, and the tray's service controls then need `sudo` too). Without it, the macOS service starts at login.

The **tray icon** is the daily control surface: open the web UI, start/stop the service, launch a model with its default profile, and set **Autorun at startup** (Off / At login / Before login) — all from the right-click menu. It needs the optional extra:

```bash
pip install -e '.[tray]'      # pystray + Pillow
```

On **Linux** the icon also needs system packages for the AppIndicator backend (especially on Wayland/GNOME), and the venv must be able to see them:

```bash
sudo apt install python3-gi gir1.2-ayatanaappindicator3-0.1
# then build the venv with --system-site-packages, or set
# include-system-site-packages = true in .venv/pyvenv.cfg
```

Run the tray manually with `llamanager tray` (it blocks the terminal — it's a GUI loop) or `llamanager tray -b` to detach. `init` and the autostart entry launch it detached for you.

Tear down with `llamanager autostart --mode off` (or the `remove-tray` alias).

> On Windows, `tray+service`/`boot-service` register a real service (run the command from an elevated shell once; it also grants your account prompt-free start/stop afterward) and need the `pywin32` extra: `pip install -e '.[windows-service]'`.

<details>
<summary>Advanced — individual per-platform installers</summary>

These are the building blocks `autostart` orchestrates; use them if you want one piece only or a custom layout.

```bash
# macOS LaunchAgent (login)            Linux user systemd unit
llamanager install-launchd             llamanager install-systemd
launchctl load -w \                    systemctl --user daemon-reload
  ~/Library/LaunchAgents/com.llamanager.plist   systemctl --user enable --now llamanager.service
```

```powershell
# Windows real service (always-on; needs pywin32 + elevation)
llamanager install-windows-service [--username "DOMAIN\svc" --password "..."]
sc query llamanager
llamanager remove-windows-service

# Windows Task Scheduler (logon-triggered, no extra deps)
llamanager install-windows
schtasks /Create /XML "$env:USERPROFILE\.llamanager\llamanager.task.xml" /TN llamanager
```

`install-tray` / `remove-tray` remain as aliases for `autostart --mode tray+service` / `--mode off`. For a system-wide Linux service, copy the generated unit to `/etc/systemd/system/` and use `sudo systemctl ...`.
</details>

### Quick restart
```bash
systemctl --user restart llamanager.service
```
Bounces the manager/proxy. Use it to pick up new code after updating a
source / editable (`pip install -e .`) checkout, or after editing the unit —
no reinstall needed. Any model already loaded keeps its old launch args until
you reload it.

## Uninstall

One command stops llamanager and removes every autostart entry (the service unit, launch agents, scheduled tasks, tray):

```bash
llamanager uninstall
```

It keeps your data by default. To also delete config, `state.db`, logs, and keys — but keep models:

```bash
llamanager uninstall --purge
```

To delete everything including downloaded models:

```bash
llamanager uninstall --purge-models
```

It prompts before doing anything (skip with `--yes`). On Windows the service removal step needs an elevated shell; on macOS a `--pre-login` system service is removed with `sudo`. Finally, remove the package itself:

```bash
pip uninstall llamanager
```

## CLI

Service and installer commands:

```text
llamanager init [--no-launch] [--binary PATH]   # guided first-run setup (start here)
llamanager serve [--host ...] [--port ...] [--log-level info]
llamanager tray [-b|--background]            # run the tray/menu-bar app (needs [tray])
llamanager autostart --mode off|boot-service|login-tray|tray+service
                                             # configure how it runs at boot/login
llamanager uninstall [--purge] [--purge-models] [--yes]
                                             # stop + remove autostart, optionally data
llamanager init-config [--path PATH]
llamanager status                            # prints last persisted runtime.json
llamanager --config /path/to/config.toml <subcommand>

# advanced / per-platform building blocks (autostart orchestrates these):
llamanager install-launchd [--label com.llamanager] [--port 7200] [--binary PATH]
llamanager install-systemd [--unit llamanager.service] [--port 7200] [--binary PATH]
llamanager install-windows [--task llamanager] [--port 7200] [--binary PATH]
llamanager install-windows-service [--startup auto|manual|delayed|disabled]
                                   [--username USER] [--password PASS] [--no-start]
llamanager remove-windows-service
llamanager install-tray / remove-tray        # aliases for autostart tray+service / off
```

### Admin verbs (drive a running service)

These talk to a running `llamanager serve` over `/admin/*`, so an agent or shell script can manage models, queue, and origins without a custom client. They print JSON to stdout (pipe through `jq` to slice the result).

Auth resolution order:

1. `--admin-key` flag
2. `$LLAMANAGER_ADMIN_KEY`
3. `admin_key` under a `[cli]` section in `config.toml`

Base URL resolution order:

1. `--url` flag
2. `$LLAMANAGER_URL`
3. derived from config (`http://<bind>:<port>`, with `0.0.0.0` rewritten to `127.0.0.1` since the CLI usually runs on the same host)

```text
llamanager server status                    # full service snapshot
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

llamanager models set-default <model_id>                  # configured default LLM
llamanager models add-existing <path/to/file.gguf>        # symlink or copy into models_dir
llamanager models set-dir <path>                          # change models_dir + persist

llamanager queue cancel-all                               # cancel every queued + in-flight request
llamanager origins update <id> [--priority N] [--allowed M ...] [--admin on|off]

llamanager profiles list <model_id>                       # LLM profiles for a model
llamanager profiles create <model_id> <name>              # see flags below
        [--mmproj PATH] [--ctx-size N] [--vram-limit-gb X | --vram-unlimited]
        [--ram-spill-policy default|ram_only|limited] [--ram-spill-limit-gb X]
        [--thinking on|off] [--arg KEY=VALUE ...] [--make-default]
llamanager profiles update <model_id> <name> [...same flags...] [--rename NEW]
llamanager profiles delete <model_id> <name>
llamanager profiles clone <model_id> <name> <new_name>
llamanager profiles set-default <model_id> [--profile NAME]

llamanager setup llama-binary <path>                      # llama-server binary path
llamanager setup hidream [--python PATH] [--repo PATH]
llamanager setup z-image <python_path>
llamanager setup flux2 [--sd-cli PATH] [--device-index N | --clear-device-index]
llamanager setup coexistence [--unload-text-on-arrival on|off]
                              [--restart-text-after-image on|off]
                              [--allow-concurrent on|off]
llamanager setup default-args <llama|mlx> --arg KEY=VALUE ...
llamanager setup autolaunch on|off                        # default LLM on service startup
llamanager setup autorestart on|off                       # supervisor crash auto-restart
llamanager setup install-llama-server [--source ...] [--backend ...] [--version TAG]
llamanager setup check-updates [--variant <id>]           # installed vs latest (one or all installed variants)
llamanager setup engine-versions <variant_id>             # list installable builds (newest first)
llamanager setup install-llama-server-status <variant_id>
llamanager setup switch-variant <variant_id>
llamanager setup auto-update list                         # per-engine switches + cadence
llamanager setup auto-update <engine-key> on|off          # variant id / diffusion engine / "llamanager"
llamanager setup auto-update settings [--idle-seconds N] [--check-interval-seconds N]

llamanager diffusion engines                              # per-engine install state + GPU detection
llamanager diffusion install <engine> [--patch-flash-attn] [--diffusers-version X | --reset-diffusers]
llamanager diffusion versions <engine>                    # installable diffusers versions + installed/target
llamanager diffusion cancel-install <engine>
llamanager diffusion models                               # installed + catalog of installable
llamanager diffusion activate <model_id>                  # set as dashboard/API default
llamanager diffusion profiles list <model_id>
llamanager diffusion profiles create <model_id> <name> [--field K=V ...] [--make-default]
llamanager diffusion profiles update <model_id> <name> [--field K=V ...] [--rename NEW]
llamanager diffusion profiles delete <model_id> <name>
llamanager diffusion profiles clone <model_id> <name> <new_name>
llamanager diffusion profiles set-default <model_id> [--profile NAME]
llamanager diffusion profiles materialize-defaults <model_id> <engine>

llamanager update [--check]                               # pip install --upgrade + restart (same as the /ui/about button)

llamanager slots status                                   # enabled flag + per-slot dashboard
llamanager slots list                                     # alias for `slots status`
llamanager slots enable                                   # turn on (force-disables exclusive mode)
llamanager slots disable                                  # turn off — drains slots 1..N
llamanager slots add                                      # allocate the next free slot id + port
llamanager slots remove <slot_id>                         # stop + remove (slot 0 cannot be removed)
llamanager slots load <slot_id> --model M [--profile P]   # load (or swap) a model into a slot
        [--arg KEY=VALUE ...] [--force]
llamanager slots unload <slot_id>                         # stop the model in a slot
llamanager slots coex on|off                              # diffusion-coexistence (on = image keeps LLMs loaded)
```

The CLI mirrors the web UI feature-for-feature: every page that lets you click something has a `llamanager` verb that does the same thing against `/admin/*`. `llamanager profiles` covers the LLM-profile editor on `/ui/models`, `llamanager diffusion` covers the Diffusion engines + Diffusion models pages, `llamanager setup` covers paths/coexistence/autolaunch and the llama-server installer, `llamanager slots` (beta, see [Multi-slot LLM (beta)](#multi-slot-llm-beta)) covers the parallel-model dashboard, `llamanager models {set-default,add-existing,set-dir}` covers the LLM-model housekeeping rows, and `llamanager queue cancel-all` + `llamanager origins update` round out the existing groups. `update` runs `pip install --upgrade llamanager` against the service's venv and SIGTERMs so the supervisor restarts it — exactly what the `/ui/about` Update button does. `--check` reports the latest GitHub tag plus the detected install mode without doing anything. If the service was installed in editable / developer mode (`pip install -e .` from a git checkout), the auto-update refuses with instructions to run `git pull && pip install -e .` in the checkout yourself — the operator's checkout is the source of truth in that case.

Example, an agent that wants to swap models before a long batch and revert after:

```bash
export LLAMANAGER_ADMIN_KEY=lm_...
llamanager server swap --profile my-vision-profile
llamanager models list | jq '.[].model_id'
llamanager queue list
llamanager server swap --profile my-default-profile
```

## Configuration

Lives at `~/.llamanager/config.toml` (Windows: `%USERPROFILE%\.llamanager\config.toml`). Hot-reload with `POST /admin/reload` or `SIGHUP` (POSIX only). Full schema is in spec §7.

A minimal example:

```toml
[server]
bind = "0.0.0.0"
port = 7200
llama_server_binary = "llama-server"   # or 'C:\path\to\llama-server.exe' as a TOML literal
llama_server_port = 7201
data_dir = "~/.llamanager"

# Multi-slot LLM (beta, see "Multi-slot LLM (beta)" section). Off by
# default; live toggle via /admin/slots/enable or `llamanager slots enable`.
# multi_slot_enabled   = false
# multi_slot_base_port = 7201   # slot 0 uses this; 1..N take the next free ports above
# multi_slot_max       = 4

[defaults]
model = ""             # set by the top bar's LLM star toggle
image_model = ""       # set by the top bar's diffusion star toggle
image_profile = ""

[downloads]
# 0 = no cap. The actual partition free-space check still applies.
# Set to a non-zero GB value to cap the cumulative size of the models dir.
max_disk_gb = 0

[queue]
# Operator lock. When true, a request that would start the engine, swap
# models, or change profile is rejected (status=failed) instead of loading;
# requests the already-loaded model can serve still run. Toggle live from
# Settings → Lock model loading. Off by default.
lock_model_loading = false

[image]
# Per-engine paths. Each is filled either by clicking Install dependencies
# on the engine card (auto), or by pasting an existing install path.
# hidream_python     = "/path/to/.venv-hidream/bin/python"
# hidream_repo       = "/path/to/HiDream-O1-Image"
# flux2_sd_cli       = "/path/to/sd-cli"
# flux2_device_index = 1                  # GGML_VK_VISIBLE_DEVICES
# z_image_python     = "/path/to/.venv-z-image/bin/python"
max_disk_gb = 10                          # cap for the on-disk image gallery

[coexistence]
unload_text_on_arrival = true
restart_text_after_image = true
allow_concurrent = false
# Multi-slot only (ignored when multi_slot_enabled = false): when true,
# image tasks do NOT unload LLM slots. VRAM headroom is on you.
# allow_diffusion_with_slots = false

[auto_update]
# Auto-update engines when the service is idle. Off for every engine by
# default; opt in per engine from the UI switch or `llamanager setup
# auto-update <engine> on`. See "Updating engines" below.
idle_seconds = 300              # quiet window required before an update fires
check_interval_seconds = 21600  # how often each enabled engine checks upstream (6h)

[auto_update.engines]
# Keys: a llama variant id, a diffusion engine name, or "llamanager".
# "llama.cpp-cuda" = true
# "z_image"        = false
# "llamanager"     = false
```

> **Windows paths in TOML.** Backslashes inside double-quoted strings are escape sequences; `"C:\Users\..."` will fail to parse (the `\U` looks like a Unicode escape). Use TOML literal strings (single quotes), e.g. `'C:\Soulthread\Models'`, or escape every backslash: `"C:\\Soulthread\\Models"`.

Per-model profiles are auto-created when you pull a model. You can edit them on the LLM models or Diffusion engines page, or write them by hand:

```toml
[models."your-model.gguf"]
default_profile = "balanced"

[models."your-model.gguf".profiles.balanced]
ctx_size = 4096
thinking = "off"        # optional; "on" / "off" / omit. See "Reasoning / thinking control" above.
reasoning_budget = 2000 # optional; cap thinking tokens (→ --reasoning-budget). Blank = unbounded, 0 = off.
parallel = 1            # optional; concurrent request slots (→ --parallel). Blank = auto. Lower to free VRAM.
args = { temp = 0.7 }
```

The VRAM / RAM-spill knobs are basic fields that translate into a
computed `--n-gpu-layers` value at launch time, based on the model's
GGUF header (layer count + file size) plus a KV-cache budget derived
from `ctx_size`. They're llama.cpp-only — MLX uses unified memory and
ignores them.

## API

| path        | purpose                                 | auth               |
|-------------|-----------------------------------------|--------------------|
| `/v1/*`     | OpenAI-compatible inference (queued)    | bearer (any key)   |
| `/admin/*`  | control plane (lifecycle, models, etc.) | bearer (admin key) |
| `/ui/*`     | web UI                                  | cookie session     |
| `/health`   | liveness check                          | none               |

Full endpoint list is in spec §5.

## Filesystem layout

```text
~/.llamanager/
├── config.toml             static config (you edit this)
├── state.db                sqlite: origins, requests, downloads, events, engine_installs
├── runtime.json            current run state (atomic writes; describes slot 0)
├── slots.json              multi-slot manifest (created when slots is enabled)
├── .session-secret         signed-cookie secret (mode 0600)
├── bin/                    installed llama.cpp / fork binaries
├── venvs/                  per-engine Python venvs created by the installer
│   ├── z_image/
│   └── hidream/
├── models/                 downloaded weights, mixed text + image
├── images/                 generated PNGs, organised YYYY-MM-DD/<origin>/
├── refs/                   transient reference-image staging (cleaned after each run)
├── logs/
│   ├── llamanager.log      llamanager's own log (rotated 5 × 50 MB)
│   ├── llama-server.log    captured stdout/stderr of the text engine
│   ├── hidream.log         captured output of the HiDream subprocess
│   ├── flux2.log           captured output of FLUX 2 / sd.cpp
│   └── z_image.log         captured output of the Z-Image subprocess
└── llamanager.task.xml     windows: generated by `install-windows` (if used)
```

On Windows, `~` resolves to `%USERPROFILE%`, e.g. `C:\Users\<you>\.llamanager`.

## Troubleshooting

**`llama-server: command not found` on launch.** The binary is not on `PATH`. Either add it, or set `[server].llama_server_binary` in `config.toml` to the absolute path.

**Port 7200 or 7201 already in use.** Change `[server].port` (llamanager) or `[server].llama_server_port` (the upstream `llama-server`) in `config.toml`. Make sure the same port shows up under any matching `args = { ... port = ... }` profile entry.

**TOML parse error: "Invalid hex value (at line N)".** A Windows path with single backslashes in a double-quoted string. Switch the value to a single-quoted literal string, e.g. `models_dir = 'C:\Soulthread\Models'`.

**"would exceed max_disk_gb" when pulling a model.** The historical default cap was 80 GB and is easy to trip with diffusion checkpoints. New installs default to `0` (no cap, only the partition free-space check applies). To raise an existing config: set `[downloads].max_disk_gb = 0` and reload.

**Lost the bootstrap key.** No recovery, argon2id is one-way. Stop the service, delete `~/.llamanager/state.db`, restart, and a new bootstrap key prints. Existing origins and keys go with it, so this is only safe if you have not created real origins yet.

**Web UI says "invalid admin key" but the key is admin.** The verification cache is per-process. If you just rotated the key, restart the service or wait 5 minutes for the cache TTL.

**Crash loop, `state: crashed` and 503s.** The 3-in-5 restart policy gave up. Check `~/.llamanager/logs/llama-server.log` (or `hidream.log` / `flux2.log` / `z_image.log` for the image engines), fix the cause, then `POST /admin/server/start` (or use the UI).

**A request shows "running" or "queued" forever after a restart.** If the service crashed or was restarted mid-flight, its in-memory queue is gone but the database rows survive. llamanager now reconciles these at startup — any row left `queued`/`swapping_model`/`running` by a previous process is resolved to `failed` (`interrupted by daemon restart`) before the queue starts, so the dashboard is truthful. A row that still lingers in-session can be cleared with `llamanager queue cancel <request_id>`; cancel now force-clears a stale row even when no live handler backs it.

**Requests fail with `model loading is locked`.** The model-loading lock is on (Settings → *Lock model loading*). Either load the model the request wants manually (top-bar **Load**, or `llamanager server swap`), or turn the lock off. See [Calling the API](#calling-the-api).

**Z-Image `Install dependencies` finishes but `pipe(...)` fails at runtime.** The Z-Image auto-install picks a generic torch wheel. For AMD ROCm, Apple Silicon, or specific CUDA versions, build the venv yourself with the vendor wheels and point the engine at it, then skip the install button. (HiDream's auto-installer does this for you on AMD — that path is generic only for Z-Image and FLUX 2 today.)

**HiDream errors with `AssertionError: CUDA is required for inference.` on AMD.** PyTorch keeps the `torch.cuda.*` namespace even when built against ROCm/HIP; the assertion really means "no HIP device visible." Usually this is the `render` group — the llamanager process needs gid `render` to open `/dev/kfd`. Add the user (`sudo usermod -aG render <user>`) and log out / in. The Diffusion engines page surfaces this as a warning on the HiDream card when it's wrong.

**HiDream errors with `AssertionError: Flash attention is not available.` on AMD.** The upstream hidream-source pipeline hardcodes flash-attn, which isn't shipped for AMD. The HiDream install card has a checkbox that patches `<hidream_repo>/models/pipeline.py` (`"use_flash_attn": True` → `False`); it's pre-checked on AMD hosts. The non-flash path uses a 4D attention mask — somewhat slower at large image sizes, but functional.

**"No compatible GPU detected" on Windows with an AMD card.** Older builds required `rocm-smi`. Update llamanager: the new path reads the adapter name and VRAM total from the driver registry and live VRAM usage from the Windows GPU performance counters, without any vendor CLI.

**Windows: argon2-cffi fails to install.** Python 3.11+ ships prebuilt wheels for argon2-cffi on Windows. If you hit a build error, upgrade pip (`python -m pip install -U pip`) and retry. As a last resort, install Microsoft's [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

**Update fails with `fatal: detected dubious ownership` or `would be overwritten by merge`.** These are git's safeguards firing against the *old* git-pull-based updater that shipped before 0.2.7. The current updater calls `pip install --upgrade llamanager` and doesn't touch your checkout at all — upgrade past 0.2.7 once (using whichever path works for your install: `pip install --upgrade llamanager` against the service's venv, or `git pull && pip install -e .` if you cloned the repo) and these failure modes go away.

**`llamanager update` refuses with "editable install detected".** You ran `pip install -e .` from a clone, so the service is running directly out of your checkout. The auto-updater treats the checkout as the source of truth and won't touch it — update by running `git pull && pip install -e .` in that directory, then restart the service (your supervisor will pick the new code up). The About page surfaces the same instructions instead of an Update button.

**`pip install --upgrade llamanager` fails with "externally-managed-environment".** PEP 668 distros (newer Debian/Ubuntu, Homebrew) refuse to mutate the system Python. The service should be running in a venv anyway — re-install into one (`python -m venv ~/.venvs/llamanager && ~/.venvs/llamanager/bin/pip install llamanager`), update your `llamanager.service` / launch agent to point at that venv's `llamanager` binary, and the updater will work against that venv from then on.

## Releasing a new version

The version is defined in one place: the `VERSION` file in the project root. `pyproject.toml`, `llamanager.__version__`, and the web UI all read from it.

### 1. Bump the version

Edit `VERSION`, then reinstall so package metadata is updated:

```bash
# source .venv/bin/activate                  # macos / linux
# .\.venv\Scripts\Activate.ps1             # windows powershell
# .\.venv\Scripts\activate.bat             # windows cmd
pip install -e .
python -c "import llamanager; print(llamanager.__version__)"
```

### 2. Tag

```bash
git tag "v$(cat VERSION)"
git push origin main
git push origin "v$(cat VERSION)"
```

### 3. Create a GitHub release

Tags are pointers. A GitHub Release adds notes and makes the version visible on the repo's Releases page; the llamanager update checker looks for releases first and falls back to tags.

**Option A — GitHub web UI** (recommended)

1. Go to the repo → Releases → Draft a new release
2. Choose a tag → select the `v...` tag you just pushed
3. Set the title and write release notes
4. Publish release

**Option B — GitHub CLI** (`gh`)

```bash
gh release create "v$(cat VERSION)" --title "v$(cat VERSION)" --notes "Release notes here."
```

> `gh` requires the [GitHub CLI](https://cli.github.com/) to be installed and authenticated (`gh auth login`). If `gh release create` fails with a permission error, use the web UI instead.

### How the update checker works

When a user clicks Check for updates on the About page, llamanager:

1. Queries `https://api.github.com/repos/{repo}/releases/latest`
2. If no releases exist (404), falls back to `https://api.github.com/repos/{repo}/tags`
3. Compares the remote version against the local one using semver (major.minor.patch)
4. Shows the update banner only if the remote version is strictly greater, never suggesting a downgrade

## Credits

llamanager builds on excellent open-source projects:

| component | project |
|-----------|---------|
| LLM inference | [llama.cpp / llama-server](https://github.com/ggerganov/llama.cpp), [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple Silicon) |
| Alternative LLM engine | [Atomic TurboQuant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) |
| HiDream | [HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) |
| Z-Image | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image), [SeeSee21/Z-Anime](https://huggingface.co/SeeSee21/Z-Anime) |
| FLUX 2 | [Black Forest Labs / FLUX 2](https://huggingface.co/black-forest-labs/FLUX.2-dev) via [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) |
| ML runtime | [PyTorch](https://pytorch.org), [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [Accelerate](https://github.com/huggingface/accelerate) |
| Web framework | [FastAPI](https://fastapi.tiangolo.com) on [Uvicorn](https://www.uvicorn.org) |
| Templates | [Jinja2](https://jinja.palletsprojects.com) |
| Interactivity | [HTMX](https://htmx.org) + [idiomorph](https://github.com/bigskysoftware/idiomorph) |
| Model hub | [Hugging Face Hub](https://huggingface.co) (`huggingface_hub`) |
| Storage | [SQLite](https://www.sqlite.org) (via [aiosqlite](https://github.com/omnilib/aiosqlite)), [tomlkit](https://github.com/python-poetry/tomlkit) |
| Auth + crypto | [argon2-cffi](https://github.com/hynek/argon2-cffi), [itsdangerous](https://itsdangerous.palletsprojects.com) |
| Typography | [Fraunces](https://fonts.google.com/specimen/Fraunces), [Inter](https://fonts.google.com/specimen/Inter), [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono) |

## License

Copyright 2025–2026 Mounir Ould Setti / [SoulThread Technologies](https://soulthread.group)

Licensed under the [Apache License, Version 2.0](LICENSE). See `LICENSE` for the full text.
