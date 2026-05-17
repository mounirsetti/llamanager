"""Live UI smoke test driven by Playwright.

Boots a real llamanager FastAPI app (with auth + UI mounted) on a free
port, with a tmpdir-scoped data dir, seeds the bootstrap admin key, and
exercises the profile create/edit/delete flow on /ui/models. The
``llama-server`` binary is set to ``true`` so we never actually try to
launch the inference engine — we're testing the configuration UI.

Run with:
    python tests/playwright_live.py
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, deadline_s: float = 15.0) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def run(headed: bool = False) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="llamanager-pw-"))
    port = _pick_port()
    runner = Path(__file__).parent / "_pw_server.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, str(runner), str(tmp), str(port)],
        env=env,
        cwd=str(Path(__file__).parent.parent),
    )
    try:
        if not _wait_for_server(port):
            print("server did not start in time", file=sys.stderr)
            return 2
        key_path = tmp / "bootstrap.key"
        # Wait until the key file is written (race with uvicorn boot).
        deadline = time.time() + 5.0
        while not key_path.exists() and time.time() < deadline:
            time.sleep(0.1)
        bootstrap_key = key_path.read_text(encoding="utf-8").strip()

        base = f"http://127.0.0.1:{port}"
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not headed)
            ctx = browser.new_context()
            page = ctx.new_page()
            page.set_default_timeout(8000)

            # ---- login ----
            page.goto(f"{base}/ui/login")
            page.fill('input[name="api_key"]', bootstrap_key)
            page.click('button[type="submit"]')
            # The login POST 303's to /ui/ which then routes onward.
            page.wait_for_url(lambda url: "/ui/login" not in url, timeout=8000)

            # ---- models page ----
            page.goto(f"{base}/ui/models")
            page.wait_for_load_state("networkidle")

            # Both seeded models should show.
            expect(page.locator("text=demo-org/demo-model.gguf")).to_be_visible()
            expect(page.locator("text=mlx-demo")).to_be_visible()

            # Engine badge: GGUF row says "engine: llama".
            gguf_card = page.locator(
                'article[data-model-id="demo-org/demo-model.gguf"]'
            )
            expect(gguf_card.locator("text=engine:")).to_be_visible()
            expect(gguf_card.locator("code:has-text('llama')").first).to_be_visible()

            mlx_card = page.locator('article[data-model-id="mlx-demo"]')
            expect(mlx_card.locator("code:has-text('mlx')").first).to_be_visible()

            # ---- create a basic profile on the GGUF model ----
            add_summary = gguf_card.locator("summary.lm-profile-add__summary")
            add_summary.click()
            add_form = gguf_card.locator("form[action='/ui/models/profiles/create']")
            add_form.locator('input[name="name"]').fill("fast")
            # Toggle into Basic tab (it's the default but double-check).
            add_form.locator('button.lm-tab[data-tab="basic"]').click()
            # Set context size + a VRAM cap (uncheck "no cap" first).
            no_cap = add_form.locator('input[name="vram_unlimited"]')
            if no_cap.is_checked():
                no_cap.uncheck()
            add_form.locator('input[name="vram_limit_gb"]').fill("8")
            add_form.locator('select[name="ram_spill_policy"]').select_option(
                "limited"
            )
            # The limit input becomes visible after the policy change.
            limit_in = add_form.locator('input[name="ram_spill_limit_gb"]')
            limit_in.wait_for(state="visible")
            limit_in.fill("4")
            add_form.locator('input[name="make_default"]').check()
            add_form.locator('button[type="submit"]').click()
            page.wait_for_load_state("networkidle")

            # New profile should show up under the GGUF model.
            gguf_card = page.locator(
                'article[data-model-id="demo-org/demo-model.gguf"]'
            )
            expect(gguf_card.locator(".lm-profile__name", has_text="fast")).to_be_visible()
            # And it should be marked as default.
            expect(gguf_card.locator(".lm-profile.is-default")).to_be_visible()

            # ---- toggle to advanced view on the new profile ----
            prof_li = gguf_card.locator(".lm-profile.is-default").first
            prof_li.locator("summary.lm-profile__summary").click()
            edit_form = prof_li.locator(
                "form[action='/ui/models/profiles/fast/update']"
            )
            edit_form.locator('button.lm-tab[data-tab="advanced"]').click()
            # The args JSON textarea should now be visible.
            args_ta = edit_form.locator('textarea[name="args_json"]')
            expect(args_ta).to_be_visible()
            # The basic panel should be hidden.
            basic_panel = edit_form.locator('.lm-tab-panel[data-panel="basic"]')
            expect(basic_panel).to_be_hidden()

            # ---- create a profile on the MLX model — ctx_size should be read-only ----
            mlx_card = page.locator('article[data-model-id="mlx-demo"]')
            mlx_card.locator("summary.lm-profile-add__summary").click()
            mlx_form = mlx_card.locator("form[action='/ui/models/profiles/create']")
            mlx_form.locator('input[name="name"]').fill("default-mlx")
            # Verify the ctx_size in basic view is disabled / informational.
            ctx_in = mlx_form.locator(
                '.lm-tab-panel[data-panel="basic"] input[disabled]'
            )
            expect(ctx_in.first).to_be_visible()
            mlx_form.locator('button[type="submit"]').click()
            page.wait_for_load_state("networkidle")
            mlx_card = page.locator('article[data-model-id="mlx-demo"]')
            expect(mlx_card.locator(".lm-profile__name", has_text="default-mlx")).to_be_visible()

            # ---- engine defaults section ----
            expect(page.locator("text=Engine defaults")).to_be_visible()
            # Open the llama defaults editor and read the JSON.
            llama_summary = page.locator(
                "summary.lm-profile-add__summary:has(code:has-text('llama'))"
            ).first
            # Already open if it had keys; click to ensure expanded.
            llama_details = llama_summary.locator("xpath=..")
            if llama_details.evaluate("(d) => !d.open"):
                llama_summary.click()
            llama_form = llama_details.locator(
                "form[action='/ui/models/default-args/save']"
            )
            current = llama_form.locator('textarea[name="args_json"]').input_value()
            assert '"temp"' in current, f"expected temp in llama defaults, got: {current}"

            # ---- delete the new fast profile ----
            gguf_card = page.locator(
                'article[data-model-id="demo-org/demo-model.gguf"]'
            )
            page.once("dialog", lambda dialog: dialog.accept())
            gguf_card.locator(
                "form[action='/ui/models/profiles/fast/delete'] button"
            ).click()
            page.wait_for_load_state("networkidle")
            gguf_card = page.locator(
                'article[data-model-id="demo-org/demo-model.gguf"]'
            )
            expect(gguf_card.locator(".lm-profile__name", has_text="fast")).to_have_count(0)

            print("OK: live UI smoke passed", flush=True)
            browser.close()
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(run(headed=("--headed" in sys.argv)))
