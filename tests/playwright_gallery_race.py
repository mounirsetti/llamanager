"""Lightbox race on the public gallery pages, driven in a real browser.

Open a slow clip, give up on it, open a fast one: the slow fetch lands last
and — before the generation guard in openLightbox — painted its bytes into
the panel showing the OTHER clip. The symptom is a lightbox whose metadata
says one video while the player holds another.

Not part of the pytest suite: it needs playwright and ffmpeg. Run it directly.

    .venv/bin/python tests/playwright_gallery_race.py

Prints PASS/FAIL. To watch it fail, delete the `gen === lbGen` checks from
videos_public.html first — the control run shows ~37 KB (clip A) in a panel
captioned fast_B.
"""
import asyncio, json, socket, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from playwright.async_api import async_playwright

REPO = str(Path(__file__).resolve().parent.parent)
def pick_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

data = Path(tempfile.mkdtemp(prefix="lm-vid-")); port = pick_port()
proc = subprocess.Popen([sys.executable, "tests/_pw_server.py", str(data), str(port)], cwd=REPO)
base = f"http://127.0.0.1:{port}"
for _ in range(120):
    try: urllib.request.urlopen(base + "/ui/login", timeout=1); break
    except Exception: time.sleep(0.5)
key = (data / "bootstrap.key").read_text().strip()

day = time.strftime("%Y-%m-%d")
gal = data / "images" / day / "bootstrap"
gal.mkdir(parents=True, exist_ok=True)
# Real, decodable clips: a broken tile deliberately swallows clicks, so a
# fake mp4 would never reach the lightbox at all.
for name, secs, size in (("slow_A.mp4", 6, "640x480"), ("fast_B.mp4", 1, "128x96")):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=size={size}:rate=24:duration={secs}",
                    "-pix_fmt", "yuv420p", str(gal / name)], check=True)
    (gal / (name + ".json")).write_text(json.dumps({"prompt": name, "model_id": "m"}))
print("clip sizes:", {n: (gal / n).stat().st_size for n in ("slow_A.mp4", "fast_B.mp4")})

async def main():
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        ctx = await b.new_context(viewport={"width": 1400, "height": 900})
        await ctx.add_init_script(f"localStorage.setItem('lm-images-api-key', {key!r})")
        pg = await ctx.new_page()
        pg.on("pageerror", lambda e: print("PAGEERROR:", e))
        pg.on("console", lambda m: print("CONSOLE:", m.type, m.text[:160]) if m.type in ("error","warning") else None)

        async def slow(route):
            # Hold A's ORIGINAL back (not its thumbnail) so the newer click
            # resolves first — the exact ordering the bug needs.
            await asyncio.sleep(2.5)
            await route.continue_()
        await pg.route(lambda u: "slow_A.mp4" in u and "thumb" not in u, slow)

        await pg.goto(base + "/videos", wait_until="networkidle")
        await pg.wait_for_timeout(1500)
        names = await pg.eval_on_selector_all(
            ".gen-card",
            "els=>els.map(e=>{const m=e.outerHTML.match(/slow_A|fast_B/);return m?m[0]:''})")
        print("cards:", names)
        cards = pg.locator(".gen-card")
        ia = names.index("slow_A"); ib = names.index("fast_B")
        # The lightbox is modal, so the real sequence is: open the slow clip,
        # give up on it, open another. A's bytes land while B is on screen.
        await cards.nth(ia).click()          # slow clip, still fetching
        await pg.wait_for_timeout(250)
        await pg.keyboard.press("Escape")    # close before it loads
        await pg.wait_for_timeout(100)
        await cards.nth(ib).click()          # the one the operator wants
        await pg.wait_for_timeout(5000)      # long enough for A to land too

        print("lightbox hidden:", await pg.eval_on_selector("#gen-lightbox", "e=>e.hidden"))
        print("video els:", await pg.locator("#gen-lb-video").count())
        prompt = await pg.eval_on_selector("#gen-lb-prompt", "e=>e.textContent.trim()")
        vsrc = await pg.eval_on_selector(
            "#gen-lb-video", "v=>v.getAttribute('src')||''") \
            if await pg.locator("#gen-lb-video").count() else ""
        blob_of = await pg.evaluate(
            "async (s) => { if(!s) return ''; const r = await fetch(s);"
            " const t = await r.text(); return t.length; }", vsrc)
        print("panel prompt :", prompt)
        print("video bytes  :", blob_of)
        # The panel says B; the bytes in the <video> must be B's too, or a
        # stale fetch painted over it after the fact.
        ok = prompt.startswith("fast_B") and 3000 < int(blob_of or 0) < 12000
        print("RESULT:", "PASS — newest click owns the panel" if ok
              else f"FAIL — stale item shown (prompt={prompt!r} bytes={blob_of})")
        await b.close()

try:
    asyncio.run(main())
finally:
    proc.terminate(); proc.wait(timeout=10)
