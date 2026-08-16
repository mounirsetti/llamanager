#!/usr/bin/env python3
"""Regenerate the llamanager wordmark and icon SVGs.

Run with a Python that has ``fonttools`` and ``brotli``:

    pip install fonttools brotli
    python tools/build_brand_assets.py

Why this exists: the SVGs used to carry a base64 woff2 and draw live
``<text>``, with the accent dot parked at a hardcoded x. That only lines up
while the embedded font loads *and* shapes identically — with the declared
fallback the dot landed on the letters, and even with the real font the
wordmark's dot overlapped the final "r" by 1.2px. Outlining the glyphs makes
the geometry fixed: the dot is placed a defined optical gap after the true
ink edge, and there is no font to fail to load.

The gap and the icon proportions are taken from assets/icon-*-512.png, the
tray icon this project already ships and likes.
"""
from __future__ import annotations

import pathlib

from fontTools.misc.transform import Transform
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FONT = ASSETS / "fraunces-wordmark.woff2"

#: The wordmark's typographic identity, matching the CSS (.lm-wordmark).
AXES = {"opsz": 96, "SOFT": 30}

#: Ink colours. Light = ink on paper, dark = cream on a dark tile.
INK_LIGHT, INK_DARK = "#181510", "#f3f1ee"
DOT_LIGHT, DOT_DARK = "#cf2f13", "#ef593f"
TILE_DARK, TILE_LIGHT = "#201e17", "#f8f6f3"

#: Icon lockup: gap between the last glyph's ink and the dot, in dot
#: *diameters*. Measured off icon-light-512.png (15px gap, 54px dot) — the
#: tray icon this project already ships.
GAP_RATIO = 0.28

#: Wordmark lockup: the same gap expressed in *em*, because a line of text
#: spaces optically with its type size rather than with the dot. Kept
#: identical to the CSS component (--wm-gap in .lm-wordmark) so the SVG and
#: the in-page wordmark are the same lockup at any size.
GAP_EM = 0.26


def _shaped(text: str, size: float, tracking: float):
    """Return (svg path data, ink bbox, pen-advance width) for ``text``."""
    font = instancer.instantiateVariableFont(TTFont(FONT), AXES, inplace=False)
    glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
    upem = font["head"].unitsPerEm
    scale = size / upem

    parts, x = [], 0.0
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for ch in text:
        name = cmap[ord(ch)]
        glyph = glyphs[name]
        # SVG's y grows downward, the font's upward: flip while scaling.
        t = Transform().translate(x, 0).scale(scale, -scale)
        pen = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}")
        glyph.draw(TransformPen(pen, t))
        d = pen.getCommands()
        if d:
            parts.append(d)
            bounds = BoundsPen(glyphs)
            glyph.draw(TransformPen(bounds, t))
            if bounds.bounds:
                x0, y0, x1, y1 = bounds.bounds
                xmin, ymin = min(xmin, x0), min(ymin, y0)
                xmax, ymax = max(xmax, x1), max(ymax, y1)
        x += glyph.width * scale + tracking
    return " ".join(parts), (xmin, ymin, xmax, ymax), x


def _svg(body: str, w: float, h: float) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" '
            f'fill="none">\n{body}</svg>\n')


def build_wordmark(dark: bool) -> str:
    """"llamanager" + dot, for page headers and the README."""
    size, tracking, baseline, pad = 52.0, -0.8, 54.0, 6.0
    d, (x0, _y0, x1, _y1), _adv = _shaped("llamanager", size, tracking)
    r = 5.5
    cx = pad + x1 + GAP_EM * size + r              # gap measured from real ink
    cy = baseline - 14.0                           # sits at x-height, as before
    width = cx + r + pad
    ink, dot = (INK_DARK, DOT_DARK) if dark else (INK_LIGHT, DOT_LIGHT)
    body = (f'  <path transform="translate({pad:g} {baseline:g})" '
            f'fill="{ink}" d="{d}"/>\n'
            f'  <circle cx="{cx:.2f}" cy="{cy:g}" r="{r:g}" fill="{dot}"/>\n')
    return _svg(body, round(width, 1), 72)


def build_icon(dark: bool) -> str:
    """Square "llam" tile — the tray icon's layout, as an SVG.

    Proportions come from icon-*-512.png: ink from 0.027W to 0.847W, band
    centred, dot at 0.9287W / 0.5967W with r 0.0527W.
    """
    W = 512.0
    r = 0.0527 * W
    dot_cx, dot_cy = 0.9287 * W, 0.5967 * W
    left = 0.027 * W
    right = dot_cx - r - GAP_RATIO * 2 * r         # where the ink must end
    top, bottom = 0.326 * W, 0.699 * W             # measured ink band

    # Shape once at a nominal size, then fit the ink box to the measured band.
    nominal = 128.0
    d, (x0, y0, x1, y1), _ = _shaped("llam", nominal, -1.0)
    s = min((right - left) / (x1 - x0), (bottom - top) / (y1 - y0))
    tx, ty = left - x0 * s, top - y0 * s

    ink, dot = (INK_DARK, DOT_DARK) if dark else (INK_LIGHT, DOT_LIGHT)
    # Both variants get a tile. The tray picks between them by *reachability*,
    # not by desktop theme (tray.py), so an ink-on-transparent light icon
    # disappeared on dark panels; the tile also gives the favicon a silhouette
    # on any tab strip.
    tile = (f'  <rect width="{W:g}" height="{W:g}" rx="{0.1855 * W:.1f}" '
            f'fill="{TILE_DARK if dark else TILE_LIGHT}"/>\n')
    body = (tile
            + f'  <path transform="translate({tx:.2f} {ty:.2f}) scale({s:.5f})" '
              f'fill="{ink}" d="{d}"/>\n'
              f'  <circle cx="{dot_cx:.1f}" cy="{dot_cy:.1f}" r="{r:.1f}" '
              f'fill="{dot}"/>\n')
    return _svg(body, W, W)


def rasterise() -> None:
    """Re-cut the PWA/tray PNGs from the icon SVGs.

    Separate from the SVG step because it needs a browser: playwright is the
    renderer already used for this project's UI screenshots, so the PNGs come
    out of the same engine that draws the favicon in a tab.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("skip PNGs: pip install playwright && playwright install chromium")
        return
    jobs = [("favicon.svg", "icon-light"), ("favicon-dark.svg", "icon-dark")]
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for src, stem in jobs:
            svg = (ASSETS / src).read_text(encoding="utf-8")
            for px in (192, 512):
                page = browser.new_page(viewport={"width": px, "height": px})
                page.set_content(
                    f'<body style="margin:0">{svg.replace(chr(60) + "svg ", chr(60) + f"svg width={px} height={px} ")}</body>')
                page.wait_for_timeout(120)
                page.screenshot(path=str(ASSETS / f"{stem}-{px}.png"))
                page.close()
                print(f"wrote {stem}-{px}.png")
        browser.close()


def main() -> None:
    for name, svg in (
        ("logo.svg", build_wordmark(dark=False)),
        ("logo-dark.svg", build_wordmark(dark=True)),
        ("favicon.svg", build_icon(dark=False)),
        ("favicon-dark.svg", build_icon(dark=True)),
    ):
        (ASSETS / name).write_text(svg, encoding="utf-8")
        print(f"wrote {name} ({len(svg)} bytes)")
    rasterise()


if __name__ == "__main__":
    main()
