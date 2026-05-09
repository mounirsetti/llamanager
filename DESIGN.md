# Design

## Visual theme

Editorial control room. Warm-ink dark surface (not navy, not pure black) with a single sharp accent used like a pull-quote mark. Asymmetric two-column layout on desktop with a heavy left rail for navigation and metadata, and a main canvas that mixes display-serif headings with tight sans body and mono data. The page reads more like a typeset front-page summary than a SaaS admin tool.

The aesthetic anchor is print-grade typography on a calm dark surface. Think Vercel observability crossed with The Economist's data pages, not Linear and not Bloomberg Terminal.

## Color palette

OKLCH, warm-tinted toward hue 80 (sepia/olive ink) so neutrals never read as cold blue. Two themes share one accent — a vermilion red used at <8% surface coverage for active state, the live indicator, and destructive emphasis. Tokens are theme-agnostic (`--bg-*` / `--fg-*`) so component CSS reads on both surfaces.

### Dark (default)

```
--bg-0:        oklch(0.14 0.008 80)   /* page background */
--bg-1:        oklch(0.18 0.008 80)   /* surface */
--bg-2:        oklch(0.22 0.010 80)   /* raised surface, zebra stripes */
--bg-3:        oklch(0.28 0.012 80)   /* hairline border */
--bg-4:        oklch(0.40 0.010 80)   /* divider on dark surface */

--fg-0:        oklch(0.96 0.005 80)   /* primary text, parchment-warm */
--fg-1:        oklch(0.82 0.005 80)   /* secondary text */
--fg-2:        oklch(0.62 0.008 80)   /* tertiary, captions */
--fg-3:        oklch(0.46 0.010 80)   /* disabled, placeholder */

--accent:      oklch(0.66 0.19 32)    /* vermilion — live, active, destructive */
--accent-soft: oklch(0.66 0.19 32 / 0.14)
--on-accent:   oklch(0.14 0.008 80)   /* text on a vermilion fill */

--ok:          oklch(0.70 0.13 155)   /* running — muted emerald */
--warn:        oklch(0.78 0.14 80)    /* swapping, starting — ochre */
--bad:         oklch(0.62 0.18 28)    /* crashed — accent's deeper sibling */
```

### Light

```
--bg-0:        oklch(0.97 0.006 80)   /* warm cream page, never #fff */
--bg-1:        oklch(0.94 0.008 80)
--bg-2:        oklch(0.91 0.010 80)
--bg-3:        oklch(0.84 0.012 80)
--bg-4:        oklch(0.74 0.010 80)

--fg-0:        oklch(0.20 0.010 80)   /* warm graphite, never #000 */
--fg-1:        oklch(0.36 0.010 80)
--fg-2:        oklch(0.50 0.010 80)
--fg-3:        oklch(0.64 0.010 80)

--accent:      oklch(0.56 0.20 32)    /* darker vermilion for AA on cream */
--accent-soft: oklch(0.56 0.20 32 / 0.10)
--on-accent:   oklch(0.97 0.006 80)

--ok:          oklch(0.50 0.14 155)
--warn:        oklch(0.55 0.14 70)
--bad:         oklch(0.52 0.20 28)
```

The user picks a theme; the choice persists in `localStorage` and is applied via `[data-theme]` on `<html>` before paint to prevent a light-flash on dark systems. Default falls back to `prefers-color-scheme`.

Color strategy: **Restrained** on both themes. One accent at <10% coverage. Status colors are muted; nothing in this palette is allowed to glow or saturate at full chroma.

## Typography

Three faces, system fallback always present.

- **Display serif**: `Fraunces` (variable, opsz + wght). Used only at sizes ≥24px. Page titles, single hero numbers, section markers. System fallback: `'Iowan Old Style', 'Charter', 'Georgia', serif`.
- **Body sans**: `Inter` variable. Used for body, labels, nav, buttons. System fallback: `system-ui, -apple-system, 'Segoe UI', sans-serif`.
- **Mono**: `JetBrains Mono` variable. Used for model IDs, keys, IP addresses, log lines, queue IDs. System fallback: `ui-monospace, 'SF Mono', Menlo, Consolas, monospace`.

Scale (1.25 ratio, slightly compressed at body sizes):

```
--t-mega:    clamp(2.5rem, 4vw + 1rem, 4rem)    /* hero numbers, display serif */
--t-h1:      clamp(1.75rem, 2vw + 1rem, 2.25rem) /* page titles, display serif */
--t-h2:      1.375rem    /* section markers, display serif italic */
--t-h3:      1.0625rem   /* subsection, sans semibold uppercase tracked */
--t-body:    0.9375rem   /* body sans */
--t-small:   0.8125rem   /* captions, table cells */
--t-tiny:    0.6875rem   /* labels, uppercase tracked */
```

Tracking: tight on display (-0.02em), normal on body, +0.08em uppercase on tiny labels and section markers.

## Layout

- **Desktop**: 240px left rail (nav + identity strip) + fluid main canvas, max width 1200px. Main canvas uses an asymmetric 8-column grid with deliberate "blank" cells — not every row fills the width. Section dividers are typographic (small caps section marker + hairline rule), never boxes.
- **Tablet (≤960px)**: rail collapses to a top strip, canvas stretches.
- **Phone (≤640px)**: single column, sticky condensed top bar with the live model + queue depth always visible. This is the Tailscale-glance surface — it must feel intentional, not "shrunk desktop."

Spacing scale: `4 / 8 / 12 / 16 / 24 / 32 / 48 / 72 / 112` px. Vary on purpose — section breaks at 48–72, in-row gaps at 8–12.

Cards: deliberately rare. Tables, hairline rules, and typography do most of the structural work. Where a "card" appears (e.g. the loaded-model summary), it's a single bordered surface, never a grid of identical ones.

## Components

- **Status pill**: 11px uppercase mono, dot glyph + label. Filled background only for `running` and `crashed`; the rest are outlined to keep the page calm.
- **Live indicator**: 6px vermilion dot with a subtle 1.4s breathing ring (disabled under `prefers-reduced-motion`).
- **Progress bar**: 2px hairline rule that becomes a solid vermilion segment, no shadow, no gradient.
- **Tables**: zebra at `--ink-1 / --ink-2`, hairline column dividers, 11px uppercase tracked headers in `--paper-2`.
- **Buttons**: text + 1px outline default; primary is solid vermilion on `--ink-0`. Destructive is text + vermilion underline-on-hover, not a red filled button.
- **Inputs**: bottom-rule only (no full borders), label sits above in tracked tiny caps. On focus, the rule thickens to 2px and changes to vermilion.
- **Code / mono blocks**: `--ink-1` background, hairline border, 13px JetBrains Mono.
- **Nav (left rail)**: vertical list, item is small caps tracked. Active item gets a leading vermilion bar (1px wide × full text height) + paper-0 text. No filled chip, no rounded background.
- **Toast / error**: full-width strip pinned to the top of the canvas, vermilion left rule (1px), no card.

## Motion

- All transitions: `cubic-bezier(0.22, 1, 0.36, 1)` (ease-out-quart), 180ms default, 280ms for layout-affecting changes.
- No bounce, no elastic, no pulsing rings except the live indicator.
- HTMX swap: 120ms opacity fade only; no slide, no scale.
- `prefers-reduced-motion: reduce` → all motion ≤80ms or removed entirely; live indicator becomes static.

## Iconography

Lucide line icons at 16px / 1.5px stroke, paper-1 by default. No filled icons. Inline SVG, no icon font.
