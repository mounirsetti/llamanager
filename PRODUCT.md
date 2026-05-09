# Product

## Register

product

## Users

A solo home-lab operator running llamanager on their own workstation. They check the dashboard from a desktop browser at home and from a phone over Tailscale, often glancing during long inference jobs or while triaging a misbehaving model. They are technically fluent — they know what a GGUF is, they read logs without flinching, they have opinions about quantization. Their context is "is the box healthy and is the queue moving," not "teach me what an origin is."

## Product Purpose

llamanager turns one local GPU into a private inference service that survives multiple clients trampling each other. The UI exists so the operator can see queue depth, model state, origin activity, and crash history at a glance, and so they can pull, swap, or kill a model without dropping into the CLI. Success looks like: glance at the page, know within two seconds whether something needs attention, never get blocked from doing the obvious next action.

## Brand Personality

Quiet, competent, opinionated. Three words: precise, unhurried, infrastructural. The voice should feel like a well-written README from someone who actually ships, not a marketing page. No exclamation marks, no "Welcome back," no friendly mascots. Status changes are stated, not celebrated.

## Anti-references

- **Stock Pico / Bootstrap admin panels** — the current state. Readable but anonymous; could be any tool.
- **Generic SaaS dashboards** — Stripe-clone navy, hero metrics, identical card grids, big number + small label + gradient accent.
- **Crypto / AI neon-on-black** — cyan + magenta gradients, glassmorphism, glow text.
- **Consumer chat-product warmth** — soft pastels, oversized rounded buttons, lots of friendly whitespace. This is infra, not a chat app.

## Design Principles

1. **Glanceable before interactive.** The first frame must answer "is anything wrong" without scrolling, clicking, or reading. Status pills, queue depth, and model state are the load-bearing elements; everything else is supporting evidence.
2. **Editorial, not dashboard.** Treat the page like a layout in a print publication: asymmetric grid, deliberate negative space, a single typographic personality. Density is fine; uniformity is not.
3. **Density without monotony.** This is an operator tool — show a lot, but vary scale and rhythm so the eye knows where to land. Tables earn their place; cards do not.
4. **Phone is a real surface, not a courtesy.** Tailscale glances on a phone are a primary use case. Layout reflows to one column without losing the at-a-glance summary.
5. **No ceremony around destructive actions.** Confirm where it matters (delete a model, force-stop), but don't moat every action behind a modal. Inline confirms beat dialog stacks.

## Accessibility & Inclusion

- WCAG 2.1 AA contrast on all text and status indicators. Status is never conveyed by color alone — every pill has a glyph or label.
- Respect `prefers-reduced-motion`: progress bars and live indicators settle to a static state, no pulsing.
- Keyboard navigable. Visible focus rings, not the browser default — but never invisible.
- Mono and serif fall back gracefully when web fonts fail; the design must hold up on system fonts alone.
