"""Build the manuscript AGN pipeline workflow figure.

The figure is intentionally generated from code so the manuscript-ready PDF and
SVG stay reproducible while preserving the supplied flowchart design.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


BOX_BLUE = colors.HexColor("#EAF3FB")
BOX_GREEN = colors.HexColor("#EAF7F2")
BOX_ORANGE = colors.HexColor("#FFF2DE")
BOX_PURPLE = colors.HexColor("#F0EEFF")
BOX_RED = colors.HexColor("#FBECE7")
BOX_PINK = colors.HexColor("#FCECF2")
BOX_GRAY = colors.HexColor("#F3F2EE")

STROKE_BLUE = colors.HexColor("#2B6DA8")
STROKE_GREEN = colors.HexColor("#1B7F68")
STROKE_ORANGE = colors.HexColor("#9A641C")
STROKE_PURPLE = colors.HexColor("#6657BF")
STROKE_RED = colors.HexColor("#A34A2B")
STROKE_PINK = colors.HexColor("#A64068")
STROKE_GRAY = colors.HexColor("#6B6B67")
ARROW = colors.HexColor("#6F6F6F")
FONT_REGULAR = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"


def _register_fonts() -> None:
    font_root = Path("/usr/local/texlive/2025/texmf-dist/fonts/truetype/public/dejavu")
    if FONT_REGULAR not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(font_root / "DejaVuSans.ttf")))
    if FONT_BOLD not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_BOLD, str(font_root / "DejaVuSans-Bold.ttf")))


def _wrap_text(text: str, font: str, size: float, max_width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if stringWidth(candidate, font, size) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_box(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    detail: Iterable[str] = (),
    *,
    fill: colors.Color,
    stroke: colors.Color,
) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1.1)
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)

    title_lines = _wrap_text(title, FONT_BOLD, 8.9, w - 18)
    detail_lines: list[str] = []
    for item in detail:
        detail_lines.extend(_wrap_text(item, FONT_REGULAR, 7.4, w - 18))

    all_lines = title_lines + detail_lines
    line_gap = 10.5
    block_h = (len(all_lines) - 1) * line_gap
    start_y = y + h / 2 + block_h / 2

    c.setFillColor(stroke)
    for i, line in enumerate(title_lines):
        c.setFont(FONT_BOLD, 8.9)
        c.drawCentredString(x + w / 2, start_y - i * line_gap, line)
    c.setFont(FONT_REGULAR, 7.4)
    for j, line in enumerate(detail_lines):
        c.drawCentredString(x + w / 2, start_y - (len(title_lines) + j) * line_gap, line)


def _arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float) -> None:
    c.setStrokeColor(ARROW)
    c.setFillColor(ARROW)
    c.setLineWidth(1.1)
    c.line(x1, y1, x2, y2)
    # Simple triangular head at the line end, sufficient for the vertical and
    # orthogonal connectors used here.
    if abs(x2 - x1) < abs(y2 - y1):
        direction = 1 if y2 > y1 else -1
        c.line(x2, y2, x2 - 3.2, y2 - direction * 5.0)
        c.line(x2, y2, x2 + 3.2, y2 - direction * 5.0)
    else:
        direction = 1 if x2 > x1 else -1
        c.line(x2, y2, x2 - direction * 5.0, y2 - 3.2)
        c.line(x2, y2, x2 - direction * 5.0, y2 + 3.2)


def _poly_arrow(c: canvas.Canvas, points: list[tuple[float, float]]) -> None:
    c.setStrokeColor(ARROW)
    c.setLineWidth(1.1)
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path)
    _arrow(c, points[-2][0], points[-2][1], points[-1][0], points[-1][1])


def build_pdf(output: Path) -> None:
    _register_fonts()
    width, height = 680, 720
    c = canvas.Canvas(str(output), pagesize=(width, height))
    c.setTitle("AGN pipeline workflow")
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    left_x, center_x, right_x = 40, 240, 480
    center_w, side_w = 200, 160
    center_mid = center_x + center_w / 2
    top = 674

    boxes = {
        "full": (center_x, top, center_w, 36),
        "holdout": (center_x, 606, center_w, 50),
        "obs": (center_x, 540, center_w, 42),
        "roles": (left_x, 448, side_w, 64),
        "vgae": (265, 448, 150, 64),
        "valid": (right_x, 448, side_w, 64),
        "latent": (center_x, 372, center_w, 48),
        "nodes": (center_x, 310, center_w, 42),
        "features": (center_x, 232, center_w, 58),
        "attach": (center_x, 154, center_w, 58),
        "aug": (center_x, 92, center_w, 42),
        "eval": (190, 16, 300, 56),
    }

    _draw_box(c, *boxes["full"], "Full reference graph G", fill=BOX_BLUE, stroke=STROKE_BLUE)
    _draw_box(
        c,
        *boxes["holdout"],
        "Node holdout",
        ("ρ = 5%, 10%, 20%",),
        fill=BOX_BLUE,
        stroke=STROKE_BLUE,
    )
    _draw_box(c, *boxes["obs"], "Observed graph G_obs", fill=BOX_BLUE, stroke=STROKE_BLUE)
    _draw_box(
        c,
        *boxes["roles"],
        "Structural role extraction",
        ("degree, community,", "PageRank, k-core"),
        fill=BOX_GREEN,
        stroke=STROKE_GREEN,
    )
    _draw_box(
        c,
        *boxes["vgae"],
        "VGAE training",
        ("on G_obs",),
        fill=BOX_BLUE,
        stroke=STROKE_BLUE,
    )
    _draw_box(
        c,
        *boxes["valid"],
        "Validation on G_obs",
        ("No hidden-node leakage",),
        fill=BOX_ORANGE,
        stroke=STROKE_ORANGE,
    )
    _draw_box(c, *boxes["latent"], "Role-conditioned latent generation", fill=BOX_PURPLE, stroke=STROKE_PURPLE)
    _draw_box(c, *boxes["nodes"], "Generate |H| synthetic nodes", fill=BOX_RED, stroke=STROKE_RED)
    _draw_box(
        c,
        *boxes["features"],
        "Feature generation",
        ("decoder + role interpolation + MMD",),
        fill=BOX_RED,
        stroke=STROKE_RED,
    )
    _draw_box(
        c,
        *boxes["attach"],
        "Hybrid attachment",
        ("learned + preferential +", "role similarity"),
        fill=BOX_PINK,
        stroke=STROKE_PINK,
    )
    _draw_box(c, *boxes["aug"], "Augmented graph (G')", fill=BOX_BLUE, stroke=STROKE_BLUE)
    _draw_box(
        c,
        *boxes["eval"],
        "Evaluation against full reference",
        ("endpoint precision@k, feature MMD,", "topology, downstream robustness"),
        fill=BOX_GRAY,
        stroke=STROKE_GRAY,
    )

    def bottom(name: str) -> tuple[float, float]:
        x, y, w, _h = boxes[name]
        return x + w / 2, y

    def top_edge(name: str) -> tuple[float, float]:
        x, y, w, h = boxes[name]
        return x + w / 2, y + h

    _arrow(c, *bottom("full"), *top_edge("holdout"))
    _arrow(c, *bottom("holdout"), *top_edge("obs"))
    _poly_arrow(c, [(center_mid - 55, 540), (center_mid - 115, 526), (120, 526), top_edge("roles")])
    _arrow(c, center_mid, 540, top_edge("vgae")[0], top_edge("vgae")[1])
    _poly_arrow(c, [(center_mid + 55, 540), (center_mid + 115, 526), (560, 526), top_edge("valid")])
    _poly_arrow(c, [(120, 448), (120, 434), (center_mid, 434), top_edge("latent")])
    _poly_arrow(c, [(560, 448), (560, 434), (center_mid, 434), top_edge("latent")])
    _arrow(c, *bottom("latent"), *top_edge("nodes"))
    _arrow(c, *bottom("nodes"), *top_edge("features"))
    _arrow(c, *bottom("features"), *top_edge("attach"))
    _arrow(c, *bottom("attach"), *top_edge("aug"))
    _arrow(c, *bottom("aug"), *top_edge("eval"))

    c.showPage()
    c.save()


def _svg_box(x: int, y: int, w: int, h: int, title: str, lines: list[str], fill: str, stroke: str) -> str:
    content = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.1"/>'
    ]
    all_lines = [title] + lines
    start_y = y + h / 2 - (len(all_lines) - 1) * 7
    for i, line in enumerate(all_lines):
        weight = "700" if i == 0 else "400"
        size = "14" if i == 0 else "12"
        content.append(
            f'<text x="{x + w / 2}" y="{start_y + i * 17}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{stroke}">{line}</text>'
        )
    return "\n".join(content)


def build_svg(output: Path) -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="680" height="720" viewBox="0 0 680 720" role="img">
<title>AGN pipeline workflow</title>
<rect width="680" height="720" fill="#ffffff"/>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="#6f6f6f" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<g fill="none" stroke="#6f6f6f" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow)">
<path d="M340 674 L340 656"/>
<path d="M340 606 L340 582"/>
<path d="M285 540 L225 526 L120 526 L120 512"/>
<path d="M340 540 L340 512"/>
<path d="M395 540 L455 526 L560 526 L560 512"/>
<path d="M120 448 L120 434 L340 434 L340 420"/>
<path d="M560 448 L560 434 L340 434 L340 420"/>
<path d="M340 372 L340 352"/>
<path d="M340 310 L340 290"/>
<path d="M340 232 L340 212"/>
<path d="M340 154 L340 134"/>
<path d="M340 92 L340 72"/>
</g>
"""
    boxes = [
        (240, 674, 200, 36, "Full reference graph G", [], "#EAF3FB", "#2B6DA8"),
        (240, 606, 200, 50, "Node holdout", ["ρ = 5%, 10%, 20%"], "#EAF3FB", "#2B6DA8"),
        (240, 540, 200, 42, "Observed graph G_obs", [], "#EAF3FB", "#2B6DA8"),
        (40, 448, 160, 64, "Structural role extraction", ["degree, community,", "PageRank, k-core"], "#EAF7F2", "#1B7F68"),
        (265, 448, 150, 64, "VGAE training", ["on G_obs"], "#EAF3FB", "#2B6DA8"),
        (480, 448, 160, 64, "Validation on G_obs", ["No hidden-node leakage"], "#FFF2DE", "#9A641C"),
        (240, 372, 200, 48, "Role-conditioned", ["latent generation"], "#F0EEFF", "#6657BF"),
        (240, 310, 200, 42, "Generate |H| synthetic nodes", [], "#FBECE7", "#A34A2B"),
        (240, 232, 200, 58, "Feature generation", ["decoder + role", "interpolation + MMD"], "#FBECE7", "#A34A2B"),
        (240, 154, 200, 58, "Hybrid attachment", ["learned + preferential +", "role similarity"], "#FCECF2", "#A64068"),
        (240, 92, 200, 42, "Augmented graph (G')", [], "#EAF3FB", "#2B6DA8"),
        (190, 16, 300, 56, "Evaluation against full reference", ["endpoint precision@k, feature MMD,", "topology, downstream robustness"], "#F3F2EE", "#6B6B67"),
    ]
    svg += "\n".join(_svg_box(*box) for box in boxes)
    svg += "\n</svg>\n"
    output.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(args.pdf)
    build_svg(args.svg)


if __name__ == "__main__":
    main()
