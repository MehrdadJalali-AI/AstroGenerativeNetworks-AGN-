from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


def render(spec_path: str | Path, output_path: str | Path) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    W, H = float(spec["width"]), float(spec["height"])
    c = canvas.Canvas(str(output_path), pagesize=(W, H))

    def txt(x, y, s, size=11, bold=False, color="#111111", anchor="start"):
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)
        c.setFillColor(HexColor(color))
        width = c.stringWidth(s, font, size)
        xx = x if anchor == "start" else x - width / 2 if anchor == "middle" else x - width
        c.drawString(xx, y, s)

    def edge(p1, p2, kind):
        color = {"observed": "#222222", "removed": "#b8b8b8", "agn": "#ff5a00"}[kind]
        width = {"observed": 1.2, "removed": 1.0, "agn": 1.8}[kind]
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(width)
        c.setDash([5, 4] if kind == "removed" else [])
        c.line(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))
        c.setDash([])

    def node(x, y, kind):
        x, y = float(x), float(y)
        if kind == "observed":
            c.setFillColor(HexColor("#2b7ed8"))
            c.setStrokeColor(HexColor("#111111"))
            c.setLineWidth(1.2)
            c.circle(x, y, 7.5, fill=1, stroke=1)
        elif kind == "hidden":
            c.setFillColor(HexColor("#d2d2d2"))
            c.setStrokeColor(HexColor("#666666"))
            c.setLineWidth(1.4)
            c.circle(x, y, 8.0, fill=1, stroke=1)
        elif kind == "hidden_ghost":
            c.setFillColor(HexColor("#ffffff"))
            c.setStrokeColor(HexColor("#888888"))
            c.setLineWidth(1.2)
            c.setDash([4, 3])
            c.circle(x, y, 8.0, fill=0, stroke=1)
            c.setDash([])
        else:
            c.setFillColor(HexColor("#ffa735"))
            c.setStrokeColor(HexColor("#111111"))
            c.setLineWidth(1.2)
            c.circle(x, y, 8.2, fill=1, stroke=1)

    c.setFillColor(HexColor("#ffffff"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    for panel in spec["panels"]:
        x, y, w, h = float(panel["x"]), float(panel["y"]), float(panel["w"]), float(panel["h"])
        c.setFillColor(HexColor("#fbfcfd"))
        c.setStrokeColor(HexColor("#cfd6dc"))
        c.setLineWidth(1.0)
        c.roundRect(x, y, w, h, 8, fill=1, stroke=1)
        txt(x + 12, y + h - 28, panel["label"], 24, True)
        txt(x + w / 2, y + h - 25, panel["title"], 14, True, anchor="middle")
    for e in spec["edges"]:
        edge(e["source"], e["target"], e["kind"])
    for n in spec["nodes"]:
        node(n["pos"][0], n["pos"][1], n["kind"])

    legend_y = 32
    node(160, legend_y + 6, "observed")
    txt(174, legend_y + 2, "Observed node", 10)
    node(300, legend_y + 6, "hidden")
    txt(315, legend_y + 2, "Hidden true node", 10)
    node(460, legend_y + 6, "generated")
    txt(475, legend_y + 2, "Generated synthetic node", 10)
    edge((665, legend_y + 6), (708, legend_y + 6), "observed")
    txt(716, legend_y + 2, "Observed edge", 10)
    edge((810, legend_y + 6), (853, legend_y + 6), "removed")
    txt(861, legend_y + 2, "Removed edge", 10)
    edge((955, legend_y + 6), (998, legend_y + 6), "agn")
    txt(1006, legend_y + 2, "AGN edge", 10)
    c.showPage()
    c.save()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render real AGN graph example JSON to vector PDF.")
    parser.add_argument("spec_json")
    parser.add_argument("output_pdf")
    args = parser.parse_args(argv)
    render(args.spec_json, args.output_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
