#!/usr/bin/env python3
"""Render the portfolio GIF and still from a single tools/smoke_test.py JSONL run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1040, 468
BG = "#0b1712"
GRID = "#294338"
INK = "#edf4e7"
MUTED = "#a5c2ac"
LIME = "#d8f989"


def load_trace(path: Path) -> list[dict]:
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(samples) < 2:
        raise ValueError("capture at least two requests")
    times = [sample["t"] for sample in samples]
    if times[0] != 0 or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("expected a single fresh run, starting at t=0")
    if len({(s["rows_in"], s["rows_out"], s["speed"]) for s in samples}) != 1:
        raise ValueError("expected fixed input/output sizes and playback speed")
    return samples


def render(samples: list[dict], font_dir: Path, output_dir: Path) -> None:
    mono = font_dir / "DejaVuSansMono.ttf"
    bold = font_dir / "DejaVuSans-Bold.ttf"
    label = ImageFont.truetype(str(mono), 16)
    small = ImageFont.truetype(str(mono), 14)
    title = ImageFont.truetype(str(bold), 42)
    number = ImageFont.truetype(str(bold), 35)

    end = samples[-1]
    peak = max(sample["live"] for sample in samples)
    time_step = max(1, math.ceil(end["t"] / 3))
    count_step = max(1, math.ceil(peak / 3))
    left, top, right, bottom = 82, 166, 709, 371

    def xs(t: float) -> float:
        return left + (right - left) * t / (3 * time_step)

    def ys(count: int) -> float:
        return bottom - (bottom - top) * count / (3 * count_step)

    points = [(xs(s["t"]), ys(s["live"])) for s in samples]

    def steps(stop: int) -> list[tuple[float, float]]:
        line = [points[0]]
        for x, y in points[1:stop]:
            line += [(x, line[-1][1]), (x, y)]
        return line

    base = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle((1, 1, WIDTH - 2, HEIGHT - 2), radius=18, outline=GRID, width=2)
    draw.text((32, 23), "02 / VERIFICATION", font=label, fill=LIME, anchor="lt")
    draw.text((32, 53), "CPU smoke trace", font=title, fill=INK, anchor="lt")
    draw.rounded_rectangle((816, 21, 1008, 54), radius=16, fill="#203b2c")
    draw.ellipse((830, 34, 837, 41), fill=LIME)
    draw.text((848, 37), "SCRIPTED POLICY", font=small, fill=INK, anchor="lm")
    draw.line((32, 113, 1008, 113), fill=GRID)
    draw.line((738, 134, 738, 409), fill=GRID)
    draw.text((left, 132), "ACTIVE PLANS", font=label, fill=MUTED, anchor="lt")

    for i in range(4):
        count, t = i * count_step, i * time_step
        y, x = ys(count), xs(t)
        draw.line((left, y, right, y), fill=GRID)
        draw.line((x, top, x, bottom), fill=GRID)
        draw.text((left - 13, y), str(count), font=small, fill=MUTED, anchor="rm")
        draw.text((x, bottom + 16), str(t), font=small, fill=MUTED, anchor="mt")
    draw.text((right, 411), "SIM TIME [s]", font=small, fill=MUTED, anchor="rb")

    draw.line(steps(len(points)), fill="#5a8167", width=2)
    for x, y in points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#78a485")

    stats = [
        (str(len(samples)), "REQUESTS"),
        (f'{end["rows_in"]} → {end["rows_out"]}', "PLAN ROWS → COMMANDS"),
        (str(peak), "PEAK LIVE PLANS"),
    ]
    for i, (value, name) in enumerate(stats):
        y = 132 + i * 94
        draw.rounded_rectangle((760, y, 1008, y + 82), radius=9, fill="#172b21", outline=GRID)
        draw.text((779, y + 12), value, font=number, fill=LIME if i == 1 else INK, anchor="lt")
        draw.text((781, y + 59), name, font=small, fill=MUTED, anchor="lt")

    draw.line((32, 424, 1008, 424), fill=GRID)
    draw.text(
        (32, 440), f'BOUNDARY MODE  /  {end["speed"]:.2f}× PLAYBACK',
        font=label, fill=MUTED, anchor="lt",
    )

    def frame(index: int) -> Image.Image:
        im = base.copy()
        d = ImageDraw.Draw(im)
        x, y = points[index]
        for tick in range(top, bottom, 8):
            d.line((x, tick, x, min(tick + 4, bottom)), fill="#769578")
        d.line(steps(index + 1), fill=LIME, width=3)
        d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=BG, outline=LIME)
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=LIME)
        d.text(
            (1008, 440), f't={samples[index]["t"]:0.2f} s  ·  REQ {index + 1:02d}',
            font=label, fill=LIME, anchor="rt",
        )
        return im

    output_dir.mkdir(parents=True, exist_ok=True)
    still = frame(len(points) - 1)
    still.save(output_dir / "cpu-trace.png", optimize=True)
    count = min(38, len(points))
    frames = [frame(round(i * (len(points) - 1) / (count - 1))) for i in range(count)]
    palette = frames[count // 2].quantize(colors=96)
    indexed = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
    # No loop extension: play once, then leave the complete chart on screen.
    indexed[0].save(
        output_dir / "cpu-trace.gif", save_all=True, append_images=indexed[1:],
        duration=[100] * (count - 1) + [1600], optimize=True, disposal=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--font-dir", type=Path, default=Path("/usr/share/fonts/truetype/dejavu"))
    args = parser.parse_args()
    render(load_trace(args.trace), args.font_dir, args.output_dir)


if __name__ == "__main__":
    main()
