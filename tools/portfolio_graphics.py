"""Render the GitHub exhibits from recorded command pairs and feed counters."""
from __future__ import annotations

import math
import struct
from pathlib import Path
from xml.sax.saxutils import escape

BG, LIME, INK = "#0c1813", "#d8f989", "#edf4e7"
LINE, MUTE, AMBER = "#345240", "#a8c2ac", "#f4c48b"
MONO = "DejaVu Sans Mono,Menlo,monospace"
SANS = "DejaVu Sans,Arial,sans-serif"


def label(x, y, text, size=16, fill=MUTE, *, mono=True, bold=False, end=False):
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-family="{MONO if mono else SANS}" '
        f'font-size="{size}" font-weight="{700 if bold else 400}" '
        f'text-anchor="{"end" if end else "start"}">{escape(text)}</text>'
    )


def rect(x, y, w, h, fill, stroke="none", radius=0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" rx="{radius}"/>'


def path(d, stroke, width=1, *, fill="none", dash=""):
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
        f'stroke-dasharray="{dash or "none"}" stroke-linejoin="round"/>'
    )


def poly(points, stroke, width=1, *, fill="none", closed=False, dash=""):
    d = "M" + "L".join(f"{x:.2f} {y:.2f}" for x, y in points) + ("Z" if closed else "")
    return path(d, stroke, width, fill=fill, dash=dash)


def circle(x, y, r, fill, stroke="none"):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{stroke}"/>'


def write_svg(file: Path, w: int, h: int, title: str, content: list[str]) -> None:
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">',
        f'<title>{escape(title)}</title>',
        rect(1, 1, w - 2, h - 2, BG, LINE, 12),
        *content,
        '</svg>',
    ]
    file.write_text("\n".join(svg) + "\n", encoding="utf-8")


def panel_grid(x, y, w, h):
    pattern = (
        '<defs><pattern id="paper" width="16" height="16" patternUnits="userSpaceOnUse">'
        '<path d="M0 16V0h16" fill="none" stroke="#274236" stroke-width="1"/></pattern></defs>'
    )
    return [pattern, rect(x, y, w, h, "#10241a", LINE, 4), rect(x, y, w, h, "url(#paper)", radius=4)]


def pose(angles, ox, oy, scale=1):
    # Same angular offsets as the stage, kept schematically proportioned.
    shoulder = math.radians(-59) + angles[0]
    elbow = shoulder + math.radians(66) + angles[1]
    x = ox + 108 * scale * math.cos(shoulder)
    y = oy + 108 * scale * math.sin(shoulder)
    return [(ox, oy), (x, y), (x + 116 * scale * math.cos(elbow), y + 116 * scale * math.sin(elbow))]


def flatten(run):
    return [p for request in run["requests"] for p in request["commands"]]


def arm_art(samples):
    locate = lambda cmd: pose(cmd, 129, 171, .86)
    points = locate(samples[-1])
    d = [(x, y) for x, y in points]
    return [
        *panel_grid(20, 50, 380, 140),
        circle(129, 171, 67, "none", LINE),
        path("M33 171h348M129 67v108", LINE, dash="2 4"),
        poly([locate(c)[2] for c in samples[::8]], LIME, 1.5, dash="2 4"),
        rect(86, 176, 84, 6, "#345338", "#91ae75", 2),
        poly(d, "#a8cd92", 25), poly(d, "#4b7953", 21), poly(d, "#273f31", 7),
        *(circle(x, y, 11, "#1b3226", "#a8cd92") for x, y in points[:2]),
        *(circle(x, y, 4, LIME) for x, y in points),
        label(388, 68, "CPU / 250 Hz", 10, INK, end=True),
    ]


def step_line(samples, px, py):
    points = [(px(samples[0]["t"]), py(samples[0]["live"]))]
    for row in samples[1:]:
        x, y = px(row["t"]), py(row["live"])
        points.extend(((x, points[-1][1]), (x, y)))
    return points


def expiry_art(keep, drop):
    px = lambda t: 78 + 297 * t / 3.04
    py = lambda n: 170 - 85 * n / 16
    return [
        *panel_grid(20, 50, 380, 140), path("M78 69v101h297", LINE),
        *(path(f"M78 {py(y)}H375", LINE) for y in [5, 10, 15]),
        poly(step_line(keep["requests"], px, py), AMBER, 2, dash="4 4"),
        poly(step_line(drop["requests"], px, py), LIME, 2),
        label(68, py(15) + 3, str(keep["live_peak"]), 10, AMBER, end=True),
        label(68, py(12) + 3, str(drop["live_peak"]), 10, LIME, end=True),
        label(384, 67, "1,8× / A:B", 10, INK, end=True),
    ]


def full_expiry(file, keep, drop):
    px = lambda t: 77 + 617 * t / 3.04
    py = lambda n: 342 - 211 * n / 16
    horizon_ms = (drop["lead"] + drop["row_step"] * drop["rows_per_plan"]) * 1000
    panels = [
        label(35, 37, "FIG. 02  /  A/B · CPU", 16, LIME),
        label(35, 84, "Когда план заканчивается раньше", 39, INK, mono=False, bold=True),
        label(1037, 37, "1,8× / ОДИН ТЕМП", 16, MUTE, end=True),
        path("M35 104h1010M719 123v276", LINE),
        label(77, 130, "АКТИВНЫЕ ПЛАНЫ К КОНЦУ ОТВЕТА", 14),
    ]
    for count in (0, 5, 10, 15):
        y = py(count)
        panels += [path(f"M77 {y}H694", LINE), label(66, y + 4, str(count), 14, end=True)]
    for t in range(4):
        x = px(t)
        panels += [path(f"M{x} 142V342", LINE), label(x - 5, 365, str(t), 14)]
    panels += [
        poly(step_line(keep["requests"], px, py), AMBER, 3, dash="6 5"),
        poly(step_line(drop["requests"], px, py), LIME, 3),
        label(694, 388, "СИМ. ВРЕМЯ [с]", 14, end=True),
        rect(740, 124, 304, 82, "#172b21", LINE, 8),
        label(756, 151, "A / БЕЗ ОТСЕЧКИ", 14, AMBER),
        label(756, 186, f'{keep["live_peak"]} планов в пике', 24, INK, mono=False, bold=True),
        rect(740, 216, 304, 82, "#172b21", LINE, 8),
        label(756, 243, "B / С ОТСЕЧКОЙ", 14, LIME),
        label(756, 278, f'{drop["live_peak"]} планов в пике', 24, INK, mono=False, bold=True),
        label(742, 326, f'{drop["requests"][-1]["frozen_dropped"]} хвостов отброшено в B', 18, INK, mono=False),
        label(742, 353, f'{horizon_ms:.0f} мс до последней строки', 16, MUTE, mono=False),
        label(742, 378, f'{drop["blend_life"] * 1000:.0f} мс — предел жизни плана', 16, MUTE, mono=False),
        path("M35 404h1010", LINE),
        label(35, 431, "ДВА ПРОГОНА · ТЕСТОВАЯ ПОЛИТИКА · НЕ GREEN SIM", 14),
        label(1045, 431, f'{len(drop["requests"])} ЗАПРОСОВ В КАЖДОМ', 14, end=True),
    ]
    write_svg(file, 1080, 452, "A/B: число активных планов при 1,8×, с отсечкой и без", panels)


def relief_grid(run):
    # Raised ribs on a single sealed heightfield. x = request, y = tick in reply.
    count = run["commands_per_request"]
    grid = []

    def push(x, heights):
        grid.append([(x, j * 3.4, z) for j, z in enumerate(heights)])

    bed = [1.8] * count
    push(0., bed)
    for i, request in enumerate(run["requests"]):
        center = (i + .5) * 3.5
        heights = [2.2 + 22. * pair[1] for pair in request["commands"]]
        push(center - .9, heights)
        push(center + .9, heights)
        push((i + 1) * 3.5, bed)
    return grid


def make_stl(file, grid):
    depth, width = len(grid[0]), len(grid)
    triangles = []

    def quad(a, b, c, d):
        triangles.extend(((a, b, c), (a, c, d)))

    for i in range(width - 1):
        for j in range(depth - 1):
            quad(grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1])
    border = (
        [col[0] for col in grid[:-1]] + grid[-1][:-1]
        + [col[-1] for col in reversed(grid[1:])] + list(reversed(grid[0][1:]))
    )
    center = (grid[-1][-1][0] / 2, grid[-1][-1][1] / 2, 0.)
    for a, b in zip(border, border[1:] + border[:1]):
        u, v = (*a[:2], 0.), (*b[:2], 0.)
        quad(u, v, b, a)
        triangles.append((center, v, u))

    with file.open("wb") as out:
        out.write(b"Green Challenge / CPU elbow command relief, playback 1.3x".ljust(80, b"\0"))
        out.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
            normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
            length = math.hypot(*normal)
            out.write(struct.pack("<12fH", *(n / length for n in normal), *a, *b, *c, 0))


def relief_art(grid):
    def iso(point):
        x, y, z = point
        return 2.8 * x - 2. * y, .78 * x + .93 * y - 5.6 * z

    faces = []
    # A lean SVG preview. The binary mesh retains all 20 tick samples per rib.
    ticks = list(range(0, len(grid[0]) - 1, 5)) + [len(grid[0]) - 1]
    for i, (left, right) in enumerate(zip(grid, grid[1:])):
        for j, k in zip(ticks, ticks[1:]):
            corners = [left[j], right[j], right[k], left[k]]
            fill = ["#4c7850", "#84ae70", "#304f37"][i % 3]
            faces.append((left[j][0] + left[j][1], poly(list(map(iso, corners)), "#273f30", .46, fill=fill, closed=True)))
    art = [art for _, art in sorted(faces, key=lambda pair: pair[0])]
    # The two visible base walls of the isometric view.
    for boundary in ([col[-1] for col in grid], grid[-1]):
        wall = [(*boundary[0][:2], 0.), *boundary, (*boundary[-1][:2], 0.)]
        art.append(poly(list(map(iso, wall)), "#6b9368", .7, fill="#172d22", closed=True))
    return art


def full_relief(file, run, art):
    panels = [
        label(35, 39, "FIG. 03  /  SLICE OF TIME", 16, LIME),
        label(1044, 39, "STL · CPU / 1,3×", 16, MUTE, end=True),
        path("M35 57h1010M35 408h1010", LINE),
        label(36, 129, "Рельеф", 54, INK, mono=False, bold=True),
        label(36, 185, "команд", 54, INK, mono=False, bold=True),
        label(38, 232, f'{len(run["requests"])} рёбер / запросы', 20, LIME),
        label(38, 264, f'{run["commands_per_request"]} точек / такты', 20, LIME),
        label(38, 306, "Высота — команда локтя.", 19, MUTE, mono=False),
        label(38, 335, "Не геометрия робота.", 19, MUTE, mono=False),
        *panel_grid(387, 76, 658, 310),
        '<g transform="translate(557 175) scale(1.13)">', *art, '</g>',
        label(414, 101, "Y / ТАКТ В ОТВЕТЕ", 13),
        label(1020, 367, "X / НОМЕР ЗАПРОСА →", 13, end=True),
        label(35, 438, "38 × 20 / 760 ОТСЧЁТОВ · ЗАМКНУТАЯ СЕТКА", 14),
        label(1045, 438, "ПОВЕРНУТЬ НА GITHUB ↗", 14, LIME, end=True),
    ]
    write_svg(file, 1080, 459, "Рельеф 760 отсчётов команды локтя, 1,3×", panels)


def tile(file, code, title, drawing):
    write_svg(file, 420, 256, title, [
        label(20, 32, code, 15, LIME), label(399, 32, "GC / 26", 13, MUTE, end=True),
        *drawing, label(20, 233, title, 29, INK, mono=False, bold=True),
        label(394, 231, "↗", 22, LIME, end=True),
    ])


def arm_film(assets, run, font_dir):
    from PIL import Image, ImageDraw, ImageFont

    samples = flatten(run)
    total = len(samples)
    origin = (600, 278)
    locate = lambda cmd: pose(cmd, *origin, 1.14)

    def font(name, size):
        return ImageFont.truetype(str(font_dir / f"DejaVuSans{name}.ttf"), size)

    big, mono, small = font("-Bold", 48), font("Mono", 16), font("Mono", 12)
    base = Image.new("RGB", (960, 418), BG)
    d = ImageDraw.Draw(base)
    d.rounded_rectangle((1, 1, 958, 416), radius=16, outline=LINE)
    d.text((32, 27), "FIG. 01  /  CPU REPLAY", font=mono, fill=LIME, anchor="lt")
    d.text((929, 27), "1,3× / SCRIPTED", font=mono, fill=MUTE, anchor="rt")
    d.line((32, 53, 929, 53), fill=LINE)
    d.text((29, 83), "Команды\nво времени", font=big, fill=INK, spacing=6)
    d.text((32, 216), "ДВА УГЛА  /  4 МС НА ТАКТ", font=mono, fill=MUTE, anchor="lt")
    d.rounded_rectangle((32, 250, 369, 332), radius=8, fill="#192c22", outline=LINE)
    d.text((47, 265), "ПЛЕЧО", font=small, fill=MUTE, anchor="lt")
    d.text((204, 265), "ЛОКОТЬ", font=small, fill=MUTE, anchor="lt")
    d.rounded_rectangle((398, 67, 929, 332), radius=8, fill="#11271c", outline=LINE)
    for x in range(416, 927, 18):
        d.line((x, 73, x, 327), fill="#223d2e")
    for y in range(81, 327, 18):
        d.line((404, y, 923, y), fill="#223d2e")
    d.ellipse((origin[0] - 104, origin[1] - 104, origin[0] + 104, origin[1] + 45), outline="#466b50")
    d.text((412, 84), "JOINT SPACE / SCHEMATIC", font=small, fill=MUTE, anchor="lt")
    for x, y in (locate(c)[2] for c in samples[::12]):
        d.ellipse((x - 1, y - 1, x + 1, y + 1), fill="#93bc84")
    d.polygon([(539, 307), (569, 290), (628, 290), (665, 307), (665, 318), (539, 318)], fill="#2c533b", outline=MUTE)
    d.line((539, 307, 665, 307), fill=MUTE)
    d.line((32, 349, 929, 349), fill=LINE)
    d.line((32, 370, 929, 370), fill=LINE, width=2)
    d.text((32, 388), "ВЫХОДЫ АДАПТЕРА / НЕ ПОЗА РОБОТА", font=small, fill=MUTE, anchor="lt")

    def frame(i):
        im = base.copy()
        draw = ImageDraw.Draw(im)
        a, b, c = locate(samples[i])
        for p, q in ((a, b), (b, c)):
            draw.line((p, q), fill="#a0c68a", width=28)
            draw.line((p, q), fill="#528560", width=24)
            draw.line((p, q), fill="#2c523c", width=9)
        for (x, y), r in [(a, 18), (b, 15), (c, 8)]:
            draw.ellipse((x - r, y - r, x + r, y + r), fill="#172d23", outline=MUTE, width=2)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=LIME)
        draw.text((47, 297), f"{samples[i][0]:+.4f} rad", font=mono, fill=INK, anchor="lt")
        draw.text((204, 297), f"{samples[i][1]:+.4f} rad", font=mono, fill=INK, anchor="lt")
        x = 32 + i * 897 / (total - 1)
        draw.line((32, 370, x, 370), fill=LIME, width=2)
        draw.ellipse((x - 4, 366, x + 4, 374), fill=LIME)
        packet, tick = divmod(i, run["commands_per_request"])
        draw.text((929, 387), f't={i * run["control_dt"]:.3f} s  /  {packet + 1:02d}:{tick + 1:02d}', font=mono, fill=LIME, anchor="rt")
        return im

    frames = [frame(round((total - 1) * i / 35)) for i in range(36)]
    frames[-1].save(assets / "arm-replay.png", optimize=True)
    palette = frames[len(frames) // 2].quantize(colors=96)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frames[0].save(assets / "arm-replay.gif", save_all=True, append_images=frames[1:],
                   duration=[100] * 35 + [1400], optimize=True, disposal=1)


def render_exhibits(captures, assets: Path, font_dir: Path):
    first = captures["v13"]
    keep, drop = captures["v18-keep"], captures["v18-drop"]
    grid = relief_grid(first)
    art = relief_art(grid)
    tile(assets / "card-01.svg", "01 / REPLAY", "Пульт исполнения", arm_art(flatten(first)))
    tile(assets / "card-02.svg", "02 / A:B", "Два прогона", expiry_art(keep, drop))
    tile(assets / "card-03.svg", "03 / STL", "Рельеф времени", [
        *panel_grid(20, 50, 380, 140), '<g transform="translate(158 73) scale(.57)">', *art, '</g>',
        label(387, 68, "38 × 20", 10, INK, end=True),
    ])
    full_expiry(assets / "expiry-ab.svg", keep, drop)
    full_relief(assets / "command-relief.svg", first, art)
    make_stl(assets / "command-relief.stl", grid)
    arm_film(assets, first, font_dir)
