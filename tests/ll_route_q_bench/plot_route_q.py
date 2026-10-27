#!/usr/bin/env python3
"""把 route_q_bench.csv 画成 VF / scalar 两条折线（横轴核数，纵轴每次计算耗时）。"""

import csv
import sys


def load(path):
    series = {"scalar": [], "vf": []}
    shape = ""
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            name = "scalar" if row["mode"] == "0" else "vf"
            series[name].append((float(row["cores"]), float(row["us_per_rep"])))
            shape = "routes=%s experts=%s" % (row["routes"], row["experts"])
    for values in series.values():
        values.sort()
    return series, shape


def plot_matplotlib(series, shape, out_png, out_svg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    style = {"scalar": ("#c0392b", "o"), "vf": ("#1f6feb", "s")}
    for name in ("scalar", "vf"):
        points = series[name]
        if not points:
            continue
        xs, ys = zip(*points)
        color, marker = style[name]
        axis.plot(xs, ys, color=color, marker=marker, linewidth=1.8, markersize=4, label=name)
        for x, y in points:
            axis.annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(0, 6),
                          ha="center", fontsize=7, color=color)
    all_y = [y for name in series for _, y in series[name] if y > 0]
    if all_y:
        axis.set_ylim(min(all_y) * 0.7, max(all_y) * 1.4)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("AIV cores used")
    axis.set_ylabel("time per build (us)")
    axis.set_title("Dispatch route table build (%s)" % shape)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_svg)


def plot_svg(series, shape, out_svg):
    """matplotlib 缺失时的最小可用折线图，保证一键流程仍产出图片。"""
    width, height, pad = 720, 440, 60
    values = [y for name in series for _, y in series[name]]
    xs = sorted({x for name in series for x, _ in series[name]})
    if not values or not xs:
        raise SystemExit("empty CSV")
    x_min, x_max = min(xs), max(xs)
    y_max = max(values) * 1.1 or 1.0

    def px(x):
        if x_max == x_min:
            return pad + (width - 2 * pad) / 2.0
        return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)

    def py(y):
        return height - pad - y / y_max * (height - 2 * pad)

    colors = {"scalar": "#c0392b", "vf": "#1f6feb"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (width, height),
        '<rect width="100%%" height="100%%" fill="white"/>',
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (pad, pad, pad, height - pad),
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (pad, height - pad, width - pad, height - pad),
        '<text x="%d" y="20" font-size="14" font-family="sans-serif">Route table build (%s), us per build</text>'
        % (pad, shape),
    ]
    for name in ("scalar", "vf"):
        points = series[name]
        if not points:
            continue
        coords = " ".join("%.1f,%.1f" % (px(x), py(y)) for x, y in points)
        parts.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (colors[name], coords))
        for x, y in points:
            parts.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (px(x), py(y), colors[name]))
            parts.append('<text x="%.1f" y="%.1f" font-size="9" fill="%s">%.2f</text>'
                         % (px(x) + 4, py(y) - 4, colors[name], y))
        parts.append('<text x="%d" y="%d" font-size="12" fill="%s">%s</text>'
                     % (width - pad + 4, pad + 16 * (1 + list(series).index(name)), colors[name], name))
    for x in xs:
        parts.append('<text x="%.1f" y="%d" font-size="10" text-anchor="middle">%d</text>'
                     % (px(x), height - pad + 16, int(x)))
    parts.append('<text x="%d" y="%d" font-size="12" text-anchor="middle">AIV cores</text>'
                 % (width // 2, height - 15))
    parts.append("</svg>")
    with open(out_svg, "w") as handle:
        handle.write("\n".join(parts))


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: plot_route_q.py <csv> [--out figure.png]")
    csv_path = sys.argv[1]
    out = "route_q_bench.png"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    series, shape = load(csv_path)
    svg = out.rsplit(".", 1)[0] + ".svg"
    try:
        plot_matplotlib(series, shape, out, svg)
        print("figure: %s" % out)
    except ImportError:
        plot_svg(series, shape, svg)
        print("matplotlib 不可用，已写 SVG: %s" % svg)


if __name__ == "__main__":
    main()
