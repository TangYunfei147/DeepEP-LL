#!/usr/bin/env python3
"""把 route_q_bench.csv 画成单条折线：横轴核数，纵轴单轮 q 构建耗时。

实线 = 后一半采样的平均，虚线 = p99；p50 与原始采样留在 CSV 里。
"""

import csv
import sys

COLOR = "#1f6feb"


def load(path):
    points = []
    shape = ""
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            points.append((float(row["cores"]), float(row["us_avg"]), float(row["us_p99"])))
            shape = "routes=%s experts=%s" % (row["routes"], row["experts"])
    points.sort()
    return points, shape


def plot_matplotlib(points, shape, out_png, out_svg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    xs = [p[0] for p in points]
    axis.plot(xs, [p[1] for p in points], color=COLOR, marker="o", linewidth=1.8, markersize=4, label="avg")
    axis.plot(xs, [p[2] for p in points], color=COLOR, linestyle="--", linewidth=1.1, alpha=0.85, label="p99")
    for x, avg, _ in points:
        axis.annotate("%.2f" % avg, (x, avg), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7,
                      color=COLOR)
    values = [v for _, avg, p99 in points for v in (avg, p99) if v > 0]
    if values:
        axis.set_ylim(min(values) * 0.7, max(values) * 1.5)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("AIV cores used")
    axis.set_ylabel("q build time (us), last-half average / p99")
    axis.set_title("Dispatch route-table q build (%s)" % shape)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_svg)


def plot_svg(points, shape, out_svg):
    """matplotlib 缺失时的最小折线图，保证一键流程仍产出图片。"""
    width, height, pad = 720, 440, 60
    values = [v for _, avg, p99 in points for v in (avg, p99)]
    xs = [p[0] for p in points]
    if not values or not xs:
        raise SystemExit("empty CSV")
    x_min, x_max, y_max = min(xs), max(xs), max(values) * 1.15 or 1.0

    def px(x):
        return pad + (width - 2 * pad) / 2.0 if x_max == x_min else pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)

    def py(y):
        return height - pad - y / y_max * (height - 2 * pad)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (pad, pad, pad, height - pad),
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>' % (pad, height - pad, width - pad, height - pad),
        '<text x="%d" y="20" font-size="14" font-family="sans-serif">q build (%s), us: solid=avg dash=p99</text>'
        % (pad, shape),
        '<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>'
        % (COLOR, " ".join("%.1f,%.1f" % (px(x), py(y)) for x, y, _ in points)),
        '<polyline fill="none" stroke="%s" stroke-width="1" stroke-dasharray="4 3" points="%s"/>'
        % (COLOR, " ".join("%.1f,%.1f" % (px(x), py(y)) for x, _, y in points)),
    ]
    for x, y, _ in points:
        parts.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (px(x), py(y), COLOR))
        parts.append('<text x="%.1f" y="%.1f" font-size="9" fill="%s">%.2f</text>' % (px(x) + 4, py(y) - 4, COLOR, y))
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
    points, shape = load(csv_path)
    svg = out.rsplit(".", 1)[0] + ".svg"
    try:
        plot_matplotlib(points, shape, out, svg)
        print("figure: %s" % out)
    except ImportError:
        plot_svg(points, shape, svg)
        print("matplotlib 不可用，已写 SVG: %s" % svg)


if __name__ == "__main__":
    main()
