#!/usr/bin/env python3
"""把 route_q_bench.csv 画成 VF / scalar 折线（横轴核数，纵轴单轮耗时）。

实线 = 后一半采样的平均，虚线 = p99；p50 与原始采样留在 CSV 里。
"""

import csv
import sys

SERIES = ("scalar", "vf", "prod")
COLOR = {"scalar": "#c0392b", "vf": "#1f6feb", "prod": "#2e7d32"}
MARKER = {"scalar": "o", "vf": "s", "prod": "^"}
MODE_NAME = {"0": "scalar", "1": "vf", "2": "prod"}
LABEL = {"scalar": "scalar (expert scan)", "vf": "VF (expert scan)", "prod": "production q (histogram)"}


def load(path):
    data = {name: [] for name in SERIES}
    shape = ""
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            name = MODE_NAME[row["mode"]]
            data[name].append((float(row["cores"]), float(row["us_avg"]), float(row["us_p99"])))
            shape = "routes=%s experts=%s" % (row["routes"], row["experts"])
    for values in data.values():
        values.sort()
    return data, shape


def plot_matplotlib(data, shape, out_png, out_svg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.4, 4.6), dpi=160)
    for name in SERIES:
        points = data[name]
        if not points:
            continue
        xs = [p[0] for p in points]
        axis.plot(xs, [p[1] for p in points], color=COLOR[name], marker=MARKER[name], linewidth=1.8,
                  markersize=4, label="%s avg" % LABEL[name])
        axis.plot(xs, [p[2] for p in points], color=COLOR[name], linestyle="--", linewidth=1.1, alpha=0.85,
                  label="%s p99" % LABEL[name])
        for x, avg, _ in points:
            axis.annotate("%.2f" % avg, (x, avg), textcoords="offset points", xytext=(0, 7), ha="center",
                          fontsize=7, color=COLOR[name])
    values = [v for name in SERIES for _, avg, p99 in data[name] for v in (avg, p99) if v > 0]
    if values:
        axis.set_ylim(min(values) * 0.7, max(values) * 1.5)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("AIV cores used")
    axis.set_ylabel("time per build (us), last-half average / p99")
    axis.set_title("Dispatch route table build (%s)" % shape)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_svg)


def plot_svg(data, shape, out_svg):
    """matplotlib 缺失时的最小折线图，保证一键流程仍产出图片。"""
    width, height, pad = 720, 440, 60
    values = [v for name in SERIES for _, avg, p99 in data[name] for v in (avg, p99)]
    xs = sorted({x for name in SERIES for x, _, _ in data[name]})
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
        '<text x="%d" y="20" font-size="14" font-family="sans-serif">Route table build (%s), us: solid=avg dash=p99</text>'
        % (pad, shape),
    ]
    for name in SERIES:
        if not data[name]:
            continue
        avg = " ".join("%.1f,%.1f" % (px(x), py(y)) for x, y, _ in data[name])
        p99 = " ".join("%.1f,%.1f" % (px(x), py(y)) for x, _, y in data[name])
        parts.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (COLOR[name], avg))
        parts.append('<polyline fill="none" stroke="%s" stroke-width="1" stroke-dasharray="4 3" points="%s"/>'
                     % (COLOR[name], p99))
        for x, y, _ in data[name]:
            parts.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (px(x), py(y), COLOR[name]))
        parts.append('<text x="%d" y="%d" font-size="12" fill="%s">%s</text>'
                     % (width - pad - 90, pad + 16 * (1 + SERIES.index(name)), COLOR[name], name))
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
    data, shape = load(csv_path)
    svg = out.rsplit(".", 1)[0] + ".svg"
    try:
        plot_matplotlib(data, shape, out, svg)
        print("figure: %s" % out)
    except ImportError:
        plot_svg(data, shape, svg)
        print("matplotlib 不可用，已写 SVG: %s" % svg)


if __name__ == "__main__":
    main()
