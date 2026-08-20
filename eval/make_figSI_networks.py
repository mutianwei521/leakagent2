"""
Supplementary Figure - topologies of the three water distribution networks used
in the results (EXA7 in-silico, BattLeDIM L-Town organizer-generated benchmark,
City H municipal in-silico, City D utility model, KY4 public benchmark). Pipes are drawn from the EPANET INP coordinates; pressure-sensor
nodes, sources and tanks are overlaid. Each network is scaled into a shared unit
box (aspect preserved) so the panels align; layout is 2 + 3 (a, b over c, d, e). Shares eval/figstyle.py.

Run (needs wntr):  <wds_rag python> eval/make_figSI_networks.py
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import wntr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

ROOT = fs.ROOT
fs.apply_rc()

# (label, provenance subtitle, inp, sensor-fingerprints json)
NETS = [
    ("EXA7", "in-silico benchmark", "data/exa7/Exa7.inp", "data/exa7/sensor_fingerprints.json"),
    ("L-Town", "BattLeDIM, generated benchmark", "data/ltown/L-TOWN.inp", "data/ltown/sensor_fingerprints.json"),
    ("City H", "municipal model, in-silico", "data/city_h/city_h.inp", "data/city_h/sensor_fingerprints.json"),
    ("City D", "field register, utility model", "data/city_d/city_d.inp", "data/city_d/sensor_fingerprints.json"),
    ("KY4", "public benchmark, in-silico", "data/ky4/ky4.inp", "data/ky4/sensor_fingerprints.json"),
]


def node_xy(wn, name):
    c = wn.get_node(name).coordinates
    return c if c else None


# 2 + 3 grid on a 6-column lattice: the two top panels span 3 columns each,
# the three bottom panels 2 columns each, so both rows fill the full width and
# every panel keeps a 1:1 map inside its cell.
W = fs.SIZE_2PANEL[0]
fig = plt.figure(figsize=(W, W * 0.80))
gs = fig.add_gridspec(2, 6, wspace=0.10, hspace=0.34, left=0.02, right=0.98,
                      top=0.955, bottom=0.10, height_ratios=[1.35, 1.0])
axes = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6]),
        fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])]

for i, (ax, (name, prov, inp, fp)) in enumerate(zip(axes, NETS)):
    if not os.path.exists(os.path.join(ROOT, inp)):
        # public checkout without the on-request utility model: placeholder panel
        ax.axis("off")
        ax.text(0.5, 0.5, name + " network model\navailable on request\n(see Data availability)",
                transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
                color="#777777", bbox=dict(boxstyle="round,pad=0.6", facecolor="#F4F4F4",
                                           edgecolor="#BBBBBB"))
        ax.text(0.0, 1.04, "abcde"[i], transform=ax.transAxes, fontsize=8.5,
                fontweight="bold", va="bottom", ha="left")
        ax.text(0.085, 1.04, name, transform=ax.transAxes, fontsize=7.3, va="bottom",
                ha="left", color=fs.INK)
        print(f"  [skip] {name}: {inp} not present, placeholder panel drawn")
        continue
    wn = wntr.network.WaterNetworkModel(os.path.join(ROOT, inp))
    # shared unit-box transform (centre + scale by the larger extent, aspect preserved)
    cs = [node_xy(wn, n) for n in wn.node_name_list if node_xy(wn, n)]
    xs, ys = [c[0] for c in cs], [c[1] for c in cs]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0

    def T(p):
        return (0.5 + (p[0] - cx) / span * 0.92, 0.5 + (p[1] - cy) / span * 0.92)

    # pipes
    segs = []
    for pn in wn.pipe_name_list:
        p = wn.get_link(pn)
        a, b = node_xy(wn, p.start_node_name), node_xy(wn, p.end_node_name)
        if a and b:
            segs.append([T(a), T(b)])
    ax.add_collection(LineCollection(segs, colors=fs.GREY_LT, linewidths=0.35, zorder=1))

    # pressure sensors (fingerprints files vary: keyed-by-node vs {"sensor_nodes": [...]})
    nodeset = set(wn.node_name_list)
    raw = json.load(open(os.path.join(ROOT, fp), encoding="utf-8"))
    cand = (raw.get("sensors") or raw.get("sensor_nodes") or list(raw.keys())) if isinstance(raw, dict) else raw
    sensors = [str(s) for s in cand if str(s) in nodeset]
    sc = [T(node_xy(wn, s)) for s in sensors]
    if sc:
        ax.scatter([q[0] for q in sc], [q[1] for q in sc], s=11, facecolor=fs.SYS_DEEP,
                   edgecolor="white", linewidths=0.4, zorder=3)
    # sources (reservoirs) and tanks
    rc = [T(node_xy(wn, n)) for n in wn.reservoir_name_list if node_xy(wn, n)]
    tc = [T(node_xy(wn, n)) for n in wn.tank_name_list if node_xy(wn, n)]
    if rc:
        ax.scatter([q[0] for q in rc], [q[1] for q in rc], s=32, marker="s",
                   facecolor=fs.RED, edgecolor="white", linewidths=0.4, zorder=4)
    if tc:
        ax.scatter([q[0] for q in tc], [q[1] for q in tc], s=34, marker="^",
                   facecolor=fs.AMBER, edgecolor=fs.INK, linewidths=0.4, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.0, 1.04, "abcde"[i], transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="left")
    ax.text(0.085, 1.04, name, transform=ax.transAxes, fontsize=7.3, va="bottom",
            ha="left", color=fs.INK)
    ax.text(0.5, -0.02,
            f"{prov}\n{wn.num_junctions} nodes · {wn.num_pipes} pipes\n{len(sc)} sensors",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.2, color=fs.GREY_D,
            linespacing=1.4)

handles = [
    Line2D([0], [0], color=fs.GREY_LT, lw=1.0, label="pipe"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=fs.SYS_DEEP,
           markeredgecolor="white", markersize=5, label="pressure sensor"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=fs.RED,
           markeredgecolor="white", markersize=5, label="reservoir / source"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor=fs.AMBER,
           markeredgecolor=fs.INK, markersize=5, label="tank"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.4,
           bbox_to_anchor=(0.5, 0.0))

fs.save(fig, "FigS1_networks")
