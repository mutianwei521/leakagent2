# -*- coding: utf-8 -*-
"""Partition (hydraulic zoning) maps in the project's territory style:
Voronoi cells of the node assignments merged per zone, clipped to the frame,
orphan fragments and gaps reassigned to the nearest zone (machinery adapted
from the author's earlier partition-figure generator). Two outputs:
  Fig5_partitions_main : EXA7 + City D (main text)
  FigS3_partitions     : L-Town + City H + KY4 (SI)
Zone fills are categorical pastels (zone identity only); markers follow
figstyle (red sensors, dark reservoirs)."""
import os
import sys
import json
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

ROOT = fs.ROOT
fs.apply_rc()

ZONE_FILLS = ['#D6EAF8', '#D5F5E3', '#FCF3CF', '#E8DAEF', '#D1F2EB', '#FDEBD0',
              '#D4E6F1', '#D0ECE7', '#FEF9E7', '#D7BDE2', '#A9DFBF', '#AED6F1',
              '#F9E79F', '#D2B4DE', '#A3E4D7', '#FADBD8', '#85C1E9', '#82E0AA',
              '#F8C471', '#BB8FCE', '#76D7C4', '#D5D8DC', '#5DADE2', '#58D68D',
              '#F4D03F', '#AF7AC5', '#48C9B0', '#AEB6BF', '#73C6B6', '#85929E']
ZONE_BORDERS = ['#2980B9', '#27AE60', '#D4AC0D', '#7D3C98', '#1ABC9C', '#CA6F1E',
                '#2471A3', '#17A589', '#B7950B', '#6C3483', '#229954', '#2E86C1',
                '#D68910', '#884EA0', '#138D75', '#566573', '#1F618D', '#1E8449',
                '#B9770E', '#6C3483', '#117A65', '#7F8C8D', '#2471A3', '#28B463',
                '#B7950B', '#7D3C98', '#117864', '#5D6D7E', '#148F77', '#515A5A']
FILL_ALPHA = 0.45

NETS = {
    "exa7":    dict(inp="Exa7.inp",   K=15, title="EXA7 (381 junctions)"),
    "city_d":  dict(inp="city_d.inp", K=15, title="City D (541 junctions)"),
    "ltown":   dict(inp="L-TOWN.inp", K=25, title="L-Town (782 junctions)"),
    "city_h": dict(inp="city_h.inp", K=25, title="City H (920 junctions)"),
    "ky4":     dict(inp="ky4.inp",    K=25, title="KY4 (959 junctions)"),
}


def parse_inp(path):
    nodes, pipes, sec = {}, [], None
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line.upper()
            continue
        p = line.split()
        if sec == "[JUNCTIONS]" and p:
            nodes[p[0]] = {"type": "junction"}
        elif sec == "[RESERVOIRS]" and p:
            nodes[p[0]] = {"type": "reservoir"}
        elif sec == "[TANKS]" and p:
            nodes[p[0]] = {"type": "tank"}
        elif sec in ("[PIPES]", "[PUMPS]", "[VALVES]") and len(p) >= 3:
            pipes.append((p[1], p[2]))
        elif sec == "[COORDINATES]" and len(p) >= 3:
            nodes.setdefault(p[0], {"type": "junction"})
            nodes[p[0]]["x"], nodes[p[0]]["y"] = float(p[1]), float(p[2])
    return nodes, pipes


def voronoi_zones(pos, partition, margin_frac=0.06):
    nids = [n for n in partition if n in pos]
    if not nids:
        return {}
    pts = np.array([pos[n] for n in nids])
    pid = [partition[n] for n in nids]
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    dx, dy = (xmax - xmin) * margin_frac, (ymax - ymin) * margin_frac
    bbox = box(xmin - dx, ymin - dy, xmax + dx, ymax + dy)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    span = max(xmax - xmin, ymax - ymin) * 4
    mirror = np.array([[cx - span, cy - span], [cx, cy - span], [cx + span, cy - span],
                       [cx - span, cy], [cx + span, cy],
                       [cx - span, cy + span], [cx, cy + span], [cx + span, cy + span]])
    vor = Voronoi(np.vstack([pts, mirror]))
    cells = {}
    for i in range(len(pts)):
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            continue
        try:
            poly = Polygon([vor.vertices[v] for v in reg])
            if not poly.is_valid:
                poly = poly.buffer(0)
            c = poly.intersection(bbox)
            if not c.is_empty:
                cells[i] = c
        except Exception:
            continue
    grp = defaultdict(list)
    for i, poly in cells.items():
        grp[pid[i]].append(poly)
    zones, orphans = {}, []
    for z, polys in grp.items():
        try:
            m = unary_union(polys)
            if isinstance(m, MultiPolygon):
                comp = sorted(m.geoms, key=lambda g: g.area, reverse=True)
                zones[z] = comp[0]
                orphans += [f for f in comp[1:] if f.area > 0]
            elif not m.is_empty:
                zones[z] = m
        except Exception:
            continue
    for frag in orphans:
        best = min(zones, key=lambda z: zones[z].distance(frag.centroid))
        try:
            zones[best] = unary_union([zones[best], frag])
        except Exception:
            pass
    for _ in range(5):
        try:
            gap = bbox.difference(unary_union(list(zones.values())))
            if gap.is_empty or gap.area < 1e-10:
                break
            pieces = ([g for g in gap.geoms if isinstance(g, Polygon) and g.area > 0]
                      if isinstance(gap, (MultiPolygon, GeometryCollection))
                      else ([gap] if isinstance(gap, Polygon) and gap.area > 0 else []))
            if not pieces:
                break
            for piece in pieces:
                best = min(zones, key=lambda z: zones[z].distance(piece.centroid))
                zones[best] = unary_union([zones[best], piece])
        except Exception:
            break
    return zones


def draw_poly(ax, geom, fc, ec, lw, alpha=FILL_ALPHA):
    if geom.is_empty:
        return
    if isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            draw_poly(ax, g, fc, ec, lw, alpha)
        return
    r, g_, b = (int(fc[1:3], 16) / 255, int(fc[3:5], 16) / 255, int(fc[5:7], 16) / 255)
    xs, ys = geom.exterior.xy
    ax.add_patch(plt.Polygon(list(zip(xs, ys)), closed=True, facecolor=(r, g_, b, alpha),
                             edgecolor=ec, linewidth=lw, zorder=1))


def draw_net(ax, key, cfg, panel):
    d = os.path.join(ROOT, "data", key)
    inp_path = os.path.join(d, NETS[key]["inp"])
    if not os.path.exists(inp_path):
        # public checkout without the on-request utility model: placeholder panel
        ax.axis("off")
        ax.text(0.5, 0.5, NETS[key]["title"].split(" (")[0] + " network model\n"
                "available on request\n(see Data availability)", transform=ax.transAxes,
                ha="center", va="center", fontsize=7.5, color="#777777",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#F4F4F4", edgecolor="#BBBBBB"))
        ax.text(0.0, 1.02, f"{panel}  {NETS[key]['title']}, K = {NETS[key]['K']}",
                transform=ax.transAxes, fontsize=7.2, fontweight="bold",
                ha="left", va="bottom")
        print(f"  [skip] {key}: {inp_path} not present, placeholder panel drawn")
        return
    nodes, pipes = parse_inp(inp_path)
    parts = pickle.load(open(os.path.join(d, "partitions.pkl"), "rb"))
    n2c = {str(n): int(c) for n, c in parts[NETS[key]["K"]]["node_to_community"].items()}
    sensors = [str(s) for s in json.load(open(os.path.join(d, "sensor_fingerprints.json"),
                                              encoding="utf-8"))["sensor_nodes"]]
    pos = {n: (v["x"], v["y"]) for n, v in nodes.items() if "x" in v}
    zones = voronoi_zones(pos, {n: c for n, c in n2c.items() if n in pos})
    for z in sorted(zones):
        draw_poly(ax, zones[z], ZONE_FILLS[z % len(ZONE_FILLS)],
                  ZONE_BORDERS[z % len(ZONE_BORDERS)], cfg["zone_lw"])
    for a, b in pipes:
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], "-",
                    color="#555555", lw=cfg["pipe_lw"], alpha=0.30, zorder=2)
    ss = set(sensors)
    for n, (x, y) in pos.items():
        if n in ss:
            continue
        if nodes[n]["type"] in ("reservoir", "tank"):
            ax.plot(x, y, "s", color="#333333", ms=cfg["res_ms"],
                    markeredgecolor="black", markeredgewidth=0.4, zorder=4)
        else:
            ax.plot(x, y, "o", color="#555555", ms=cfg["node_ms"],
                    markeredgecolor="none", alpha=0.6, zorder=3)
    for s in sensors:
        if s in pos:
            ax.plot(pos[s][0], pos[s][1], "o", color=fs.RED, ms=cfg["sensor_ms"],
                    markeredgecolor="black", markeredgewidth=0.5, zorder=6)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    tot = (max(xs) - min(xs)) * (max(ys) - min(ys))
    for z in sorted(zones):
        g = zones[z]
        if g.area / tot > 0.001:
            ax.text(g.centroid.x, g.centroid.y, str(z + 1), ha="center", va="center",
                    fontsize=cfg["label_fs"], fontweight="bold", color="black", zorder=7,
                    alpha=0.9, bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                         edgecolor="none", alpha=0.5))
    ax.set_aspect("auto")
    ax.axis("off")
    mx, my = (max(xs) - min(xs)) * 0.07, (max(ys) - min(ys)) * 0.07
    ax.set_xlim(min(xs) - mx, max(xs) + mx)
    ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.text(0.0, 1.02, f"{panel}  {NETS[key]['title']}, K = {NETS[key]['K']}, "
            f"{len(sensors)} sensors", transform=ax.transAxes, fontsize=7.2,
            fontweight="bold", ha="left", va="bottom")


LEGEND = [Line2D([0], [0], color="#555555", lw=1.2, label="Pipe"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555", ms=4,
                 label="Junction"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor=fs.RED, ms=7,
                 markeredgecolor="black", markeredgewidth=0.5, label="Pressure sensor"),
          Line2D([0], [0], marker="s", color="w", markerfacecolor="#333333", ms=5,
                 markeredgecolor="black", markeredgewidth=0.3, label="Reservoir / Tank")]

# ---- main-text figure: EXA7 + City D ----
fig, axes = plt.subplots(1, 2, figsize=(7.09, 4.0),
                         gridspec_kw=dict(width_ratios=[1, 1.9]))
draw_net(axes[0], "exa7", dict(sensor_ms=3.4, node_ms=0.6, res_ms=3.0, label_fs=6.4,
                               zone_lw=0.7, pipe_lw=0.3), "a")
draw_net(axes[1], "city_d", dict(sensor_ms=3.2, node_ms=0.5, res_ms=3.0, label_fs=6.4,
                                 zone_lw=0.7, pipe_lw=0.3), "b")
fig.legend(handles=LEGEND, loc="lower center", ncol=4, fontsize=6.4, frameon=True,
           fancybox=True, framealpha=0.9)
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.09, top=0.925, wspace=0.06)
fs.save(fig, "Fig5_partitions_main")
print("saved Fig5_partitions_main")

# ---- SI figure: L-Town + City H + KY4 ----
fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.9))
for ax, key, panel in zip(axes, ("ltown", "city_h", "ky4"), "abc"):
    draw_net(ax, key, dict(sensor_ms=2.4, node_ms=0.4, res_ms=2.4, label_fs=4.8,
                           zone_lw=0.55, pipe_lw=0.2), panel)
fig.legend(handles=LEGEND, loc="lower center", ncol=4, fontsize=6.2, frameon=True,
           fancybox=True, framealpha=0.9)
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.13, top=0.90, wspace=0.05)
fs.save(fig, "FigS3_partitions")
print("saved FigS3_partitions")
