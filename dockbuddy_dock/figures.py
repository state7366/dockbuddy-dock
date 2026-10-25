# -*- coding: utf-8 -*-
"""libdock.figures —— figure-ready 可视化(零依赖自包含 SVG, B类 #313)。

================================================================================
设计约束(最高优先级):
- **零依赖**: 只产出标准 SVG 字符串, 不 import matplotlib / 任何绘图库(本环境无
  matplotlib)。SVG 自包含、可在浏览器/论文直接嵌入。
- **诚实可视化**: 图元只反映喂入的数值; 误差棒/置信带/警告框如实呈现, 不美化成
  "确定结论"。涉及伪重复风险(ensemble)时显式画红框警示, 不掩盖。
- 所有文本经 XML 转义, 防止 `& < >` 破坏 SVG 良构。
================================================================================
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET


def _esc(s: Any) -> str:
    """XML 文本转义。"""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _color_ramp(t: float) -> str:
    """t in [0,1] → 蓝(低)→黄→红(高) 简单三段插值, 返回 #rrggbb。"""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        # 蓝(40,80,200) → 黄(240,220,60)
        u = t / 0.5
        r = int(40 + (240 - 40) * u)
        g = int(80 + (220 - 80) * u)
        b = int(200 + (60 - 200) * u)
    else:
        # 黄(240,220,60) → 红(200,40,40)
        u = (t - 0.5) / 0.5
        r = int(240 + (200 - 240) * u)
        g = int(220 + (40 - 220) * u)
        b = int(60 + (40 - 60) * u)
    return f"#{r:02x}{g:02x}{b:02x}"


def _wrap_text(text: str, width: int = 60) -> List[str]:
    """中文/英文混排按字符数粗略断行(用于长注记)。"""
    out: List[str] = []
    cur = ""
    for ch in text:
        cur += ch
        if len(cur) >= width and ch in ("。", ";", "；", " ", "，"):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out or [""]


def ensemble_aggregation_svg(stats: Dict[str, Any], *,
                             width: int = 780, height: int = 460,
                             title: str = "Ensemble 跨模型能量聚合") -> str:
    """把 aggregate() 的 stats 画成跨模型能量均值±std 柱状图(含 overall 参考线
    与伪重复风险红框警示)。

    图元:
      - 每个 AF 模型一根柱: 高度=energy_mean, 误差棒=±energy_std(无 std 则只画点)。
      - 水平虚线 = overall.energy_mean(全体 run 池均值)。
      - 若 pseudo_replication.degenerate_models 为真 → 顶部红框 + 截断注记。
    """
    per = (stats.get("per_model") or {})
    overall = stats.get("overall") or {}
    pseudo = stats.get("pseudo_replication") or {}

    models = []
    for k in sorted(per):
        b = per[k]
        models.append((k, b.get("energy_mean"), b.get("energy_std"),
                       b.get("n_runs")))

    # y 轴范围
    vals: List[float] = []
    for _, m, s, _ in models:
        if m is not None:
            vals.append(m - (s or 0))
            vals.append(m + (s or 0))
    if overall.get("energy_mean") is not None:
        vals.append(overall["energy_mean"])
    if not vals:
        vals = [0.0, 1.0]
    ylo, yhi = min(vals), max(vals)
    pad = (yhi - ylo) * 0.12 or 1.0
    ylo, yhi = ylo - pad, yhi + pad

    left, right = 80, width - 30
    top, bottom = 70, height - 110
    plot_w = right - left
    plot_h = bottom - top

    def y2px(v: float) -> float:
        return bottom - (v - ylo) / (yhi - ylo) * plot_h

    n = max(len(models), 1)
    slot = plot_w / n
    bw = min(slot * 0.55, 70)

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2:.0f}" y="28" font-size="16" text-anchor="middle" '
        f'fill="#222">{_esc(title)}</text>',
    ]

    # y 轴
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#444"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#444"/>')
    for f in range(0, 5):
        v = ylo + (yhi - ylo) * f / 4
        y = y2px(v)
        parts.append(f'<line x1="{left-4}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                     f'stroke="#e0e0e0"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" font-size="10" '
                     f'text-anchor="end" fill="#666">{v:.2f}</text>')
    parts.append(f'<text x="14" y="{(top+bottom)/2:.0f}" font-size="11" '
                 f'text-anchor="middle" fill="#444" transform="rotate(-90 14 '
                 f'{(top+bottom)/2:.0f})">能量 kcal/mol (越低越好)</text>')

    # overall 参考线
    if overall.get("energy_mean") is not None:
        oy = y2px(overall["energy_mean"])
        parts.append(f'<line x1="{left}" y1="{oy:.1f}" x2="{right}" y2="{oy:.1f}" '
                     f'stroke="#d9534f" stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{right-4}" y="{oy-5:.1f}" font-size="10" '
                     f'text-anchor="end" fill="#d9534f">overall mean '
                     f'{overall["energy_mean"]:.2f} (SEM '
                     f'{overall.get("energy_sem_over_models")})</text>')

    # 柱
    for idx, (k, m, s, nr) in enumerate(models):
        cx = left + slot * (idx + 0.5)
        if m is None:
            parts.append(f'<text x="{cx:.1f}" y="{(top+bottom)/2:.0f}" font-size="10" '
                         f'text-anchor="middle" fill="#999">无数据</text>')
        else:
            y = y2px(m)
            bh = bottom - y
            parts.append(f'<rect x="{cx-bw/2:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                         f'height="{bh:.1f}" fill="#4a90d9"/>')
            if s is not None and s > 0:
                yb = y2px(m + s)
                yt = y2px(m - s)
                parts.append(f'<line x1="{cx:.1f}" y1="{yt:.1f}" x2="{cx:.1f}" '
                             f'y2="{yb:.1f}" stroke="#222" stroke-width="1.2"/>')
                parts.append(f'<line x1="{cx-5:.1f}" y1="{yt:.1f}" x2="{cx+5:.1f}" '
                             f'y2="{yt:.1f}" stroke="#222" stroke-width="1.2"/>')
                parts.append(f'<line x1="{cx-5:.1f}" y1="{yb:.1f}" x2="{cx+5:.1f}" '
                             f'y2="{yb:.1f}" stroke="#222" stroke-width="1.2"/>')
            parts.append(f'<text x="{cx:.1f}" y="{y-6:.1f}" font-size="10" '
                         f'text-anchor="middle" fill="#222">{m:.2f}</text>')
        # x 标签(模型号, 旋转)
        parts.append(f'<text x="{cx:.1f}" y="{bottom+16:.1f}" font-size="10" '
                     f'text-anchor="middle" fill="#444">{_esc(k)}</text>')
        if nr:
            parts.append(f'<text x="{cx:.1f}" y="{bottom+30:.1f}" font-size="9" '
                         f'text-anchor="middle" fill="#888">{nr} runs</text>')

    # 伪重复风险红框
    if pseudo.get("degenerate_models"):
        parts.append(f'<rect x="{left}" y="{top-44:.1f}" width="{plot_w:.1f}" '
                     f'height="34" fill="#fdecec" stroke="#d9534f"/>')
        note = _wrap_text("⚠ 模型退化(跨模型 Cα RMSD 极小): "
                          + str(pseudo.get("note", ""))[:120], width=78)
        for i, line in enumerate(note[:2]):
            parts.append(f'<text x="{left+6}" y="{top-30+i*13:.1f}" font-size="9" '
                         f'fill="#a94442">{_esc(line)}</text>')

    # 脚注
    foot = (f"n_models={overall.get('n_models')}, n_runs={overall.get('n_runs')}, "
            f"独立单元={pseudo.get('n_independent_units')} (跨模型 SEM 为准)")
    parts.append(f'<text x="{left}" y="{height-18:.1f}" font-size="10" '
                 f'fill="#666">{_esc(foot)}</text>')

    parts.append('</svg>')
    return "".join(parts)


def dca_heatmap_svg(pairs: List[Dict[str, Any]], n_positions: int, *,
                    use_apc: bool = False, cell: int = 22,
                    title: str = "DCA 共进化耦合热图") -> str:
    """把 dca_pairs() 的 top 对画成 L×L 对称热图(仅 top 对上色, 其余灰)。

    参数:
      pairs          dca_pairs() 返回的 pairs(含 i,j 与 mi/apc_mi)
      n_positions    L(比对列数, 1-based 残基号上限)
      use_apc        True 用 apc_mi 上色, False 用原始 mi
      cell           每格像素
    """
    L = n_positions
    if L <= 0:
        return _empty_svg(title, "n_positions 无效")
    key = "apc_mi" if use_apc else "mi"
    grid = {}
    mx = 0.0
    for p in pairs:
        i = p.get("i")
        j = p.get("j")
        v = p.get(key)
        if i and j and v is not None:
            grid[(i, j)] = float(v)
            grid[(j, i)] = float(v)
            mx = max(mx, abs(float(v)))
    mx = mx or 1.0

    pad_l, pad_t = 40, 40
    w = pad_l + L * cell + 20
    h = pad_t + L * cell + 20

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="DejaVu Sans, Arial, sans-serif">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<text x="{w/2:.0f}" y="20" font-size="14" text-anchor="middle" '
        f'fill="#222">{_esc(title)}{"(APC 校正)" if use_apc else "(原始 MI)"}</text>',
    ]
    # 轴标签(每隔 max(1, L//10) 标一次)
    step = max(1, L // 10)
    for r in range(L):
        y = pad_t + r * cell + cell / 2
        if (r + 1) % step == 0 or r == 0:
            parts.append(f'<text x="{pad_l-6:.0f}" y="{y+4:.0f}" font-size="9" '
                         f'text-anchor="end" fill="#666">{r+1}</text>')
        x = pad_l + r * cell + cell / 2
        if (r + 1) % step == 0 or r == 0:
            parts.append(f'<text x="{x:.0f}" y="{pad_t-6:.0f}" font-size="9" '
                         f'text-anchor="middle" fill="#666">{r+1}</text>')
    for r in range(L):
        for c in range(L):
            x = pad_l + c * cell
            y = pad_t + r * cell
            if r == c:
                parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cell}" '
                             f'height="{cell}" fill="#f2f2f2"/>')
                continue
            v = grid.get((r + 1, c + 1))
            if v is None:
                fill = "#ededed"
            else:
                fill = _color_ramp(abs(v) / mx)
            parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cell}" '
                         f'height="{cell}" fill="{fill}" stroke="#ffffff" '
                         f'stroke-width="0.3"/>')
    # 色标
    parts.append(f'<text x="{pad_l:.0f}" y="{h-4:.0f}" font-size="9" '
                 f'fill="#666">0</text>')
    parts.append(f'<text x="{pad_l+L*cell-30:.0f}" y="{h-4:.0f}" font-size="9" '
                 f'text-anchor="end" fill="#666">|{key}| max={mx:.2f}</text>')
    parts.append('</svg>')
    return "".join(parts)


def ranking_scatter_svg(ranked: List[Dict[str, Any]], x_name: str, y_name: str, *,
                        width: int = 640, height: int = 480,
                        title: str = "突变候选多目标散点") -> str:
    """把排名结果画成两目标归一化散点(颜色按 rank: 绿=前排, 红=末位)。

    点坐标取 ranked[i]["components"][x_name / y_name](均为 min-max 归一化,
    越大越优); 任一为 None 的点跳过并在脚注提示。
    """
    left, right = 70, width - 30
    top, bottom = 60, height - 70
    pw, ph = right - left, bottom - top

    pts: List[Tuple[float, float, str, int]] = []
    skipped = 0
    n = max(len(ranked), 1)
    for r in ranked:
        comp = r.get("components") or {}
        xv = comp.get(x_name)
        yv = comp.get(y_name)
        if xv is None or yv is None:
            skipped += 1
            continue
        pts.append((float(xv), float(yv), str(r.get("id", "?")), int(r.get("rank", 0))))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2:.0f}" y="24" font-size="15" text-anchor="middle" '
        f'fill="#222">{_esc(title)}</text>',
        f'<text x="{width/2:.0f}" y="42" font-size="10" fill="#666">'
        f'{_esc(x_name)} → (越大越优) ； {_esc(y_name)} ↑ (越大越优)</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#444"/>',
        f'<text x="{(left+right)/2:.0f}" y="{height-50:.0f}" font-size="11" '
        f'text-anchor="middle" fill="#444">{_esc(x_name)}</text>',
        f'<text x="16" y="{(top+bottom)/2:.0f}" font-size="11" '
        f'text-anchor="middle" fill="#444" transform="rotate(-90 16 '
        f'{(top+bottom)/2:.0f})">{_esc(y_name)}</text>',
    ]

    if not pts:
        parts.append(f'<text x="{width/2:.0f}" y="{height/2:.0f}" font-size="12" '
                     f'text-anchor="middle" fill="#999">无可用数据点</text>')
    else:
        for x, y, label, rk in pts:
            px = left + x * pw
            py = bottom - y * ph
            # 颜色: rank 1 → 绿(46,160,67), rank n → 红(200,60,60)
            frac = 0.0 if n == 1 else (rk - 1) / (n - 1)
            col = _color_ramp(frac)
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" '
                         f'fill="{col}" fill-opacity="0.8" stroke="#333" '
                         f'stroke-width="0.6"/>')
            parts.append(f'<text x="{px+10:.1f}" y="{py+4:.1f}" font-size="10" '
                         f'fill="#222">#{rk} {_esc(label)}</text>')

    if skipped:
        parts.append(f'<text x="{left}" y="{height-24:.0f}" font-size="9" '
                     f'fill="#a94442">⚠ {skipped} 个候选因缺失 '
                     f'{_esc(x_name)}/{_esc(y_name)} 未绘</text>')
    parts.append(f'<text x="{left}" y="{height-8:.0f}" font-size="9" '
                 f'fill="#888">颜色: 绿=排名前, 红=排名末; 仅优先级提示, 非活性预测</text>')
    parts.append('</svg>')
    return "".join(parts)


def _empty_svg(title: str, msg: str) -> str:
    return ("<svg xmlns='http://www.w3.org/2000/svg' width='320' height='160'>"
            f"<rect width='320' height='160' fill='#fff'/>"
            f"<text x='160' y='30' font-size='13' text-anchor='middle' "
            f"fill='#222'>{_esc(title)}</text>"
            f"<text x='160' y='80' font-size='11' text-anchor='middle' "
            f"fill='#a94442'>{_esc(msg)}</text></svg>")


def write_figure(path: str, svg: str) -> str:
    """把 SVG 字符串落盘; 返回路径。"""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return path


def _is_well_formed(svg: str) -> bool:
    """用标准 XML 解析校验 SVG 良构(测试/运行自检用)。"""
    try:
        ET.fromstring(svg)
        return True
    except ET.ParseError:
        return False


__all__ = ["ensemble_aggregation_svg", "dca_heatmap_svg", "ranking_scatter_svg",
           "write_figure", "_is_well_formed"]
