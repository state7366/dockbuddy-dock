# -*- coding: utf-8 -*-
"""run_report.py —— 命令落盘 JSON 后自动生成人类可读字段报告。

===================================================================================
定位(2026-10-06 审计新增功能)
===================================================================================
`dock` / `mutate` / `pocket-tunnel` / `pocket-decomp` / `deacylation` / `msa-report`
/ `product-egress` / `compare` 每条命令在写完自己的 JSON 后, 调
`write_run_report(json_path, payload, doc_type)` 在同目录生成 `<stem>.report.md`:

  - 头部: 来源 JSON / 命令行 / 生成时间 / 字段覆盖率;
  - 关键指标表(该文档类型的少量核心字段, 快速判读用);
  - 全量字段表: 扁平化路径 + 值 + **详细解释**(解释来自 field_docs.FIELDS,
    单一事实源 —— 代码里改字段含义时同步改 field_docs 即可);
  - 脚注: 未登记字段清单(字段覆盖审计, 新代码加字段忘登记时顶出)。

设计约束:
  - **永不阻断主流程**: 任何异常只 print 告警, 报告失败不影响 JSON 产物;
  - 纯 stdlib, 与 field_docs 同款「不进 libdock/__init__ 再导出」约定;
  - 数组折叠: dict 数组只展开首元素字段(路径 `key[i].sub`), 概览行标注项数;
  - 环境开关: DOCK_RUN_REPORT=0 关闭(批量重跑想省 IO 时)。

调用点(五处, 均已接入):
  cli/dock.py run_pipeline(6.97 之后, docking_summary + run_meta 各一份)
  pipeline/mutate.py _mutate_one(.mutate.json 落盘后)
  cli/pocket_tunnel_cmd.py(tunnel.json 落盘后)
  cli/analysis_cmd.py _dump_json(四个分析命令 + compare 分项共用一个出口)
  cli/analysis_cmd.py run_compare_suite(compare_summary.json)
===================================================================================
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from . import field_docs  # 相对导入: 同包内纯 stdlib 模块


# ----------------------------------------------------------------------------
# 值格式化
# ----------------------------------------------------------------------------
def _fmt_value(v: Any, limit: int = 64) -> str:
    """把扁平化后的叶子值转成表格单元格文本(截断到 limit 字符)。"""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        s = f"{v:.4g}"
        return s
    if isinstance(v, (list, tuple)):
        if not v:
            return "[]"
        joined = ", ".join(_fmt_value(x, 24) for x in v[:6])
        more = f" …(共{len(v)}项)" if len(v) > 6 else ""
        return (joined + more)[:limit]
    if isinstance(v, dict):
        s = json.dumps(v, ensure_ascii=False)
        return (s[:limit] + "…") if len(s) > limit else s
    s = str(v)
    return (s[:limit] + "…") if len(s) > limit else s


def _lookup(payload: dict, dotted: str) -> Any:
    """按点路径取值(支持 [i] 模板段取首元素); 取不到返回 None。"""
    cur: Any = payload
    for seg in dotted.split("."):
        m = seg.endswith("]") and "[" in seg
        key = seg[:seg.index("[")] if m else seg
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
        if m and isinstance(cur, list):
            cur = cur[0] if cur else None
            if cur is None:
                return None
    return cur


def _cell(s: str) -> str:
    """表格单元格转义: 竖线(如 |Δ|≥0.5)与换行会破坏 markdown 表格。"""
    return s.replace("|", "\\|").replace("\n", " ")


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------
def write_run_report(json_path: str, payload: dict, doc_type: str,
                     command_line: str = "") -> str:
    """为一份已落盘的 JSON 生成 `<stem>.report.md`; 返回报告路径(失败/关闭返回 "")。

    json_path  : JSON 文件路径(报告写在其旁, 同名换 .report.md 后缀)
    payload    : 已写出的 payload(内存中的同一对象, 免二次读盘)
    doc_type   : field_docs.FIELD_DOCS 的键(如 "docking_summary")
    command_line: 可选命令行留痕(缺省取 sys.argv)
    """
    if os.environ.get("DOCK_RUN_REPORT", "1") == "0":
        return ""
    spec = field_docs.get_spec(doc_type)
    try:
        if spec is None:
            print(f"[run_report] 未知文档类型 {doc_type!r}, 跳过(已知: "
                  f"{', '.join(sorted(field_docs.FIELD_DOCS))})")
            return ""
        fields_doc = spec.get("fields", {})
        report_path = os.path.splitext(json_path)[0] + ".report.md"

        # 头部
        lines = [f"# {spec.get('title', doc_type)} —— 字段报告", ""]
        lines.append(f"- 来源 JSON: `{os.path.basename(json_path)}`")
        lines.append(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if not command_line:
            command_line = " ".join(sys.argv)
        if command_line:
            lines.append(f"- 命令行: `{command_line[:200]}`")

        # 覆盖率(先算, 头部引用)
        cov = field_docs.coverage(doc_type, payload)
        lines.append(f"- 字段覆盖: {cov['n_documented']}/{cov['n_total']} 已登记"
                     + ("" if not cov["undocumented"]
                        else f"(⚠ 未登记: {', '.join(cov['undocumented'][:8])}"
                             + ("…" if len(cov["undocumented"]) > 8 else "") + ")"))
        lines.append("")

        # 关键指标
        key_metrics = spec.get("key_metrics") or []
        if key_metrics:
            lines.append("## 关键指标")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|---|---|")
            for m in key_metrics:
                label = m.rsplit(".", 1)[-1] if "." in m else m
                val = _lookup(payload, m)
                lines.append(f"| {label} | {_cell(_fmt_value(val, 48))} |")
            lines.append("")

        # 全量字段表
        lines.append("## 全量字段(含解释)")
        lines.append("")
        lines.append("| 字段 | 值 | 解释 |")
        lines.append("|---|---|---|")
        _dynamic = tuple(spec.get("dynamic", ()))
        for path, val in field_docs.iter_flat_fields(payload, dynamic=_dynamic):
            # 文档匹配(与 coverage 同款通配), 拿解释
            doc = ""
            for cand in field_docs._coverage_candidates(path):
                if cand in fields_doc:
                    doc = fields_doc[cand]
                    break
            if not doc:
                # 容器行(其子字段已登记)标注为结构行; 真未登记标 ⚠
                doc = "(结构行, 子字段见下)" if path.endswith("[i]") else "⚠ 未登记"
            lines.append(f"| {_cell(f'`{path}`')} | {_cell(_fmt_value(val))} | {_cell(doc)} |")
        lines.append("")

        # 脚注: 未登记字段
        if cov["undocumented"]:
            lines.append("## ⚠ 未登记字段(需在 field_docs.py 补解释)")
            lines.append("")
            for p in cov["undocumented"]:
                lines.append(f"- `{p}`")
            lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return report_path
    except Exception as e:  # noqa: BLE001 —— 报告失败绝不阻断主流程
        print(f"[run_report] 报告生成失败(不影响 JSON 产物): {type(e).__name__}: {e}")
        return ""


def attach_run_report(json_path: str, payload: dict, doc_type: str) -> None:
    """write_run_report 的静默包装(调用点一行接入; 已含异常吞噬)。"""
    write_run_report(json_path, payload, doc_type)


__all__ = ["write_run_report", "attach_run_report"]
