#!/usr/bin/env python3
"""
RCDT 全脑动力学拓扑分析系统 V1.0
Author: Haolong Wang | 开发完成日期：2026年2月

生成软著登记用源程序文档

输出：docs/source_code_for_copyright.txt（含页眉，每页约50行）
用于中国版权保护中心软件著作权登记 - 源程序材料
"""

import os

LINES_PER_PAGE = 50
HEADER = "RCDT 全脑动力学拓扑分析系统 V1.1 | Author: Haolong Wang"
# RCDT 完整软件：统一入口 + 参数/TDA 公共模块 + 两大核心脚本
SOURCE_FILES = [
    "main.py",
    "rcdt_params.py",                 # 优化: 参数集中与 [D]_crit 定量
    "rcdt_tda.py",                    # 优化: 公共 TDA 管道与 τ 选取、替代数据
    "figure1_persistence_diagram.py",
    "figure2_simulation.py",
]


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "source_code_for_copyright.txt")

    lines = []
    for fname in SOURCE_FILES:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            lines.append(f"\n{'='*60}\n# 文件: {fname}\n{'='*60}\n")
            lines.extend(content.splitlines())

    full_text = "\n".join(lines)
    all_lines = full_text.splitlines()
    total_lines = len(all_lines)
    total_pages = (total_lines + LINES_PER_PAGE - 1) // LINES_PER_PAGE

    # 按页输出，每页加页眉
    output = []
    for p in range(total_pages):
        start = p * LINES_PER_PAGE
        end = min((p + 1) * LINES_PER_PAGE, total_lines)
        page_lines = all_lines[start:end]
        header = f"\n----- 第 {p+1} 页 / 共 {total_pages} 页 | {HEADER} -----\n"
        output.append(header + "\n".join(page_lines))

    result = "\n".join(output)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"软件著作权登记 - 源程序文档\n{HEADER}\n")
        f.write(f"总行数: {total_lines} | 总页数: {total_pages} (每页约{LINES_PER_PAGE}行)\n")
        f.write("=" * 60 + "\n")
        f.write(result)

    print(f"已生成: {out_path}")
    print(f"总行数: {total_lines}, 总页数: {total_pages}")

    # 同时生成 HTML 版，便于浏览器打印为 PDF
    html_path = os.path.join(out_dir, "source_code_for_copyright.html")
    _write_html(html_path, output, total_pages, HEADER, LINES_PER_PAGE)
    print(f"已生成(HTML): {html_path}")
    print("提示: 用浏览器打开 HTML，Ctrl+P 打印为 PDF 即可提交。")

    if total_pages <= 60:
        print("说明: 总页数≤60，登记时通常提交全部源程序即可。")
    else:
        print("说明: 总页数>60，登记时提交前30页+后30页。")


def _write_html(out_path, output, total_pages, header, lines_per_page):
    """生成 HTML，含页眉与分页，便于打印为 PDF。"""
    import html
    body_parts = []
    for i, block in enumerate(output):
        escaped = html.escape(block).replace("\n", "<br>\n")
        body_parts.append(
            f'<div class="page">'
            f'<div class="page-header">{header} | 第 {i+1} 页 / 共 {total_pages} 页</div>'
            f'<pre class="code">{escaped}</pre>'
            f'</div>'
        )
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>源程序文档 - RCDT 全脑动力学拓扑分析系统</title>
<style>
body {{ font-family: Consolas, "Courier New", monospace; font-size: 11px; line-height: 1.3; margin: 0; }}
.page {{ page-break-after: always; padding: 15px; }}
.page:last-child {{ page-break-after: auto; }}
.page-header {{ font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
pre {{ margin: 0; white-space: pre-wrap; word-wrap: break-word; }}
@media print {{ .page {{ page-break-after: always; }} .page:last-child {{ page-break-after: auto; }} }}
</style>
</head>
<body>
{chr(10).join(body_parts)}
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    main()
