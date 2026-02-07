#!/usr/bin/env python3
"""
将源程序文档导出为 PDF（软著提交用）
RCDT 全脑动力学拓扑分析系统 V1.1 | Author: Haolong Wang
"""

import os
import sys

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(base, "source_code_for_copyright.txt")
    pdf_path = os.path.join(base, "source_code_for_copyright.pdf")

    if not os.path.exists(txt_path):
        print(f"错误: 未找到 {txt_path}")
        sys.exit(1)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Preformatted, Spacer, PageBreak
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        print("请先安装: pip install reportlab")
        sys.exit(1)

    font_name = "Courier"
    for font_path in [
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("CustomFont", font_path))
                font_name = "CustomFont"
                break
            except Exception:
                pass

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=30,
        bottomMargin=30,
    )

    style = ParagraphStyle(
        name="Code",
        fontName=font_name,
        fontSize=8,
        leading=10,
        leftIndent=0,
    )

    flow = []
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    for i, line in enumerate(content.split("\n")):
        line = line.replace("\r", "").replace("<", "&lt;").replace(">", "&gt;")
        if len(line) > 130:
            line = line[:127] + "..."
        try:
            flow.append(Preformatted(line, style))
        except Exception:
            flow.append(Preformatted(line.encode("ascii", "replace").decode(), style))

    doc.build(flow)
    print(f"已生成: {pdf_path}")


if __name__ == "__main__":
    main()
