#!/usr/bin/env python3
"""
RCDT 全脑动力学拓扑分析系统 V1.1
Author: Haolong Wang | 开发完成日期：2026年2月

统一入口：调度 TDA 校验（figure1）与全脑仿真/拓扑分析（figure2）。
运行方式：
  python main.py figure1        # 生成 Fig1 TDA 校验图
  python main.py figure2        # 生成 Fig2/Fig3 主图与持久熵
  python main.py figure2 --quick
  python main.py figure2 --shuffled
  python main.py figure2 --sweep
"""

import argparse
import os
import subprocess
import sys

SOFTWARE_NAME = "RCDT 全脑动力学拓扑分析系统"
VERSION = "V1.1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_module(module_name, extra_args=None):
    """在项目根目录下以子进程运行指定模块，保持工作目录与当前环境一致。"""
    module_path = os.path.join(SCRIPT_DIR, module_name)
    if not os.path.isfile(module_path):
        print(f"错误: 未找到模块文件 {module_path}", file=sys.stderr)
        sys.exit(1)
    cmd = [sys.executable, module_path]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    sys.exit(result.returncode)


def cmd_figure1(args):
    """执行 Figure 1：TDA 管道校验（Van der Pol / Lorenz）。优化: 支持 --seed 复现。"""
    extra = []
    if getattr(args, "seed", None) is not None:
        extra.extend(["--seed", str(args.seed)])
    _run_module("figure1_persistence_diagram.py", extra if extra else None)


def cmd_figure2(args):
    """执行 Figure 2/3：全脑仿真与持久熵分析。优化: 支持 --seed, --surrogate, --n-shuffles, --n-seeds-sweep。"""
    extra = []
    if getattr(args, "quick", False):
        extra.append("--quick")
    if getattr(args, "shuffled", False):
        extra.append("--shuffled")
    if getattr(args, "sweep", False):
        extra.append("--sweep")
    if getattr(args, "surrogate", False):
        extra.append("--surrogate")
    if getattr(args, "seed", None) is not None:
        extra.extend(["--seed", str(args.seed)])
    if getattr(args, "n_shuffles", 1) != 1:
        extra.extend(["--n-shuffles", str(args.n_shuffles)])
    if getattr(args, "n_seeds_sweep", 1) != 1:
        extra.extend(["--n-seeds-sweep", str(args.n_seeds_sweep)])
    _run_module("figure2_simulation.py", extra if extra else None)


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=f"{SOFTWARE_NAME} {VERSION} — 统一运行入口",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # figure1
    p1 = subparsers.add_parser("figure1", help="生成 Figure 1：TDA 校验（极限环 vs 混沌）")
    p1.add_argument("--seed", type=int, default=42, help="随机种子（复现论文图）")
    p1.set_defaults(func=cmd_figure1)

    # figure2
    p2 = subparsers.add_parser("figure2", help="生成 Figure 2/3：全脑仿真与持久熵")
    p2.add_argument("--quick", action="store_true", help="快速模式（缩短仿真时间）")
    p2.add_argument("--shuffled", action="store_true", help="同时运行受体洗牌对照")
    p2.add_argument("--sweep", action="store_true", help="分岔参数扫描（k vs 持久熵）")
    p2.add_argument("--surrogate", action="store_true", help="替代数据检验（相位随机化）")
    p2.add_argument("--n-shuffles", type=int, default=1, help="洗牌重复次数（>1 得 PE 分布）")
    p2.add_argument("--n-seeds-sweep", type=int, default=1, help="分岔扫描 seed 数（>1 报告 k_crit 稳定性）")
    p2.add_argument("--seed", type=int, default=42, help="主随机种子")
    p2.set_defaults(func=cmd_figure2)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        print("\n示例: python main.py figure1  或  python main.py figure2 --quick")
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
