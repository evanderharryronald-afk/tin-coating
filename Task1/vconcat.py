"""
vconcat.py
----------
将多个 xlsx / csv 文件纵向拼接成一张大表。

用法：
    python vconcat.py                    # 使用同目录下的 vconcat.yaml
    python vconcat.py --config my.yaml   # 指定配置文件
"""

import sys
import time
import argparse
from pathlib import Path

import pandas as pd
import yaml

def log(msg):
    print(msg, flush=True)


def die(msg):
    print(f"❌ ERROR: {msg}", flush=True)
    sys.exit(1)


def load_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(str(path), header=0)
    elif suffix == '.csv':
        try:
            df = pd.read_csv(str(path), header=0, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(str(path), header=0, encoding='gbk')
    else:
        die(f"不支持的文件格式: {path.suffix}，仅支持 xlsx/xls/csv。")
    return df


def main():
    parser = argparse.ArgumentParser(description="多表纵向拼接工具")
    parser.add_argument('--config', default=None, help="配置文件路径（默认：同目录下的 vconcat.yaml）")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else Path(__file__).parent / "vconcat.yaml"
    if not cfg_path.exists():
        die(f"配置文件不存在: {cfg_path}")

    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    base_dir = cfg_path.parent
    t_start = time.time()

    # ── 收集所有输入文件 ──────────────────────────────────────────
    inputs = cfg.get('inputs', [])
    if not inputs:
        die("配置文件中 inputs 为空，请至少指定一个输入文件或目录。")

    file_paths = []
    for item in inputs:
        p = Path(item) if Path(item).is_absolute() else base_dir / item
        if p.is_dir():
            # 目录：收集其下所有 xlsx/csv
            found = sorted([
                f for f in p.iterdir()
                if f.suffix.lower() in ('.xlsx', '.xls', '.csv')
            ])
            if not found:
                log(f"⚠️  目录 {p} 下未找到任何 xlsx/csv 文件，跳过。")
            file_paths.extend(found)
        elif p.is_file():
            file_paths.append(p)
        else:
            die(f"路径不存在: {p}")

    if not file_paths:
        die("未找到任何可拼接的文件。")

    log(f"共找到 {len(file_paths)} 个文件待拼接。")

    # ── 逐个读取 ─────────────────────────────────────────────────
    dfs = []
    col_ref = None  # 以第一个文件的列名为基准

    for fp in file_paths:
        log(f"  📂 读取: {fp.name}")
        df = load_file(fp)

        # 列名一致性检查
        if col_ref is None:
            col_ref = list(df.columns)
        else:
            if list(df.columns) != col_ref:
                missing = set(col_ref) - set(df.columns)
                extra   = set(df.columns) - set(col_ref)
                msg = f"文件 '{fp.name}' 列名与第一个文件不一致。"
                if missing:
                    msg += f"\n   缺少列: {sorted(missing)}"
                if extra:
                    msg += f"\n   多余列: {sorted(extra)}"

                on_mismatch = cfg.get('on_column_mismatch', 'error').strip().lower()
                if on_mismatch == 'error':
                    die(msg)
                elif on_mismatch == 'warn':
                    print(f"⚠️  WARNING: {msg}，仍继续拼接（列不对齐处填 NaN）。", flush=True)
                # ignore：静默继续

        log(f"     {len(df)} 行 × {len(df.columns)} 列")
        dfs.append(df)

    # ── 纵向拼接 ─────────────────────────────────────────────────
    log(f"\n纵向拼接中...")
    final_df = pd.concat(dfs, ignore_index=True)
    log(f"拼接完成：{len(final_df)} 行 × {len(final_df.columns)} 列")

    # ── 输出 ─────────────────────────────────────────────────────
    out_cfg   = cfg.get('output', {})
    out_path  = base_dir / out_cfg.get('path', 'vconcat_result.xlsx')
    out_sheet = out_cfg.get('sheet_name', 'Result')

    log(f"\n💾 写出结果到: {out_path.name}")
    with pd.ExcelWriter(str(out_path), engine='openpyxl') as writer:
        final_df.to_excel(writer, sheet_name=out_sheet, index=False)

    elapsed = time.time() - t_start
    log(f"\n✅ 全部完成！输出文件: {out_path}")
    log(f"   结果表: {len(final_df)} 行 × {len(final_df.columns)} 列")
    log(f"   总耗时: {elapsed:.1f} 秒")


if __name__ == '__main__':
    main()
