"""
run_merge.py
------------
读取 config.yaml，执行主副表合并，输出结果 xlsx。

用法：
    python run_merge.py                    # 使用同目录下的 config.yaml
    python run_merge.py --config my.yaml   # 指定配置文件
"""
import pandas as pd
import sys
import argparse
import warnings
from pathlib import Path
from collections import Counter
import time
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, column_index_from_string
import yaml
import xlwings as xw


# ══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════════

def log(msg):
    print(msg, flush=True)


def warn(msg):
    print(f"⚠️  WARNING: {msg}", flush=True)


def die(msg):
    print(f"❌ ERROR: {msg}", flush=True)
    sys.exit(1)


def build_merge_map(ws):
    """展开所有合并单元格：{(row, col): 左上角值}。"""
    m = {}
    for mr in ws.merged_cells.ranges:
        top_val = ws.cell(row=mr.min_row, column=mr.min_col).value
        for r in range(mr.min_row, mr.max_row + 1):
            for c in range(mr.min_col, mr.max_col + 1):
                m[(r, c)] = top_val
    return m


def get_val(ws, merge_map, row, col):
    return merge_map.get((row, col), ws.cell(row=row, column=col).value)


def ffill(lst):
    result, last = [], None
    for v in lst:
        sv = str(v).strip() if v is not None else ''
        if sv:
            last = sv
        result.append(last)
    return result


def clean_str(s):
    if s is None:
        return ''
    return str(s).replace('\n', '').replace('\r', '').strip()


def detect_max_col(ws, merge_map):
    """从合并单元格map和表头行直接取最大列号，避免逐列扫描。"""
    # 从 merge_map 取最大列号
    max_col = max((c for r, c in merge_map.keys()
                   if HEADER_ROW_START <= r <= HEADER_ROW_END), default=0)
    # 再扫一遍表头行的非合并普通单元格（防止最后几列没有合并单元格）
    for r in range(HEADER_ROW_START, HEADER_ROW_END + 1):
        for c in range(max_col, max_col + 20):  # 只往后多扫20列做保险
            v = ws.cell(row=r, column=c).value
            if v is not None and clean_str(v):
                max_col = max(max_col, c)
    return max_col

def generate_column_names(ws, merge_map, max_col):
    """
    生成主表所有列的最终列名。
    返回：list[str]，长度 = max_col，index 从 0 开始对应 col 1。
    """
    raw = []
    for row_idx in range(HEADER_ROW_START, HEADER_ROW_END + 1):
        raw.append([get_val(ws, merge_map, row_idx, c) for c in range(1, max_col + 1)])

    l1f = ffill(raw[0])
    l2f = ffill(raw[1])
    l3f = ffill(raw[2])
    l4  = [clean_str(v) for v in raw[3]]

    col_names = []
    for i in range(max_col):
        parts = []
        for lvl in [l1f[i], l2f[i], l3f[i], l4[i]]:
            sv = clean_str(lvl)
            if sv and (not parts or parts[-1] != sv):
                parts.append(sv)
        col_names.append('_'.join(parts) if parts else f'col_{i+1:03d}')

    # 去重加后缀
    counts = Counter()
    final = []
    for name in col_names:
        counts[name] += 1
        final.append(f"{name}_{counts[name]-1}" if counts[name] > 1 else name)
    return final


# ══════════════════════════════════════════════════════════════════════════════
#  主表读取
# ══════════════════════════════════════════════════════════════════════════════

def load_master(path: Path, keep_col_letters: list[str]) -> pd.DataFrame:
    """
    读取主表，只保留 keep_col_letters 指定的列。
    返回 DataFrame，列名为生成的语义列名，同时保留原列字母作为参考信息
    （存在 df.attrs['col_letter_map'] 中）。
    """
    log(f"📂 读取主表: {path.parent.name}/{path.name}")
    log("   展开合并单元格中...")


    # 第一次加载：read_only=False，只用来读合并单元格和表头，读完立即关闭
    wb_full = load_workbook(str(path), read_only=False, data_only=True)
    ws_full = wb_full.active
    merge_map = build_merge_map(ws_full)
    max_col = detect_max_col(ws_full, merge_map)
    log(f"   检测到 {max_col} 列")
    all_col_names = generate_column_names(ws_full, merge_map, max_col)
    wb_full.close()

    # 第二次加载：read_only=True，流式读取数据行，速度快
    wb = load_workbook(str(path), read_only=True, data_only=True)
    ws = wb.active

    # 将列字母转为 0-indexed 列索引，验证有效性
    keep_indices = []
    col_letter_map = {}   # 语义名 -> 列字母
    for letter in keep_col_letters:
        letter = letter.strip().upper()
        idx = column_index_from_string(letter) - 1   # 0-indexed
        if idx >= max_col:
            die(f"列 {letter} 超出主表范围（共 {max_col} 列）")
        semantic_name = all_col_names[idx]
        keep_indices.append(idx)
        col_letter_map[semantic_name] = letter

    # 读取数据行，只取需要的列
    log("   读取数据行中...")
    rows = []
    for row in ws.iter_rows(min_row=DATA_START_ROW, values_only=True):
        # 截断：遇到全空行停止
        if all(v is None for v in row):
            break
        rows.append([row[i] if i < len(row) else None for i in keep_indices])

    if not rows:
        die("主表没有读取到任何数据行，请检查 DATA_START_ROW 设置。")

    selected_names = [all_col_names[i] for i in keep_indices]
    df = pd.DataFrame(rows, columns=selected_names)
    df.attrs['col_letter_map'] = col_letter_map

    log(f"   ✅ 主表读取完成：{len(df)} 行 × {len(df.columns)} 列")
    return df
def parse_master_schema(path: Path, keep_col_letters: list[str]):
    """
    只做一次：展开合并单元格、生成列名、确定 keep_indices。
    返回 (keep_indices, selected_names, col_letter_map)。
    在多张同结构主表的循环外调用一次即可。
    """
    log(f"📐 解析主表结构（仅需一次）: {path.parent.name}/{path.name}")
    log("   展开合并单元格中...")
    wb_full = load_workbook(str(path), read_only=False, data_only=True)
    ws_full = wb_full.active
    merge_map = build_merge_map(ws_full)
    max_col = detect_max_col(ws_full, merge_map)
    log(f"   检测到 {max_col} 列")
    all_col_names = generate_column_names(ws_full, merge_map, max_col)
    wb_full.close()

    keep_indices = []
    col_letter_map = {}
    for letter in keep_col_letters:
        letter = letter.strip().upper()
        idx = column_index_from_string(letter) - 1
        if idx >= max_col:
            die(f"列 {letter} 超出主表范围（共 {max_col} 列）")
        semantic_name = all_col_names[idx]
        keep_indices.append(idx)
        col_letter_map[semantic_name] = letter

    selected_names = [all_col_names[i] for i in keep_indices]
    log(f"   ✅ 结构解析完成，保留 {len(keep_indices)} 列")
    return keep_indices, selected_names, col_letter_map


def load_master_data(path: Path, keep_indices: list, selected_names: list, col_letter_map: dict) -> pd.DataFrame:
    """
    只读数据行，表头结构复用 parse_master_schema 的结果。
    """
    log(f"📂 读取主表数据: {path.parent.name}/{path.name}")
    wb = load_workbook(str(path), read_only=True, data_only=True)
    ws = wb.active

    rows = []
    for row in ws.iter_rows(min_row=DATA_START_ROW, values_only=True):
        if all(v is None for v in row):
            break
        rows.append([row[i] if i < len(row) else None for i in keep_indices])

    wb.close()

    if not rows:
        die(f"主表 {path.parent.name}/{path.name} 没有读取到任何数据行。")

    df = pd.DataFrame(rows, columns=selected_names)
    df.attrs['col_letter_map'] = col_letter_map
    log(f"   ✅ 读取完成：{len(df)} 行 × {len(df.columns)} 列")
    return df

# ══════════════════════════════════════════════════════════════════════════════
#  合并策略
# ══════════════════════════════════════════════════════════════════════════════

VALID_STRATEGIES = {
    'first_nonempty', 'first', 'mean', 'sum', 'max', 'min', 'join_unique'
}


def infer_strategy(series: pd.Series) -> str:
    """根据列数据类型推断默认策略。"""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 'first_nonempty'
    if pd.api.types.is_numeric_dtype(non_null):
        return 'mean'
    try:
        pd.to_numeric(non_null)
        return 'mean'
    except (ValueError, TypeError):
        return 'first_nonempty'


def apply_strategy(group: pd.Series, strategy: str) -> object:
    """
    对一个分组的 Series 应用指定策略，返回聚合结果。
    空值/无有效数值时静默返回 None，由调用方统计后汇总报警。
    """
    non_null = group.dropna()

    if strategy == 'first_nonempty':
        return non_null.iloc[0] if len(non_null) > 0 else None

    elif strategy == 'first':
        return group.iloc[0] if len(group) > 0 else None

    elif strategy == 'mean':
        numeric = pd.to_numeric(non_null, errors='coerce').dropna()
        return numeric.mean() if len(numeric) > 0 else None

    elif strategy == 'sum':
        numeric = pd.to_numeric(non_null, errors='coerce').dropna()
        return numeric.sum() if len(numeric) > 0 else None

    elif strategy == 'max':
        numeric = pd.to_numeric(non_null, errors='coerce').dropna()
        return numeric.max() if len(numeric) > 0 else None

    elif strategy == 'min':
        numeric = pd.to_numeric(non_null, errors='coerce').dropna()
        return numeric.min() if len(numeric) > 0 else None

    elif strategy == 'join_unique':
        vals = non_null.astype(str).unique().tolist()
        return ','.join(vals) if vals else None

    else:
        die(f"未知策略: '{strategy}'，合法值为: {VALID_STRATEGIES}")


# ══════════════════════════════════════════════════════════════════════════════
#  副表读取 & 合并
# ══════════════════════════════════════════════════════════════════════════════

def load_sub(path: Path) -> pd.DataFrame:
    """读取副表（标准格式，第一行是表头）。"""
    suffix = path.suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(str(path), header=0)
    elif suffix == '.csv':
        # 尝试 utf-8，失败则 gbk
        try:
            df = pd.read_csv(str(path), header=0, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(str(path), header=0, encoding='gbk')
    else:
        die(f"副表格式不支持: {path.suffix}，仅支持 xlsx/xls/csv。")
    return df


def merge_one_sub(master_df: pd.DataFrame, sub_cfg: dict, base_dir: Path) -> pd.DataFrame:
    """
    执行单个副表的 LEFT JOIN 合并。
    master_df 的列名是语义列名；sub_cfg 来自 config.yaml。
    """
    sub_path = base_dir / sub_cfg['path']
    if not sub_path.exists():
        die(f"副表文件不存在: {sub_path}")

    log(f"\n📂 读取副表: {sub_path.name}")
    sub_df = load_sub(sub_path)
    log(f"   副表原始: {len(sub_df)} 行 × {len(sub_df.columns)} 列")

    # ── 关键字段 ──
    join_key    = sub_cfg['join_key']
    master_key_letter = join_key['master_col'].strip().upper()
    sub_key_col       = join_key['sub_col'].strip()

    # 把主表列字母转换成语义列名
    master_key_col = None
    for col in master_df.columns:
        if master_df.attrs.get('col_letter_map', {}).get(col) == master_key_letter:
            master_key_col = col
            break
    if master_key_col is None:
        die(f"主表中找不到列字母 '{master_key_letter}' 对应的列，"
            f"请确认该列在 master.keep_columns 中。")

    if sub_key_col not in sub_df.columns:
        die(f"副表 '{sub_path.name}' 中找不到关键字段列 '{sub_key_col}'，"
            f"实际列名为: {list(sub_df.columns)}")

    # ── 类型一致性检查 ──
    master_key_series = master_df[master_key_col]
    sub_key_series    = sub_df[sub_key_col]

    master_numeric = pd.api.types.is_numeric_dtype(master_key_series.dropna())
    sub_numeric    = pd.api.types.is_numeric_dtype(sub_key_series.dropna())
    if master_numeric != sub_numeric:
        die(f"关键字段类型不一致：主表 '{master_key_col}' 为 "
            f"{'数值' if master_numeric else '文本'}型，"
            f"副表 '{sub_key_col}' 为 "
            f"{'数值' if sub_numeric else '文本'}型。"
            f"\n   请检查数据或在合并前进行类型转换。")

    # ── 副表保留列配置 ──
    keep_cols_cfg = sub_cfg.get('keep_columns', [])
    if not keep_cols_cfg:
        warn(f"副表 '{sub_path.name}' 未配置 keep_columns，将不添加任何副表列。")
        return master_df

    # 验证副表列存在，推断默认策略
    processed_keep = []
    for item in keep_cols_cfg:
        col_name = item['col'].strip()
        if col_name not in sub_df.columns:
            die(f"副表 '{sub_path.name}' 中找不到列 '{col_name}'，"
                f"实际列名为: {list(sub_df.columns)}")
        strategy = item.get('strategy', None)
        if strategy is None:
            strategy = infer_strategy(sub_df[col_name])
            log(f"   列 '{col_name}' 未指定策略，自动推断为: {strategy}")
        elif strategy not in VALID_STRATEGIES:
            die(f"列 '{col_name}' 的策略 '{strategy}' 无效，"
                f"合法值为: {sorted(VALID_STRATEGIES)}")
        processed_keep.append((col_name, strategy))

    # ── 未匹配行统计 ──
    master_keys = set(master_df[master_key_col].dropna().unique())
    sub_keys    = set(sub_df[sub_key_col].dropna().unique())
    unmatched   = master_keys - sub_keys
    # 读取 unmatched 行为配置（drop=删除未匹配行；keep=保留填空值，默认）
    unmatched_action = join_key.get('unmatched', 'keep').strip().lower()
    if unmatched_action not in ('drop', 'keep'):
        die(f"join_key.unmatched 的值 '{unmatched_action}' 无效，合法值为: drop / keep")
    all_null_action = join_key.get('all_null', 'keep').strip().lower()

    if all_null_action not in ('drop', 'keep'):
        die(f"join_key.all_null 的值 '{all_null_action}' 无效，合法值为: drop / keep")

    if unmatched:
        action_desc = "这些行将被删除" if unmatched_action == 'drop' else "这些行的副表列将填入空值"
        warn(f"主表中有 {len(unmatched)} 个关键字段值在副表中无匹配，{action_desc}。")
        if len(unmatched) <= 10:
            warn(f"   未匹配值: {sorted(str(v) for v in unmatched)}")
        else:
            sample = sorted(str(v) for v in list(unmatched)[:10])
            warn(f"   前10个未匹配值: {sample} ...")

    # ── 聚合副表（处理一对多）──
    sub_keep_df = sub_df[[sub_key_col] + [c for c, _ in processed_keep]].copy()

    # 全列空值检查（一次性）
    for col_name, _ in processed_keep:
        if sub_keep_df[col_name].isna().all():
            warn(f"副表列 '{col_name}' 全为空值。")

    # 按关键字段分组，对每个保留列应用策略
    # null_counts 统计每列产生 None 结果的次数，最后汇总报一次
    grouped = sub_keep_df.groupby(sub_key_col, sort=False)
    agg_rows = {}
    null_counts = Counter()
    for key, group in grouped:
        row_result = {}
        for col_name, strategy in processed_keep:
            val = apply_strategy(group[col_name], strategy)
            if val is None:
                null_counts[col_name] += 1
            row_result[col_name] = val
        agg_rows[key] = row_result

    # 汇总报警：每列只报一次，说明有多少个 key 产生了空结果
    for col_name, cnt in null_counts.items():
        strategy = dict(processed_keep)[col_name]
        warn(f"列 '{col_name}'（策略: {strategy}）有 {cnt} 个分组无有效值，已填入 None。")

    # 构建聚合后的副表 DataFrame
    agg_df = pd.DataFrame.from_dict(agg_rows, orient='index')
    agg_df.index.name = sub_key_col
    agg_df = agg_df.reset_index()

    # ── 列名冲突检查 ──
    master_cols = set(master_df.columns)
    for col_name, _ in processed_keep:
        if col_name in master_cols and col_name != master_key_col:
            new_name = f"{col_name}_sub"
            warn(f"副表列 '{col_name}' 与主表列重名，重命名为 '{new_name}'。")
            agg_df = agg_df.rename(columns={col_name: new_name})

    # ── JOIN（left=保留未匹配行，inner=删除未匹配行）──
    join_how = 'inner' if unmatched_action == 'drop' else 'left'
    result = master_df.merge(
        agg_df,
        left_on=master_key_col,
        right_on=sub_key_col,
        how=join_how
    )

    if unmatched_action == 'drop' and unmatched:
        log(f"   已删除 {len(master_df) - len(result)} 行未匹配记录，"
            f"保留 {len(result)} 行。")
        # ── 过滤副表列全为空的行 ──
        if all_null_action == 'drop':
            sub_cols_in_result = [c for c, _ in processed_keep if c in result.columns]
            before = len(result)
            result = result.dropna(subset=sub_cols_in_result, how='all')
            dropped = before - len(result)
            if dropped > 0:
                log(f"   已删除 {dropped} 行（副表保留列全为空值），保留 {len(result)} 行。")

    # 如果关键字段名不同，合并后会多一列，删掉副表的那列
    if sub_key_col != master_key_col and sub_key_col in result.columns:
        result = result.drop(columns=[sub_key_col])

    # 保留 attrs
    result.attrs['col_letter_map'] = master_df.attrs.get('col_letter_map', {})

    log(f"   ✅ 合并完成：{len(result)} 行 × {len(result.columns)} 列")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global HEADER_ROW_START, HEADER_ROW_END, DATA_START_ROW
    t_start = time.time()
    parser = argparse.ArgumentParser(description="主副表合并工具")
    parser.add_argument('--config', default=None, help="配置文件路径（默认：同目录下的 config.yaml）")
    args = parser.parse_args()

    # 确定配置文件路径
    if args.config:
        cfg_path = Path(args.config)
    else:
        cfg_path = Path(__file__).parent / "config.yaml"

    if not cfg_path.exists():
        die(f"配置文件不存在: {cfg_path}")

    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    base_dir = cfg_path.parent

    # ── 读取主表 ──
    master_cfg = cfg['master']
    search_root = Path(master_cfg['search_root'])
    pattern = master_cfg.get('search_pattern', f"*/{master_cfg['filename']}")
    master_paths = sorted(search_root.glob(pattern))
    if not master_paths:
        die(f"在 {search_root} 下按 '{pattern}' 未找到任何主表文件。")
    log(f"共找到 {len(master_paths)} 个主表文件：")
    for p in master_paths:
        log(f"  - {p.parent.name}/{p.name}")
    HEADER_ROW_START = master_cfg.get('header_row_start', 11)
    HEADER_ROW_END = master_cfg.get('header_row_end', 14)
    DATA_START_ROW = master_cfg.get('data_start_row', 15)

    # # 检查是否有 .xls 文件
    # xls_files = [p for p in master_paths if p.suffix.lower() == '.xls']
    # if xls_files:
    #     log(f"\n❌ 发现 {len(xls_files)} 个不支持的 .xls 文件，请手动转换为 .xlsx 后重新运行：")
    #     for p in xls_files:
    #         log(f"   - {p.parent.name}/{p.name}")
    #     log(f"\n   转换方法：用 Excel 打开该文件 → 另存为 → 选择 xlsx 格式")
    #     sys.exit(1)

    keep_col_letters = cfg['master'].get('keep_columns', [])
    if not keep_col_letters:
        die("config.yaml 中 master.keep_columns 为空，请至少指定一列。")

    sub_tables = cfg.get('sub_tables', [])
    if not sub_tables:
        warn("config.yaml 中未配置任何副表（sub_tables 为空），将直接输出主表筛选结果。")

    # all_results = []
    #
    # for master_path in master_paths:
    #     result_df = load_master(master_path, keep_col_letters)
    #     all_results.append(result_df)


    # 用第一张表解析结构，只做一次
    keep_indices, selected_names, col_letter_map = parse_master_schema(
        master_paths[0], keep_col_letters
    )

    all_results = []
    for master_path in master_paths:
        result_df = load_master_data(master_path, keep_indices, selected_names, col_letter_map)
        all_results.append(result_df)


    log(f"纵向拼接 {len(all_results)} 张表...")
    # col_letter_map = all_results[0].attrs.get('col_letter_map', {}) # 删掉
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.attrs['col_letter_map'] = col_letter_map

    for i, sub_cfg in enumerate(sub_tables, 1):
        log(f"\n── 合并第 {i} 个副表 ──")
        final_df = merge_one_sub(final_df, sub_cfg, base_dir)

    # ── 输出 ──
    out_cfg   = cfg.get('output', {})
    out_path  = base_dir / out_cfg.get('path', 'merged_result.xlsx')
    out_sheet = out_cfg.get('sheet_name', 'Result')

    log(f"\n💾 写出结果到: {out_path.name}")
    with pd.ExcelWriter(str(out_path), engine='openpyxl') as writer:
        final_df.to_excel(writer, sheet_name=out_sheet, index=False)

    log(f"\n✅ 全部完成！输出文件: {out_path}")
    log(f"   结果表: {len(final_df)} 行 × {len(final_df.columns)} 列")
    elapsed = time.time() - t_start
    log(f"   总耗时: {elapsed:.1f} 秒")

if __name__ == '__main__':
    main()