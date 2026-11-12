"""审批 Gate —— 唯一有权力覆盖线上正式模型的脚本。

严格规则：
  1. 默认只读（--list / --show / --active）
  2. --approve / --rollback 必须显式指定，且需要 --yes 二次确认
  3. 绝不允许自动触发（没有任何 cron / scheduler 会调本脚本的 --approve 分支）
  4. 所有动作都在 registry.json 里留历史记录，可回滚

用法:
  python monitoring/gate.py --list                                           # 列出所有组+面的版本历史
  python monitoring/gate.py --list --surface Top                            # 只看 Top
  python monitoring/gate.py --show Top2.799_Bot2.799 Top                    # 看 Top 当前 active 详情
  python monitoring/gate.py --approve Top2.799_Bot2.799 Top --version XXX --yes  # 部署 Top
  python monitoring/gate.py --rollback Top2.799_Bot2.799 Top --yes          # 回滚到上一个版本
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK1_ROOT = HERE.parent
if str(TASK1_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK1_ROOT))

from registry import ModelRegistry  # noqa: E402


GROUP_LABEL = "Top2.799_Bot2.799"
ONLINE_JOBS_DIR = TASK1_ROOT / "result/trained_models/by_group" / GROUP_LABEL


def _show(reg: ModelRegistry, group_label: str, surface: str):
    history = reg.get_history(group_label, surface)
    active = reg.get_active(group_label, surface)
    if not history:
        print(f"  ({group_label} / {surface}) 无历史")
        return

    print(f"  ({group_label} / {surface}) — active={active.get('version') if active else None}")
    for i, e in enumerate(reversed(history)):
        mark = "●" if e.get("active") else "○"
        cv_str = ""
        if e.get("cv_rmse_mean"):
            cv_str = f" CV={e['cv_rmse_mean']:.4f}±{e.get('cv_rmse_std', 0):.4f}"
        ho_str = ""
        if e.get("holdout_test_rmse"):
            ho_str = f" ho={e['holdout_test_rmse']:.4f}"
        print(f"  {mark} [{len(history)-1-i}] {e.get('version','?')}{cv_str}{ho_str} "
              f"(features={e.get('features')}, samples={e.get('training_samples')})")
        print(f"     path={e.get('path','?')}")


def _list(reg: ModelRegistry, surface_filter: str | None):
    # 先 seed: 如果 registry 是空的，把线上已在跑的模型注册进去作为 active 初始版本
    _ensure_seeded(reg)

    all_groups = set()
    import json as _json
    if reg.path.exists():
        with open(reg.path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        all_groups = set(data.keys())

    if not all_groups:
        print("registry.json 为空")
        return

    for group in sorted(all_groups):
        print(f"\n== {group} ==")
        for surface in ["Top", "Bot"]:
            if surface_filter and surface != surface_filter:
                continue
            _show(reg, group, surface)


def _ensure_seeded(reg: ModelRegistry):
    """首次运行 gate.py 时，把线上已在跑的模型注册为 active 初始版本。"""
    if reg.get_active(GROUP_LABEL, "Top") is None:
        top_job = ONLINE_JOBS_DIR / "Top_model.joblib"
        if top_job.exists():
            reg.seed_from_artifact(
                GROUP_LABEL, "Top", str(top_job),
                training_samples=2813, features=8,
            )
    if reg.get_active(GROUP_LABEL, "Bot") is None:
        bot_job = ONLINE_JOBS_DIR / "Bot_model.joblib"
        if bot_job.exists():
            reg.seed_from_artifact(
                GROUP_LABEL, "Bot", str(bot_job),
                training_samples=2813, features=8,
            )


def _approve(reg: ModelRegistry, group_label: str, surface: str, version: str | None, yes: bool):
    _ensure_seeded(reg)

    # 找到候选版本
    history = reg.get_history(group_label, surface)
    if not history:
        print(f"❌ {group_label}/{surface} registry 为空，无法 approve")
        sys.exit(1)

    target = None
    target_idx = None
    for i, e in enumerate(history):
        if version is None and not e.get("active") and ".pending." in e.get("path", ""):
            target, target_idx = e, i
            break
        if version and e.get("version") == version:
            target, target_idx = e, i
            break

    if target is None:
        print(f"❌ 找不到 version={version} 的候选 (--version 可省略，默认选第一个 pending)")
        sys.exit(1)

    pending_path = Path(target["path"])
    if not pending_path.exists():
        print(f"❌ 候选模型文件不存在: {pending_path}")
        sys.exit(1)

    dest = ONLINE_JOBS_DIR / f"{surface}_model.joblib"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 审批 Gate：二次确认
    old = reg.get_active(group_label, surface)
    print(f"\n审批确认（{group_label} / {surface}）:")
    print(f"  当前 active : version={old.get('version')}  path={old.get('path')}")
    print(f"  候选        : version={target.get('version')}  path={pending_path.name}")
    if old.get("cv_rmse_mean") and target.get("cv_rmse_mean"):
        delta = target["cv_rmse_mean"] - old["cv_rmse_mean"]
        print(f"  CV RMSE 对比: old={old['cv_rmse_mean']:.4f} ± {old.get('cv_rmse_std',0):.4f}"
              f"  new={target['cv_rmse_mean']:.4f} ± {target.get('cv_rmse_std',0):.4f}  "
              f"(Δ={delta:+.4f})")
    print(f"  部署动作    : 复制 {pending_path.name} → {dest.name}")

    if not yes:
        print("\n⚠️ 未传 --yes，已中止（纯预览）。")
        sys.exit(0)

    # 执行部署
    shutil.copy2(pending_path, dest)
    reg.activate(group_label, surface, target["version"])

    print(f"\n✅ 已部署: {dest} (version={target['version']})")
    print(f"   registry.json 已更新，可以随时 --rollback 回滚到上一个版本")


def _rollback(reg: ModelRegistry, group_label: str, surface: str, yes: bool):
    _ensure_seeded(reg)

    history = reg.get_history(group_label, surface)
    if len(history) < 2:
        print(f"❌ 历史版本不足（{len(history)} 条），无法回滚")
        sys.exit(1)

    active_idx = next((i for i, e in enumerate(history) if e.get("active")),
                      len(history) - 1)
    target = history[active_idx - 1]
    dest = ONLINE_JOBS_DIR / f"{surface}_model.joblib"

    print(f"\n回滚确认（{group_label} / {surface}）:")
    print(f"  当前 active : version={history[active_idx].get('version')}")
    print(f"  回滚到      : version={target.get('version')}  path={target.get('path')}")

    if not yes:
        print("\n⚠️ 未传 --yes，已中止（纯预览）。")
        sys.exit(0)

    src = Path(target["path"])
    if src.exists():
        shutil.copy2(src, dest)
    else:
        print(f"⚠️ 回滚目标文件不存在: {src}，registry 仍会更新 active 标记")

    reg.rollback(group_label, surface)
    print(f"\n✅ 已回滚: version={target['version']}")


def main():
    parser = argparse.ArgumentParser(description="模型审批 Gate（只读默认；approve/rollback 需显式 --yes）")
    parser.add_argument("--list", action="store_true", help="列出所有规格组版本历史（默认动作）")
    parser.add_argument("--show", nargs=2, metavar=("GROUP", "SURFACE"),
                        help="查看指定组+面的详细版本信息")
    parser.add_argument("--group", default=GROUP_LABEL, help="规格组标签")
    parser.add_argument("--surface", default=None, help="Top / Bot；--list 时过滤用")
    parser.add_argument("--approve", nargs="?", const="__auto_pick_pending__",
                        help="审批候选版本；--version 可选（不指定就选第一个 pending）")
    parser.add_argument("--rollback", action="store_true", help="回滚到上一个版本")
    parser.add_argument("--version", default=None, help="指定 approve 的目标版本号")
    parser.add_argument("--yes", action="store_true", help="跳过二次确认直接执行")
    args = parser.parse_args()

    reg = ModelRegistry()

    if args.show:
        group, surface = args.show
        _show(reg, group, surface)
        return

    if args.approve is not None:
        version = args.version if args.version else None
        if args.approve != "__auto_pick_pending__":
            version = args.approve
        _approve(reg, args.group, args.surface or "Top", version, args.yes)
        return

    if args.rollback:
        _rollback(reg, args.group, args.surface or "Top", args.yes)
        return

    _list(reg, args.surface)


if __name__ == "__main__":
    main()
