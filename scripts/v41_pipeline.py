#!/usr/bin/env python3
"""
CnS v4.1+ 傻瓜式一键流水线：代码同步 / 续训 / 全量生成与合并 / 版本对比。

纯 Python 标准库实现（本地 Windows 无需安装 PyTorch 即可调度 llm24g RTX 4090）。

常用示例：
  # 1. 同步代码与 ids.txt 到 llm24g（自动补齐缺失的 ids.txt 并校验 SHA256）
  python scripts/v41_pipeline.py sync

  # 2. 基于 v4.1 权重续训 15 轮，产出 v42 权重
  python scripts/v41_pipeline.py resume --base_ckpt checkpoints/v41_zh_sdf_global.pt --tag v42 --epochs 15 --lr 2e-5

  # 3. 用 v42 权重全量生成 Missing + SCVariants + Full 合并字体，并自动拉回本地校验 SHA256
  python scripts/v41_pipeline.py build --tag v42

  # 4. 生成 v41 vs v42 可视化对比图与量化指标差值表
  python scripts/v41_pipeline.py compare --a v41 --b v42

  # 5. 一键全自动：同步 -> 续训 -> 生成与合并 -> 拉回 -> 对比
  python scripts/v41_pipeline.py all --base_ckpt checkpoints/v41_zh_sdf_global.pt --tag v42 --epochs 15 --lr 2e-5 --compare_with v41
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
HOST = "llm24g"
CONDA_ENV = "comfyui"
REMOTE_ROOT = "~/cns_font"
NOTO_REMOTE = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
IDS_URL = "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt"

SYNC_FILES = [
    "src/v4/__init__.py",
    "src/v4/model_v4.py",
    "src/v4/dataset_v4.py",
    "src/v4/losses_v4.py",
    "src/v4/synthesizer_v4.py",
    "src/v3/curve_constrain.py",
    "src/v3/hb_light.py",
    "src/v3/structural_gate.py",
    "src/v3/variant_classifier.py",
    "src/v3/local_inpainter.py",
    "src/v3/otf_packager.py",
    "src/v2/hb_outlines.py",
    "scripts/v4_train.py",
    "scripts/v4_build_tg8105.py",
    "scripts/v3_merge_fonts.py",
    "scripts/v41_compare_tags.py",
]


def _sha256_local(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: List[str], dry_run: bool = False, capture: bool = False) -> str:
    shown = " ".join(cmd)
    if dry_run:
        print(f"  [DRY-RUN] {shown}")
        return ""
    res = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
    )
    if res.returncode != 0:
        if capture:
            print(res.stdout, end="")
            print(res.stderr, end="", file=sys.stderr)
        raise SystemExit(f"[ERROR] Command failed (exit {res.returncode}): {shown}")
    return res.stdout if capture else ""


def remote_exec(remote_cmd: str, dry_run: bool = False, capture: bool = False) -> str:
    """Execute command on llm24g via safe_exec.py inside Conda env `comfyui`."""
    return _run(
        [sys.executable, str(ROOT / "safe_exec.py"), "-e", CONDA_ENV, HOST, remote_cmd],
        dry_run=dry_run,
        capture=capture,
    )


def scp_push(local_path: Path, remote_rel: str, dry_run: bool = False) -> None:
    _run(["scp", "-o", "BatchMode=yes", str(local_path), f"{HOST}:{REMOTE_ROOT}/{remote_rel}"], dry_run=dry_run)


def scp_pull(remote_rel: str, local_path: Path, dry_run: bool = False) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _run(["scp", "-o", "BatchMode=yes", f"{HOST}:{REMOTE_ROOT}/{remote_rel}", str(local_path)], dry_run=dry_run)


def verify_sha256(rel_paths: List[str], dry_run: bool = False) -> None:
    if dry_run or not rel_paths:
        return
    remote_files = " ".join(f"{REMOTE_ROOT}/{p}" for p in rel_paths)
    out = remote_exec(f"sha256sum {remote_files}", capture=True)
    remote_map = {}
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            remote_map[Path(parts[1]).name] = parts[0].lower()
    for p in rel_paths:
        lp = ROOT / p
        loc_h = _sha256_local(lp)
        rem_h = remote_map.get(lp.name)
        if loc_h != rem_h:
            raise SystemExit(f"[SHA256 MISMATCH] {p}: local={loc_h} vs remote={rem_h}")
        print(f"  [SHA256 OK] {p} ({loc_h[:12]}...)")


def cmd_sync(args: argparse.Namespace) -> None:
    print("[1/2] Checking data_v2/ids.txt ...")
    ids_local = ROOT / "data_v2" / "ids.txt"
    if not ids_local.exists() or ids_local.stat().st_size < 500_000:
        if args.dry_run:
            print(f"  [DRY-RUN] Download {IDS_URL} -> {ids_local}")
        else:
            ids_local.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Downloading cjkvi-ids -> {ids_local} ...")
            urllib.request.urlretrieve(IDS_URL, str(ids_local))
            print(f"  Downloaded {ids_local.stat().st_size:,} bytes.")

    print("[2/2] Syncing source files & ids.txt to llm24g ...")
    remote_exec(
        f"mkdir -p {REMOTE_ROOT}/src/v4 {REMOTE_ROOT}/src/v3 {REMOTE_ROOT}/src/v2 "
        f"{REMOTE_ROOT}/scripts {REMOTE_ROOT}/data_v2 {REMOTE_ROOT}/outputs "
        f"{REMOTE_ROOT}/checkpoints {REMOTE_ROOT}/manual_review",
        dry_run=args.dry_run,
    )
    to_verify = ["data_v2/ids.txt"]
    scp_push(ids_local, "data_v2/ids.txt", dry_run=args.dry_run)
    for rel in SYNC_FILES:
        lp = ROOT / rel
        if lp.exists():
            scp_push(lp, rel, dry_run=args.dry_run)
            to_verify.append(rel)
    verify_sha256(to_verify, dry_run=args.dry_run)
    print("[Sync Done]")


def cmd_resume(args: argparse.Namespace) -> None:
    tag = args.tag
    base_ckpt_rel = args.base_ckpt.replace("\\", "/")
    out_ckpt_rel = f"checkpoints/{tag}_zh_sdf_global.pt"
    out_log_rel = f"outputs/{tag}_train_log.json"

    print(f"[Resume] {base_ckpt_rel} -> {out_ckpt_rel} ({args.epochs} epochs, lr={args.lr}, batch={args.batch_size})")
    remote_cmd = (
        f"cd {REMOTE_ROOT} && set -o pipefail && PYTHONUNBUFFERED=1 "
        f"python scripts/v4_train.py "
        f"--noto_sans {NOTO_REMOTE} "
        f"--resume {base_ckpt_rel} "
        f"--batch_size {args.batch_size} "
        f"--stage2_epochs {args.epochs} "
        f"--stage2_lr {args.lr} "
        f"--ckpt_out {out_ckpt_rel} "
        f"--out_json {out_log_rel} "
        f"2>&1 | tee outputs/{tag}_train_stdout.log"
    )
    remote_exec(remote_cmd, dry_run=args.dry_run)

    print(f"[Resume] Pulling {out_ckpt_rel} & {out_log_rel} ...")
    scp_pull(out_ckpt_rel, ROOT / out_ckpt_rel, dry_run=args.dry_run)
    scp_pull(out_log_rel, ROOT / out_log_rel, dry_run=args.dry_run)
    verify_sha256([out_ckpt_rel, out_log_rel], dry_run=args.dry_run)
    print("[Resume Done]")


def cmd_build(args: argparse.Namespace) -> None:
    tag = args.tag
    ckpt_rel = (args.ckpt or f"checkpoints/{tag}_zh_sdf_global.pt").replace("\\", "/")
    miss_rel = f"outputs/CnS-SkipPro-B-{tag}-Missing.otf"
    var_rel = f"outputs/CnS-SkipPro-B-{tag}-SCVariants.otf"
    full_rel = f"outputs/CnS-SkipPro-B-{tag}-Full.otf"
    met_rel = f"outputs/{tag}_metrics.json"
    man_rel = f"outputs/{tag}_glyph_manifest.json"
    rev_json_rel = f"manual_review/{tag}_needs_review.json"
    rev_otf_rel = f"manual_review/{tag}_needs_review.otf"

    print(f"[Build] tag={tag}, ckpt={ckpt_rel}, merge={not args.no_merge}")
    build_cmd = (
        f"cd {REMOTE_ROOT} && set -o pipefail && PYTHONUNBUFFERED=1 "
        f"python scripts/v4_build_tg8105.py "
        f"--noto_sans {NOTO_REMOTE} "
        f"--ckpt_path {ckpt_rel} "
        f"--version_tag {tag} "
        f"2>&1 | tee outputs/{tag}_build_stdout.log"
    )
    remote_exec(build_cmd, dry_run=args.dry_run)

    pull_list = [miss_rel, var_rel, met_rel, man_rel, rev_json_rel]
    if not args.no_merge:
        merge_cmd = (
            f"cd {REMOTE_ROOT} && python scripts/v3_merge_fonts.py "
            f"--base_otf FOT-SkipPro-B.OTF "
            f"--missing_otf {miss_rel} "
            f"--variants_otf {var_rel} "
            f"--out_otf {full_rel} "
            f"--emit_merged"
        )
        remote_exec(merge_cmd, dry_run=args.dry_run)
        pull_list.append(full_rel)

    print(f"[Build] Pulling outputs for {tag} ...")
    for rel in pull_list:
        scp_pull(rel, ROOT / rel, dry_run=args.dry_run)
    # Optional review OTF (only created if blocked_to_review > 0)
    try:
        scp_pull(rev_otf_rel, ROOT / rev_otf_rel, dry_run=args.dry_run)
    except SystemExit:
        pass
    verify_sha256(pull_list, dry_run=args.dry_run)
    print("[Build Done]")


def cmd_compare(args: argparse.Namespace) -> None:
    png_rel = f"outputs/compare_{args.a}_vs_{args.b}.png"
    chars_arg = f" --chars {args.chars}" if args.chars else ""
    print(f"[Compare] {args.a} vs {args.b} -> {png_rel}")
    remote_exec(
        f"cd {REMOTE_ROOT} && python scripts/v41_compare_tags.py --a {args.a} --b {args.b} --out_png {png_rel}{chars_arg}",
        dry_run=args.dry_run,
    )
    scp_pull(png_rel, ROOT / png_rel, dry_run=args.dry_run)
    verify_sha256([png_rel], dry_run=args.dry_run)
    print(f"[Compare Done] Open {ROOT / png_rel}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_sync(args)
    cmd_resume(args)
    args.ckpt = f"checkpoints/{args.tag}_zh_sdf_global.pt"
    cmd_build(args)
    if args.compare_with:
        args.a = args.compare_with
        args.b = args.tag
        cmd_compare(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="CnS v4.1+ 傻瓜式一键续训、生成、合并与比对流水线")
    parser.add_argument("--dry_run", action="store_true", help="仅打印将要执行的命令，不实际运行")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # sync
    sub.add_parser("sync", help="同步代码与 ids.txt 至 llm24g 并校验 SHA256")

    # resume
    p_res = sub.add_parser("resume", help="基于已有权重在 RTX 4090 上续训")
    p_res.add_argument("--base_ckpt", default="checkpoints/v41_zh_sdf_global.pt", help="基础权重路径")
    p_res.add_argument("--tag", required=True, help="新版本标签（如 v42），不会覆盖旧权重")
    p_res.add_argument("--epochs", type=int, default=15, help="续训轮数")
    p_res.add_argument("--lr", type=float, default=2e-5, help="续训最大学习率（cosine 衰减）")
    p_res.add_argument("--batch_size", type=int, default=16)

    # build
    p_bld = sub.add_parser("build", help="全量生成 Missing + SCVariants + Full 合并字体")
    p_bld.add_argument("--tag", required=True, help="版本标签（如 v41 或 v42）")
    p_bld.add_argument("--ckpt", default=None, help="权重路径（默认 checkpoints/<tag>_zh_sdf_global.pt）")
    p_bld.add_argument("--no_merge", action="store_true", help="仅生成分卷字体，不合成 Full.otf")

    # compare
    p_cmp = sub.add_parser("compare", help="渲染双版本对比图并输出指标差值")
    p_cmp.add_argument("--a", required=True, help="基线版本标签（如 v4 或 v41）")
    p_cmp.add_argument("--b", required=True, help="待测版本标签（如 v41 或 v42）")
    p_cmp.add_argument("--chars", default="", help="可选：额外追加一行自定义抽检汉字")

    # all
    p_all = sub.add_parser("all", help="一键全流程：sync -> resume -> build(+merge) -> compare")
    p_all.add_argument("--base_ckpt", default="checkpoints/v41_zh_sdf_global.pt")
    p_all.add_argument("--tag", required=True, help="新版本标签（如 v42）")
    p_all.add_argument("--epochs", type=int, default=15)
    p_all.add_argument("--lr", type=float, default=2e-5)
    p_all.add_argument("--batch_size", type=int, default=16)
    p_all.add_argument("--no_merge", action="store_true")
    p_all.add_argument("--compare_with", default="v41", help="生成后自动与该版本对比（设为空则跳过）")
    p_all.add_argument("--chars", default="")

    args = parser.parse_args()
    dispatch = {
        "sync": cmd_sync,
        "resume": cmd_resume,
        "build": cmd_build,
        "compare": cmd_compare,
        "all": cmd_all,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
