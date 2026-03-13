#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path

CANDIDATE_REPO_ROOTS = [
    Path.home() / "ibrido_ws" / "src" / "AugMPC",
    Path.home() / "work" / "containers" / "ibrido-singularity" / "ibrido_ws" / "src" / "AugMPC",
    Path.home() / "work" / "containers" / "ibrido-singularity-xbot" / "ibrido_ws" / "src" / "AugMPC",
    Path.home() / "work" / "containers" / "ibrido-singularity-u24" / "ibrido_ws" / "src" / "AugMPC",
]
for candidate in CANDIDATE_REPO_ROOTS:
    if candidate.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from aug_mpc.utils.model_bundle import infer_checkpoint_file, load_existing_framework_repos, write_bundle_manifest

IGNORE_DIR_PATTERNS = {
    "__pycache__",
    "env_db_checkpoints",
    "wandb",
}
IGNORE_FILE_PATTERNS = {
    "*.pyc",
    "*.hdf5",
    "*.db3",
}
IGNORE_TOPLEVEL_PATTERNS = {
    "rosbag_*",
}


def should_ignore_dir(name: str, rel_path: str) -> bool:
    if name in IGNORE_DIR_PATTERNS:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in IGNORE_TOPLEVEL_PATTERNS if "/" not in rel_path)


def should_ignore_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in IGNORE_FILE_PATTERNS)


def _is_extra_checkpoint_file(name: str, main_checkpoint_file: str) -> bool:
    if name == main_checkpoint_file:
        return False
    return ("_checkpoint" in name and "_model" in name) or name.endswith(".ckpt")


def copy_filtered_bundle(
    src_bundle: Path,
    dst_bundle: Path,
    main_checkpoint_file: str,
    drop_extra_checkpoints: bool,
) -> None:
    if dst_bundle.exists():
        shutil.rmtree(dst_bundle)
    dst_bundle.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(src_bundle):
        root_path = Path(root)
        rel_root = root_path.relative_to(src_bundle)
        rel_root_str = rel_root.as_posix()

        if rel_root == Path('.'):
            dirs[:] = [
                d for d in dirs
                if not any(fnmatch.fnmatch(d, pat) for pat in IGNORE_TOPLEVEL_PATTERNS)
                and not should_ignore_dir(d, rel_root_str)
            ]
        else:
            dirs[:] = [d for d in dirs if not should_ignore_dir(d, rel_root_str)]

        if rel_root_str.startswith("ibrido_run_"):
            files = [f for f in files if fnmatch.fnmatch(f, "training_cfg_*")]
        else:
            files = [f for f in files if not should_ignore_file(f)]

        if drop_extra_checkpoints:
            files = [f for f in files if not _is_extra_checkpoint_file(f, main_checkpoint_file)]

        dst_root = dst_bundle / rel_root
        dst_root.mkdir(parents=True, exist_ok=True)
        for fname in files:
            shutil.copy2(root_path / fname, dst_root / fname)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an AugMPC model bundle into AugMPCModels format")
    parser.add_argument("--src_bundle", required=True, help="Source bundle directory")
    parser.add_argument(
        "--dst_root",
        default=str(Path.home() / "training_data" / "AugMPCModels" / "bundles"),
        help="Destination root containing bundle groups",
    )
    parser.add_argument("--group", default=None, help="Optional destination group inserted under --dst_root")
    parser.add_argument("--workspace_src", default=None, help="Workspace src root for git metadata")
    parser.add_argument("--checkpoint_file", default=None, help="Optional checkpoint filename override")
    parser.add_argument("--drop_extra_checkpoints", action="store_true", help="Exclude checkpoint snapshots such as *_model_checkpoint* from the exported bundle")
    args = parser.parse_args()

    src_bundle = Path(args.src_bundle).expanduser().resolve()
    if not src_bundle.is_dir():
        raise NotADirectoryError(f"Source bundle does not exist: {src_bundle}")

    dst_root = Path(args.dst_root).expanduser().resolve()
    dst_bundle = dst_root / src_bundle.name
    if args.group is not None:
        dst_bundle = dst_root / args.group / src_bundle.name
    checkpoint_file = args.checkpoint_file or infer_checkpoint_file(src_bundle)

    if load_existing_framework_repos(src_bundle) is None:
        raise RuntimeError(
            f"Source bundle {src_bundle} does not contain a valid training-time framework snapshot in bundle.yaml. "
            "Refusing to regenerate bundle metadata from the current workspace during export."
        )

    copy_filtered_bundle(
        src_bundle,
        dst_bundle,
        main_checkpoint_file=checkpoint_file,
        drop_extra_checkpoints=args.drop_extra_checkpoints,
    )
    manifest = write_bundle_manifest(
        bundle_dir=dst_bundle,
        checkpoint_file=checkpoint_file,
        src_root=args.workspace_src,
        preserve_existing_framework=True,
    )
    print(dst_bundle)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
