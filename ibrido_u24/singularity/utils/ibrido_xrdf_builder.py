#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import shlex
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def split_words(value: str | None) -> list[str]:
    if not value:
        return []
    return shlex.split(value)


def custom_xacro_args(
    names_s: str | None, dtypes_s: str | None, vals_s: str | None
) -> list[str]:
    names = split_words(names_s)
    dtypes = split_words(dtypes_s)
    vals = split_words(vals_s)
    if not names and not dtypes and not vals:
        return []
    if not (len(names) == len(dtypes) == len(vals)):
        raise SystemExit("custom arg names, dtypes, and values have different lengths")
    return [
        f"{name}:={val}"
        for name, dtype, val in zip(names, dtypes, vals)
        if dtype == "xacro"
    ]


def merge_xacro_cmds(prev_cmds: list[str], new_cmds: list[str]) -> list[str]:
    merged = {}
    for cmd in prev_cmds + new_cmds:
        if ":=" not in cmd:
            raise SystemExit(f"invalid xacro command: {cmd}")
        name, value = cmd.split(":=", 1)
        merged[name.strip()] = value.strip()
    return [f"{name}:={value}" for name, value in merged.items()]


def default_xrdf_cmds(urdf_descr_root_path: str) -> list[str]:
    root = urdf_descr_root_path
    root_l = root.lower()

    if "centauro" in root_l:
        return [
            "legs:=true",
            "big_wheel:=true",
            "upper_body:=true",
            "battery:=true",
            "velodyne:=false",
            "realsense:=false",
            "floating_joint:=false",
            "use_abs_mesh_paths:=true",
            "end_effector_left:=ball",
            "end_effector_right:=ball",
            f"root:={root}",
        ]

    if "kyon" in root_l:
        return [
            "wheels:=false",
            "upper_body:=false",
            "dagana:=false",
            "sensors:=false",
            "floating_joint:=false",
            "payload:=false",
            "use_abs_mesh_paths:=true",
            f"root:={root}",
        ]

    if "b2w" in root_l:
        return [
            "use_abs_mesh_paths:=true",
            "floating_joint:=false",
            f"root:={root}",
        ]

    if "talos" in root_l:
        talos_repo_root = str(Path(root).parent)
        ws_src_root = str(Path(talos_repo_root).parent)
        return [
            "foot_collision:=thinbox",
            "head_type:=default",
            "flexibility:=False",
            "test:=false",
            "use_fixed_base:=false",
            "use_sim:=false",
            "enable_crane:=false",
            "disable_gazebo_camera:=true",
            "use_capsule_collision:=false",
            "multiple:=false",
            "gazebo_version:=classic",
            "include_gazebo:=false",
            "include_ros2_control:=false",
            "include_head_sensors:=false",
            "include_torso_imu:=false",
            "include_grippers:=false",
            "floating_joint:=false",
            "use_abs_mesh_paths:=true",
            "use_local_filesys_for_meshes:=false",
            f"root:={root}",
            f"talos_description_inertial_root:={talos_repo_root}/talos_description_inertial",
            f"talos_description_calibration_root:={talos_repo_root}/talos_description_calibration",
            f"pal_urdf_utils_root:={ws_src_root}/pal_urdf_utils",
        ]

    raise SystemExit(f"unsupported robot description root for XRDF generation: {root}")


def robot_name(value: str | None) -> str:
    name = value or "xbot_robot"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def xml_root(path: Path) -> str:
    try:
        return ET.parse(path).getroot().tag
    except ET.ParseError as exc:
        raise SystemExit(f"invalid XML generated at {path}: {exc}") from exc


def copy_xml(src: Path, dst: Path, expected_root: str) -> Path:
    if not src.is_file():
        raise SystemExit(f"XRDF input not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if xml_root(dst) != expected_root:
        raise SystemExit(f"{dst} root is not <{expected_root}>")
    return dst


def generated_path(kind: str, src: Path, dump_dir: Path, name: str, cmds: list[str]) -> Path:
    if src.suffix == ".xacro":
        if kind == "urdf":
            generated = generate_xrdf("urdf", src, dump_dir / f"{name}.urdf", cmds)
        elif kind == "srdf":
            generated = generate_xrdf("srdf", src, dump_dir / f"{name}.srdf", cmds)
        else:
            raise AssertionError(kind)
        return generated

    suffix = ".urdf" if kind == "urdf" else ".srdf"
    return copy_xml(src, dump_dir / f"{name}{suffix}", "robot")


def xacro_binary() -> str:
    found = shutil.which("xacro")
    if found:
        return found
    for candidate in (
        Path("/opt/conda/envs/ibrido/bin/xacro"),
        Path("/opt/conda/envs/ibrido_isaac_py11/bin/xacro"),
        Path("/opt/ros/jazzy/bin/xacro"),
    ):
        if candidate.is_file():
            return str(candidate)
    raise SystemExit("xacro executable not found")


def generate_xrdf(kind: str, src: Path, dst: Path, cmds: list[str]) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = [xacro_binary(), str(src), *cmds, "-o", str(dst)]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"failed to generate {kind}: {' '.join(command)}") from exc
    if xml_root(dst) != "robot":
        raise SystemExit(f"{dst} root is not <robot>")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf-path", required=True)
    parser.add_argument("--srdf-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--robot-name")
    parser.add_argument("--custom-args-names")
    parser.add_argument("--custom-args-dtype")
    parser.add_argument("--custom-args-vals")
    args = parser.parse_args()

    urdf_path = Path(args.urdf_path).expanduser().resolve()
    srdf_path = Path(args.srdf_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    name = robot_name(args.robot_name)

    if not urdf_path.is_file():
        raise SystemExit(f"URDF/Xacro path not found: {urdf_path}")
    if not srdf_path.is_file():
        raise SystemExit(f"SRDF/Xacro path not found: {srdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    descr_root = urdf_path.parent.parent if len(urdf_path.parts) > 2 else urdf_path.parent
    cmds = merge_xacro_cmds(
        prev_cmds=default_xrdf_cmds(urdf_descr_root_path=str(descr_root)),
        new_cmds=custom_xacro_args(
            args.custom_args_names, args.custom_args_dtype, args.custom_args_vals
        ),
    )

    generated_urdf = generated_path("urdf", urdf_path, output_dir, name, cmds)
    generated_srdf = generated_path("srdf", srdf_path, output_dir, name, cmds)
    print(generated_urdf)
    print(generated_srdf)


if __name__ == "__main__":
    main()
