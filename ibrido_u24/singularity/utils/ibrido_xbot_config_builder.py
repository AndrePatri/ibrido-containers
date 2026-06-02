#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a YAML mapping")
    return data


def replace_pwd(value: Any, runtime_dir: Path) -> Any:
    if isinstance(value, str):
        return value.replace("$PWD", str(runtime_dir))
    if isinstance(value, list):
        return [replace_pwd(item, runtime_dir) for item in value]
    if isinstance(value, dict):
        return {key: replace_pwd(item, runtime_dir) for key, item in value.items()}
    return value


def runtime_name(xbot_config: Path, impedance_config: Path) -> str:
    token = hashlib.sha1(
        f"{xbot_config.resolve()}::{impedance_config.resolve()}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{xbot_config.stem}.{token}.runtime.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xbot-config", required=True)
    parser.add_argument("--impedance-config", required=True)
    parser.add_argument("--output-dir", default="/tmp/ibrido_xbot_configs")
    args = parser.parse_args()

    xbot_config = Path(args.xbot_config).expanduser().resolve()
    impedance_config = Path(args.impedance_config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not xbot_config.is_file():
        raise SystemExit(f"XBot config not found: {xbot_config}")
    if not impedance_config.is_file():
        raise SystemExit(f"impedance config not found: {impedance_config}")

    runtime = replace_pwd(load_yaml(xbot_config), xbot_config.parent)
    impedance = load_yaml(impedance_config)

    for section in ("motor_pd", "startup_motor_pd"):
        if section not in impedance:
            raise SystemExit(f"{impedance_config} is missing required section '{section}'")
        runtime[section] = impedance[section]

    if "motor_vel" in impedance:
        runtime["motor_vel"] = impedance["motor_vel"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / runtime_name(xbot_config, impedance_config)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(runtime, stream, sort_keys=False)

    print(output_path)


if __name__ == "__main__":
    main()
