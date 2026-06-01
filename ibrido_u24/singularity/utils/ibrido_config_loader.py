#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import sys
from collections import OrderedDict
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("ibrido_config_loader.py requires PyYAML") from exc

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def scalar_to_string(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def expand_value(value, env):
    text = scalar_to_string(value)
    for _ in range(8):
        new_text = _VAR_RE.sub(lambda m: scalar_to_string(env.get(m.group(1) or m.group(2), "")), text)
        if new_text == text:
            return new_text
        text = new_text
    return text


def merge_custom_arg(base, name, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"custom arg '{name}' must be a mapping")
    if "dtype" not in spec or "value" not in spec:
        raise ValueError(f"custom arg '{name}' requires dtype and value")
    base["custom_args"][name] = {
        "dtype": scalar_to_string(spec["dtype"]),
        "value": scalar_to_string(spec["value"]),
    }


def merge_config(base, overlay):
    for key, value in overlay.get("vars", {}).items():
        base["vars"][key] = scalar_to_string(value)

    for key, spec in overlay.get("custom_args", {}).items():
        if not isinstance(spec, dict):
            raise ValueError(f"custom arg '{key}' must be a mapping")
        if "dtype" in spec or "value" in spec:
            merge_custom_arg(base, key, spec)
            continue
        for arg_name, arg_spec in spec.items():
            merge_custom_arg(base, arg_name, arg_spec)


def load_one(path, seen, stack):
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic config include detected at {path}")
    seen.add(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")

    result = {"vars": OrderedDict(), "custom_args": OrderedDict()}
    includes = data.get("include", [])
    if isinstance(includes, str):
        includes = [includes]
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        child = load_one(include_path, seen, stack)
        merge_config(result, child)

    merge_config(result, data)
    stack.append(path)
    seen.remove(path)
    return result


def resolve_config(path):
    stack = []
    cfg = load_one(Path(path), set(), stack)
    env = dict(os.environ)
    env.update(cfg["vars"])

    original_env = dict(os.environ)
    for _ in range(8):
        changed = False
        for key, value in list(cfg["vars"].items()):
            expansion_env = dict(env)
            if key in original_env:
                expansion_env[key] = original_env[key]
            else:
                expansion_env.pop(key, None)
            expanded = expand_value(value, expansion_env)
            if expanded != value:
                cfg["vars"][key] = expanded
                env[key] = expanded
                changed = True
        if not changed:
            break

    for spec in cfg["custom_args"].values():
        spec["dtype"] = expand_value(spec["dtype"], env)
        spec["value"] = expand_value(spec["value"], env)

    cfg["vars"]["CUSTOM_ARGS_NAMES"] = " ".join(cfg["custom_args"].keys())
    cfg["vars"]["CUSTOM_ARGS_DTYPE"] = " ".join(spec["dtype"] for spec in cfg["custom_args"].values())
    cfg["vars"]["CUSTOM_ARGS_VALS"] = " ".join(spec["value"] for spec in cfg["custom_args"].values())
    cfg["vars"]["IBRIDO_CFG_STACK"] = ":".join(str(p) for p in stack)
    return cfg


def print_shell(cfg):
    for key, value in cfg["vars"].items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise ValueError(f"invalid environment variable name: {key}")
        print(f"export {key}={shlex.quote(scalar_to_string(value))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--shell", action="store_true", help="emit shell export commands")
    args = parser.parse_args()

    try:
        cfg = resolve_config(args.config)
        if args.shell:
            print_shell(cfg)
        else:
            yaml.safe_dump({"vars": dict(cfg["vars"]), "custom_args": dict(cfg["custom_args"])}, sys.stdout, sort_keys=False)
    except Exception as exc:
        raise SystemExit(f"ibrido_config_loader.py: {exc}") from exc


if __name__ == "__main__":
    main()
