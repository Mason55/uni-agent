"""SWEBench-specific dataset that injects verl-standard reward fields."""

from __future__ import annotations

import os

from verl.utils.dataset.rl_dataset import RLHFDataset


def _use_local_sandbox() -> bool:
    sandbox_type = os.getenv("SWE_AGENT_SANDBOX_TYPE", "openyuanrong")
    return sandbox_type == "local"


def _remap_image_to_local(image_name: str) -> str:
    parts = image_name.split("/")
    if len(parts) > 1 and "." in parts[0]:
        basename = parts[-1]
    else:
        basename = image_name
    basename = basename.replace("_1776_", "__")
    if ":" in basename:
        basename = basename.rsplit(":", 1)[0]
    return f"{basename}:latest"


def _normalize_env_images(env_config: dict) -> dict:
    if not _use_local_sandbox():
        return env_config

    image = env_config.get("image")
    if image:
        env_config["image"] = _remap_image_to_local(image)

    deployment = env_config.get("deployment")
    if isinstance(deployment, dict):
        dep_image = deployment.get("image")
        if dep_image:
            deployment["image"] = _remap_image_to_local(dep_image)
    return env_config


def extract_image(env_config: dict) -> str:
    """Extract Docker image from env config, supporting both flat and nested formats.

    Flat:   env_config["image"]
    Nested: env_config["deployment"]["image"]
    """
    env_config = _normalize_env_images(env_config)
    image = env_config.get("image")
    if image:
        return image
    deployment = env_config.get("deployment")
    if isinstance(deployment, dict):
        image = deployment.get("image")
        if image:
            return image
    return ""


class SWEBenchDataset(RLHFDataset):

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        extra_info = row_dict.get("extra_info", {})
        tools_kwargs = extra_info.get("tools_kwargs", {})
        env_config = tools_kwargs.get("env")
        if isinstance(env_config, dict):
            _normalize_env_images(env_config)
        reward_config = tools_kwargs.get("reward", {})

        row_dict.setdefault("data_source", reward_config.get("name", "unknown"))
        row_dict.setdefault("reward_model", {"ground_truth": {}})

        return row_dict
