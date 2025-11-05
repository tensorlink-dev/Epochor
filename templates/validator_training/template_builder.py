"""Helpers for constructing the validator training Chute template."""

from __future__ import annotations

from typing import Any, List, Optional

from chutes.chute import Chute, NodeSelector
from chutes.image import Image


DEFAULT_PIP_PACKAGES = [
    "torch==2.4.0",
    "safetensors==0.4.4",
    "numpy==1.26.4",
]


def build_validator_training_template(
    username: str,
    *,
    name: str = "epo-validator-training",
    gpu_count: int = 1,
    min_vram_gb_per_gpu: int = 24,
    concurrency: int = 1,
    max_instances: Optional[int] = None,
    timeout_seconds: int = 3900,
    extra_pip: Optional[List[str]] = None,
    python_version: str = "3.11",
    entry_file: str = "trainer_entry.py",
    entry_point: str = "run",
    copy_from: str = "./templates/validator_training",
) -> Chute:
    """Construct a Chute configured for validator-owned training jobs."""

    pip_packages = list(DEFAULT_PIP_PACKAGES)
    if extra_pip:
        pip_packages.extend(extra_pip)

    image = (
        Image(username=username, name=name, tag="latest", python_version=python_version)
        .pip_install(pip_packages)
        .copy_files(copy_from, "/app")
    )

    node_selector = NodeSelector(gpu_count=gpu_count, min_vram_gb_per_gpu=min_vram_gb_per_gpu)

    chute_kwargs: dict[str, Any] = {}
    if max_instances is not None:
        chute_kwargs["max_instances"] = max_instances

    return Chute(
        username=username,
        name=name,
        image=image,
        entry_file=entry_file,
        entry_point=entry_point,
        node_selector=node_selector,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        environment={
            "SUBMISSION_DIR": "/submission",
            "ARTIFACTS_DIR": "/artifacts",
        },
        **chute_kwargs,
    )


__all__ = ["build_validator_training_template", "DEFAULT_PIP_PACKAGES"]
