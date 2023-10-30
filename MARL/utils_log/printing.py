# Copyright 2022 The Nerfstudio Team. All rights reserved.
# Copyright 2023 RTPlayground Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of common strings and print statements used throughout the codebase."""

from math import floor, log

import rich.syntax
import rich.tree
from rich.console import Console
from omegaconf import DictConfig, OmegaConf

from .decorators import check_main_thread

CONSOLE = Console(width=120)


def print_tcnn_speed_warning(method_name: str):
    """Prints a warning about the speed of the TCNN."""
    CONSOLE.line()
    CONSOLE.print(f"[bold yellow]WARNING: Using a slow implementation of {method_name}. ")
    CONSOLE.print(
        "[bold yellow]:person_running: :person_running: "
        + "Install tcnn for speedups :person_running: :person_running:"
    )
    CONSOLE.print("[yellow]pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch")
    CONSOLE.line()


def human_format(num):
    """Format a number in a more human readable way

    Args:
        num: number to format
    Error: num = 0, i.e. in the duration time or number of items.
    """
    units = ["", "K", "M", "B", "T", "P"]
    k = 1000.0
    magnitude = int(floor(log(num, k)))
    return f"{(num / k**magnitude):.2f} {units[magnitude]}"


@check_main_thread
def print_config(
    config: DictConfig,
    # fields: Sequence[str] = (
    #     "trainer",
    #     "model",
    #     "datamodule",
    #     "train",
    #     "callbacks",
    #     "logger",
    #     "seed",
    # ),
    resolve: bool = True,
    save_cfg=True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_cfg:
        with open("config_tree.txt", "w") as fp:
            rich.print(tree, file=fp)