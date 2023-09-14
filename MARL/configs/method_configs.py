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

"""
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

from MARL.configs.base_config import ViewerConfig
from MARL.configs.trainer_config import TrainerConfig

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "basic": "Sample.",
}

method_configs["basic"] = TrainerConfig(
    method_name="basic",
    entity='utokyo-marl',
    project = 'testlog',
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=1000,
    mixed_precision=True,
    viewer=ViewerConfig(),
    vis="wandb"#"tensorboard"#"viewer", # check experiment_config
)
