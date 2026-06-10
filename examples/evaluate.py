# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import yaml

from pikerag.utils.config_loader import load_dot_env
from pikerag.workflows.evaluate import EvaluationWorkflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="the path of the yaml config file you want to use")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fin:
        # 兼容不同版本的 PyYAML
        try:
            yaml_config: dict = yaml.safe_load(fin, Loader=yaml.FullLoader)
        except TypeError:
            # 旧版本不支持 Loader 参数
            yaml_config: dict = yaml.safe_load(fin)

    load_dot_env(env_path=yaml_config.get("dotenv_path", None))

    workflow = EvaluationWorkflow(yaml_config)
    workflow.run()
