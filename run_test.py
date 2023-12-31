# -*- coding: utf-8 -*-
from core import Test
from core.config import Config
import torch
import os
import sys

sys.dont_write_bytecode = True


PATH = "/root/autodl-tmp/dsy/results/DeepEMD-miniImageNet--ravi-resnet12emd-5-1-Dec-31-2023-13-58-28"
VAR_DICT = {
    "test_epoch": 5,
    "device_ids": "0",
    "n_gpu": 1,
    "test_episode": 600,
    "episode_size": 1,
}


def main(rank, config):
    test = Test(rank, config, PATH)
    test.test_loop()


if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"),
                    VAR_DICT).get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(
            main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
