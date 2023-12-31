# -*- coding: utf-8 -*-
from core import Trainer
from core.config import Config
import os
import torch
import sys

sys.dont_write_bytecode = True


def main(rank, config):
    trainer = Trainer(rank, config)
    trainer.train_loop(rank)


if __name__ == "__main__":
    config = Config("./config/deepemd.yaml").get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(
            main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
