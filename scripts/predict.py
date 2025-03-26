from drp.train.predicting import Predicting
from drp.data.dataset import Drp3dMMapDataset
from drp.utils.config import Config
from drp.utils.arguments import parse_args
import os
from torch.utils.data import DataLoader
import torch
from drp.utils.hostlist import expand_hostlist

from drp.utils.ddp_setup import setup, prepare_dataset


def main(
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 0,
        config: Config = None
):

    if config.range:
        test_dataset = Drp3dMMapDataset(config, train_flag="Train", type_model="idx")
    else:
        test_dataset = Drp3dMMapDataset(config, train_flag="Train", type_model="classification")

    if config.distributed:
        hostnames = expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
        setup(rank, world_size, IP_ad=hostnames[0])  # initialize ddp
        test_dataloader, _, _, train_sampler = prepare_dataset(
            test_dataset, test_dataset, test_dataset, config, rank, world_size
        )
        trainer = Predicting(
            config=config,
            rank=rank,
            local_rank=local_rank,
            sampler_train=train_sampler,
        )
    else:

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            persistent_workers=True,
        )

        trainer = Predicting(config=config)
    trainer.predict(
        loader=test_dataloader)


if __name__ == "__main__":

    args = parse_args()
    config = Config.create_from_args(args)
    torch.manual_seed(config.seed)
    torch.rand(config.seed).numpy()
    # get SLURM variables
    if config.distributed:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        main(rank=rank, local_rank=local_rank, world_size=world_size, config=config)
    else:
        main(config=config)





