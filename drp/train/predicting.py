import torch
import torch.nn as nn
import torch.distributed as dist

from drp.utils.iterative_mean import Mean
from drp.handlers.logger import Logger
from drp.handlers.checkpoint import CheckPoint
from drp.utils.config import Config
from drp.utils.timer import Timer
from torch.utils.data import DistributedSampler
from drp.metrics.r2_metric import DrpR2Metric
from drp.metrics.scatter_metric import DrpScatterMetricPredict, DrpScatterMetricAllSamples

from drp.train.basemethod import BaseMethod
from drp.metrics.accuracy_at_k import TopKAccuracy, TopKAccuracyTest


class Predicting(BaseMethod):
    def __init__(self,
                 config: Config,
                 rank: int = 0,
                 local_rank: int = 0,
                 sampler_train: DistributedSampler = None,
                 ) -> None:
        """
            Initialize the prediction model with configuration and distributed settings.

            Args:
                config: Configuration object
                rank: Rank of the current process
                local_rank: Local rank of the current process
                sampler_train: Distributed sampler for training data
        """
        super().__init__(config, rank, local_rank, sampler_train)

        # Model setup
        self.backbone = self._create_distributed_model()

        # Loss function
        self.L1_Loss = nn.L1Loss()
        self.MSE =  nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # Metrics setup
        self._setup_metrics()

        # Test loop selection based on task type
        self._configure_test_loop()

    def _configure_test_loop(self):
        """
        Configure the test loop based on task type and tqdm setting.
        """
        if self.config["regression"]:
            inner_method = self.__inner_test_regression
        else:
            inner_method = self.__inner_validate_classification

        self.test_loop = (
            self._tqdm_loop_(inner_method)
            if self.config["tqdm"]
            else self._batch_loop(inner_method)
        )

    def _setup_metrics(self):
        """
        Initialize metrics based on task type.
        """
        if self.config["regression"]:
            self.r2_metric = DrpR2Metric().to(device=self.device)
            self.scatter_metric = DrpScatterMetricPredict().to(device=self.device)
        else:
            self.top_k_accuracy = TopKAccuracyTest(
                num_classes=self.config["num_classes"],
                top_k=(1,)
            ).to(device=self.device)

    def _create_distributed_model(self):
        """
        Create and distribute the model across available GPUs.

        Returns:
            Distributed model
        """
        backbone = self._create_model()
        checkpoint = CheckPoint(
            run_id=self.config["ckp_runid"],
            best_ckp=True,
            finetune_ssl=False,
            finetune_mae=True
        )
        checkpoint.init_model(backbone)

        return self._distribute_model(backbone)

    def _tqdm_loop_(self, function):
        """
        This is a decorator that encapsulates the inner learning process.
        Iterations over all batches of one epoch.
        This decorator displays a progress bar and computes some times.
        """

        def new_function(epoch, loader, description, mean1, mean2):
            if self.rank == 0:  # Initialize tqdm only on GPU 0
                timer = Timer()
                timer.start()
                size_by_batch = len(loader)
                step = max(size_by_batch // self.config["n_steps_by_batch"], 1)

                loss1_tmp, loss2_tmp = [], []  # Initialize empty lists to store loss values

                for batch_idx, batch in enumerate(loader):
                    loss1, loss2 = function(batch, epoch, batch_idx)

                    timer.tick()
                    mean1.update(loss1)
                    mean2.update(loss2)

                    if mean1.iter % step == 0:
                        loss1_tmp.append(mean1.value)
                    if mean2.iter % step == 0:
                        loss2_tmp.append(mean2.value)

                # Log average losses
                self._log_average_losses(epoch, loss1_tmp, loss2_tmp)

                timer.log()
                timer.stop()

            else:  # For other GPUs, run without tqdm progress bar
                for batch_idx, batch in enumerate(loader):
                    loss1, loss2 = function(batch, epoch, batch_idx)
                    mean1.update(loss1)
                    mean2.update(loss2)

        return new_function

    def _log_average_losses(self, epoch, losses1, losses2):
        """
        Log average losses to the logger.

        Args:
            epoch: Current epoch number
            losses1: First type of losses
            losses2: Second type of losses
        """
        if losses1:
            avg_loss1 = sum(losses1) / len(losses1)
            self.logger.report_metric(
                metrics={"L1_avg_loss": avg_loss1},
                epoch=epoch
            )

        if losses2:
            avg_loss2 = sum(losses2) / len(losses2)
            self.logger.report_metric(
                metrics={"L2_avg_loss": avg_loss2},
                epoch=epoch
            )

    def __inner_test_regression(self, batch, epoch, batch_idx):
        """
            Inner method for regression testing.

            Args:
                batch: Input batch
                epoch: Current epoch
                batch_idx: Batch index

            Returns:
                L1 and MSE loss values
            """
        x, y_reg, idx = batch
        x, y, idx = x.to(self.device), y_reg.to(self.device), idx.to(self.device)

        y_hat = self.backbone(x)

        if self.distributed:
            y_hat = self.gather_across_gpus(y_hat)
            y = self.gather_across_gpus(y)
            idx = self.gather_across_gpus(idx)

            loss1 = self.L1_Loss(y_hat, y)
            loss2 = self.MSE(y_hat, y)

            if self.rank == 0:
                self.r2_metric.update(y_hat, y)
                self.scatter_metric.update(y_hat, y, idx)

            dist.all_reduce(loss1, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss2, op=dist.ReduceOp.SUM)

            loss1 = loss1 / dist.get_world_size()
            loss2 = loss2 / dist.get_world_size()
        else:
            loss1 = self.L1_Loss(y_hat, y)
            loss2 = self.MSE(y_hat, y)

            self.r2_metric.update(y_hat, y)
            self.scatter_metric.update(y_hat, y, idx)

        return loss1.item(), loss2.item()

    def __inner_validate_classification(self, batch, epoch, batch_idx):
        x, _, y_class = batch
        x, y = x.to(self.device), y_class.to(self.device)

        y_hat = self.backbone(x)
        loss = self.cross_entropy_loss(y_hat, y)

        if self.distributed:
            y_hat = self.gather_across_gpus(y_hat)
            y = self.gather_across_gpus(y)

            if self.rank == 0:
                self.top_k_accuracy.update(y_hat, y)

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()
        else:
            self.top_k_accuracy.update(y_hat, torch.unsqueeze(y, dim=1))

        return loss.item(), loss.item()

    def test_(self, epoch, valid_loader, valid_loss1, valid_loss2):
        self.backbone.eval()
        with torch.no_grad():
            return self.test_loop(epoch, valid_loader, "Valid", valid_loss1, valid_loss2)

    def predict(self, loader):
        """
        Run prediction on given data loader.

        Args:
            loader: Data loader for prediction
        """
        test_loss1 = Mean.create("ewm")
        test_loss2 = Mean.create("ewm")
        epoch = 0

        self._setup_logger(loader)
        self.test_(epoch=epoch, valid_loader=loader, valid_loss1=test_loss1, valid_loss2=test_loss2)

        if self.distributed:
            dist.barrier()

        if self.rank == 0:
            self._report_results()
            self.logger.close()

        if self.distributed:
            torch.distributed.destroy_process_group()

    def _setup_logger(self, loader):
        """
        Setup logger for the prediction run.

        Args:
            loader: Data loader
        """
        if self.rank == 0:
            self.logger = Logger(self._config, run_id=None)
            self.logger.start()
            self.logger.set_signature(loader)
            self.logger.summary()
        else:
            self.logger = None

    def _report_results(self):
        """
        Report results based on task type.
        """
        if self.config["regression"]:
            fig = self.scatter_metric.compute()
            self.logger.report_figure(fig, "scatter_predict.png")

            metric = self.r2_metric.compute(train_flag="Test")
            self.logger.report_metric(metric)

            self.r2_metric.reset()
            self.scatter_metric.reset()
        else:
            metric = self.top_k_accuracy.compute(train=False)
            self.logger.report_metric(metric)
