import os
import json
from functools import partial
from datetime import datetime
from omegaconf import OmegaConf

import torch
from torch.optim import Adam, lr_scheduler
import torch.distributed as dist

from tqdm.auto import tqdm
from perfgen.argparse import generate_parser_from_dict

from perfgen.models.compute_image_metrics import (
    fls_score, kid_fid_precision_recall_score)

from ddpm_torch import Trainer, DummyScheduler, ModelWrapper, GaussianDiffusion
from ddpm_torch import UNet, get_dataloader, DATASET_DICT, DATASET_INFO
from ddpm_torch import seed_all, get_param, ConfigDict, get_beta_schedule
from ddpm_torch import Evaluator
from ddpm_torch.ddim import DDIM, get_selection_schedule


class DDPM():
    """
    DDPM style Diffusion.
    Mostly adapted from https://github.com/tqch/ddpm-torch/blob/master/train.py
    """
    def __init__(
            self, num_layers = 5, dim=32, hidden_size=128,
            sampling_timesteps=50,
            beta_schedule='linear',
            eval_batch_size=256,
            eval_total_size=50_000,
            train_device='cpu',
            eval_device='cpu',
            num_accum=1,
            block_size_=1,
            seed=0,
            dry_run=False,
            num_workers=1,
            chkpt_intv=120,
            image_intv=10,
            subseq_size=50,
            chkpt_dir_="./chkpts",
            chkpt_name="",
            image_dir_="./images",  # TODO change this
            skip_schedule="linear",  # TODO quadratic?
            resume=False,  # TODO change to enable pretrained models
            use_ddim=False,
            distributed=False, dataset="cifar10",
            root="/network/datasets/cifar10.var/cifar10_torchvision"):

        self.num_layers = num_layers
        self.dim = dim
        self.hidden_size = hidden_size

        self.sampling_timesteps = sampling_timesteps
        self.beta_schedule = beta_schedule
        self.distributed = distributed

        root = os.path.expanduser(root)
        # TODO rm for submission
        config_dir = "/home/mila/q/quentin.bertrand/ddpm-torch/configs/"
        config_path = os.path.join(config_dir, dataset + ".json")
        with open(config_path, "r") as f:
            meta_config = json.load(f)
        exp_name = os.path.basename(config_path)[:-5]

        # dataset basic info
        dataset = meta_config.get("dataset", dataset)
        in_channels = DATASET_INFO[dataset]["channels"]
        image_res = DATASET_INFO[dataset]["resolution"]
        image_shape = (in_channels, ) + image_res

        # set seed for RNGs
        seed = meta_config.get("seed", seed)
        seed_all(seed)

        ###################################################################
        # extract training-specific hyperparameters
        yaml_config = OmegaConf.load(config_path)
        parser = generate_parser_from_dict(yaml_config)
        args = parser.parse_args()

        gettr = partial(
            get_param, obj_1=meta_config.get("train", {}), obj_2=args.train)
        train_config = ConfigDict(**{
            k: gettr(k) for k in (
                "batch_size", "beta1", "beta2", "lr", "epochs", "grad_norm", "warmup",
                "num_samples", "use_ema", "ema_decay")})
        train_config.batch_size //= num_accum
        # extract diffusion-specific hyperparameters
        getdif = partial(
            get_param, obj_1=meta_config.get("diffusion", {}),
            obj_2=args.diffusion)
        diffusion_config = ConfigDict(**{
            k: getdif(k) for k in (
                "beta_schedule", "beta_start", "beta_end", "timesteps",
                "model_mean_type", "model_var_type", "loss_type")})

        diffusion_config.timesteps = sampling_timesteps

        betas = get_beta_schedule(
            diffusion_config.beta_schedule,
            beta_start=diffusion_config.beta_start,
            beta_end=diffusion_config.beta_end,
            timesteps=diffusion_config.timesteps)
        diffusion = GaussianDiffusion(betas=betas, **diffusion_config)

        # extract model-specific hyperparameters
        out_channels = 2 * in_channels if diffusion_config.model_var_type == "learned" else in_channels
        model_config = meta_config["model"]
        block_size = model_config.pop("block_size", block_size_)
        model_config["in_channels"] = in_channels * block_size ** 2
        model_config["out_channels"] = out_channels * block_size ** 2
        _model = UNet(**model_config)

        if block_size > 1:
            pre_transform = torch.nn.PixelUnshuffle(block_size)  # space-to-depth
            post_transform = torch.nn.PixelShuffle(block_size)  # depth-to-space
            _model = ModelWrapper(_model, pre_transform, post_transform)

        if self.distributed:
            raise NotImplementedError
        else:
            rank = local_rank = 0
            model = _model.to(train_device)

        is_leader = rank == 0  # rank 0: leader in the process group

        self.logger(f"Dataset: {dataset}")
        self.logger(
            f"Effective batch-size is {train_config.batch_size} * {num_accum}"
            f" = {train_config.batch_size * num_accum}.")
        optimizer = Adam(model.parameters(), lr=train_config.lr, betas=(train_config.beta1, train_config.beta2))
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda t: min((t + 1) / train_config.warmup, 1.0)
        ) if train_config.warmup > 0 else None

        split = "all" if dataset == "celeba" else "train"
        trainloader, sampler = get_dataloader(
            dataset, batch_size=train_config.batch_size, split=split, val_size=0., random_seed=seed,
            root=root, drop_last=True, pin_memory=True, num_workers=num_workers, distributed=distributed
        )  # drop_last to have a static input shape; num_workers > 0 to enable asynchronous data loading

        if dry_run:
            self.logger("This is a dry run.")
            chkpt_intv = 1
            image_intv = 1

        chkpt_dir = os.path.join(chkpt_dir_, exp_name)
        self.chkpt_path = os.path.join(chkpt_dir, chkpt_name or f"{exp_name}.pt")

        self.logger(f"Checkpoint will be saved to {os.path.abspath(self.chkpt_path)}", end=" ")
        self.logger(f"every {chkpt_intv} epoch(s)")

        self.image_dir = os.path.join(image_dir_, "train", exp_name)
        self.logger(f"Generated images will be saved to {os.path.abspath(self.image_dir)}", end=" ")
        self.logger(f"every {train_config.image_intv} epoch(s)")

        if is_leader:
            model_config["block_size"] = block_size
            hps = {
                "dataset": dataset,
                "seed": seed,
                "diffusion": diffusion_config,
                "model": model_config,
                "train": train_config
            }
            timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")

            if not os.path.exists(chkpt_dir):
                os.makedirs(chkpt_dir)
            # keep a record of hyperparameter settings used for this experiment run
            with open(os.path.join(chkpt_dir, f"exp_{timestamp}.info"), "w") as f:
                json.dump(hps, f, indent=2)
            if not os.path.exists(self.image_dir):
                os.makedirs(self.image_dir)

        self.trainer = Trainer(
            model=model,
            optimizer=optimizer,
            diffusion=diffusion,
            epochs=train_config.epochs,
            trainloader=trainloader,
            sampler=None,  # do sampler=sampler for distributed training
            scheduler=scheduler,
            num_accum=num_accum,
            use_ema=train_config.use_ema,
            grad_norm=train_config.grad_norm,
            shape=image_shape,
            device=train_device,
            chkpt_intv=chkpt_intv,
            image_intv=image_intv,
            num_samples=train_config.num_samples,
            ema_decay=args.train.ema_decay,
            rank=rank,
            distributed=distributed,
            dry_run=dry_run
        )

        if use_ddim:
            subsequence = get_selection_schedule(
                skip_schedule, size=subseq_size, timesteps=diffusion_config.timesteps)
            diffusion_eval = DDIM.from_ddpm(
                diffusion, eta=0., subsequence=subsequence)
        else:
            diffusion_eval = diffusion

        self.evaluator = Evaluator(
            dataset=dataset,
            diffusion=diffusion_eval,
            eval_batch_size=eval_batch_size,
            eval_total_size=eval_total_size,
            device=eval_device
        )

        if resume:
            try:
                map_location = {"cuda:0": f"cuda:{local_rank}"} if distributed else train_device
                self.trainer.load_checkpoint(
                    self.chkpt_path, map_location=map_location)
            except FileNotFoundError:
                self.logger("Checkpoint file does not exist!")
                self.logger("Starting from scratch...")

        # use cudnn benchmarking algorithm to select the best conv algorithm
        if torch.backends.cudnn.is_available():  # noqa
            torch.backends.cudnn.benchmark = True  # noqa
            self.logger(f"cuDNN benchmark: ON")

    def logger(self, msg, **kwargs):
        if not self.distributed or dist.get_rank() == 0:
            print(msg, **kwargs)

    def train(self, train_loader, n_epochs=200):
        self.logger("Training starts...", flush=True)
        self.trainer.epochs = n_epochs
        self.trainer.trainloader = train_loader
        self.trainer.train(
            self.evaluator, chkpt_path=self.chkpt_path,
            image_dir=self.image_dir)

    @torch.inference_mode()
    def generate(self, n_samples, batchsize_sampling=4096, save_path=None):
        def generate(self, n_samples, batchsize_sampling=4096, save_path=None):
            if n_samples == 0:
                return torch.tensor([])
        self.trainer.model.eval()
        n_batches = n_samples // batchsize_sampling
        list_samples = []
        for batch in tqdm(range(n_batches), desc='Generation loop'):
            samples = self.trainer.sample_fn(
                sample_size=batchsize_sampling,
                sample_seed=self.trainer.sample_seed).cpu()
            list_samples.append(samples)
        samples = self.trainer.sample_fn(
                sample_size=n_samples % batchsize_sampling,
                sample_seed=self.trainer.sample_seed).cpu()
        list_samples.append(samples)

        samples = torch.vstack(list_samples)
        self.trainer.model.train()
        return samples.detach()
        # x = self.sample_fn(sample_size=self.num_samples, sample_seed=self.sample_seed).cpu()

    @torch.inference_mode()
    def eval(self, train_loader, test_loader, gen_data_tensor):
        # raise NotImplementedError
        with torch.no_grad():
            metrics = {}
            train_dataset = train_loader.dataset
            test_dataset = test_loader.dataset
            gen_dataset = torch.utils.data.TensorDataset(gen_data_tensor)
            # metrics['FLS']  = fls_score(
            #     train_dataset, test_dataset, gen_data_tensor.cpu().detach())
            metrics['KID'], metrics['FID'], metrics['Precision'], metrics['Recall']  = kid_fid_precision_recall_score(
                train_dataset, test_dataset, gen_data_tensor.cpu().detach())
        return metrics

    def log_prob(self, data):
        raise NotImplementedError

    def load(self, path):
        self.trainer.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.trainer.model.state_dict(), path)

    def cold_start(self):
        raise NotImplementedError
