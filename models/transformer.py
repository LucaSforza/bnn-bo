import pfns
import torch
from .model import Model
from typing import Optional, Callable, List, Any
from botorch.posteriors import Posterior

from pfns.priors import Batch


# in our convention we name the `num_datasets` -> `batch_size`, and the `num_points_in_each_dataset` -> `seq_len`
def get_batch_for_ridge_regression(
    batch_size=2,
    seq_len=100,
    num_features=1,
    hyperparameters=None,
    device="cpu",
    **kwargs,
):
    if hyperparameters is None:
        hyperparameters = {"a": 0.1, "b": 1.0}
    ws = torch.distributions.Normal(
        torch.zeros(num_features + 1), hyperparameters["b"]
    ).sample((batch_size,))

    xs = torch.rand(batch_size, seq_len, num_features)
    ys = torch.distributions.Normal(
        torch.einsum(
            "nmf, nf -> nm", torch.cat([xs, torch.ones(batch_size, seq_len, 1)], 2), ws
        ),
        hyperparameters["a"],
    ).sample()[..., None]

    # get_batch functions return two different ys, let's come back to this later, though.
    return Batch(x=xs.to(device), y=ys.to(device), target_y=ys.to(device))


from pfns.train import train


from torch import nn
from pfns import utils
from pfns.model import bar_distribution
from pfns.train import (
    train,
    MainConfig,
    OptimizerConfig,
    TransformerConfig,
    BatchShapeSamplerConfig,
)
from pfns.model.encoders import EncoderConfig
from pfns.model.bar_distribution import BarDistributionConfig
from pfns.priors.prior import AdhocPriorConfig
import math


def train_a_pfn(
    get_batch_function,
    epochs=10,
    num_features=10,
    max_dataset_size=10,
    hps=None,
    batch_size=256,
    steps_per_epoch=100,
    device="cpu",
    nhid=1024,
):
    # define a bar distribution (riemann distribution) criterion with 1000 bars
    ys = get_batch_function(100000, 20, num_features, hyperparameters=hps).target_y
    # we define our bar distribution adaptively with respect to the above sample of target ys from our prior
    borders = bar_distribution.get_bucket_borders(num_outputs=1_000, ys=ys).tolist()

    config = MainConfig(
        priors=[
            AdhocPriorConfig(
                get_batch_methods=[get_batch_function],
                prior_kwargs={"num_features": num_features, "hyperparameters": hps},
            )
        ],
        optimizer=OptimizerConfig("adamw", lr=0.0003),
        model=TransformerConfig(
            criterion=BarDistributionConfig(full_support=True, borders=borders),
            emsize=512,
            nhead=8,
            nhid=nhid,
            nlayers=6,
            features_per_group=num_features,
            attention_between_features=False,
            # The encoder config ensures the uniform inputs between 0 and 1 have mean 0 and var 1
            encoder=EncoderConfig(
                constant_normalization_mean=0.5,
                constant_normalization_std=math.sqrt(1 / 12),
            ),
        ),
        batch_shape_sampler=BatchShapeSamplerConfig(
            batch_size=batch_size,
            max_seq_len=max_dataset_size,
            min_num_features=num_features,
            max_num_features=num_features,
        ),
        epochs=epochs,
        warmup_epochs=epochs // 4,
        steps_per_epoch=steps_per_epoch,
        num_workers=0,
    )
    train_result = train(config, device=device, reusable_config=False)
    return train_result


class TransformerModel(Model):
    def __init__(self, args, input_dim, output_dim, device):
        super().__init__()

        # problem dimensions
        self.input_dim = input_dim
        self.problem_output_dim = output_dim

        # architecture
        self.regnet_dims = args["regnet_dims"]
        self.regnet_activation = args["regnet_activation"]
        self.prior_var = args["prior_var"] if "prior_var" in args else 1.0
        self.noise_var = args["noise_var"] if "noise_var" in args else torch.tensor(1.0)
        self.adapt_noise = args["adapt_noise"]
        self.network_output_dim = output_dim
        self.iterative = args["iterative"] if "iterative" in args else True

        # standardize y values before training
        # self.standardize_y = args["standardize_y"]
        self.mean = 0.0
        self.std = 1.0
        self.model = None

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return self.model(X)

    def fit_and_save(self, train_x, train_y, save_dir):
        result = train_a_pfn(
            get_batch_for_ridge_regression,
            num_features=self.input_dim,
            nhid=len(self.regnet_dims),
        )
        self.model = result["model"]
        print("Risultati:\n", result)
