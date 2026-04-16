import copy
import math
import logging
import dataclasses
from collections import OrderedDict
import typing as tp
from pathlib import Path
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf
from tqdm import tqdm

from url_benchmark import utils
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from .ddpg import MetaDict, make_aug_encoder
from .fb_modules import (
    Actor,
    DiagGaussianActor,
    ForwardMap,
    BackwardMap,
    mlp,
    OnlineCov,
)
from url_benchmark.dmc import TimeStep
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import os
from collections import OrderedDict


logger = logging.getLogger(__name__)



import torch.nn.functional as F



def match_sim_target(sim_obs, targ_obs, sim_reward, targ_reward, obs_threshold=15.0):
    """
    Match simulation and target samples based on observation similarity.
    
    Args:
        sim_obs (Tensor): Simulation observations of shape [N, d].
        targ_obs (Tensor): Target observations of shape [M, d].
        sim_reward (Tensor): Simulation rewards of shape [N, 1].
        targ_reward (Tensor): Target rewards of shape [M, 1].
        obs_threshold (float): Threshold for matching observations.
    
    Returns:
        matched_sim_reward (Tensor): Simulation rewards for matched samples.
        matched_targ_reward (Tensor): Target rewards corresponding to the matched simulation samples.
    """
    # Compute pairwise L2 distances between simulation and target observations.
    distances = torch.cdist(sim_obs, targ_obs, p=2)  # [N, M]
    # For each simulation sample, find the closest target sample.
    min_vals, min_indices = torch.min(distances, dim=1)  # [N]
    # Select simulation samples with distance below the threshold.
    matched_mask = min_vals < obs_threshold
    matched_sim_reward = sim_reward[matched_mask]
    matched_targ_reward = targ_reward[min_indices[matched_mask]]
    return matched_sim_reward, matched_targ_reward

def train_reward_adapter_with_matching(adapter, optimizer, sim_obs, sim_reward, targ_obs, targ_reward, obs_threshold=10.0, num_epochs=50, batch_size=128):
    """
    Train the reward adapter using matched simulation and target rewards based on observation similarity.
    
    Args:
        adapter (nn.Module): The reward adapter module.
        optimizer (torch.optim.Optimizer): Optimizer for the adapter.
        sim_obs (Tensor): Simulation observations, shape [N, d].
        sim_reward (Tensor): Simulation rewards, shape [N, 1].
        targ_obs (Tensor): Target observations, shape [M, d].
        targ_reward (Tensor): Target rewards, shape [M, 1].
        obs_threshold (float): Matching threshold for observations.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
    
    Returns:
        loss_history (list): List of loss values per epoch.
    """
    matched_sim_reward, matched_targ_reward = match_sim_target(sim_obs, targ_obs, sim_reward, targ_reward, obs_threshold)
    dataset_size = matched_sim_reward.shape[0]
    print("Number of matched samples for adapter training:", dataset_size)
    if dataset_size == 0:
        print("No matched samples found, cannot train adapter.")
        return []
    
    loss_history = []
    for epoch in range(num_epochs):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i: i + batch_size]
            batch_sim_reward = matched_sim_reward[indices]
            batch_targ_reward = matched_targ_reward[indices]
            optimizer.zero_grad()
            # Compute adapted reward: f(r) = a * r + b
            batch_adapted_reward = adapter(batch_sim_reward)
            loss = ((batch_adapted_reward - batch_targ_reward) ** 2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        epoch_loss /= num_batches
        loss_history.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Adapter Loss: {epoch_loss:.4f}")
    return loss_history


def contrastive_loss(phi, phi_targ, margin=1.0):
    # Calculate pairwise distances
    distances = torch.norm(phi - phi_targ, p=2, dim=1)

    # Contrastive loss computation
    loss = torch.mean(
        (distances**2) * (distances <= margin).float() +
        (margin - distances).clamp(min=0)**2 * (distances > margin).float()
    )
    return loss

def vis_adaptation(reward, reward_adapted, save_dir):
    import matplotlib.pyplot as plt
    import os

    reward_np = reward.squeeze().cpu().numpy()          # shape: [N]
    reward_adapted_np = reward_adapted.squeeze().cpu().numpy()  # shape: [N]

    # Create a scatter plot for comparison
    indices = range(len(reward_np))
    plt.figure(figsize=(10, 6))
    plt.scatter(indices, reward_np, color='blue', label='Original Reward')
    plt.scatter(indices, reward_adapted_np, color='red', label='Adapted Reward')
    plt.xlabel("Index")
    plt.ylabel("Reward Value")
    plt.title("Comparison of Original and Adapted Rewards")
    plt.legend()
    plt.grid(True)

    # Save the figure to the specified directory
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "reward_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

@dataclasses.dataclass
class SFAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.sf.SFAgent"
    name: str = "sf"
    obs_type: str = omegaconf.MISSING  # to be specified later
    image_wh: int = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    lr_coef: float = 5
    sf_target_tau: float = 0.01  # 0.001-0.01
    update_every_steps: int = 1
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING  # ???  # to be specified later
    num_inference_steps: int = 10000
    hidden_dim: int = 1024  # 128, 2048
    phi_hidden_dim: int = 512  # 128, 2048
    feature_dim: int = 512  # 128, 1024
    z_dim: int = 50  # 30-200
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)"  # 0,  0.1, 0.2
    stddev_clip: float = 0.3  # 1
    update_z_every_step: int = 300
    nstep: int = 1
    batch_size: int = 1024
    init_sf: bool = True
    update_encoder: bool = omegaconf.II("update_encoder")  # ${update_encoder}
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)  # param for DiagGaussianActor
    temp: float = 1  # temperature for DiagGaussianActor
    boltzmann: bool = False  # set to true for DiagGaussianActor
    debug: bool = False
    preprocess: bool = True
    num_sf_updates: int = 1
    feature_learner: str = "hilp"
    mix_ratio: float = 0.5
    q_loss: bool = True
    update_cov_every_step: int = 1000
    add_trunk: bool = False

    feature_type: str = "state"  # 'state', 'diff', 'concat'
    hilp_discount: float = 0.98
    hilp_expectile: float = 0.5


cs = ConfigStore.instance()
cs.store(group="agent", name="sf", node=SFAgentConfig)


class FeatureLearner(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        self.feature_net: nn.Module = mlp(
            obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2"
        )
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        return None


class Identity(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.feature_net = nn.Identity()


class HILP(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, cfg) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.z_dim = z_dim
        self.cfg = cfg

        if self.cfg.feature_type != "concat":
            feature_dim = z_dim
        else:
            assert z_dim % 2 == 0
            feature_dim = z_dim // 2

        layers = [obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", feature_dim]

        self.phi1 = mlp(*layers)
        self.phi2 = mlp(*layers)
        self.target_phi1 = mlp(*layers)
        self.target_phi2 = mlp(*layers)
        self.target_phi1.load_state_dict(self.phi1.state_dict())
        self.target_phi2.load_state_dict(self.phi2.state_dict())

        self.apply(utils.weight_init)

        # Define a running mean and std
        self.register_buffer("running_mean", torch.zeros(feature_dim))
        self.register_buffer("running_std", torch.ones(feature_dim))

    def feature_net(self, obs):
        phi = self.phi1(obs)
        phi = phi - self.running_mean
        return phi

    def value(self, obs: torch.Tensor, goals: torch.Tensor, is_target: bool = False):
        if is_target:
            phi1 = self.target_phi1
            phi2 = self.target_phi2
        else:
            phi1 = self.phi1
            phi2 = self.phi2

        phi1_s = phi1(obs)
        phi1_g = phi1(goals)

        phi2_s = phi2(obs)
        phi2_g = phi2(goals)

        squared_dist1 = ((phi1_s - phi1_g) ** 2).sum(dim=-1)
        v1 = -torch.sqrt(torch.clamp(squared_dist1, min=1e-6))
        squared_dist2 = ((phi2_s - phi2_g) ** 2).sum(dim=-1)
        v2 = -torch.sqrt(torch.clamp(squared_dist2, min=1e-6))

        if is_target:
            v1 = v1.detach()
            v2 = v2.detach()

        return v1, v2

    def expectile_loss(self, adv, diff, expectile=0.7):
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        goals = future_obs
        rewards = (torch.linalg.norm(obs - goals, dim=-1) < 1e-6).float()
        masks = 1.0 - rewards
        rewards = rewards - 1.0

        next_v1, next_v2 = self.value(next_obs, goals, is_target=True)
        next_v = torch.minimum(next_v1, next_v2)
        q = rewards + self.cfg.hilp_discount * masks * next_v

        v1_t, v2_t = self.value(obs, goals, is_target=True)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = rewards + self.cfg.hilp_discount * masks * next_v1
        q2 = rewards + self.cfg.hilp_discount * masks * next_v2
        v1, v2 = self.value(obs, goals, is_target=False)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.cfg.hilp_expectile).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.cfg.hilp_expectile).mean()
        value_loss = value_loss1 + value_loss2

        utils.soft_update_params(self.phi1, self.target_phi1, 0.005)
        utils.soft_update_params(self.phi2, self.target_phi2, 0.005)

        with torch.no_grad():
            phi1 = self.phi1(obs)
            self.running_mean = 0.995 * self.running_mean + 0.005 * phi1.mean(dim=0)
            self.running_std = 0.995 * self.running_std + 0.005 * phi1.std(dim=0)

        return value_loss, {
            "hilp/value_loss": value_loss,
            "hilp/v_mean": v.mean(),
            "hilp/v_max": v.max(),
            "hilp/v_min": v.min(),
            "hilp/abs_adv_mean": torch.abs(adv).mean(),
            "hilp/adv_mean": adv.mean(),
            "hilp/adv_max": adv.max(),
            "hilp/adv_min": adv.min(),
            "hilp/accept_prob": (adv >= 0).float().mean(),
        }


class Laplacian(FeatureLearner):
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        del action
        del future_obs
        phi = self.feature_net(obs)
        next_phi = self.feature_net(next_obs)
        loss = (phi - next_phi).pow(2).mean()
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        return loss


class ContrastiveFeature(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        del action
        del next_obs
        assert future_obs is not None
        phi = self.feature_net(obs)
        future_mu = self.mu_net(future_obs)
        phi = F.normalize(phi, dim=1)
        future_mu = F.normalize(future_mu, dim=1)
        logits = torch.einsum("sd, td-> st", phi, future_mu)  # batch x batch
        I = torch.eye(*logits.size(), device=logits.device)
        off_diag = ~I.bool()
        logits_off_diag = logits[off_diag].reshape(logits.shape[0], logits.shape[0] - 1)
        loss = -logits.diag() + torch.logsumexp(logits_off_diag, dim=1)
        loss = loss.mean()
        return loss


class ContrastiveFeaturev2(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        del action
        del next_obs
        assert future_obs is not None
        future_phi = self.feature_net(future_obs)
        mu = self.mu_net(obs)
        future_phi = F.normalize(future_phi, dim=1)
        mu = F.normalize(mu, dim=1)
        logits = torch.einsum("sd, td-> st", mu, future_phi)  # batch x batch
        I = torch.eye(*logits.size(), device=logits.device)
        off_diag = ~I.bool()
        logits_off_diag = logits[off_diag].reshape(logits.shape[0], logits.shape[0] - 1)
        loss = -logits.diag() + torch.logsumexp(logits_off_diag, dim=1)
        loss = loss.mean()
        return loss


class ICM(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.inverse_dynamic_net = mlp(
            2 * z_dim, hidden_dim, "irelu", hidden_dim, "irelu", action_dim, "tanh"
        )
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        del future_obs
        phi = self.feature_net(obs)
        next_phi = self.feature_net(next_obs)
        predicted_action = self.inverse_dynamic_net(torch.cat([phi, next_phi], dim=-1))
        backward_error = (action - predicted_action).pow(2).mean()
        icm_loss = backward_error
        return icm_loss


class TransitionModel(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.forward_dynamic_net = mlp(
            z_dim + action_dim, hidden_dim, "irelu", hidden_dim, "irelu", obs_dim
        )
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        del future_obs
        phi = self.feature_net(obs)
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (predicted_next_obs - next_obs).pow(2).mean()
        return forward_error


class TransitionLatentModel(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.forward_dynamic_net = mlp(
            z_dim + action_dim, hidden_dim, "irelu", hidden_dim, "irelu", z_dim
        )
        self.target_feature_net = mlp(
            obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2"
        )
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        del future_obs
        phi = self.feature_net(obs)
        with torch.no_grad():
            next_phi = self.target_feature_net(next_obs)
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (predicted_next_obs - next_phi.detach()).pow(2).mean()
        utils.soft_update_params(self.feature_net, self.target_feature_net, 0.01)

        return forward_error


class AutoEncoder(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.decoder = mlp(z_dim, hidden_dim, "irelu", hidden_dim, "irelu", obs_dim)
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        del future_obs
        del next_obs
        del action
        phi = self.feature_net(obs)
        predicted_obs = self.decoder(phi)
        reconstruction_error = (predicted_obs - obs).pow(2).mean()
        return reconstruction_error


class SVDSR(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.target_feature_net = mlp(
            obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2"
        )
        self.target_mu_net = mlp(
            obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim
        )
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        del future_obs
        phi = self.feature_net(obs)
        mu = self.mu_net(next_obs)
        SR = torch.einsum("sd, td -> st", phi, mu)
        with torch.no_grad():
            target_phi = self.target_feature_net(next_obs)
            target_mu = self.target_mu_net(next_obs)
            target_SR = torch.einsum("sd, td -> st", target_phi, target_mu)

        I = torch.eye(*SR.size(), device=SR.device)
        off_diag = ~I.bool()
        loss = (
            -2 * SR.diag().mean()
            + (SR - 0.99 * target_SR.detach())[off_diag].pow(2).mean()
        )

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        utils.soft_update_params(self.feature_net, self.target_feature_net, 0.01)
        utils.soft_update_params(self.mu_net, self.target_mu_net, 0.01)

        return loss


class SVDSRv2(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.target_feature_net = mlp(
            obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2"
        )
        self.target_mu_net = mlp(
            obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim
        )
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        del future_obs
        phi = self.feature_net(next_obs)
        mu = self.mu_net(obs)
        SR = torch.einsum("sd, td -> st", mu, phi)
        with torch.no_grad():
            target_phi = self.target_feature_net(next_obs)
            target_mu = self.target_mu_net(next_obs)
            target_SR = torch.einsum("sd, td -> st", target_mu, target_phi)

        I = torch.eye(*SR.size(), device=SR.device)
        off_diag = ~I.bool()
        loss = (
            -2 * SR.diag().mean()
            + (SR - 0.98 * target_SR.detach())[off_diag].pow(2).mean()
        )

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        utils.soft_update_params(self.feature_net, self.target_feature_net, 0.01)
        utils.soft_update_params(self.mu_net, self.target_mu_net, 0.01)

        return loss



# Define a simple reward adapter that performs a linear mapping: f(r) = a * r + b.
class RewardAdapter(nn.Module):
    def __init__(self):
        super(RewardAdapter, self).__init__()
        # Initialize a and b as learnable parameters.
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
    
    def forward(self, r):
        # r: Tensor of shape [batch_size, 1]
        return self.a * r + self.b


class SVDP(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(
            obs_dim + action_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim
        )
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
    ):
        del future_obs
        phi = self.feature_net(next_obs)
        mu = self.mu_net(torch.cat([obs, action], dim=1))
        P = torch.einsum("sd, td -> st", mu, phi)
        I = torch.eye(*P.size(), device=P.device)
        off_diag = ~I.bool()
        loss = -2 * P.diag().mean() + P[off_diag].pow(2).mean()

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        return loss


class SFAgent:
    def __init__(self, **kwargs: tp.Any):
        cfg = SFAgentConfig(**kwargs)
        self.cfg = cfg
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]
        self.solved_meta: tp.Any = None

        # models
        if cfg.obs_type == "pixels":
            self.aug, self.encoder = make_aug_encoder(cfg)
            self.obs_dim = self.encoder.repr_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = cfg.obs_shape[0]
        if cfg.feature_learner == "identity":
            cfg.z_dim = self.obs_dim
            self.cfg.z_dim = self.obs_dim
        # create the network
        if self.cfg.boltzmann:
            self.actor: nn.Module = DiagGaussianActor(
                cfg.obs_type,
                self.obs_dim,
                cfg.z_dim,
                self.action_dim,
                cfg.hidden_dim,
                cfg.log_std_bounds,
            ).to(cfg.device)
        else:
            self.actor = Actor(
                self.obs_dim,
                cfg.z_dim,
                self.action_dim,
                cfg.feature_dim,
                cfg.hidden_dim,
                preprocess=cfg.preprocess,
                add_trunk=self.cfg.add_trunk,
            ).to(cfg.device)
        self.successor_net = ForwardMap(
            self.obs_dim,
            cfg.z_dim,
            self.action_dim,
            cfg.feature_dim,
            cfg.hidden_dim,
            preprocess=cfg.preprocess,
            add_trunk=self.cfg.add_trunk,
        ).to(cfg.device)
        # build up the target network
        self.successor_target_net = ForwardMap(
            self.obs_dim,
            cfg.z_dim,
            self.action_dim,
            cfg.feature_dim,
            cfg.hidden_dim,
            preprocess=cfg.preprocess,
            add_trunk=self.cfg.add_trunk,
        ).to(cfg.device)

        learner = dict(
            icm=ICM,
            transition=TransitionModel,
            latent=TransitionLatentModel,
            contrastive=ContrastiveFeature,
            autoencoder=AutoEncoder,
            lap=Laplacian,
            random=FeatureLearner,
            svd_sr=SVDSR,
            svd_p=SVDP,
            contrastivev2=ContrastiveFeaturev2,
            svd_srv2=SVDSRv2,
            identity=Identity,
            hilp=HILP,
        )[self.cfg.feature_learner]
        extra_kwargs = dict()
        if self.cfg.feature_learner == "hilp":
            extra_kwargs = dict(
                cfg=self.cfg,
            )
        self.feature_learner = learner(
            self.obs_dim, self.action_dim, cfg.z_dim, cfg.phi_hidden_dim, **extra_kwargs
        ).to(cfg.device)

        # load the weights into the target networks
        self.successor_target_net.load_state_dict(self.successor_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.obs_type == "pixels":
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.sf_opt = torch.optim.Adam(self.successor_net.parameters(), lr=cfg.lr)
        self.phi_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.feature_learner not in ["random", "identity"]:
            self.phi_opt = torch.optim.Adam(
                self.feature_learner.parameters(), lr=cfg.lr_coef * cfg.lr
            )
        self.train()
        self.successor_target_net.train()

        self.inv_cov = torch.eye(
            self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device
        )

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.encoder, self.actor, self.successor_net]:
            net.train(training)
        if self.phi_opt is not None:
            self.feature_learner.train()

    def init_from(self, other) -> None:
        # copy parameters over
        names = ["encoder", "actor"]
        if self.cfg.init_sf:
            names += ["successor_net", "feature_learner", "successor_target_net"]
        for name in names:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def get_goal_meta(
        self, goal_array: np.ndarray, obs_array: np.ndarray = None
    ) -> MetaDict:
        assert self.cfg.feature_learner == "hilp"

        obs = torch.tensor(obs_array).unsqueeze(0).to(self.cfg.device)
        desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.cfg.device)

        with torch.no_grad():
            obs = self.encoder(obs)
            desired_goal = self.encoder(desired_goal)

        with torch.no_grad():
            z_g = self.feature_learner.feature_net(desired_goal)
            z_s = self.feature_learner.feature_net(obs)

        z = z_g - z_s
        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta["z"] = z
        return meta
    


    # add the obs_next - obs = delta
    def infer_meta_from_obs_and_rewards_sim2real(
        self, 
        obs: torch.Tensor, 
        reward: torch.Tensor, 
        next_obs: torch.Tensor, 
        obs_targ: torch.Tensor,
        reward_targ: torch.Tensor,
        next_obs_targ: torch.Tensor,
        vis: bool = False,
        lam: float = 5.0  # Hyperparameter to balance target loss (tunable): 0.1 - 10
    ):
        import math
        import torch.nn.functional as F

        # Encode simulation and target observations and next-observations.
        with torch.no_grad():
            sim_obs_enc = self.encoder(obs)
            sim_next_obs_enc = self.encoder(next_obs)
            targ_obs_enc = self.encoder(obs_targ)
            targ_next_obs_enc = self.encoder(next_obs_targ)
        
        # Compute delta features (differences between next and current state representations)
        with torch.no_grad():
            if self.cfg.feature_type in ["state", "diff"]:
                delta_phi_sim = self.feature_learner.feature_net(sim_next_obs_enc) - self.feature_learner.feature_net(sim_obs_enc)
                delta_phi_targ = self.feature_learner.feature_net(targ_next_obs_enc) - self.feature_learner.feature_net(targ_obs_enc)
                # Debug print to verify branch selection
                print("Finished computing delta features with type:", self.cfg.feature_type)
            else:
                phi_sim_obs = self.feature_learner.feature_net(sim_obs_enc)
                phi_sim_next = self.feature_learner.feature_net(sim_next_obs_enc)
                phi_targ_obs = self.feature_learner.feature_net(targ_obs_enc)
                phi_targ_next = self.feature_learner.feature_net(targ_next_obs_enc)
                delta_phi_sim = phi_sim_next - phi_sim_obs
                delta_phi_targ = phi_targ_next - phi_targ_obs

        # Create lambda tensor on proper device/dtype.
        lam_tensor = torch.tensor(lam, dtype=torch.float32, device=delta_phi_sim.device)
        
        # Construct the combined least-squares system.
        A_sim = delta_phi_sim
        A_targ = torch.sqrt(lam_tensor) * delta_phi_targ
        A = torch.cat([A_sim, A_targ], dim=0)

        b_sim = reward
        b_targ = torch.sqrt(lam_tensor) * reward_targ
        b = torch.cat([b_sim, b_targ], dim=0)
        
        # Add regularization to stabilize the least-squares solve.
        reg_epsilon = 1e-4 # 1e-5, 0.1
        A_T = A.transpose(0, 1)
        # Solve (A^T A + εI)z = A^T b, which is the Tikhonov regularized least squares solution.
        I = torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
        z = torch.linalg.solve(A_T.matmul(A) + reg_epsilon * I, A_T.matmul(b))
        
        # Check for NaN or Inf in z
        if not torch.isfinite(z).all():
            raise ValueError("Computed latent vector z contains NaN or Inf values.")
        
        # Normalize and scale the latent vector.
        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
        
        meta = OrderedDict()
        meta["z"] = z.squeeze().cpu().numpy()
        return meta
        
    # * v2: assume - obs_next aligns
    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, 
    #     obs: torch.Tensor, 
    #     reward: torch.Tensor, 
    #     next_obs: torch.Tensor, 
    #     obs_targ: torch.Tensor,
    #     reward_targ: torch.Tensor,
    #     next_obs_targ: torch.Tensor,
    #     vis: bool = False
    # ):
    #     import math
    #     import torch.nn.functional as F

    #     # Encode the simulation observations and target observations
    #     with torch.no_grad():
    #         sim_obs_enc = self.encoder(obs)
    #         sim_next_obs_enc = self.encoder(next_obs)
    #         targ_obs_enc = self.encoder(obs_targ)
    #         targ_next_obs_enc = self.encoder(next_obs_targ)
        
    #     # Compute feature representations based on the chosen feature type.
    #     with torch.no_grad():
    #         if self.cfg.feature_type == "state":
    #             phi_sim = self.feature_learner.feature_net(sim_obs_enc)
    #             phi_targ = self.feature_learner.feature_net(targ_obs_enc)
    #         elif self.cfg.feature_type == "diff":
    #             phi_sim = self.feature_learner.feature_net(sim_next_obs_enc) - self.feature_learner.feature_net(sim_obs_enc)
    #             phi_targ = self.feature_learner.feature_net(targ_next_obs_enc) - self.feature_learner.feature_net(targ_obs_enc)
    #         else:
    #             phi_sim = torch.cat([
    #                 self.feature_learner.feature_net(sim_obs_enc),
    #                 self.feature_learner.feature_net(sim_next_obs_enc)
    #             ], dim=-1)
    #             phi_targ = torch.cat([
    #                 self.feature_learner.feature_net(targ_obs_enc),
    #                 self.feature_learner.feature_net(targ_next_obs_enc)
    #             ], dim=-1)
        
    #     # Set lambda to balance simulation and target reward objectives.
    #     lam = 4  # Hyperparameter; can be tuned
        
    #     # Construct the combined least squares system.
    #     # A = [phi_sim; sqrt(lam)*phi_targ] and b = [reward; sqrt(lam)*reward_targ]
    #     A_sim = phi_sim
    #     A_targ = torch.sqrt(torch.tensor(lam, device=A_sim.device)) * phi_targ
    #     A = torch.cat([A_sim, A_targ], dim=0)
        
    #     b_sim = reward
    #     b_targ = torch.sqrt(torch.tensor(lam, device=b_sim.device)) * reward_targ
    #     b = torch.cat([b_sim, b_targ], dim=0)
        
    #     # Solve the least-squares problem to obtain z.
    #     z = torch.linalg.lstsq(A, b).solution
        
    #     # Normalize and scale the latent vector.
    #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
        
    #     meta = OrderedDict()
    #     meta["z"] = z.squeeze().cpu().numpy()
    #     return meta
        
    # * my modify version: 
    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, 
    #     obs: torch.Tensor, 
    #     reward: torch.Tensor, 
    #     next_obs: torch.Tensor, 
    #     obs_targ: torch.Tensor,
    #     reward_targ: torch.Tensor,
    #     next_obs_targ: torch.Tensor,
    #     vis: bool = False
    #     ):
    #     import os
    #     # Setup adapter save directory and log file path.
    #     adapter_save_dir = "/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/a_analysis_25/adapter"
    #     os.makedirs(adapter_save_dir, exist_ok=True)
    #     adapter_save_path = os.path.join(adapter_save_dir, "reward_adapter.pth")
    #     log_path = os.path.join(adapter_save_dir, "log.txt")
        
    #     # Check if adapter exists; if not, train it.
    #     if not os.path.exists(adapter_save_path):
    #         print("Training reward adapter...")
    #         adapter = RewardAdapter().to(self.cfg.device)
    #         optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    #         # Train the adapter using matched samples (based on obs similarity).
    #         loss_history = train_reward_adapter_with_matching(
    #             adapter, optimizer,
    #             sim_obs=obs, 
    #             sim_reward=reward, 
    #             targ_obs=obs_targ, 
    #             targ_reward=reward_targ, 
    #             obs_threshold=10.0, 
    #             num_epochs=200, 
    #             batch_size=128
    #         )
    #         # Save loss history to log.txt.
    #         with open(log_path, 'w') as f:
    #             for epoch, loss in enumerate(loss_history):
    #                 f.write(f"Epoch {epoch}: Loss = {loss}\n")
    #         torch.save(adapter.state_dict(), adapter_save_path)
    #     else:
    #         adapter = RewardAdapter().to(self.cfg.device)
    #         adapter.load_state_dict(torch.load(adapter_save_path))
        
    #     # Use the adapter to adjust the simulation reward.
    #     with torch.no_grad():
    #         reward_adapted = adapter(reward)
            
    #         save_dir = "/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/a_analysis_25/img"
    #         vis_adaptation(reward=reward, reward_adapted=reward_adapted, save_dir=save_dir)
            
        
    #     # Encode observations using the pre-trained encoder.
    #     with torch.no_grad():
    #         obs_encoded = self.encoder(obs)
    #         next_obs_encoded = self.encoder(next_obs)
        
    #     # Compute representation phi based on feature_type.
    #     with torch.no_grad():
    #         if self.cfg.feature_type == "state":
    #             phi = self.feature_learner.feature_net(obs_encoded)
    #         elif self.cfg.feature_type == "diff":
    #             phi = self.feature_learner.feature_net(next_obs_encoded) - self.feature_learner.feature_net(obs_encoded)
    #         else:
    #             phi = torch.cat([
    #                 self.feature_learner.feature_net(obs_encoded),
    #                 self.feature_learner.feature_net(next_obs_encoded)
    #             ], dim=-1)
        
    #     # Solve for z using the adapted reward.
    #     z = torch.linalg.lstsq(phi, reward_adapted).solution
    #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
        
    #     meta = OrderedDict()
    #     meta["z"] = z.squeeze().cpu().numpy()
    #     return meta

    # * original version: 

    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, 
    #     obs_targ: torch.Tensor,
    #     reward_targ: torch.Tensor,
    #     next_obs_targ: torch.Tensor,
    #     vis: bool = False
    # ):
    #     with torch.no_grad():
    #         obs = self.encoder(obs)
    #         next_obs = self.encoder(next_obs)

    #     with torch.no_grad():
    #         if self.cfg.feature_type == "state":
    #             # This is the representation function: phi (trained network)
    #             phi = self.feature_learner.feature_net(obs)
    #         elif self.cfg.feature_type == "diff":
    #             # This is to get the difference between two observations
    #             phi = self.feature_learner.feature_net(
    #                 next_obs
    #             ) - self.feature_learner.feature_net(obs)
    #         else:
    #             phi = torch.cat(
    #                 [
    #                     self.feature_learner.feature_net(obs),
    #                     self.feature_learner.feature_net(next_obs),
    #                 ],
    #                 dim=-1,
    #             )
    #     # this will calculate the result z:
    #     # computes the vector z that minimizes the squared Euclidean norm of the matrix-vector product phi * z - reward
    #     z = torch.linalg.lstsq(phi, reward).solution

    #     # preparing parameters in a NN, ensuring they are normalized and scaled appropriately for the task (given the dim) or the architecture's expectations
    #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)

    #     meta = OrderedDict()
    #     meta["z"] = z.squeeze().cpu().numpy()
    #     return meta

    # def calculate_difference(self, 
    #     obs: torch.Tensor,
    #     reward: torch.Tensor,
    #     next_obs: torch.Tensor,
    #     obs_targ: torch.Tensor,
    #     reward_targ: torch.Tensor,
    #     next_obs_targ: torch.Tensor,
    #     ):
    #     # Number of most similar elements to find
    #     k = 100

    #     # Compute pairwise distances between obs and obs_targ
    #     obs_distances = torch.cdist(obs, obs_targ, p=2)  # Pairwise Euclidean distances
    #     most_similar_obs_indices = torch.topk(obs_distances, k, largest=False, dim=1).indices

    #     differences = []

    #     # For each observation, find the most similar action based on reward similarity
    #     for i in range(obs.shape[0]):
    #         # Get the indices of the top k most similar obs_targ
    #         similar_indices = most_similar_obs_indices[i]
            
    #         # Extract the corresponding rewards and actions from the target
    #         reward_candidates = reward_targ[similar_indices]
            
    #         # Find the most similar reward (action similarity proxy)
    #         action_similarity_index = torch.argmin(torch.abs(reward[i] - reward_candidates))
    #         matched_index = similar_indices[action_similarity_index]
            
    #         # Compute the difference between next_obs and next_obs_targ for this match
    #         difference = next_obs[i] - next_obs_targ[matched_index]
    #         differences.append(difference)

    #     # Convert the differences into a matrix
    #     difference_matrix = torch.stack(differences)
    #     return difference_matrix

    def calculate_difference(
        self, 
        obs: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        obs_targ: torch.Tensor,
        reward_targ: torch.Tensor,
        next_obs_targ: torch.Tensor,
        vis: bool = False
    ):
        """
        Captures discrepancies in dynamics and reward transitions between two environments:
        1. Align each observation in Env1 with top-k most similar observations in Env2.
        2. Among top-k, match using closest reward.
        3. Calculate differences: observation, reward, and next-state (transition).
        """
        k = 100  # Number of nearest neighbors to consider

        # Ensure compatibility with tensor operations
        obs, obs_targ = obs.float(), obs_targ.float()
        reward, reward_targ = reward.squeeze(-1), reward_targ.squeeze(-1)

        # Pairwise distances between obs and obs_targ
        obs_distances = torch.cdist(obs, obs_targ, p=2)  # [N, M]
        most_similar_obs_indices = torch.topk(obs_distances, k, largest=False, dim=1).indices

        # Store differences
        obs_differences, reward_differences, transition_differences = [], [], []

        for i in range(obs.shape[0]):
            # Find top-k closest observations in obs_targ
            similar_indices = most_similar_obs_indices[i]
            reward_candidates = reward_targ[similar_indices]
            next_obs_candidates = next_obs_targ[similar_indices]

            # Match based on reward proximity
            reward_diff = torch.abs(reward_candidates - reward[i])
            best_match_idx = torch.argmin(reward_diff)
            matched_index = similar_indices[best_match_idx]

            # Calculate differences
            obs_difference = torch.norm(obs[i] - obs_targ[matched_index], p=2).item()
            reward_difference = torch.abs(reward[i] - reward_targ[matched_index]).item()
            transition_difference = torch.norm(next_obs[i] - next_obs_targ[matched_index], p=2).item()

            obs_differences.append(obs_difference)
            reward_differences.append(reward_difference)
            transition_differences.append(transition_difference)

        # Aggregate statistics
        mean_obs_diff = sum(obs_differences) / len(obs_differences)
        mean_reward_diff = sum(reward_differences) / len(reward_differences)
        mean_transition_diff = sum(transition_differences) / len(transition_differences)

        # Optional visualization
        if vis:
            import matplotlib.pyplot as plt
            os.makedirs("/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/vis", exist_ok=True)
            indices = range(len(obs_differences))

            plt.figure(figsize=(12, 8))
            plt.plot(indices, obs_differences, label="Observation Differences", alpha=0.7)
            plt.plot(indices, reward_differences, label="Reward Differences", alpha=0.7)
            plt.plot(indices, transition_differences, label="Transition Differences", alpha=0.7)

            plt.xlabel("Sample Index")
            plt.ylabel("Difference Value")
            plt.title(
                f"Discrepancy Metrics (Mean Obs: {mean_obs_diff:.3f}, "
                f"Mean Reward: {mean_reward_diff:.3f}, Mean Transition: {mean_transition_diff:.3f})"
            )
            plt.legend()
            plt.grid()
            plt.savefig("/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/vis/dynamics_discrepancy.png", dpi=300)
            plt.close()

        # Return transition differences for policy adaptation
        return transition_differences

    def sample_z(self, size):
        gaussian_rdv = torch.randn((size, self.cfg.z_dim), dtype=torch.float32)
        z = math.sqrt(self.cfg.z_dim) * F.normalize(gaussian_rdv, dim=1)
        return z

    def init_meta(self) -> MetaDict:
        if self.solved_meta is not None:
            print("solved_meta")
            return self.solved_meta
        else:
            z = self.sample_z(1)
            z = z.squeeze().numpy()
            meta = OrderedDict()
            meta["z"] = z
        return meta

    # pylint: disable=unused-argument
    def update_meta(
        self,
        meta: MetaDict,
        global_step: int,
        time_step: TimeStep,
        finetune: bool = False,
        replay_loader: tp.Optional[ReplayBuffer] = None,
    ) -> MetaDict:
        if global_step % self.cfg.update_z_every_step == 0:
            return self.init_meta()
        return meta

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        obs = torch.as_tensor(
            obs, device=self.cfg.device, dtype=torch.float32
        ).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta["z"], device=self.cfg.device).unsqueeze(0)  # type: ignore
        if self.cfg.boltzmann:
            dist = self.actor(h, z)
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(h, z, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_sf(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: tp.Optional[torch.Tensor],
        z: torch.Tensor,
        step: int,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():
            if self.cfg.boltzmann:
                dist = self.actor(next_obs, z)
                next_action = dist.sample()
            else:
                stddev = utils.schedule(self.cfg.stddev_schedule, step)
                dist = self.actor(next_obs, z, stddev)
                next_action = dist.sample(clip=self.cfg.stddev_clip)
            next_F1, next_F2 = self.successor_target_net(
                next_obs, z, next_action
            )  # batch x z_dim
            if self.cfg.feature_type == "state":
                target_phi = self.feature_learner.feature_net(
                    next_obs
                ).detach()  # batch x z_dim
            elif self.cfg.feature_type == "diff":
                target_phi = (
                    self.feature_learner.feature_net(next_obs).detach()
                    - self.feature_learner.feature_net(obs).detach()
                )
            else:
                target_phi = torch.cat(
                    [
                        self.feature_learner.feature_net(obs).detach(),
                        self.feature_learner.feature_net(next_obs).detach(),
                    ],
                    dim=-1,
                )
            next_Q1, next_Q2 = [
                torch.einsum("sd, sd -> s", next_Fi, z)
                for next_Fi in [next_F1, next_F2]
            ]
            next_F = torch.where((next_Q1 < next_Q2).reshape(-1, 1), next_F1, next_F2)
            target_F = target_phi + discount * next_F

        F1, F2 = self.successor_net(obs, z, action)
        if self.cfg.q_loss:
            Q1, Q2 = [torch.einsum("sd, sd -> s", Fi, z) for Fi in [F1, F2]]
            target_Q = torch.einsum("sd, sd -> s", target_F, z)
            sf_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        else:
            sf_loss = F.mse_loss(F1, target_F) + F.mse_loss(F2, target_F)

        # compute feature loss
        if self.cfg.feature_learner == "hilp":
            phi_loss, info = self.feature_learner(
                obs=obs, action=action, next_obs=next_obs, future_obs=future_obs
            )
        else:
            phi_loss = self.feature_learner(
                obs=obs, action=action, next_obs=next_obs, future_obs=future_obs
            )
            info = None

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics["target_F"] = target_F.mean().item()
            metrics["F1"] = F1.mean().item()
            metrics["phi"] = target_phi.mean().item()
            metrics["phi_norm"] = torch.norm(target_phi, dim=-1).mean().item()
            metrics["z_norm"] = torch.norm(z, dim=-1).mean().item()
            metrics["sf_loss"] = sf_loss.item()
            if phi_loss is not None:
                metrics["phi_loss"] = phi_loss.item()

            if isinstance(self.sf_opt, torch.optim.Adam):
                metrics["sf_opt_lr"] = self.sf_opt.param_groups[0]["lr"]

            if info is not None:
                for key, val in info.items():
                    metrics[key] = val.item()

        # optimize SF
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.sf_opt.zero_grad(set_to_none=True)
        if self.phi_opt is not None:
            self.phi_opt.zero_grad(set_to_none=True)
            phi_loss.backward(retain_graph=True)
        sf_loss.backward()
        self.sf_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        if self.phi_opt is not None:
            self.phi_opt.step()

        return metrics

    def update_actor(
        self, obs: torch.Tensor, z: torch.Tensor, step: int
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        if self.cfg.boltzmann:
            dist = self.actor(obs, z)
            action = dist.rsample()
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(obs, z, stddev)
            action = dist.sample(clip=self.cfg.stddev_clip)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.successor_net(obs, z, action)
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)
        Q = torch.min(Q1, Q2)
        actor_loss = (
            (self.cfg.temp * log_prob - Q).mean() if self.cfg.boltzmann else -Q.mean()
        )

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()

        return metrics

    def aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.cfg.update_every_steps != 0:
            return metrics

        for _ in range(self.cfg.num_sf_updates):
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs = batch.obs
            action = batch.action
            discount = batch.discount
            next_obs = batch.next_obs
            future_obs = batch.future_obs

            z = self.sample_z(self.cfg.batch_size).to(self.cfg.device)
            if not z.shape[-1] == self.cfg.z_dim:
                raise RuntimeError("There's something wrong with the logic here")

            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)
            future_obs = self.aug_and_encode(future_obs)
            next_obs = next_obs.detach()

            if self.cfg.mix_ratio > 0:
                perm = torch.randperm(self.cfg.batch_size)
                with torch.no_grad():
                    if self.cfg.feature_type == "state":
                        desired_obs = next_obs[perm]
                        phi = self.feature_learner.feature_net(desired_obs)
                    elif self.cfg.feature_type == "diff":
                        desired_obs = obs[perm]
                        desired_next_obs = next_obs[perm]
                        phi = self.feature_learner.feature_net(
                            desired_next_obs
                        ) - self.feature_learner.feature_net(desired_obs)
                    else:
                        desired_obs = obs[perm]
                        desired_next_obs = next_obs[perm]
                        phi = torch.cat(
                            [
                                self.feature_learner.feature_net(desired_obs),
                                self.feature_learner.feature_net(desired_next_obs),
                            ],
                            dim=-1,
                        )
                # compute inverse of cov of phi
                cov = torch.matmul(phi.T, phi) / phi.shape[0]
                inv_cov = torch.linalg.pinv(cov)

                mix_idxs: tp.Any = np.where(
                    np.random.uniform(size=self.cfg.batch_size) < self.cfg.mix_ratio
                )[0]
                with torch.no_grad():
                    new_z = phi[mix_idxs]

                new_z = torch.matmul(new_z, inv_cov)  # batch_size x z_dim
                new_z = math.sqrt(self.cfg.z_dim) * F.normalize(new_z, dim=1)
                z[mix_idxs] = new_z

            metrics.update(
                self.update_sf(
                    obs=obs,
                    action=action,
                    discount=discount,
                    next_obs=next_obs,
                    future_obs=future_obs,
                    z=z,
                    step=step,
                )
            )

            # update actor
            metrics.update(self.update_actor(obs.detach(), z, step))

            # update critic target
            utils.soft_update_params(
                self.successor_net, self.successor_target_net, self.cfg.sf_target_tau
            )

        return metrics
