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
from torch.autograd import Function
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

from torch.utils.data import TensorDataset, DataLoader
logger = logging.getLogger(__name__)

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, λ):
        ctx.lambda_ = λ
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, λ):
    return GradReverse.apply(x, λ)

class TaskEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, z_dim):
        super().__init__()
        # simulation-domain encoder
        self.enc_sim = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # target-domain encoder
        self.enc_targ = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # decoder to z
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        # adversarial discriminator for domain alignment
        self.discrim = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, phi_sim, r_sim, phi_targ, r_targ):
        # encode sim and real pairs
        m_sim  = self.enc_sim (torch.cat([phi_sim,  r_sim ], dim=-1))
        m_targ = self.enc_targ(torch.cat([phi_targ, r_targ], dim=-1))
        # simple average pooling
        m_task = torch.cat([m_sim, m_targ], dim=0).mean(dim=0, keepdim=True)
        # decode to z
        return self.decoder(m_task).squeeze(0)

    def domain_loss(self, phi_sim, r_sim, phi_targ, r_targ, lambda_=1.0):
        # encode
        m_sim  = self.enc_sim (torch.cat([phi_sim,  r_sim ], dim=-1))
        m_targ = self.enc_targ(torch.cat([phi_targ, r_targ], dim=-1))
        # gradient reversal for adversarial training
        m_all = torch.cat([
            grad_reverse(m_sim,  lambda_),
            grad_reverse(m_targ,  lambda_),
        ], dim=0)
        # domain labels: 0 for sim, 1 for real
        labels = torch.cat([
            torch.zeros(m_sim.size(0), dtype=torch.long, device=m_sim.device),
            torch.ones (m_targ.size(0), dtype=torch.long, device=m_targ.device),
        ], dim=0)
        logits = self.discrim(m_all)
        return F.cross_entropy(logits, labels)

class DeepSetStat(nn.Module):
    def __init__(self, feat_dim, hid=128, out_dim=128):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(feat_dim, hid), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(hid, out_dim), nn.ReLU())
    def forward(self, feats):    # feats: [N, F]
        h = self.phi(feats)      # [N, hid]
        return self.rho(h.mean(0,keepdim=True)).squeeze(0)  # [out_dim]
    

# class ZAdapter(nn.Module):
#     def __init__(self, z_dim, feat_dim, hidden_dim=256):
#         super().__init__()
#         # 输入维度 = z_dim + feat_dim*2（mean + var）
#         self.net = nn.Sequential(
#             nn.Linear(z_dim + feat_dim*2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, z_dim),
#         )
#     def forward(self, z_src, dist_stat):
#         # z_src: [D], dist_stat: [2F]
#         x = torch.cat([z_src, dist_stat], dim=-1).unsqueeze(0)  # [1, D+2F]
#         return self.net(x).squeeze(0)  # [D]   
    

import torch.nn.functional as F

def contrastive_loss(phi, phi_targ, margin=1.0):
    # Calculate pairwise distances
    distances = torch.norm(phi - phi_targ, p=2, dim=1)

    # Contrastive loss computation
    loss = torch.mean(
        (distances**2) * (distances <= margin).float() +
        (margin - distances).clamp(min=0)**2 * (distances > margin).float()
    )
    return loss


class ZAdapter(nn.Module):
    """
    g_theta(z_src, stat) -> R^D
    这里将 [z_src || stat] 拼接后用 MLP 映射到 D 维向量。
    - z_src: [D] 或 [1, D]
    - stat : [K] 或 [1, K]   (K = 任意统计向量维度，比如 DeepSet 输出 dim)
    """
    def __init__(self, z_dim, stat_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + stat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z_src, stat):
        if z_src.dim() == 1:
            z_src = z_src.unsqueeze(0)   # [1, D]
        if stat.dim() == 1:
            stat = stat.unsqueeze(0)     # [1, K]
        x = torch.cat([z_src, stat], dim=-1)  # [1, D+K]
        return self.net(x)               # [1, D]


@dataclasses.dataclass
class SFAgentConfig:

    # added value: 
    lambda_wls: float = 1.2           # 加权最小二乘里真实域权重 λ>1
    use_adapter: bool = True          # 是否启用 DeepSet + Adapter
    adapter_steps: int = 1 # 1 - 40
    adapter_batch_size: int = 256
    adapter_lr: float = 1e-3
    deepset_out_dim: int = 128        # DeepSet 输出维度 K（同时作为 adapter 的 stat_dim）


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


        # added new: 
        if self.cfg.feature_learner == "hilp":
            if self.cfg.feature_type != "concat":
                phi_feat_dim = self.cfg.z_dim            # state/diff 情况
            else:
                phi_feat_dim = self.cfg.z_dim // 2       # concat 情况：feature_net 输出是 z_dim//2
        else:
            # 其他 learner 的 feature_net 默认输出 z_dim
            phi_feat_dim = self.cfg.z_dim

        # DeepSet 输出维度 K
        K = self.cfg.deepset_out_dim

        # 初始化 DeepSet（输入维度 = feature_net 输出维度；输出维度 = K）
        self.deepset = DeepSetStat(feat_dim=phi_feat_dim, hid=128, out_dim=K).to(cfg.device)

        # 初始化 Adapter（输入统计维度 = K）
        self.adapter = ZAdapter(z_dim=self.cfg.z_dim, stat_dim=K, hidden_dim=256).to(cfg.device)




        #### origina: 

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

        # here is how they infer the z
        # def infer_meta_from_obs_and_rewards_sim2real(
    
    # explore Jun12 #TODO: Baseline 2:
    # def infer_meta_from_obs_and_rewards_sim2real(
    #         self, 
    #         obs: torch.Tensor, 
    #         reward: torch.Tensor, 
    #         next_obs: torch.Tensor, 
    #         obs_targ: torch.Tensor,
    #         reward_targ: torch.Tensor,
    #         next_obs_targ: torch.Tensor,
    #         vis: bool = False
    #     ):
    #         import math
    #         import torch.nn.functional as F

    #         # Encode the simulation observations and target observations
    #         with torch.no_grad():
    #             sim_obs_enc = self.encoder(obs)
    #             sim_next_obs_enc = self.encoder(next_obs)
    #             targ_obs_enc = self.encoder(obs_targ)
    #             targ_next_obs_enc = self.encoder(next_obs_targ)
            
    #         # Compute feature representations based on the chosen feature type.
    #         with torch.no_grad():
    #             if self.cfg.feature_type == "state":
    #                 phi_sim = self.feature_learner.feature_net(sim_obs_enc)
    #                 phi_targ = self.feature_learner.feature_net(targ_obs_enc)
            
    #         # Set lambda to balance simulation and target reward objectives.
    #         lam = 5  # Hyperparameter; can be tuned
            
    #         # Construct the combined least squares system.
    #         # A = [phi_sim; sqrt(lam)*phi_targ] and b = [reward; sqrt(lam)*reward_targ]
    #         A_sim = phi_sim
    #         A_targ = torch.sqrt(torch.tensor(lam, device=A_sim.device)) * phi_targ
    #         A = torch.cat([A_sim, A_targ], dim=0)
            
    #         b_sim = reward
    #         b_targ = torch.sqrt(torch.tensor(lam, device=b_sim.device)) * reward_targ
    #         b = torch.cat([b_sim, b_targ], dim=0)
            
    #         # Solve the least-squares problem to obtain z.
    #         z = torch.linalg.lstsq(A, b).solution
            
    #         # Normalize and scale the latent vector.
    #         z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
            
    #         meta = OrderedDict()
    #         meta["z"] = z.squeeze().cpu().numpy()
    #         return meta

    # # TODO: Baseline 3
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
    #         device = obs.device
    #         D = self.cfg.z_dim
    #         lam = 5.0

    #         # 1) 编码 & 抽特征 φ(s)
    #         with torch.no_grad():
    #             sim_enc  = self.encoder(obs)
    #             targ_enc = self.encoder(obs_targ)
    #             phi_sim  = self.feature_learner.feature_net(sim_enc)   # [N, F]
    #             phi_targ = self.feature_learner.feature_net(targ_enc)  # [N, F]

    #         # 2) 加权最小二乘求 z_src
    #         A = torch.cat([
    #             phi_sim,
    #             torch.sqrt(torch.tensor(lam, device=device)) * phi_targ
    #         ], dim=0)                                                    # [2N, F]
    #         b = torch.cat([
    #             reward.view(-1,1),
    #             torch.sqrt(torch.tensor(lam, device=device)) * reward_targ.view(-1,1)
    #         ], dim=0).squeeze(-1)                                        # [2N]
    #         z_src = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)  # [F]
    #         z_src = F.normalize(z_src, dim=0) * math.sqrt(D)                     # [D]

    #         # 3) 构造分布统计量（mean & var）
    #         mean_phi = phi_targ.mean(dim=0)                             # [F]
    #         var_phi  = phi_targ.var(dim=0, unbiased=False)              # [F]
    #         dist_stat = torch.cat([mean_phi, var_phi], dim=-1)          # [2F]

    #         # 4) 确保有 adapter 网络
    #         if not hasattr(self, 'z_adapter'):
    #             feat_dim = phi_targ.size(1)
    #             self.z_adapter = ZAdapter(z_dim=D, feat_dim=feat_dim).to(device)

    #         # 5) 在目标域上微调 adapter
    #         opt = torch.optim.Adam(self.z_adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    #         self.z_adapter.train()
    #         for _ in range(300):
    #             z_pred = self.z_adapter(z_src, dist_stat)               # [D]
    #             pred_r = (phi_targ @ z_pred.unsqueeze(-1)).squeeze(-1)  # [N]
    #             loss   = F.mse_loss(pred_r, reward_targ.view(-1))
    #             print(loss)
    #             opt.zero_grad(); loss.backward(); opt.step()

    #         # 6) 推断并归一化最终 z
    #         self.z_adapter.eval()
    #         with torch.no_grad():
    #             z_final = self.z_adapter(z_src, dist_stat)             
    #             z_final = F.normalize(z_final, dim=0) * math.sqrt(D)

    #         return OrderedDict(z=z_final.cpu().numpy())


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
    #     device = obs.device
    #     D = self.cfg.z_dim
    #     lam = 5.0

    #     # 1) 编码 & 抽特征 φ(s)
    #     with torch.no_grad():
    #         sim_enc  = self.encoder(obs)
    #         targ_enc = self.encoder(obs_targ)
    #         phi_sim  = self.feature_learner.feature_net(sim_enc)   # [N, F]
    #         phi_targ = self.feature_learner.feature_net(targ_enc)  # [N, F]

    #     # 2) 加权最小二乘求 z_src
    #     A = torch.cat([
    #         phi_sim,
    #         torch.sqrt(torch.tensor(lam, device=device)) * phi_targ
    #     ], dim=0)                                                    # [2N, F]
    #     b = torch.cat([
    #         reward.view(-1,1),
    #         torch.sqrt(torch.tensor(lam, device=device)) * reward_targ.view(-1,1)
    #     ], dim=0).squeeze(-1)                                        # [2N]
    #     z_src = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)  # [F]
    #     z_src = F.normalize(z_src, dim=0) * math.sqrt(D)                     # [D]

    #     # 3) 构造分布统计量（mean & var）
    #     mean_phi = phi_targ.mean(dim=0)                             # [F]
    #     var_phi  = phi_targ.var(dim=0, unbiased=False)              # [F]
    #     dist_stat = torch.cat([mean_phi, var_phi], dim=-1)          # [2F]

    #     # 4) 确保有 adapter 网络
    #     if not hasattr(self, 'z_adapter'):
    #         feat_dim = phi_targ.size(1)
    #         self.z_adapter = ZAdapter(z_dim=D, feat_dim=feat_dim).to(device)

    #     # 5) 在目标域上微调 adapter
    #     opt = torch.optim.Adam(self.z_adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    #     self.z_adapter.train()
    #     for _ in range(500):
    #         z_pred = self.z_adapter(z_src, dist_stat)               # [D]
    #         pred_r = (phi_targ @ z_pred.unsqueeze(-1)).squeeze(-1)  # [N]
    #         loss   = F.mse_loss(pred_r, reward_targ.view(-1))
    #         print(loss)
    #         opt.zero_grad(); loss.backward(); opt.step()

    #     # 6) 推断并归一化最终 z
    #     self.z_adapter.eval()
    #     with torch.no_grad():
    #         z_final = self.z_adapter(z_src, dist_stat)             
    #         z_final = F.normalize(z_final, dim=0) * math.sqrt(D)

    #     return OrderedDict(z=z_final.cpu().numpy())

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
    #     lam = 5  # Hyperparameter; can be tuned
        
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

    
    
    
    # try 3
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
    #     from collections import OrderedDict

    #     device = obs.device
    #     D = self.cfg.z_dim
    #     lam = 5.0

    #     # 1) 编码 & 抽取 φ(s), φ(s_targ)
    #     with torch.no_grad():
    #         sim_enc  = self.encoder(obs)
    #         targ_enc = self.encoder(obs_targ)
    #         phi_sim  = self.feature_learner.feature_net(sim_enc)   # [N, F]
    #         phi_targ = self.feature_learner.feature_net(targ_enc)  # [N, F]

    #     # 2) 加权最小二乘求 z_src
    #     A = torch.cat([
    #         phi_sim,
    #         math.sqrt(lam) * phi_targ
    #     ], dim=0)                                                    # [2N, F]
    #     b = torch.cat([
    #         reward.view(-1,1),
    #         math.sqrt(lam) * reward_targ.view(-1,1)
    #     ], dim=0).squeeze(-1)                                        # [2N]
    #     z_src = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)  # [F]
    #     z_src = F.normalize(z_src, dim=0) * math.sqrt(D)                     # [D]

    #     # 3) 构造目标域分布统计量（mean & var）
    #     mean_phi = phi_targ.mean(dim=0)                             # [F]
    #     var_phi  = phi_targ.var(dim=0, unbiased=False)              # [F]
    #     dist_stat = torch.cat([mean_phi, var_phi], dim=-1)          # [2F]
    #     stat_dim = dist_stat.shape[-1]                              # =2F

    #     # 4) 确保有 adapter 网络
    #     if not hasattr(self, 'z_adapter'):
    #         self.z_adapter = ZAdapter(z_dim=D, stat_dim=stat_dim).to(device)

    #     # 5) 在目标域上微调 adapter
    #     opt = torch.optim.Adam(self.z_adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    #     self.z_adapter.train()
    #     for _ in range(300):
    #         z_pred = self.z_adapter(z_src, dist_stat)               # [D]
    #         pred_r = (phi_targ @ z_pred.unsqueeze(-1)).squeeze(-1)  # [N]
    #         loss   = F.mse_loss(pred_r, reward_targ.view(-1))
    #         opt.zero_grad(); loss.backward(); opt.step()

    #     # 6) 推断并归一化最终 z
    #     self.z_adapter.eval()
    #     with torch.no_grad():
    #         z_final = self.z_adapter(z_src, dist_stat)
    #         z_final = F.normalize(z_final, dim=0) * math.sqrt(D)

    #     # 7) 返回给外部策略
    #     return OrderedDict(z=z_final.cpu().numpy())


    # try2
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
    #     from collections import OrderedDict

    #     device = obs.device
    #     D = self.cfg.z_dim
    #     lam = 5.0

    #     # 1) 编码 & 抽特征 φ(s)
    #     with torch.no_grad():
    #         sim_enc  = self.encoder(obs)
    #         targ_enc = self.encoder(obs_targ)
    #         φ_sim    = self.feature_learner.feature_net(sim_enc)    # [N, F]
    #         φ_targ   = self.feature_learner.feature_net(targ_enc)   # [N, F]

    #     # 2) 加权最小二乘求 z_src （闭式解）
    #     A = torch.cat([
    #         φ_sim,
    #         math.sqrt(lam) * φ_targ
    #     ], dim=0)                                                    # [2N, F]
    #     b = torch.cat([
    #         reward.view(-1,1),
    #         math.sqrt(lam) * reward_targ.view(-1,1)
    #     ], dim=0).squeeze(-1)                                        # [2N]
    #     z_src = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)  # [F]
    #     z_src = F.normalize(z_src, dim=0) * math.sqrt(D)                     # [D]

    #     # 3) 深度集成统计量代替 mean+var
    #     with torch.no_grad():
    #         if not hasattr(self, 'set_stat'):
    #             feat_dim = φ_targ.size(1)
    #             self.set_stat = DeepSetStat(feat_dim=feat_dim, hid=128, out_dim=128).to(device)
    #         dist_stat = self.set_stat(φ_targ).detach()   # [128]

    #     # 4) 确保有 adapter 并 detach 固定 z_src
    #     z_src = z_src.detach()
    #     if not hasattr(self, 'z_adapter'):
    #         stat_dim = dist_stat.shape[0]
    #         self.z_adapter = ZAdapter(z_dim=D, stat_dim=stat_dim).to(device)

    #     # 5) 内置微调 Adapter —— 在一个全新的计算图上
    #     self.z_adapter.train()
    #     optimizer = torch.optim.Adam(self.z_adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    #     for _ in range(300):
    #         optimizer.zero_grad()
    #         # 前向
    #         z_pred = self.z_adapter(z_src, dist_stat)               # [D]
    #         pred_r = (φ_targ @ z_pred.unsqueeze(-1)).squeeze(-1)    # [N]
    #         loss   = F.mse_loss(pred_r, reward_targ.view(-1))       # 计算误差
    #         # 反向 + 更新（每次都是新图）
    #         loss.backward()
    #         optimizer.step()

    #     # 6) 推断最终 z 并归一化
    #     self.z_adapter.eval()
    #     with torch.no_grad():
    #         z_final = self.z_adapter(z_src, dist_stat)
    #         z_final = F.normalize(z_final, dim=0) * math.sqrt(D)

    #     return OrderedDict(z=z_final.cpu().numpy())


    # try 1: 
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
    #         import math
    #         import torch.nn.functional as F
    #         from collections import OrderedDict

    #         device = obs.device
    #         D = self.cfg.z_dim
    #         lam = 5.0

    #         # 1) 编码 & 抽特征 φ(s)
    #         with torch.no_grad():
    #             sim_enc  = self.encoder(obs)
    #             targ_enc = self.encoder(obs_targ)
    #             φ_sim    = self.feature_learner.feature_net(sim_enc)    # [N, F]
    #             φ_targ   = self.feature_learner.feature_net(targ_enc)   # [N, F]

    #         # 2) 加权最小二乘求 z_src （闭式解）
            
    #         # method 1:
    #         # A = torch.cat([
    #         #     φ_sim,
    #         #     math.sqrt(lam) * φ_targ
    #         # ], dim=0)                                                    # [2N, F]
    #         # b = torch.cat([
    #         #     reward.view(-1,1),
    #         #     math.sqrt(lam) * reward_targ.view(-1,1)
    #         # ], dim=0).squeeze(-1)                                        # [2N]
    #         # z_src = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)  # [F]
    #         # z_src = F.normalize(z_src, dim=0) * math.sqrt(D)                     # [D]


    #         # method 2:
    #         r_sim = reward.view(-1,1)                  # [N, 1]
    #         # least-squares solution for z_src:  [F,1] → squeeze to [F]
    #         z_src = torch.linalg.lstsq(φ_sim, r_sim).solution.squeeze(-1)
    #         # normalize and scale to √D
    #         z_src = F.normalize(z_src, dim=0) * math.sqrt(D)   # [F]

    #         # 3) 深度集成统计量代替 mean+var
    #         with torch.no_grad():
    #             if not hasattr(self, 'set_stat'):
    #                 feat_dim = φ_targ.size(1)
    #                 self.set_stat = DeepSetStat(feat_dim=feat_dim, hid=128, out_dim=128).to(device)
    #             dist_stat = self.set_stat(φ_targ).detach()   # [128]

    #         # 4) 确保有 adapter 并 detach 固定 z_src
    #         z_src = z_src.detach()
    #         if not hasattr(self, 'z_adapter'):
    #             stat_dim = dist_stat.shape[0]
    #             self.z_adapter = ZAdapter(z_dim=D, stat_dim=stat_dim).to(device)

    #         # 5) 内置微调 Adapter —— 在一个全新的计算图上
    #         self.z_adapter.train()
    #         optimizer = torch.optim.Adam(self.z_adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    #         for _ in range(300):
    #             optimizer.zero_grad()
    #             # 前向
    #             z_pred = self.z_adapter(z_src, dist_stat)               # [D]
    #             pred_r = (φ_targ @ z_pred.unsqueeze(-1)).squeeze(-1)    # [N]
    #             loss   = F.mse_loss(pred_r, reward_targ.view(-1))       # 计算误差
    #             # 反向 + 更新（每次都是新图）
    #             loss.backward()
    #             optimizer.step()

    #         # 6) 推断最终 z 并归一化
    #         self.z_adapter.eval()
    #         with torch.no_grad():
    #             z_final = self.z_adapter(z_src, dist_stat)
    #             z_final = F.normalize(z_final, dim=0) * math.sqrt(D)

    #         return OrderedDict(z=z_final.cpu().numpy())



    # TODO: today update - Sep.8: hahahahahha
    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self,
    #     obs: torch.Tensor,          # 模拟域 N×...   (source / sim)
    #     reward: torch.Tensor,       # 模拟域 N×1 or N
    #     next_obs: torch.Tensor,     # 模拟域 N×...

    #     obs_targ: torch.Tensor,     # 真实域 M×...   (target / real)
    #     reward_targ: torch.Tensor,  # 真实域 M×1 or M
    #     next_obs_targ: torch.Tensor,# 真实域 M×...
    #     vis: bool = False
    # ):
    #     """
    #     Sim->Real 推理期自适应：
    #     (a) 加权联合最小二乘，得到 z_src
    #     (b) 用 DeepSet(phi(s_targ)) 得到环境统计 η
    #     (c) 用 Adapter g_theta(z_src, η) 做轻量微调，得到 z_final
    #     返回 meta["z"] (numpy), 做了 L2 归一化并乘 sqrt(D)
    #     """
    #     device = obs.device
    #     D = int(self.cfg.z_dim)

    #     # --------------------------
    #     # 0) 编码观测 -> 表征 phi
    #     # --------------------------
    #     with torch.no_grad():
    #         enc_sim     = self.encoder(obs)           # N x ...
    #         enc_sim_nxt = self.encoder(next_obs)      # N x ...
    #         enc_targ    = self.encoder(obs_targ)      # M x ...
    #         enc_targ_nxt= self.encoder(next_obs_targ) # M x ...

    #     # helper: 从编码特征构造 phi 矩阵 (对齐你原有的 feature_type 逻辑)
    #     def build_phi(encoded, encoded_next):
    #         if self.cfg.feature_type == "state":
    #             return self.feature_learner.feature_net(encoded)  # [*, D]
    #         elif self.cfg.feature_type == "diff":
    #             return (self.feature_learner.feature_net(encoded_next)
    #                     - self.feature_learner.feature_net(encoded))  # [*, D]
    #         else:  # "concat"
    #             return torch.cat(
    #                 [self.feature_learner.feature_net(encoded),
    #                 self.feature_learner.feature_net(encoded_next)],
    #                 dim=-1
    #             )  # 注意：若 concat，D 应与 cfg.z_dim 一致

    #     with torch.no_grad():
    #         Phi_sim  = build_phi(enc_sim, enc_sim_nxt)          # N x D
    #         Phi_targ = build_phi(enc_targ, enc_targ_nxt)        # M x D

    #         # 奖励向量 reshape 成 (N,1)/(M,1)
    #         r_sim   = reward.reshape(-1, 1).to(device)          # N x 1
    #         r_targ  = reward_targ.reshape(-1, 1).to(device)     # M x 1

    #     # --------------------------
    #     # 1) (a) 加权联合最小二乘  (式 (5))
    #     #     A = [Phi_sim;
    #     #          sqrt(lambda)*Phi_targ],
    #     #     b = [r_sim;
    #     #          sqrt(lambda)*r_targ]
    #     #     z_src = argmin ||Az - b||^2  => lstsq
    #     # --------------------------
    #     lam = float(getattr(self.cfg, "lambda_wls", 4.0))
    #     lam_sqrt = math.sqrt(lam)

    #     with torch.no_grad():
    #         A = torch.cat([Phi_sim,  lam_sqrt * Phi_targ], dim=0)  # (N+M) x D
    #         b = torch.cat([r_sim,    lam_sqrt * r_targ],  dim=0)   # (N+M) x 1

    #         # 最小二乘解 ẑ = (A^T A)^(-1) A^T b     -> 用 lstsq 更稳健
    #         z_src = torch.linalg.lstsq(A, b).solution  # D x 1 (或 D x 1 视 torch 版本可能返回 D x 1)
    #         # 归一化 + 尺度匹配
    #         z_src = math.sqrt(D) * F.normalize(z_src.squeeze(-1), dim=0)  # (D,)

    #     # --------------------------
    #     # 2) (b) DeepSet 统计 η = DeepSet({phi(s_targ)})
    #     #    论文里是对 {phi(s^t_j)} 做置换不变聚合。
    #     #    即便 feature_type == "diff"，这里建议用 "state" 表征统计环境分布。
    #     # --------------------------
    #     if getattr(self.cfg, "use_adapter", True):
    #         with torch.no_grad():
    #             phi_state_targ = self.feature_learner.feature_net(enc_targ)  # M x D_state(=D if一致)
    #             # 若 concat 模式导致 D 不一致，可改为用 self.feature_learner.feature_net(enc_targ) 的输出维度
    #             eta = self.deepset(phi_state_targ)  # K 维向量
    #             if eta.dim() == 1:
    #                 eta = eta.unsqueeze(0)          # 1 x K
    #     else:
    #         eta = None  # 不用 adapter

    #     # --------------------------
    #     # 3) (c) Adapter: g_theta(z_src, η) -> refine z
    #     #    轻量内环微调，在目标域上拟合: Phi_targ @ g_theta(...) ≈ r_targ   (式 (6))
    #     # --------------------------
    #     if getattr(self.cfg, "use_adapter", True):
    #         # 复制一份 adapter 的优化器（临时微调，不破坏全局训练状态的话，可用新的优化器）
    #         adapter_lr = float(getattr(self.cfg, "adapter_lr", 1e-3))
    #         adapter_steps = int(getattr(self.cfg, "adapter_steps", 300))
    #         adapter_bs = int(getattr(self.cfg, "adapter_batch_size", 256))

    #         # 构造 DataLoader（只在目标域上做几步拟合）
    #         targ_ds = TensorDataset(Phi_targ.detach(), r_targ.detach())
    #         targ_dl = DataLoader(targ_ds, batch_size=min(adapter_bs, len(targ_ds)), shuffle=True, drop_last=False)

    #         # 将 z_src/eta 固定为张量输入
    #         z_src_tensor = z_src.detach().unsqueeze(0)  # 1 x D

    #         # 你可以在 adapter/g_theta 内部做 concat([z_src, eta]) 再 MLP -> R^D
    #         self.adapter.train()
    #         optimizer = torch.optim.Adam(self.adapter.parameters(), lr=adapter_lr)

    #         # 简单训练若干步
    #         step = 0
    #         while step < adapter_steps:
    #             for Phi_b, r_b in targ_dl:
    #                 optimizer.zero_grad()
    #                 # g_theta 输入：z_src, eta  -> 输出：D 维向量
    #                 # 需要 adapter 支持批外生条件，可实现为：g([z_src, eta])，与 batch 无关
    #                 z_refined = self.adapter(z_src_tensor, eta)  # 1 x D
    #                 # 预测奖励: Phi_b (B x D) @ z_refined^T (D x 1) -> (B x 1)
    #                 pred = Phi_b @ z_refined.transpose(0,1)      # B x 1
    #                 loss = F.mse_loss(pred, r_b)
    #                 loss.backward()
    #                 optimizer.step()

    #                 step += 1
    #                 if step >= adapter_steps:
    #                     break

    #         self.adapter.eval()
    #         with torch.no_grad():
    #             z_final = self.adapter(z_src_tensor, eta).squeeze(0)  # D,
    #             z_final = math.sqrt(D) * F.normalize(z_final, dim=0)  # 归一化 + 尺度
    #     else:
    #         # 只做 (a)，不做 (b)(c)：直接把 z_src 当作最终 z
    #         z_final = z_src  # 已经 normalize * sqrt(D)

    #     # --------------------------
    #     # 4) 打包返回
    #     # --------------------------
    #     meta = OrderedDict()
    #     meta["z"] = z_final.detach().cpu().numpy()
    #     return meta

    def infer_meta_from_obs_and_rewards_sim2real(
    self,
    obs: torch.Tensor,          # 模拟域 N×...   (source / sim)
    reward: torch.Tensor,       # 模拟域 N×1 or N
    next_obs: torch.Tensor,     # 模拟域 N×...

    obs_targ: torch.Tensor,     # 真实域 M×...   (target / real)
    reward_targ: torch.Tensor,  # 真实域 M×1 or M
    next_obs_targ: torch.Tensor,# 真实域 M×...
    vis: bool = False
    ):
        vis = True
        """
        Sim->Real 推理期自适应：
        (a) 加权联合最小二乘，得到 z_src
        (b) 用 DeepSet(phi(s_targ)) 得到环境统计 η
        (c) 用 Adapter g_theta(z_src, η) 做轻量微调，得到 z_final
        返回 meta["z"] (numpy), 做了 L2 归一化并乘 sqrt(D)
        """
        device = obs.device
        D = int(self.cfg.z_dim)

        # --------------------------
        # 0) 编码观测 -> 表征 phi
        # --------------------------
        with torch.no_grad():
            enc_sim     = self.encoder(obs)           # N x ...
            enc_sim_nxt = self.encoder(next_obs)      # N x ...
            enc_targ    = self.encoder(obs_targ)      # M x ...
            enc_targ_nxt= self.encoder(next_obs_targ) # M x ...

        # helper: 从编码特征构造 phi 矩阵 (对齐你原有的 feature_type 逻辑)
        def build_phi(encoded, encoded_next):
            if self.cfg.feature_type == "state":
                return self.feature_learner.feature_net(encoded)  # [*, D]
            elif self.cfg.feature_type == "diff":
                return (self.feature_learner.feature_net(encoded_next)
                        - self.feature_learner.feature_net(encoded))  # [*, D]
            else:  # "concat"
                return torch.cat(
                    [self.feature_learner.feature_net(encoded),
                    self.feature_learner.feature_net(encoded_next)],
                    dim=-1
                )  # 注意：若 concat，D 应与 cfg.z_dim 一致

        with torch.no_grad():
            Phi_sim  = build_phi(enc_sim, enc_sim_nxt)          # N x D
            Phi_targ = build_phi(enc_targ, enc_targ_nxt)        # M x D

            # 奖励向量 reshape 成 (N,1)/(M,1)
            r_sim   = reward.reshape(-1, 1).to(device)          # N x 1
            r_targ  = reward_targ.reshape(-1, 1).to(device)     # M x 1

        # --------------------------
        # 1) (a) 加权联合最小二乘  (式 (5))
        # --------------------------
        lam = float(getattr(self.cfg, "lambda_wls", 4.0))
        lam_sqrt = math.sqrt(lam)

        with torch.no_grad():
            A = torch.cat([Phi_sim,  lam_sqrt * Phi_targ], dim=0)  # (N+M) x D
            b = torch.cat([r_sim,    lam_sqrt * r_targ],  dim=0)   # (N+M) x 1

            # 最小二乘解 ẑ = (A^T A)^(-1) A^T b
            z_src = torch.linalg.lstsq(A, b).solution  # D x 1
            # 归一化 + 尺度匹配
            z_src = math.sqrt(D) * F.normalize(z_src.squeeze(-1), dim=0)  # (D,)

        # ===== 可视化/IO 准备（仅在 vis=True 时） =====
        if vis:
            from pathlib import Path
            import numpy as np
            import matplotlib.pyplot as plt

            # 统一字体配置
            FONT_CFG = {
                "font.family": "Times New Roman",
                "font.size": 16,
                "axes.titlesize": 18,
                "axes.labelsize": 16,
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
                "legend.fontsize": 18,
                "figure.titlesize": 20,
            }
            plt.rcParams.update(FONT_CFG)

            save_root = Path("/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/0official/lossanalysis")
            save_root.mkdir(parents=True, exist_ok=True)

            # 简单的一维向量 heatmap
            def plot_vector_heat(vec_np, title, out_path):
                v = vec_np.astype(np.float32)
                vmax = float(np.max(np.abs(v))) + 1e-8
                fig, ax = plt.subplots(figsize=(10, 1.6))
                im = ax.imshow(v[np.newaxis, :], aspect='auto', cmap='RdBu_r',
                            interpolation='nearest', vmin=-vmax, vmax=vmax)
                ax.set_yticks([]); ax.set_xlabel("dimension index"); ax.set_title(title)
                # 稀疏刻度（可读性）
                D_loc = v.size
                step = max(1, D_loc // 8)
                ax.set_xticks(np.arange(0, D_loc, step))
                ax.set_xticklabels([str(i) for i in range(0, D_loc, step)])
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.025, pad=0.02)
                cbar.set_label("value")
                fig.tight_layout(); fig.savefig(out_path, dpi=220); plt.close(fig)

            # 画 loss 曲线
            def plot_loss_curve(loss_arr, out_path, title="Adapter loss (MSE) vs step"):
                fig, ax = plt.subplots(figsize=(6.4, 3.6))
                ax.plot(np.arange(1, len(loss_arr)+1), loss_arr, lw=1.5)
                ax.set_xlabel("step"); ax.set_ylabel("loss (MSE)"); ax.set_title(title)
                ax.grid(alpha=0.3, linestyle='--', linewidth=0.7)
                fig.tight_layout(); fig.savefig(out_path, dpi=220); plt.close(fig)
        # ============================================

        # --------------------------
        # 2) (b) DeepSet 统计 η = DeepSet({phi(s_targ)})
        # --------------------------
        if getattr(self.cfg, "use_adapter", True):
            with torch.no_grad():
                phi_state_targ = self.feature_learner.feature_net(enc_targ)  # M x D_state(=D if一致)
                eta = self.deepset(phi_state_targ)  # K 维向量
                if eta.dim() == 1:
                    eta = eta.unsqueeze(0)          # 1 x K
        else:
            eta = None  # 不用 adapter

        # --------------------------
        # 3) (c) Adapter: g_theta(z_src, η) -> refine z
        # --------------------------
        if getattr(self.cfg, "use_adapter", True):
            adapter_lr = float(getattr(self.cfg, "adapter_lr", 1e-3))
            adapter_steps = int(getattr(self.cfg, "adapter_steps", 300))
            adapter_bs = int(getattr(self.cfg, "adapter_batch_size", 256))

            targ_ds = TensorDataset(Phi_targ.detach(), r_targ.detach())
            targ_dl = DataLoader(targ_ds, batch_size=min(adapter_bs, len(targ_ds)), shuffle=True, drop_last=False)

            z_src_tensor = z_src.detach().unsqueeze(0)  # 1 x D

            self.adapter.train()
            optimizer = torch.optim.Adam(self.adapter.parameters(), lr=adapter_lr)

            # ===== 记录 loss =====
            loss_log = []  # 按 step 追加
            # ======================

            step = 0
            while step < adapter_steps:
                for Phi_b, r_b in targ_dl:
                    optimizer.zero_grad()
                    z_refined = self.adapter(z_src_tensor, eta)   # 1 x D
                    pred = Phi_b @ z_refined.transpose(0,1)       # B x 1
                    loss = F.mse_loss(pred, r_b)
                    loss.backward()
                    optimizer.step()

                    # ===== 记录本步 loss =====
                    loss_log.append(float(loss.item()))
                    # ========================

                    step += 1
                    if step >= adapter_steps:
                        break

            self.adapter.eval()
            with torch.no_grad():
                z_final = self.adapter(z_src_tensor, eta).squeeze(0)  # D,
                z_final = math.sqrt(D) * F.normalize(z_final, dim=0)  # 归一化 + 尺度

            # ===== 落盘 CSV + 读回并可视化（loss 曲线） =====
            if vis:
                # 写 CSV（单列：loss）
                csv_path = save_root / f"loss_steps{adapter_steps}.csv"
                with open(csv_path, "w") as f:
                    f.write("loss\n")
                    for v in loss_log:
                        f.write(f"{v}\n")

                # 读回 CSV（再画图）
                import numpy as np
                loaded = np.loadtxt(csv_path, dtype=float, delimiter=",", skiprows=1)
                if loaded.ndim == 0:  # 只有一个值的极端情况
                    loaded = np.array([float(loaded)])
                plot_loss_curve(loaded, str(save_root / f"loss_curve_steps{adapter_steps}.png"))

            # ===== 可视化（向量 heatmap，各保存一张） =====
            if vis:
                z_src_np = z_src.detach().cpu().numpy()
                z_fin_np = z_final.detach().cpu().numpy()
                plot_vector_heat(z_src_np, "z_src (normalized)",
                                str(save_root / f"z_src_heat_steps{adapter_steps}.png"))
                plot_vector_heat(z_fin_np, "z_refined (last)",
                                str(save_root / f"z_refined_heat_steps{adapter_steps}.png"))

        else:
            # 只做 (a)，不做 (b)(c)：直接把 z_src 当作最终 z
            z_final = z_src  # 已经 normalize * sqrt(D)
            if vis:
                # 仅保存 z_src heatmap；无 loss 记录
                z_src_np = z_src.detach().cpu().numpy()
                plot_vector_heat(z_src_np, "z_src (no-adapter)",
                                str(save_root / f"z_src_heat_steps0.png"))

        # --------------------------
        # 4) 打包返回
        # --------------------------
        meta = OrderedDict()
        meta["z"] = z_final.detach().cpu().numpy()
        return meta



    # * TODO: original version: Verified!!!!!!!!

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


    # def calculate_difference(
    #     self, 
    #     obs: torch.Tensor,
    #     reward: torch.Tensor,
    #     next_obs: torch.Tensor,
    #     obs_targ: torch.Tensor,
    #     reward_targ: torch.Tensor,
    #     next_obs_targ: torch.Tensor,
    #     vis: bool = False
    # ):
    #     """
    #     Captures discrepancies in dynamics and reward transitions between two environments:
    #     1. Align each observation in Env1 with top-k most similar observations in Env2.
    #     2. Among top-k, match using closest reward.
    #     3. Calculate differences: observation, reward, and next-state (transition).
    #     """
    #     k = 100  # Number of nearest neighbors to consider

    #     # Ensure compatibility with tensor operations
    #     obs, obs_targ = obs.float(), obs_targ.float()
    #     reward, reward_targ = reward.squeeze(-1), reward_targ.squeeze(-1)

    #     # Pairwise distances between obs and obs_targ
    #     obs_distances = torch.cdist(obs, obs_targ, p=2)  # [N, M]
    #     most_similar_obs_indices = torch.topk(obs_distances, k, largest=False, dim=1).indices

    #     # Store differences
    #     obs_differences, reward_differences, transition_differences = [], [], []

    #     for i in range(obs.shape[0]):
    #         # Find top-k closest observations in obs_targ
    #         similar_indices = most_similar_obs_indices[i]
    #         reward_candidates = reward_targ[similar_indices]
    #         next_obs_candidates = next_obs_targ[similar_indices]

    #         # Match based on reward proximity
    #         reward_diff = torch.abs(reward_candidates - reward[i])
    #         best_match_idx = torch.argmin(reward_diff)
    #         matched_index = similar_indices[best_match_idx]

    #         # Calculate differences
    #         obs_difference = torch.norm(obs[i] - obs_targ[matched_index], p=2).item()
    #         reward_difference = torch.abs(reward[i] - reward_targ[matched_index]).item()
    #         transition_difference = torch.norm(next_obs[i] - next_obs_targ[matched_index], p=2).item()

    #         obs_differences.append(obs_difference)
    #         reward_differences.append(reward_difference)
    #         transition_differences.append(transition_difference)

    #     # Aggregate statistics
    #     mean_obs_diff = sum(obs_differences) / len(obs_differences)
    #     mean_reward_diff = sum(reward_differences) / len(reward_differences)
    #     mean_transition_diff = sum(transition_differences) / len(transition_differences)

    #     # Optional visualization
    #     if vis:
    #         import matplotlib.pyplot as plt
    #         os.makedirs("/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/vis", exist_ok=True)
    #         indices = range(len(obs_differences))

    #         plt.figure(figsize=(12, 8))
    #         plt.plot(indices, obs_differences, label="Observation Differences", alpha=0.7)
    #         plt.plot(indices, reward_differences, label="Reward Differences", alpha=0.7)
    #         plt.plot(indices, transition_differences, label="Transition Differences", alpha=0.7)

    #         plt.xlabel("Sample Index")
    #         plt.ylabel("Difference Value")
    #         plt.title(
    #             f"Discrepancy Metrics (Mean Obs: {mean_obs_diff:.3f}, "
    #             f"Mean Reward: {mean_reward_diff:.3f}, Mean Transition: {mean_transition_diff:.3f})"
    #         )
    #         plt.legend()
    #         plt.grid()
    #         plt.savefig("/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/vis/dynamics_discrepancy.png", dpi=300)
    #         plt.close()

    #     # Return transition differences for policy adaptation
    #     return transition_differences



    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, obs_targ: torch.Tensor, reward_targ: torch.Tensor, next_obs_targ: torch.Tensor
    #     ):
    #     with torch.no_grad():
    #         obs = self.encoder(obs) # transform into a latent feature space (phi_obs)
    #         next_obs = self.encoder(next_obs) # (phi_nextObs)

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

    #     # Calculate the discrepancy matrix using the provided method
    #     diff_matrix = self.calculate_difference(obs, reward, next_obs, obs_targ, reward_targ, next_obs_targ, vis=True) # torch.Size([10000, 24])

    #     # Normalize the diff_matrix for stability
    #     diff_matrix = torch.tensor(diff_matrix)
    #     smoothed_diff_matrix = torch.nn.functional.avg_pool1d(diff_matrix.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
    #     normalized_diff_matrix = smoothed_diff_matrix / (smoothed_diff_matrix.max() + 1e-6)
    #     normalized_diff_matrix = normalized_diff_matrix.to(phi.device)

    #     # + 0.1
    #     # normalized_diff_matrix = torch.where(normalized_diff_matrix > 0.1, normalized_diff_matrix, torch.zeros_like(normalized_diff_matrix))

    #     # Create the delta matrix (diagonal) as described in the image
    #     discrepancy_weights = torch.matmul(phi.T, normalized_diff_matrix.unsqueeze(1)).squeeze()
    #     delta = torch.diag(discrepancy_weights)

    #     # Compute terms for policy guidance
    #     phi_t_phi = torch.matmul(phi.T, phi)  # Covariance matrix of features
    #     phi_t_r = torch.matmul(phi.T, reward)  # Correlation of features with rewards
    #     lambda_delta = 0.001 * delta  # Regularization term

    #     # Solve for z* using the formula from the image
    #     z_star = torch.linalg.solve(phi_t_phi + lambda_delta, phi_t_r)

    #     # Prepare meta information
    #     meta = OrderedDict()
    #     meta["z"] = z_star.squeeze().cpu().numpy()  # Convert z* to a numpy array for further use

    #     return meta
    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, 
    #     obs_targ: torch.Tensor, reward_targ: torch.Tensor, next_obs_targ: torch.Tensor
    # ):
    #     """
    #     Zero-shot adaptation of policy by leveraging observed discrepancies between source
    #     (Env1) and target (Env2) dynamics during inference.
    #     """
    #     with torch.no_grad():
    #         obs = self.encoder(obs)
    #         next_obs = self.encoder(next_obs)

    #     # Feature extraction
    #     with torch.no_grad():
    #         if self.cfg.feature_type == "state":
    #             phi = self.feature_learner.feature_net(obs)
    #         elif self.cfg.feature_type == "diff":
    #             phi = self.feature_learner.feature_net(next_obs) - self.feature_learner.feature_net(obs)
    #         else:
    #             phi = torch.cat(
    #                 [self.feature_learner.feature_net(obs), self.feature_learner.feature_net(next_obs)], dim=-1
    #             )

    #     # Calculate discrepancies
    #     diff_matrix = self.calculate_difference(obs, reward, next_obs, obs_targ, reward_targ, next_obs_targ, vis=True)
    #     diff_matrix = torch.tensor(diff_matrix, device=phi.device)

    #     # Normalize discrepancies for stability
    #     normalized_diff_matrix = diff_matrix / (diff_matrix.max() + 1e-6)
    #     discrepancy_weights = torch.matmul(phi.T, normalized_diff_matrix.unsqueeze(1)).squeeze()

    #     # Create delta regularization
    #     delta = torch.diag(discrepancy_weights)

    #     # Solve for optimal policy parameters z*
    #     phi_t_phi = torch.matmul(phi.T, phi)
    #     phi_t_r = torch.matmul(phi.T, reward)
    #     lambda_delta = 0.01 * delta  # Soft regularization
    #     z_star = torch.linalg.solve(phi_t_phi + lambda_delta, phi_t_r)

    #     # Return meta parameters
    #     meta = OrderedDict()
    #     meta["z"] = z_star.squeeze().cpu().numpy()
    #     return meta


    # working 1: basic KL divergence:


    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, 
    #     obs_targ: torch.Tensor, reward_targ: torch.Tensor, next_obs_targ: torch.Tensor
    # ):
    #     """
    #     Zero-shot adaptation of policy by leveraging discrepancies between Env1 and Env2.
    #     """
    #     with torch.no_grad():
    #         # Encode observations into latent space (shared encoder)
    #         obs = self.encoder(obs)  # Source environment latent features
    #         next_obs = self.encoder(next_obs)
    #         obs_targ = self.encoder(obs_targ)  # Target environment latent features
    #         next_obs_targ = self.encoder(next_obs_targ)

    #         # Feature extraction using feature learner (phi computation)
    #         if self.cfg.feature_type == "state":
    #             phi = self.feature_learner.feature_net(obs)
    #             phi_targ = self.feature_learner.feature_net(obs_targ)
    #         elif self.cfg.feature_type == "diff":
    #             phi = self.feature_learner.feature_net(next_obs) - self.feature_learner.feature_net(obs)
    #             phi_targ = self.feature_learner.feature_net(next_obs_targ) - self.feature_learner.feature_net(obs_targ)
    #         else:
    #             phi = torch.cat(
    #                 [
    #                     self.feature_learner.feature_net(obs),
    #                     self.feature_learner.feature_net(next_obs),
    #                 ],
    #                 dim=-1,
    #             )
    #             phi_targ = torch.cat(
    #                 [
    #                     self.feature_learner.feature_net(obs_targ),
    #                     self.feature_learner.feature_net(next_obs_targ),
    #                 ],
    #                 dim=-1,
    #             )

    #     # Normalize phi and phi_targ for stability
    #     eps = 1e-8  # Small constant for stability
    #     phi_norm = F.normalize(phi, dim=1) + eps
    #     phi_targ_norm = F.normalize(phi_targ, dim=1) + eps

    #     # Clamp normalized values to avoid log(0)
    #     phi_norm = torch.clamp(phi_norm, min=eps)
    #     phi_targ_norm = torch.clamp(phi_targ_norm, min=eps)

    #     # Match dimensions for KL divergence
    #     if phi_norm.size(0) > phi_targ_norm.size(0):
    #         indices = torch.randint(0, phi_targ_norm.size(0), (phi_norm.size(0),), device=phi_targ_norm.device)
    #         phi_targ_norm_expanded = phi_targ_norm[indices]
    #     else:
    #         phi_norm = phi_norm[:phi_targ_norm.size(0)]
    #         phi_targ_norm_expanded = phi_targ_norm

    #     # Calculate KL divergence
    #     try:
    #         kl_divergence = torch.sum(
    #             phi_norm * (phi_norm.log() - phi_targ_norm_expanded.log()), dim=1
    #         ).mean().item()
    #     except Exception as e:
    #         kl_divergence = float('nan')
    #         print(f"KL Divergence calculation error: {e}")

    #     # Check and replace NaNs in KL divergence
    #     if not (kl_divergence > 0):  # Handle cases where KL divergence is NaN or negative
    #         kl_divergence = 0.0

    #     # Compute the latent vector z (minimizing the reward prediction loss)
    #     z = torch.linalg.lstsq(phi, reward).solution

    #     # Regularize z using the KL divergence, with safe scaling
    #     print("kl_divergence:::::")
    #     print(kl_divergence)

    #     if kl_divergence > 0:  # Ensure KL divergence is positive
    #         z = z - 0.001 * kl_divergence * z

    #     # Normalize z for numerical stability
    #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)

    #     # Prepare meta information for policy guidance
    #     meta = OrderedDict()
    #     meta["z"] = z.squeeze().cpu().numpy()
    #     meta["kl_divergence"] = kl_divergence  # Add KL divergence as diagnostic information

    #     return meta



    # golden version: 
    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, 
    #         obs_targ: torch.Tensor, reward_targ: torch.Tensor, next_obs_targ: torch.Tensor
    #     ):
    #     """
    #     Zero-shot adaptation of policy by leveraging discrepancies between Env1 and Env2.
    #     """
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

    #     # add noise to z:
    #     # noise = torch.rand_like(z) * 10.0 - 5.0  # Uniform noise in range [-5, 5]
    #     # z += noise
    #     # finish nosie


    #     meta = OrderedDict()
    #     meta["z"] = z.squeeze().cpu().numpy()
    #     return meta


        # new methos;
        # with torch.no_grad():
        #     obs = self.encoder(obs_targ)
        #     next_obs = self.encoder(next_obs_targ)

        # with torch.no_grad():
        #     if self.cfg.feature_type == "state":
        #         # This is the representation function: phi (trained network)
        #         phi = self.feature_learner.feature_net(obs)
        #     elif self.cfg.feature_type == "diff":
        #         # This is to get the difference between two observations
        #         phi = self.feature_learner.feature_net(
        #             next_obs
        #         ) - self.feature_learner.feature_net(obs)
        #     else:
        #         phi = torch.cat(
        #             [
        #                 self.feature_learner.feature_net(obs),
        #                 self.feature_learner.feature_net(next_obs),
        #             ],
        #             dim=-1,
        #         )
        # # this will calculate the result z:
        # # computes the vector z that minimizes the squared Euclidean norm of the matrix-vector product phi * z - reward
        # z = torch.linalg.lstsq(phi, reward_targ).solution

        # # preparing parameters in a NN, ensuring they are normalized and scaled appropriately for the task (given the dim) or the architecture's expectations
        # z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)

        # # add noise to z:
        # # noise = torch.rand_like(z) * 10.0 - 5.0  # Uniform noise in range [-5, 5]
        # # z += noise
        # # finish nosie


        # meta = OrderedDict()
        # meta["z"] = z.squeeze().cpu().numpy()
        # return meta


    # 25_v1:

    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, 
    #         obs_targ: torch.Tensor, reward_targ: torch.Tensor, next_obs_targ: torch.Tensor
    #     ):
    #     """
    #     Zero-shot adaptation of policy using observations only from the target environment.
    #     """
    #     with torch.no_grad():
    #         # Encode original and target observations
    #         obs = self.encoder(obs)  # Original env latent features
    #         next_obs = self.encoder(next_obs)
    #         obs_targ = self.encoder(obs_targ)  # Target env latent features
    #         next_obs_targ = self.encoder(next_obs_targ)

    #         # Feature extraction
    #         if self.cfg.feature_type == "state":
    #             phi_origin = self.feature_learner.feature_net(obs)
    #             phi_targ = self.feature_learner.feature_net(obs_targ)
    #         elif self.cfg.feature_type == "diff":
    #             phi_origin = self.feature_learner.feature_net(next_obs) - self.feature_learner.feature_net(obs)
    #             phi_targ = self.feature_learner.feature_net(next_obs_targ) - self.feature_learner.feature_net(obs_targ)
    #         else:
    #             phi_origin = torch.cat(
    #                 [self.feature_learner.feature_net(obs), self.feature_learner.feature_net(next_obs)], dim=-1
    #             )
    #             phi_targ = torch.cat(
    #                 [self.feature_learner.feature_net(obs_targ), self.feature_learner.feature_net(next_obs_targ)], dim=-1
    #             )

    #     # Normalize embeddings for stability
    #     phi_origin = F.normalize(phi_origin, dim=1)
    #     phi_targ = F.normalize(phi_targ, dim=1)

    #     # Match dimensions for discrepancy computation
    #     if phi_origin.size(0) > phi_targ.size(0):
    #         indices = torch.randint(0, phi_targ.size(0), (phi_origin.size(0),), device=phi_targ.device)
    #         phi_targ = phi_targ[indices]
    #     else:
    #         phi_origin = phi_origin[:phi_targ.size(0)]

    #     # Compute discrepancy (e.g., MMD or cosine similarity)
    #     def gaussian_kernel(x, y, sigma=1.0):
    #         beta = 1 / (2 * sigma**2)
    #         dist = torch.cdist(x, y, p=2)  # Pairwise Euclidean distances
    #         return torch.exp(-beta * dist**2)

    #     kernel_xx = gaussian_kernel(phi_origin, phi_origin)
    #     kernel_yy = gaussian_kernel(phi_targ, phi_targ)
    #     kernel_xy = gaussian_kernel(phi_origin, phi_targ)

    #     mmd = kernel_xx.mean() + kernel_yy.mean() - 2 * kernel_xy.mean()
    #     mmd_value = mmd.item()

    #     # Compute latent vector z (based only on original rewards and embeddings)
    #     z = torch.linalg.lstsq(phi_origin, reward).solution

    #     # Regularize z using the discrepancy
    #     z = z - 0.01 * mmd_value * z

    #     # Normalize z for stability
    #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)

    #     # Prepare meta information
    #     meta = OrderedDict()
    #     meta["z"] = z.squeeze().cpu().numpy()
    #     meta["mmd"] = mmd_value  # Add MMD as diagnostic information

    #     return meta


    # 25_v2:
    import torch
    import torch.nn.functional as F
    from collections import OrderedDict

    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, 
    #         obs_targ: torch.Tensor, reward_targ: torch.Tensor, next_obs_targ: torch.Tensor
    #     ):
    #     """
    #     Zero-shot adaptation of policy using observations only from the target environment.
    #     """
    #     obs = self.encoder(obs)  # Original env latent features
    #     next_obs = self.encoder(next_obs)
    #     obs_targ = self.encoder(obs_targ)
    #     next_obs_targ = self.encoder(next_obs_targ)

    #     # Feature extraction
    #     phi_origin = self.feature_learner.feature_net(obs)
    #     phi_targ = self.feature_learner.feature_net(obs_targ)

    #     # Dynamics modeling (self-supervised learning)
    #     dynamics_model = torch.nn.Linear(phi_targ.size(1), next_obs_targ.size(1)).to(phi_targ.device)
    #     optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=1e-3)

    #     # Train dynamics model
    #     for _ in range(50):  # Train for a few iterations
    #         predicted_next_obs = dynamics_model(phi_targ)
    #         loss = F.mse_loss(predicted_next_obs, next_obs_targ)  # Ensure requires_grad=True
    #         optimizer.zero_grad()
    #         loss.backward(retain_graph=True)  # Retain the graph for subsequent backward passes
    #         optimizer.step()

    #     dynamics_loss = loss.item()

    #     # Normalize embeddings for stability
    #     phi_origin = F.normalize(phi_origin, dim=1)
    #     phi_targ = F.normalize(phi_targ, dim=1)

    #     # Domain-specific weighting based on similarity
    #     similarities = torch.mm(phi_targ, phi_origin.T)  # Cosine similarity
    #     weights = F.softmax(similarities, dim=1)  # Normalize weights

    #     weighted_phi_origin = torch.mm(weights, phi_origin)

    #     # Contrastive learning loss
    #     temperature = 0.1
    #     contrastive_similarities = torch.mm(phi_origin, phi_targ.T) / temperature
    #     contrastive_labels = torch.arange(phi_origin.size(0), device=phi_origin.device)
    #     contrastive_loss = F.cross_entropy(contrastive_similarities, contrastive_labels)

    #     # Compute Maximum Mean Discrepancy (MMD) with Gaussian kernel
    #     def gaussian_kernel(x, y, sigma=1.0):
    #         beta = 1 / (2 * sigma**2)
    #         dist = torch.cdist(x, y, p=2)  # Pairwise Euclidean distances
    #         return torch.exp(-beta * dist**2)

    #     kernel_xx = gaussian_kernel(phi_origin, phi_origin)
    #     kernel_yy = gaussian_kernel(phi_targ, phi_targ)
    #     kernel_xy = gaussian_kernel(phi_origin, phi_targ)

    #     mmd = kernel_xx.mean() + kernel_yy.mean() - 2 * kernel_xy.mean()
    #     mmd_value = mmd.item()

    #     # Compute latent vector z (based only on original rewards and embeddings)
    #     z = torch.linalg.lstsq(weighted_phi_origin, reward).solution

    #     # Regularize z using dynamics loss, MMD, and contrastive loss
    #     z = z - 0.01 * dynamics_loss * z
    #     z = z - 0.01 * mmd_value * z
    #     z = z - 0.01 * contrastive_loss.item() * z

    #     # Normalize z for stability
    #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)

    #     # Prepare meta information
    #     meta = OrderedDict()
    #     meta["z"] = z.squeeze().detach().cpu().numpy()
    #     meta["dynamics_loss"] = dynamics_loss
    #     meta["mmd"] = mmd_value
    #     meta["contrastive_loss"] = contrastive_loss.item()

    #     return meta

    # # JS divergence: 
    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, 
    #         obs_targ: torch.Tensor, reward_targ: torch.Tensor, next_obs_targ: torch.Tensor
    #     ):
    #     """
    #     Zero-shot adaptation of policy by leveraging discrepancies between Env1 and Env2.
    #     """
    #     with torch.no_grad():
    #         # Encode observations into latent space (shared encoder)
    #         obs = self.encoder(obs)  # Source environment latent features
    #         next_obs = self.encoder(next_obs)
    #         obs_targ = self.encoder(obs_targ)  # Target environment latent features
    #         next_obs_targ = self.encoder(next_obs_targ)

    #         # Feature extraction using feature learner (phi computation)
    #         if self.cfg.feature_type == "state":
    #             phi = self.feature_learner.feature_net(obs)
    #             phi_targ = self.feature_learner.feature_net(obs_targ)
    #         elif self.cfg.feature_type == "diff":
    #             phi = self.feature_learner.feature_net(next_obs) - self.feature_learner.feature_net(obs)
    #             phi_targ = self.feature_learner.feature_net(next_obs_targ) - self.feature_learner.feature_net(obs_targ)
    #         else:
    #             phi = torch.cat(
    #                 [
    #                     self.feature_learner.feature_net(obs),
    #                     self.feature_learner.feature_net(next_obs),
    #                 ],
    #                 dim=-1,
    #             )
    #             phi_targ = torch.cat(
    #                 [
    #                     self.feature_learner.feature_net(obs_targ),
    #                     self.feature_learner.feature_net(next_obs_targ),
    #                 ],
    #                 dim=-1,
    #             )

    #     # Normalize phi and phi_targ for stability
    #     eps = 1e-8  # Small constant for stability
    #     phi_norm = F.normalize(phi, dim=1) + eps
    #     phi_targ_norm = F.normalize(phi_targ, dim=1) + eps

    #     # Clamp normalized values to avoid log(0)
    #     phi_norm = torch.clamp(phi_norm, min=eps)
    #     phi_targ_norm = torch.clamp(phi_targ_norm, min=eps)

    #     # Match dimensions for JS divergence
    #     if phi_norm.size(0) > phi_targ_norm.size(0):
    #         indices = torch.randint(0, phi_targ_norm.size(0), (phi_norm.size(0),), device=phi_targ_norm.device)
    #         phi_targ_norm_expanded = phi_targ_norm[indices]
    #     else:
    #         phi_norm = phi_norm[:phi_targ_norm.size(0)]
    #         phi_targ_norm_expanded = phi_targ_norm

    #     # Compute the midpoint distribution
    #     midpoint = 0.5 * (phi_norm + phi_targ_norm_expanded)

    #     # Calculate JS divergence
    #     try:
    #         kl_phi_midpoint = torch.sum(phi_norm * (phi_norm.log() - midpoint.log()), dim=1).mean()
    #         kl_targ_midpoint = torch.sum(phi_targ_norm_expanded * (phi_targ_norm_expanded.log() - midpoint.log()), dim=1).mean()
    #         js_divergence = 0.5 * (kl_phi_midpoint + kl_targ_midpoint).item()
    #     except Exception as e:
    #         js_divergence = float('nan')
    #         print(f"JS Divergence calculation error: {e}")

    #     # Check and replace NaNs in JS divergence
    #     if not (js_divergence > 0):  # Handle cases where JS divergence is NaN or negative
    #         js_divergence = 0.0

    #     # Compute the latent vector z (minimizing the reward prediction loss)
    #     z = torch.linalg.lstsq(phi, reward).solution

    #     # Regularize z using the JS divergence, with safe scaling
    #     if js_divergence > 0:  # Ensure JS divergence is positive
    #         z = z - 0.1 * js_divergence * z

    #     # Normalize z for numerical stability
    #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)

    #     # Prepare meta information for policy guidance
    #     meta = OrderedDict()
    #     meta["z"] = z.squeeze().cpu().numpy()
    #     meta["js_divergence"] = js_divergence  # Add JS divergence as diagnostic information

    #     return meta

    # MMD: 
    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, 
    #     obs_targ: torch.Tensor, reward_targ: torch.Tensor, next_obs_targ: torch.Tensor
    #     ):
    #     """
    #     Zero-shot adaptation of policy by leveraging discrepancies between Env1 and Env2.
    #     """
    #     with torch.no_grad():
    #         # Encode observations into latent space (shared encoder)
    #         obs = self.encoder(obs)  # Source environment latent features
    #         next_obs = self.encoder(next_obs)
    #         obs_targ = self.encoder(obs_targ)  # Target environment latent features
    #         next_obs_targ = self.encoder(next_obs_targ)

    #         # Feature extraction using feature learner (phi computation)
    #         if self.cfg.feature_type == "state":
    #             phi = self.feature_learner.feature_net(obs)
    #             phi_targ = self.feature_learner.feature_net(obs_targ)
    #         elif self.cfg.feature_type == "diff":
    #             phi = self.feature_learner.feature_net(next_obs) - self.feature_learner.feature_net(obs)
    #             phi_targ = self.feature_learner.feature_net(next_obs_targ) - self.feature_learner.feature_net(obs_targ)
    #         else:
    #             phi = torch.cat(
    #                 [
    #                     self.feature_learner.feature_net(obs),
    #                     self.feature_learner.feature_net(next_obs),
    #                 ],
    #                 dim=-1,
    #             )
    #             phi_targ = torch.cat(
    #                 [
    #                     self.feature_learner.feature_net(obs_targ),
    #                     self.feature_learner.feature_net(next_obs_targ),
    #                 ],
    #                 dim=-1,
    #             )

    #     # Normalize phi and phi_targ for stability
    #     eps = 1e-8  # Small constant for stability
    #     phi_norm = F.normalize(phi, dim=1) + eps
    #     phi_targ_norm = F.normalize(phi_targ, dim=1) + eps

    #     # Match dimensions for MMD computation
    #     if phi_norm.size(0) > phi_targ_norm.size(0):
    #         indices = torch.randint(0, phi_targ_norm.size(0), (phi_norm.size(0),), device=phi_targ_norm.device)
    #         phi_targ_norm = phi_targ_norm[indices]
    #     else:
    #         phi_norm = phi_norm[:phi_targ_norm.size(0)]

    #     # Compute Maximum Mean Discrepancy (MMD) with Gaussian kernel
    #     def gaussian_kernel(x, y, sigma=1.0):
    #         beta = 1 / (2 * sigma**2)
    #         dist = torch.cdist(x, y, p=2)  # Pairwise Euclidean distances
    #         return torch.exp(-beta * dist**2)

    #     kernel_xx = gaussian_kernel(phi_norm, phi_norm)  # Source vs Source
    #     kernel_yy = gaussian_kernel(phi_targ_norm, phi_targ_norm)  # Target vs Target
    #     kernel_xy = gaussian_kernel(phi_norm, phi_targ_norm)  # Source vs Target

    #     mmd = kernel_xx.mean() + kernel_yy.mean() - 2 * kernel_xy.mean()

    #     # Convert MMD to a scalar value
    #     mmd_value = mmd.item()

    #     # Compute the latent vector z (minimizing the reward prediction loss)
    #     z = torch.linalg.lstsq(phi, reward).solution

    #     # Regularize z using the MMD, with safe scaling
    #     if mmd_value > 0:  # Ensure MMD is positive
    #         z = z - 0.1 * mmd_value * z

    #     # Normalize z for numerical stability
    #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)

    #     # Prepare meta information for policy guidance
    #     meta = OrderedDict()
    #     meta["z"] = z.squeeze().cpu().numpy()
    #     meta["mmd"] = mmd_value  # Add MMD as diagnostic information

    #     return meta




    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, 
    #     obs_targ: torch.Tensor, reward_targ: torch.Tensor, next_obs_targ: torch.Tensor
    # ):
    #     """
    #     Zero-shot adaptation of policy by leveraging discrepancies between Env1 and Env2.
    #     Includes diversity regularization.
    #     """
    #     with torch.no_grad():
    #         # Encode observations into latent space (shared encoder)
    #         obs = self.encoder(obs)
    #         next_obs = self.encoder(next_obs)
    #         obs_targ = self.encoder(obs_targ)
    #         next_obs_targ = self.encoder(next_obs_targ)

    #         # Feature extraction using feature learner (phi computation)
    #         phi = self.feature_learner.feature_net(obs)
    #         phi_targ = self.feature_learner.feature_net(obs_targ)

    #     # Normalize phi and phi_targ for stability
    #     eps = 1e-8
    #     phi_norm = F.normalize(phi, dim=1) + eps
    #     phi_targ_norm = F.normalize(phi_targ, dim=1) + eps

    #     # Clamp normalized values to avoid log(0)
    #     phi_norm = torch.clamp(phi_norm, min=eps)
    #     phi_targ_norm = torch.clamp(phi_targ_norm, min=eps)

    #     # Match dimensions for dynamics discrepancy
    #     if next_obs.size(0) > next_obs_targ.size(0):
    #         indices = torch.randint(0, next_obs_targ.size(0), (next_obs.size(0),), device=next_obs.device)
    #         next_obs_targ_expanded = next_obs_targ[indices]
    #     else:
    #         next_obs = next_obs[:next_obs_targ.size(0)]
    #         next_obs_targ_expanded = next_obs_targ

    #     # Compute dynamics discrepancy
    #     dynamics_discrepancy = torch.mean(torch.norm(next_obs - next_obs_targ_expanded, p=2, dim=1))

    #     # Calculate KL divergence
    #     if phi_norm.size(0) > phi_targ_norm.size(0):
    #         indices = torch.randint(0, phi_targ_norm.size(0), (phi_norm.size(0),), device=phi_targ_norm.device)
    #         phi_targ_norm_expanded = phi_targ_norm[indices]
    #     else:
    #         phi_norm = phi_norm[:phi_targ_norm.size(0)]
    #         phi_targ_norm_expanded = phi_targ_norm

    #     kl_divergence = torch.sum(
    #         phi_norm * (phi_norm.log() - phi_targ_norm_expanded.log()), dim=1
    #     ).mean().item()

    #     if not (kl_divergence > 0):
    #         kl_divergence = 0.0

    #     # Compute diversity regularization (maximize diversity among latent vectors)
    #     diversity_loss = 0
    #     num_samples = phi.size(0)
    #     for i in tqdm(range(num_samples)):
    #         for j in range(i + 1, num_samples):
    #             diversity_loss += F.cosine_similarity(phi[i], phi[j], dim=0)

    #     diversity_loss = diversity_loss / (num_samples * (num_samples - 1) / 2)  # Average similarity
    #     diversity_loss = 1.0 - diversity_loss  # Minimize similarity to promote diversity

    #     # Compute the latent vector z
    #     z = torch.linalg.lstsq(phi, reward).solution

    #     # Regularize z using KL divergence, dynamics discrepancy, and diversity
    #     lambda_kl = min(0.01, max(0.001, kl_divergence / 10.0))
    #     z = z - lambda_kl * kl_divergence * z
    #     z = z - 0.01 * dynamics_discrepancy * z
    #     z = z + 0.01 * diversity_loss * z  # Encourage diversity

    #     # Normalize z for numerical stability
    #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)

    #     # Prepare meta information
    #     meta = OrderedDict()
    #     meta["z"] = z.squeeze().cpu().numpy()
    #     meta["kl_divergence"] = kl_divergence
    #     meta["dynamics_discrepancy"] = dynamics_discrepancy.item()
    #     meta["diversity_loss"] = diversity_loss.item()

    #     return meta



    # contrasive loss, running, but no improve than above KL
    
    # def infer_meta_from_obs_and_rewards_sim2real(
    #     self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, 
    #     obs_targ: torch.Tensor, reward_targ: torch.Tensor, next_obs_targ: torch.Tensor
    # ):
    #     """
    #     Zero-shot adaptation of policy by leveraging discrepancies between Env1 and Env2.
    #     """
    #     with torch.no_grad():
    #         # Encode observations into latent space (shared encoder)
    #         obs = self.encoder(obs)
    #         next_obs = self.encoder(next_obs)
    #         obs_targ = self.encoder(obs_targ)
    #         next_obs_targ = self.encoder(next_obs_targ)

    #         # Feature extraction using feature learner (phi computation)
    #         phi = self.feature_learner.feature_net(obs)
    #         phi_targ = self.feature_learner.feature_net(obs_targ)

    #     # Normalize phi and phi_targ for stability
    #     eps = 1e-8
    #     phi_norm = F.normalize(phi, dim=1) + eps
    #     phi_targ_norm = F.normalize(phi_targ, dim=1) + eps

    #     # Clamp normalized values to avoid log(0)
    #     phi_norm = torch.clamp(phi_norm, min=eps)
    #     phi_targ_norm = torch.clamp(phi_targ_norm, min=eps)

    #     # Match dimensions for dynamics discrepancy
    #     if next_obs.size(0) > next_obs_targ.size(0):
    #         indices = torch.randint(0, next_obs_targ.size(0), (next_obs.size(0),), device=next_obs.device)
    #         next_obs_targ_expanded = next_obs_targ[indices]
    #     else:
    #         next_obs = next_obs[:next_obs_targ.size(0)]
    #         next_obs_targ_expanded = next_obs_targ

    #     # Compute dynamics discrepancy
    #     dynamics_discrepancy = torch.mean(torch.norm(next_obs - next_obs_targ_expanded, p=2, dim=1))

    #     # Calculate KL divergence
    #     if phi_norm.size(0) > phi_targ_norm.size(0):
    #         indices = torch.randint(0, phi_targ_norm.size(0), (phi_norm.size(0),), device=phi_targ_norm.device)
    #         phi_targ_norm_expanded = phi_targ_norm[indices]
    #     else:
    #         phi_norm = phi_norm[:phi_targ_norm.size(0)]
    #         phi_targ_norm_expanded = phi_targ_norm

    #     kl_divergence = torch.sum(
    #         phi_norm * (phi_norm.log() - phi_targ_norm_expanded.log()), dim=1
    #     ).mean().item()

    #     if not (kl_divergence > 0):
    #         kl_divergence = 0.0

    #     # Contrastive loss between source and target embeddings
    #     margin = 1.0  # Contrastive loss margin
    #     distances = torch.norm(phi - phi_targ_norm_expanded, p=2, dim=1)
    #     contrastive_loss = torch.mean(
    #         (distances**2) * (distances <= margin).float() +
    #         (margin - distances).clamp(min=0)**2 * (distances > margin).float()
    #     )

    #     # Compute the latent vector z
    #     z = torch.linalg.lstsq(phi, reward).solution

    #     # Regularize z using KL divergence, dynamics discrepancy, and contrastive loss
    #     lambda_kl = min(0.01, max(0.001, kl_divergence / 10.0))
    #     lambda_contrastive = 0.01  # Scaling factor for contrastive loss
    #     z = z - lambda_kl * kl_divergence * z
    #     z = z - 0.01 * dynamics_discrepancy * z
    #     z = z - lambda_contrastive * contrastive_loss * z

    #     # Normalize z for numerical stability
    #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)

    #     # Prepare meta information
    #     meta = OrderedDict()
    #     meta["z"] = z.squeeze().cpu().numpy()
    #     meta["kl_divergence"] = kl_divergence
    #     meta["dynamics_discrepancy"] = dynamics_discrepancy.item()
    #     meta["contrastive_loss"] = contrastive_loss.item()

    #     return meta

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
