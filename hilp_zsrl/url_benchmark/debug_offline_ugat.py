# test_offline: is used actual debug

import platform
import os
import sys
import tempfile
import typing as tp
import warnings
from pathlib import Path

import hydra
import numpy as np
import omegaconf as omgcf
import toml
import torch
import wandb
from dm_env import specs
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from runtime_paths import load_repo_parameters

CONTEXT, parameters = load_repo_parameters(__file__)
sys.path.append(f"{parameters['config']['path']}/hilp_zsrl/")

try:
    from dlc.utils import load_customized_config
except ModuleNotFoundError as exc:
    def load_customized_config(*args, **kwargs):
        raise ModuleNotFoundError(
            "Missing optional dependency: dlc. "
            "Install the DLC package or avoid code paths that require customized configs."
        ) from exc

from url_benchmark import agent as agents
from url_benchmark import dmc, utils
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.my_utils import record_video
from url_benchmark.video import VideoRecorder

if "mac" in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

from pathlib import Path
import sys
base = CONTEXT.project_dir
for fp in [base, base / "url_benchmark"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))

import logging
import torch
import warnings

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore', category=DeprecationWarning)

import json
import dataclasses
import tempfile
import typing as tp
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import wandb
import omegaconf as omgcf

from omegaconf import OmegaConf

from url_benchmark import dmc
from dm_env import specs
from url_benchmark import utils
from url_benchmark import agent as agents
from url_benchmark.logger import Logger
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.video import VideoRecorder
from url_benchmark.my_utils import record_video


@dataclasses.dataclass
class Config:
    agent: tp.Any
    # agent.feature_learner: str = "hilp"
    # agent.hilp_expectile: float = 0.5
    # agent.hilp_discount : float = 0.96
    # agent.q_loss: bool =False
    # misc
    run_group: str = "EXP"
    seed: int = 0
    device: str = "cuda"
    save_video: bool = True
    use_tb: bool = False
    use_wandb: bool = True
    # experiment
    experiment: str = "offline"
    # task settings
    task: str = "walker_run"
    obs_type: str = "states"  # [states, pixels]
    frame_stack: int = 3  # only works if obs_type=pixels
    image_wh: int = 64
    action_repeat: int = 1
    discount: float = 0.98
    future: float = 0.99  # discount of future sampling, future=1 means no future sampling
    p_currgoal: float = 0  # current goal ratio
    p_randomgoal: float = 0.375  # random goal ratio
    # eval
    num_eval_episodes: int = 1 # 
    num_final_eval_episodes: int = 1 #

    eval_every_steps: int = 10000
    video_every_steps: int = 100000
    num_skip_frames: int = 2
    custom_reward: tp.Optional[str] = None  # activates custom eval if not None
    # checkpoint
    snapshot_at: tp.Tuple[int, ...] = ()
    checkpoint_every: int = 100000
    load_model: tp.Optional[str] = None
    # training
    num_grad_steps: int = 1000000
    log_every_steps: int = 1000
    num_seed_frames: int = 0
    replay_buffer_episodes: int = 5000
    replay_buffer_env_target_episodes: int = 4000 # the episode value from the target envrionment
    update_encoder: bool = True
    batch_size: int = omgcf.II("agent.batch_size")
    goal_eval: bool = False
    # dataset: align, s
    load_replay_buffer: tp.Optional[str] = (
        f"{parameters['config']['path_exorl_learn']}/datasets/walker/rnd/replay.pt"  # from the command 
        # f"{parameters['config']['path_exorl_learn_earth_aligned']}/datasets/walker/rnd/replay.pt"  # from the command
    )
    # align gravity - 24
    # load_replay_buffer_env_target: tp.Optional[str] = (
    #     # f"{parameters['config']['path_exorl_learn_24_true_RMSPROP']}/datasets/walker/rnd/replay.pt"  # from the command
    #     f"{parameters['config']['path_exorl_learn_24_aligned']}/datasets/walker/rnd/replay.pt"  # from the command
    # )

    # align gravity - 44
    load_replay_buffer_env_target: tp.Optional[str] = ( 
        None
        # f"{parameters['config']['path_exorl_learn_24_true_RMSPROP']}/datasets/walker/rnd/replay.pt"  # from the command
        # f"{parameters['config']['path_exorl_learn_44_aligned']}/datasets/walker/rnd/replay.pt"  # from the command 


        # f"{parameters['config']['path_exorl_learn_15_aligned_dlc']}/datasets/walker/rnd/replay.pt"  # from the command 
        # f"{parameters['config']['path_exorl_learn_24_aligned_dlc']}/datasets/walker/rnd/replay.pt"  # from the command 
        # f"{parameters['config']['path_exorl_learn_34_aligned_dlc']}/datasets/walker/rnd/replay.pt"  # from the command 
        # f"{parameters['config']['path_exorl_learn_44_aligned_dlc']}/datasets/walker/rnd/replay.pt"  # from the command 


        # f"{parameters['config']['path_exorl_learn_fri_4_aligned_dlc']}/datasets/walker/rnd/replay.pt"  # from the command 
        # f"{parameters['config']['path_exorl_learn_fri_5_aligned_dlc']}/datasets/walker/rnd/replay.pt"  # from the command 
        # f"{parameters['config']['path_exorl_learn_fri_6_aligned_dlc']}/datasets/walker/rnd/replay.pt"  # from the command 
        # f"{parameters['config']['path_exorl_learn_fri_7_aligned_dlc']}/datasets/walker/rnd/replay.pt"  # from the command 
        # f"{parameters['config']['path_exorl_learn_fri_8_aligned_dlc']}/datasets/walker/rnd/replay.pt"  # from the command 
        # f"{parameters['config']['path_exorl_learn_fri_18_aligned_dlc']}/datasets/walker/rnd/replay.pt"  # from the command 

        
    )


    # dataset: align s, a ==> s_t+a
    # load_replay_buffer: tp.Optional[str] = (
    #     # f"{parameters['config']['path_exorl_learn']}/datasets/walker/rnd/replay.pt"  # from the command 
    #     f"{parameters['config']['path_exorl_learn_earth_aligned_s_a']}/datasets/walker/rnd/replay.pt"  # from the command
    # )
    # load_replay_buffer_env_target: tp.Optional[str] = (
    #     # f"{parameters['config']['path_exorl_learn_24_true_RMSPROP']}/datasets/walker/rnd/replay.pt"  # from the command
    #     f"{parameters['config']['path_exorl_learn_24_aligned_s_a']}/datasets/walker/rnd/replay.pt"  # from the command
    # )

    # new config:
    # path_exorl_learn_24_aligned_s_a = "/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/url_verify_solved/url_benchmark/exp_local/2025.04.24/align_24_s_a"
    # path_exorl_learn_earth_aligned_s_a = "/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/url_verify_solved/url_benchmark/exp_local/2025.04.24/alin_earth_s_a"

    # path_exorl_learn_44
    # path_exorl_learn_44_true
    # path_exorl_learn_24_true
    # path_exorl_learn_mixed_true
    
    # /scratch/longchao/project/Sim2Real/url_benchmark/exp_local/2024.12.02/saved_buffer/datasets/walker/rnd/replay.pt
    
    expl_agent: str = "rnd"
    replay_buffer_dir: str = omgcf.SI("../../../../datasets")

    feature_learner: str = 'hilp'  # from the command
    hilp_expectile: float = 0.5  # from the command
    hilp_discount: float = 0.90  # from the command
    q_loss: bool = False  # from the command
    save_path: tp.Optional[str] = None


ConfigStore.instance().store(name="workspace_config", node=Config)


class BaseReward:
    def __init__(self, seed: tp.Optional[int] = None) -> None:
        self._env: dmc.EnvWrapper  # to be instantiated in subclasses
        self._rng = np.random.RandomState(seed)

    def get_goal(self, goal_space: str) -> np.ndarray:
        raise NotImplementedError

    def from_physics(self, physics: np.ndarray) -> float:
        "careful this is not threadsafe"
        with self._env.physics.reset_context():
            self._env.physics.set_state(physics)
        return self.from_env(self._env)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        raise NotImplementedError


class DmcReward(BaseReward):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        env_name, task_name = name.split("_", maxsplit=1)
        from dm_control import suite  # import
        from url_benchmark import custom_dmc_tasks as cdmc
        if 'jaco' not in env_name:
            make = suite.load if (env_name, task_name) in suite.ALL_TASKS else cdmc.make
            self._env = make(env_name, task_name)
        else:
            self._env = cdmc.make_jaco(task_name, obs_type='states', seed=0)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        return float(self._env.task.get_reward(env.physics))


def make_agent(
        obs_type: str, image_wh, obs_spec, action_spec, num_expl_steps: int, cfg: omgcf.DictConfig
) -> tp.Union[agents.FBDDPGAgent, agents.DDPGAgent]:
    cfg.obs_type = obs_type
    cfg.image_wh = image_wh
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = (action_spec.num_values, ) if isinstance(action_spec, specs.DiscreteArray) \
        else action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)

# # original
# def _init_eval_meta(workspace, custom_reward: BaseReward = None):
#     num_steps = workspace.agent.cfg.num_inference_steps
#     obs_list, reward_list, next_obs_list = [], [], []
#     batch_size = 0
#     while batch_size < num_steps:
#         batch = workspace.replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
#         batch = batch.to(workspace.cfg.device)
#         if isinstance(workspace.agent, agents.FBDDPGAgent) or (isinstance(workspace.agent, agents.SFAgent) and workspace.agent.cfg.feature_type == 'state'):
#             obs_list.append(batch.next_obs)
#             next_obs_list.append(batch.next_obs)
#         else:
#             obs_list.append(batch.obs)
#             next_obs_list.append(batch.next_obs)
#         reward_list.append(batch.reward)
#         batch_size += batch.next_obs.size(0)
#     obs, reward, next_obs = torch.cat(obs_list, 0), torch.cat(reward_list, 0), torch.cat(next_obs_list, 0)
#     obs_t, reward_t, next_obs_t = obs[:num_steps], reward[:num_steps], next_obs[:num_steps]
#     return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t, next_obs_t, None, None, None, False)



def get_obs_difference(obs: torch.Tensor, obs_targ: torch.Tensor, save_path: str, num_samples: 10) -> None:
    """
    Sample 10 observations from obs (shape [10000, 24]) with seed=44,
    for each observation, find the closest observation from obs_targ (based on overall L2 distance),
    then compute and plot the differences of each of the 24 dimensions.
    
    The resulting figure contains 10 subplots (one for each sampled pair),
    where the x-axis represents the 24 dimensions and the y-axis represents the difference in that dimension.
    
    The final figure is saved to the given save_path.
    
    Args:
        obs (torch.Tensor): Tensor of shape [10000, 24].
        obs_targ (torch.Tensor): Tensor of shape [10000, 24].
        save_path (str): Path to save the output figure.
    """
    # Set the random seed for reproducibility
    np.random.seed(44)
   
    # Number of samples to take
    
    # Convert obs to numpy for sampling, if necessary
    # (假设 obs 为 torch.Tensor，直接使用 torch.randperm 也可以)
    total_samples = obs.shape[0]
    # Get random indices from obs
    indices = torch.randperm(total_samples)[:num_samples]
    
    # Create subplots: 10 rows, 1 column
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 30))
    
    # Ensure axes is an array
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        sample_obs = obs[idx]              # shape: [24]
        # Compute L2 distance from sample_obs to all obs_targ samples
        # Note: expand sample_obs to match obs_targ shape for broadcasting
        distances = torch.norm(obs_targ - sample_obs.unsqueeze(0), dim=1)  # shape: [N]
        # Get the index of the closest obs_targ
        best_match_idx = torch.argmin(distances)
        matched_obs = obs_targ[best_match_idx]  # shape: [24]
        # Compute dimension-wise difference (这里取原始差值，也可以取绝对值)
        diff = sample_obs - matched_obs  # shape: [24]
        diff_np = diff.cpu().numpy()
        dims = np.arange(diff_np.shape[0])  # x-axis: 0~23
        
        # Plot bar chart in each subplot
        ax = axes[i]
        ax.bar(dims, diff_np, color='blue', alpha=0.7)
        ax.set_xlabel("Dimension Index")
        ax.set_ylabel("Difference Value")
        ax.set_title(f"Sample {i} (obs index: {idx.item()}), Matched obs_targ index: {best_match_idx.item()}")
        ax.grid(True)
    
    plt.tight_layout()
    # Save the figure to the specified path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# original
# def _init_eval_meta(workspace, custom_reward: BaseReward = None):
#     num_steps = workspace.agent.cfg.num_inference_steps
#     obs_list, reward_list, next_obs_list = [], [], []
#     batch_size = 0
#     while batch_size < num_steps:
#         batch = workspace.replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
#         batch = batch.to(workspace.cfg.device)
#         if isinstance(workspace.agent, agents.FBDDPGAgent) or (isinstance(workspace.agent, agents.SFAgent) and workspace.agent.cfg.feature_type == 'state'):
#             obs_list.append(batch.next_obs)
#             next_obs_list.append(batch.next_obs)
#         else:
#             obs_list.append(batch.obs)
#             next_obs_list.append(batch.next_obs)
#         reward_list.append(batch.reward)
#         batch_size += batch.next_obs.size(0)
#     obs, reward, next_obs = torch.cat(obs_list, 0), torch.cat(reward_list, 0), torch.cat(next_obs_list, 0)
#     obs_t, reward_t, next_obs_t = obs[:num_steps], reward[:num_steps], next_obs[:num_steps]
#     return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t, next_obs_t, None, None, None, False)

def corrupt_target_subset(
    obs: torch.Tensor,
    rew: torch.Tensor,
    nxt: torch.Tensor,
    mode: str = "drop",
    pct: float = 0.3,
    seed: int = 123,
    noise_std: float = 0.1,
    noise_dim_pct: float = 1.0,   # <— NEW
):
    """
    Corrupt a percentage of rows in (obs, rew, nxt) for *target* data.
    - mode="drop": remove pct of rows (length becomes smaller).
    - mode="mask": set pct of rows to zeros (length unchanged).
    - mode="noise": add Gaussian noise to pct of rows (length unchanged).
    Returns (obs2, rew2, nxt2, kept_idx, affected_idx).
    """
    assert 0.0 <= pct <= 1.0, "pct must be in [0,1]"
    N = obs.shape[0]

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    dev = obs.device
    g = torch.Generator(device='cuda' if dev.type == 'cuda' else 'cpu')
    g.manual_seed(seed)

    # choose which rows to affect
    idx = torch.randperm(N, generator=g, device=dev)
    k_keep = int(round((1 - pct) * N))
    keep_idx    = idx[:k_keep]
    affect_idx  = idx[k_keep:]

    if mode == "drop":
        # shrink the dataset by keeping only (1 - pct)
        return obs[keep_idx], rew[keep_idx], nxt[keep_idx], keep_idx, affect_idx

    elif mode == "mask":
        # zero out the selected rows (shape preserved)
        obs2, rew2, nxt2 = obs.clone(), rew.clone(), nxt.clone()
        obs2[affect_idx] = 0
        nxt2[affect_idx] = 0
        rew2[affect_idx] = 0
        return obs2, rew2, nxt2, keep_idx, affect_idx

    # elif mode == "noise":
    #     # add N(0, noise_std * feature_std) to selected rows (shape preserved)
    #     obs2, rew2, nxt2 = obs.clone(), rew.clone(), nxt.clone()

    #     eps = 1e-8
    #     # per-dimension std (avoid zero-std)
    #     obs_std = obs.std(dim=0, unbiased=False) + eps
    #     nxt_std = nxt.std(dim=0, unbiased=False) + eps
    #     rew_std = rew.std() + eps  # scalar

    #     # build noise using the same generator for reproducibility
    #     obs_noise = torch.randn((affect_idx.numel(), obs.shape[1]), device=obs.device, generator=g) * (noise_std * obs_std)
    #     nxt_noise = torch.randn((affect_idx.numel(), nxt.shape[1]), device=nxt.device, generator=g) * (noise_std * nxt_std)

    #     # rewards: handle shape (N,) or (N,1)
    #     if rew.dim() == 1:
    #         rew_noise = torch.randn((affect_idx.numel(),), device=rew.device, generator=g) * (noise_std * rew_std)
    #     else:  # (N,1) or (N,k)
    #         rew_noise = torch.randn((affect_idx.numel(), rew.shape[1]), device=rew.device, generator=g) * (noise_std * rew_std)

    #     obs2[affect_idx] += obs_noise
    #     nxt2[affect_idx] += nxt_noise
    #     rew2[affect_idx] += rew_noise
    #     return obs2, rew2, nxt2, keep_idx, affect_idx
    elif mode == "noise":
        obs2, rew2, nxt2 = obs.clone(), rew.clone(), nxt.clone()
        eps = 1e-8
        obs_std = obs.std(dim=0, unbiased=False) + eps
        nxt_std = nxt.std(dim=0, unbiased=False) + eps
        rew_std = rew.std() + eps

        # 只对一部分维度加噪：更“集中”的破坏
        D_obs = obs.shape[1]; D_nxt = nxt.shape[1]
        gdev = obs.device
        g = torch.Generator(device='cuda' if gdev.type == 'cuda' else 'cpu'); g.manual_seed(seed)

        d_obs = max(1, int(round(NOISE_DIM_PCT * D_obs)))
        d_nxt = max(1, int(round(NOISE_DIM_PCT * D_nxt)))
        dim_idx_obs = torch.randperm(D_obs, generator=g, device=gdev)[:d_obs]
        dim_idx_nxt = torch.randperm(D_nxt, generator=g, device=gdev)[:d_nxt]

        # 高斯噪声（更重就把 NOISE_STD 拉大）
        obs_noise = torch.randn((affect_idx.numel(), d_obs), device=gdev, generator=g) * (NOISE_STD * obs_std[dim_idx_obs])
        nxt_noise = torch.randn((affect_idx.numel(), d_nxt), device=gdev, generator=g) * (NOISE_STD * nxt_std[dim_idx_nxt])
        obs2[affect_idx][:, dim_idx_obs] += obs_noise
        nxt2[affect_idx][:, dim_idx_nxt] += nxt_noise

        # 奖励噪声（可选：也可以把 NOISE_DIM_PCT 用于 reward 的一部分通道）
        if rew.dim() == 1:
            rew_noise = torch.randn((affect_idx.numel(),), device=gdev, generator=g) * (NOISE_STD * rew_std)
        else:
            rew_noise = torch.randn((affect_idx.numel(), rew.shape[1]), device=gdev, generator=g) * (NOISE_STD * rew_std)
        rew2[affect_idx] += rew_noise
        return obs2, rew2, nxt2, keep_idx, affect_idx

    else:
        raise ValueError(f"Unknown mode: {mode}")
# my exploration

def _init_eval_meta(workspace, custom_reward: BaseReward = None):
    show_obs_alignment = True
    print("------workspace.agent.cfg------")
    print(workspace.agent.cfg)
    num_steps = workspace.agent.cfg.num_inference_steps
    
    # 1. Get data for original env:
    obs_list, reward_list, next_obs_list = [], [], []
    batch_size = 0
    while batch_size < num_steps:
        batch = workspace.replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
        batch = batch.to(workspace.cfg.device)
        if isinstance(workspace.agent, agents.FBDDPGAgent) or (isinstance(workspace.agent, agents.SFAgent) and workspace.agent.cfg.feature_type == 'state'):
            obs_list.append(batch.next_obs)
            next_obs_list.append(batch.next_obs)
        else:
            obs_list.append(batch.obs)
            next_obs_list.append(batch.next_obs)
        reward_list.append(batch.reward)
        batch_size += batch.next_obs.size(0)
    obs, reward, next_obs = torch.cat(obs_list, 0), torch.cat(reward_list, 0), torch.cat(next_obs_list, 0)
    obs_t, reward_t, next_obs_t = obs[:num_steps], reward[:num_steps], next_obs[:num_steps]
    # len(obs_t) = 10000, len(obs_t[0]) = 24: here we have obs_t.shape: ([10000, 24])
    # reward_t shape: ([10000, 1])
    # next_obs_t: ([10000, 24])
    
    # 2. Get for the target environment
    obs_list_env_targ, reward_list_env_targ, next_obs_list_env_targ = [], [], []
    batch_size_env_targ = 0

    num_steps_env_targ = num_steps # 10000

    # 这一步，while 有可能会产生一些内部加载过程的缓冲变化，导致最终结果可能稍微好于原本的结果，（+50～60）左右

    while batch_size_env_targ < num_steps_env_targ:
        batch_env_targ = workspace.replay_loader_env_targ.sample(
            workspace.cfg.batch_size, custom_reward=custom_reward
        )
        batch_env_targ = batch_env_targ.to(workspace.cfg.device)
        if isinstance(workspace.agent, agents.FBDDPGAgent) or (
            isinstance(workspace.agent, agents.SFAgent) and workspace.agent.cfg.feature_type == "state"
        ):
            obs_list_env_targ.append(batch_env_targ.next_obs)
            next_obs_list_env_targ.append(batch_env_targ.next_obs)
        else:
            obs_list_env_targ.append(batch_env_targ.obs)
            next_obs_list_env_targ.append(batch_env_targ.next_obs)
        reward_list_env_targ.append(batch_env_targ.reward)
        batch_size_env_targ += batch_env_targ.next_obs.size(0)
    
    obs_env_targ = torch.cat(obs_list_env_targ, 0)
    reward_env_targ = torch.cat(reward_list_env_targ, 0)
    next_obs_env_targ = torch.cat(next_obs_list_env_targ, 0)

    obs_t_env_targ = obs_env_targ[:num_steps_env_targ]
    reward_t_env_targ = reward_env_targ[:num_steps_env_targ]
    next_obs_t_env_targ = next_obs_env_targ[:num_steps_env_targ]

    # save_diff  = "/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/0official/sep24/quality/diff/"

    # get the most close obs (sample 10)
    # if show_obs_alignment:
    #     get_obs_difference(obs, obs_env_targ, save_path=save_diff + "obs_diff.png", num_samples=10)
    #     get_obs_difference(obs_t, obs_t_env_targ, save_path=save_diff + "obs_diff_t.png", num_samples=10)

    # CORRUPT_MODE = "drop"     # one of: "drop", "mask", "noise"
    # CORRUPT_PCT  = 0      # e.g., 0.30 => affect 30% of target samples
    # CORRUPT_SEED = 123        # reproducible random subset
    # # NOISE_STD    = 0.10       # used only when CORRUPT_MODE == "noise" (relative scale)
    # NOISE_STD      = 0.8         # ↑ 提高就更“重”（例如 0.5~1.0 甚至 2.0）
    # NOISE_DIM_PCT  = 1.0         # 只对选中样本的前 NOISE_DIM_PCT 维加噪（1.0=所有维度）

    # if CORRUPT_PCT > 0:
    #     (obs_t_env_targ,
    #     reward_t_env_targ,
    #     next_obs_t_env_targ,
    #     kept_idx, affected_idx) = corrupt_target_subset(
    #         obs_t_env_targ, reward_t_env_targ, next_obs_t_env_targ,
    #         mode=CORRUPT_MODE, pct=CORRUPT_PCT, seed=CORRUPT_SEED, noise_std=NOISE_STD, noise_dim_pct=NOISE_DIM_PCT
    #     )
    #     print(f"[target corruption] mode={CORRUPT_MODE}, pct={CORRUPT_PCT:.2f}, "
    #         f"affected={affected_idx.numel()} / {kept_idx.numel()+affected_idx.numel()}")

    # this has been updated: dlc (in sf.py!!!!!)
    # infer_meta_from_obs_and_rewards_sim2real
    # return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t, next_obs_t)
    # # Mock example usage (assuming var1 and var2 are numpy arrays of shape (1000, 24)):
    # visualize the distances:
    
    # result = visualize_distribution_difference(obs_t.cpu(), obs_t_env_targ.cpu())
    # new_reward_t_env_targ = relabel_rewards_with_pca(obs_t, obs_t_env_targ, reward_t)

    # return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t, next_obs_t, obs_t_env_targ, reward_t_env_targ, next_obs_t_env_targ, False)
    # return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t, next_obs_t, obs_t_env_targ, reward_t_env_targ, next_obs_t_env_targ)
    return workspace.agent.infer_meta_from_obs_and_rewards_sim2real(obs_t, reward_t, next_obs_t, obs_t_env_targ, reward_t_env_targ, next_obs_t_env_targ, False)

def relabel_rewards(obs_t, obs_t_env_targ, reward_t):
    """
    Relabel the rewards for obs_t_env_targ based on the closest match in obs_t.
    
    Args:
        obs_t (torch.Tensor): The source observation tensor of shape (N, D).
        obs_t_env_targ (torch.Tensor): The target observation tensor of shape (M, D).
        reward_t (torch.Tensor): The rewards for obs_t of shape (N,).
        
    Returns:
        torch.Tensor: The relabeled rewards for obs_t_env_targ of shape (M,).
    """
    # Ensure the inputs are of the correct shape
    assert obs_t.size(1) == obs_t_env_targ.size(1), "Dimension mismatch between obs_t and obs_t_env_targ."
    assert obs_t.size(0) == reward_t.size(0), "Mismatch between obs_t and reward_t."

    # Compute pairwise distances between obs_t_env_targ and obs_t
    distances = torch.cdist(obs_t_env_targ, obs_t, p=2)  # Shape: (M, N)

    # Find the index of the closest match in obs_t for each obs_t_env_targ
    closest_indices = torch.argmin(distances, dim=1)  # Shape: (M,)

    # Use the closest indices to map rewards from obs_t to obs_t_env_targ
    new_reward_t_env_targ = reward_t[closest_indices] * 2 # Shape: (M,)

    return new_reward_t_env_targ

from sklearn.decomposition import PCA
import torch

def relabel_rewards_with_pca(obs_t, obs_t_env_targ, reward_t, n_components=8):
    """
    Relabel the rewards for obs_t_env_targ based on the closest match in obs_t after PCA dimensionality reduction.

    Args:
        obs_t (torch.Tensor): The source observation tensor of shape (N, D).
        obs_t_env_targ (torch.Tensor): The target observation tensor of shape (M, D).
        reward_t (torch.Tensor): The rewards for obs_t of shape (N,).
        n_components (int): Number of components for PCA dimensionality reduction.

    Returns:
        torch.Tensor: The relabeled rewards for obs_t_env_targ of shape (M,).
    """
    # Ensure the inputs are of the correct shape
    assert obs_t.size(1) == obs_t_env_targ.size(1), "Dimension mismatch between obs_t and obs_t_env_targ."
    assert obs_t.size(0) == reward_t.size(0), "Mismatch between obs_t and reward_t."

    # Convert tensors to numpy for PCA
    obs_t_np = obs_t.cpu().numpy()
    obs_t_env_targ_np = obs_t_env_targ.cpu().numpy()

    # Apply PCA to reduce dimensions
    pca = PCA(n_components=n_components)
    obs_t_reduced = pca.fit_transform(obs_t_np)  # Shape: (N, n_components)
    obs_t_env_targ_reduced = pca.transform(obs_t_env_targ_np)  # Shape: (M, n_components)

    # Convert reduced observations back to tensors
    obs_t_reduced = torch.tensor(obs_t_reduced, device=obs_t.device)
    obs_t_env_targ_reduced = torch.tensor(obs_t_env_targ_reduced, device=obs_t_env_targ.device)

    # Compute pairwise distances between reduced obs_t_env_targ and obs_t
    distances = torch.cdist(obs_t_env_targ_reduced, obs_t_reduced, p=2)  # Shape: (M, N)

    # Find the index of the closest match in obs_t for each obs_t_env_targ
    closest_indices = torch.argmin(distances, dim=1)  # Shape: (M,)

    # Use the closest indices to map rewards from obs_t to obs_t_env_targ
    new_reward_t_env_targ = reward_t[closest_indices]  # Shape: (M,)

    return new_reward_t_env_targ


def visualize_distribution_difference(var1, var2, save_path="distribution_difference.png"):
    """
    Visualize and compare the distribution differences between two high-dimensional tensors.
    
    Args:
        var1 (np.ndarray): First input tensor of shape (N, D).
        var2 (np.ndarray): Second input tensor of shape (N, D).
        save_path (str): Path to save the visualization image.
    
    Returns:
        dict: A dictionary containing statistical summaries of differences.
    """
    # Calculate pairwise Wasserstein distances (Earth Mover's Distance)
    wasserstein_dists = [
        wasserstein_distance(var1[:, i], var2[:, i]) for i in range(var1.shape[1])
    ]
    
    # Calculate mean and standard deviation for differences
    mean_diff = np.mean(wasserstein_dists)
    std_diff = np.std(wasserstein_dists)

    # Plotting the Wasserstein distances
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(wasserstein_dists)), wasserstein_dists, color='blue', alpha=0.7)
    plt.axhline(mean_diff, color='red', linestyle='--', label=f"Mean: {mean_diff:.3f}")
    plt.fill_between(
        range(len(wasserstein_dists)),
        mean_diff - std_diff,
        mean_diff + std_diff,
        color='orange',
        alpha=0.2,
        label=f"Std Dev: {std_diff:.3f}"
    )
    plt.title("Wasserstein Distance Between Corresponding Dimensions")
    plt.xlabel("Feature Index")
    plt.ylabel("Wasserstein Distance")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    # Return summary statistics
    return {
        "mean_wasserstein_distance": mean_diff,
        "std_wasserstein_distance": std_diff,
        "individual_distances": wasserstein_dists
    }

# # Mock example usage (assuming var1 and var2 are numpy arrays of shape (1000, 24)):
# var1 = np.random.randn(1000, 24)
# var2 = np.random.randn(1000, 24) + 0.5  # Slightly shifted
# result = visualize_distribution_difference(obs_t.cpu(), obs_t_env_targ.cpu())


# ==== GAT + EDL UQ 直接替换版（含 Workspace）====
import os, json, math, tempfile
from pathlib import Path
import typing as tp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------- 基础 MLP -------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, layers=2):
        super().__init__()
        layers_list, d = [], in_dim
        for _ in range(layers):
            layers_list += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        layers_list += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers_list)
    def forward(self, x): return self.net(x)

# ------------------- Forward: (s,a)->s' -------------------
class ForwardModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, layers=2, predict_delta=False):
        super().__init__()
        self.predict_delta = predict_delta
        self.core = MLP(obs_dim + act_dim, obs_dim, hidden, layers)
    def forward(self, s, a):
        out = self.core(torch.cat([s, a], dim=-1))
        return s + out if self.predict_delta else out

# ------------------- EDL 组件（NIG 回归） -------------------
def softplus(x): return torch.nn.functional.softplus(x)

class EDLInverseHead(nn.Module):
    """
    对每个动作维度输出 (mu, v, alpha, beta) 的 NIG 参数:
      mu: 预测均值
      v, alpha, beta > 0 由 softplus 限制
    """
    def __init__(self, in_dim, act_dim, hidden=256, layers=2):
        super().__init__()
        self.body = MLP(in_dim, hidden, hidden=hidden, layers=max(layers-1,1))
        self.out_mu     = nn.Linear(hidden, act_dim)
        self.out_v      = nn.Linear(hidden, act_dim)
        self.out_alpha  = nn.Linear(hidden, act_dim)
        self.out_beta   = nn.Linear(hidden, act_dim)

    def forward(self, x):
        h = self.body(x)
        mu    = self.out_mu(h)
        v     = softplus(self.out_v(h)) + 1e-4
        alpha = softplus(self.out_alpha(h)) + 1.0  # alpha>1 保证方差有限
        beta  = softplus(self.out_beta(h)) + 1e-4
        return mu, v, alpha, beta

def nig_nll(y, mu, v, alpha, beta):
    """
    Evidential Regression NLL（Amini+ 2020）
    y, mu, v, alpha, beta: [B, D]
    """
    two_beta_v = 2.0 * beta * (1.0 + v)
    nll = 0.5*torch.log(math.pi/v) \
          - alpha*torch.log(two_beta_v) \
          + (alpha + 0.5)*torch.log(v*(y - mu)**2 + two_beta_v) \
          + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return nll

def nig_regularizer(y, mu, v, alpha, beta, lam=1e-3):
    """
    证据正则项：鼓励在大误差时减小证据（降低过度自信）
    """
    err = (y - mu).abs()
    reg = err * (2.0*v + alpha)
    return lam * reg

def edl_regression_loss(y, mu, v, alpha, beta, lam=1e-3):
    nll = nig_nll(y, mu, v, alpha, beta)
    reg = nig_regularizer(y, mu, v, alpha, beta, lam)
    return (nll + reg).mean()

def nig_predictive_variance(v, alpha, beta):
    """
    NIG 预测方差：sigma^2 = beta / (v*(alpha-1))
    """
    return beta / (v * (alpha - 1.0) + 1e-8)

# ------------------- Inverse: (s, s'_des) -> a  + EDL UQ -------------------
class InverseModelEDL(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, layers=2):
        super().__init__()
        self.head = EDLInverseHead(in_dim=obs_dim*2, act_dim=act_dim, hidden=hidden, layers=layers)

    def forward(self, s, s_next_desired):
        x = torch.cat([s, s_next_desired], dim=-1)
        mu, v, alpha, beta = self.head(x)
        return mu, v, alpha, beta  # mu 是 a_hat，后面用 var 计算不确定性

# ------------------- GAT + EDL 容器 -------------------
class GATModule:
    """
    训练：
      - forward 用 target/real replay: (s,a)->s'
      - inverse-EDL 用 sim replay:    (s, s'_des=f_real(s,a))->a，损失用 EDL 回归
    推理：
      - 产出 grounded 动作 a_g 以及不确定性 u（NIG 预测方差的均值）
      - 在 Workspace.eval 中与上一步 u 比较，决定是否采纳 a_g（UGAT 不确定性门控思想 ）
    """
    def __init__(self, obs_dim, act_dim, device, cfg):
        self.device = device
        self.obs_dim, self.act_dim = obs_dim, act_dim
        hidden = int(getattr(cfg, "gat_hidden", 256))
        layers = int(getattr(cfg, "gat_layers", 2))
        predict_delta = bool(getattr(cfg, "gat_predict_delta", False))
        self.alpha_mix = float(getattr(cfg, "gat_alpha", 1.0))  # 与上一版一致：动作线性混合因子

        self.f_real   = ForwardModel(obs_dim, act_dim, hidden=hidden, layers=layers, predict_delta=predict_delta).to(device)
        self.f_inv_edl= InverseModelEDL(obs_dim, act_dim, hidden=hidden, layers=layers).to(device)

        # 路径
        default_dir = CONTEXT.project_dir / "url_benchmark" / "agent" / "gat"
        self.ckpt_dir = Path(getattr(cfg, "gat_weights_dir", str(default_dir)))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.fwd_path = self.ckpt_dir / "forward.pt"
        self.inv_path = self.ckpt_dir / "inverse_edl.pt"

        # 训练超参
        self.epochs = int(getattr(cfg, "gat_epochs", 200))
        self.bs = int(getattr(cfg, "gat_batch_size", 1024))
        self.lr_fwd = float(getattr(cfg, "gat_lr_forward", 1e-3))
        self.lr_inv = float(getattr(cfg, "gat_lr_inverse", 1e-3))
        self.n_forward = int(getattr(cfg, "gat_forward_samples", 50000))
        self.n_inverse = int(getattr(cfg, "gat_inverse_samples", 50000))
        self.edl_reg_lam = float(getattr(cfg, "gat_edl_reg_lambda", 1e-3))

    def _gather_from_buffer(self, replay, max_n, device):
        obs_list, act_list, next_list = [], [], []
        total, bs = 0, min(self.bs, 4096)
        while total < max_n:
            batch = replay.sample(bs, custom_reward=None).to(device)
            if not hasattr(batch, "action"):
                raise RuntimeError("ReplayBuffer batch 缺少 action，无法训练 GAT/UGAT。")
            obs_list.append(batch.obs if hasattr(batch, "obs") else batch.next_obs*0)
            act_list.append(batch.action)
            next_list.append(batch.next_obs)
            total += batch.next_obs.size(0)
        S = torch.cat(obs_list, 0)[:max_n]
        A = torch.cat(act_list, 0)[:max_n]
        Sp= torch.cat(next_list, 0)[:max_n]
        return S, A, Sp

    def maybe_load(self):
        ok = False
        if self.fwd_path.exists():
            self.f_real.load_state_dict(torch.load(self.fwd_path, map_location=self.device))
            ok = True
        if self.inv_path.exists():
            self.f_inv_edl.load_state_dict(torch.load(self.inv_path, map_location=self.device))
            ok = ok and True
        return ok

    def train_models(self, replay_sim, replay_real):
        # 数据
        S_r, A_r, Sp_r = self._gather_from_buffer(replay_real, self.n_forward, self.device)
        S_s, A_s, _    = self._gather_from_buffer(replay_sim,  self.n_inverse, self.device)

        # ---------- 训练 forward ----------
        opt_f = optim.Adam(self.f_real.parameters(), lr=self.lr_fwd)
        mse = nn.MSELoss()
        n = S_r.size(0); idx = torch.arange(n, device=self.device)
        for ep in range(self.epochs):
            perm = idx[torch.randperm(n)]
            for i in range(0, n, self.bs):
                j = min(i+self.bs, n)
                s, a, sp = S_r[perm[i:j]], A_r[perm[i:j]], Sp_r[perm[i:j]]
                pred = self.f_real(s, a)
                loss = mse(pred, sp)
                opt_f.zero_grad(); loss.backward(); opt_f.step()

        # ---------- 训练 inverse（EDL 回归） ----------
        with torch.no_grad():
            Sp_des = self.f_real(S_s, A_s)               # s'_des = f_real(s, a)
        opt_i = optim.Adam(self.f_inv_edl.parameters(), lr=self.lr_inv)
        n2 = S_s.size(0); idx2 = torch.arange(n2, device=self.device)
        for ep in range(self.epochs):
            perm = idx2[torch.randperm(n2)]
            for i in range(0, n2, self.bs):
                j = min(i+self.bs, n2)
                s, sp_des, a_tgt = S_s[perm[i:j]], Sp_des[perm[i:j]], A_s[perm[i:j]]
                mu, v, alpha, beta = self.f_inv_edl(s, sp_des)
                loss = edl_regression_loss(a_tgt, mu, v, alpha, beta, lam=self.edl_reg_lam)
                opt_i.zero_grad(); loss.backward(); opt_i.step()

        torch.save(self.f_real.state_dict(), self.fwd_path)
        torch.save(self.f_inv_edl.state_dict(), self.inv_path)

    @torch.no_grad()
    def transform_action_with_uq(self, obs_np, action_np):
        """
        返回: a_grounded, u_val
        u_val = 预测方差的均值（各动作维度平均）
        """
        s = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.as_tensor(action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        s_real_next = self.f_real(s, a)
        mu, v, alpha, beta = self.f_inv_edl(s, s_real_next)   # mu 即 a_hat
        var = nig_predictive_variance(v, alpha, beta)         # [1, act_dim]
        u = var.mean(dim=-1)                                  # 标量不确定性
        a_hat = mu
        a_g = self.alpha_mix * a_hat + (1.0 - self.alpha_mix) * a
        return a_g.squeeze(0).cpu().numpy(), float(u.item())

# ------------------- Workspace：集成 UGAT-EDL 的动作门控 -------------------
class Workspace:
    def __init__(self, cfg: Config) -> None:
        print("Config save_path:", cfg.get("save_path", "save_path not found"))
        print(cfg.save_path)
        self.work_dir = Path(cfg.save_path) if cfg.save_path else Path(f"{parameters['config']['path']}/hilp_zsrl/exp_local/EXP/saved_training")
        self.latest_dir=Path(f"{parameters['config']['path']}/hilp_zsrl/exp_local/EXP/saved_training")
        print(f"Workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        if not torch.cuda.is_available():
            if cfg.device != "cpu":
                logger.warning(f"Falling back to cpu as {cfg.device} is not available")
                cfg.device = "cpu"; cfg.agent.device = "cpu"
        self.device = torch.device(cfg.device)

        task = cfg.task
        self.domain = task.split('_', maxsplit=1)[0]

        self.train_env = self._make_env()
        self.eval_env  = self._make_env()
        self.train_env.reset()

        self.agent = make_agent(cfg.obs_type, cfg.image_wh,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        if cfg.use_wandb:
            exp_name = ''
            exp_name += f'sd{cfg.seed:03d}_'
            if 'SLURM_JOB_ID' in os.environ: exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
            if 'SLURM_PROCID' in os.environ: exp_name += f'{os.environ["SLURM_PROCID"]}.'
            exp_name += '_'.join([cfg.run_group, cfg.agent.name, self.domain,])
            wandb_output_dir = tempfile.mkdtemp()
            wandb.init(project='hilp_zsrl', group=cfg.run_group, name=exp_name,
                       config=omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                       dir=wandb_output_dir)

        self.replay_loader = ReplayBuffer(max_episodes=cfg.replay_buffer_episodes, discount=cfg.discount, future=cfg.future)
        self.replay_loader_env_targ = ReplayBuffer(max_episodes=cfg.replay_buffer_env_target_episodes, discount=cfg.discount, future=cfg.future)

        cam_id = 0 if 'quadruped' not in self.domain else 2
        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, task=self.cfg.task,
                                            camera_id=cam_id, use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self.global_step = 0
        self.global_episode = 0
        self.eval_rewards_history: tp.List[float] = []
        self._checkpoint_filepath = self.latest_dir / "models" / "latest.pt"
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        elif cfg.load_model is not None:
            self.load_checkpoint(cfg.load_model, exclude=["replay_loader"])

        datasets_dir = self.latest_dir / cfg.replay_buffer_dir
        replay_dir = datasets_dir.resolve() / self.domain / cfg.expl_agent / "buffer"
        print(f"replay dir: {replay_dir}")

        print("loading Replay from %s", self.cfg.load_replay_buffer)
        self.load_checkpoint(self.cfg.load_replay_buffer, only=["replay_loader"], num_episodes=cfg.replay_buffer_episodes, use_pixels=(cfg.obs_type == 'pixels'))

        print(f"loading target_env:{self.cfg.load_replay_buffer_env_target}")
        self.load_checkpoint_demo_env_targ(self.cfg.load_replay_buffer_env_target, only=["replay_loader"], num_episodes=cfg.replay_buffer_env_target_episodes, use_pixels=(cfg.obs_type == 'pixels'))

        self.replay_loader._future = cfg.future
        self.replay_loader._discount = cfg.discount
        self.replay_loader._p_currgoal = cfg.p_currgoal
        self.replay_loader._p_randomgoal = cfg.p_randomgoal
        self.replay_loader._frame_stack = cfg.frame_stack if cfg.obs_type == 'pixels' else None
        self.replay_loader._max_episodes = len(self.replay_loader._storage["discount"])

        # ===== 初始化 UGAT (含 EDL UQ) =====
        self.use_gat = bool(getattr(self.cfg, "use_gat", True))
        self.last_u = float('inf')   # 比较式门控：第一步默认更可能接受 grounded（也可改为 None）
        if self.use_gat:
            b = self.replay_loader.sample(max(8, getattr(self.cfg, "batch_size", 512))).to(self.cfg.device)
            if not hasattr(b, "action"):
                print("[UGAT-EDL] ReplayBuffer batch 不含 action，禁用 UGAT。")
                self.use_gat = False
            else:
                obs_dim = (b.obs if hasattr(b, "obs") else b.next_obs).shape[-1]
                act_dim = b.action.shape[-1]
                self.gat = GATModule(obs_dim, act_dim, self.device, self.cfg)
                loaded = self.gat.maybe_load()
                if not loaded:
                    print("[UGAT-EDL] 训练 forward/inverse-EDL ...")
                    self.gat.train_models(self.replay_loader, self.replay_loader_env_targ)
                    print("[UGAT-EDL] 训练完成，已保存。")
                else:
                    print("[UGAT-EDL] 已加载现有权重。")

    def _make_env(self) -> dmc.EnvWrapper:
        cfg = self.cfg
        return dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, image_wh=cfg.image_wh)

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    def _make_custom_reward(self) -> tp.Optional[BaseReward]:
        if self.cfg.custom_reward is None:
            return None
        return DmcReward(self.cfg.custom_reward)

    def get_argmax_goal(self, custom_reward):
        num_steps = self.agent.cfg.num_inference_steps
        reward_list, next_obs_list = [], []
        batch_size = 0
        while batch_size < num_steps:
            batch = self.replay_loader.sample(self.cfg.batch_size, custom_reward=custom_reward)
            batch = batch.to(self.cfg.device)
            next_obs_list.append(batch.next_obs)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        reward, next_obs = torch.cat(reward_list, 0), torch.cat(next_obs_list, 0)
        reward_t, next_obs_t = reward[:num_steps], next_obs[:num_steps]
        return next_obs_t[torch.argmax(reward_t)].detach().cpu().numpy()

    def test(self):
        self.global_step = 1
        self.finalize()

    def eval(self, final_eval=False, video_file_name="eval_video.mp4"):
        step, episode = 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        physics_agg = dmc.PhysicsAggregator()
        rewards: tp.List[float] = []
        custom_reward = self._make_custom_reward()
        meta = _init_eval_meta(self, custom_reward)
        videos = []
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            if self.cfg.goal_eval:
                goal = self.get_argmax_goal(custom_reward)
                meta = self.agent.get_goal_meta(goal_array=goal, obs_array=time_step.observation)

            total_reward = 0.0
            video_enabled = True
            self.video_recorder.init(self.eval_env, enabled=video_enabled)

            # 每个 episode 重置比较基准（也可跨 episode 累积）
            self.last_u = float('inf')

            while not time_step.last():
                if self.cfg.goal_eval and self.cfg.agent.name == 'sf' and self.cfg.agent.feature_learner == 'hilp':
                    meta = self.agent.get_goal_meta(goal_array=goal, obs_array=time_step.observation)
                with torch.no_grad(), utils.eval_mode(self.agent):
                    raw_action = self.agent.act(time_step.observation, meta, self.global_step, eval_mode=True)
                    action = raw_action
                    if self.use_gat:
                        try:
                            a_g, u_curr = self.gat.transform_action_with_uq(time_step.observation, raw_action)
                            # ====== UGAT 不确定性门控（比较式）：若 u_t < u_{t-1} 则采纳 grounded ======
                            if u_curr < self.last_u:
                                action = a_g
                                took_grounded = 1
                            else:
                                action = raw_action
                                took_grounded = 0
                            self.last_u = u_curr
                            if self.cfg.use_wandb:
                                wandb.log({"UGAT/u_curr": u_curr,
                                           "UGAT/took_grounded": took_grounded,
                                           "UGAT/global_frame": self.global_frame}, step=self.global_frame)
                        except Exception as e:
                            print(f"[UGAT-EDL] transform 失败，回退原始动作。err={e}")
                            action = raw_action

                time_step = self.eval_env.step(action)
                physics_agg.add(self.eval_env)
                if step % self.cfg.num_skip_frames == 0:
                    self.video_recorder.record(self.eval_env)
                if custom_reward is not None:
                    time_step.reward = custom_reward.from_env(self.eval_env)
                total_reward += time_step.reward
                step += 1

            if video_enabled:
                videos.append(self.video_recorder.frames)
            rewards.append(total_reward)
            episode += 1
            self.video_recorder.save(video_file_name)

        self.eval_rewards_history.append(float(np.mean(rewards)))
        self.video_recorder.save(f"TrajVideo_{video_file_name}.mp4")

        if final_eval:
            return {'episode_reward': self.eval_rewards_history[-1]}, videos

        if len(videos) > 0:
            video = record_video(f'TrajVideo_{self.global_frame}', videos, skip_frames=2)
            wandb.log({'TrajVideo': video}, step=self.global_frame)

    _CHECKPOINTED_KEYS = ('agent', 'global_step', 'global_episode', "replay_loader")

    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        assert isinstance(self.replay_loader, ReplayBuffer)
        payload = {k: self.__dict__[k] for k in self._CHECKPOINTED_KEYS if k not in exclude}
        with fp.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def load_checkpoint_demo_env_targ(self, fp: tp.Union[Path, str], only: tp.Optional[tp.Sequence[str]] = None,
                                      exclude: tp.Sequence[str] = (), num_episodes=None, use_pixels=False) -> None:
        print(f"loading checkpoint from {fp}")
        fp = Path(fp)
        with fp.open('rb') as f:
            payload = torch.load(f)

        if num_episodes is not None:
            payload._episodes_length = payload._episodes_length[:num_episodes]
            payload._max_episodes = min(payload._max_episodes, num_episodes)
            for key, value in payload._storage.items():
                payload._storage[key] = value[:num_episodes]
        if use_pixels:
            payload._storage['observation'] = payload._storage['pixel']
            del payload._storage['pixel']
            payload._batch_names.remove('pixel')

        if isinstance(payload, ReplayBuffer):
            payload = {"replay_loader": payload}
        if only is not None:
            only = list(only); assert all(x in self._CHECKPOINTED_KEYS for x in only)
            payload = {x: payload[x] for x in only}
        exclude = list(exclude); assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        for x in exclude: payload.pop(x, None)
        for name, val in payload.items():
            if name == "agent":
                self.agent.init_from(val)
            elif name == "replay_loader":
                assert isinstance(val, ReplayBuffer)
                val._current_episode.clear()
                val._future = self.cfg.future
                val._discount = self.cfg.discount
                val._max_episodes = len(val._storage["discount"])
                self.replay_loader_env_targ = val
            else:
                assert hasattr(self, name)
                setattr(self, name, val)

    def load_checkpoint(self, fp: tp.Union[Path, str], only: tp.Optional[tp.Sequence[str]] = None,
                        exclude: tp.Sequence[str] = (), num_episodes=None, use_pixels=False) -> None:
        print(f"loading checkpoint from {fp}")
        fp = Path(fp)
        with fp.open('rb') as f:
            payload = torch.load(f)

        if num_episodes is not None:
            payload._episodes_length = payload._episodes_length[:num_episodes]
            payload._max_episodes = min(payload._max_episodes, num_episodes)
            for key, value in payload._storage.items():
                payload._storage[key] = value[:num_episodes]
        if use_pixels:
            payload._storage['observation'] = payload._storage['pixel']
            del payload._storage['pixel']
            payload._batch_names.remove('pixel')

        if isinstance(payload, ReplayBuffer):
            payload = {"replay_loader": payload}
        if only is not None:
            only = list(only); assert all(x in self._CHECKPOINTED_KEYS for x in only)
            payload = {x: payload[x] for x in only}
        exclude = list(exclude); assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        for x in exclude: payload.pop(x, None)
        for name, val in payload.items():
            if name == "agent":
                self.agent.init_from(val)
            elif name == "replay_loader":
                assert isinstance(val, ReplayBuffer)
                val._current_episode.clear()
                val._future = self.cfg.future
                val._discount = self.cfg.discount
                val._max_episodes = len(val._storage["discount"])
                self.replay_loader = val
            else:
                assert hasattr(self, name)
                setattr(self, name, val)

    def finalize(self) -> None:
        print("Running final test", flush=True)
        domain_tasks = {
            "cheetah": ['walk', 'walk_backward', 'run', 'run_backward'],
            "quadruped": ['stand', 'walk', 'run', 'jump'],
            "walker": ['stands', 'walks', 'runs', 'flip'],
            "jaco": ['reach_top_left', 'reach_top_right', 'reach_bottom_left', 'reach_bottom_right'],
        }
        if self.domain not in domain_tasks:
            return
        eval_hist = self.eval_rewards_history
        rewards, videos, infos = {}, {}, {}
        for name in domain_tasks[self.domain]:
            task = "_".join([self.domain, name])
            self.cfg.task = task
            self.cfg.custom_reward = task
            self.cfg.seed += 10000000
            self.eval_env = self._make_env()
            self.eval_rewards_history = []
            self.cfg.num_eval_episodes = self.cfg.num_final_eval_episodes
            video_file_name = f"{self.cfg.task}_final_eval.mp4"
            info, video = self.eval(final_eval=True, video_file_name=video_file_name)
            rewards[name] = self.eval_rewards_history
            infos[name] = info
            videos[name] = video
        self.eval_rewards_history = eval_hist
        with (self.work_dir / "test_rewards.json").open("w") as f:
            json.dump(rewards, f)

# ---- cfg <-> json ----
def save_cfg_to_json(cfg, file_path):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(file_path, 'w') as f:
        json.dump(cfg_dict, f, indent=4)

def read_cfg_from_json(file_path):
    with open(file_path, 'r') as f:
        cfg_dict = json.load(f)
    return OmegaConf.create(cfg_dict)

@hydra.main(config_path='.', config_name='base_config')
def main(cfg: omgcf.DictConfig) -> None:
    custmized_cfg = load_customized_config(f"{parameters['config']['path']}/hilp_zsrl/dlc/allcfg.yaml")
    workspace = Workspace(cfg)
    workspace.test()

if __name__ == '__main__':
    main()
