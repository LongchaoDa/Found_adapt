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
SCRIPT_DIR = CONTEXT.script_dir
PROJECT_DIR = CONTEXT.project_dir
REPO_ROOT = CONTEXT.repo_root
DATA_ROOT = CONTEXT.data_root
SOURCE_REPLAY_PATH = DATA_ROOT / "exorl_learn" / "datasets" / "walker" / "rnd" / "replay.pt"
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from url_benchmark import agent as agents
from url_benchmark import dmc, utils
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.my_utils import record_video
from url_benchmark.video import VideoRecorder

if "mac" in platform.platform():
    pass
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

from pathlib import Path
import sys
base = Path(__file__).absolute().parents[1]
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
        str(SOURCE_REPLAY_PATH)
        # f"{parameters['config']['path_exorl_learn_earth_aligned']}/datasets/walker/rnd/replay.pt"  # from the command
    )
    load_replay_buffer_env_target: tp.Optional[str] = ( 
        None
    )



    
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
    np.random.seed(44)
   
    total_samples = obs.shape[0]
    indices = torch.randperm(total_samples)[:num_samples]
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
        # Compute dimension-wise difference
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

def _init_eval_meta(workspace, custom_reward: BaseReward = None, mode: str = "Direct", lambda_wls_set =4.0):
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

    return workspace.agent.infer_meta_from_obs_and_rewards_sim2real(obs_t, reward_t, next_obs_t, obs_t_env_targ, reward_t_env_targ, next_obs_t_env_targ, False, mode, lambda_wls_set)

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


class Workspace:
    def __init__(self, cfg: Config) -> None:
        # The path for the saved model (should define the path as the root for the saved folder)
        print("Config save_path:", cfg.get("save_path", "save_path not found"))
        print(cfg.save_path)
        default_saved_training = PROJECT_DIR / "exp_local" / "EXP" / "saved_training"
        self.work_dir = Path(cfg.save_path) if cfg.save_path else default_saved_training
        if cfg.load_model is not None:
            self.latest_dir = Path(cfg.load_model).resolve().parents[1]
        else:
            self.latest_dir = default_saved_training
        print(f"Workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        if not torch.cuda.is_available():
            if cfg.device != "cpu":
                logger.warning(f"Falling back to cpu as {cfg.device} is not available")
                cfg.device = "cpu"
                cfg.agent.device = "cpu"
        self.device = torch.device(cfg.device)

        task = cfg.task
        self.domain = task.split('_', maxsplit=1)[0]

        self.train_env = self._make_env()
        
        # Evaluation env:
        self.eval_env = self._make_env()
        # create agent
        self.train_env.reset()
        self.agent = make_agent(cfg.obs_type,
                                cfg.image_wh,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        if cfg.use_wandb:
            exp_name = ''
            exp_name += f'sd{cfg.seed:03d}_'
            if 'SLURM_JOB_ID' in os.environ:
                exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
            if 'SLURM_PROCID' in os.environ:
                exp_name += f'{os.environ["SLURM_PROCID"]}.'
            exp_name += '_'.join([
                cfg.run_group, cfg.agent.name, self.domain,
            ])
            wandb_output_dir = tempfile.mkdtemp()
            wandb.init(project='hilp_zsrl', group=cfg.run_group, name=exp_name,
                       config=omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                       dir=wandb_output_dir)

        self.replay_loader = ReplayBuffer(max_episodes=cfg.replay_buffer_episodes, discount=cfg.discount, future=cfg.future) # current: replay_buffer_episodes=5000,
        
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
        # self.eval()
        """
        DLC:
        After training, we conduct the finalize() action.
        """
        self.finalize()

    def eval(self, final_eval=False, video_file_name="eval_video.mp4", mode="Direct", lambda_wls_set = 4.0):
        step, episode = 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes) # Until(action_repeat=1, until=10)
        physics_agg = dmc.PhysicsAggregator() # initiate the url_benchmark.dmc.PhysicsAggregator object
        rewards: tp.List[float] = []
        custom_reward = self._make_custom_reward()  # not None only if final_eval
        meta = _init_eval_meta(self, custom_reward, mode=mode, lambda_wls_set =lambda_wls_set) # Ours
        videos = []
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            if self.cfg.goal_eval:
                goal = self.get_argmax_goal(custom_reward)
                meta = self.agent.get_goal_meta(goal_array=goal, obs_array=time_step.observation)

            total_reward = 0.0
            video_enabled = (episode < 2) and (self.global_frame % self.cfg.video_every_steps == 0)
            video_enabled = video_enabled and self.cfg.save_video
            self.video_recorder.init(self.eval_env, enabled=video_enabled)
            while not time_step.last():
                if self.cfg.goal_eval and self.cfg.agent.name == 'sf' and self.cfg.agent.feature_learner == 'hilp':
                    # Recompute z every step
                    meta = self.agent.get_goal_meta(goal_array=goal, obs_array=time_step.observation)
                with torch.no_grad(), utils.eval_mode(self.agent):
                    # dlc: Here the agent will take actions: 
                    action = self.agent.act(time_step.observation, meta, self.global_step, eval_mode=True)
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
            if video_enabled:
                self.video_recorder.save(video_file_name)

        self.eval_rewards_history.append(float(np.mean(rewards)))

        # Manually save the video:
        # video = record_video(f'TrajVideo_{self.global_frame}', videos, skip_frames=2)
        if len(videos) > 0:
            self.video_recorder.save(f"TrajVideo_{video_file_name}.mp4")

        
        if final_eval:
            return {
                'episode_reward': self.eval_rewards_history[-1],
            }, videos

        if len(videos) > 0:
            video = record_video(f'TrajVideo_{self.global_frame}', videos, skip_frames=2)
            wandb.log({'TrajVideo': video}, step=self.global_frame)


    _CHECKPOINTED_KEYS = ('agent', 'global_step', 'global_episode', "replay_loader")

    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        # logger.info(f"Saving checkpoint to {fp}")
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        assert isinstance(self.replay_loader, ReplayBuffer), "Is this buffer designed for checkpointing?"
        # this is just a dumb security check to not forget about it
        payload = {k: self.__dict__[k] for k in self._CHECKPOINTED_KEYS if k not in exclude}
        with fp.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def load_checkpoint_demo_env_targ(
            self,
            fp: tp.Union[Path, str],
            only: tp.Optional[tp.Sequence[str]] = None,
            exclude: tp.Sequence[str] = (),
            num_episodes=None,
            use_pixels=False
    ) -> None:
        print(f"loading checkpoint from {fp}")
        fp = Path(fp)
        map_location = self.device if self.device.type == "cpu" else None
        with fp.open('rb') as f:
            try:
                payload = torch.load(f, map_location=map_location, weights_only=False)
            except TypeError:
                payload = torch.load(f, map_location=map_location)

        if num_episodes is not None:
            payload._episodes_length = payload._episodes_length[:num_episodes]
            payload._max_episodes = min(payload._max_episodes, num_episodes)
            for key, value in payload._storage.items():
                payload._storage[key] = value[:num_episodes]
        if use_pixels:
            payload._storage['observation'] = payload._storage['pixel']
            del payload._storage['pixel']
            payload._batch_names.remove('pixel')

        if isinstance(payload, ReplayBuffer):  # compatibility with pure buffers pickles
            payload = {"replay_loader": payload}
        if only is not None:
            only = list(only)
            assert all(x in self._CHECKPOINTED_KEYS for x in only)
            payload = {x: payload[x] for x in only}
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        for x in exclude:
            payload.pop(x, None)
        for name, val in payload.items():
            if name == "agent":
                self.agent.init_from(val)
            elif name == "replay_loader":
                assert isinstance(val, ReplayBuffer)
                # pylint: disable=protected-access
                val._current_episode.clear()  # make sure we can start over
                val._future = self.cfg.future
                val._discount = self.cfg.discount
                val._max_episodes = len(val._storage["discount"])
                # The only change: assign to self.replay_loader_env_targ instead of self.replay_loader
                self.replay_loader_env_targ = val
            else:
                assert hasattr(self, name)
                setattr(self, name, val)


    # We have the load checkpoint function here
    def load_checkpoint(self, fp: tp.Union[Path, str], only: tp.Optional[tp.Sequence[str]] = None, exclude: tp.Sequence[str] = (), num_episodes=None, use_pixels=False) -> None:
        print(f"loading checkpoint from {fp}")
        fp = Path(fp)
        map_location = self.device if self.device.type == "cpu" else None
        with fp.open('rb') as f:
            try:
                payload = torch.load(f, map_location=map_location, weights_only=False)
            except TypeError:
                payload = torch.load(f, map_location=map_location)

        if num_episodes is not None:
            payload._episodes_length = payload._episodes_length[:num_episodes]
            payload._max_episodes = min(payload._max_episodes, num_episodes)
            for key, value in payload._storage.items():
                payload._storage[key] = value[:num_episodes]
        if use_pixels:
            payload._storage['observation'] = payload._storage['pixel']
            del payload._storage['pixel']
            payload._batch_names.remove('pixel')

        if isinstance(payload, ReplayBuffer):  # compatibility with pure buffers pickles
            payload = {"replay_loader": payload}
        if only is not None:
            only = list(only)
            assert all(x in self._CHECKPOINTED_KEYS for x in only)
            payload = {x: payload[x] for x in only}
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        for x in exclude:
            payload.pop(x, None)
        for name, val in payload.items():
            # logger.info("Reloading %s from %s", name, fp)
            if name == "agent":
                self.agent.init_from(val)
            elif name == "replay_loader":
                assert isinstance(val, ReplayBuffer)
                # pylint: disable=protected-access
                # drop unecessary meta which could make a mess
                val._current_episode.clear()  # make sure we can start over
                val._future = self.cfg.future
                val._discount = self.cfg.discount
                val._max_episodes = len(val._storage["discount"])
                # replay loader has been re-written here: dlc
                self.replay_loader = val
            else:
                assert hasattr(self, name)
                setattr(self, name, val)
                # if name == "global_episode":
                #     logger.warning(f"Reloaded agent at global episode {self.global_episode}")

    def finalize(self) -> None:
        print("Running final test", flush=True)

        domain_tasks = {
            # "cheetah": ['walk', 'walk_backward', 'run', 'run_backward'],
            # "quadruped": ['stand', 'walk', 'run', 'jump'],
            "walker": ['stands'],
            # "jaco": ['reach_top_left', 'reach_top_right', 'reach_bottom_left', 'reach_bottom_right'],
        }
        if self.domain not in domain_tasks:
            return
        eval_hist = self.eval_rewards_history
        rewards = {}
        videos = {}
        infos = {}
        for name in domain_tasks[self.domain]:
            task = "_".join([self.domain, name])
            self.cfg.task = task
            self.cfg.custom_reward = task  # for the replay buffer
            self.cfg.seed += 10000000  # for the sake of avoiding similar seeds
            self.eval_env = self._make_env()
            self.eval_rewards_history = []
            self.cfg.num_eval_episodes = self.cfg.num_final_eval_episodes
            video_file_name = f"{self.cfg.task}_final_eval.mp4"
            # info, video = self.eval(final_eval=True, video_file_name=video_file_name, mode="Direct")
            info, video = self.eval(final_eval=True, video_file_name=video_file_name, mode=self.cfg.algo_mode, lambda_wls_set = self.cfg.algo_lambda_wls_set)
            # mode=self.cfg.algo_mode
            rewards[name] = self.eval_rewards_history
            infos[name] = info
            videos[name] = video

        self.eval_rewards_history = eval_hist  # restore
        with (self.work_dir / "test_rewards.json").open("w") as f:
            json.dump(rewards, f)





def save_cfg_to_json(cfg, file_path):
    # Convert DictConfig to a plain Python dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Convert the dictionary to JSON and save it to the specified file
    with open(file_path, 'w') as f:
        json.dump(cfg_dict, f, indent=4)


def read_cfg_from_json(file_path):
    # Load the JSON file and convert it to a plain Python dictionary
    with open(file_path, 'r') as f:
        cfg_dict = json.load(f)
    
    # Convert the dictionary back to a DictConfig object
    cfg = OmegaConf.create(cfg_dict)
    return cfg



@hydra.main(config_path='.', config_name='base_config')
def main(cfg: omgcf.DictConfig) -> None:
    workspace = Workspace(cfg)

    workspace.test()


if __name__ == '__main__':
    main()
