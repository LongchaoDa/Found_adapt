import os
# 使用 EGL 作为离屏渲染后端（确保你的系统支持 EGL）
os.environ.setdefault("MUJOCO_GL", "egl")
import subprocess
import re
import shutil
import json
from pathlib import Path
import sys
from runtime_paths import load_repo_parameters
try:
    from dm_env import specs
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency: dm_env. Install with: pip install dm-env"
    ) from exc
from datetime import datetime
import argparse

CONTEXT, parameters = load_repo_parameters(__file__)
SCRIPT_DIR = CONTEXT.script_dir
PROJECT_DIR = CONTEXT.project_dir
REPO_ROOT = CONTEXT.repo_root
DATA_ROOT = CONTEXT.data_root
MODEL_PATH = DATA_ROOT / "Sim2RealFoundationPolicy" / "hilp_zsrl" / "exp_local" / "EXP" / "saved_training" / "models" / "latest.pt"
SOURCE_REPLAY_PATH = DATA_ROOT / "exorl_learn" / "datasets" / "walker" / "rnd" / "replay.pt"
ALIGN15_REPLAY_PATH = DATA_ROOT / "url_verify_solved" / "url_benchmark" / "exp_local" / "2025.05.13" / "align_15" / "datasets" / "walker" / "rnd" / "replay.pt"


# =========================
# NEW: command-line args
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None,
                    help="Optional manual seed override. If omitted, the script uses the default seed for the selected config.")
parser.add_argument("--mode", type=str, default="Direct", choices=["Direct", "Ours"],
                    help='mode = "Direct" # Direct, Ours, ...')
parser.add_argument("--config", type=str, default="config_g0",
                    help='Specify variable config name, e.g., "config_g0". Use "all" to run all configs.')
parser.add_argument("--repeat", type=int, default=1,
                    help="Repeat the whole run N times (default 1).")
parser.add_argument("--lambda_wls_set", type=float, default=4.0,
                    help="The lambda_wls value to use in Ours mode (default 4.0).")
args = parser.parse_args()

mode = args.mode # Direct, Ours, ...
lambda_wls_set = args.lambda_wls_set

DEFAULT_SEEDS = {
    "config_g0": 88,
    "config_g1": 72,
    "config_g2": 169,
    "config_g3": 181,
    "config_g4": 91,
}

if args.seed is not None:
    seed = args.seed
elif args.config == "all":
    seed = DEFAULT_SEEDS["config_g0"]
else:
    seed = DEFAULT_SEEDS.get(args.config, DEFAULT_SEEDS["config_g0"])

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

def backup_original_xml(xml_path: Path) -> Path:
    """Creates a backup of the original XML file without copying file mode."""
    backup_path = xml_path.with_suffix(".backup.xml")
    shutil.copyfile(xml_path, backup_path)
    return backup_path

def restore_original_xml(xml_path: Path, backup_path: Path):
    """Restores the original XML file from the backup without copying file mode."""
    shutil.copyfile(backup_path, xml_path)
    backup_path.unlink()

def update_gravity_in_xml(xml_path: Path, new_gravity: tuple):
    """
    Updates the gravity setting in a MuJoCo XML file.
    If any value in the tuple is None, reads and prints the current gravity without modifying.
    """
    with open(xml_path, "r") as file:
        xml = file.read()
    pat = r'gravity="([^"]+)"'
    m = re.search(pat, xml)
    if not m:
        print("No gravity attribute found in the XML.")
        return
    curr = m.group(1)
    if None in new_gravity:
        print(f"Current gravity values in the XML: {curr}")
        return
    new_str = f'gravity="{new_gravity[0]} {new_gravity[1]} {new_gravity[2]}"'
    modified = re.sub(pat, new_str, xml)
    with open(xml_path, "w") as file:
        file.write(modified)
    print(f"Updated gravity values to: {new_str}")

def update_friction_in_xml(xml_path: Path, new_friction: tuple):
    """
    Updates the friction setting specifically for the <geom name="floor"> in a MuJoCo XML file.
    If any value in the tuple is None, reads and prints the current friction without modifying.
    """
    with open(xml_path, "r") as file:
        xml_content = file.read()

    floor_pattern = r'(<geom[^>]*\bname="floor"[^>]*?)\bfriction="([^"]+)"([^>]*>)'
    match = re.search(floor_pattern, xml_content, flags=re.DOTALL)
    if not match:
        print("No <geom name='floor'> with a friction attribute found in the XML.")
        return

    current_fric = match.group(2)
    if None in new_friction:
        print(f"Current friction values in the XML: {current_fric}")
        return

    new_fric_str = f'friction="{new_friction[0]} {new_friction[1]} {new_friction[2]}"'
    modified = re.sub(
        floor_pattern,
        lambda m: f"{m.group(1)} {new_fric_str}{m.group(3)}",
        xml_content,
        flags=re.DOTALL
    )
    with open(xml_path, "w") as file:
        file.write(modified)
    print(f"Updated friction values to: {new_fric_str}")

def run_test_offline(save_folder: Path, env_replay: str):
    """Runs the test_offline.py command with specified parameters."""
    test_video_folder = save_folder / "test_video"
    test_video_folder.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, str(SCRIPT_DIR / "debug_offline.py"),
        "run_group=EXP", "device=cuda",
        "save_video=False", "use_wandb=False",
        "agent=sf", "agent.feature_learner=hilp",
        "p_randomgoal=0.375",
        "agent.hilp_expectile=0.5", "agent.hilp_discount=0.96",
        "agent.q_loss=False", f"seed={seed}", "task=walker_run", # seed = 44, 0 , 88, 100
        "expl_agent=rnd",
        f"load_model={MODEL_PATH}",
        f"load_replay_buffer={SOURCE_REPLAY_PATH}",
        f"load_replay_buffer_env_target={env_replay}",
        "replay_buffer_episodes=5000",
        f"save_path={save_folder}",
        # NEW: set algo_mode for infer_meta_from_obs_and_rewards_sim2real in debug_offline.py
        # NOTE: this requires debug_offline.py's Hydra Config to include `algo_mode` (or allow it via structured config).
        f"+algo_mode={mode}",
        f"+algo_lambda_wls_set={lambda_wls_set}",
    ]
    env = os.environ.copy()
    if shutil.which("xvfb-run"):
        # Prefer a virtual X server on headless/shared machines where EGL is unreliable.
        env["MUJOCO_GL"] = "glfw"
        command = ["xvfb-run", "-a"] + command
    print("RUNNING OFFLINE:", command)
    subprocess.run(command, cwd=PROJECT_DIR, check=True, env=env)
    _print_run_results(save_folder)


def _replay_file_from_config_key(config_key: str) -> Path:
    base = Path(parameters["config"][config_key])
    return base / "datasets" / "walker" / "rnd" / "replay.pt"


def resolve_env_replay(cfg_name: str) -> Path:
    candidate_keys = {
        "config_g0": [],
        "config_g1": [
            "path_exorl_learn_15_aligned_dlc",
            "path_exorl_learn_24_aligned",
            "path_exorl_learn_24_true_RMSPROP",
            "path_exorl_learn_24_true",
        ],
        "config_g2": [
            "path_exorl_learn_24_aligned_dlc",
            "path_exorl_learn_24_aligned",
            "path_exorl_learn_24_true_RMSPROP",
            "path_exorl_learn_24_true",
        ],
        "config_g3": [
            "path_exorl_learn_34_aligned_dlc",
            "path_exorl_learn_mixed_true",
            "path_exorl_learn_24_true",
        ],
        "config_g4": [
            "path_exorl_learn_44_aligned_dlc",
            "path_exorl_learn_44_true",
            "path_exorl_learn_44",
        ],
        "config_f1": ["path_exorl_learn_fri_4_aligned_dlc"],
        "config_f2": ["path_exorl_learn_fri_5_aligned_dlc"],
        "config_f3": ["path_exorl_learn_fri_6_aligned_dlc"],
        "config_f4": ["path_exorl_learn_fri_7_aligned_dlc"],
        "config_f5": ["path_exorl_learn_fri_8_aligned_dlc"],
        "config_f6": ["path_exorl_learn_fri_18_aligned_dlc"],
    }

    if cfg_name == "config_g0":
        return SOURCE_REPLAY_PATH

    for key in candidate_keys.get(cfg_name, []):
        if key in parameters["config"]:
            replay_path = _replay_file_from_config_key(key)
            if replay_path.exists():
                return replay_path

    if SOURCE_REPLAY_PATH.exists():
        print(f"[WARN] No target replay found for {cfg_name}; falling back to source replay {SOURCE_REPLAY_PATH}")
        return SOURCE_REPLAY_PATH

    raise FileNotFoundError(f"No usable replay buffer found for {cfg_name}")


def _print_run_results(save_folder: Path):
    """Print one-line result per method and the result file path."""
    results_path = save_folder / "test_rewards.json"
    print(f"[RESULT] output_dir={save_folder}")
    print(f"[RESULT] rewards_file={results_path}")
    if not results_path.exists():
        print("[RESULT] rewards_file not found (run may have failed or not produced results).")
        return
    try:
        with results_path.open("r") as f:
            rewards = json.load(f)
    except Exception as exc:
        print(f"[RESULT] failed to read rewards_file: {exc}")
        return

    for method, vals in rewards.items():
        if isinstance(vals, list) and len(vals) > 0:
            try:
                mean_val = sum(vals) / len(vals)
                print(f"[RESULT] {method}: mean={mean_val:.4f} (n={len(vals)})")
            except Exception:
                print(f"[RESULT] {method}: {vals}")
        else:
            print(f"[RESULT] {method}: {vals}")

def run_tests_with_different_friction(xml_path: Path, frictions: dict, base_result_folder: Path):
    backup_path = backup_original_xml(xml_path)
    try:
        for surface, friction in frictions.items():
            print(f"\n--- Friction test {surface} → {friction} ---")

            # 1. 如果是 default 摩擦，先打印并显式设为 Earth 重力
            if None in friction:
                update_friction_in_xml(xml_path, friction)
                update_gravity_in_xml(xml_path, (0, 0, -9.81))
                env_replay = (
                    f"{ALIGN15_REPLAY_PATH}"
                )
            else:
                # 非默认摩擦：修改摩擦、保持默认重力
                update_friction_in_xml(xml_path, friction)
                update_gravity_in_xml(xml_path, (0, 0, -9.81))
                env_replay = (
                    f"{parameters['config'][f'path_exorl_learn_fri_{surface}_aligned_dlc']}"
                    f"/datasets/walker/rnd/replay.pt"
                )

            # 2. 准备结果文件夹
            result_folder = base_result_folder / f"friction_{surface}"
            result_folder.mkdir(parents=True, exist_ok=True)

            # 3. 运行离线测试
            run_test_offline(result_folder, env_replay)

        print("\n✓ All friction tests completed.")
    finally:
        restore_original_xml(xml_path, backup_path)
        print("Original XML file has been restored.")

# =========================
# NEW: variables-based runner
# =========================
def run_tests_with_variables(xml_path: Path, variables: dict, base_result_folder: Path, default_gravity=(0, 0, -9.81)):
    """
    variables: dict[str, tuple[tuple|None, tuple|None]]
        key -> config name (used in result folder name and env key suffix)
        value -> (gravity_tuple_or_None, friction_tuple_or_None)
                 gravity: (gx, gy, gz) or None (means force default_gravity)
                 friction: (sx, sy, sz) or tuple containing None(s) to only read/print
    Behavior:
      - If gravity is None => force set to default_gravity (as required when only changing friction).
      - If friction contains None => read/print friction and use the 'origin' env path.
      - Else => write friction and use fri_{config} env path.
    """
    backup_path = backup_original_xml(xml_path)
    try:
        for cfg_name, pair in variables.items():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                print(f"[WARN] Skip invalid variables[{cfg_name}]={pair}")
                continue
            gravity, friction = pair
            print(f"\n--- Config {cfg_name} ---")
            print(f"Gravity: {gravity} | Friction: {friction}")

            # Gravity handling
            if gravity is None:
                update_gravity_in_xml(xml_path, default_gravity)
            else:
                update_gravity_in_xml(xml_path, gravity)

            # Friction handling + env path
            if friction is None or (isinstance(friction, (list, tuple)) and (None in friction)):
                update_friction_in_xml(xml_path, friction if friction is not None else (None, None, None))
                env_replay = str(resolve_env_replay(cfg_name))
                result_folder = base_result_folder / f"variables_{cfg_name}_origin"
            else:
                update_friction_in_xml(xml_path, friction)
                env_replay = str(resolve_env_replay(cfg_name))
                result_folder = base_result_folder / f"variables_{cfg_name}"

            result_folder.mkdir(parents=True, exist_ok=True)
            run_test_offline(result_folder, env_replay)

        print("\n✓ All variables-based tests completed.")
    finally:
        restore_original_xml(xml_path, backup_path)
        print("Original XML file has been restored.")

# =========================
# Example usage:
# =========================
xml_path = PROJECT_DIR / "url_benchmark" / "custom_dmc_tasks" / "walker.xml"
# base_result_folder = Path(f"{parameters['config']['path']}/hilp_zsrl/exp_local/0official/lossanalysis")
# base_result_folder = Path(f"{parameters['config']['path']}/hilp_zsrl/exp_local/0rebuttal/lambda/3_testtime_{seed}_{mode}")
base_result_folder = PROJECT_DIR / "exp_local" / "0CameraRevalidate" / "InDomain_Test_camera_lamda_verify" / f"pinPoint_Test_{seed}_{mode}_lambda_{lambda_wls_set}"


# 原有：仅按摩擦测试
# frictions = {
#     "Origin": (None, None, None), # 1.0 0.1 0.1
#     # "4": (4, 0.4, 0.4),
#     # "5": (5, 0.5, 0.5),
# }

# 新增：按 (gravity, friction) 成对控制
variables = {
    # garvity: 
    "config_g0": ((0, 0,  -9.81), (1.0, 0.1, 0.1)),  # 第一个（gravity的参数），第二个（friction的参数）
    "config_g1": ((0, 0, -15), (1.0, 0.1, 0.1)),  # 第一个（gravity的参数），第二个（friction的参数）
    "config_g2": ((0, 0, -24), (1.0, 0.1, 0.1)),  # 第一个（gravity的参数），第二个（friction的参数）
    "config_g3": ((0, 0, -34), (1.0, 0.1, 0.1)),  # 第一个（gravity的参数），第二个（friction的参数）
    "config_g4": ((0, 0, -44), (1.0, 0.1, 0.1)),  # 第一个（gravity的参数），第二个（friction的参数）
    # # friction: 
    "config_f1": ((0, 0, -9.81), (4, 0.4, 0.4)),  # 第一个（gravity的参数），第二个（friction的参数）
    "config_f2": ((0, 0, -9.81), (5, 0.5, 0.5)),  # 第一个（gravity的参数），第二个（friction的参数）
    "config_f3": ((0, 0, -9.81), (6, 0.6, 0.6)),  # 第一个（gravity的参数），第二个（friction的参数）
    "config_f4": ((0, 0, -9.81), (7, 0.7, 0.7)),  # 第一个（gravity的参数），第二个（friction的参数）
    "config_f5": ((0, 0, -9.81), (8, 0.8, 0.8)),  # 第一个（gravity的参数），第二个（friction的参数）
    "config_f6": ((0, 0, -9.81), (18, 1.8, 1.8)),  # 第一个（gravity的参数），第二个（friction的参数）
    # "config2":  ((0, 0, -15), (4, 0.4, 0.4)),       # gravity=None => 强制默认重力；设定摩擦
    # "config3":  ((0, 0, -24), (5, 0.5, 0.5)),
    # "config4":  ((0, 0, -34), (6, 0.6, 0.6)),
    # "config5":  ((0, 0, -44), (7, 0.7, 0.7)),
    # "config3": ((0, 0, -15), (None, None, None)),  # 只打印摩擦，用 origin env
}

# 如需沿用旧流程：
# run_tests_with_different_friction(xml_path, frictions, base_result_folder)

# NEW: allow selecting a single config from command line
if args.config != "all":
    if args.config not in variables:
        raise ValueError(f"--config {args.config} not found in variables. Available: {list(variables.keys())}")
    variables_to_run = {args.config: variables[args.config]}
else:
    variables_to_run = variables

start = datetime.now()
for _ in range(args.repeat):
    # 新流程：使用 variables 控制 (gravity, friction)
    run_tests_with_variables(xml_path, variables_to_run, base_result_folder, default_gravity=(0, 0, -9.81))
end = datetime.now()

dur = (end - start).total_seconds()
ave = dur / max(args.repeat, 1)

print(f"[time] run_tests_with_variables took {ave:.2f}s")


# TODO: Baseline:
# python /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/test_multi_surface_offline_bothfrictionandGarvity0.py --seed 88 --mode Direct --config config_g0

# TODO: run Gravity directTransfer: 
# python /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/test_multi_surface_offline_bothfrictionandGarvity0.py --seed 0 --mode Direct --config all

# ✅ Reproduce G1 (seed 72)
# python /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/test_multi_surface_offline_bothfrictionandGarvity0.py \
#   --seed 72 \
#   --mode Direct \
#   --config config_g1

# ✅ Reproduce G2 (seed 169)
# python /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/test_multi_surface_offline_bothfrictionandGarvity0.py \
#   --seed 169 \
#   --mode Direct \
#   --config config_g2

# ✅ Reproduce G3 (seed 181)

# python /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/test_multi_surface_offline_bothfrictionandGarvity0.py \
#   --seed 181 \
#   --mode Direct \
#   --config config_g3

# ✅ Reproduce G4 (seed 91)
# python /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/test_multi_surface_offline_bothfrictionandGarvity0.py \
#   --seed 91 \
#   --mode Direct \
#   --config config_g4

# Our method: 

# ✅ Reproduce G1 (seed 72)
# python /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/test_multi_surface_offline_bothfrictionandGarvity0.py \
#   --seed 72 \
#   --mode Ours \
#   --config config_g1

# ✅ Reproduce G2 (seed 169)
# python /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/test_multi_surface_offline_bothfrictionandGarvity0.py \
#   --seed 169 \
#   --mode Ours \
#   --config config_g2

# ✅ Reproduce G3 (seed 181)

# python /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/test_multi_surface_offline_bothfrictionandGarvity0.py \
#   --seed 181 \
#   --mode Ours \
#   --config config_g3

# ✅ Reproduce G4 (seed 91)
# python /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/test_multi_surface_offline_bothfrictionandGarvity0.py \
#   --seed 91 \
#   --mode Ours \
#   --config config_g4
