import os
# 使用 EGL 作为离屏渲染后端（确保你的系统支持 EGL）
os.environ['MUJOCO_GL'] = 'egl'

import subprocess
import re
import shutil
from pathlib import Path
import toml
import sys

from dm_env import specs

# 读取配置参数
with open("/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/parameters.toml", "r") as f:
    parameters = toml.load(f)

# 将工程路径加入到 sys.path 中
sys.path.append(f"{parameters['config']['path']}/hilp_zsrl/")


def backup_original_xml(xml_path: Path) -> Path:
    """
    Creates a backup of the original XML file.
    
    Parameters:
        xml_path (Path): 原始 XML 文件路径.
        
    Returns:
        Path: 备份文件路径.
    """
    backup_path = xml_path.with_suffix(".backup.xml")
    shutil.copyfile(xml_path, backup_path)
    return backup_path


def restore_original_xml(xml_path: Path, backup_path: Path):
    """
    Restores the original XML file from the backup.
    
    Parameters:
        xml_path (Path): 原始 XML 文件路径.
        backup_path (Path): 备份文件路径.
    """
    shutil.copyfile(backup_path, xml_path)
    backup_path.unlink()  # 恢复后删除备份文件


def update_gravity_in_xml(xml_path: Path, new_gravity: tuple):
    """
    Updates the gravity setting in a MuJoCo XML file.

    Parameters:
        xml_path (Path): Path to the XML file.
        new_gravity (tuple): New gravity vector as (x, y, z).
            如果 tuple 中任一值为 None，则打印文件中的当前重力值。
    """
    with open(xml_path, "r") as file:
        xml_content = file.read()

    # Regex 用于匹配 gravity 属性
    gravity_pattern = r'gravity="([^"]+)"'

    # 查找 gravity 属性
    match = re.search(gravity_pattern, xml_content)
    if match:
        current_gravity = match.group(1)
        if None in new_gravity:
            print(f"Current gravity values in the XML: {current_gravity}")
            return
    else:
        print("No gravity attribute found in the XML.")
        return

    # 构建新的 gravity 字符串
    new_gravity_str = f'gravity="{new_gravity[0]} {new_gravity[1]} {new_gravity[2]}"'

    # 替换 XML 中的 gravity 属性
    modified_xml = re.sub(gravity_pattern, new_gravity_str, xml_content)

    # 保存更新后的 XML 文件
    with open(xml_path, "w") as file:
        file.write(modified_xml)

    print(f"Updated gravity values to: {new_gravity_str}")


def run_test_offline(save_folder: Path):
    """
    Runs the test_offline.py command with specified parameters.

    Parameters:
        save_folder (Path): 保存测试结果的文件夹.
    """
    # 创建测试视频保存目录
    test_video_folder = save_folder / "test_video"
    test_video_folder.mkdir(parents=True, exist_ok=True)

    # 构建运行命令
    command = [
        "python",
        "url_benchmark/debug_offline.py",
        "run_group=EXP",
        "device=cuda",
        "agent=sf",
        "agent.feature_learner=hilp",
        "p_randomgoal=0.375",
        "agent.hilp_expectile=0.5",
        "agent.hilp_discount=0.96",
        "agent.q_loss=False",
        "seed=0",
        "task=walker_run",
        "expl_agent=rnd",
        f"load_model={parameters['config']['path']}/hilp_zsrl/exp_local/EXP/saved_training/models/latest.pt",
        # "load_model=/scratch/longchao/project/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/EXP/explore_train/moon train/models/latest.pt",
        # "load_model=/scratch/longchao/project/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/EXP/explore_train/jupiter_train/models/latest.pt",
        # /scratch/longchao/project/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/EXP/explore_train/moon train/models/latest.pt
        f"load_replay_buffer={parameters['config']['path_exorl_learn']}/datasets/walker/rnd/replay.pt",  # original one
        # f"load_replay_buffer={parameters['config']['path_exorl_learn_44']}/datasets/walker/rnd/replay.pt", # original representation guided NN with guidance totally collect from env 44: saved_gravity_pickout44_with44demo
        "replay_buffer_episodes=5000",
        f"save_path={save_folder}"
    ]
    print(command)
    # 执行命令
    subprocess.run(command)


def run_tests_with_different_gravity(xml_path: Path, gravities: dict, base_result_folder: Path):
    """
    Runs tests with different gravity settings by updating the MuJoCo XML.

    Parameters:
        xml_path (Path): Path to the MuJoCo XML file.
        gravities (dict): 一个字典，键为测试名称，值为重力设置 tuple.
        base_result_folder (Path): 基础结果保存文件夹.
    """
    # Step 1: 备份原始 XML 文件
    backup_path = backup_original_xml(xml_path)

    try:
        for planet, gravity in gravities.items():
            print(f"\nRunning test {planet} with gravity: {gravity}")

            # Step 2: 更新 XML 中的重力设置
            update_gravity_in_xml(xml_path, gravity)

            # Step 3: 为该重力设置创建结果保存文件夹
            result_folder = base_result_folder / f"saved_training_{planet}"
            result_folder.mkdir(parents=True, exist_ok=True)

            # Step 4: 运行 test_offline.py 命令，并保存结果
            run_test_offline(result_folder)

        print("\nAll tests completed.")
    finally:
        # Step 5: 测试结束后恢复原始 XML 文件
        restore_original_xml(xml_path, backup_path)
        print("Original XML file has been restored.")


# Example usage:
xml_path = Path(f"{parameters['config']['path']}/hilp_zsrl/url_benchmark/custom_dmc_tasks/walker.xml")
base_result_folder = Path(
    f"{parameters['config']['path']}/hilp_zsrl/exp_local/dlc_25/earth"
)
gravities = {
    # "Origin": (None, None, None),
    # "Earth": (0, 0, -9.81),
    # "Mars": (0, 0, -3.71),
    # "Jupiter": (0, 0, -24.79),
    # "Moon": (0, 0, -1.62)  # Earth's Moon
    # "Origin": (None, None, None),
    "Earth": (0, 0, -9.81),
    # "-15": (0, 0, -15),
    # "-24": (0, 0, -24.79),
    # "-34": (0, 0, -34),  # Earth's Moon
    # "-44": (0, 0, -44),  # Earth's Moon
}  # 重力设置列表

# 执行不同重力测试
run_tests_with_different_gravity(xml_path, gravities, base_result_folder)

# 以下为历史代码备注:
# path_collect = "/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/url_verify_solved/url_benchmark/exp_local/2024.12.17/034327_rnd/verify/datasets/walker/rnd/replay.pt"
# 44 replay collected data: /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/url_verify_solved/url_benchmark/exp_local/2024.12.17/034327_rnd/verify/datasets/walker/rnd/replay.pt
