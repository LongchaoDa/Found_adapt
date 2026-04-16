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

with open("/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/parameters.toml", "r") as f:
    parameters = toml.load(f)
sys.path.append(f"{parameters['config']['path']}/hilp_zsrl/")
def backup_original_xml(xml_path: Path) -> Path:
    """Creates a backup of the original XML file."""
    backup_path = xml_path.with_suffix(".backup.xml")
    shutil.copyfile(xml_path, backup_path)
    return backup_path

def restore_original_xml(xml_path: Path, backup_path: Path):
    """Restores the original XML file from the backup."""
    shutil.copyfile(backup_path, xml_path)
    backup_path.unlink()  # Delete the backup after restoration


def update_gravity_in_xml(xml_path: Path, new_gravity: tuple):
    """
    Updates the gravity setting in a MuJoCo XML file.

    Parameters:
    - xml_path (Path): Path to the XML file.
    - new_gravity (tuple): New gravity vector as (x, y, z).
      If any value in the tuple is None, reads the current gravity values from the file and prints them.
    """
    with open(xml_path, "r") as file:
        xml_content = file.read()

    # Regex to match the gravity attribute
    gravity_pattern = r'gravity="([^"]+)"'

    # Search for the gravity attribute
    match = re.search(gravity_pattern, xml_content)
    if match:
        current_gravity = match.group(1)
        if None in new_gravity:
            print(f"Current gravity values in the XML: {current_gravity}")
            return
    else:
        print("No gravity attribute found in the XML.")
        return

    # Create a string for the new gravity values
    new_gravity_str = f'gravity="{new_gravity[0]} {new_gravity[1]} {new_gravity[2]}"'

    # Replace the gravity attribute in the XML content
    modified_xml = re.sub(gravity_pattern, new_gravity_str, xml_content)

    # Save the updated XML
    with open(xml_path, "w") as file:
        file.write(modified_xml)

    print(f"Updated gravity values to: {new_gravity_str}")

def run_test_offline(save_folder: Path):

    test_video_folder = save_folder / "test_video"
    test_video_folder.mkdir(parents=True, exist_ok=True)
    """Runs the test_offline.py command with specified parameters."""
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
        # f"load_model=/scratch/longchao/project/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/EXP/explore_train/moon train/models/latest.pt",
        # f"load_model=/scratch/longchao/project/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/EXP/explore_train/jupiter_train/models/latest.pt",
        # /scratch/longchao/project/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/EXP/explore_train/moon train/models/latest.pt
        # f"load_replay_buffer={parameters['config']['path_exorl_learn']}/datasets/walker/rnd/replay.pt", # original one
        # f"load_replay_buffer={parameters['config']['path_exorl_learn_44']}/datasets/walker/rnd/replay.pt", # original representation guided NN with guidance totally collect from env 44: saved_gravity_pickout44_with44demo
        "replay_buffer_episodes=5000",
        f"save_path={save_folder}"
    ]
    print(command)
    # Run the command
    subprocess.run(command)

def run_tests_with_different_gravity(xml_path: Path, gravities: dict, base_result_folder: Path):
    # Step 1: Back up the original XML
    backup_path = backup_original_xml(xml_path)

    try:
        for planet, gravity in gravities.items():
            print(f"\nRunning test {planet} with gravity: {gravity}")

            # Step 2: Update gravity in XML
            update_gravity_in_xml(xml_path, gravity)

            # Step 3: Create a results folder for this gravity setting
            result_folder = base_result_folder / f"saved_training_{planet}"
            result_folder.mkdir(parents=True, exist_ok=True)

            # Step 4: Run the test_offline.py command and save results
            run_test_offline(result_folder)

        print("\nAll tests completed.")
        
    finally:
        # Step 5: Restore the original XML after all tests
        restore_original_xml(xml_path, backup_path)
        print("Original XML file has been restored.")

# Example usage:
# /scratch/longchao/project/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/exp_local/EXP/saved_gravity_verify
xml_path = Path(f"{parameters['config']['path']}/hilp_zsrl/url_benchmark/custom_dmc_tasks/walker.xml")
base_result_folder = Path(f"{parameters['config']['path']}/hilp_zsrl/exp_local/dlc_25/try_considerNextobs")
gravities = {
    # "Origin": (None, None, None),
    # "Earth": (0, 0, -9.81),
    # "Mars": (0, 0, -3.71),
    # "Jupiter": (0, 0, -24.79),
    # "Moon": (0, 0, -1.62)  # Earth's Moon
    # "Origin": (None, None, None),
    # "Earth": (0, 0, -9.81),
    # "-15": (0, 0, -15),
    "-24": (0, 0, -24.79),
    # "-34": (0, 0, -34),  # Earth's Moon
    # "-44": (0, 0, -44),  # Earth's Moon
}  # List of gravity settings

run_tests_with_different_gravity(xml_path, gravities, base_result_folder)

# path_collect = "/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/url_verify_solved/url_benchmark/exp_local/2024.12.17/034327_rnd/verify/datasets/walker/rnd/replay.pt"

# 44 replay collected data: /home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/url_verify_solved/url_benchmark/exp_local/2024.12.17/034327_rnd/verify/datasets/walker/rnd/replay.pt