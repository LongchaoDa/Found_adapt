import subprocess
import re
import shutil
from pathlib import Path
import toml
import sys

# to test different surface conditions and other parameters
from dm_env import specs

with open("parameters.toml", "r") as f:
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

# def update_wind_in_xml(xml_path: Path, new_wind: tuple):
#     """
#     Updates or adds the wind setting in a MuJoCo XML file.
    
#     Parameters:
#     - xml_path (Path): Path to the XML file.
#     - new_wind (tuple): New wind values as (x, y, z).
#     """
#     with open(xml_path, "r") as file:
#         xml_content = file.read()
    
#     # Create a string for the new wind values
#     new_wind_str = f'wind="{new_wind[0]} {new_wind[1]} {new_wind[2]}"'
    
#     # Replace the wind setting using regex
#     modified_xml = re.sub(r'wind="[^"]*"', new_wind_str, xml_content)
    
#     # Save the updated XML
#     with open(xml_path, "w") as file:
#         file.write(modified_xml)
def update_wind_in_xml(xml_path: Path, new_wind: tuple):
    """
    Updates or adds the wind setting in a MuJoCo XML file.

    Parameters:
    - xml_path (Path): Path to the XML file.
    - new_wind (tuple): New wind values as (x, y, z).
      If any value in the tuple is None, reads the current wind values from the file and prints them.
    """
    with open(xml_path, "r") as file:
        xml_content = file.read()

    # Regex to match the wind attribute
    wind_pattern = r'wind="([^"]+)"'

    # Search for the wind attribute in the file
    match = re.search(wind_pattern, xml_content)
    if match:
        current_wind = match.group(1)
        if None in new_wind:
            print(f"Current wind values in the XML: {current_wind}")
            return
    else:
        if None in new_wind:
            print("No wind attribute found in the XML.")
            return

    # Create a string for the new wind values
    new_wind_str = f'wind="{new_wind[0]} {new_wind[1]} {new_wind[2]}"'

    # Replace or add the wind attribute in the XML content
    if match:
        # Replace existing wind attribute
        modified_xml = re.sub(wind_pattern, new_wind_str, xml_content)
    else:
        # Add the wind attribute to the root element if it doesn't exist
        modified_xml = re.sub(
            r'(<mujoco[^>]*>)', r'\1\n  ' + new_wind_str, xml_content, count=1
        )

    # Save the updated XML
    with open(xml_path, "w") as file:
        file.write(modified_xml)

    print(f"Updated wind values to: {new_wind_str}")

def run_test_offline(save_folder: Path):
    """Runs the test_offline.py command with specified parameters."""
    test_video_folder = save_folder / "test_video"
    test_video_folder.mkdir(parents=True, exist_ok=True)
    
    command = [
        "python",
        "url_benchmark/test_offline.py",
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
        f"load_replay_buffer={parameters['config']['path_exorl_learn']}/datasets/walker/rnd/replay.pt",
        "replay_buffer_episodes=5000",
        f"save_path={save_folder}"
    ]
    print(command)
    # Run the command
    subprocess.run(command)

def run_tests_with_different_winds(xml_path: Path, winds: dict, base_result_folder: Path):
    # Step 1: Back up the original XML
    backup_path = backup_original_xml(xml_path)

    try:
        for condition, wind in winds.items():
            print(f"\nRunning test {condition} with wind: {wind}")

            # Step 2: Update wind in XML
            update_wind_in_xml(xml_path, wind)

            # Step 3: Create a results folder for this wind setting
            result_folder = base_result_folder / f"saved_training_{condition}"
            result_folder.mkdir(parents=True, exist_ok=True)

            # Step 4: Run the test_offline.py command and save results
            run_test_offline(result_folder)

        print("\nAll tests completed.")
        
    finally:
        # Step 5: Restore the original XML after all tests
        restore_original_xml(xml_path, backup_path)
        print("Original XML file has been restored.")

# Example usage:
xml_path = Path(f"{parameters['config']['path']}/hilp_zsrl/url_benchmark/custom_dmc_tasks/walker.xml")
base_result_folder = Path(f"{parameters['config']['path']}/hilp_zsrl/exp_local/EXP/saved_winds")
winds = {
    # "Origin": (None, None, None), # (5, 0, 0)
    # "LightWind": (5, 0, 0),
    # "15": (15.0, 0, 0),
    # "30": (30.0, 0, 0),
    # "50": (50.0, 0, 0),
    # "60": (60.0, 0, 0),
    # "70": (70.0, 0, 0),
    "0_5_0": (0.0, 5.0, 0),
    "0_0_5": (0.0, 0, 5.0),
    "0_100_0": (0.0, 100.0, 0),
    "0_0_100": (0.0, 0, 100.0),
    # "ReverseWind": (-5.0, 0, 0),
    # "StrongReverseWind": (-15.0, 0, 0),
    # "DiagonalWind": (6.0, 6.0, 0),
    # "VerticalWind": (0, 0, 100.0)
}

run_tests_with_different_winds(xml_path, winds, base_result_folder)
