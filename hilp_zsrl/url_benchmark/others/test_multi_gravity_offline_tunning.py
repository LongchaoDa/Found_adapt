import os
import re
import sys
import shutil
import subprocess
from pathlib import Path
import toml
import numpy as np

# Load parameters from the configuration TOML file.
with open("/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/parameters.toml", "r") as f:
    parameters = toml.load(f)

# Define file paths for the two files to be modified.
sf_py_path = Path(f"{parameters['config']['path']}/hilp_zsrl/url_benchmark/agent/sf.py")
test_multi_gravity_path = Path(f"{parameters['config']['path']}/hilp_zsrl/url_benchmark/test_multi_gravity_offline.py")

# Define the grid search values.
lam_list = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # You can expand this list if needed
reg_epsilon_list = np.linspace(1e-5, 0.1, 11).tolist()  # 11 values from 1e-5 to 0.1

def modify_file(file_path: Path, patterns_replacements: dict):
    """
    Modify file content by replacing all occurrences that match the regular expression.
    """
    with file_path.open("r") as f:
        content = f.read()
    for pattern, replacement in patterns_replacements.items():
        content = re.sub(pattern, replacement, content)
    with file_path.open("w") as f:
        f.write(content)

def update_sf_file(new_reg_epsilon: float, new_lam: float):
    """
    Update sf.py with new reg_epsilon and lam values.
    Assumes the file contains lines like:
        reg_epsilon = 1e-4 # 1e-5, 0.1
        lam: float = 5.0  # Hyperparameter to balance target loss (tunable): 0.1 - 10
    """
    sf_patterns = {
        r"reg_epsilon\s*=\s*[\d\.\-e]+": f"reg_epsilon = {new_reg_epsilon}",
        r"lam:\s*float\s*=\s*[\d\.\-e]+": f"lam: float = {new_lam}"
    }
    modify_file(sf_py_path, sf_patterns)
    print(f"Updated sf.py: reg_epsilon={new_reg_epsilon}, lam={new_lam}")

def update_test_script_file(new_reg_epsilon: float, new_lam: float):
    """
    Update test_multi_gravity_offline.py with a new base_result_folder.
    Assumes the file contains a line similar to:
         base_result_folder = Path(f"{parameters['config']['path']}/hilp_zsrl/exp_local/dlc_25/try_considerNextobs")
    which we will replace with a folder name that incorporates the current reg_epsilon and lam.
    """
    new_folder_str = f"try_{new_reg_epsilon}_{new_lam}"
    pattern = r'base_result_folder\s*=\s*Path\(f"\{parameters\[\s*\'config\'\]\[\'path\'\]\}/hilp_zsrl/exp_local/dlc_25/try_considerNextobs"\)'
    replacement = f'base_result_folder = Path(f"{{parameters[\'config\'][\'path\']}}/hilp_zsrl/exp_local/dlc_25_grid/{new_folder_str}")'
    modify_file(test_multi_gravity_path, {pattern: replacement})
    print(f"Updated test_multi_gravity_offline.py: base_result_folder set to .../dlc_25_grid/{new_folder_str}")

def run_test_script(result_folder: Path):
    """
    Run the test_multi_gravity_offline.py script using subprocess.
    """
    command = [
        "python",
        str(test_multi_gravity_path),
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
        f"save_path={result_folder}"
    ]
    print("Executing command:", command)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result

# Prepare a log file for grid search progress.
log_file = Path("/home/local/ASURITE/longchao/Desktop/project/sim2realFoun/Sim2Real/Sim2RealFoundationPolicy/hilp_zsrl/a_analysis_25/adapter/log.txt")
with log_file.open("w") as lf:
    lf.write("Grid Search Log\n")
    lf.write("reg_epsilon, lam, return_code, stdout, stderr\n")

# Begin grid search.
for new_reg_epsilon in reg_epsilon_list:
    for new_lam in lam_list:
        # Update the two Python files with new parameter values.
        update_sf_file(new_reg_epsilon, new_lam)
        update_test_script_file(new_reg_epsilon, new_lam)

        # Set base result folder for this configuration.
        result_folder = Path(f"{parameters['config']['path']}/hilp_zsrl/exp_local/dlc_25_grid/try_{new_reg_epsilon}_{new_lam}")
        result_folder.mkdir(parents=True, exist_ok=True)

        print(f"Running grid search for reg_epsilon={new_reg_epsilon}, lam={new_lam}")
        result = run_test_script(result_folder)

        # Log results.
        with log_file.open("a") as lf:
            lf.write(f"{new_reg_epsilon}, {new_lam}, {result.returncode}, {result.stdout}, {result.stderr}\n")
        
        print(f"Completed reg_epsilon={new_reg_epsilon}, lam={new_lam}\n\n")

print("Grid search complete.")
