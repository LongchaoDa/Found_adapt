#Code to quickly test xml modifications before performing test



import re
from pathlib import Path

def create_multiple_temp_xmls_with_winds(xml_path: Path, winds: dict, output_dir: Path):
    """
    Creates multiple XML files with different wind settings.

    Parameters:
    - xml_path (Path): Path to the original XML file.
    - winds (dict): Dictionary of wind conditions with values as tuples (x, y, z).
    - output_dir (Path): Directory to save the modified XML files.

    Returns:
    - list: List of paths to the newly created XML files.
    """
    with open(xml_path, "r") as file:
        original_xml_content = file.read()

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    modified_files = []

    for condition, wind in winds.items():
        # Reset to the original XML content for each wind condition
        xml_content = original_xml_content

        # Create a string for the new wind values
        new_wind_str = f'wind="{wind[0]} {wind[1]} {wind[2]}"'

        # Replace the wind setting in the XML
        modified_xml = re.sub(r'wind="[^"]*"', new_wind_str, xml_content)

        # Save the modified XML to a new file
        output_file = output_dir / f"temp_{condition}.xml"
        with open(output_file, "w") as file:
            file.write(modified_xml)

        modified_files.append(output_file)

    return modified_files

# Example usage:
xml_path = Path("C:/Users/thiru/OneDrive/Desktop/DRAL/sim2real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/custom_dmc_tasks/walker.xml")
output_dir = Path("C:/Users/thiru/OneDrive/Desktop/DRAL/sim2real/Sim2RealFoundationPolicy/hilp_zsrl/url_benchmark/custom_dmc_tasks/temp_winds")
winds = {
    "NoWind": (0, 0, 0),
    "LightWind": (0.5, 0, 0),
    "ModerateWind": (1.0, 0, 0),
    "StrongWind": (3.0, 0, 0),
    "ReverseWind": (-1.0, 0, 0),
    "StrongReverseWind": (-3.0, 0, 0),
    "DiagonalWind": (1.0, 1.0, 0),
    "VerticalWind": (0, 0, 2.0)
}

modified_files = create_multiple_temp_xmls_with_winds(xml_path, winds, output_dir)
print("Modified XML files with different winds created:")
for file in modified_files:
    print(file)
