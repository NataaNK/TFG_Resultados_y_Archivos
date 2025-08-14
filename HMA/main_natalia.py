import argparse
import yaml
import subprocess
import os
import sys

def update_yaml_file(input_dir, yaml_path):
    """Update the dataroot_lq field in the YAML file."""
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        # Update the dataroot_lq field
        if 'datasets' in data and 'test_1' in data['datasets']:
            data['datasets']['test_1']['dataroot_lq'] = input_dir
        else:
            raise KeyError("'datasets:test_1' structure not found in the YAML file.")

        # Save the updated YAML file
        with open(yaml_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        print(f"Updated 'dataroot_lq' to '{input_dir}' in {yaml_path}")
    except Exception as e:
        print(f"Error updating YAML file: {e}")
        raise

def execute_test_script():
    """Main function to handle argument parsing and script execution."""
    parser = argparse.ArgumentParser(description="Run HMA test with updated YAML input directory.")
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Path to the input directory to update in the YAML file."
    )
    args = parser.parse_args()

    # Paths
    yaml_path = "H:/MasOrange/CUDA/HMA-CUDA/options/test/NATALIA_x4.yml"
    test_script = "hma/test.py"

    # Update the YAML file
    update_yaml_file(args.input_dir, yaml_path)

    # Run the test script
    command = [sys.executable, test_script, "-opt", yaml_path]
    try:
        subprocess.run(command, check=True)
        print("HMA test script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing the test script: {e}")

if __name__ == "__main__":
    execute_test_script()
