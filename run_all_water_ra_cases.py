import os
import glob
import subprocess
import sys
from datetime import datetime

def run_command(command, log_file):
    """Executes a command and logs its output."""
    try:
        # Using shell=True for simpler command execution on Windows, but be aware of security implications.
        # For this specific use case with controlled inputs, it's acceptable.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        with open(log_file, 'w') as f:
            for line in process.stdout:
                sys.stdout.write(line) # Print to console in real-time
                f.write(line)
        process.wait()
        return process.returncode
    except Exception as e:
        print(f"An error occurred while running command: {command}")
        print(f"Error: {e}")
        with open(log_file, 'a') as f:
            f.write(f"\n\n--- SCRIPT EXECUTION FAILED ---\n{e}\n")
        return 1

def main():
    """Main function to run the training and visualization pipeline."""
    print("======================================================================")
    print("== Starting Training Pipeline for all 30 Water-Ra cases (Python Runner)")
    print("======================================================================")

    # --- Setup Directories ---
    main_results_dir = "water_ra_training_all_results"
    if not os.path.exists(main_results_dir):
        os.makedirs(main_results_dir)
        print(f"All results will be saved in: {main_results_dir}")
    else:
        print(f"Results directory '{main_results_dir}' already exists. Files may be overwritten.")
    
    failed_log_path = os.path.join(main_results_dir, "failed_trainings.txt")
    data_dir = os.path.join("data", "Water", "Water_Ra", "Ra")

    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found at '{data_dir}'. Exiting.")
        return

    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    total_files = len(mat_files)
    
    if total_files == 0:
        print(f"ERROR: No .mat files found in '{data_dir}'. Exiting.")
        return

    # --- Main Loop ---
    failed_cases = []
    for i, mat_file_path in enumerate(mat_files):
        mat_file_name = os.path.splitext(os.path.basename(mat_file_path))[0]
        
        print("\n" + "="*70)
        print(f"== Processing Case {i+1}/{total_files}: {mat_file_name}")
        print("="*70)

        result_subdir = os.path.join(main_results_dir, mat_file_name)
        os.makedirs(result_subdir, exist_ok=True)
        
        current_time = lambda: datetime.now().strftime("%H:%M:%S")

        # --- Step 1: Train the model ---
        print(f"[{current_time()}] Starting training for {mat_file_name}...")
        train_log = os.path.join(result_subdir, "training.log")
        train_command = (
            f"python train_complete_physics.py "
            f"--data_dir \"{data_dir}\" "
            f"--data_file \"{mat_file_path}\" "
            f"--save_dir \"{result_subdir}\" "
            f"--epochs 500 "
            f"--batch_size 32 "
            f"--learning_rate 0.001"
        )
        
        return_code = run_command(train_command, train_log)
        
        if return_code != 0:
            print(f"ERROR: Training failed for {mat_file_name}. See log for details.")
            failed_cases.append(f"{mat_file_name}\n")
            continue # Skip to the next file

        print(f"[{current_time()}] Training finished.")

        checkpoint_path = os.path.join(result_subdir, "complete_physics_model_checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint file not found for {mat_file_name}. Skipping visualizations.")
            failed_cases.append(f"{mat_file_name} (checkpoint_missing)\n")
            continue

        # --- Step 2: Generate Training Plots ---
        print(f"[{current_time()}] Generating training plots...")
        plot_log = os.path.join(result_subdir, "plotting.log")
        plot_command = (
            f"python create_training_plots.py "
            f"--checkpoint_path \"{checkpoint_path}\" "
            f"--save_dir \"{result_subdir}\""
        )
        run_command(plot_command, plot_log)
        print(f"[{current_time()}] Plotting finished.")
        
        # --- Step 3: Generate Field Animation ---
        print(f"[{current_time()}] Generating field animation...")
        anim_log = os.path.join(result_subdir, "animation.log")
        anim_command = (            f"python create_complete_animation.py "            f"--checkpoint_path \"{checkpoint_path}\" "            f"--data_path \"{mat_file_path}\" "            f"--output_dir \"{os.path.join(result_subdir, 'animations')}\" "        )
        run_command(anim_command, anim_log)
        print(f"[{current_time()}] Field animation finished.")

        # --- Step 4: Generate Streamline/Isotherm Plots & Animation ---
        print(f"[{current_time()}] Generating streamline and isotherm plots...")
        flow_log = os.path.join(result_subdir, "flow_viz.log")
        flow_command = (
            f"python create_streamline_isotherm_plots.py "
            f"--checkpoint_path \"{checkpoint_path}\" "
            f"--data_path \"{mat_file_path}\" "
            f"--output_dir \"{os.path.join(result_subdir, 'flow_visuals')}\" "
            f"--create_animation"
        )
        run_command(flow_command, flow_log)
        print(f"[{current_time()}] Flow visualization finished.")

        # --- Step 5: Generate Error Analysis Plots ---
        print(f"[{current_time()}] Generating error analysis plots...")
        error_plot_log = os.path.join(result_subdir, "error_plotting.log")
        error_plot_command = (
            f"python create_error_analysis_plots.py "
            f"--checkpoint_path \"{checkpoint_path}\" "
            f"--data_path \"{mat_file_path}\" "
            f"--output_dir \"{result_subdir}\""
        )
        return_code = run_command(error_plot_command, error_plot_log)
        if return_code != 0:
            print(f"ERROR: Error analysis plot generation failed for {mat_file_name}. See log for details.")
            failed_cases.append(f"{mat_file_name} (Error Plotting)\n")
            # Continue to the next file even if this step fails
            
    # --- Final Summary ---
    print("\n" + "="*70)
    print("== Pipeline Finished!")
    print("="*70)
    if failed_cases:
        with open(failed_log_path, "a") as f:
            f.writelines(failed_cases)
        print(f"WARNING: {len(failed_cases)} case(s) failed. Check '{failed_log_path}' for a list.")
    else:
        print("SUCCESS: All cases processed successfully.")
    print(f"Results are in: {main_results_dir}")


if __name__ == "__main__":
    main()
