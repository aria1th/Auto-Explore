import os
import subprocess
import time
import logging
import torch

# Global Variables
root_dir = "generated_outputs"
num_folders = 200 # Number of folders to create for smaller file count per folder
stop_file = "stop.txt"

# Configure Logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [Main] %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)

def prepare_folders(root_dir, num_folders=200):
    os.makedirs(root_dir, exist_ok=True)
    for i in range(num_folders):
        folder_path = os.path.join(root_dir, f"folder_{i+1}")
        os.makedirs(folder_path, exist_ok=True)
    logging.info(f"Prepared {num_folders} folders in '{root_dir}'.")

def main():
    process_start_time_to_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    prepare_folders(root_dir, num_folders)

    num_gpus = torch.cuda.device_count()
    processes = []

    for i in range(num_gpus):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(i)}
        cmd = ['python', 'worker.py', str(i), root_dir, stop_file, process_start_time_to_str]
        cmd = cmd
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        logging.info(f"Started worker {i} with PID {p.pid}")

    logging.info("All workers started. Press Ctrl+C to stop or create 'stop.txt' file.")

    try:
        while True:
            time.sleep(1)
            if os.path.exists(stop_file):
                logging.info("Main process: Stop file detected. Initiating shutdown.")
                break
    except KeyboardInterrupt:
        logging.info("Main process: KeyboardInterrupt detected. Initiating shutdown.")

    # Wait for all workers to finish
    for p in processes:
        p.wait()
        logging.info(f"Worker with PID {p.pid} has exited.")

    logging.info("All workers have exited. Main process terminating.")

if __name__ == '__main__':
    main()
