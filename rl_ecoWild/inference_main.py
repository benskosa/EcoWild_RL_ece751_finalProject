import numpy as np
import json
from datetime import datetime

from stable_baselines3 import TD3

import torch

from multiprocessing import Pool
import os

import sys

from data_utils import prepare_dataset
from wildfire_env import WildfireEnv, Logger

# Set deterministic CPU behavior
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# reduces nondeterminism from GPU kernel choices
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run_inference_with_sensor_id(sensor_id, sensor_df, start_offset, config):

    file_name = config["file_name"]

    # Recreate environment and logger
    env = WildfireEnv(sensor_df, config, start_offset=start_offset)
    model = TD3.load(
        "wildfire_td3_20250622_041653_RL1to30min_beta0p9_0.90036452TP_0.58399005FP_noOffset_7daysReservedEng_50perLoss_37571840.zip",
        env=env,
        custom_objects={
            "action_noise": None,
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        }
    )
    obs = env.reset(sensor_override=sensor_id)

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    np.save(f"Inference/episode_plots_step_reward{file_name}/rewards_{sensor_id}.npy", env.cumulative_rewards_list)
    np.save(f"Inference/episode_plots_step_reward{file_name}/average_rewards_{sensor_id}.npy", env.average_rewards_list)
    np.save(f"Inference/episode_plots_step_reward{file_name}/tensorboard_{sensor_id}.npy", env.tensorboard_rewards_list)
    np.save(f"Inference/episode_plots_step_reward{file_name}/avg_sampling_time_{sensor_id}.npy", env.avg_sampling_time_list)
    np.save(f"Inference/episode_plots_step_reward{file_name}/detection_time_{sensor_id}.npy", env.detection_time_list)
    
    step_log_array = np.array(sorted(env.step_reward_log, key=lambda x: x[0]))
    step_sampling_log_array = np.array(sorted(env.step_sampling_log, key=lambda x: x[0]))
    np.save(f"Inference/episode_plots_step_reward{file_name}/step_rewards_{sensor_id}.npy", step_log_array)
    np.save(f"Inference/episode_plots_step_reward{file_name}/step_sampling_time_{sensor_id}.npy", step_sampling_log_array)

    print(f"Inference finished for sensor {sensor_id}")


if len(sys.argv) < 2:
    print("Usage: python run.py <config_file.json>")
    sys.exit(1)

config_path = sys.argv[1]
# Load configuration from JSON file
with open(config_path, "r") as json_file:
    config = json.load(json_file)


max_offset_per_sensor = config["max_offset_per_sensor"]
batch_size = config["parallel_batch_size"]


df = prepare_dataset(config)

if __name__ == '__main__':
    
    file_name = config["file_name"]
    folder = f"Inference/episode_plots_step_reward{file_name}"
    os.makedirs(folder, exist_ok=True)

    log_file_path = f"{folder}/wildfire_rl_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_name}.txt"
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout
    
    # All sensors → each one becomes a parallel task
    all_sensors = sorted(df["Sensor"].unique())
    
    start_offset_per_sensor = {
        sensor: np.random.randint(0, max_offset_per_sensor - 1) for sensor in all_sensors
    }
    
    def run_batch(sensor_list):
        args = [
            (sensor, df[df["Sensor"] == sensor].copy(), start_offset_per_sensor[sensor], config)
            for sensor in sensor_list
        ]

        with Pool(processes=len(sensor_list)) as pool:
            pool.starmap(run_inference_with_sensor_id, args)

    # === Batch all sensors in chunks of ... ===
    for i in range(0, len(all_sensors), batch_size):
        current_batch = all_sensors[i:i + batch_size]
        print(f"\nRunning batch {i // batch_size + 1} with {len(current_batch)} sensors...")
        run_batch(current_batch)
        print(f"Batch {i // batch_size + 1} complete.")

