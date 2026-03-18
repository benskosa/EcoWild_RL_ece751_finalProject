import gym
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gym import spaces
from joblib import load

from data_utils import normalize_feature


class Logger(object):
    def __init__(self, filename):
        import sys
        self.terminal = sys.__stdout__
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Define RL Environment
class WildfireEnv(gym.Env):
    
    def __init__(self, df, config, start_offset=0):
        super(WildfireEnv, self).__init__()
        self.df = df  # Store full dataset, select a sensor later
        self.config = config
        self.start_offset = start_offset

        self.dt_model = load("weather_fire_detection_model.pkl")
        self.sensor_selection_count = {sensor: 0 for sensor in df["Sensor"].unique()} 
        self.sensor_data = None  # Will be set in reset()
        self.current_sensor = None  # Track the current sensor
        self.current_step = 0
        self.last_image_timestamp = None
        self.last_sampling_time = None
        # Randomly choose one of the battery levels each episode
        battery_energy_dict = self.config["Initial_Battery_Levels"]
        battery_choice = np.random.choice(list(battery_energy_dict.keys()))
        initial_energy = battery_energy_dict[battery_choice]
        self.battery_energy = initial_energy  # Start with full battery capacity
        self.max_battery_energy = initial_energy  # Full battery capacity
        self.energy_budget = initial_energy - self.config["Energy_Constraints"]["reserved_energy"]
        self.previous_ml_result = 0  # No fire detected at start
        self.missed_fire = 0
        self.missed_fire_time = 0  # Initialize missed fire time tracking
        
        self.episode_counter = 0  # Track episode number
        
        self.step_rewards_list = []
        
        self.total_env_steps = 0  # Track global step count for logging
        self.step_reward_log = []  # Store (step, reward)
        self.step_sampling_log = []  # Store (step, sampling)

        self.cumulative_rewards_list = []
        self.average_rewards_list = []
        self.tensorboard_rewards_list = []
        self.avg_sampling_time_list = []
        self.detection_time_list = []
        
        
        self.reward = 0
        self.data_length = 0
        
        self.fire_start_time = None
        self.fire_detection_time = None
        self.battery_depletion_time = None

        # Store episode data for plotting
        self.episode_data = {
            "timestamps": [],
            "battery_levels": [],
            "energy_budgets": [],
            "missed_fire_times": [],
            "sampling_time": [],
            "harvested_energy": [],
            "consumed_energy": [],
            "temperature": [],
            "humidity": [],
            "HDWI_score": [],
            "wind_speed": [],
            "ml_result": [],
            "take_a_picture": [],
            "label": [],
            "reward": []
        }

        # Observation space (state)
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        # Action space (continuous: [1 min, 60 min])
        self.action_space = spaces.Box(low=np.array([self.config["TD3_params"]["min_sampling_time"]]), high=np.array([self.config["TD3_params"]["max_sampling_time"]]), dtype=np.float32)

    def reset(self, sensor_override=None):
        if sensor_override is None:
            print("Sensor override must be provided for sequential execution.")
            return None  
        self.current_sensor = sensor_override

        # Set sensor data dynamically
        self.sensor_data = self.df[self.df["Sensor"] == self.current_sensor].reset_index(drop=True)
        
        self.sensor_selection_count[self.current_sensor] += 1  # Update visit count
        
        self.last_sampling_time = 0
        
        # Randomly choose one of the battery levels each episode
        battery_energy_dict = self.config["Initial_Battery_Levels"]
        battery_choice = np.random.choice(list(battery_energy_dict.keys()))
        initial_energy = battery_energy_dict[battery_choice]

        self.battery_energy = initial_energy  # Reset to full capacity
        self.max_battery_energy = initial_energy  # Full battery capacity
        self.energy_budget = initial_energy - self.config["Energy_Constraints"]["reserved_energy"]
        self.previous_ml_result = 0
        self.missed_fire = 0
        self.missed_fire_time = 0 # Reset missed fire time
        
        self.step_rewards_list = []
        
        self.reward = 0
        
        # Initialize fire start time as the first Label == 1 timestamp
        fire_rows = self.sensor_data[self.sensor_data["Label"] == 1]
        self.fire_start_time = fire_rows["Timestamp"].iloc[0] if not fire_rows.empty else None
        
        if not fire_rows.empty:
            fire_start_time = fire_rows["Timestamp"].iloc[0]
            allowed_indices = self.sensor_data[self.sensor_data["Timestamp"] <= (fire_start_time - pd.Timedelta(days=7))].index
        else:
            # No fire at all → just pick from the whole dataset
            allowed_indices = self.sensor_data.index

        # Find the start timestamp index + offset
        if len(self.sensor_data) <= self.start_offset:
            self.current_step = 0
        else:
            self.current_step = 0 # self.start_offset
        
        self.last_image_timestamp = self.sensor_data["Timestamp"].iloc[self.current_step]
          
        self.data_length = max(1, len(self.sensor_data) - (self.current_step + 1))
        self.fire_detection_time = None
        self.battery_depletion_time = None
        
        # Reset episode data
        self.episode_data = {
            "timestamps": [],
            "battery_levels": [],
            "energy_budgets": [],
            "missed_fire_times": [],
            "sampling_time": [],
            "harvested_energy": [],
            "consumed_energy": [],
            "temperature": [],
            "humidity": [],
            "HDWI_score": [],
            "wind_speed": [],
            "ml_result": [],
            "take_a_picture": [],
            "label": [],
            "reward": []
        }
        
        return self.get_state()

    def get_state(self):
        row = self.sensor_data.iloc[self.current_step]
        time_since_last_image = (row["Timestamp"] - self.last_image_timestamp).total_seconds() / 60

        return np.array([
            row["Temperature_2m_normalized"],
            row["Relative_Humidity_2m_normalized"],
            row["Wind_Speed_10m"],
            # row["Rain"],
            row["HDWI"],
            row["solar_energy"],
            normalize_feature(self.last_sampling_time, 1, self.config["TD3_params"]["max_sampling_time"]),
            # normalize_feature(self.battery_energy, 0, self.max_battery_energy),
            normalize_feature(self.energy_budget, 0, self.max_battery_energy - self.config["Energy_Constraints"]["reserved_energy"]),
            normalize_feature(time_since_last_image, 0, 120),
            row["Time_of_Day"],
            row["Season"],  # Encoded as a categorical variable
            self.previous_ml_result,  # Fire detected previously (or not)
        ], dtype=np.float32)

    def step(self, action):
        self.last_sampling_time = int(action) # Weather Sensor Read Interval

        # Find the next timestamp based on the RL decision
        next_timestamp = self.sensor_data.iloc[self.current_step]["Timestamp"] + pd.Timedelta(minutes=self.last_sampling_time)
        
        next_steps = self.sensor_data[self.sensor_data["Timestamp"] >= next_timestamp].index
        
        # Get all rows within the skipped time range
        skipped_data = self.sensor_data[
            (self.sensor_data["Timestamp"] > self.sensor_data.iloc[self.current_step]["Timestamp"]) &
            (self.sensor_data["Timestamp"] < next_timestamp)
        ]

        if not next_steps.empty:
            self.current_step = next_steps[0]
        else:
            self.current_step = len(self.sensor_data) - 1  # Stop at last timestamp

        row = self.sensor_data.iloc[self.current_step]
        
        features = {
            "avgtempC": row["Temperature_2m"],
            "humid": row["Relative_Humidity_2m"]
        }
        df_features = pd.DataFrame([features])
        take_picture = int(self.dt_model.predict(df_features)[0])
        
        # Combine skipped rows and the RL-decided row
        fire_rows = pd.concat([skipped_data, self.sensor_data.iloc[[self.current_step]]])

        """ # Check if fire exists and update fire_start_time if not set
        if self.fire_start_time is None and (fire_rows["Label"] == 1).any():
            self.fire_start_time = fire_rows.loc[fire_rows["Label"] == 1, "Timestamp"].iloc[0] """

        ml_result = 0

        if take_picture:
            ml_result = np.random.choice([1, 0], p=[self.config["ML_Performance"]["TP_rate"], 1 - self.config["ML_Performance"]["TP_rate"]]) if row["Label"] == 1 else \
                        np.random.choice([1, 0], p=[self.config["ML_Performance"]["FP_rate"], 1 - self.config["ML_Performance"]["FP_rate"]])

            if row["Label"] == 1 and ml_result == 1:
                self.fire_detection_time = row["Timestamp"] # Fire detection tracking
                    
            self.last_image_timestamp = row["Timestamp"]
        
        neighbor_comm_energy = self.config["Neighborhood_Communication"]["num_neighbors"] * self.config["Neighborhood_Communication"]["E_comm_neighbor"]

        # Check for missed fires safely
        if not skipped_data.empty and "Label" in skipped_data.columns:
            self.missed_fire = any((skipped_data["Label"] == 1) &
                            ((row["Timestamp"] - skipped_data["Timestamp"]).dt.total_seconds() / 60 > self.config["max_missing_fire_min"]))

        if row["Label"] == 1:
            self.missed_fire_time = (row["Timestamp"] - self.fire_start_time).total_seconds() / 60

        # Sum up harvested energy for every minute in the skipped time
        total_harvested_energy = (fire_rows["solar_energy"] * self.config["harvested_energy_loss"]).sum() if not fire_rows.empty else 0
        
        time_skipped_hours = self.last_sampling_time / 60  # Convert to hours
        standby_power_used = (
            self.config["Standby_Power_Components"]["P_temp_humidity_standby"] + self.config["Standby_Power_Components"]["P_anemometer_standby"] +
            self.config["Standby_Power_Components"]["P_camera_standby"] + self.config["Standby_Power_Components"]["P_comm_standby"]
        ) * time_skipped_hours  # Multiply by time skipped

        # Energy management with standby power
        energy_used = (
            self.config["Energy_Constraints"]["E_proc_rl"] + self.config["Energy_Constraints"]["E_temp_humidity_sensor"] + self.config["Energy_Constraints"]["E_anemometer_sensor"] + (self.config["Energy_Constraints"]["E_proc_ml"] + self.config["Energy_Constraints"]["E_camera_host"] if take_picture else 0) +
            standby_power_used + (self.config["Energy_Constraints"]["E_comm"] + neighbor_comm_energy if ml_result and take_picture else 0) 
        )
        
        # Simulate minute-by-minute depletion during skipped time
        if self.battery_depletion_time is None:
            current_batt = self.battery_energy
            max_batt_energy = self.max_battery_energy
            E_used = (
                self.config["Energy_Constraints"]["E_proc_rl"] + self.config["Energy_Constraints"]["E_temp_humidity_sensor"] + self.config["Energy_Constraints"]["E_anemometer_sensor"] + (self.config["Energy_Constraints"]["E_proc_ml"] + self.config["Energy_Constraints"]["E_camera_host"] if take_picture else 0) +
                (self.config["Energy_Constraints"]["E_comm"] + neighbor_comm_energy if ml_result and take_picture else 0)
            )
            for i, ts in enumerate(fire_rows["Timestamp"]):
                harvested = fire_rows.iloc[i]["solar_energy"] * self.config["harvested_energy_loss"]
                standby_used = standby_power_used / len(fire_rows) # Distribute standby cost equally
                net_energy = harvested - standby_used 
                # No sensing/ML/comm costs since device is idle, just standby and harvested

                current_batt += net_energy
                current_batt *= (1 - self.config["E_battery_leakage_percentage"])
                max_batt_energy *= (1 - self.config["E_battery_leakage_percentage"])
                current_batt = max(0, min(max_batt_energy, current_batt))

                if current_batt - self.config["Energy_Constraints"]["reserved_energy"] <= 0:
                    self.battery_depletion_time = ts
                    break  # First time battery hits 0
                
            if  current_batt - self.config["Energy_Constraints"]["reserved_energy"] > 0 and current_batt - E_used - self.config["Energy_Constraints"]["reserved_energy"] <= 0:
                self.battery_depletion_time = row["Timestamp"]
            
        # Update battery energy first
        self.battery_energy += total_harvested_energy - energy_used
        self.battery_energy *= (1 - self.config["E_battery_leakage_percentage"])  # Apply battery leakage loss
        self.max_battery_energy *= (1 - self.config["E_battery_leakage_percentage"])  # Apply battery leakage loss
        self.battery_energy = max(0, min(self.max_battery_energy, self.battery_energy))  # Ensure valid range

        # Update energy budget after battery energy is updated
        self.energy_budget = max(0, self.battery_energy - self.config["Energy_Constraints"]["reserved_energy"])  

        self.previous_ml_result = ml_result
        
        # Default: Episode continues
        done = self.current_step >= len(self.sensor_data) - 1  # Stops when dataset is finished

        # Stop the episode if fire is detected, missed, or energy is 0
        # if (row["Label"] == 1 and ml_result == 1) or self.missed_fire_time >= 15 or self.energy_budget <= 0:
        if (row["Label"] == 1 and ml_result == 1) or self.energy_budget <= 0 or self.battery_depletion_time is not None:
            print(f'Next sensor time: {self.last_sampling_time}, Stopping Condition Met: Fire Detected: {row["Label"] == 1 and ml_result == 1}, Missed Fire for {self.missed_fire_time} min: {self.missed_fire_time >= 30}, Energy Depleted: {self.energy_budget <= 0}, Battery Depletion Time: {self.battery_depletion_time}')
            self.detection_time_list.append(self.missed_fire_time)
            done = True # Stops the episode
            
        # Store episode data for plotting
        self.episode_data["timestamps"].append(row["Timestamp"])
        self.episode_data["battery_levels"].append(self.battery_energy)
        self.episode_data["energy_budgets"].append(self.energy_budget)
        self.episode_data["missed_fire_times"].append(self.missed_fire_time)
        self.episode_data["sampling_time"].append(self.last_sampling_time)
        self.episode_data["harvested_energy"].append(row["solar_energy"])
        self.episode_data["consumed_energy"].append(energy_used)
        self.episode_data["temperature"].append(row["Temperature_2m_normalized"])
        self.episode_data["humidity"].append(row["Relative_Humidity_2m_normalized"])
        self.episode_data["HDWI_score"].append(row["HDWI"])
        self.episode_data["wind_speed"].append(row["Wind_Speed_10m"])
        self.episode_data["ml_result"].append(ml_result)
        self.episode_data["take_a_picture"].append(take_picture)
        self.episode_data["label"].append(row["Label"])

        """ if self.battery_energy > 5:
            self.reward = - k1 * self.last_sampling_time
        else:
            self.reward = k1 * self.last_sampling_time """
        
        self.reward  = self.config["Reward_Params"]["beta"] * self.last_sampling_time + (1 - self.config["Reward_Params"]["beta"]) * self.reward
        
        step_reward = self.last_sampling_time * (1 - 2 * take_picture) / self.data_length
        
        self.episode_data["reward"].append(step_reward)
        
        self.step_sampling_log.append((self.total_env_steps, self.last_sampling_time))
        

        # If episode ends, generate plot
        if done:
            self.episode_counter += 1  # Increment episode counter
            avg_sampling_time = np.mean(self.episode_data["sampling_time"]) 
            final_reward, reason = self.calculate_final_reward() # -k1 * self.reward, "xx" #self.calculate_final_reward()
            print("final_reward", final_reward)
            self.step_rewards_list.append(final_reward)  # Store final reward globally
            
            self.step_reward_log.append((self.total_env_steps, final_reward))
            
            self.total_env_steps += 1
            
            self.average_rewards_list.append(sum(self.step_rewards_list) / len(self.step_rewards_list))
            self.cumulative_rewards_list.append(sum(self.step_rewards_list))
            self.tensorboard_rewards_list.append(sum(self.cumulative_rewards_list) / len(self.cumulative_rewards_list))
        
            self.avg_sampling_time_list.append(avg_sampling_time)
            print("avg_sampling_time", avg_sampling_time)

            self.plot_episode_metrics(reason, final_reward)
            return self.get_state(), final_reward, done, {}
        
        self.step_rewards_list.append(step_reward)  # Store step reward globally
        self.step_reward_log.append((self.total_env_steps, step_reward))
        self.total_env_steps += 1

        return self.get_state(), step_reward, done, {}
    
    def calculate_final_reward(self):
        """Calculate the reward at the end of the episode."""
        if self.battery_depletion_time is not None and (not self.fire_start_time or self.battery_depletion_time < self.fire_start_time):
            # Case 1: Battery depletes before fire
            t_deplete_minus_t_start = (self.battery_depletion_time - self.sensor_data["Timestamp"].iloc[0]).total_seconds() / 60
            reward = -self.config["Reward_Params"]["alpha_B"] * (1 / (t_deplete_minus_t_start)) - self.config["Reward_Params"]["R_min"]
            reason = "case1"
            print('reward case1 ', reward, "t_deplete_minus_t_start", t_deplete_minus_t_start)
        else:
            reward = -self.config["Reward_Params"]["k1"] * self.reward  
            
            reason = "case2/3"
            print('reward case2/3 ', reward)

        return reward, reason
    
    def plot_episode_metrics(self, reason, final_reward):
        """ if self.episode_counter % 10 != 0:
            return  # Skip saving unless episode number is a multiple of 100 """
        """Generates and saves episode-specific plots inside `episode_plots/`."""
        file_name = self.config["file_name"]
        folder = f"Inference/episode_plots_step_reward{file_name}"
        os.makedirs(folder, exist_ok=True)
        
        # Save the loaded config into the folder
        with open(os.path.join(folder, "config_setup.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        
        df2 = pd.DataFrame(self.episode_data)

        # Save it to a CSV file
        df2.to_csv(f"{folder}/episode_{self.episode_counter}_{self.current_sensor}.csv", index=False)  # index=False avoids adding an extra index column

        fig, axs = plt.subplots(7, 2, figsize=(15, 20), sharex=True)
        axs = axs.flatten()

        axs[0].scatter(self.episode_data["timestamps"], self.episode_data["harvested_energy"], label="Harvested Energy (Wh)", color='green')
        axs[1].scatter(self.episode_data["timestamps"], self.episode_data["consumed_energy"], label="Consumed Energy (Wh)", color='green')
        axs[2].scatter(self.episode_data["timestamps"], self.episode_data["energy_budgets"], label="Energy Budget (Wh)", color='green')
        axs[3].scatter(self.episode_data["timestamps"], self.episode_data["battery_levels"], label="Battery Energy (Wh)", color='green')
        
        
        axs[4].scatter(self.episode_data["timestamps"], self.episode_data["temperature"], label="Temperature (C)", color='green')
        axs[5].scatter(self.episode_data["timestamps"], self.episode_data["humidity"], label="Humidity (%)", color='green')
        axs[6].scatter(self.episode_data["timestamps"], self.episode_data["HDWI_score"], label="HDWI score", color='green')
        axs[7].scatter(self.episode_data["timestamps"], self.episode_data["wind_speed"], label="Wind Speed (km/h)", color='green')
        
        axs[8].scatter(self.episode_data["timestamps"], self.episode_data["sampling_time"], label="Sampling Time (min)", color='green')
        axs[9].scatter(self.episode_data["timestamps"], self.episode_data["take_a_picture"], label="Take a picture [0, 1]", color='green')
        axs[10].scatter(self.episode_data["timestamps"], self.episode_data["ml_result"], label="ML result [0, 1]", color='green')
        
        axs[11].scatter(self.episode_data["timestamps"], self.episode_data["missed_fire_times"], label="Missed Fire Time (min)", color='green')
        
        axs[12].scatter(self.episode_data["timestamps"], self.episode_data["reward"], label=f"Step Reward {reason} {final_reward}", color='green')
        axs[-1].set_xlabel("Timestamp (min)", fontsize=16, fontweight='bold')
        
        if self.fire_start_time is not None:
            for ax in axs:
                ax.axvline(x=self.fire_start_time, color='red', linestyle='--', label='First Fire Detected')

        for ax in axs:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y\n%I:%M %p'))  # Format timestamps
            ax.tick_params(axis='x', labelsize=12, width=2, rotation=45)  # Rotate & set label size
            ax.tick_params(axis='y', labelsize=12, width=2)
            ax.legend()
            ax.grid(True)

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{folder}/episode_{self.episode_counter}_{self.current_sensor}.png", dpi=300)
        plt.close()

        print(f"Episode {self.episode_counter} plot saved.")
