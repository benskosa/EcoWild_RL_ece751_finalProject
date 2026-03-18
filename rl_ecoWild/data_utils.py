import pandas as pd
import numpy as np

def filter_last_month(group, episode_length_days):
    last_ts = group["Timestamp"].max()
    start_ts = last_ts - pd.Timedelta(days=episode_length_days)
    return group[group["Timestamp"] >= start_ts]

# Interpolate weather data to minutely frequency for each sensor
def interpolate_group(group):
    # Sort by timestamp and ensure duplicates are handled
    group = group.sort_values(by="Label", ascending=False).drop_duplicates(
        subset="Timestamp", keep="first"
    )

    # Ensure data is sorted properly
    group = group.sort_values(by="Timestamp")
    
    # Generate a full minutely timestamp range
    minutely_index = pd.date_range(
        start=group["Timestamp"].min(),
        end=group["Timestamp"].max()
        + pd.Timedelta(hours=1)
        - pd.Timedelta(minutes=1),
        freq="1T",
    )

    # Merge with existing data to maintain 1-minute intervals
    group = group.set_index("Timestamp").reindex(minutely_index)

    # Interpolate numeric columns but preserve categorical labels carefully
    numeric_cols = [
        "Temperature_2m",
        "Relative_Humidity_2m",
        "Wind_Speed_10m",
        "HDWI",
        "Rain",
    ]

    group[numeric_cols] = group[numeric_cols].interpolate(method="linear")

    # Forward-fill the hourly values, then divide by 60 to distribute energy correctly
    group["solar_energy"] = group["solar_energy"].ffill() / 60

    # Forward-fill and backward-fill labels
    group["Label"] = group["Label"].ffill().bfill()
    group["Sensor"] = group["Sensor"].ffill().bfill()
    group["Location"] = group["Location"].ffill().bfill()

    # Reset index to bring Timestamp back as a column
    group = group.reset_index().rename(columns={"index": "Timestamp"})
    return group

# Function to get season from a month
def get_season(month):
    if 3 <= month <= 5:
        return 0.25  # Spring
    elif 6 <= month <= 8:
        return 0.50  # Summer
    elif 9 <= month <= 11:
        return 0.75  # Fall
    else:
        return 1.00  # Winter


def normalize_feature(x, min_val, max_val):
    return np.clip((x - min_val) / (max_val - min_val), 0, 1)

# Normalize data
def normalize(data):
    range_val = data.max() - data.min()
    return (data - data.min()) / range_val if range_val != 0 else data * 0


def prepare_dataset(config):
    """
    Loads and preprocesses dataset exactly as original inference script.
    No logic changes.
    """

    episode_length_days = config["episode_length_days"]
    
    # Load dataset
    df = pd.read_csv("grouped_weather_data_with_solar_energy.csv")
    df["Timestamp"] = (
        pd.to_datetime(df["Timestamp"], unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Etc/GMT+8")
    )
    
    # Apply interpolation to all sensor groups
    df = df.sort_values(by="Timestamp")
    df = df.groupby("Sensor").apply(interpolate_group).reset_index(drop=True)

    # Filter last 1 month only
    df = df.groupby("Sensor", group_keys=False).apply(
        lambda g: filter_last_month(g, episode_length_days)
    )

    duplicates_to_remove = [
        "bh-s-mobo-c1",
        "bh-w-mobo-c2",
        "bl-s-mobo-c1",
        "bm-w-mobo-c1",
        "hp-w-mobo-c1",
        "lp-n-mobo-c3",
        "mlo-s-mobo-c1",
        "om-w-mobo-c1",
        "pi-w-mobo-c1",
        "so-w-mobo-c1",
        "sp-w-mobo-c1",
        "lp-n-mobo1",
        "om-n-mobo-c1",
        "lp-s-mobo-c1",
        "bh-n-mobo-c2",
        "sdsc-e-mobo-c1",
        "hp-w-mobo-c2",
    ]

    # Remove those sensors
    df = df[~df["Sensor"].isin(duplicates_to_remove)].copy()

    df["Temperature_2m_normalized"] = normalize(df["Temperature_2m"])
    df["Relative_Humidity_2m_normalized"] = normalize(df["Relative_Humidity_2m"])
    df["Wind_Speed_10m"] = normalize(df["Wind_Speed_10m"])
    df["HDWI"] = normalize(df["HDWI"])
    df["Rain"] = normalize(df["Rain"])

    # Add season and time of day
    df["Season"] = df["Timestamp"].dt.month.apply(get_season)
    df["Time_of_Day"] = df["Timestamp"].dt.hour / 23

    return df