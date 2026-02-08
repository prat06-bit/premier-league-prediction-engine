import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

season_map = {
    "2015.csv": "2015/16",
    "2016.csv": "2016/17",
    "2017.csv": "2017/18",
    "2018.csv": "2018/19",
    "2019.csv": "2019/20",
    "2020.csv": "2020/21",
    "2021.csv": "2021/22",
    "2022.csv": "2022/23",
    "2023.csv": "2023/24",
    "2024.csv": "2024/25",
}

dfs = []

for file, season in season_map.items():
    df = pd.read_csv(DATA_DIR / file)
    df["Season"] = season
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

data["Date"] = pd.to_datetime(
    data["Date"],
    dayfirst=True,
    errors="coerce",
    format="mixed"
)
    
data = data.sort_values("Date").reset_index(drop=True)

data.to_csv(DATA_DIR / "merged_matches.csv", index=False)

print("Merged dataset created")
print("Total matches:", data.shape[0])
