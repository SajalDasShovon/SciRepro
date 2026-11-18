"""
Exercise vs rest analysis.

For every answered EMA, average the wrist activity (ENMO, mg) in a
+/-15 minute window around the EMA time. Average per participant and
condition, then compare exercise with rest (sitting or lying down) using
a paired t-test across participants.
"""
import pandas as pd
from scipy import stats

WINDOW = pd.Timedelta(minutes=15)
CONDITION = {"exercise": "exercise", "sitting": "rest", "lying down": "rest"}


def load_data():
    sensor = pd.read_csv("sensor.csv", parse_dates=["timestamp"])
    ema = pd.read_csv("ema.csv", parse_dates=["timestamp"])
    ema = ema.dropna(subset=["activity"])  # missed prompts
    return sensor, ema


def activity_around(sensor, ema):
    """Mean ENMO in the window around each EMA."""
    enmo = {
        pid: group.set_index("timestamp")["enmo_mg"].sort_index()
        for pid, group in sensor.groupby("participant_id")
    }
    means = []
    for row in ema.itertuples():
        series = enmo[row.participant_id]
        means.append(series.loc[row.timestamp - WINDOW:row.timestamp + WINDOW].mean())
    return ema.assign(window_enmo=means)


def run():
    sensor, ema = load_data()
    ema = activity_around(sensor, ema)
    ema["condition"] = ema["activity"].map(CONDITION)  # other answers are dropped
    ema = ema.dropna(subset=["condition", "window_enmo"])

    per_person = ema.pivot_table(
        index="participant_id", columns="condition",
        values="window_enmo", aggfunc="mean",
    ).dropna()
    t, p = stats.ttest_rel(per_person["exercise"], per_person["rest"])
    effect = (per_person["exercise"] - per_person["rest"]).mean()

    print(f"participants={len(per_person)}")
    print(f"exercise_mean_mg={per_person['exercise'].mean():.1f}")
    print(f"rest_mean_mg={per_person['rest'].mean():.1f}")
    print(f"effect_mg={effect:.1f}")
    print(f"paired_t={t:.2f} p={p:.3g}")


if __name__ == "__main__":
    run()
