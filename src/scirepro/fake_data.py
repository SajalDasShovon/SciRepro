"""
fake_data.py - generate the data of the fake mHealth study used to test the agent.

    python fake_data.py     # (re)generate the data after changing this file

The fake study lives in fake_study/: pipeline.py and study.txt are written
by hand, and this file generates sensor.csv and ema.csv next to them. The
data are generated once and then reused by every run.

The study: 10 participants, 7 days, a wrist sensor (1-minute acceleration
and heart rate) and smartphone EMA surveys. The data behave like real
wearable data: sleep at night, sitting/standing/walking bouts, exercise
sessions, charging gaps, a dead battery, missed prompts, heart-rate
dropouts and occasional wrong answers.

SPOILER - the hidden flaw: the wearable exports timestamps in UTC, while the
EMA app logs local time (US Pacific, summer, UTC-7). Neither file says so,
so the sensor clock looks 7 hours ahead and the shipped pipeline compares
the wrong windows. The agent has to find this. Keep this file out of the
agent's reach.
"""

import csv
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

FAKE_STUDY_DIR = Path(__file__).resolve().parent / "fake_study"

DATA_SEED = 7
HIDDEN_OFFSET_HOURS = 7          # UTC minus US Pacific daylight time

START = datetime(2025, 7, 7)     # local midnight on the first study day
N_DAYS = 7
N_PARTICIPANTS = 10
DAY = 24 * 60                    # minutes

# Median wrist ENMO (milli-g) and heart-rate rise over resting (bpm).
BEHAVIOR = {
    "sleep": (3, -8),
    "sedentary": (12, 4),
    "light": (45, 14),
    "walking": (150, 28),
    "exercise": (300, 65),
}
# Waking behavior is a Markov chain: mean bout length (minutes) and what follows.
BOUT_MINUTES = {"sedentary": 35, "light": 8, "walking": 6}
NEXT_STATE = {
    "sedentary": (["light", "walking"], [0.7, 0.3]),
    "light": (["sedentary", "walking"], [0.7, 0.3]),
    "walking": (["sedentary", "light"], [0.6, 0.4]),
}
# How people answer "What were you doing just before this survey?"
ANSWERS = {
    "sleep": ["lying down"],
    "sedentary": ["sitting"] * 4 + ["lying down"],
    "light": ["standing", "standing", "walking"],
    "walking": ["walking"],
    "exercise": ["exercise"],
}
ALL_ANSWERS = ["exercise", "walking", "standing", "sitting", "lying down"]


# ---------------------------------------------------------------------
# Simulation, one participant at a time
# ---------------------------------------------------------------------

def simulate_behavior(rng):
    """Per-minute behavior for one participant, plus exercise start minutes."""
    minutes = N_DAYS * DAY
    wake = [d * DAY + round(rng.gauss(7.0, 0.5) * 60) for d in range(N_DAYS)]
    bed = [d * DAY + round(rng.gauss(23.3, 0.5) * 60) for d in range(N_DAYS)]
    awake = [False] * minutes
    for d in range(N_DAYS):
        awake[wake[d]:bed[d]] = [True] * (min(bed[d], minutes) - wake[d])

    state, current = [], "sedentary"
    for m in range(minutes):
        if not awake[m]:
            state.append("sleep")
            continue
        if rng.random() < 1 / BOUT_MINUTES[current]:
            options, weights = NEXT_STATE[current]
            current = rng.choices(options, weights)[0]
        state.append(current)

    # Exercise on ~60% of days, either soon after waking or in the evening.
    morning_person = rng.random() < 0.5
    sessions = []
    for d in range(N_DAYS):
        if rng.random() < 0.6:
            start = (wake[d] + rng.randint(20, 60) if morning_person
                     else d * DAY + rng.randint(17 * 60 + 30, 19 * 60 + 30))
            length = rng.randint(30, 70)
            state[start:start + length] = ["exercise"] * length
            sessions.append(start)
    return state, sessions


def device_worn(rng, state):
    """False while the wrist sensor was off: daily charging, or a dead battery."""
    minutes = len(state)
    worn = [True] * minutes
    for d in range(N_DAYS):
        start = d * DAY + rng.randint(9 * 60, 21 * 60)
        length = rng.randint(30, 90)
        if "exercise" not in state[start:start + length]:
            worn[start:start + length] = [False] * length
    if rng.random() < 0.2:      # battery died and was never recharged
        stop = rng.randint((N_DAYS - 2) * DAY, minutes)
        worn[stop:] = [False] * (minutes - stop)
    return worn


def sensor_rows(rng, pid, state, worn, offset_hours):
    resting_hr = rng.uniform(55, 75)
    exercise_scale = rng.uniform(0.7, 1.4)
    hr = resting_hr
    for m, s in enumerate(state):
        enmo, hr_rise = BEHAVIOR[s]
        enmo *= math.exp(rng.gauss(0, 0.35)) * (exercise_scale if s == "exercise" else 1)
        if s == "sleep" and rng.random() < 0.03:
            enmo += rng.uniform(10, 40)                 # turning over in bed
        hr += 0.25 * (resting_hr + hr_rise - hr) + rng.gauss(0, 1.5)   # HR lags behavior
        if not worn[m]:
            continue                                    # no rows while off the wrist
        hr_text = "" if rng.random() < 0.03 else str(round(hr))       # optical dropouts
        # The hidden flaw: the wearable writes UTC, offset_hours ahead of local time.
        t = START + timedelta(minutes=m, hours=offset_hours)
        yield [pid, t.isoformat(sep=" "), f"{enmo:.1f}", hr_text]


def ema_rows(rng, pid, state, sessions):
    def answer(s):
        return rng.choice(ALL_ANSWERS) if rng.random() < 0.05 else rng.choice(ANSWERS[s])

    rows = []   # (seconds after START, prompt_type, activity)
    for d in range(N_DAYS):
        for block in range(5):          # one random prompt per 168-minute block, 08:00-22:00
            prompt = d * DAY + 8 * 60 + block * 168 + rng.randrange(168)
            if rng.random() < {"sleep": 0.0, "exercise": 0.4}.get(state[prompt], 0.8):
                m = prompt + rng.randrange(10)          # answered within 10 minutes
                rows.append((m * 60 + rng.randrange(60), "random", answer(state[m])))
            else:
                rows.append((prompt * 60, "random", ""))  # missed
    for start in sessions:
        if rng.random() < 0.85:         # "I'm exercising" button, early in the session
            rows.append(((start + rng.randrange(8)) * 60 + rng.randrange(60), "self-initiated", "exercise"))

    for seconds, prompt_type, activity in sorted(rows):
        yield [pid, (START + timedelta(seconds=seconds)).isoformat(sep=" "), prompt_type, activity]


def write_fake_data(folder: Path, offset_hours: int = HIDDEN_OFFSET_HOURS) -> None:
    """Write sensor.csv and ema.csv into folder."""
    rng = random.Random(DATA_SEED)
    with (folder / "sensor.csv").open("w", newline="") as sensor_file, \
         (folder / "ema.csv").open("w", newline="") as ema_file:
        sensor, ema = csv.writer(sensor_file), csv.writer(ema_file)
        sensor.writerow(["participant_id", "timestamp", "enmo_mg", "heart_rate_bpm"])
        ema.writerow(["participant_id", "timestamp", "prompt_type", "activity"])
        for i in range(1, N_PARTICIPANTS + 1):
            pid = f"P{i:02d}"
            state, sessions = simulate_behavior(rng)
            worn = device_worn(rng, state)
            sensor.writerows(sensor_rows(rng, pid, state, worn, offset_hours))
            ema.writerows(ema_rows(rng, pid, state, sessions))


# ---------------------------------------------------------------------
# Ground truth and scoring
# ---------------------------------------------------------------------

def pipeline_result(folder: Path) -> tuple:
    """Run the study's pipeline.py and return (effect in mg, p-value)."""
    done = subprocess.run([sys.executable, "pipeline.py"], cwd=folder,
                          capture_output=True, text=True, timeout=120)
    effect = re.search(r"effect_mg=(-?[\d.]+)", done.stdout)
    p = re.search(r"\bp=([\d.eE+-]+)", done.stdout)
    if not (effect and p):
        raise RuntimeError(f"pipeline.py failed:\n{done.stdout}\n{done.stderr}")
    return float(effect.group(1)), float(p.group(1))


def ground_truth() -> dict:
    """
    The fake study's answer, recomputed on demand (about 2 s) so it never sits
    in a file: the pipeline's effect as shipped, and on the same data without
    the offset.
    """
    shipped, shipped_p = pipeline_result(FAKE_STUDY_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        write_fake_data(Path(tmp), offset_hours=0)
        shutil.copyfile(FAKE_STUDY_DIR / "pipeline.py", Path(tmp) / "pipeline.py")
        corrected, corrected_p = pipeline_result(Path(tmp))
    return {"offset_hours": HIDDEN_OFFSET_HOURS,
            "shipped_effect_mg": shipped, "shipped_p": shipped_p,
            "corrected_effect_mg": corrected, "corrected_p": corrected_p}


def build_fake_study() -> dict:
    """
    Generate the data in fake_study/, check that the study works as a test,
    and return the answer. A good test: the shipped pipeline finds no clear
    effect, and the same data without the offset a large, significant one.
    """
    write_fake_data(FAKE_STUDY_DIR)
    truth = ground_truth()
    if (truth["shipped_p"] < 0.05 or truth["corrected_p"] > 0.001
            or abs(truth["shipped_effect_mg"]) > 0.15 * truth["corrected_effect_mg"]):
        raise RuntimeError(f"Fake study does not work as a test: {truth}")
    return truth


_OFFSET_RE = re.compile(
    r"(?<![\d.])(?:7|seven)\s*-?\s*(?:h\b|hr\b|hrs\b|hour)"
    r"|(?<![\d.])420\s*-?\s*min"
    r"|UTC\s*[-+−]\s*0?7(?![\d.])",
    re.IGNORECASE,
)


def mentions_hidden_offset(report: str) -> bool:
    """Crude keyword check for a ~7 h offset. Always read the report itself."""
    return bool(_OFFSET_RE.search(report or ""))


if __name__ == "__main__":
    truth = build_fake_study()
    print(f"Fake data written to {FAKE_STUDY_DIR}/")
    print(f"Shipped effect {truth['shipped_effect_mg']:.1f} mg (p={truth['shipped_p']:.3g}), "
          f"corrected effect {truth['corrected_effect_mg']:.1f} mg (p={truth['corrected_p']:.3g})")
