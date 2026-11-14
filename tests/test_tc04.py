import numpy as np
import pandas as pd


def load_timestamps(split):
    data = np.load(
        "data/windows_96_to_24.npz",
        allow_pickle=False
    )
    return pd.DataFrame(data[f"{split}_target_times"]).apply(
        pd.to_datetime,
        errors="coerce"
    )


def test_tc04_detect_missing_time_intervals():
    """TC04.1 - Phát hiện khoảng thời gian gián đoạn giữa các cửa sổ."""

    times = load_timestamps("train")
    starts = times.iloc[:, 0]

    diffs = starts.diff().dropna()
    abnormal = diffs[diffs != pd.Timedelta(hours=1)]

    print("\nTC04.1 - Missing time intervals")
    print("Abnormal intervals:", len(abnormal))

    for idx, diff in abnormal.items():
        print(
            f"Window {idx-1} -> {idx}: "
            f"{starts.iloc[idx-1]} -> {starts.iloc[idx]} "
            f"| Gap: {diff}"
        )

    assert len(abnormal) > 0


def test_tc04_detect_duplicate_timestamps_inside_window():
    """TC04.2 - Kiểm tra timestamp trùng bên trong từng cửa sổ."""

    times = load_timestamps("train")
    duplicate_windows = 0

    for _, row in times.iterrows():
        if row.duplicated().any():
            duplicate_windows += 1

    print("\nTC04.2 - Duplicate timestamps inside windows")
    print("Windows containing duplicate timestamps:", duplicate_windows)

    assert duplicate_windows == 0


def test_tc04_detect_unsorted_timestamps_inside_window():
    """TC04.3 - Kiểm tra timestamp sai thứ tự bên trong từng cửa sổ."""

    times = load_timestamps("train")
    unsorted_windows = 0

    for _, row in times.iterrows():
        if not row.is_monotonic_increasing:
            unsorted_windows += 1

    print("\nTC04.3 - Unsorted timestamps inside windows")
    print("Windows containing unsorted timestamps:", unsorted_windows)

    assert unsorted_windows == 0
