"""
Analyzes 2D pose sequences from JSON files to evaluate motion smoothness.
Computes Spectral Arc Length (SAL) and Mean Squared Jerk (MSJ) per sequence, compares interpolation methods, and saves results to CSV.
"""
import os
import json
import numpy as np
import csv
from statistics import mean, median, stdev
from tabulate import tabulate

def extract_xy(flat_array):
    x = flat_array[0::3]
    y = flat_array[1::3]
    return [val for pair in zip(x, y) for val in pair]

def get_positions_array(data):
    num_frames = len(data)
    if num_frames < 2:
        return None, 0

    num_pose = len(data[0]['pose_keypoints_2d']) // 3
    num_left_hand = len(data[0]['hand_left_keypoints_2d']) // 3
    num_right_hand = len(data[0]['hand_right_keypoints_2d']) // 3
    num_face = len(data[0]['face_keypoints_2d']) // 3
    total_keypoints = num_pose + num_left_hand + num_right_hand + num_face

    positions = np.zeros((num_frames, 2 * total_keypoints))

    for t in range(num_frames):
        frame = data[t]
        pose = frame['pose_keypoints_2d']
        left = frame['hand_left_keypoints_2d']
        right = frame['hand_right_keypoints_2d']
        face = frame['face_keypoints_2d']

        all_positions = extract_xy(pose) + extract_xy(left) + extract_xy(right) + extract_xy(face)
        positions[t, :] = all_positions

    return positions, num_frames

def compute_sal(data):
    """
    Computes Spectral Arc Length (SAL) as a smoothness metric.
    - Higher (closer to 0) = smoother
    - Lower (more negative) = jerkier
    """
    positions, num_frames = get_positions_array(data)
    if positions is None or num_frames < 2:
        return None

    diffs = np.diff(positions, axis=0)
    speed = np.linalg.norm(diffs, axis=1)

    fft_result = np.fft.fft(speed)
    magnitude = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(speed))

    m = len(speed)
    freq_pos = frequencies[:m//2 + 1]
    mag_pos = magnitude[:m//2 + 1]

    cumsum = np.cumsum(mag_pos) / np.sum(mag_pos)
    cutoff_idx = np.where(cumsum >= 0.99)[0][0] if np.any(cumsum >= 0.99) else len(freq_pos) - 1
    delta_f = freq_pos[1] - freq_pos[0] if len(freq_pos) > 1 else 1
    arc_length = np.sum(np.sqrt(np.diff(mag_pos[:cutoff_idx + 1])**2 + delta_f**2)) if cutoff_idx > 0 else 0

    return -arc_length  # Lower arc length = smoother, so we return negative

def compute_msj(data):
    """
    Computes Mean Squared Jerk (MSJ).
    - Lower MSJ = smoother motion
    - Higher MSJ = more jerky motion
    """
    positions, num_frames = get_positions_array(data)
    if positions is None or num_frames < 4:
        return None

    jerk = np.diff(positions, n=3, axis=0)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    msj = np.mean(jerk_magnitude ** 2)
    return msj

def process_folder(folder_path, label):
    sal_values = []
    msj_values = []
    results = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json"):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, 'r') as f:
                try:
                    data = json.load(f)
                    sal = compute_sal(data)
                    msj = compute_msj(data)
                    if sal is not None and msj is not None:
                        sal_values.append(sal)
                        msj_values.append(msj)
                        results.append((filename, label, sal, msj))
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    return results, sal_values, msj_values

def get_stats(values):
    if not values:
        return ("N/A", "N/A", "N/A")
    return (
        f"{mean(values):.4f}",
        f"{median(values):.4f}",
        f"{stdev(values):.4f}" if len(values) > 1 else "N/A"
    )

# === PATHS ===
folder_no_interp = "./pose-sequence-eval/no_interpolation"
folder_interp = "./pose-sequence-eval/linear_interpolation"
folder_bspline = "./pose-sequence-eval/bspline_interpolation"
csv_output = "./pose-sequence-eval/sal_msj_results.csv"

# === PROCESS DATA ===
results_no_interp, sals_no, msjs_no = process_folder(folder_no_interp, "no_interpolation")
results_interp, sals_interp, msjs_interp = process_folder(folder_interp, "linear_interpolation")
results_bspline, sals_bspline, msjs_bspline = process_folder(folder_bspline, "bspline_interpolation")

# === SAVE TO CSV ===
all_results = results_no_interp + results_interp + results_bspline
with open(csv_output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "type", "SAL", "MSJ"])
    writer.writerows(all_results)

# === STATS ===
sal_stats_no = get_stats(sals_no)
sal_stats_interp = get_stats(sals_interp)
sal_stats_bspline = get_stats(sals_bspline)

msj_stats_no = get_stats(msjs_no)
msj_stats_interp = get_stats(msjs_interp)
msj_stats_bspline = get_stats(msjs_bspline)

# === COMPARISON TABLE ===
table = [
    ["Metric", "Better Direction", "Stat", "No Interpolation", "Linear Interpolation", "B-spline Interpolation"],
    ["SAL", "Higher (closer to 0)", "Mean", sal_stats_no[0], sal_stats_interp[0], sal_stats_bspline[0]],
    ["SAL", "", "Median", sal_stats_no[1], sal_stats_interp[1], sal_stats_bspline[1]],
    ["SAL", "", "Std Dev", sal_stats_no[2], sal_stats_interp[2], sal_stats_bspline[2]],
    ["MSJ", "Lower", "Mean", msj_stats_no[0], msj_stats_interp[0], msj_stats_bspline[0]],
    ["MSJ", "", "Median", msj_stats_no[1], msj_stats_interp[1], msj_stats_bspline[1]],
    ["MSJ", "", "Std Dev", msj_stats_no[2], msj_stats_interp[2], msj_stats_bspline[2]],
]

print("\n📊 Motion Smoothness Metrics Comparison\n")
print(tabulate(table, headers="firstrow", tablefmt="grid"))

print(f"\n✅ CSV results saved to: {csv_output}")
print("\nℹ️  Interpretation:")
print("  - SAL (Spectral Arc Length): Closer to 0 = smoother. More negative = jerkier.")
print("  - MSJ (Mean Squared Jerk): Lower = smoother. Higher = more abrupt/jerky movement.")
