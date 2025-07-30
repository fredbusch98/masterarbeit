import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# Data
categories = ['dgs-total', 'g2p-total', 'g2p-dict']
file_sizes = [131.64, 53.14, 0.92]  # in GB
pose_frames = [17566500, 3762615, 164873]

# Percentages for annotation
file_size_percent = [100, 40.4, 0.7]
pose_frame_percent = [100, 21.4, 0.9]

# Bar positions
x = np.arange(len(categories))
width = 0.6

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# --- File Size Bar Plot ---
axs[0].bar(x, file_sizes, width, color=['#4c72b0', '#55a868', '#c44e52'])
for i, v in enumerate(file_sizes):
    axs[0].text(x[i], v + 5, f"{file_size_percent[i]}%", ha='center', fontweight='bold')
axs[0].set_title('Total File Size (GB)')
axs[0].set_ylabel('Size (GB)')
axs[0].set_xticks(x)
axs[0].set_xticklabels(categories)
axs[0].set_ylim(0, 150)

# --- Pose Frames Bar Plot with Y-axis in Millions ---
axs[1].bar(x, pose_frames, width, color=['#4c72b0', '#55a868', '#c44e52'])
for i, v in enumerate(pose_frames):
    axs[1].text(x[i], v + 5e5, f"{pose_frame_percent[i]}%", ha='center', fontweight='bold')
axs[1].set_title('Total Pose Frames (Millions)')
axs[1].set_ylabel('Number of Frames (Millions)')
axs[1].set_xticks(x)
axs[1].set_xticklabels(categories)
axs[1].set_ylim(0, 2e7)
axs[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*1e-6:.1f}M'))

# --- Layout ---
fig.suptitle('Pose Data Reduction Visualization: File Size & Pose Frames', fontsize=16, fontweight='bold')
plt.tight_layout()

# --- Save the plot ---
plt.savefig("pose_data_reduction_plot.png", dpi=300)  # or use .pdf, .svg as needed
