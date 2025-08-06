import json
import csv
import matplotlib.pyplot as plt

# Paths to the input files
gloss_dict_path = "../../pipeline/resources/gloss2pose_dictionary.json"
gloss_times_path = "../../pipeline/resources/gloss_times_for_frame_interpolation.csv"
output_chart_path = "gloss_stretching_piechart.png"

# Constants
FPS = 50
MS_PER_FRAME = 1000 / FPS  # 20 ms per frame

print("Loading gloss2pose dictionary...")
# Load the gloss2pose_dictionary
with open(gloss_dict_path, 'r', encoding='utf-8') as f:
    gloss2pose_dict = json.load(f)
print(f"Loaded {len(gloss2pose_dict)} glosses from the dictionary.")

print("Loading gloss times CSV...")
# Load gloss_times and build a mapping from gloss to median_gd
gloss_to_median_gd = {}
with open(gloss_times_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    count = 0
    for row in reader:
        gloss = row["gloss"]
        median_gd = float(row["median_gd"])
        gloss_to_median_gd[gloss] = median_gd
        count += 1
print(f"Loaded median durations for {count} glosses.")

# Count glosses whose pose sequence duration (in ms) is shorter than median_gd
print("Evaluating gloss durations...")
shorter_count = 0
skipped_count = 0

for i, (gloss, pose_sequence) in enumerate(gloss2pose_dict.items(), 1):
    if gloss not in gloss_to_median_gd:
        skipped_count += 1
        continue  # Skip glosses without median_gd data

    num_frames = len(pose_sequence)
    duration_ms = num_frames * MS_PER_FRAME
    median_gd = gloss_to_median_gd[gloss]

    if duration_ms < median_gd:
        shorter_count += 1

    if i % 100 == 0 or i == len(gloss2pose_dict):
        print(f"Processed {i}/{len(gloss2pose_dict)} glosses...")

# Total glosses in the dictionary
total_glosses = len(gloss2pose_dict)
non_shorter_count = total_glosses - shorter_count

# Output the results
print("\n--- Summary ---")
print(f"{shorter_count} glosses will get Single Gloss Pose Sequence Stretching in the Gloss2Pose Translator module.")
if total_glosses > 0:
    percentage = (shorter_count / total_glosses) * 100
    print(f"That is {percentage:.2f}% of the {total_glosses} glosses in the dictionary.")
else:
    print("The gloss2pose dictionary is empty.")
print(f"{skipped_count} glosses were skipped due to missing median_gd data.")

# Create and save pie chart
labels = [
    "Glosses that will get stretching",
    "Glosses that won't get stretching"
]
sizes = [shorter_count, non_shorter_count]
colors = ["#ff0000","#0080ff"]
explode = (0.1, 0)  # explode 1st slice

plt.figure(figsize=(8, 6))
wedges, texts, autotexts = plt.pie(
    sizes,
    explode=explode,
    colors=colors,
    autopct='%1.1f%%',
    shadow=True,
    startangle=140
)
plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
plt.legend(wedges, labels, title="Legend", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(output_chart_path, dpi=300)
print(f"\nPie chart saved as '{output_chart_path}'")
