"""
Reads JSON files containing frame-by-frame x and y coordinates for a transition, 
then plots and compares the coordinates with and without interpolation. 
Saves the comparison as a PNG plot.
"""
import os
import json
import matplotlib.pyplot as plt

# Directory containing the JSON files
input_dir = "./transition-examples"

def read_json_data(filepath):
    """
    Reads x, y coordinates from a JSON file.
    Returns a list of (frame, x, y) tuples, where x and y are floats or None.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as jsonfile:
        json_data = json.load(jsonfile)
        for row in json_data:
            frame = int(row['frame'])
            x = float(row['x']) if row['x'] is not None else None
            y = float(row['y']) if row['y'] is not None else None
            data.append((frame, x, y))
    return data

def plot_transition(without_interp_data, with_interp_data, output_dir):
    """
    Creates a plot comparing x and y coordinates for the transition.
    Saves the plot as a PNG file.
    """
    # Extract frame indices and coordinates
    frames_without = [d[0] for d in without_interp_data]
    x_without = [d[1] for d in without_interp_data]
    y_without = [d[2] for d in without_interp_data]
    
    frames_with = [d[0] for d in with_interp_data]
    x_with = [d[1] for d in with_interp_data]
    y_with = [d[2] for d in with_interp_data]
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Plot x-coordinates
    plt.subplot(1, 2, 1)
    plt.plot(frames_with, x_with, 'bo-', label='With Interpolation', marker='o')
    plt.plot(frames_without, x_without, 'rx-', label='Without Interpolation', marker='x')
    plt.xlabel('Frame')
    plt.ylabel('X Coordinate')
    plt.title('Transition from Gloss N to Gloss N + 1: X Coordinate')
    plt.legend()
    plt.grid(True)
    
    # Plot y-coordinates
    plt.subplot(1, 2, 2)
    plt.plot(frames_with, y_with, 'bo-', label='With Interpolation', marker='o')
    plt.plot(frames_without, y_without, 'rx-', label='Without Interpolation', marker='x')
    plt.xlabel('Frame')
    plt.ylabel('Y Coordinate')
    plt.title('Transition from Gloss N to Gloss N + 1: Y Coordinate')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, "transition_plot.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def main():
    # Ensure output directory exists
    os.makedirs(input_dir, exist_ok=True)
    
    # File paths for the single transition
    without_interp_file = os.path.join(input_dir, "transition_without_interpolation.json")
    with_interp_file = os.path.join(input_dir, "transition_with_interpolation.json")
    
    # Check if both files exist
    if not os.path.exists(without_interp_file):
        print(f"Error: {without_interp_file} not found")
        return
    if not os.path.exists(with_interp_file):
        print(f"Error: {with_interp_file} not found")
        return
    
    # Read data
    without_interp_data = read_json_data(without_interp_file)
    with_interp_data = read_json_data(with_interp_file)
    
    # Generate plot
    plot_transition(without_interp_data, with_interp_data, input_dir)

if __name__ == "__main__":
    main()