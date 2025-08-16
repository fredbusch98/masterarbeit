"""
Downloads videos, SRT transcripts, and OpenPose data from the DGS-Korpus Release 3 webpage, 
saving each entry in separate folders on an external drive, 
and decompresses any OpenPose .gz files automatically.
Essentially doing the Data Collection part.
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import gzip
import shutil

# URL of the webpage
url = "https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start-name_de.html"

# Set a limit for the number of entries to process
max_entries = -1  # If set to -1 all entries will be downloaded at once!

# Set the folder path on the external hard drive
external_drive_path = "/Volumes/IISY/DGSKorpus"
os.makedirs(external_drive_path, exist_ok=True)

def download_file(file_url, file_name):
    """Helper function to download a file."""
    response = requests.get(file_url, stream=True)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded: {file_name}")
    else:
        print(f"Failed to download {file_name}. Status code: {response.status_code}")

def decompress_gz(gz_file_path, json_file_path):
    """Decompress a .gz file and save as a .json file."""
    with gzip.open(gz_file_path, 'rb') as f_in:
        with open(json_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_file_path)  # Remove the .gz file after decompression
    print(f"Decompressed: {json_file_path}")

# Step 1: Request the webpage
response = requests.get(url)
if response.status_code != 200:
    print(f"Failed to fetch the page. Status code: {response.status_code}")
    exit()

# Step 2: Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Step 3: Locate the table and headers
table = soup.find('table')  # Assuming there's only one table
headers = table.find_all('th')

# Find the indices for the required columns
header_mapping = {header.text.strip(): idx for idx, header in enumerate(headers)}
required_columns = ['Video A', 'Video B', 'SRT', 'OpenPose']

if not all(col in header_mapping for col in required_columns):
    print("Required columns not found in the table headers.")
    exit()

# Step 4: Loop through all rows in the table
rows = table.find_all('tr')[1:]  # Exclude the header row
if not rows:
    print("No rows found in the table.")
    exit()

# Base URL of the website
base_url = "https://www.sign-lang.uni-hamburg.de/meinedgs/ling/"

# Determine how many entries to process (use all entries if max_entries is -1)
entries_to_process = len(rows) if max_entries == -1 else min(max_entries, len(rows))

# Track the progress
print(f"Total entries to process: {entries_to_process}")

# Process each row and download files
for idx, row in enumerate(rows[:entries_to_process]):  # Only process up to entries_to_process
    # Show progress as a percentage
    progress_percentage = (idx + 1) / entries_to_process * 100
    print(f"Processing entry {idx + 1}/{entries_to_process} ({progress_percentage:.2f}%)")

    cells = row.find_all('td')
    
    # Check if SRT link exists (mandatory)
    srt_idx = header_mapping['SRT']
    srt_button = cells[srt_idx].find('a')  # Assuming the button is an <a> tag
    if not srt_button or 'href' not in srt_button.attrs:
        print(f"Skipping entry {idx}: No SRT file found (SRT is mandatory).")
        continue  # Skip this entry entirely

    # Create a folder for this table entry
    entry_folder = os.path.join(external_drive_path, f"entry_{idx}")
    os.makedirs(entry_folder, exist_ok=True)

    # Download SRT file (since it is mandatory)
    srt_url = urljoin(base_url, srt_button['href'])
    srt_file_name = os.path.join(entry_folder, "transcript.srt")
    download_file(srt_url, srt_file_name)

    # Download Video A if available
    video_a_downloaded = False
    video_a_idx = header_mapping['Video A']
    video_a_button = cells[video_a_idx].find('a')
    if video_a_button and 'href' in video_a_button.attrs:
        video_a_url = urljoin(base_url, video_a_button['href'])
        video_a_file_name = os.path.join(entry_folder, "video-a.mp4")  # Fixed filename for Video A
        download_file(video_a_url, video_a_file_name)
        video_a_downloaded = True

    # Download Video B if available
    video_b_downloaded = False
    video_b_idx = header_mapping['Video B']
    video_b_button = cells[video_b_idx].find('a')
    if video_b_button and 'href' in video_b_button.attrs:
        video_b_url = urljoin(base_url, video_b_button['href'])
        video_b_file_name = os.path.join(entry_folder, "video-b.mp4")  # Fixed filename for Video B
        download_file(video_b_url, video_b_file_name)
        video_b_downloaded = True

    if not video_a_downloaded and not video_b_downloaded:
        print(f"Entry {idx}: No Video A or Video B found.")

    # Download OpenPose (if available)
    openpose_idx = header_mapping['OpenPose']
    openpose_button = cells[openpose_idx].find('a')
    openpose_found = False
    if openpose_button and 'href' in openpose_button.attrs:
        openpose_url = urljoin(base_url, openpose_button['href'])
        openpose_file_name = os.path.join(entry_folder, "openpose.json.gz")  # Fixed OpenPose filename (gz)
        download_file(openpose_url, openpose_file_name)
        
        # Check if the file is a .gz file and decompress it
        decompressed_file_name = os.path.join(entry_folder, "openpose.json")
        decompress_gz(openpose_file_name, decompressed_file_name)
        openpose_found = True

    if not openpose_found:
        # Rename the folder if no OpenPose file is found
        no_openpose_folder = os.path.join(external_drive_path, f"entry_{idx}_no_openpose")
        os.rename(entry_folder, no_openpose_folder)
        print(f"Entry {idx}: No OpenPose file found. Renamed folder to {no_openpose_folder}")

print(f"Processing completed. Processed up to {entries_to_process} entries.")
