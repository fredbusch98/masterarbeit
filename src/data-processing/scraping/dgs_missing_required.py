import requests
from bs4 import BeautifulSoup

# URL of the webpage
url = "https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start-name_de.html"

# Step 1: Request the webpage
response = requests.get(url)
if response.status_code != 200:
    print(f"Failed to fetch the page. Status code: {response.status_code}")
    exit()

# Step 2: Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Step 3: Locate the table and headers
table = soup.find('table')
if not table:
    print("No table found on the page.")
    exit()

headers = table.find_all('th')
header_mapping = {header.text.strip(): idx for idx, header in enumerate(headers)}
required_columns = ['SRT', 'OpenPose']

if not all(col in header_mapping for col in required_columns):
    print("Required columns not found in the table headers.")
    exit()

# Step 4: Loop through all rows and count missing entries
rows = table.find_all('tr')[1:]  # Exclude the header row

total_entries = len(rows)
missing_srt = 0
missing_openpose = 0
missing_both = 0

for row in rows:
    cells = row.find_all('td')
    
    # Check for SRT link
    srt_cell = cells[header_mapping['SRT']]
    srt_link = srt_cell.find('a')
    srt_exists = srt_link is not None and srt_link.get('href') is not None

    # Check for OpenPose link
    openpose_cell = cells[header_mapping['OpenPose']]
    openpose_link = openpose_cell.find('a')
    openpose_exists = openpose_link is not None and openpose_link.get('href') is not None

    # Update counts based on existence
    if not srt_exists:
        missing_srt += 1
    if not openpose_exists:
        missing_openpose += 1
    if not srt_exists and not openpose_exists:
        missing_both += 1

# Print the results
print("Total entries:", total_entries)
print("Entries missing SRT transcript:", missing_srt)
print("Entries missing OpenPose:", missing_openpose)
print("Entries missing both SRT and OpenPose:", missing_both)
