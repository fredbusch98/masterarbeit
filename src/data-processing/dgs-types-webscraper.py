import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# URL of the website
url = "https://www.sign-lang.uni-hamburg.de/meinedgs/ling/types_de.html"

# Send a GET request to the website with explicit encoding handling
response = requests.get(url)
response.encoding = 'utf-8'  # Ensure the response uses UTF-8 encoding

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the webpage with UTF-8 encoding
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the div with class 'textWrapper'
    div = soup.find('div', class_='textWrapper')

    # Initialize a set to store unique elements
    unique_elements = set()

    # Iterate over all <p> elements within the div
    for p in div.find_all('p'):
        # Extract the text content of the <p> element
        p_text = p.get_text(strip=True)

        # Use regex to find the leading gloss (e.g., ALPHA1, $ALPHA1, etc.)
        # This assumes the gloss is at the start of the <p> text and ends before any non-gloss characters
        gloss_match = re.match(r'^([A-Za-z$][\w^-]*)', p_text)
        if gloss_match:
            gloss = gloss_match.group(1)
            unique_elements.add(gloss)

        # Check if <p> contains an <a> tag
        a_tag = p.find('a')
        if a_tag and a_tag.text.strip():  # Ensure <a> tag has text
            text = a_tag.text.strip()
            unique_elements.add(text)

    # Convert the set to a list and create a DataFrame
    unique_elements_list = sorted(unique_elements)  # Sort for consistency
    df = pd.DataFrame(unique_elements_list, columns=["Text"])

    # Save the DataFrame to a CSV file with UTF-8 encoding
    csv_filename = "/Volumes/IISY/DGSKorpus/all-types-dgs.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8')

    print(f"CSV file '{csv_filename}' created with {len(unique_elements_list)} unique elements.")
else:
    print(f"Failed to fetch the webpage. Status code: {response.status_code}")