import requests
from bs4 import BeautifulSoup
import csv

def scrape_meaning_to_gloss():
    # Base URL format
    base_url = "https://www.sign-lang.uni-hamburg.de/glex/bedeut/x{}.html"

    # Total number of pages to scrape
    total_pages = 18

    # Prepare the CSV file for writing
    with open('dgs-gloss-dictionary.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write the header row
        csvwriter.writerow(["Meaning", "Glosses"])

        # Loop through all pages
        for page_num in range(1, total_pages + 1):
            # Construct the URL for the current page
            url = base_url.format(page_num)
            print(f"Scraping page: {url}")

            # Send a GET request to fetch the content of the page
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception if the request failed

            # Set the correct encoding and parse the HTML content
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the table with class 'hitsmeaning'
            table = soup.find('table', class_='hitsmeaning')

            # Check if the table exists
            if not table:
                print(f"Table with class 'hitsmeaning' not found on page: {url}")
                continue

            # Iterate over all rows in the table
            for row in table.find_all('tr'):
                # Extract the td.text-top element (Meaning)
                text_top_td = row.find('td', class_='text-top')

                # Extract all a.Gloss elements (Glosses)
                gloss_td = row.find_all('td')[1] if len(row.find_all('td')) > 1 else None
                gloss_links = gloss_td.find_all('a', class_='Gloss') if gloss_td else []

                # Check if both elements are present
                if text_top_td and gloss_links:
                    text_top_name = text_top_td.get_text(strip=True)  # Extract Meaning
                    gloss_names = [gloss.get_text(strip=True) for gloss in gloss_links]  # Extract Gloss names

                    # Write to the CSV file
                    csvwriter.writerow([text_top_name] + gloss_names)
                    print(f"Extracted: Meaning = {text_top_name}, Glosses = {gloss_names}")

    print("Scraping completed! Data saved to 'gloss-dictionary.csv'.")

if __name__ == "__main__":
    scrape_meaning_to_gloss()
