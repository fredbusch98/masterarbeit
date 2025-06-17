import csv

# Read and process the CSV file
with open('eval.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # The first column is a string (e.g., name or label)
        label = row[0]
        # Convert the remaining values to floats
        numbers = list(map(float, row[1:]))
        # Calculate the average
        avg = sum(numbers) / len(numbers)
        # Print in the specified format
        print(f"{label}: {avg}")
