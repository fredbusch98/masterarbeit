import csv

# Read and process the CSV file
with open('eval.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # Convert each value in the row to a float
        numbers = list(map(float, row))
        # Calculate the average
        avg = sum(numbers) / len(numbers)
        # Print the average
        print(avg)