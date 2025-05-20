# Read the original file
with open("glosses.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Remove numbers after the comma
cleaned_lines = [line.split(',')[0] + '\n' for line in lines if ',' in line]

# Write the cleaned lines to a new file or overwrite the existing one
with open("glosses_cleaned.txt", "w", encoding="utf-8") as file:
    file.writelines(cleaned_lines)

print("Cleaned file saved as glosses_cleaned.txt")
