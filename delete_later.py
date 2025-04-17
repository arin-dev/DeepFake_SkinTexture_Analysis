# import csv
# import argparse
# from collections import defaultdict

# def main():
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description='Count frequency of values in CSV column grouped by step size')
#     parser.add_argument('csv_file', help='Path to the CSV file')
#     parser.add_argument('--step', type=float, required=True, help='Step size for grouping values (can be decimal)')
#     args = parser.parse_args()

#     try:
#         with open(args.csv_file, 'r') as file:
#             reader = csv.reader(file)
#             # Skip header row
#             next(reader)
#             # Get 4th column (index 3 since Python is 0-based)
#             values = [float(row[3]) for row in reader if len(row) > 3]
            
#             if not values:
#                 print("No valid data found in 4th column")
#                 return

#             # Calculate frequency distribution
#             max_val = max(values)
#             freq_dist = defaultdict(int)
            
#             for val in values:
#                 # Determine which step bucket the value falls into (works for both integer and decimal steps)
#                 bucket = int(val // args.step) * args.step
#                 freq_dist[bucket] += 1
            
#             # Print results sorted by bucket
#             print(f"Frequency distribution (step size: {args.step}):")
#             for bucket in sorted(freq_dist.keys()):
#                 print(f"{bucket:.2f}+ to {bucket + args.step:.2f}: {freq_dist[bucket]}")

#     except FileNotFoundError:
#         print(f"Error: File '{args.csv_file}' not found")
#     except ValueError:
#         print("Error: Non-numeric value found in 4th column")
#     except Exception as e:
#         print(f"An unexpected error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()