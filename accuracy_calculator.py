import csv
import sys
import os

def count_folders(csv_file, data_folder):
    two_id_count_0 = 0
    two_id_count_1 = 0
    one_id_count_0 = 0
    one_id_count_1 = 0
    Deepfake_DF_count = 0
    one_id_count_minus_1 = 0
    two_id_count_minus_1 = 0
    Deepfake_R_count = 0
    Real_DF_count = 0
    Real_R_count = 0

    # Get all existing subfolders in data_folder
    existing_folders = set()
    for f in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, f)):
            # Extract base name before _number
            # base_name = '_'.join(f.split('_')[:-1])
            # existing_folders.add(base_name)
            existing_folders.add(f)

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            # print(row)
            folder, cluster_label = row[0], row[1]
            
            # Only process if folder exists in data_folder
            if folder not in existing_folders:
                continue
                
            id_count = folder.count('id')
            
            if id_count == 2:
                if cluster_label == "0":
                    two_id_count_0 += 1
                elif cluster_label == "1":
                    two_id_count_1 += 1
                else:
                    two_id_count_minus_1 += 1
            elif id_count == 1:
                if cluster_label == "0":
                    one_id_count_0 += 1
                elif cluster_label == "1":
                    one_id_count_1 += 1
                else:
                    one_id_count_minus_1 += 1
            if id_count == 2:
                if cluster_label == "Deepfake":
                    Deepfake_DF_count += 1
                elif cluster_label == "Real":
                    Deepfake_R_count += 1
            elif id_count == 1:
                if cluster_label == "Deepfake":
                    Real_DF_count += 1
                elif cluster_label == "Real":
                    Real_R_count += 1

    print(f"Two 'id' folders with cluster_label 0: {two_id_count_0}")
    print(f"Two 'id' folders with cluster_label 1: {two_id_count_1}")
    print(f"Two 'id' folders with cluster_label -1: {two_id_count_minus_1}")
    print(f"One 'id' folders with cluster_label 0: {one_id_count_0}")
    print(f"One 'id' folders with cluster_label 1: {one_id_count_1}")
    print(f"One 'id' folders with cluster_label -1: {one_id_count_minus_1}")
    print(f"Two 'id' folders with cluster_label 0 | Deepfake_DF_count: {Deepfake_DF_count}")
    print(f"Two 'id' folders with cluster_label 1 | Deepfake_R_count: {Deepfake_R_count}")
    print(f"One 'id' folders with cluster_label 0 | Real_DF_count: {Real_DF_count}")
    print(f"One 'id' folders with cluster_label 1 | Real_R_count: {Real_R_count}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python accuracy_calculator.py <csv_file> <data_folder>")
    else:
        count_folders(sys.argv[1], sys.argv[2])
