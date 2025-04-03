# import cv2
# import matplotlib.pyplot as plt
# import dlib
# import os
# import sys
# from lib.vaf_util import get_crops_landmarks

# def show_cropped_face(image_path, face_detector, sp68):
#     img = cv2.imread(image_path)
    
#     if img is None:
#         print(f"Could not open image file: {image_path}")
#         return

#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print(f"{gray_img.dtype} and {gray_img.shape}")

#     # colors = ['Red', 'Green', 'Blue']
#     # for i in range(3):
#     #     plt.imshow(img[:, :, i])
#     #     plt.title(colors[i])
#     #     plt.axis('off')
#     #     plt.show()
#     # plt.imshow(gray_img)
#     # plt.axis('off')
#     # plt.show()

#     # face_crops, _ = get_crops_landmarks(face_detector, sp68, img)
#     # face_crops, _ = get_crops_landmarks(face_detector, sp68, gray_img)
#     dets, scores, idx = face_detector.run(img, 0, 0)

#     print(len(dets))
#     # if len(face_crops) > 0:
#     #     plt.imshow(cv2.cvtColor(face_crops[0], cv2.COLOR_BGR2RGB))
#     #     plt.axis('off')
#     #     plt.show()
#     # else:
#     #     print("No face detected.")

# def load_face_detector(face_detector_path):
#     if not os.path.isfile(face_detector_path):
#         print("Could not find shape_predictor_68_face_landmarks.dat")
#         sys.exit()
#     face_detector = dlib.get_frontal_face_detector()
#     sp68 = dlib.shape_predictor(face_detector_path)
#     return face_detector, sp68

# def main():
#     if len(sys.argv) != 3:
#         print("Usage: python testing_functions.py <image_path> <face_detector_path>")
#         sys.exit(1)
    
#     image_path = sys.argv[1]
#     face_detector_path = sys.argv[2]
    
#     face_detector, sp68 = load_face_detector(face_detector_path)
#     show_cropped_face(image_path, face_detector, sp68)

# if __name__ == "__main__":
#     main()












# ##### DELETE RANDOM FILES:
# import os
# import random

# # Set the folder path (replace with your folder's path if it's not the current directory)
# folder_path = './function_test_data/Celeb-synthesis'

# # Get all files in the folder
# files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# # Check if there are more than 96 files
# if len(files) > 96:
#     # Calculate how many files need to be deleted
#     files_to_delete = len(files) - 96
    
#     # Randomly select files to delete
#     files_to_delete_randomly = random.sample(files, files_to_delete)
    
#     # Delete the selected files
#     for file in files_to_delete_randomly:
#         file_path = os.path.join(folder_path, file)
#         try:
#             os.remove(file_path)
#             print(f"Deleted {file}")
#         except Exception as e:
#             print(f"Error deleting {file}: {e}")

#     print(f"Done! {files_to_delete} files deleted, 96 files remain.")
# else:
#     print(f"You're already at or below 96 files. No deletion needed.")


import os
import random
import shutil

def delete_random_subfolders(folder_path, max_subfolders=10):
    # Get all subfolders in the directory
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # Check if there are more subfolders than the specified maximum
    if len(subfolders) > max_subfolders:
        # Calculate how many subfolders need to be deleted
        subfolders_to_delete = len(subfolders) - max_subfolders
        
        # Randomly select subfolders to delete
        subfolders_to_delete_randomly = random.sample(subfolders, subfolders_to_delete)
        
        # Delete the selected subfolders and their contents
        for subfolder in subfolders_to_delete_randomly:
            subfolder_path = os.path.join(folder_path, subfolder)
            try:
                shutil.rmtree(subfolder_path)  # Deletes the subfolder and all its contents
                print(f"Deleted subfolder: {subfolder}")
            except Exception as e:
                print(f"Error deleting subfolder {subfolder}: {e}")

        print(f"Done! {subfolders_to_delete} subfolders deleted, {max_subfolders} subfolders remain.")
    else:
        print(f"You're already at or below {max_subfolders} subfolders. No deletion needed.")

def main():
    folder_path = './frames_function_test_data/'
    max_subfolders = 50
    delete_random_subfolders(folder_path, max_subfolders)

if __name__ == "__main__":
    main()