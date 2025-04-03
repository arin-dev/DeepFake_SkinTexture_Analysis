# # import cv2
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import os

# # # Texture analysis functions (unchanged from original)
# # def generate_law_filters():
# #     LAWS = {'L5': [1, 4, 6, 4, 1], 
# #             'E5': [-1, -2, 0, 2, 1], 
# #             'S5': [-1, 0, 2, 0, -1], 
# #             'R5': [1, -4, 6, -4, 1]}
# #     law_masks = {}
# #     for type1, vector1 in LAWS.items():
# #         for type2, vector2 in LAWS.items():
# #             mask_type = type1 + type2
# #             filter_mask = np.asarray(vector1)[:, np.newaxis].T * np.asarray(vector2)[:, np.newaxis]
# #             law_masks[mask_type] = filter_mask
# #     return law_masks

# # def generate_mean_kernel(size):
# #     mean_kernel = np.ones((size, size), dtype=np.float32)
# #     return mean_kernel / mean_kernel.size

# # def preprocess_image(img, size=15):
# #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# #     mean_kernel = generate_mean_kernel(size)
# #     local_means = cv2.filter2D(img, -1, mean_kernel)
# #     return img - local_means

# # def filter_image(img, law_masks):
# #     law_images = {}
# #     for name, filter_kernel in law_masks.items():
# #         law_images[name] = cv2.filter2D(img, -1, filter_kernel)
# #     return law_images

# # def compute_energy(law_images, m_size):
# #     laws_energy = {}
# #     mean_kernel = generate_mean_kernel(m_size)
# #     for name, law_image in law_images.items():
# #         laws_energy[name] = cv2.filter2D(np.abs(law_image), -1, mean_kernel)
    
# #     # Combine symmetric filters
# #     laws_energy_final = {
# #         'L5E5_2': (laws_energy['L5E5'] + laws_energy['E5L5']) / 2.0,
# #         'L5R5_2': (laws_energy['L5R5'] + laws_energy['R5L5']) / 2.0,
# #         'E5S5_2': (laws_energy['S5E5'] + laws_energy['E5S5']) / 2.0,
# #         'L5S5_2': (laws_energy['S5L5'] + laws_energy['L5S5']) / 2.0,
# #         'E5R5_2': (laws_energy['E5R5'] + laws_energy['R5E5']) / 2.0,
# #         'S5R5_2': (laws_energy['S5R5'] + laws_energy['R5S5']) / 2.0,
# #         'S5S5': laws_energy['S5S5'],
# #         'R5R5': laws_energy['R5R5'],
# #         'E5E5': laws_energy['E5E5']
# #     }
# #     return laws_energy_final

# # def analyze_face_texture(img_path, law_filter_size=15, energy_window_size=15):
# #     # Load image
# #     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
# #     # Texture analysis pipeline
# #     laws_masks = generate_law_filters()
# #     preprocessed = preprocess_image(img, size=law_filter_size)
# #     filtered = filter_image(preprocessed, laws_masks)
# #     energy = compute_energy(filtered, energy_window_size)
    
# #     # Visualization
# #     plt.figure(figsize=(18, 12))
# #     plt.suptitle(os.path.basename(img_path))
    
# #     plots = [
# #         ('Original Image', img, None),
# #         ('Zero-Mean', preprocessed, 'gray'),
# #         ('L5E5 Energy', energy['L5E5_2'], 'jet'),
# #         ('E5S5 Energy', energy['E5S5_2'], 'jet'),
# #         ('S5S5 Energy', energy['S5S5'], 'jet'),
# #         ('Combined Energy', sum(energy.values())/len(energy), 'jet')
# #     ]
    
# #     for i, (title, data, cmap) in enumerate(plots, 1):
# #         plt.subplot(2, 3, i)
# #         plt.imshow(data, cmap=cmap)
# #         plt.title(title)
# #         plt.axis('off')
    
# #     plt.tight_layout()
# #     # plt.show()
# #     output_path = os.path.splitext(img_path)[0] + '_texture_analysis.png'
# #     plt.savefig(output_path)
# #     plt.close()

# # def process_folder(folder_path, law_filter_size=15, energy_window_size=15):
# #     if not os.path.isdir(folder_path):
# #         print(f"Error: {folder_path} is not a valid directory")
# #         return
    
# #     for filename in os.listdir(folder_path):
# #         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
# #             img_path = os.path.join(folder_path, filename)
# #             analyze_face_texture(img_path, law_filter_size, energy_window_size)

# # if __name__ == "__main__":
# #     import sys
# #     if len(sys.argv) < 2:
# #         print("Usage: python vaf_ext.py <image_path_or_folder> [law_filter_size] [energy_window_size]")
# #         sys.exit(1)
    
# #     path = sys.argv[1]
# #     law_size = int(sys.argv[2]) if len(sys.argv) > 2 else 15
# #     energy_size = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
# #     if os.path.isfile(path):
# #         analyze_face_texture(path, law_size, energy_size)
# #     elif os.path.isdir(path):
# #         process_folder(path, law_size, energy_size)
# #     else:
# #         print(f"Error: {path} is not a valid file or directory")

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# def generate_sharper_law_filters():
#     LAWS = {
#         'L3': [1, 2, 1],
#         'E3': [-1, 0, 1],
#         'S3': [-1, 2, -1]
#     }
#     law_masks = {}
#     for type1, vector1 in LAWS.items():
#         for type2, vector2 in LAWS.items():
#             mask_type = type1 + type2
#             filter_mask = np.outer(vector1, vector2)
#             law_masks[mask_type] = filter_mask
#     return law_masks

# def preprocess_sharp(img, size=7):
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     gray = cv2.bilateralFilter(gray, 9, 75, 75)
#     mean_kernel = np.ones((size, size), np.float32)/(size*size)
#     local_means = cv2.filter2D(gray, -1, mean_kernel)
#     return gray - local_means

# def filter_image(img, law_masks):
#     law_images = {}
#     for name, filter_kernel in law_masks.items():
#         law_images[name] = cv2.filter2D(img, -1, filter_kernel)
#     return law_images

# def compute_sharp_energy(law_images, m_size=5):
#     laws_energy = {}
#     for name, law_image in law_images.items():
#         laws_energy[name] = cv2.filter2D(law_image**2, -1, 
#                                        np.ones((m_size,m_size))/(m_size*m_size))
#     return laws_energy

# def analyze_face_texture_sharp(img_path, output_dir=None):
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
#     laws_masks = generate_sharper_law_filters()
#     preprocessed = preprocess_sharp(img, size=7)
#     filtered = filter_image(preprocessed, laws_masks)
#     energy = compute_sharp_energy(filtered, m_size=3)
    
#     plt.figure(figsize=(15,8))
    
#     selected_energies = {
#         'L3E3': energy['L3E3'],
#         'E3S3': energy['E3S3'], 
#         'S3S3': energy['S3S3'],
#         'Combined': 0.3*energy['L3L3'] + 0.5*energy['E3E3'] + 0.2*energy['S3S3']
#     }
    
#     # Plot original
#     plt.subplot(2,4,1)
#     plt.imshow(img)
#     plt.title('Original')
#     plt.axis('off')
    
#     # Plot processed versions
#     for i, (name, energy_map) in enumerate(selected_energies.items(), 2):
#         plt.subplot(2,4,i)
#         plt.imshow(energy_map, cmap='viridis')
#         plt.title(f'{name} Energy')
#         plt.axis('off')
    
#     plt.tight_layout()
    
#     # if output_dir:
#     #     os.makedirs(output_dir, exist_ok=True)
#     # output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '_texture.png')
#     # output_path = os.path.join(img_path, os.path.splitext(os.path.basename(img_path))[0] + '_texture.png')
#     output_path = os.path.splitext(img_path)[0] + '_texture_analysis.png'
#     plt.savefig(output_path)
#     plt.close()
#     # else:
#     #     plt.show()

# def process_folder(input_folder, output_folder=None):
#     if not os.path.isdir(input_folder):
#         print(f"Error: {input_folder} is not a valid directory")
#         return
    
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             img_path = os.path.join(input_folder, filename)
#             analyze_face_texture_sharp(img_path, output_folder)

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python texture_analysis.py <input_path> [output_folder]")
#         print("       input_path can be image file or directory")
#         sys.exit(1)
    
#     input_path = sys.argv[1]
#     output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    
#     if os.path.isfile(input_path):
#         analyze_face_texture_sharp(input_path, output_folder)
#     elif os.path.isdir(input_path):
#         process_folder(input_path, output_folder)
#     else:
#         print(f"Error: {input_path} is not a valid file or directory")


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.morphology import convex_hull_image

class FaceTextureAnalyzer:
    def __init__(self, law_size=3, energy_size=5):
        self.LAWS = {
            'L3': [1, 2, 1],
            'E3': [-1, 0, 1], 
            'S3': [-1, 2, -1]
        }
        self.law_size = law_size
        self.energy_size = energy_size
        self.law_masks = self._generate_law_filters()
        
    def _generate_law_filters(self):
        masks = {}
        for t1, v1 in self.LAWS.items():
            for t2, v2 in self.LAWS.items():
                masks[f"{t1}{t2}"] = np.outer(v1, v2)
        return masks
    
    def _preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        kernel = np.ones((self.law_size, self.law_size), np.float32)/(self.law_size**2)
        return gray - cv2.filter2D(gray, -1, kernel)
    
    def _compute_energy(self, img):
        energy = {}
        kernel = np.ones((self.energy_size, self.energy_size), np.float32)/(self.energy_size**2)
        
        for name, law_img in self._filter_image(img).items():
            energy[name] = cv2.filter2D(law_img**2, -1, kernel)
        
        return {
            'L3E3': energy['L3E3'],
            'E3S3': energy['E3S3'],
            'S3S3': energy['S3S3'],
            'Combined': 0.4*energy['L3L3'] + 0.4*energy['E3E3'] + 0.2*energy['S3S3']
        }
    
    def _filter_image(self, img):
        return {name: cv2.filter2D(img, -1, kernel) 
                for name, kernel in self.law_masks.items()}
    
    def analyze(self, img_path, output_path=None):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        preprocessed = self._preprocess(img)
        energy = self._compute_energy(preprocessed)
        
        plt.figure(figsize=(16, 8))
        
        # Original and preprocessed
        plt.subplot(2, 4, 1)
        plt.imshow(img)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(2, 4, 2)
        plt.imshow(preprocessed, cmap='gray')
        plt.title('Preprocessed')
        plt.axis('off')
        
        # Energy maps
        for i, (name, e_map) in enumerate(energy.items(), 3):
            plt.subplot(2, 4, i)
            plt.imshow(e_map, cmap='viridis')
            plt.title(name)
            plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
            
        return energy

def batch_process(input_dir, output_dir):
    analyzer = FaceTextureAnalyzer()
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in os.listdir(input_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_texture.png")
            analyzer.analyze(img_path, output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python texture_analysis.py <input_path> [output_dir]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if os.path.isfile(input_path):
        FaceTextureAnalyzer().analyze(input_path, 
                                    output_dir and os.path.join(output_dir, 
                                    f"{os.path.splitext(os.path.basename(input_path))[0]}_texture.png"))
    elif os.path.isdir(input_path):
        batch_process(input_path, output_dir or os.path.join(input_path, "texture_results"))
    else:
        print(f"Error: Invalid input path {input_path}")