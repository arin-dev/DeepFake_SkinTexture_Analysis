import cv2
import numpy as np

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
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # gray = cv2.GaussianBlur(gray, (3,3), 0)  # More precise than bilateral for small images
        # return gray - cv2.blur(gray, (3,3)) 
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        kernel = np.ones((self.law_size, self.law_size), np.float32)/(self.law_size**2)
        return gray - cv2.filter2D(gray, -1, kernel)
    
    def _compute_energy(self, img):
        energy = {}
        kernel = np.ones((self.energy_size, self.energy_size), np.float32)/(self.energy_size**2)
        
        for name, law_img in self._filter_image(img).items():
            energy[name] = cv2.filter2D(law_img**2, -1, kernel)
        
        # return {
        #     'L3E3': energy['L3E3'],
        #     'E3S3': energy['E3S3'],
        #     'S3S3': energy['S3S3'],
        #     'Combined': 0.4*energy['L3L3'] + 0.4*energy['E3E3'] + 0.2*energy['S3S3']
        # }
    
        return {
            'L3E3': energy['L3E3'],
            'E3S3': energy['E3S3'],
            'S3S3': energy['S3S3'],
            'Combined': 0.4*energy['L3L3'] + 0.4*energy['E3E3'] + 0.2*energy['S3S3']
        }
    
    def _filter_image(self, img):
        return {name: cv2.filter2D(img, -1, kernel) 
                for name, kernel in self.law_masks.items()}
    
    def extract_features(self, image_array):
        """Extract texture features from image array (RGB format)
        Returns: 16-dimensional feature vector (mean, std, 25p, 75p for each energy map)
        """
        preprocessed = self._preprocess(image_array)
        energy = self._compute_energy(preprocessed)
        
        return energy
        # features = []
        # for e_map in energy.values():
        #     features.extend([
        #         np.mean(e_map),
        #         np.std(e_map),
        #         np.percentile(e_map, 25),
        #         np.percentile(e_map, 75)
        #     ])
        
        # return np.array(features)