import numpy as np
import cv2

class FattalToneMapping:
    m_alpha_multiplier = 0.1
    m_bheta = 0.83
    
    @staticmethod
    def apply_tone_mapping(log_luma, div_g):
        pyramid = []
        FattalToneMapping.build_gaussian_py(log_luma, pyramid)
        
        print(f"Built pyramid with {len(pyramid)} levels")
        
        scaling_vector = []
        for i in range(len(pyramid)):
            grad_x, grad_y = FattalToneMapping.calculate_gradient_py(pyramid[i], i)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Check for issues
            if not np.all(np.isfinite(grad_mag)):
                print(f"Warning: Non-finite gradient magnitude at level {i}")
                grad_mag = np.where(np.isfinite(grad_mag), grad_mag, 0.0)
            
            scaling = FattalToneMapping.calculate_scaling(grad_mag)
            
            if not np.all(np.isfinite(scaling)):
                print(f"Warning: Non-finite scaling at level {i}")
                scaling = np.where(np.isfinite(scaling), scaling, 1.0)
            
            scaling_vector.append(scaling)
            print(f"Level {i}: grad_mag range [{np.min(grad_mag):.6f}, {np.max(grad_mag):.6f}], "
                  f"scaling range [{np.min(scaling):.6f}, {np.max(scaling):.6f}]")
        
        attenuation = FattalToneMapping.calculate_attenuations(scaling_vector)
        
        if not np.all(np.isfinite(attenuation)):
            print(f"Warning: Non-finite attenuation values")
            attenuation = np.where(np.isfinite(attenuation), attenuation, 1.0)
        
        print(f"Attenuation range: [{np.min(attenuation):.6f}, {np.max(attenuation):.6f}]")
        
        cv2.imshow("Attenuation", attenuation)
        cv2.waitKey(0)
        
        attenuated_grad_x, attenuated_grad_y = FattalToneMapping.calculate_attenuated_gradient(
            log_luma, attenuation)
        
        if not np.all(np.isfinite(attenuated_grad_x)) or not np.all(np.isfinite(attenuated_grad_y)):
            print(f"Warning: Non-finite attenuated gradients")
            attenuated_grad_x = np.where(np.isfinite(attenuated_grad_x), attenuated_grad_x, 0.0)
            attenuated_grad_y = np.where(np.isfinite(attenuated_grad_y), attenuated_grad_y, 0.0)
        
        div_g[:] = FattalToneMapping.calculate_divergence(attenuated_grad_x, attenuated_grad_y)
        
        return True
    
    @staticmethod
    def build_gaussian_py(image, pyramid):
        down_image = image.copy()
        while down_image.shape[1] > 32 and down_image.shape[0] > 32:
            pyramid.append(down_image.copy())
            down_image = cv2.pyrDown(down_image, 
                                    dstsize=(down_image.shape[1] // 2, 
                                            down_image.shape[0] // 2))
    
    @staticmethod
    def calculate_divergence(gx, gy):
        rows, cols = gx.shape
        div_g = np.zeros((rows, cols), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                div_g[i, j] = gx[i, j] + gy[i, j]
                if j > 0:
                    div_g[i, j] -= gx[i, j - 1]
                if i > 0:
                    div_g[i, j] -= gy[i - 1, j]
                
                if j == 0:
                    div_g[i, j] += gx[i, j]
                if i == 0:
                    div_g[i, j] += gy[i, j]
        
        return div_g
    
    @staticmethod
    def calculate_gradient_py(image, level):
        rows, cols = image.shape
        grad_x = np.zeros((rows, cols), dtype=np.float32)
        grad_y = np.zeros((rows, cols), dtype=np.float32)
        
        divisor = 2.0 ** (level + 1)
        
        for i in range(rows):
            for j in range(cols):
                if i == 0:
                    grad_y[i, j] = (image[i + 1, j] - image[i, j]) / divisor
                elif i == rows - 1:
                    grad_y[i, j] = (image[i, j] - image[i - 1, j]) / divisor
                else:
                    grad_y[i, j] = (image[i + 1, j] - image[i - 1, j]) / divisor
                
                if j == 0:
                    grad_x[i, j] = (image[i, j + 1] - image[i, j]) / divisor
                elif j == cols - 1:
                    grad_x[i, j] = (image[i, j] - image[i, j - 1]) / divisor
                else:
                    grad_x[i, j] = (image[i, j + 1] - image[i, j - 1]) / divisor
        
        return grad_x, grad_y
    
    @staticmethod
    def set_mapping_settings(bheta, alpha_multiplier):
        FattalToneMapping.m_bheta = bheta
        FattalToneMapping.m_alpha_multiplier = alpha_multiplier
    
    @staticmethod
    def calculate_attenuated_gradient(image, phi):
        rows, cols = image.shape
        grad_x = np.zeros((rows, cols), dtype=np.float32)
        grad_y = np.zeros((rows, cols), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                if j + 1 >= cols:
                    grad_x[i, j] = (image[i, cols - 2] - image[i, j]) * 0.5 * \
                                   (phi[i, cols - 2] + phi[i, j])
                else:
                    grad_x[i, j] = (image[i, j + 1] - image[i, j]) * 0.5 * \
                                   (phi[i, j + 1] + phi[i, j])
                
                if i + 1 >= rows:
                    grad_y[i, j] = (image[rows - 2, j] - image[i, j]) * 0.5 * \
                                   (phi[rows - 2, j] + phi[i, j])
                else:
                    grad_y[i, j] = (image[i + 1, j] - image[i, j]) * 0.5 * \
                                   (phi[i + 1, j] + phi[i, j])
        
        return grad_x, grad_y
    
    @staticmethod
    def calculate_scaling(grad_mag):
        alpha = FattalToneMapping.m_alpha_multiplier * np.mean(grad_mag)
        
        if alpha <= 0:
            print(f"Warning: alpha is {alpha}, using default")
            alpha = 0.1
        
        # Compute scaling with proper handling of zero gradients
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = np.power(grad_mag / alpha, FattalToneMapping.m_bheta)
            scaling = (alpha / grad_mag) * temp
        
        # Replace inf/nan with 0 (where grad_mag was zero), because scaling should be 0 there
        scaling = np.where(np.isfinite(scaling), scaling, 0.0)
        
        return scaling.astype(np.float32)
    
    @staticmethod
    def calculate_attenuations(scalings):
        for i in range(len(scalings) - 2, -1, -1):
            temp = cv2.resize(scalings[i + 1], 
                            (scalings[i].shape[1], scalings[i].shape[0]),
                            interpolation=cv2.INTER_LINEAR)
            scalings[i] = scalings[i] * temp
        
        return scalings[0]