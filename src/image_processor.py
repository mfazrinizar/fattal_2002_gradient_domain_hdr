import numpy as np
import cv2
from src.hdr_loader import HDRLoader

class ImageProcessor:
    @staticmethod
    def load_and_prepare_image(filepath):
        """
        Load image and prepare it exactly as main.py does.
        Detects format and converts LDR to HDR if needed.
        
        Returns:
            tuple: (success, hdr_image, luminance, width, height)
        """
        # Detect if it's HDR or LDR
        if filepath.lower().endswith('.hdr'):
            return ImageProcessor._load_hdr_image(filepath)
        else:
            return ImageProcessor._load_ldr_image(filepath)
    
    @staticmethod
    def _load_hdr_image(filepath):
        """Load HDR image using existing HDRLoader."""
        print("Loading HDR image...")
        success, result = HDRLoader.load(filepath)
        
        if not success:
            print("Failed to load HDR image")
            return False, None, None, 0, 0
        
        width = result.width
        height = result.height
        
        # Create HDR image array
        hdr_image = np.reshape(result.cols, (height, width, 3)).astype(np.float32)
        
        # Find min/max of HDR image
        min_val = np.min(hdr_image)
        max_val = np.max(hdr_image)
        
        print(f"HDR range: {min_val} to {max_val}")
        
        # Normalize
        if max_val > 0:
            hdr_image = hdr_image / max_val
        
        # Display input HDR
        lin_hdr = np.clip(hdr_image * 255, 0, 255).astype(np.uint8)
        cv2.imshow("Input HDR (linear)", lin_hdr)
        cv2.waitKey(0)
        
        # Convert RGB to XYZ
        hdr_xyz = cv2.cvtColor(hdr_image, cv2.COLOR_RGB2XYZ)
        
        # Extract luminance (Y channel = channels[1])
        channels = cv2.split(hdr_xyz)
        luminance = channels[1].copy()
        
        # Ensure no zeros or negative values in luminance
        luminance = np.maximum(luminance, 1e-4)
        
        # Display input luminance
        min_val = np.min(luminance)
        max_val = np.max(luminance)
        print(f"Luminance range: {min_val} to {max_val}")
        
        lin_hdr_luma = np.clip((luminance / max_val) * 255, 0, 255).astype(np.uint8)
        cv2.imshow("Inputluminance", lin_hdr_luma)
        cv2.waitKey(0)
        
        return True, hdr_image, luminance, width, height
    
    @staticmethod
    def _load_ldr_image(filepath):
        """Load LDR image and convert to HDR format."""
        print("Loading LDR image...")
        
        try:
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to load image: {filepath}")
                return False, None, None, 0, 0
            
            height, width = img.shape[:2]
            print(f"Image size: {width}x{height}")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to float and normalize to [0, 1]
            img_float = img_rgb.astype(np.float32) / 255.0
            
            # Apply inverse gamma to linearize (sRGB gamma correction)
            # This converts LDR to linear HDR space
            gamma = 2.2
            hdr_image = np.power(img_float, gamma)
            
            # Find min/max
            min_val = np.min(hdr_image)
            max_val = np.max(hdr_image)
            print(f"Converted HDR range: {min_val} to {max_val}")
            
            if max_val > 0:
                hdr_image = hdr_image / max_val
            
            # Display input 
            lin_hdr = np.clip(hdr_image * 255, 0, 255).astype(np.uint8)
            # Convert back to BGR for display
            lin_hdr_bgr = cv2.cvtColor(lin_hdr, cv2.COLOR_RGB2BGR)
            cv2.imshow("Input HDR (linear)", lin_hdr_bgr)
            cv2.waitKey(0)
            
            # Convert RGB to XYZ 
            hdr_xyz = cv2.cvtColor(hdr_image, cv2.COLOR_RGB2XYZ)
            
            # Extract luminance (Y channel = channels[1])
            channels = cv2.split(hdr_xyz)
            luminance = channels[1].copy()
            
            # Ensure no zeros or negative values in luminance
            luminance = np.maximum(luminance, 1e-4)
            
            # Display input luminance
            min_val = np.min(luminance)
            max_val = np.max(luminance)
            print(f"Luminance range: {min_val} to {max_val}")
            
            lin_hdr_luma = np.clip((luminance / max_val) * 255, 0, 255).astype(np.uint8)
            cv2.imshow("Inputluminance", lin_hdr_luma)
            cv2.waitKey(0)
            
            return True, hdr_image, luminance, width, height
            
        except Exception as e:
            print(f"Error loading LDR image: {e}")
            return False, None, None, 0, 0
    
    @staticmethod
    def create_ldr_output(hdr_image, luminance, out_luma, saturation):
        """
        Create color LDR image exactly as main.py does.
        
        Args:
            hdr_image: Normalized HDR image in RGB
            luminance: Original luminance
            out_luma: Output luminance after tone mapping
            saturation: Saturation parameter
            
        Returns:
            LDR image ready for saving
        """
        print("Creating color LDR image...")
        color_channels = cv2.split(hdr_image)
        out_color_channels = []
        
        # Ensure luminance is safe for division
        luminance_safe = np.maximum(luminance, 1e-4)
        
        # Process in BGR order (order: 2, 1, 0)
        for color_idx in [2, 1, 0]:  # B, G, R
            temp = color_channels[color_idx] / luminance_safe
            temp = np.clip(temp, 0, 100)  # Prevent extreme values
            temp = np.power(temp, saturation)
            ldr_channel = temp * out_luma
            out_color_channels.append(ldr_channel)
        
        ldr_image = cv2.merge(out_color_channels)
        
        # Normalize
        min_val = np.min(ldr_image)
        max_val = np.max(ldr_image)
        print(f"LDR image range: {min_val} to {max_val}")
        
        if max_val > 0 and np.isfinite(max_val):
            ldr_image_255 = np.clip((ldr_image / max_val) * 255, 0, 255).astype(np.uint8)
        else:
            print("Error: Invalid LDR image")
            return None
        
        return ldr_image_255
    
    @staticmethod
    def save_and_display(ldr_image_255, output_filepath, window_name="result"):
        """Save and display LDR image."""
        if ldr_image_255 is None:
            return False
        
        cv2.imwrite(output_filepath, ldr_image_255)
        cv2.imshow(window_name, ldr_image_255)
        cv2.waitKey(0)
        return True