import sys
import numpy as np
import cv2
from src.image_processor import ImageProcessor
from src.fattal_tone_mapping import FattalToneMapping
from src.poisson_solver import PoissonSolver

def main():
    # Default parameters from the paper's implementation
    alpha_multiplier = 0.18
    bheta = 0.87
    s = 0.55
    
    if len(sys.argv) < 3:
        print("Usage: python main.py InputPath OutputPath [AlphaMultiplier] [Bheta] [S]")
        print("\nSupported formats: .hdr, .png, .jpg, .jpeg, .bmp, .tiff")
        print(f"\nDefault parameters:")
        print(f"  AlphaMultiplier: {alpha_multiplier}")
        print(f"  Bheta: {bheta}")
        print(f"  S (saturation): {s}")
        return 0
    
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    
    if len(sys.argv) >= 4:
        alpha_multiplier = float(sys.argv[3])
    if len(sys.argv) >= 5:
        bheta = float(sys.argv[4])
    if len(sys.argv) >= 6:
        s = float(sys.argv[5])
    
    print(f"Processing: {input_file_path}")
    print(f"Parameters: alpha={alpha_multiplier}, beta={bheta}, saturation={s}")
    
    # Load and prepare image (handles both HDR and LDR)
    success, hdr_image, luminance, width, height = ImageProcessor.load_and_prepare_image(input_file_path)
    
    if not success:
        print("Failed to load image")
        return -1
    
    print(f"Image loaded successfully: {width}x{height}")
    
    # Log luminance (safe because we ensured luminance > 0)
    print("Computing log luminance...")
    log_luma = np.log(luminance).astype(np.float32)
    
    # Check for inf/nan in log_luma
    if not np.all(np.isfinite(log_luma)):
        print("Warning: Non-finite values in log luminance")
        log_luma = np.where(np.isfinite(log_luma), log_luma, -10.0)
    
    # Apply tone mapping
    print("Applying tone mapping...")
    div_g = np.zeros_like(log_luma, dtype=np.float32)
    FattalToneMapping.set_mapping_settings(bheta, alpha_multiplier)
    FattalToneMapping.apply_tone_mapping(log_luma, div_g)
    
    # Check div_g for issues
    if not np.all(np.isfinite(div_g)):
        print("Warning: Non-finite values in divergence")
        div_g = np.where(np.isfinite(div_g), div_g, 0.0)
    
    print(f"Divergence range: {np.min(div_g)} to {np.max(div_g)}")
    
    # Solve Poisson equation
    print("Solving Poisson equation...")
    h1, h2, a1, a2 = 1.0, 1.0, 1.0, 1.0
    bd_value = PoissonSolver.neumann_compat(div_g, a1, a2, h1, h2)
    print(f"Neumann boundary value: {bd_value}")
    
    if not np.isfinite(bd_value):
        print("Warning: Non-finite boundary value, using 0.0")
        bd_value = 0.0
    
    U = PoissonSolver.poisolve(div_g, a1, a2, h1, h2, bd_value, 'neumann')
    
    # Check U for issues
    if not np.all(np.isfinite(U)):
        print("Warning: Non-finite values in solution")
        U = np.where(np.isfinite(U), U, 0.0)
    
    print(f"Solution range: {np.min(U)} to {np.max(U)}")
    
    # Exponential to get output luminance
    out_luma = np.exp(U)
    
    # Normalize with max value
    min_val = np.min(out_luma)
    max_val = np.max(out_luma)
    print(f"Output luminance range: {min_val} to {max_val}")
    
    if max_val > 0 and np.isfinite(max_val):
        out_luma = out_luma / max_val
    else:
        print("Error: Invalid output luminance")
        return -1
    
    out_luma_255 = np.clip(out_luma * 255, 0, 255).astype(np.uint8)
    
    # Save and display output luminance
    output_dir = output_file_path.rsplit('.', 1)[0] if '.' in output_file_path else output_file_path
    cv2.imwrite(output_dir + "_LDRLuminance.png", out_luma_255)
    cv2.imshow("outluminance", out_luma_255)
    cv2.waitKey(0)
    
    # Create color LDR image
    ldr_image_255 = ImageProcessor.create_ldr_output(hdr_image, luminance, out_luma, s)
    
    if ldr_image_255 is None:
        print("Failed to create LDR image")
        return -1
    
    # Save and display final result
    print("Saving final result...")
    if ImageProcessor.save_and_display(ldr_image_255, output_dir + "_LDRImage.png", "result"):
        print("Processing complete!")
        print(f"Output saved to: {output_dir}_LDRImage.png")
        return 0
    else:
        print("Failed to save output image")
        return -1

if __name__ == "__main__":
    sys.exit(main())