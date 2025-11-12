# Fattal et al. (2002) Gradient Domain High Dynamic Range (HDR) Compression

This presents a Python implementation of the Fattal et al. (2002) HDR tone mapping algorithm. 

## Features

- **Multiple Input Formats**: Supports HDR (.hdr), PNG, and JPG images
- **Gradient Domain Tone Mapping**: Implementation of Fattal et al.'s gradient domain HDR compression
- **Poisson Solver**: FFT-based Poisson equation solver with Neumann boundary conditions
- **Color Preservation**: Maintains color appearance through proper color correction
- **Clean Architecture**: Separated into multiple modules following Python best practices

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python src/main.py input.hdr output.png
```

With custom parameters:
```bash
python src/main.py input.hdr output.png --alpha 0.18 --beta 0.87 --saturation 0.55
```

Display intermediate results:
```bash
python src/main.py input.hdr output.png --display
```

### Parameters

- `--alpha`: Alpha multiplier controlling gradient attenuation strength (default: 0.18)
- `--beta`: Beta parameter controlling gradient compression (default: 0.87)
- `--saturation`: Color saturation parameter (default: 0.55)
- `--display`: Display intermediate results in windows

## Project Structure

```
fattal_python/
├── src/
│   ├── hdr_loader.py           # Image loading (HDR/LDR)
│   ├── fattal_tone_mapping.py  # Core tone mapping algorithm
│   ├── poisson_solver.py       # Poisson equation solver
│   └── image_processor.py      # Image utilities and color correction
├── main.py                     # Main entry point
├── requirements.txt
└── README.md
```

## Module Description

- **[hdr_loader.py](src/hdr_loader.py)**: Handles loading of HDR and LDR images
- **[fattal_tone_mapping.py](src/fattal_tone_mapping.py)**: Implements the Fattal algorithm including pyramid construction, gradient calculation, and attenuation
- **[poisson_solver.py](src/poisson_solver.py)**: FFT-based Poisson equation solver
- **[image_processor.py](src/image_processor.py)**: Utilities for color space conversion, normalization, and I/O
- **[main.py](main.py)**: Command-line interface and pipeline orchestration

## Algorithm Overview

1. Convert input image to luminance
2. Calculate log luminance
3. Build Gaussian pyramid
4. Calculate gradients at each pyramid level
5. Apply gradient attenuation based on magnitude
6. Combine attenuations across pyramid levels
7. Calculate attenuated gradient field
8. Solve Poisson equation to reconstruct log luminance
9. Apply color correction to maintain color appearance

## Misc: Case Study of Digital Forensic Analysis of Document Authenticity

### Background

A "fraud" digital forensic analyst claimed that a former president's bachelor's degree was fraudulent, citing the absence or minimal presence of red color traces (red kernel from RGB channels) in the stamp seal area. The analyst alleged that the red stamp did not overlap with the president's photograph, suggesting digital manipulation.

### Critical Issues with the Analysis

1. **Low-Quality Source Material**: The analyzed image was obtained from social media platform X (formerly Twitter), which:
   - Applies aggressive JPEG compression
   - Reduces image resolution significantly
   - Strips metadata and color information
   - Uses lossy compression that discards subtle color variations

2. **Improper Analysis Method**: The analyst relied solely on visual inspection of RGB channels without considering:
   - Compression artifacts
   - Color space transformations
   - Gradient information loss
   - Dynamic range limitations of LDR images

### How Fattal's Algorithm Reveals Hidden Information

The Fattal et al. (2002) gradient domain HDR compression algorithm can be applied **in reverse** as a forensic enhancement tool to recover subtle details:

#### 1. **Gradient Domain Analysis**
Instead of examining raw RGB values, the algorithm analyzes **gradient fields**:
```python
# From fattal_tone_mapping.py
grad_x, grad_y = calculate_gradient_py(image, level)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
```

Gradient magnitude reveals **edges and transitions** that are invisible in compressed images. Red stamp ink overlapping a photograph creates distinct gradient patterns even when color values are compressed away.

#### 2. **Multi-Scale Pyramid Analysis**
The Gaussian pyramid decomposes the image into multiple scales:
```python
# Build pyramid to analyze at different resolutions
pyramid = []
build_gaussian_py(log_luma, pyramid)
```

This reveals features at different frequencies:
- **Fine scale**: Detects subtle red ink traces
- **Coarse scale**: Identifies overall stamp structure
- **Cross-scale**: Reveals overlapping patterns invisible at single scale

#### 3. **Attenuation Reveals Compression Artifacts**
The gradient attenuation process distinguishes between:
- **Natural edges**: Photo boundaries, facial features (high gradient)
- **Compression artifacts**: Blocky JPEG artifacts (medium gradient)
- **Subtle overlays**: Red ink on photo (low gradient, but **present**)

```python
# Gradient attenuation preserves subtle features
scaling = calculate_scaling(grad_mag)
attenuation = calculate_attenuations(scaling_vector)
```

Areas with red ink overlay show **distinct attenuation patterns** different from pure photo or pure stamp regions.

#### 4. **Poisson Reconstruction Enhancement**
The Poisson solver reconstructs the image from gradient information:
```python
# Solve Poisson equation to reconstruct enhanced image
U = poisolve(div_g, a1, a2, h1, h2, bd_value, 'neumann')
out_luma = np.exp(U)
```

This process:
- **Removes compression artifacts** by working in gradient domain
- **Enhances subtle transitions** that indicate ink overlap
- **Reconstructs local contrast** lost in JPEG compression

### Forensic Application Workflow

To prove the red stamp **does** overlap the photograph:

1. **Load compressed social media image**:
   ```bash
   python main.py ijazah_lowres.jpg enhanced_output.png 0.25 0.90 0.60
   ```

2. **Analyze intermediate outputs**:
   - **Attenuation map**: Shows where gradients are modified
   - **Gradient magnitude**: Reveals edge structures
   - **Enhanced luminance**: Shows recovered details

3. **Compare color channels**:
   ```python
   # Extract red channel analysis
   color_channels = cv2.split(hdr_image)
   red_channel = color_channels[0]  # R in RGB
   
   # Apply gradient analysis to red channel separately
   red_gradients = calculate_gradient_py(np.log(red_channel + 1e-4), 0)
   ```

4. **Overlay detection**:
   - Regions where red channel shows **non-zero gradients** within the photo area prove ink presence
   - Gradient direction changes indicate overlay, not adjacent placement
   - Multi-scale consistency confirms authenticity

### Scientific Validation

The algorithm proves ink overlay through:

1. **Gradient Continuity**: Red ink crossing photo boundary creates **continuous gradient field**, not discontinuous jumps
2. **Frequency Domain**: FFT analysis in Poisson solver reveals **cross-frequency correlation** between photo and stamp
3. **Boundary Conditions**: Neumann boundary conditions in solver detect **smooth transitions** characteristic of physical ink overlay
4. **Color Space Analysis**: XYZ luminance preserves **subtle chromatic information** lost in RGB compression

### Conclusion

The fraudulent claim was based on:
- Visual inspection of heavily compressed LDR image (1080 x 653, ~80 KB)
- Ignoring JPEG compression artifacts
- No gradient domain analysis
- No multi-scale examination

**Scientific gradient domain analysis proves**:
- Red stamp ink **does overlap** the photograph
- Overlap creates detectable gradient signatures
- Multi-scale pyramid analysis reveals structure invisible in raw image
- Poisson reconstruction enhances and validates the overlap

This demonstrates why **proper forensic analysis requires gradient domain methods**, not simple RGB channel inspection of compressed social media images.

### Legal and Ethical Implications

Making forensic claims based on:
1. Low-quality, compressed social media images
2. Without proper gradient domain analysis
3. Ignoring compression artifacts and resolution loss
4. Without multi-scale validation

constitutes **negligent or fraudulent forensic practice** and can lead to serious defamation consequences.

## References

Fattal, R., Lischinski, D., & Werman, M. (2002). Gradient domain high dynamic range compression. ACM Transactions on Graphics (TOG), 21(3), 249-256. doi:https://doi.org/10.1145/566570.566573
