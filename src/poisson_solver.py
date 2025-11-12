import numpy as np
from scipy.fftpack import dctn, idctn

class PoissonSolver:
    @staticmethod
    def neumann_compat(F, a1, a2, h1, h2):
        """
        Calculate Neumann boundary compatibility value.
        For Neumann boundary conditions, the integral of F must be zero.
        This function calculates the mean that should be subtracted.
        """
        n1, n2 = F.shape
        
        # Calculate the mean of F
        mean_F = np.mean(F)
        
        # For Neumann boundaries, we need to subtract this mean
        # to ensure compatibility (integral of RHS = 0)
        return mean_F
    
    @staticmethod
    def poisolve(F, a1=1.0, a2=1.0, h1=1.0, h2=1.0, 
                 bound_value=0.0, boundary_type='neumann'):
        """
        Solve Poisson equation using FFT method.
        """
        n1, n2 = F.shape
        
        # Neumann compatibility
        rhs = F.copy()
        if boundary_type == 'neumann':
            # Subtract the mean to ensure compatibility
            rhs = rhs - bound_value
        
        # Apply DCT-I (REDFT00 in FFTW)
        # Note that: scipy's dctn with type=1 corresponds to FFTW's REDFT00
        rhs_transformed = dctn(rhs, type=1, norm=None)
        
        # Create eigenvalue grids
        i_indices = np.arange(n1)
        j_indices = np.arange(n2)
        
        # Eigenvalues for DCT-I (matching FFTW REDFT00)
        # lambda[k] = -2 * (1 - cos(pi * k / (n - 1)))
        lambda1 = -2.0 * (1.0 - np.cos(np.pi * i_indices / (n1 - 1)))
        lambda2 = -2.0 * (1.0 - np.cos(np.pi * j_indices / (n2 - 1)))
        
        # Create 2D grids
        L1, L2 = np.meshgrid(lambda2, lambda1, indexing='xy')
        
        # Calculate divisor
        div = (a1 * L1 / (h1 * h1)) + (a2 * L2 / (h2 * h2))
        
        # Solve U = rhs / div, handling the zero frequency term
        U = np.zeros_like(rhs_transformed)
        mask = np.abs(div) > 1e-15
        U[mask] = rhs_transformed[mask] / div[mask]
        U[~mask] = 0.0  # Set DC component to 0
        
        # Apply inverse DCT-I
        # FFTW normalization: forward and inverse each multiply by 2(n1-1)*2(n2-1)
        # Therefore, we need to divide by 4*(n1-1)*(n2-1)
        U = idctn(U, type=1, norm=None) / (4.0 * (n1 - 1) * (n2 - 1))
        
        return U.astype(np.float32)