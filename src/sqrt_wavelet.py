import numpy as np
from typing import Callable, Optional, Tuple


def _morlet_wavelet(u: np.ndarray, w: float = 5.0, sigma: float = 1.0) -> np.ndarray:
    """Simple Morlet-like wavelet function for scalar/array *u*.

    ψ(u) = exp(-u² / (2 σ²)) · cos(w u)
    """
    return np.exp(-u ** 2 / (2 * sigma ** 2)) * np.cos(w * u)


class SqrtWaveletTransform:
    """Implements the specialised transform

    W(k, τ) = ∫₀^∞ V(t) ψ(√t / τ) e^{-i k √t} dt

    The integral is discretised using trapezoidal rule on the sampled signal.
    """

    def __init__(self,
                 sampling_rate: float = 1.0,
                 wavelet_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 taus: Optional[np.ndarray] = None,
                 ks: Optional[np.ndarray] = None):
        """Parameters
        ----------
        sampling_rate
            Samples per second of *V(t)*.
        wavelet_func
            Callable ψ(u).  Defaults to a Morlet-like function.
        taus
            1-D array of scale values τ.  Defaults to log-space 0.1–10 (32 steps).
        ks
            1-D array of *k* (spatial-frequency) values.  Defaults to linear 0.1–10 (64 steps).
        """
        self.sampling_rate = sampling_rate
        self.wavelet_func = wavelet_func if wavelet_func is not None else _morlet_wavelet
        self.taus = taus if taus is not None else np.logspace(-1, 1, 32)  # 0.1 … 10
        self.ks = ks if ks is not None else np.linspace(0.1, 10.0, 64)

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------
    def transform(self, signal_data: np.ndarray) -> np.ndarray:
        """Compute W(k, τ) for a 1-D signal.

        Returns
        -------
        coeffs : ndarray, shape (len(τ), len(k))
            Complex-valued transform coefficients.
        """
        # Ensure float64 for numerical stability
        signal_data = np.asarray(signal_data, dtype=np.float64)
        n = signal_data.size
        if n == 0:
            raise ValueError("signal_data must contain at least one sample")

        dt = 1.0 / self.sampling_rate
        t = np.arange(n) * dt  # start at t=0
        sqrt_t = np.sqrt(t)

        # Pre-compute ψ(√t / τ) for all τ
        psi_mat = self.wavelet_func(sqrt_t[None, :] / self.taus[:, None])  # (τ, t)

        # Weight by signal once to save multiplications in k-loop
        base_mat = psi_mat * signal_data[None, :]

        # Allocate output (τ, k)
        coeffs = np.zeros((self.taus.size, self.ks.size), dtype=np.complex128)

        for j, k in enumerate(self.ks):
            exp_vec = np.exp(-1j * k * sqrt_t)  # (t,)
            integrand = base_mat * exp_vec[None, :]
            # Trapezoidal integration along time axis
            coeffs[:, j] = np.trapz(integrand, t, axis=1)

        return coeffs

    # ------------------------------------------------------------------
    # Convenience wrapper similar to EnhancedWaveletAnalysis
    # ------------------------------------------------------------------
    def analyze_signal(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run transform and derive magnitude & phase.

        Returns
        -------
        coeffs : complex ndarray (τ, k)
        magnitude : float ndarray (τ, k)
        phase : float ndarray (τ, k) in radians
        """
        coeffs = self.transform(signal_data)
        magnitude = np.abs(coeffs)
        phase = np.angle(coeffs)
        return coeffs, magnitude, phase 