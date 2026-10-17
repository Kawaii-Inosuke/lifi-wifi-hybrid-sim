"""Bit-error rate of OOK and unipolar M-PAM: closed-form approximations and Monte Carlo."""
from __future__ import annotations

import numpy as np
from scipy.special import erfc


def qfunc(x: np.ndarray) -> np.ndarray:
    """Gaussian Q function."""
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2))


def ber_pam(snr: np.ndarray, M: int) -> np.ndarray:
    """BER of Gray-coded unipolar M-PAM (M=2 is OOK) at electrical SNR (linear).

    SNR = (mean photocurrent)^2 / noise variance. Levels 0 .. 2a are equally spaced,
    so the half-distance is a/(M-1). BER ~ SER/log2(M) (nearest-neighbour approximation).
    """
    k = np.log2(M)
    return 2 * (M - 1) / (M * k) * qfunc(np.sqrt(snr) / (M - 1))


def ber_ook(snr: np.ndarray) -> np.ndarray:
    """OOK BER = Q(sqrt(SNR))."""
    return ber_pam(snr, 2)


def monte_carlo_pam(snr_db: np.ndarray, M: int, n_bits: int, rng: np.random.Generator
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Simulate unipolar Gray-coded M-PAM over AWGN. Returns (BER, bit-error counts).

    Per SNR point it draws rng.integers(0, M, n_sym) then rng.normal(0, 1, n_sym).
    For M=2 this is exactly the draw order of the prototype's OOK validation.
    """
    k = int(np.log2(M))
    n_sym = n_bits // k
    ber, errs = [], []
    for s in np.atleast_1d(snr_db):
        a = np.sqrt(10 ** (s / 10)) * 1.0           # mean amplitude, sigma = 1
        sym = rng.integers(0, M, n_sym)
        noise = rng.normal(0, 1.0, n_sym)
        step = 2 * a / (M - 1)                       # spacing between levels
        rx = step * sym + noise
        if M == 2:
            sym_hat = (rx > a).astype(sym.dtype)     # threshold at the mean, as prototype
        else:
            sym_hat = np.clip(np.rint(rx / step), 0, M - 1).astype(sym.dtype)
        diff = (sym ^ (sym >> 1)) ^ (sym_hat ^ (sym_hat >> 1))   # Gray-code bit differences
        e = int(sum(((diff >> b) & 1).sum() for b in range(k)))
        errs.append(e)
        ber.append(e / (n_sym * k))
    return np.array(ber), np.array(errs)
