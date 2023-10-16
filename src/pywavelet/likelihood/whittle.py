"""Whittle Likelihood (in the wavelet domain)
See eq 18, Cornish 2020 "Time-Frequency Analysis of Gravitational Wave Data"

LnL(d|h) = -1/2 * Sum_{ti,fi} [ ln(2pi) + ln(PSD[ti,fi]) + (d[ti,fi]-h[ti,fi])^2/PSD[ti,fi]]

where
- d[ti,fi] is the data in the wavelet domain,
- h[ti,fi] is the model in the wavelet domain,
- PSD[ti,fi] is the _evolutionary_ power spectral density in the wavelet domain.

For stationary noise:
PSD[ti,fi] = PSD[fi * Nt/2] Delta_f,
where
- fi * Nt/2 is the frequency index in the wavelet domain,
- Delta_f is the frequency bin width in the wavelet domain.


"""

from .likelihood_base import LikelihoodBase

class WhittleLikelihood(LikelihoodBase):
    pass


