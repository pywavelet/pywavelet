import matplotlib.pyplot as plt
import numpy as np

from pywavelet.transforms import omega, phi_vec, phitilde_vec_norm


def _plot(t, phi, f, phitilde, fname):
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(7, 5),
    )
    ax[0].plot(t, phi / max(phi))
    ax[0].set_ylabel("$\phi(t)/\Delta t$")
    ax[0].set_xlabel("$t/\Delta T$")

    ax[1].plot(f, phitilde / max(phitilde))
    ax[1].set_ylabel("$\\tilde{\phi(f)}/\Delta t$")
    ax[1].set_xlabel("$f/\Delta F$")
    plt.tight_layout()
    plt.savefig(fname)


def test_phi(plot_dir):
    d, q = 4.0, 16
    Nf, Nt = 3, 64
    dt = 2

    delT = Nf * dt
    delF = 1.0 / (2 * Nf * dt)

    phi = phi_vec(Nf=Nf, d=d, q=q) / delT
    phitilde = phitilde_vec_norm(Nf=Nf, Nt=Nt, d=d) / delF

    assert len(phi) == 2 * q * Nf, f"{len(phi)} != {2 * q * Nf}"
    assert len(phitilde) == Nt // 2 + 1, f"{len(phitilde)} != {Nt // 2 + 1}"

    t = np.linspace(-q, q, len(phi))
    f = omega(Nf, Nt) / delF

    _plot(t, phi, f, phitilde, f"{plot_dir}/phi.png")
