import matplotlib.pyplot as plt
import numpy as np

from pywavelet.transforms import (
    phi_vec,
    phitilde_vec,
    phitilde_vec_norm,
)


def test_phi(plot_dir):
    d, q = 4.0, 16
    Nf, Nt = 3, 64
    dt = 2

    delT = Nf * dt
    delF = 1.0 / (2 * Nf * dt)

    phi = phi_vec(Nf=Nf, d=d, q=q) / delT
    t = np.linspace(-q, q, len(phi))
    f = np.linspace(-1, 1, 1000)
    phitilde = phitilde_vec(f, Nf=Nf, d=d) / delF

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
    plt.savefig(f"{plot_dir}/phi.png")
