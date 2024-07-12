import matplotlib.pyplot as plt
import numpy as np

from pywavelet.transforms.phi_computer import phi_vec, phitilde_vec_norm


def test_phi(plot_dir):
    d, q = 4.0, 16
    Nf, Nt = 64, 64
    dt = 1

    phi = phi_vec(Nf=Nf, dt=dt, d=d, q=q)
    t = np.linspace(-q, q, len(phi))
    phitilde = phitilde_vec_norm(Nf=Nf, Nt=Nt, dt=dt, d=d)
    f = np.linspace(-1, 1, len(phitilde))
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 5),
    )
    ax[0].plot(t, phi )
    ax[0].set_ylabel("$\phi(t)/\Delta t$")
    ax[0].set_xlabel("$t/\Delta T$")

    ax[1].plot(f, phitilde)
    ax[1].set_ylabel("$\\tilde{\phi(f)}/\Delta t$")
    ax[1].set_xlabel("$f/\Delta F$")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/phi.png")
