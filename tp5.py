import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import math

# =============================================================================
# Exercice 1: Simulation d'un processus de Poisson
# =============================================================================


def ex1():
    print("--- Exercice 1: Simulation d'un processus de Poisson ---")
    lam = 2.0  # Intensité du processus

    # --- Q1: Simulation des n premiers sauts ---
    def simulate_n_jumps(n, lam):
        inter_arrival_times = npr.exponential(1.0/lam, n)
        jump_times = np.cumsum(inter_arrival_times)
        return jump_times

    # --- Q2: Loi du n-ième saut (Loi Gamma/Erlang) ---
    n = 5
    n_sims = 10000
    s_n_samples = [simulate_n_jumps(n, lam)[-1] for _ in range(n_sims)]

    plt.figure(figsize=(18, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax1.hist(s_n_samples, bins=50, density=True,
             label=f'Histogramme empirique de $S_{n}$')
    # Comparaison avec la densité théorique Gamma(n, lambda)
    from scipy.stats import gamma
    x = np.linspace(0, max(s_n_samples), 200)
    pdf_gamma = gamma.pdf(x, a=n, scale=1.0/lam)
    ax1.plot(x, pdf_gamma, 'r-', lw=2,
             label=f'Densité théorique Gamma({n}, {lam})')
    ax1.set_title(f"Distribution du {n}-ième saut")
    ax1.legend()

    # --- Q3 & Q4: Simulation jusqu'à t et loi de N_t ---
    def simulate_until_t(t, lam):
        jump_times = []
        current_time = 0
        while True:
            dt = npr.exponential(1.0/lam)
            current_time += dt
            if current_time > t:
                break
            jump_times.append(current_time)
        return jump_times, len(jump_times)

    t_fixed = 4.0
    n_t_samples = [simulate_until_t(t_fixed, lam)[1] for _ in range(n_sims)]

    ax2 = plt.subplot(1, 3, 2)
    bins = np.arange(0, max(n_t_samples) + 2) - 0.5
    ax2.hist(n_t_samples, bins=bins, density=True,
             label=f'Histogramme de $N_t$')
    # Comparaison avec la PMF de Poisson(lambda*t)
    from scipy.stats import poisson
    k = np.arange(0, max(n_t_samples) + 1)
    pmf_poisson = poisson.pmf(k, mu=lam * t_fixed)
    ax2.plot(k, pmf_poisson, 'ro',
             label=f'PMF théorique Poisson({lam*t_fixed:.1f})')
    ax2.set_title(f"Distribution de $N_t$ pour t={t_fixed}")
    ax2.legend()

    # --- Q5 & Q6: Trajectoire et convergence de N_t/t ---
    T_max = 500
    jumps, _ = simulate_until_t(T_max, lam)

    # Q5: Trajectoire
    ax3 = plt.subplot(1, 3, 3)
    t_plot = np.concatenate(([0], jumps, [T_max]))
    n_plot = np.concatenate(([0], np.arange(1, len(jumps) + 1), [len(jumps)]))
    ax3.step(t_plot, n_plot, where='post', label='Trajectoire de $N_t$')
    ax3.set_xlim(0, 50)  # Zoom sur le début
    ax3.set_ylim(bottom=0)
    ax3.set_title("Trajectoire d'un processus de Poisson")

    # Q6: Convergence N_t / t
    # On utilise les points juste après les sauts pour le calcul
    t_conv = np.array(jumps)
    n_conv = np.arange(1, len(jumps) + 1)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t_conv, n_conv / t_conv, 'r--', alpha=0.7, label='$N_t/t$')
    ax3_twin.axhline(lam, color='g', linestyle=':',
                     lw=2, label=f'$\lambda={lam}$')
    ax3_twin.set_ylabel("$N_t/t$")
    ax3_twin.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


# =============================================================================
# Exercice 5: Files d'attente M/M/1
# =============================================================================
def ex5():
    print("\n--- Exercice 5: Simulation d'une file d'attente M/M/1 ---")

    # --- Q5: Fonction de simulation (Gillespie) ---
    def simulate_queue(lam, mu, T_max):
        t = 0
        k = 0  # Nombre de clients
        times = [0]
        counts = [0]

        while t < T_max:
            if k == 0:
                rate = lam
                dt = npr.exponential(1.0 / rate)
                k_new = 1
            else:
                rate = lam + mu
                dt = npr.exponential(1.0 / rate)
                if npr.rand() < lam / rate:
                    k_new = k + 1  # Arrivée
                else:
                    k_new = k - 1  # Départ

            t += dt
            k = k_new
            times.append(t)
            counts.append(k)

        return np.array(times), np.array(counts)

    # --- Q6: Observer les 3 régimes ---
    plt.figure(figsize=(15, 5))

    # Cas stable: lambda < mu
    ax1 = plt.subplot(1, 3, 1)
    lam_s, mu_s = 2.0, 3.0
    t_s, k_s = simulate_queue(lam_s, mu_s, 200)
    ax1.plot(t_s, k_s)
    ax1.set_title(f'Cas stable ($\lambda={lam_s} < \mu={mu_s}$)')
    ax1.grid(True)

    # Cas critique: lambda = mu
    ax2 = plt.subplot(1, 3, 2)
    lam_c, mu_c = 2.0, 2.0
    t_c, k_c = simulate_queue(lam_c, mu_c, 200)
    ax2.plot(t_c, k_c)
    ax2.set_title(f'Cas critique ($\lambda={lam_c} = \mu={mu_c}$)')
    ax2.grid(True)

    # Cas explosif: lambda > mu
    ax3 = plt.subplot(1, 3, 3)
    lam_e, mu_e = 3.0, 2.0
    t_e, k_e = simulate_queue(lam_e, mu_e, 200)
    ax3.plot(t_e, k_e)
    ax3.set_title(f'Cas instable ($\lambda={lam_e} > \mu={mu_e}$)')
    ax3.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Q7: Limite de W_t/t pour le cas instable ---
    T_long = 1000
    t_e_long, k_e_long = simulate_queue(lam_e, mu_e, T_long)
    plt.figure(figsize=(12, 5))
    ax_q7 = plt.subplot(1, 2, 1)
    ax_q7.plot(t_e_long, k_e_long / t_e_long, label='$W_t/t$')
    ax_q7.axhline(lam_e - mu_e, color='r', linestyle='--',
                  label=f'$\lambda-\mu = {lam_e - mu_e:.1f}$')
    ax_q7.set_title('Convergence de $W_t/t$ dans le cas instable')
    ax_q7.set_xlabel('Temps (t)')
    ax_q7.set_ylim(bottom=0)
    ax_q7.legend()
    ax_q7.grid(True)

    # --- Q8: Distribution stationnaire pour le cas stable ---
    T_long = 50000
    t_s_long, k_s_long = simulate_queue(lam_s, mu_s, T_long)

    # Calcul de la proportion de temps passé dans chaque état
    time_in_state = np.zeros(int(k_s_long.max()) + 1)
    durations = np.diff(t_s_long)
    states_visited = k_s_long[:-1]

    for i, state in enumerate(states_visited):
        time_in_state[state] += durations[i]

    pi_empirique = time_in_state / t_s_long[-1]

    # Théorique
    rho = lam_s / mu_s
    k_th = np.arange(len(pi_empirique))
    pi_theorique = (1 - rho) * (rho**k_th)

    ax_q8 = plt.subplot(1, 2, 2)
    ax_q8.bar(k_th, pi_empirique, label='Proportion de temps (empirique)')
    ax_q8.plot(k_th, pi_theorique, 'ro-', label='Loi stationnaire (théorique)')
    ax_q8.set_title('Vérification de la loi stationnaire ($\lambda < \mu$)')
    ax_q8.set_xlim(right=15)  # Zoom
    ax_q8.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    ex1()
    ex5()
