import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

# =============================================================================
# Exercice 2: Méthode de Métropolis-Hastings sur {1,2,3}
# =============================================================================


def ex2():
    print("--- Exercice 2: MH sur {1,2,3} ---")

    # --- Q2: Fonction pour simuler une chaîne de Markov ---
    def simulate_markov_chain(Q, n_steps, x0):
        X = np.zeros(n_steps, dtype=int)
        X[0] = x0
        for n in range(n_steps - 1):
            X[n+1] = npr.choice([0, 1, 2], p=Q[X[n], :])
        return X

    # --- Q4a: Implémentation de l'algorithme ---
    def metropolis_hastings_discrete(Q, nu, n_steps, x0):
        X = np.zeros(n_steps, dtype=int)
        X[0] = x0

        for n in range(n_steps - 1):
            current_x = X[n]

            # Proposer un nouvel état Y
            Y = npr.choice([0, 1, 2], p=Q[current_x, :])

            # Calculer le ratio d'acceptation
            ratio = (nu[Y] * Q[Y, current_x]) / \
                (nu[current_x] * Q[current_x, Y])
            alpha = min(1, ratio)

            # Accepter ou rejeter
            if npr.rand() < alpha:
                X[n+1] = Y
            else:
                X[n+1] = current_x

        return X

    # --- Q4: Simulation et analyse ---
    nu = np.array([3/5, 1/10, 3/10])
    Q1 = np.array([[0, 0.5, 0.5], [0.6, 0.3, 0.1], [0.3, 0.4, 0.3]])
    Q2 = np.array([[0, 0.3, 0.7], [0.3, 0.4, 0.3], [0.7, 0.3, 0.0]])
    Q3 = np.array([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])

    # --- Q4b: Vérification numérique par histogramme ---
    n_long = 100000
    traj_q2 = metropolis_hastings_discrete(Q2, nu, n_long, x0=0)

    plt.figure(figsize=(12, 9))
    plt.hist(traj_q2, bins=[-0.5, 0.5, 1.5, 2.5],
             density=True, rwidth=0.4, label='Fréquence empirique')
    plt.plot([0, 1, 2], nu, 'ro', label='Loi cible $\\nu$')
    plt.xticks([0, 1, 2])
    plt.title("Vérification de la mesure invariante pour Q2")
    plt.legend()
    plt.show()

    n_steps_total = 300
    n_sims_for_avg = 50  # Pour moyenner le bruit

    plt.figure(figsize=(10, 6))

    for Q, label in [(Q1, 'Q1'), (Q2, 'Q2'), (Q3, 'Q3')]:
        print(f"Simulation avec la matrice {label}...")

        # --- Q4c: Calcul de la distance en variation totale ---
        # On fait la moyenne sur plusieurs simulations pour lisser les courbes
        dvt_history_avg = np.zeros(n_steps_total)

        for _ in range(n_sims_for_avg):
            traj = metropolis_hastings_discrete(Q, nu, n_steps_total, x0=0)
            dvt_history_run = []
            for n in range(1, n_steps_total + 1):
                # Calculer la mesure empirique
                hist = np.bincount(traj[:n], minlength=3)
                mu_empirique = hist / n

                # Calculer la distance VT
                # Division par 2 est convention
                dvt = 0.5 * np.sum(np.abs(mu_empirique - nu))
                dvt_history_run.append(dvt)

            dvt_history_avg += np.array(dvt_history_run)

        dvt_history_avg /= n_sims_for_avg
        plt.plot(range(1, n_steps_total + 1),
                 dvt_history_avg, label=f'Matrice {label}')

    plt.title("Convergence en Variation Totale pour différentes matrices Q")
    plt.xlabel("Nombre d'itérations (n)")
    plt.ylabel("Distance en Variation Totale (moyennée)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

    # Intuition sur la convergence (Q4c):
    # - Q3 est un mauvais noyau de proposition. Il force des sauts vers l'état 1
    #   (faible proba cible), ce qui entraîne de nombreux rejets.
    #   La chaîne explore l'espace très lentement.
    # - Q2 est symétrique, ce qui en fait un bon candidat.
    # - Q1 propose des sauts qui peuvent aller 'contre' la mesure cible
    #   (ex: de 0 à 1 où nu(1) est faible).
    # On s'attend à ce que Q2 converge le mieux.

# =============================================================================
# Exercice 3: MH pour des variables à densité
# =============================================================================


def ex3():
    print("\n--- Exercice 3: MH pour une densité continue ---")
    alpha = 2.0

    def mu_alpha(x): return (1/np.sqrt(2*np.pi)) * \
        np.exp(-x**2/2) * (1 + np.sin(alpha * x))
    n_samples = 1000

    # --- Q2: Tracer la densité ---
    x_grid = np.linspace(-4, 4, 400)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_grid, mu_alpha(x_grid), label=f'$\\mu_{{{alpha}}}(x)$')
    plt.plot(x_grid, norm.pdf(x_grid), 'r--', label='N(0,1) pour comparaison')
    plt.title("Densité cible")
    plt.legend()

    # --- Q3 & Q6: Méthode du Rejet ---
    def rejection_sampling(n_target):
        samples = []
        start_time = time.time()
        while len(samples) < n_target:
            Y = npr.randn()  # Proposition N(0,1)
            U = npr.rand()
            if U <= (1 + np.sin(alpha * Y)) / 2:
                samples.append(Y)
        end_time = time.time()
        return np.array(samples), end_time - start_time

    # --- Q4 & Q5: Algorithme de Metropolis-Hastings ---
    def metropolis_hastings_continuous(n_target, burn_in=500):
        samples = np.zeros(n_target + burn_in)
        samples[0] = 0.0  # Start at 0
        start_time = time.time()

        for i in range(n_target + burn_in - 1):
            current_x = samples[i]
            # Proposition symétrique: Y = x + Z, Z ~ N(0,1)
            Y = current_x + npr.randn()

            # Ratio d'acceptation (simplifié pour Q symétrique)
            ratio = mu_alpha(Y) / mu_alpha(current_x)
            alpha_acc = min(1, ratio)

            if npr.rand() < alpha_acc:
                samples[i+1] = Y
            else:
                samples[i+1] = current_x

        end_time = time.time()
        return samples[burn_in:], end_time - start_time

    # --- Q5 & Q6: Comparaison des temps ---
    mh_samples, t_mh = metropolis_hastings_continuous(n_samples)
    rej_samples, t_rej = rejection_sampling(n_samples)

    print(f"Temps de calcul pour {n_samples} échantillons:")
    print(f"  Méthode du Rejet: {t_rej:.4f}s")
    print(f"  Metropolis-Hastings: {t_mh:.4f}s")

    plt.subplot(1, 2, 2)
    plt.hist(mh_samples, bins=50, density=True, alpha=0.7, label='MH Samples')
    plt.hist(rej_samples, bins=50, density=True,
             alpha=0.7, label='Rejection Samples')
    plt.plot(x_grid, mu_alpha(x_grid), 'k-', lw=2, label='Densité théorique')
    plt.title("Histogramme des échantillons générés")
    plt.legend()
    plt.show()

    # --- Q9: Comparaison des erreurs quadratiques ---
    print("\n[Question 9] Comparaison de l'erreur quadratique moyenne (MSE)")
    val_theorique = alpha * np.exp(-alpha**2 / 2)
    n_estimations = 1000

    # Temps de calcul de référence
    n_mh = 2000  # On garde MH comme référence
    _, t_ref = metropolis_hastings_continuous(n_mh)

    # Estimer le nombre d'itérations/samples pour un temps équivalent
    n_rej = int(n_samples * t_ref / t_rej)

    print(
        f"Pour un temps de calcul équivalent (~{t_ref:.2f}s), nous prenons :")
    print(f" - {n_rej} échantillons pour la méthode du Rejet.")
    print(f" - {n_mh} échantillons pour la méthode MH.")

    estimations_rej = []
    estimations_mh = []

    for _ in range(n_estimations):
        samples_rej, _ = rejection_sampling(n_rej)
        estimations_rej.append(np.mean(samples_rej))

        samples_mh, _ = metropolis_hastings_continuous(n_mh, burn_in=200)
        estimations_mh.append(np.mean(samples_mh))

    mse_rej = np.mean((np.array(estimations_rej) - val_theorique)**2)
    mse_mh = np.mean((np.array(estimations_mh) - val_theorique)**2)

    print(f"\nErreur Quadratique Moyenne (MSE) sur {n_estimations} runs:")
    print(f"  Estimateur par Rejet: {mse_rej:.4e}")
    print(f"  Estimateur par MH:    {mse_mh:.4e}")
    print(f"Ratio des MSE (MSE_MH / MSE_Rejet): {mse_mh/mse_rej:.2f}")

# =============================================================================
# Exercice 4: Pièce d'échec
# =============================================================================


def ex4():
    print("\n--- Exercice 4: Déplacement du cavalier ---")

    def get_knight_moves(pos):
        """Retourne la liste des mouvements possibles pour un cavalier."""
        x, y = pos
        moves = []
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                moves.append((nx, ny))
        return moves

    def simulate_knight_mh(n_steps):
        """Simule la marche du cavalier avec l'algorithme MH."""
        pos = (0, 0)  # Départ en (0,0)
        history = np.zeros((n_steps, 2))

        for i in range(n_steps):
            history[i, :] = pos

            # Mouvements possibles et leur nombre
            possible_moves = get_knight_moves(pos)
            d_x = len(possible_moves)

            # Proposition: choisir un mouvement uniformément
            next_pos_proposal = possible_moves[npr.randint(d_x)]

            # Nombre de mouvements depuis la position proposée
            d_y = len(get_knight_moves(next_pos_proposal))

            # Taux d'acceptation MH
            alpha = min(1, d_x / d_y)

            if npr.rand() < alpha:
                pos = next_pos_proposal

        return history

    n_steps = 500000
    burn_in = 2000
    trajectory_tuples = simulate_knight_mh(n_steps)

    final_trajectory = trajectory_tuples[burn_in:]

    # Calculer la distribution stationnaire empirique
    board_counts = np.zeros((8, 8))
    # On ignore le début (burn-in)
    for x, y in final_trajectory:
        board_counts[int(x), int(y)] += 1

    pi_empirique = board_counts / np.sum(board_counts)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(pi_empirique, cmap='viridis', origin='lower')
    plt.colorbar(label="Fréquence de visite")
    plt.title("Distribution stationnaire du cavalier (MH)")

    plt.subplot(1, 2, 2)
    trajectory_1d = [x * 8 + y for x, y in final_trajectory]
    plt.hist(trajectory_1d, bins=np.arange(65) -
             0.5, density=True, label='Fréquence empirique')
    uniform_freq = 1 / 64
    plt.axhline(uniform_freq, color='r', linestyle='--',
                label=f'Fréquence uniforme (1/64 = {uniform_freq:.4f})')
    plt.title("Histogramme des visites par case")

    plt.show()

    print("La distribution obtenue par MH est quasi-uniforme, comme attendu.")
    print(f"Variance de la distribution empirique: {np.var(pi_empirique):.2e}")
    print(f"(Une variance faible indique une distribution proche de l'uniforme)")

# =============================================================================
# Exercice 5: Retour sur le sac à dos (Knapsack)
# =============================================================================


def ex5():
    print("\n--- Exercice 5: Sac à dos avec MCMC ---")

    # --- Partie 1 ---
    print("\n[Partie 1] Chaîne de Markov sur {0,1}^K")
    K = 4

    def simulate_C_chain(n_steps):
        C = np.zeros(K, dtype=int)
        history = np.zeros((n_steps, K))
        for i in range(n_steps):
            history[i, :] = C
            # Choisir un indice au hasard
            idx_to_flip = npr.randint(K)
            # Créer la nouvelle configuration
            C_new = C.copy()
            C_new[idx_to_flip] = 1 - C_new[idx_to_flip]
            C = C_new
        return history

    n_p1 = 50000
    traj_C = simulate_C_chain(n_p1)
    # Vérifier numériquement la limite pour C=(0,0,0,0)
    proportion_zeros = np.mean(np.all(traj_C == 0, axis=1))
    theorique_zeros = 1 / (2**K)
    print(f"Q4: Fréquence de visite de (0,..,0): {proportion_zeros:.4f}")
    print(f"    Valeur théorique 1/2^K: {theorique_zeros:.4f}")

    # --- Partie 2 & 3: MCMC pour le problème du sac à dos ---
    print("\n[Partie 2 & 3] MCMC sur l'espace des solutions acceptables")

    def run_knapsack_mcmc(weights, capacity, beta, n_steps, c0):
        K = len(weights)
        C = c0.copy()

        history = []
        current_weight = np.dot(C, weights)

        for _ in range(n_steps):
            history.append(C.copy())

            # Proposer une modification (noyau Q symétrique)
            idx_to_flip = npr.randint(K)
            C_proposal = C.copy()
            C_proposal[idx_to_flip] = 1 - C_proposal[idx_to_flip]

            proposal_weight = np.dot(C_proposal, weights)

            # --- Vérification de l'acceptabilité ---
            if proposal_weight > capacity:
                # Rejeter car la proposition est hors de l'espace A
                # La chaîne reste au même endroit (équivalent à un rejet MH)
                continue

            # --- Calcul de l'acceptation MH pour la mesure de Gibbs ---
            # Si beta=0, on simule l'uniforme sur A.
            # L'exponentielle est e^0 = 1, le ratio est 1, on accepte toujours.
            weight_diff = proposal_weight - current_weight
            alpha = min(1, np.exp(beta * weight_diff))

            if npr.rand() < alpha:
                C = C_proposal
                current_weight = proposal_weight

        return np.array(history)

    # --- Partie 3, Q4: Estimer A pour un petit exemple ---
    K_p3 = 8
    w_p3 = np.array([3, 3, 3, 5, 5, 5, 5, 5])
    cap_p3 = 9  # Poids max = 9, ex: (3,3,3) ou (3,5)
    beta_p3 = 10
    # Poids = 10 (non-acceptable pour cap=9), on part de 0
    c0_p3 = np.array([0, 0, 0, 1, 1, 0, 0, 0])
    c0_p3_valid = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    print(f"\n[Partie 3, Q4] Temps pour trouver le poids max M={cap_p3}")

    for beta_val in [0.01, 0.1, 1, 10]:
        n_tries = 100
        steps_to_find_max = []
        for _ in range(n_tries):
            # On simule jusqu'à trouver le max
            C = c0_p3_valid.copy()
            current_weight = 0
            steps = 0
            while current_weight < cap_p3 and steps < 5000:
                steps += 1
                idx = npr.randint(K_p3)
                C_prop = C.copy()
                C_prop[idx] = 1 - C_prop[idx]
                w_prop = np.dot(C_prop, w_p3)

                if w_prop > cap_p3:
                    continue

                alpha = min(1, np.exp(beta_val * (w_prop - current_weight)))
                if npr.rand() < alpha:
                    C = C_prop
                    current_weight = w_prop
            steps_to_find_max.append(
                steps if current_weight == cap_p3 else 5000)

        print(f"  Pour beta={beta_val}: "
              f"Nombre moyen d'étapes pour atteindre M={cap_p3} est ~{np.mean(steps_to_find_max):.0f}")

    # --- Partie 4: Recuit Simulé ---
    print("\n[Partie 4] Recuit simulé")

    def simulate_annealing(weights, capacity, beta_schedule, n_steps):
        K = len(weights)
        C = np.zeros(K, dtype=int)  # Start empty

        weight_history = np.zeros(n_steps)
        current_weight = 0

        for n in range(n_steps):
            beta = beta_schedule[n]

            # Propose move
            idx = npr.randint(K)
            C_prop = C.copy()
            C_prop[idx] = 1 - C_prop[idx]
            w_prop = np.dot(C_prop, weights)

            if w_prop <= capacity:
                # Accept if valid and based on MH rule
                delta_E = w_prop - current_weight  # We want to MAXIMIZE weight
                if npr.rand() < min(1, np.exp(beta * delta_E)):
                    C = C_prop
                    current_weight = w_prop

            weight_history[n] = current_weight
        return weight_history

    n_sa = 2000
    w_sa = np.array([3, 3, 3, 5, 5, 5, 5, 5])
    cap_sa = 10

    # Define beta schedules
    betas = {
        "exp": np.exp(np.linspace(0, 5, n_sa)),
        "linear": np.linspace(0.01, 10, n_sa),
        "log": np.log(np.arange(n_sa) + 2)
    }

    ax2 = plt.subplot(1, 2, 2)
    for name, schedule in betas.items():
        hist = simulate_annealing(w_sa, cap_sa, schedule, n_sa)
        ax2.plot(hist, label=f'$\\beta_n = {name}$')

    ax2.axhline(10, color='r', linestyle='--',
                label='Optimum global (Poids=10)')
    ax2.set_title("Recuit simulé : convergence du poids du sac")
    ax2.set_xlabel("Itération")
    ax2.set_ylabel("Poids du sac")
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # --- Partie 5: Bin Packing ---
    print("\n[Partie 5] Problème du Bin Packing")

    def solve_bin_packing_sa(weights, capacity, n_steps):
        K = len(weights)
        # State: an array where state[i] is the bin number for item i
        # Start with a random assignment
        state = npr.randint(K, size=K)

        # Energy: number of bins used
        def energy(s):
            return len(np.unique(s))

        # Check validity of a state
        def is_valid(s):
            bin_weights = np.zeros(K)  # Max possible bins is K
            for item_idx, bin_idx in enumerate(s):
                bin_weights[bin_idx] += weights[item_idx]
            return np.all(bin_weights <= capacity)

        current_energy = energy(state)
        best_state = state.copy()
        best_energy = current_energy

        beta_schedule = 0.5 * np.log(np.arange(n_steps) + 2)

        for n in range(n_steps):
            # Propose a move: move one item to a different bin
            proposal = state.copy()
            item_to_move = npr.randint(K)
            # Move to one of the existing bins or a new one
            target_bin = npr.randint(current_energy + 1)
            proposal[item_to_move] = target_bin

            if is_valid(proposal):
                proposal_energy = energy(proposal)
                delta_E = proposal_energy - current_energy

                if npr.rand() < min(1, np.exp(-beta_schedule[n] * delta_E)):
                    state = proposal
                    current_energy = proposal_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state = state.copy()

        return best_energy, best_state

    # Test on 100 random items
    w_bp = npr.randint(1, 11, size=100)
    cap_bp = 10

    print("Résolution du Bin Packing pour 100 objets aléatoires...")
    min_bins, final_config = solve_bin_packing_sa(w_bp, cap_bp, n_steps=50000)
    print(f"Le recuit simulé a trouvé une solution avec {min_bins} sacs.")
    # A lower bound for the number of bins is sum(weights)/capacity
    lower_bound = np.ceil(np.sum(w_bp)/cap_bp)
    print(f"Borne inférieure théorique : {lower_bound:.0f} sacs.")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    ex2()
    ex3()
    ex4()
    ex5()
