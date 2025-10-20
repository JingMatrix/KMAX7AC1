import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import math

# =============================================================================
# Exercice 3: Coalescent de Kingman
# =============================================================================


def ex3():
    print("--- Exercice 3: Coalescent de Kingman (TP) ---")

    def simulate_coalescent(n_initial, c=1.0):
        """Simule une trajectoire du coalescent de Kingman."""
        n_lineages = n_initial
        current_time = 0

        history_time = [0]
        history_lineages = [n_initial]

        while n_lineages > 1:
            rate = c * n_lineages * (n_lineages - 1) / 2
            waiting_time = npr.exponential(1 / rate)
            current_time += waiting_time
            n_lineages -= 1

            history_time.append(current_time)
            history_lineages.append(n_lineages)

        return np.array(history_time), np.array(history_lineages)

    # 5a. Tracer le graphe
    n = 10000
    c = 1.0
    time_hist, lineage_hist = simulate_coalescent(n, c)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(time_hist, lineage_hist)
    plt.xlabel("Temps (en remontant le passé)")
    plt.ylabel("Nombre de lignées ancestrales")
    plt.title(f"Trajectoire du coalescent de Kingman (n={n})")
    plt.grid(True)
    print("Observation (5a): La coalescence est très rapide au début puis ralentit considérablement.")

    # 5b. Estimer E(T_MRCA)
    n_simulations = 5000
    n_small = 100  # Utiliser un n plus petit pour des simulations rapides
    tmrca_samples = []
    for _ in range(n_simulations):
        t, _ = simulate_coalescent(n_small, c)
        tmrca_samples.append(t[-1])

    estimated_mean_tmrca = np.mean(tmrca_samples)
    theoretical_mean_tmrca = (2 / c) * (1 - 1 / n_small)

    print(f"\nPour n={n_small}:")
    print(f"  E(T_MRCA) estimée: {estimated_mean_tmrca:.4f}")
    print(f"  E(T_MRCA) théorique: {theoretical_mean_tmrca:.4f}")

    # 5c. Distribution empirique de T_MRCA
    plt.subplot(1, 2, 2)
    for n_val in [10, 50, 200]:
        samples = [simulate_coalescent(n_val, c)[0][-1] for _ in range(2000)]
        plt.hist(samples, bins=50, density=True, alpha=0.6, label=f'n={n_val}')

    plt.title("$T_{MRCA}$ pour différentes valeurs de n")
    plt.xlabel("Temps")
    plt.legend()
    plt.show()
    print("Observation (5c): La distribution converge lorsque n augmente. La moyenne se stabilise vers 2/c.")


# =============================================================================
# Exercice 4: File d’attente M/M/1
# =============================================================================
def ex4():
    print("\n--- Exercice 4: File d’attente M/M/1/K (TP) ---")

    def simulate_mm1k(lam, mu, K, max_time):
        """Simule une file d'attente M/M/1/K."""
        n_clients = 0
        current_time = 0

        history_time = [0]
        history_clients = [0]
        clients_refuses = 0
        clients_arrives = 0

        while current_time < max_time:
            if n_clients == 0:
                rate = lam
                dt = npr.exponential(1 / rate)
                n_clients += 1  # Seule une arrivée est possible
                clients_arrives += 1
            elif n_clients == K:
                rate = mu
                dt = npr.exponential(1 / rate)
                # Un client qui arriverait pendant ce temps est refusé
                # On approxime en comptant le nb d'arrivées attendues
                if npr.rand() < lam / (lam + mu):  # Proba qu'une arrivée se produise avant un service
                    clients_refuses += 1
                n_clients -= 1  # Seul un départ est possible
            else:  # 0 < n_clients < K
                rate = lam + mu
                dt = npr.exponential(1 / rate)
                if npr.rand() < lam / (lam + mu):
                    n_clients += 1
                    clients_arrives += 1
                else:
                    n_clients -= 1

            current_time += dt
            history_time.append(current_time)
            history_clients.append(n_clients)

        return np.array(history_time), np.array(history_clients), clients_refuses, clients_arrives

    lam, mu, K = 0.8, 1.0, 10
    rho = lam / mu
    max_time = 10000

    time_hist, client_hist, refuses, arrives = simulate_mm1k(
        lam, mu, K, max_time)

    # Distribution stationnaire
    pi_empirique = np.bincount(client_hist, minlength=K+1) / len(client_hist)

    C = (1 - rho) / (1 - rho**(K + 1))
    pi_theorique = C * (rho**np.arange(K + 1))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time_hist, client_hist)
    plt.title("Évolution du nombre de clients")
    plt.xlabel("Temps")
    plt.ylabel("Nombre de clients")

    plt.subplot(1, 2, 2)
    plt.bar(np.arange(K+1) - 0.2, pi_empirique, width=0.4, label='Empirique')
    plt.bar(np.arange(K+1) + 0.2, pi_theorique, width=0.4, label='Théorique')
    plt.title("Distribution stationnaire $\pi^{(K)}$")
    plt.legend()
    plt.show()

    prob_refus_estimee = refuses / (refuses + arrives)
    prob_refus_theorique = pi_theorique[K]

    print(f"Probabilité de refus (estimée): {prob_refus_estimee:.4f}")
    print(
        f"Probabilité de refus (théorique = pi_K): {prob_refus_theorique:.4f}")


# =============================================================================
# Exercice 5: Monopoly
# =============================================================================
def ex5():
    print("\n--- Exercice 5: Monopoly (TP) ---")

    # --- Simulation Core ---

    JAIL = 10
    GO_TO_JAIL = 30

    def roll_dice():
        d1, d2 = npr.randint(1, 7), npr.randint(1, 7)
        return d1 + d2, d1 == d2

    def take_turn(position, turns_in_jail):
        visited_cases = []

        # --- Logic for being in jail ---
        if turns_in_jail > 0:
            total, is_double = roll_dice()
            if is_double or turns_in_jail == 3:
                position = (JAIL + total) % 40
                visited_cases.append(position)
                turns_in_jail = 0
            else:
                turns_in_jail += 1
            return position, turns_in_jail, visited_cases

        # --- Logic for a normal turn ---
        doubles_count = 0
        while True:
            total, is_double = roll_dice()
            if is_double:
                doubles_count += 1
            else:
                doubles_count = 0

            if doubles_count == 3:
                position = JAIL
                turns_in_jail = 1
                visited_cases.append(position)
                break

            position = (position + total) % 40
            visited_cases.append(position)

            if position == GO_TO_JAIL:
                position = JAIL
                turns_in_jail = 1
                visited_cases.append(position)  # Technically don't stay at 30
                break

            if not is_double:
                break

        return position, turns_in_jail, visited_cases

    # --- Simulation Run ---
    n_tours = 500000
    start_pos_counts = np.zeros(40)
    visit_counts = np.zeros(40)

    pos, jail_turns = 0, 0
    for _ in range(n_tours):
        start_pos_counts[pos] += 1
        pos, jail_turns, visited = take_turn(pos, jail_turns)
        for case in visited:
            visit_counts[case] += 1

    # --- Analysis ---
    pi_start = start_pos_counts / n_tours
    visits_per_tour = visit_counts / n_tours

    print("\n[Q2] 5 cases de DÉBUT de tour les plus fréquentes:")
    top5_start = np.argsort(pi_start)[-5:][::-1]
    for case in top5_start:
        print(f"  Case {case}: {pi_start[case]*100:.2f}%")

    print("\n[Q3] 5 cases les plus VISITÉES par tour:")
    top5_visited = np.argsort(visits_per_tour)[-5:][::-1]
    for case in top5_visited:
        print(f"  Case {case}: {visits_per_tour[case]:.3f} visites/tour")

    # --- Plotting ---
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(40), pi_start)
    plt.title("Distribution stationnaire (case de début de tour)")

    plt.subplot(1, 2, 2)
    plt.bar(range(40), visits_per_tour)
    plt.title("Fréquence de visite par case")
    plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    # These are optional TPs, run them one by one if desired
    ex3()
    ex4()
    ex5()
