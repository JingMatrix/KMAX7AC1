import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import math

# =============================================================================
# Exercice 1: Les parapluies de Cherbourg
# =============================================================================


def ex1():
    print("--- Exercice 1 ---")

    def simule_parapluies(n_steps, n_parapluies, p_pluie):
        X = np.zeros(n_steps, dtype=int)
        X[0] = n_parapluies  # On suppose qu'on part avec tous les parapluies
        wet_count = 0

        for n in range(n_steps - 1):
            pleut = (npr.rand() < p_pluie)

            # Matin ou soir
            if X[n] == 0:
                if pleut:
                    wet_count += 1
                X[n+1] = n_parapluies
            else:  # X[n] > 0
                if pleut:
                    X[n+1] = n_parapluies - X[n] + 1
                else:
                    X[n+1] = n_parapluies - X[n]
        return X, wet_count

    n_parapluies = 3
    p_pluie = 0.5
    n_steps = 100000

    # 2. Simulation
    traj, wet_count = simule_parapluies(n_steps, n_parapluies, p_pluie)

    # 4. Histogramme de la mesure invariante
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    pi_theorique = [1/7, 2/7, 2/7, 2/7]
    plt.hist(traj, bins=np.arange(n_parapluies + 2) -
             0.5, density=True, label='Empirique')
    plt.plot(range(n_parapluies + 1), pi_theorique, 'ro', label='Théorique')
    plt.title(f"Distribution stationnaire ({n_parapluies} parapluies)")
    plt.legend()

    # 6. Estimation numérique
    prob_wet_empirique = wet_count / n_steps
    jours_mouilles_an = prob_wet_empirique * 365 * 2
    print(
        f"Probabilité d'être mouillé (théorique): {pi_theorique[0] * p_pluie:.4f}")
    print(f"Probabilité d'être mouillé (estimée): {prob_wet_empirique:.4f}")
    print(f"Nombre de jours mouillés par an (estimé): {jours_mouilles_an:.2f}")

    # Graphe en fonction du nombre de parapluies
    plt.subplot(1, 2, 2)
    for p in [0.1, 0.3, 0.5, 0.7]:
        jours_mouilles_vs_N = []
        parapluie_range = range(1, 8)
        for n_p in parapluie_range:
            _, wc = simule_parapluies(n_steps, n_p, p)
            jours_mouilles_vs_N.append(wc / n_steps * 365 * 2)
        plt.plot(parapluie_range, jours_mouilles_vs_N, 'o-', label=f'p={p}')
    plt.title("Jours mouillés / an vs N parapluies")
    plt.xlabel("Nombre de parapluies")
    plt.ylabel("Nombre moyen de trajets mouillés par an")
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# Exercice 2: Équilibre thermodynamique
# =============================================================================


def ex2():
    print("\n--- Exercice 2 ---")
    N = 100
    n_steps = 2000
    X = np.zeros(n_steps, dtype=int)
    X[0] = N  # Toutes les molécules à gauche

    for t in range(n_steps - 1):
        # Choisir une molécule au hasard
        mol_id = npr.randint(N)
        # Si la molécule est à droite (id < X[t])
        if mol_id < X[t]:
            X[t+1] = X[t] - 1
        else:
            X[t+1] = X[t] + 1

    # 2. Graphe de l'évolution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X)
    plt.axhline(N/2, color='r', linestyle='--')
    plt.title("Modèle d'Ehrenfest (évolution de $X_n$)")
    plt.xlabel("Temps")
    plt.ylabel("Molécules dans la pièce de droite")

    # 4. Théorème de Birkhoff
    # On compare le temps passé dans un état j à la proba stationnaire
    j = N // 2
    temps_en_j = np.mean(X == j)
    pi_j = (1/2**N) * math.comb(N, j)
    print(f"Temps passé en j={j}: {temps_en_j:.4f}")
    print(f"Proba stationnaire pi_j: {pi_j:.4f}")

    # 6. K pièces
    K = 4
    N_mol = 1000
    n_steps_k = 5000
    pieces = np.zeros((n_steps_k, K), dtype=int)
    pieces[0, 0] = N_mol  # Tout dans la première pièce

    for t in range(n_steps_k - 1):
        pieces[t+1] = pieces[t]
        # Choisir une molécule
        mol_piece = npr.choice(np.arange(K), p=pieces[t]/N_mol)

        # Choisir une direction
        direction = npr.choice([-1, 1])
        voisine = mol_piece + direction

        # Gérer les bords
        if voisine == -1:
            voisine = 1
        elif voisine == K:
            voisine = K - 2

        pieces[t+1, mol_piece] -= 1
        pieces[t+1, voisine] += 1

    plt.subplot(1, 2, 2)
    plt.plot(pieces)
    plt.title(f"Évolution pour {K} pièces")
    plt.xlabel("Temps")
    plt.ylabel("Nombre de molécules")
    plt.legend([f'Pièce {i+1}' for i in range(K)])
    plt.show()

    def run_question_7():
        # TP - Simulates the K-piece model with a fan/blower from right to left.
        print("\n[Exercice 2, Question 7] Simulation avec soufflerie:")

        K = 10
        N = 10000  # We use a smaller N than 10^6 for a reasonable simulation time
        n_steps = 500000
        burn_in = 100000  # Steps to discard to reach equilibrium

        plt.figure(figsize=(10, 6))

        # To store the state of the system over time for averaging
        history = np.zeros((n_steps - burn_in, K))

        for p_fan in [0, 0.01, 0.05, 0.2]:  # Different fan strengths
            # Start at equilibrium to speed up convergence
            pieces = np.full(K, N / K, dtype=int)

            for t in range(n_steps):
                # --- Diffusion Step ---
                # Choose a molecule based on which room it's in
                if np.sum(pieces) > 0:
                    mol_piece = npr.choice(
                        np.arange(K), p=pieces / np.sum(pieces))
                    direction = npr.choice([-1, 1])
                    voisine = mol_piece + direction
                    if voisine == -1:
                        voisine = 1
                    elif voisine == K:
                        voisine = K - 2

                    pieces[mol_piece] -= 1
                    pieces[voisine] += 1

                # --- Fan/Blower Step ---
                n_right = pieces[K-1]
                if n_right > 0:
                    n_moved = npr.binomial(n_right, p_fan)
                    pieces[K-1] -= n_moved
                    pieces[0] += n_moved

                # Store history after burn-in period
                if t >= burn_in:
                    history[t - burn_in, :] = pieces

            # Calculate the average number of molecules in each room
            avg_distribution = np.mean(history, axis=0)
            plt.plot(np.arange(K), avg_distribution,
                     'o-', label=f'p = {p_fan}')

        plt.title(f"Distribution d'équilibre moyenne ({K} pièces, N={N})")
        plt.xlabel("Numéro de la pièce (0=gauche, 9=droite)")
        plt.ylabel("Nombre moyen de molécules")
        plt.legend()
        plt.grid(True)
        plt.show()

        print("Observation: À mesure que p augmente, la soufflerie crée un flux net de droite à gauche,")
        print("ce qui engendre un gradient de concentration : les pièces de gauche deviennent plus peuplées.")

    run_question_7()

# =============================================================================
# Exercice 4: Perpetuité
# =============================================================================


def ex4():
    print("\n--- Exercice 4 ---")
    a = 0.9
    lambda_exp = 1.0
    n_steps = 1000
    n_traj = 500

    X = np.zeros((n_traj, n_steps))
    Y = npr.exponential(1/lambda_exp, size=(n_traj, n_steps))

    for t in range(n_steps - 1):
        X[:, t+1] = a * X[:, t] + Y[:, t+1]

    plt.figure(figsize=(12, 5))
    # Afficher quelques trajectoires
    plt.subplot(1, 2, 1)
    plt.plot(X[:5, :].T)
    plt.title("Quelques trajectoires de la chaîne")

    # Histogrammes à différents temps
    plt.subplot(1, 2, 2)
    plt.hist(X[:, 5], bins=30, density=True, alpha=0.5, label='n=5')
    plt.hist(X[:, 50], bins=30, density=True, alpha=0.5, label='n=50')
    plt.hist(X[:, -1], bins=30, density=True,
             alpha=0.5, label=f'n={n_steps-1}')
    plt.title("Convergence de la distribution de $X_n$")
    plt.legend()
    plt.show()

# =============================================================================
# Exercice 5: La poule et les poussins
# =============================================================================


def ex5():
    print("\n--- Exercice 5 ---")
    lambda_poisson = 10
    p_eclosion = 0.7
    n_sim = 10000

    def simule_poulee():
        N = npr.poisson(lambda_poisson)
        K = npr.binomial(N, p_eclosion)
        return K

    K_samples = [simule_poulee() for _ in range(n_sim)]

    plt.figure()
    # Loi théorique: Poisson(lambda * p)
    lambda_K = lambda_poisson * p_eclosion
    x = np.arange(0, 25)
    y_th = (np.exp(-lambda_K) * lambda_K**x) / [math.factorial(i) for i in x]

    plt.hist(K_samples, bins=x-0.5, density=True, label='Loi empirique')
    plt.plot(x, y_th, 'ro-', label=f'Loi théorique P({lambda_K:.1f})')
    plt.title("Distribution du nombre de poussins")
    plt.legend()
    plt.show()

# =============================================================================
# Exercice 6: Processus de Galton-Watson
# =============================================================================


def ex6():
    print("\n--- Exercice 6 ---")

    def simule_gw(p_geom, n_gen):
        # Pour une loi geom sur {1, 2, ...}, E = 1/p
        Z = np.zeros(n_gen, dtype=int)
        Z[0] = 1
        for n in range(n_gen - 1):
            if Z[n] == 0:
                break
            # somme de Z[n] variables géométriques
            Z[n+1] = np.sum(npr.geometric(p_geom, size=Z[n]) - 1)
        return Z

    plt.figure(figsize=(14, 6))

    # m < 1 (sous-critique) => 1/p < 1 => p > 1
    plt.subplot(1, 3, 1)
    p_sub = 0.6
    m_sub = (1-p_sub)/p_sub
    for _ in range(10):
        plt.plot(simule_gw(p_sub, 60))
    plt.title(f"Sous-critique (m={m_sub:.2f} < 1)")

    # m = 1 (critique) => p = 1
    plt.subplot(1, 3, 2)
    p_crit = 0.5
    m_crit = 1
    for _ in range(10):
        plt.plot(simule_gw(p_crit, 60))
    plt.title(f"Critique (m={m_crit:.2f} = 1)")

    # m > 1 (sur-critique) => p < 1
    plt.subplot(1, 3, 3)
    p_super = 0.4
    m_super = (1-p_super)/p_super
    for _ in range(10):
        plt.plot(simule_gw(p_super, 10))
    plt.title(f"Sur-critique (m={m_super:.2f} > 1)")
    plt.yscale('log')
    plt.show()

    print("\nObservations pour Galton-Watson (Q5):")
    print(" - Cas m < 1: Toutes les trajectoires s'éteignent très rapidement (extinction quasi certaine).")
    print(" - Cas m = 1: Toutes les trajectoires s'éteignent aussi, mais peuvent survivre plus longtemps avant de disparaître.")
    print(" - Cas m > 1: Comportement de 'tout ou rien'. La plupart des trajectoires s'éteignent,")
    print("   mais celles qui survivent explosent de manière exponentielle.")

    # Martingale Z_n / m^n
    plt.figure()
    n_gen_mart = 20
    m_values = m_super**np.arange(n_gen_mart)
    for _ in range(40):
        traj = simule_gw(p_super, n_gen_mart)
        plt.plot(traj / m_values, alpha=0.7)
    plt.title("Convergence de la martingale $Z_n/m^n$")
    plt.ylim(0, 5)
    plt.show()


# =============================================================================
# Exercice 8: Chaussettes
# =============================================================================


def ex8():
    print("\n--- Exercice 8 ---")

    p_portee = 1/50
    q_tiroir = 1/200
    n_jours = 10 * 365  # 10 ans de simulation

    # Question 1
    achats_q1 = 0
    # On suppose qu'on a toujours N paires, et on remplace instantanément
    N_paires = 7
    for _ in range(n_jours):
        # Chaussettes portées
        if npr.rand() < p_portee:
            achats_q1 += 1
        if npr.rand() < p_portee:
            achats_q1 += 1
        # Chaussettes dans le tiroir
        for _ in range(2 * (N_paires - 1)):
            if npr.rand() < q_tiroir:
                achats_q1 += 1

    # On jette la paire, donc chaque chaussette usée coûte une paire
    print(f"Q1: Nombre moyen de paires achetées/an: {achats_q1 / 10:.2f}")

    # Question 2
    n_paires_actuel = 57  # on vient d'acheter le pack
    achats_q2 = 0
    for _ in range(n_jours):
        # Vérifier si on doit racheter
        if n_paires_actuel < 7:
            n_paires_actuel += 50
            achats_q2 += 1

        # Usure
        paires_a_jeter = 0
        # Paire portée
        if npr.rand() < p_portee or npr.rand() < p_portee:
            paires_a_jeter += 1

        # Paires dans le tiroir
        for _ in range(n_paires_actuel - 1):
            if npr.rand() < q_tiroir or npr.rand() < q_tiroir:
                paires_a_jeter += 1

        n_paires_actuel -= paires_a_jeter

    paires_achetees_q2 = achats_q2 * 50
    cout_q1 = achats_q1
    cout_q2 = achats_q2 * 45
    print(
        f"Q2: Nombre moyen de paires achetées/an: {paires_achetees_q2 / 10:.2f}")
    print(f"Comparaison coût sur 10 ans: Q1={cout_q1:.0f} vs Q2={cout_q2:.0f}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    # ex1()
    # ex2()
    # ex3() is theoretical
    # ex4()
    # ex5()
    ex6()
    # ex7() is mostly theoretical
    # ex8()
