import numpy as np
import argparse
import matplotlib.pyplot as plt

# Parámetros generales
MU = 3.0  # Tasa de servicio
QUEUE_SIZE = 2
ARRIVAL_FACTORS = [0.25, 0.5, 0.75, 1.0, 1.25]
RUNS = 30
SIM_TIME = 1000  # Tiempo de simulación por corrida


def simulate_mm1(lambda_, mu, queue_size, sim_time=SIM_TIME):
    np.random.seed()
    t = 0
    n = 0  # clientes en sistema
    n_arrivals = 0
    n_denied = 0
    n_departures = 0
    area_n = 0.0
    area_q = 0.0
    busy_time = 0.0
    last_event = 0.0
    times_in_system = []
    times_in_queue = []
    queue_hist = {}
    arrival_times = []
    queue = []
    server_busy = False

    # Primer arribo
    next_arrival = np.random.exponential(1/lambda_)
    next_departure = float('inf')

    while t < sim_time:
        if next_arrival < next_departure:
            t = next_arrival
            area_n += n * (t - last_event)
            area_q += max(n-1, 0) * (t - last_event)
            busy_time += (1 if n > 0 else 0) * (t - last_event)
            last_event = t
            if n < queue_size + 1:  # +1 porque incluye el servidor
                n += 1
                n_arrivals += 1
                arrival_times.append(t)
                queue.append(t)
                if n == 1:
                    next_departure = t + np.random.exponential(1/mu)
            else:
                n_denied += 1
            next_arrival = t + np.random.exponential(1/lambda_)
        else:
            t = next_departure
            area_n += n * (t - last_event)
            area_q += max(n-1, 0) * (t - last_event)
            busy_time += (1 if n > 0 else 0) * (t - last_event)
            last_event = t
            n -= 1
            n_departures += 1
            arrival_time = queue.pop(0)
            times_in_system.append(t - arrival_time)
            times_in_queue.append((t - arrival_time) - np.random.exponential(1/mu))
            if n > 0:
                next_departure = t + np.random.exponential(1/mu)
            else:
                next_departure = float('inf')
        # Histograma de clientes en cola
        n_in_queue = max(n-1, 0)
        queue_hist[n_in_queue] = queue_hist.get(n_in_queue, 0) + 1

    # Métricas
    avg_n_system = area_n / sim_time
    avg_n_queue = area_q / sim_time
    avg_time_system = np.mean(times_in_system) if times_in_system else 0
    avg_time_queue = np.mean(times_in_queue) if times_in_queue else 0
    utilization = busy_time / sim_time
    prob_denied = n_denied / (n_arrivals + n_denied) if (n_arrivals + n_denied) > 0 else 0
    prob_n_in_queue = {k: v / sum(queue_hist.values()) for k, v in queue_hist.items()}

    return {
        'avg_n_system': avg_n_system,
        'avg_n_queue': avg_n_queue,
        'avg_time_system': avg_time_system,
        'avg_time_queue': avg_time_queue,
        'utilization': utilization,
        'prob_denied': prob_denied,
        'prob_n_in_queue': prob_n_in_queue,
        'n_denied': n_denied,
        'n_arrivals': n_arrivals,
        'n_departures': n_departures
    }

def main():
    parser = argparse.ArgumentParser(description="Simulación MM1 con cola finita")
    parser.add_argument('--mu', type=float, default=3.0, help='Tasa de servicio (mu)')
    parser.add_argument('--queue_size', type=int, default=2, help='Tamaño de la cola (fijo)')
    parser.add_argument('--arrival_factors', type=float, nargs='+', default=[0.25,0.5,0.75,1.0,1.25], help='Factores de tasa de arribo respecto a mu (separados por espacio)')
    parser.add_argument('--runs', type=int, default=30, help='Cantidad de corridas por experimento')
    parser.add_argument('--sim_time', type=float, default=1000, help='Tiempo de simulación por corrida')
    args = parser.parse_args()

    MU = args.mu
    QUEUE_SIZE = args.queue_size
    ARRIVAL_FACTORS = args.arrival_factors
    RUNS = args.runs
    SIM_TIME = args.sim_time

    metrics_names = ['avg_n_system', 'avg_n_queue', 'avg_time_system', 'avg_time_queue', 'utilization', 'prob_denied']
    results = {m: [] for m in metrics_names}

    # Para matriz de probabilidades de n en cola
    prob_n_matrix = []  # cada fila: para un lambda, columnas: n=0...QUEUE_SIZE

    print(f"\nCola finita de tamaño: {QUEUE_SIZE}")
    for factor in ARRIVAL_FACTORS:
        lambda_ = factor * MU
        res_list = []
        prob_n_list = []
        for _ in range(RUNS):
            res = simulate_mm1(lambda_, MU, QUEUE_SIZE, SIM_TIME)
            res_list.append(res)
            prob_n_list.append(res['prob_n_in_queue'])
        avg_metrics = {k: np.mean([r[k] for r in res_list]) for k in res_list[0] if not isinstance(res_list[0][k], dict)}
        for m in metrics_names:
            results[m].append(avg_metrics[m])
        # Calcular promedio de prob_n_in_queue para cada n=0...QUEUE_SIZE
        prob_n_avg = []
        for n in range(QUEUE_SIZE+1):
            vals = [p.get(n, 0.0) for p in prob_n_list]
            prob_n_avg.append(np.mean(vals))
        prob_n_matrix.append(prob_n_avg)
        print(f"Tasa de arribo: {lambda_:.2f} (factor {factor})")
        print(f"  Prom. clientes en sistema: {avg_metrics['avg_n_system']:.3f}")
        print(f"  Prom. clientes en cola: {avg_metrics['avg_n_queue']:.3f}")
        print(f"  Tiempo prom. en sistema: {avg_metrics['avg_time_system']:.3f}")
        print(f"  Tiempo prom. en cola: {avg_metrics['avg_time_queue']:.3f}")
        print(f"  Utilización del servidor: {avg_metrics['utilization']:.3f}")
        print(f"  Prob. de denegación: {avg_metrics['prob_denied']:.3f}")

    # Graficar resultados como líneas en 2 columnas
    n_metrics = len(metrics_names)
    n_cols = 2
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 2.8 * n_rows), constrained_layout=True)
    axs = axs.flatten()
    x = [factor * MU for factor in ARRIVAL_FACTORS]
    for idx, m in enumerate(metrics_names):
        ax = axs[idx]
        ax.plot(x, results[m], marker='o', color='tab:blue')
        ax.set_xlabel('Tasa de arribo (lambda)')
        ax.set_ylabel(m.replace('_', ' ').capitalize())
        ax.set_title(f'{m.replace("_", " ").capitalize()} vs Tasa de arribo (Cola={QUEUE_SIZE})')
        ax.grid(True)
        for i, val in enumerate(results[m]):
            ax.text(x[i], val, f"{val:.2f}", ha='center', va='bottom', fontsize=9, color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))
    # Ocultar ejes vacíos si los hay
    for idx in range(n_metrics, len(axs)):
        fig.delaxes(axs[idx])
    plt.show()
    print("\nGráficos de métricas mostrados para tamaño de cola fijo.")

    # --- Matriz de probabilidad de n en cola vs lambda ---
    prob_n_matrix = np.array(prob_n_matrix)  # shape: (len(ARRIVAL_FACTORS), QUEUE_SIZE+1)
    fig2, axm = plt.subplots(figsize=(8, 5))
    cax = axm.imshow(prob_n_matrix.T, aspect='auto', cmap='RdYlGn_r', origin='lower')
    axm.set_xticks(np.arange(len(ARRIVAL_FACTORS)))
    axm.set_xticklabels([f"{factor*MU:.2f}" for factor in ARRIVAL_FACTORS])
    axm.set_yticks(np.arange(QUEUE_SIZE+1))
    axm.set_yticklabels([str(n) for n in range(QUEUE_SIZE+1)])
    axm.set_xlabel('Tasa de arribo (lambda)')
    axm.set_ylabel('n en cola')
    axm.set_title('Probabilidad de encontrar n personas en la cola')
    for i in range(len(ARRIVAL_FACTORS)):
        for j in range(QUEUE_SIZE+1):
            val = prob_n_matrix[i, j]
            axm.text(i, j, f"{val:.2f}", ha='center', va='center', color='black', fontsize=9)
    fig2.colorbar(cax, ax=axm, label='Probabilidad')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()