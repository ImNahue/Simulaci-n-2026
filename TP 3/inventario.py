import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parámetros del Modelo de Inventario (s,S) 
# Se pueden ajustar para diferentes experimentos
s_param = 10  # Punto de reorden (s)
S_param = 100  # Nivel objetivo (S)

# Parámetros de la simulación
ARRIVAL_RATE = 1.0     # Tasa de arribo de clientes (clientes/unidad tiempo)
MEAN_ITEMS_PER_CUSTOMER = 2.0  # Media de ítems pedidos por cliente
MIN_DELAY = 5.0        # Demora mínima de entrega
MAX_DELAY = 15.0        # Demora máxima de entrega
CHECK_INTERVAL = 30.0    # Intervalo de chequeo de inventario
ORDER_COST = 100     # Costo fijo por orden (K)
ORDER_COST_PER_UNIT = 5  
HOLDING_COST_PER_UNIT = 0.1  # Costo de mantenimiento por unidad por dia (h)
SHORTAGE_COST_PER_UNIT = 10        # Costo de faltante por unidad (p)
SIMULATION_DURATION = 365       # Duración de cada corrida de simulación (12 meses)
NUM_RUNS_PER_EXPERIMENT = 30           # Número de corridas por cada experimento

N_BARRAS = 12  # Número de barras para el gráfico de costos

# Inventario inicial
INITIAL_INVENTORY = S_param # Empezamos con el nivel objetivo para evitar un p

def calcular_valores_teoricos(s, S, arrival_rate, mean_items_per_customer, holding_cost, shortage_cost, order_cost, unit_order_cost, min_delay, max_delay, duration):
    demanda_promedio_diaria = arrival_rate * mean_items_per_customer
    demanda_total_esperada = demanda_promedio_diaria * duration

    inventario_promedio_teorico = (s + S) / 2
    costo_mantenimiento_promedio_esperado = inventario_promedio_teorico * holding_cost * duration

    num_ordenes_esperado = demanda_total_esperada / (S - s) if (S - s) > 0 else 0
    costo_orden_promedio_esperado = num_ordenes_esperado * (order_cost + (S - s) * unit_order_cost)

    lead_time_promedio = (min_delay + max_delay) / 2
    demanda_en_lead_time_promedio = demanda_promedio_diaria * lead_time_promedio
    
    expected_shortage_per_order_cycle = max(0, demanda_en_lead_time_promedio - s)
    costo_faltante_promedio_esperado = expected_shortage_per_order_cycle * num_ordenes_esperado * shortage_cost
    
    costo_total_promedio_esperado = costo_mantenimiento_promedio_esperado + costo_orden_promedio_esperado + costo_faltante_promedio_esperado
    
    return {
        "inventario_promedio_esperado": inventario_promedio_teorico,
        "costo_mantenimiento_promedio_esperado": costo_mantenimiento_promedio_esperado,
        "costo_orden_promedio_esperado": costo_orden_promedio_esperado,
        "costo_faltante_promedio_esperado": costo_faltante_promedio_esperado,
        "costo_total_promedio_esperado": costo_total_promedio_esperado
    }
def run_event_simulation(
    s=s_param, S=S_param,
    arrival_rate=ARRIVAL_RATE,
    mean_items_per_customer=MEAN_ITEMS_PER_CUSTOMER,
    min_delay=MIN_DELAY,
    max_delay=MAX_DELAY,
    inventory_check_interval=CHECK_INTERVAL,
    unit_order_cost=ORDER_COST_PER_UNIT,
    order_cost=ORDER_COST,
    holding_cost=HOLDING_COST_PER_UNIT,
    shortage_cost=SHORTAGE_COST_PER_UNIT,
    duration=SIMULATION_DURATION,
    initial_inventory=S_param
):
    """
    Simulación basada en eventos discretos para un sistema de inventario.

    Args:
        s (int): Punto de reorden.
        S (int): Nivel objetivo.
        arrival_rate (float): Tasa de arribo de clientes (clientes/unidad tiempo).
        mean_items_per_customer (float): Media de ítems pedidos por cliente.
        min_delay (float): Demora mínima de entrega.
        max_delay (float): Demora máxima de entrega.
        inventory_check_interval (float): Intervalo de chequeo de inventario.
        order_cost (float): Costo fijo por orden.
        holding_cost (float): Costo de mantenimiento por unidad.
        shortage_cost (float): Costo de faltante por unidad.
        duration (float): Duración de la simulación.
        initial_inventory (int): Nivel de inventario al inicio de la simulación.

    Returns:
        dict: Un diccionario con los costos totales y las métricas de rendimiento.
    """

    t = 0.0
    inventory = S if initial_inventory is None else initial_inventory
    order_pending_qty = 0  # cantidad del pedido pendiente
    next_order_arrival_time = float('inf')
    total_order_cost = 0
    total_holding_cost = 0
    total_shortage_cost = 0
    total_units_shortage = 0
    num_orders_placed = 0
    inventory_history = []  # (tiempo, inventario)
    time_history = []       # solo para compatibilidad, pero no se usa en el nuevo enfoque
    shortage_history = []   # (tiempo, faltante acumulado)
    # Históricos de costos por evento
    order_cost_history = []     # (tiempo, costo acumulado)
    holding_cost_history = []   # (tiempo, costo acumulado)
    shortage_cost_history = []  # (tiempo, costo acumulado)
    # Inicializar tiempos de eventos
    next_arrival_time = round(np.random.exponential(1/arrival_rate), 8)
    next_inventory_check_time = round(inventory_check_interval, 8)
    last_inventory_update = 0.0

    def update_holding_cost(current_time):
        nonlocal inventory, total_holding_cost, last_inventory_update, holding_cost_history
        # Costo de mantenimiento proporcional al tiempo transcurrido
        total_holding_cost += inventory * holding_cost * (current_time - last_inventory_update)
        holding_cost_history.append((current_time, total_holding_cost))
        last_inventory_update = current_time
        

    def create_arrival(current_time):
        nonlocal inventory, total_units_shortage, total_shortage_cost, next_arrival_time, last_inventory_update, holding_cost_history, total_holding_cost
        items = np.random.poisson(mean_items_per_customer)

        update_holding_cost(current_time)

        if inventory >= items:
            inventory -= items
        else:
            shortage = items - inventory
            total_units_shortage += shortage
            total_shortage_cost += shortage * shortage_cost
            inventory = 0
        next_arrival_time = round(current_time + np.random.exponential(1/arrival_rate), 8)
        # Registrar costo de faltante acumulado
        shortage_cost_history.append((current_time, total_shortage_cost))

    def create_order_if_needed(current_time):
        nonlocal inventory, order_pending_qty, total_order_cost, num_orders_placed, next_inventory_check_time, next_order_arrival_time
        if inventory <= s and order_pending_qty == 0:
            order_pending_qty = S - inventory

            delay = np.random.uniform(min_delay, max_delay)
            next_order_arrival_time = round(current_time + delay, 8)
            total_order_cost += order_cost + order_pending_qty * unit_order_cost
            num_orders_placed += 1
            # Registrar costo de orden acumulado
            order_cost_history.append((current_time, total_order_cost))
        next_inventory_check_time = round(current_time + inventory_check_interval, 8)

    def receive_order(current_time):
        nonlocal inventory, order_pending_qty, next_order_arrival_time, total_holding_cost, holding_cost, last_inventory_update
        
        update_holding_cost(current_time)

        if order_pending_qty > 0 and t == next_order_arrival_time:
            inventory += order_pending_qty
            order_pending_qty = 0
            next_order_arrival_time = float('inf')

    while t < duration:
        # Calcular el próximo evento
        next_event_time = min(next_arrival_time, next_inventory_check_time, next_order_arrival_time)
        t = round(next_event_time, 8)
        # Costo de mantenimiento proporcional al tiempo transcurrido

        
        # Ejecutar el evento correspondiente usando igualdad exacta
        if t == next_arrival_time:
            create_arrival(t)
        if t == next_inventory_check_time:
            create_order_if_needed(t)
        if t == next_order_arrival_time:
            receive_order(t)
        # Guardar historial como (tiempo, valor)
        inventory_history.append((t, inventory))
        shortage_history.append((t, total_units_shortage))
        time_history.append(t)  # opcional, para compatibilidad

    update_holding_cost(duration)  # Actualizar el costo de mantenimiento al final de la simulación
    total_cost = total_order_cost + total_holding_cost + total_shortage_cost
    return {
        'total_order_cost': total_order_cost,
        'total_holding_cost': total_holding_cost,
        'total_shortage_cost': total_shortage_cost,
        'total_cost': total_cost,
        'total_units_shortage': total_units_shortage,
        'num_orders_placed': num_orders_placed,
        # Se devuelven los historiales solicitados:
        'inventory_history': inventory_history,  # lista de (tiempo, inventario)
        'shortage_history': shortage_history,    # lista de (tiempo, faltante)
        # time_history puede omitirse, pero se deja por compatibilidad
        'time_history': time_history,
        # históricos de costos
        'order_cost_history': order_cost_history,
        'holding_cost_history': holding_cost_history,
        'shortage_cost_history': shortage_cost_history
    }


def main():
    parser = argparse.ArgumentParser(description="Simulación de Inventario por Eventos")
    parser.add_argument('--s', type=int, default=s_param, help='Punto de reorden (s)')
    parser.add_argument('--S', type=int, default=S_param, help='Nivel objetivo (S)')
    parser.add_argument('--arrival_rate', type=float, default=ARRIVAL_RATE, help='Tasa de arribo de clientes (clientes/unidad tiempo)')
    parser.add_argument('--mean_items', type=float, default=MEAN_ITEMS_PER_CUSTOMER, help='Media de ítems pedidos por cliente')
    parser.add_argument('--min_delay', type=float, default=MIN_DELAY, help='Demora mínima de entrega')
    parser.add_argument('--max_delay', type=float, default=MAX_DELAY, help='Demora máxima de entrega')
    parser.add_argument('--check_interval', type=float, default=CHECK_INTERVAL, help='Intervalo de chequeo de inventario')
    parser.add_argument('--order_cost', type=float, default=ORDER_COST, help='Costo fijo por orden')
    parser.add_argument('--unit_order_cost', type=float, default=ORDER_COST_PER_UNIT, help='Costo fijo por orden')

    parser.add_argument('--holding_cost', type=float, default=HOLDING_COST_PER_UNIT, help='Costo de mantenimiento por unidad')
    parser.add_argument('--shortage_cost', type=float, default=SHORTAGE_COST_PER_UNIT, help='Costo de faltante por unidad')
    parser.add_argument('--duration', type=float, default=SIMULATION_DURATION, help='Duración de la simulación')
    parser.add_argument('--runs', type=int, default=NUM_RUNS_PER_EXPERIMENT, help='Número de corridas')
    args = parser.parse_args()

    results = []
    order_cost_histories = []
    holding_cost_histories = []
    shortage_cost_histories = []
    for _ in range(args.runs):
        res = run_event_simulation(
            s=args.s, S=args.S,
            arrival_rate=args.arrival_rate,
            mean_items_per_customer=args.mean_items,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            inventory_check_interval=args.check_interval,
            order_cost=args.order_cost,
            unit_order_cost=args.unit_order_cost,
            holding_cost=args.holding_cost,
            shortage_cost=args.shortage_cost,
            duration=args.duration,
            initial_inventory=args.S
        )
        results.append(res)
        order_cost_histories.append(res['order_cost_history'])
        holding_cost_histories.append(res['holding_cost_history'])
        shortage_cost_histories.append(res['shortage_cost_history'])
    valores_teoricos = calcular_valores_teoricos(
        s=args.s,
        S=args.S,
        arrival_rate=args.arrival_rate,
        mean_items_per_customer=args.mean_items,
        holding_cost=args.holding_cost,
        shortage_cost=args.shortage_cost,
        order_cost=args.order_cost,
        unit_order_cost=args.unit_order_cost,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        duration=SIMULATION_DURATION)

    print("\n--- Valores Teóricos Esperados ---")
    print(f"Inventario promedio esperado: {valores_teoricos['inventario_promedio_esperado']:.2f}")
    print(f"Costo de mantenimiento promedio esperado: {valores_teoricos['costo_mantenimiento_promedio_esperado']:.2f}")
    print(f"Costo de orden promedio esperado: {valores_teoricos['costo_orden_promedio_esperado']:.2f}")
    print(f"Costo de faltante promedio esperado: {valores_teoricos['costo_faltante_promedio_esperado']:.2f}")
    print(f"Costo total promedio esperado: {valores_teoricos['costo_total_promedio_esperado']:.2f}")
    

    # --- Nueva lógica para graficar con escala de tiempo y promedios por intervalo ---
    plot_interval = 3.0  # 1 unidad de tiempo = 1 unidad en el eje x
    max_time = args.duration
    num_points = int(np.ceil(max_time / plot_interval))
    plot_times = [i * plot_interval for i in range(num_points + 1)]

    # Para cada corrida y cada intervalo, tomamos el último valor registrado antes o igual al final del intervalo
    def get_last_value_before(history, t):
        # history: lista de (tiempo, valor), t: tiempo de corte
        last_val = history[0][1]
        for time, val in history:
            if time > t:
                break
            last_val = val
        return last_val

    inventory_matrix = []
    shortage_matrix = []
    for r in results:
        inv_hist = r['inventory_history']
        sh_hist = r['shortage_history']
        inv_vals = [get_last_value_before(inv_hist, t) for t in plot_times]
        sh_vals = [get_last_value_before(sh_hist, t) for t in plot_times]
        inventory_matrix.append(inv_vals)
        shortage_matrix.append(sh_vals)

    avg_inventory = np.mean(inventory_matrix, axis=0)
    avg_shortage = np.mean(shortage_matrix, axis=0)
    avg_time = plot_times

    avg_total_cost = np.mean([r['total_cost'] for r in results])
    avg_order_cost = np.mean([r['total_order_cost'] for r in results])
    avg_holding_cost = np.mean([r['total_holding_cost'] for r in results])
    avg_shortage_cost = np.mean([r['total_shortage_cost'] for r in results])
    print("---------------------")
    print("Costos de la simulación:")
    print(f"Costo de mantenimiento promedio: {avg_holding_cost:.2f}")
    print(f"Costo de orden promedio: {avg_order_cost:.2f}")
    print(f"Costo de faltante promedio: {avg_shortage_cost:.2f}")
    print(f"Costo total promedio: {avg_total_cost:.2f}")

    # --- Cálculo de costos promedio por intervalo para el gráfico de barras ---
    n_barras = N_BARRAS
    barra_intervalo = args.duration / n_barras
    barra_times = [i * barra_intervalo for i in range(n_barras + 1)]

    def get_last_cost_before(history, t):
        # history: lista de (tiempo, costo acumulado)
        if not history:
            return 0.0
        last_val = history[0][1]
        for time, val in history:
            if time > t:
                break
            last_val = val
        return last_val

    # Para cada corrida y cada barra, calcular el costo incremental en ese intervalo
    order_costs_barras = []
    holding_costs_barras = []
    shortage_costs_barras = []
    for i in range(n_barras):
        order_vals = []
        holding_vals = []
        shortage_vals = []
        t0 = barra_times[i]
        t1 = barra_times[i+1]
        for j in range(args.runs):
            oc_hist = order_cost_histories[j]
            hc_hist = holding_cost_histories[j]
            sc_hist = shortage_cost_histories[j]
            oc = get_last_cost_before(oc_hist, t1) - get_last_cost_before(oc_hist, t0)
            hc = get_last_cost_before(hc_hist, t1) - get_last_cost_before(hc_hist, t0)
            sc = get_last_cost_before(sc_hist, t1) - get_last_cost_before(sc_hist, t0)
            order_vals.append(oc)
            holding_vals.append(hc)
            shortage_vals.append(sc)
        order_costs_barras.append(np.mean(order_vals))
        holding_costs_barras.append(np.mean(holding_vals))
        shortage_costs_barras.append(np.mean(shortage_vals))

    # --- Gráficas ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 9))
    # Graficar todas las corridas individuales con baja opacidad
    for inv_vals in inventory_matrix:
        axs[0].plot(avg_time, inv_vals, color='blue', alpha=0.3, linewidth=0.8)
    # Inventario promedio
    axs[0].plot(avg_time, avg_inventory, label='Inventario promedio', color='red', linewidth=2)
    # Marcar s y S
    axs[0].axhline(y=s_param, color='green', linestyle='--', linewidth=1.5, label='s (punto de reorden)')
    axs[0].axhline(y=S_param, color='purple', linestyle='--', linewidth=1.5, label='S (nivel objetivo)')
    axs[0].set_xlabel('Tiempo')
    axs[0].set_ylabel('Inventario')
    axs[0].set_title('Inventario promedio a lo largo del tiempo')
    axs[0].grid(True)
    axs[0].legend()
    # Gráfico de barras de costos promedio por intervalo
    barra_labels = [f"{int(barra_times[i])}-{int(barra_times[i+1])}" for i in range(n_barras)]
    bar_width = 0.8
    ind = np.arange(n_barras)
    axs[1].bar(ind, order_costs_barras, bar_width, label='Costo de orden', color='royalblue')
    axs[1].bar(ind, holding_costs_barras, bar_width, bottom=order_costs_barras, label='Costo de mantenimiento', color='orange')
    axs[1].bar(ind, shortage_costs_barras, bar_width, bottom=np.array(order_costs_barras)+np.array(holding_costs_barras), label='Costo de faltante', color='crimson')
    axs[1].set_xlabel('Intervalo de tiempo')
    axs[1].set_ylabel('Costo promedio')
    axs[1].set_title('Costos promedio por intervalo de tiempo')
    axs[1].set_xticks(ind)
    axs[1].set_xticklabels(barra_labels, rotation=45)
    axs[1].legend()
    axs[1].grid(True, axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()