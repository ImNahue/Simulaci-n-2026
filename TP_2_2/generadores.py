import random
import math
import matplotlib.pyplot as plt

# ========================================================
# 1. GENERADORES DE VARIABLES ALEATORIAS (Según Tabla 1)
# ========================================================

def uniforme_continua(a, b):
    R = random.random()
    return a + (b - a) * R

def exponencial_inversa(lambda_param):
    R = random.random()
    return - (1.0 / lambda_param) * math.log(R)

def normal_box_muller(mu, sigma):
    R1 = random.random()
    R2 = random.random()
    Z0 = math.sqrt(-2.0 * math.log(R1)) * math.cos(2.0 * math.pi * R2)
    return mu + Z0 * sigma

def gamma_transformacion_inversa(k, lambda_param):
    # Enfoque por composición para k entero positivo (Erlang)
    producto_R = 1.0
    for _ in range(k):
        producto_R *= random.random()
    return - (1.0 / lambda_param) * math.log(producto_R)

def gamma_aceptacion_rechazo(alpha, beta):
    # Metodo de Aceptacion y Rechazo (Algoritmo de Ahrens-Dieter / Cheng)
    if alpha < 1.0:
        while True:
            U = random.random()
            b = (math.e + alpha) / math.e
            P = b * U
            if P <= 1.0:
                X = P ** (1.0 / alpha)
                U2 = random.random()
                if U2 <= math.exp(-X):
                    return X * beta
            else:
                X = - math.log((b - P) / alpha)
                U2 = random.random()
                if U2 <= X ** (alpha - 1.0):
                    return X * beta
    else:
        d = alpha - math.log(4.0)
        while True:
            U1 = random.random()
            U2 = random.random()
            V = math.log(U1 / (1.0 - U1)) / d
            Y = alpha * math.exp(V)
            Z = U1 * U1 * U2
            W = d * V - Y
            if W + 1.0 + math.log(4.5) >= 4.5 * Z or W >= math.log(Z):
                return Y * beta

def binomial_simulacion(n, p):
    exitos = 0
    for _ in range(n):
        if random.random() < p:
            exitos += 1
    return exitos

def pascal_inversa(r, p):
    # Distribucion Binomial Negativa por inversion de ensayos
    exitos = 0
    ensayos = 0
    while exitos < r:
        ensayos += 1
        if random.random() < p:
            exitos += 1
    return ensayos

def poisson_inversa(lambda_param):
    L = math.exp(-lambda_param)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def empirica_discreta_inversa(valores, probabilidades):
    acumuladas = []
    suma = 0.0
    for p in probabilidades:
        suma += p
        acumuladas.append(suma)
    
    R = random.random()
    for idx, f_acum in enumerate(acumuladas):
        if R <= f_acum:
            return valores[idx]

# ========================================================
# 2. GENERACIÓN MASIVA DE MUESTRAS (N=10000) Y GRÁFICOS
# ========================================================
if __name__ == "__main__":
    N = 10000
    
    # Generamos las muestras de testeo (Incluyendo las nuevas incorporaciones)
    muestras_uniforme = [uniforme_continua(2, 8) for _ in range(N)]
    muestras_exponencial = [exponencial_inversa(0.5) for _ in range(N)]
    muestras_normal = [normal_box_muller(10, 2) for _ in range(N)]
    muestras_gamma_inv = [gamma_transformacion_inversa(3, 0.5) for _ in range(N)]
    muestras_gamma_ar = [gamma_aceptacion_rechazo(2.5, 2.0) for _ in range(N)]
    muestras_binomial = [binomial_simulacion(20, 0.4) for _ in range(N)]
    muestras_pascal = [pascal_inversa(4, 0.5) for _ in range(N)]
    muestras_poisson = [poisson_inversa(4.0) for _ in range(N)]
    
    valores_emp = [10, 20, 30, 40]
    probs_emp = [0.1, 0.4, 0.3, 0.2]
    muestras_empirica = [empirica_discreta_inversa(valores_emp, probs_emp) for _ in range(N)]

    # Creamos una grilla adaptada de 4 filas x 2 columnas para albergar las 8 variantes de testeo
    fig, axs = plt.subplots(4, 2, figsize=(12, 14))
    fig.suptitle("UTN-FRRO - Testeo Estadístico Integrado de Generadores (TP 2.2)", fontsize=14, fontweight='bold')

    # 1. Uniforme
    axs[0, 0].hist(muestras_uniforme, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    axs[0, 0].set_title("Distribución Uniforme Continua U(2, 8)")
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # 2. Exponencial
    axs[0, 1].hist(muestras_exponencial, bins=30, density=True, color='lightgreen', edgecolor='black', alpha=0.7)
    axs[0, 1].set_title("Distribución Exponencial ($\lambda = 0.5$)")
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # 3. Normal
    axs[1, 0].hist(muestras_normal, bins=30, density=True, color='salmon', edgecolor='black', alpha=0.7)
    axs[1, 0].set_title("Distribución Normal ($\mu=10, \sigma=2$)")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # 4. Gamma por Inversión (Composición)
    axs[1, 1].hist(muestras_gamma_inv, bins=30, density=True, color='turquoise', edgecolor='black', alpha=0.7)
    axs[1, 1].set_title("Distribución Gamma por Inversión (k=3, $\lambda=0.5$)")
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    # 5. Gamma por Aceptación y Rechazo
    axs[2, 0].hist(muestras_gamma_ar, bins=30, density=True, color='teal', edgecolor='black', alpha=0.7)
    axs[2, 0].set_title("Distribución Gamma por Acep. y Rechazo ($\\alpha=2.5, \\beta=2.0$)")
    axs[2, 0].grid(True, linestyle='--', alpha=0.5)

    # 6. Binomial
    axs[2, 1].hist(muestras_binomial, bins=range(22), density=True, color='gold', edgecolor='black', alpha=0.7, align='left')
    axs[2, 1].set_title("Distribución Binomial (n=20, p=0.4)")
    axs[2, 1].grid(True, linestyle='--', alpha=0.5)

    # 7. Pascal (Binomial Negativa)
    axs[3, 0].hist(muestras_pascal, bins=range(max(muestras_pascal)+2), density=True, color='sandybrown', edgecolor='black', alpha=0.7, align='left')
    axs[3, 0].set_title("Distribución de Pascal por Inversión (r=4, p=0.5)")
    axs[3, 0].grid(True, linestyle='--', alpha=0.5)

    # 8. Poisson
    axs[3, 1].hist(muestras_poisson, bins=range(13), density=True, color='orchid', edgecolor='black', alpha=0.7, align='left')
    axs[3, 1].set_title("Distribución de Poisson ($\lambda = 4.0$)")
    axs[3, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # GUARDAR LA IMAGEN CON CALIDAD PROFESIONAL PARA LATEX
    plt.savefig("histogramas_simulacion.png", dpi=300)
    print("¡Imagen de testeo integrada actualizada y salvada como 'histogramas_simulacion.png'!")
    
    plt.show()