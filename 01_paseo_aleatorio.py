"""
01_paseo_aleatorio.py — El precio como partícula browniana
==========================================================

Pregunta física: ¿se mueve el precio de una acción como una partícula de polen
en el agua (movimiento browniano)?

Si fuera así, se deberían cumplir tres cosas, y las comprobamos con Google:
  (a) Los pasos no tienen memoria: el retorno de hoy no dice nada del de mañana.
  (b) El desplazamiento cuadrático medio crece LINEALMENTE con el tiempo
      (difusión): Var[r(1)+...+r(k)] = k * Var[r].
  (c) Los pasos son gaussianos.

ESQUEMA DE TIPOS DE DATOS
-------------------------
    precios        : np.ndarray[float64], forma (1047,)  precio de cierre diario
    r              : np.ndarray[float64], forma (1046,)  log-retornos diarios
    caminos        : np.ndarray[float64], forma (3, 1046) tres paseos simulados
    acf            : np.ndarray[float64], forma (20,)    autocorrelación para retardos 1..20
    horizontes     : np.ndarray[int64]                   k = 1, 2, ..., 60 días
    var_k          : np.ndarray[float64]                 varianza del retorno a k días
    resultados     : dict[str, float]                    números que se citan en el documento

ESTRUCTURA DEL PROGRAMA
-----------------------
    1. Cargar datos y calcular retornos, media y volatilidad.
    2. Simular paseos aleatorios con la misma volatilidad  -> figura 1.1
    3. Memoria: autocorrelación de r                         -> figura 1.2 (izq.)
    4. Difusión: varianza frente a horizonte                  -> figura 1.2 (dcha.)
    5. Gaussianidad: histograma y conteo de días extremos     -> figura 1.3
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import comun

rng = np.random.default_rng(1)     # generador de números aleatorios con semilla fija (reproducible)
resultados = {}                    # dict: nombre -> valor; lo imprimimos al final

# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------
fechas, precios = comun.cargar_google()
r = comun.log_retornos(precios)
n = len(r)
media, sigma = r.mean(), r.std()
resultados["dias"] = n
resultados["media_diaria_%"] = 100 * media
resultados["sigma_diaria_%"] = 100 * sigma
resultados["sigma_anual_%"] = 100 * sigma * np.sqrt(252)     # 252 días de mercado al año
resultados["sigma_mensual_%"] = 100 * sigma * np.sqrt(21)    # ~21 días de mercado al mes

# ---------------------------------------------------------------------------
# 2. Tres paseos aleatorios con la MISMA media y volatilidad que Google
# ---------------------------------------------------------------------------
# Cada paso es gaussiano e independiente: exactamente el modelo de Bachelier.
pasos = rng.normal(media, sigma, size=(3, n))
caminos = np.log(precios[0]) + np.cumsum(pasos, axis=1)     # log-precio = suma de pasos

fig, ax = plt.subplots(figsize=(6.2, 2.6))
dias = np.arange(1, n + 1)
orden = [0, 3, 1, 2]       # list[int]: la serie real (índice 3) se dibuja en segundo lugar, sin pistas
series = [np.exp(c) for c in caminos] + [precios[1:]]
colores = [comun.AZUL, comun.NARANJA, comun.VERDE, comun.GRIS]   # list[str]: un color por serie
for k, i in enumerate(orden):
    ax.plot(dias, series[i], lw=0.9, color=colores[k], label=comun.t("serie", "series") + f" {k + 1}")
ax.set_yscale("log")
ax.set_xlabel(comun.t("día", "day"))
ax.set_ylabel(comun.t("precio ($)", "price ($)"))
ax.legend(ncol=4, loc="upper left", frameon=False)
comun.guardar(fig, "fig1_paseos")
resultados["serie_real_en_figura"] = orden.index(3) + 1

# ---------------------------------------------------------------------------
# 3. Memoria: autocorrelación C(tau) = corr(r_t, r_{t+tau})
# ---------------------------------------------------------------------------
def autocorrelacion(x, max_retardo):
    """Correlación entre x_t y x_{t+tau} para tau = 1..max_retardo.
    Se resta la media y se normaliza por la varianza: C(0) = 1."""
    x = x - x.mean()
    return np.array([np.mean(x[:-t] * x[t:]) for t in range(1, max_retardo + 1)]) / x.var()

acf = autocorrelacion(r, 20)
banda = 1.96 / np.sqrt(n)            # si no hay memoria, el 95 % de los valores cae dentro de ±banda
resultados["C_r(1)"] = acf[0]
resultados["banda_95"] = banda
resultados["retardos_fuera_banda_de_20"] = int(np.sum(np.abs(acf) > banda))

# ---------------------------------------------------------------------------
# 4. Difusión: la varianza crece linealmente con el tiempo
# ---------------------------------------------------------------------------
# Retorno a k días = suma de k retornos diarios seguidos (ventanas solapadas).
horizontes = np.arange(1, 61)
r_centrado = r - media
var_k = np.array([np.var(np.convolve(r_centrado, np.ones(k), mode="valid")) for k in horizontes])
pendiente = np.polyfit(np.log(horizontes), np.log(var_k), 1)[0]   # en log-log, difusión => pendiente 1
resultados["pendiente_difusion"] = pendiente
resultados["var20_sobre_20var1"] = var_k[19] / (20 * var_k[0])

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
ax[0].bar(np.arange(1, 21), acf, color=comun.AZUL, width=0.7)
ax[0].axhspan(-banda, banda, color=comun.GRIS, alpha=0.2, lw=0)
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_xlabel(comun.t(r"retardo $\tau$ (días)", r"lag $\tau$ (days)"))
ax[0].set_ylabel(r"$C(\tau)$")
ax[0].set_title(comun.t("¿Recuerda el precio su último paso?", "Does the price remember its last step?"))
ax[0].set_ylim(-0.25, 0.25)
ax[1].loglog(horizontes, var_k / var_k[0], "o", ms=2.5, color=comun.AZUL, label="Google")
ax[1].loglog(horizontes, horizontes, color=comun.NARANJA, lw=1.2, label=comun.t("difusión: $\\propto k$", "diffusion: $\\propto k$"))
ax[1].set_xlabel(comun.t("horizonte $k$ (días)", "horizon $k$ (days)"))
ax[1].set_ylabel(comun.t("varianza / varianza diaria", "variance / daily variance"))
ax[1].set_title(comun.t("Desplazamiento cuadrático medio", "Mean squared displacement"))
ax[1].legend(frameon=False)
comun.guardar(fig, "fig1_difusion")

# ---------------------------------------------------------------------------
# 5. ¿Son gaussianos los pasos?
# ---------------------------------------------------------------------------
z = (r - media) / sigma                      # retorno medido en "desviaciones típicas"
resultados["curtosis_exceso"] = stats.kurtosis(z)          # 0 para una gaussiana
for umbral in (3, 4, 5):
    observados = int(np.sum(np.abs(z) > umbral))
    esperados = n * 2 * stats.norm.sf(umbral)              # lo que daría una gaussiana
    resultados[f"dias_mas_de_{umbral}sigma_obs"] = observados
    resultados[f"dias_mas_de_{umbral}sigma_gauss"] = esperados
# ¿Cada cuántos años ocurriría un día de más de 5 sigma si todo fuera gaussiano?
resultados["anos_entre_5sigma_si_gauss"] = 1 / (2 * stats.norm.sf(5)) / 252

fig, ax = plt.subplots(figsize=(4.6, 2.8))
bordes = np.linspace(-8, 8, 65)
ax.hist(z, bins=bordes, density=True, color=comun.AZUL, alpha=0.6, label="Google")
x = np.linspace(-8, 8, 400)
ax.plot(x, stats.norm.pdf(x), color=comun.NARANJA, lw=1.4, label=comun.t("gaussiana", "Gaussian"))
ax.set_yscale("log")
ax.set_ylim(1e-4, 1)
ax.set_xlabel(comun.t("retorno diario (en desviaciones típicas)", "daily return (in standard deviations)"))
ax.set_ylabel(comun.t("densidad de probabilidad", "probability density"))
ax.legend(frameon=False)
comun.guardar(fig, "fig1_colas")

# ---------------------------------------------------------------------------
print("\nRESULTADOS (paseo aleatorio)")
for k, v in resultados.items():
    print(f"  {k:32s} {v:10.4g}" if isinstance(v, float) else f"  {k:32s} {v:10d}")
