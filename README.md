# Statistical Physics of Financial Markets: Brownian Motion, Volatility as Temperature, Ornstein–Uhlenbeck and Ising Models

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Topic: Econophysics](https://img.shields.io/badge/topic-econophysics-orange.svg)](https://github.com/topics/econophysics)

**Econophysics with Python**: four core models of **statistical mechanics** and **condensed matter
physics** applied to real stock market data (Google 2004–2008, S&P 500 and Nasdaq 1990–2022).
Each model is implemented as a short, documented Python script and tested against the data:
the **random walk / diffusion** model of prices, **volatility as a fluctuating temperature**
(superstatistics and GARCH), **mean reversion as an overdamped Langevin / Ornstein–Uhlenbeck process**,
and a **mean-field Ising model of herding** that produces bubbles, crashes and hysteresis.

**Author:** Jesús García Izquierdo, M.Sc. student in Condensed Matter Physics.

📄 **Full write-up (PDF, 20 pages):**
[Four ideas from statistical physics, tested on stock prices](document_en/Four_ideas_statistical_physics_markets.pdf)

![Hysteresis in a mean-field Ising model of a financial market](img/fig4_respuesta.png)

<!-- GIF placeholder: add an animation of the Ising market here, e.g.
![Ising market simulation](img/ising_market.gif) -->

---

## Contents

- [Physical problem](#physical-problem)
- [Models and main results](#models-and-main-results)
- [Quick start](#quick-start)
- [Repository structure](#repository-structure)
- [Theory and references](#theory-and-references)
- [How to cite](#how-to-cite)
- [License](#license)

---

## Physical problem

A financial market is a many-body system: millions of agents interact through a single
observable, the price. As in statistical mechanics, the microscopic dynamics of each agent is
unknown, but collective regularities can be described with a small number of macroscopic
variables. This project asks **which statistical-physics models describe real price data,
where they fail, and why**, using:

| Physics concept | Market observable |
|---|---|
| Brownian motion, diffusion coefficient $D=\sigma^2/2$ | log-price, daily returns |
| Temperature of a heat bath, $\sigma^2 \leftrightarrow k_BT$ | volatility |
| Overdamped Langevin equation, Fokker–Planck, Boltzmann distribution | mean-reverting price spreads |
| Ising model, mean-field (Weiss) theory, susceptibility, Landau free energy | herding, bubbles, crashes |

## Models and main results

| Script | Model | Result on real data |
|---|---|---|
| `01_paseo_aleatorio.py` | **Random walk / diffusion** | No linear memory ($C(1)=0.024$) and variance growing linearly in time (log–log slope 1.10). Fat tails: 4 days beyond $5\sigma$ in 4 years, versus one every ~6900 years for a Gaussian. |
| `02_temperatura.py` | **Volatility as temperature** (superstatistics, GARCH) | Volatility clustering ($C_{\lvert r\rvert}(1)=0.15$). A mixture of temperatures generates fat tails. An out-of-sample GARCH forecast predicts volatility; the sign of returns remains unpredictable. |
| `03_muelle.py` | **Ornstein–Uhlenbeck / Langevin** mean reversion | Exact AR(1) estimation recovers the relaxation time. The Nasdaq–S&P 500 spread (15-month half-life) does not pass a random-walk null test ($p=0.12$). |
| `04_ising.py` | **Mean-field Ising model** of herding | Spontaneous consensus below $T_c$ (bubbles), sudden flips (crashes), susceptibility $\chi=1/(T-T_c)$ and hysteresis (crash as a first-order transition). |

![Volatility is predictable, the direction of the price is not](img/fig2_prediccion.png)

## Quick start

Requirements: Python ≥ 3.10, `numpy`, `scipy`, `matplotlib`. The datasets ship with `matplotlib`,
so no download is needed.

```bash
git clone https://github.com/jesus-garcia-izquierdo/statistical-physics-of-markets.git
cd statistical-physics-of-markets
pip install -r requirements.txt

python 01_paseo_aleatorio.py      # random walk and diffusion
python 02_temperatura.py          # volatility as temperature, GARCH
python 03_muelle.py               # Ornstein–Uhlenbeck mean reversion
python 04_ising.py                # mean-field Ising market (~15 s)
```

Each script saves its figures and prints every number quoted in the write-up. Figure labels are
in Spanish by default (`figuras/`); for English labels (`figures/`) run:

```bash
IDIOMA=en python 01_paseo_aleatorio.py
```

<!-- Colab placeholder: once a notebook is added (e.g. notebooks/quickstart.ipynb), insert:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jesus-garcia-izquierdo/statistical-physics-of-markets/blob/main/notebooks/quickstart.ipynb)
-->

## Repository structure
statistical-physics-of-markets/
├── comun.py # data loading and plotting style (shared)
├── 01_paseo_aleatorio.py # Brownian motion, diffusion, fat tails
├── 02_temperatura.py # volatility as temperature, superstatistics, GARCH
├── 03_muelle.py # Langevin / Ornstein–Uhlenbeck, Boltzmann distribution
├── 04_ising.py # mean-field Ising model, susceptibility, hysteresis
├── figures/ , figuras/ # generated figures (English / Spanish labels)
├── img/ # images used in this README
├── document_en/ # LaTeX source and PDF of the write-up
├── requirements.txt
└── LICENSE


## Theory and references

The write-up develops the theory from the general framework to the specific market system:
statistical ensembles and the Boltzmann distribution, Brownian motion and the diffusion
equation, the Langevin and Fokker–Planck equations with the fluctuation–dissipation relation,
superstatistics, and mean-field theory of phase transitions (Weiss, Landau, Ising).

Main references:
- R. N. Mantegna and H. E. Stanley, *An Introduction to Econophysics*, Cambridge University Press (2000).
- J.-P. Bouchaud and M. Potters, *Theory of Financial Risk and Derivative Pricing*, Cambridge University Press (2003).
- C. Beck and E. G. D. Cohen, "Superstatistics", *Physica A* 322, 267 (2003).
- S. Bornholdt, "Expectation bubbles in a spin model of markets", *Int. J. Mod. Phys. C* 12, 667 (2001).

## How to cite

```bibtex
@misc{garciaizquierdo2026markets,
  author       = {García Izquierdo, Jesús},
  title        = {Statistical Physics of Financial Markets: Four Ideas Tested on Stock Prices},
  year         = {2026},
  howpublished = {\url{https://github.com/jesus-garcia-izquierdo/statistical-physics-of-markets}}
}
```

## License

Released under the [MIT License](LICENSE). This is a physics research project, not investment advice.

**Keywords:** econophysics, statistical mechanics, condensed matter physics, Brownian motion,
random walk, diffusion, Langevin equation, Fokker–Planck equation, Ornstein–Uhlenbeck process,
superstatistics, GARCH, volatility clustering, fat tails, Ising model, mean-field theory, phase
transitions, financial markets, Python.