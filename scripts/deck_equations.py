"""Equation images for the deck -> figures/deck_eq_*.png (600 dpi).

    python deck_equations.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / 'figures'
plt.rcParams.update({'mathtext.fontset': 'cm', 'figure.facecolor': 'white'})
EQUATIONS = {
    'deck_eq_bank': [r'$\boldsymbol{\psi}_{\gamma}(t)=\sum_{k=0}^{\infty}\gamma^{k}\,\boldsymbol{\phi}_{t+k},\qquad \gamma\in\{\gamma_1,\ldots,\gamma_{40}\}$',
                r'$\boldsymbol{\phi}_t=[\,x,\;y,\;\dot{x},\;\dot{y},\;a_1,\ldots,a_6\,]\in\mathbb{R}^{10}$'],
    'deck_eq_rule': [r'$\mathbf{b}_i(t)=\frac{\boldsymbol{\psi}_{\gamma_{i+1}}(t)-\boldsymbol{\psi}_{\gamma_i}(t)}{|\ln\gamma_{i+1}-\ln\gamma_i|},\qquad i=1,\ldots,39$',
                r'$c_i(t)=1-\cos\angle(\mathbf{b}_i(t),\,\mathbf{b}_i(t+1))$',
                r'$\mathrm{event\ at\ }t\ \Leftrightarrow\ c_i(t)\geq P_{90}(c_i)\ \mathrm{for}\ \geq 3\ \mathrm{adjacent\ bands}\ i$'],
}
for name, lines in EQUATIONS.items():
    fig = plt.figure(figsize=(7, 0.62 * len(lines)))
    for j, line in enumerate(lines):
        fig.text(0.0, 1 - (j + .5) / len(lines), line, fontsize=17, color='#1a1a1a', va='center')
    fig.savefig(OUT / f'{name}.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(name)
