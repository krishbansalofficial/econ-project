# Regime-Aware SPY Trading Strategy Using Hidden Markov Models

This project develops a regime-aware trading framework for U.S. equity markets using Hidden Markov Models (HMMs), clustering benchmarks, and structural break detection methods.

The study uses daily SPY ETF data beginning in 2015 and combines statistical modeling, machine learning, and financial time series analysis to model changing market environments.


## Research Question

Does a regime-aware trading framework using Hidden Markov Models improve risk-adjusted investment performance relative to static benchmark strategies such as buy-and-hold investing, momentum trading, mean reversion trading, and clustering-based approaches?




## Quick Start

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python main.py
```

---

## Output Files

Generated outputs are automatically saved into the following folders:

### `data/`


* Downloaded SPY market data
* Engineered feature datasets

### `figures/`



* Price charts with detected regimes
* Regime probability visualizations
* Equity curves
* Drawdown plots
* Transition matrix heatmaps
* Regime interpretation charts



