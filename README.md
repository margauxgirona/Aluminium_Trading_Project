\# Aluminium Trading Project — ESILV 2026



\*\*Course:\*\* Commodities Markets \& Models  

\*\*Group:\*\* Martin \& Margaux  

\*\*Commodity:\*\* Aluminium  



\## Repository Structure

```

Aluminium\_Trading\_Project/

├── shared\_data/

│   └── prices\_clean.csv        # Shared dataset (14 assets, 2018-2024)

├── martin/

│   └── trading\_game\_1/

│       └── Game1\_Martin\_MarketNeutral\_1.ipynb

└── margaux/

&#x20;   └── trading\_game\_2/

&#x20;       ├── step1\_data.py

&#x20;       ├── step2\_cointegration.py

&#x20;       └── step3\_strategy.py

```



\## Trading Game #1 — Martin

Optimal Market-Neutral Trading (Yang \& Malik, 2024)  

Run: open `Game1\_Martin\_MarketNeutral\_1.ipynb` in Jupyter



\## Trading Game #2 — Margaux

Beating Passive Strategies via Pairs Trading (Palazzi, 2025)  

Run in order:

```

python step1\_data.py

python step2\_cointegration.py

python step3\_strategy.py

```

\*\*Results:\*\* Pair AA/ALI=F | Sharpe 3.59 | Ann. Return 124% vs 4.2% buy-and-hold

