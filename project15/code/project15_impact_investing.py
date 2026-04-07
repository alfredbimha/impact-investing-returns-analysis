"""
===============================================================================
PROJECT 15: Impact Investing Returns Analysis & Portfolio Optimization
===============================================================================
RESEARCH QUESTION:
    Do impact/ESG funds sacrifice returns? Can we construct an optimal
    portfolio combining impact and conventional funds?
METHOD:
    Risk-adjusted performance metrics, mean-variance optimization
DATA:
    Yahoo Finance — real ESG and conventional ETF data
===============================================================================
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

print("STEP 1: Downloading ESG and conventional fund data...")

funds = {
    'ESGU':'ESG US Equity','ESGD':'ESG Intl Dev','SUSA':'Sustainable USA',
    'ICLN':'Clean Energy','QCLN':'Clean Tech',
    'SPY':'S&P 500','QQQ':'Nasdaq 100','VTI':'Total Market',
    'AGG':'US Bonds','VWO':'EM Equity'
}

prices = {}
for t in funds:
    df = yf.download(t, start='2018-01-01', end='2025-12-31', auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if not df.empty:
        prices[t] = df['Close']
        print(f"  {t} ({funds[t]}): {len(df)} obs")

prices = pd.DataFrame(prices).dropna()
returns = np.log(prices/prices.shift(1)).dropna()
ann_returns = returns.mean() * 252 * 100
ann_vol = returns.std() * np.sqrt(252) * 100
prices.to_csv('data/fund_prices.csv')

print(f"\nSTEP 2: Computing performance metrics...")

rf = 0.04  # Risk-free rate (4%)
metrics = []
for t in returns.columns:
    r = returns[t]
    ann_r = r.mean() * 252
    ann_v = r.std() * np.sqrt(252)
    sharpe = (ann_r - rf) / ann_v
    
    # Sortino (downside deviation)
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = (ann_r - rf) / downside if downside > 0 else np.nan
    
    # Max drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100
    
    # Calmar ratio
    calmar = ann_r / abs(max_dd/100) if max_dd != 0 else np.nan
    
    is_esg = t in ['ESGU','ESGD','SUSA','ICLN','QCLN']
    
    metrics.append({
        'Fund':t, 'Name':funds[t], 'Type':'ESG/Impact' if is_esg else 'Conventional',
        'Annual_Return_pct': round(ann_r*100, 2),
        'Volatility_pct': round(ann_v*100, 2),
        'Sharpe': round(sharpe, 3),
        'Sortino': round(sortino, 3),
        'Max_Drawdown_pct': round(max_dd, 2),
        'Calmar': round(calmar, 3)
    })

metrics_df = pd.DataFrame(metrics).sort_values('Sharpe', ascending=False)
metrics_df.to_csv('output/tables/performance_metrics.csv', index=False)
print(metrics_df[['Fund','Name','Type','Annual_Return_pct','Sharpe','Max_Drawdown_pct']].to_string(index=False))

print(f"\nSTEP 3: Portfolio optimization...")

# Mean-variance optimization
equity_funds = [t for t in returns.columns if t != 'AGG']
mu = returns[equity_funds].mean() * 252
cov = returns[equity_funds].cov() * 252
n = len(equity_funds)

def neg_sharpe(w, mu, cov, rf):
    port_r = np.dot(w, mu)
    port_v = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    return -(port_r - rf) / port_v

constraints = [{'type':'eq','fun':lambda w: np.sum(w)-1}]
bounds = [(0, 0.4)] * n  # Max 40% per fund

# Optimize
x0 = np.ones(n) / n
result = minimize(neg_sharpe, x0, args=(mu, cov, rf), method='SLSQP',
                  bounds=bounds, constraints=constraints)
optimal_w = result.x

# Efficient frontier
target_returns = np.linspace(mu.min(), mu.max(), 50)
frontier = []
for target in target_returns:
    cons = [{'type':'eq','fun':lambda w: np.sum(w)-1},
            {'type':'eq','fun':lambda w, t=target: np.dot(w, mu)-t}]
    res = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov, w))),
                   x0, method='SLSQP', bounds=bounds, constraints=cons)
    if res.success:
        frontier.append({'return': target*100, 'volatility': res.fun*100})

frontier_df = pd.DataFrame(frontier)

# Save optimal portfolio
opt_port = pd.DataFrame({'Fund':equity_funds, 'Weight':optimal_w.round(4)})
opt_port = opt_port[opt_port['Weight'] > 0.01].sort_values('Weight', ascending=False)
opt_port.to_csv('output/tables/optimal_portfolio.csv', index=False)
print("\n  Optimal portfolio weights:")
print(opt_port.to_string(index=False))

print(f"\nSTEP 4: Visualizations...")

# Fig 1: Risk-return scatter
fig, ax = plt.subplots(figsize=(12, 7))
for _, row in metrics_df.iterrows():
    color = '#2ecc71' if row['Type']=='ESG/Impact' else '#3498db'
    marker = 's' if row['Type']=='ESG/Impact' else 'o'
    ax.scatter(row['Volatility_pct'], row['Annual_Return_pct'], s=100, c=color, 
              marker=marker, edgecolors='white', zorder=3)
    ax.annotate(row['Fund'], (row['Volatility_pct'], row['Annual_Return_pct']),
                fontsize=9, ha='center', va='bottom')

if not frontier_df.empty:
    ax.plot(frontier_df['volatility'], frontier_df['return'], 'k--', lw=1.5, 
            alpha=0.5, label='Efficient Frontier')

from matplotlib.lines import Line2D
legend = [Line2D([0],[0],marker='s',color='w',markerfacecolor='#2ecc71',markersize=10,label='ESG/Impact'),
          Line2D([0],[0],marker='o',color='w',markerfacecolor='#3498db',markersize=10,label='Conventional')]
ax.legend(handles=legend, fontsize=11)
ax.set_title('Risk-Return Profile: ESG vs Conventional Funds', fontweight='bold')
ax.set_xlabel('Annualized Volatility (%)'); ax.set_ylabel('Annualized Return (%)')
plt.tight_layout()
plt.savefig('output/figures/fig1_risk_return.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Cumulative returns
fig, ax = plt.subplots(figsize=(14, 7))
cum = (1 + returns).cumprod()
for t in returns.columns:
    is_esg = t in ['ESGU','ESGD','SUSA','ICLN','QCLN']
    ax.plot(cum.index, cum[t], label=funds[t], 
            linestyle='-' if is_esg else '--', lw=1.2 if is_esg else 0.8)
ax.set_title('Cumulative Returns: ESG vs Conventional', fontweight='bold')
ax.set_ylabel('Growth of $1'); ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig('output/figures/fig2_cumulative_returns.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Optimal portfolio pie chart
fig, ax = plt.subplots(figsize=(8, 8))
colors_pie = ['#2ecc71' if f in ['ESGU','ESGD','SUSA','ICLN','QCLN'] else '#3498db' for f in opt_port['Fund']]
ax.pie(opt_port['Weight'], labels=opt_port['Fund'], autopct='%1.1f%%', colors=colors_pie)
ax.set_title('Optimal Portfolio Allocation', fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/fig3_optimal_portfolio.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 4: Drawdown chart
fig, ax = plt.subplots(figsize=(14, 5))
for t in ['ESGU','ICLN','SPY','QQQ']:
    if t in returns.columns:
        cum = (1 + returns[t]).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax() * 100
        ax.plot(dd.index, dd, label=funds[t], lw=1)
ax.set_title('Drawdown Comparison', fontweight='bold')
ax.set_ylabel('Drawdown (%)'); ax.legend()
plt.tight_layout()
plt.savefig('output/figures/fig4_drawdowns.png', dpi=150, bbox_inches='tight')
plt.close()

print("  COMPLETE!")
