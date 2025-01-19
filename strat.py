from datetime import datetime, timedelta
from tqdm import tqdm
from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, PSO, SGA, SA, NELDER_MEAD, TABU, FA
from GQLib.Models import LPPL, LPPLS
from GQLib.enums import InputType
from GQLib.AssetProcessor import AssetProcessor
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

# Initialisation du framework
fw = Framework(frequency="daily", input_type=InputType.BTC)

# Paramètres de la stratégie
start_date = datetime(day=1, month=1, year=2015)
end_date = datetime(day=1, month=9, year=2024)

total_iterations = (end_date - start_date).days // 30 + 1
#total_iterations = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

nb_tc = 10
initial_capital = 100
capital = initial_capital
capital_long = initial_capital

holding = False  # Statut long
shorting = False  # Statut short

current_date = start_date

capital_values = []
positions = []
dates = []
long_only_capital_values = []

# Fonction pour récupérer le prix correspondant à une date
def get_price_at_date(current_date, global_dates, global_prices):
    for i, date in enumerate(global_dates):
        if date >= current_date: 
            return global_prices[i]
    return global_prices[-1]  

entry_price = None
# Boucle principale
with tqdm(total=total_iterations) as pbar:
    
    while current_date <= end_date:
        closing_price = get_price_at_date(current_date, fw.global_dates, fw.global_prices)

        if current_date != start_date:
            old_price = get_price_at_date(current_date - timedelta(days=30), fw.global_dates, fw.global_prices)
            capital_long *= (closing_price / old_price)  

        long_only_capital_values.append(capital_long)
        if holding:
            rdt = (closing_price / entry_price) - 1
            capital *= (1 + rdt)  # Mise à jour du capital pour position longue
            
        elif shorting:
            rdt = (entry_price / closing_price) - 1
            capital *= (1 + rdt)  # Mise à jour du capital pour position short

        entry_price = closing_price

        fw_start_dt = current_date - relativedelta(years=3)
        fw_start = fw_start_dt.strftime("%d/%m/%Y")
        fw_end = current_date.strftime("%d/%m/%Y")

        # Analyse du framework
        file_name = f"Results_strategy/strat_{current_date.strftime('%m-%Y')}"
        best_results = fw.analyze(result_json_name=file_name,significativity_tc=0.3, lppl_model=LPPLS)

        # Analyse des résultats significatifs
        significant_tc = []
        for res in best_results:
            if res["is_significant"]:
                significant_tc.append([res["bestParams"][0], res["power_value"]])

        # Calcul de la date TC pondérée par leur power
        significant_tc_value = None
        try:
            if significant_tc:
                significant_tc = sorted(significant_tc, key=lambda x: x[1], reverse=True)[:min(len(significant_tc), nb_tc)]
                sum_max_power = sum(x[1] for x in significant_tc if x[1] is not None and not np.isnan(x[1]))
                weighted_sum_tc = sum(x[0] * x[1] for x in significant_tc if x[1] is not None and not np.isnan(x[1]))
                significant_tc_value = weighted_sum_tc / sum_max_power if sum_max_power != 0 else None
        except Exception as e:
            print(f"Erreur lors du calcul des TC : {e}")

        # Validation et décision de stratégie
        if significant_tc_value is not None and isinstance(significant_tc_value, float):
            tc_index = int(round(significant_tc_value))
            if 0 <= tc_index < len(fw.global_dates):
                tc_date = fw.global_dates[tc_index]
                diff_date = tc_date - current_date
                tc_price = fw.global_prices[tc_index]
                # Short si diff_date.days < 30
                if diff_date.days < 30 and not shorting:
                    
                    if holding:
                        # Fermer position longue
                        rdt = (tc_price/entry_price)-1
                        capital*= (1+ rdt)
                        print(f"[{tc_date}] Long fermé à {tc_price}, capital = {capital:.2f}")
                        holding = False

                    # Ouvrir une position short
                    entry_price = tc_price  # Stocker le prix d'entrée pour le short
                    shorting = True
                    print(f"[{tc_date}] Short ouvert à {tc_price}")
                
                # Si diff_date.days >= 100, fermer le short
                elif shorting and diff_date.days >= 30:
                    print(f"[{current_date}] Short fermé à {closing_price}, capital = {capital:.2f}")
                    shorting = False
                
        # Si aucune condition de short, ouvrir/maintenir une position longue
        if not shorting and not holding:
            holding = True
            entry_price = closing_price
            print(f"[{current_date}] Long ouvert à {entry_price}")

        capital_values.append(capital)
        positions.append("Long" if holding else ("Short" if shorting else "None"))
        dates.append(current_date.strftime("%Y-%m-%d"))

        current_date += timedelta(days=30)
        pbar.update(1)

dates.append(current_date.strftime("%Y-%m-%d"))
positions.append('None')
closing_price = get_price_at_date(current_date, fw.global_dates, fw.global_prices)
old_price = get_price_at_date(current_date - timedelta(days=30), fw.global_dates, fw.global_prices)
capital_long *= (closing_price / old_price) 
long_only_capital_values.append(capital_long)

# Clôturer toute position ouverte à la fin
if holding:
    rdt = (closing_price/entry_price)-1
    capital*= (1+ rdt)
    print(f"[{current_date}] Position longue clôturée à {closing_price}, capital final = {capital:.2f}")
elif shorting:
    rdt = (entry_price/closing_price)-1
    capital*= (1+ rdt)
    print(f"[{current_date}] Position short clôturée à {closing_price}, capital final = {capital:.2f}")
capital_values.append(capital)
# Résultat final
gain = capital - initial_capital
print(f"Capital initial : {initial_capital:.2f}, Capital final : {capital:.2f}, Gain total : {gain:.2f} ({(gain / initial_capital) * 100:.2f}%)")



fig = go.Figure()


fig.add_trace(go.Scatter(
    x=dates, 
    y=capital_values, 
    mode='lines', 
    name='Strategy', 
    line=dict(color='red')
))
fig.add_trace(go.Scatter(
    x=dates, 
    y=long_only_capital_values, 
    mode='lines', 
    name='Long Only', 
    line=dict(color='blue', dash='dash')
))


current_position = None
start_date = None

for idx, signal in enumerate(positions):
    if signal != current_position:
        # Fin de la zone précédente
        if current_position in ["Short", "Long"] and start_date is not None:
            fig.add_shape(
                type="rect",
                x0=start_date,
                x1=dates[idx],
                y0=min(min(capital_values), min(long_only_capital_values)),
                y1=max(max(capital_values), max(long_only_capital_values)),
                fillcolor="red" if current_position == "Short" else "blue",
                opacity=0.2,
                layer="below",
                line_width=0
            )

        start_date = dates[idx]
        current_position = signal

if current_position in ["Short", "Long"] and start_date is not None:
    fig.add_shape(
        type="rect",
        x0=start_date,
        x1=dates[-1],
        y0=min(min(capital_values), min(long_only_capital_values)),
        y1=max(max(capital_values), max(long_only_capital_values)),
        fillcolor="red" if current_position == "Short" else "blue",
        opacity=0.2,
        layer="below",
        line_width=0
    )


fig.add_trace(go.Scatter(
    x=[None], 
    y=[None], 
    mode='markers', 
    name='Short Zone', 
    marker=dict(size=10, color='red', opacity=0.5)
))
fig.add_trace(go.Scatter(
    x=[None], 
    y=[None], 
    mode='markers', 
    name='Long Zone', 
    marker=dict(size=10, color='blue', opacity=0.5)
))


fig.update_layout(
    title='Strategy Evolution', 
    xaxis=dict(
        title='Date',
        showline=True,
        linecolor='black',
        linewidth=1,
        mirror=True
    ),
    yaxis=dict(
        title="Capital (USD)",
        showline=True,
        linecolor='black',
        linewidth=1,
        mirror=True
    ),
    showlegend=True, 
    plot_bgcolor='white', 
    paper_bgcolor='white'
)

fig.show()