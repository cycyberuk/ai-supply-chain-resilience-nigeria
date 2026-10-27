"""
Inventory Optimisation Module — Module 2
AI-Driven Supply Chain Resilience in African E-Commerce
Input:  nigerian_retail_and_ecommerce_supply_chain_logistics_data.parquet
        module1_demand_forecasts.csv
Output: Table 4 — stockout frequency, fill rate, holding days
"""

import pandas as pd
import numpy as np
import simpy
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================
# 1. LOAD AND PREPARE THE SUPPLY CHAIN DATA
# ============================================================
PARQUET = "nigerian_retail_and_ecommerce_supply_chain_logistics_data.parquet"
df = pd.read_parquet(PARQUET)

# Convert date columns
for col in ["ship_date", "expected_delivery_date", "actual_delivery_date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Derive lead time only for shipments that actually delivered
delivered = df[df["delivery_status"] == "delivered"].copy()
delivered = delivered.dropna(subset=["actual_delivery_date", "ship_date"])
delivered["lead_time_days"] = (
    delivered["actual_delivery_date"] - delivered["ship_date"]
).dt.days

# Filter to plausible lead times (0 to 30 days)
lead_times = delivered["lead_time_days"][
    (delivered["lead_time_days"] >= 0) & (delivered["lead_time_days"] <= 30)
]

print("=== Empirical parameters from dataset ===")
print(f"Delivered shipments used for lead-time: {len(lead_times)}")
print(f"Lead time (days): mean={lead_times.mean():.2f}, std={lead_times.std():.2f}, "
      f"min={lead_times.min()}, max={lead_times.max()}")
print(f"Median lead time: {lead_times.median():.1f} days")

# Cost per unit (proxy for holding cost)
delivered["cost_per_unit"] = delivered["shipping_cost_ngn"] / delivered["quantity"].clip(lower=1)
cost_per_unit = delivered["cost_per_unit"].clip(upper=delivered["cost_per_unit"].quantile(0.99))
print(f"Cost per unit (NGN): mean={cost_per_unit.mean():.2f}, "
      f"median={cost_per_unit.median():.2f}")

# Empirical lead-time distribution (bootstrap from the actual data)
LEAD_TIME_MEAN = float(lead_times.mean())
LEAD_TIME_STD  = float(lead_times.std())
HOLDING_COST_PER_UNIT_PER_DAY = float(cost_per_unit.median()) * 0.01  # 1% of unit cost per day
STOCKOUT_COST_PER_UNIT = float(cost_per_unit.median()) * 2.0          # penalty = 2x unit cost

print(f"\nSimulation parameters:")
print(f"  Holding cost / unit / day: {HOLDING_COST_PER_UNIT_PER_DAY:.2f} NGN")
print(f"  Stockout cost / unit:      {STOCKOUT_COST_PER_UNIT:.2f} NGN")

# ============================================================
# 2. LOAD MODULE 1 DEMAND FORECASTS
# ============================================================
forecasts = pd.read_csv("module1_demand_forecasts.csv", parse_dates=["ds"])
forecasts = forecasts.sort_values("ds").reset_index(drop=True)

# Daily demand = actual y (for simulation) and xgb_pred (for AI policy)
demand_actual = forecasts["y"].values
demand_forecast = forecasts["xgb_pred"].values
N_DAYS = len(forecasts)
print(f"\nDemand series loaded: {N_DAYS} days")
print(f"Mean daily demand: {demand_actual.mean():.1f} units")

# ============================================================
# 3. SIMPY INVENTORY SIMULATION
# ============================================================
class InventorySim:
    def __init__(self, env, demand, forecast, policy="fixed",
                 initial_stock=None, lead_time_mean=LEAD_TIME_MEAN,
                 lead_time_std=LEAD_TIME_STD):
        self.env = env
        self.demand = demand
        self.forecast = forecast
        self.policy = policy
        self.lead_time_mean = lead_time_mean
        self.lead_time_std = lead_time_std

        # Policy parameters
        if policy == "fixed":
            self.rop = int(np.mean(demand) * self.lead_time_mean * 1.2 )    
            self.order_qty = int(np.mean(demand) * 14)  
        else:  # AI-dynamic
            self.rop = None
            self.order_qty = None

        # State
        self.stock = initial_stock if initial_stock is not None else int(np.mean(demand) * 7)
        self.pipeline = []  # list of (arrival_day, qty)
        self.stockout_events = 0
        self.stockout_units = 0
        self.total_demand = 0
        self.units_fulfilled = 0
        self.daily_holding = []
        self.daily_stock = []
        self.on_order = 0

        self.env.process(self.run())

    def place_order(self, qty, day):
        lead_time = max(1, int(np.random.normal(self.lead_time_mean, self.lead_time_std)))
        arrival_day = day + lead_time
        self.pipeline.append((arrival_day, qty))
        self.on_order += qty

    def receive_orders(self, day):
        arrived = [(a, q) for a, q in self.pipeline if a <= day]
        for a, q in arrived:
            self.stock += q
            self.on_order -= q
            self.pipeline.remove((a, q))

    def run(self):
        for day in range(N_DAYS):
            self.receive_orders(day)

            # Policy decision BEFORE demand is realised
            if self.policy == "fixed":
                position = self.stock + self.on_order
                if position <= self.rop:
                    self.place_order(self.order_qty, day)
            else:  # AI-dynamic: use forecast for next few days
                horizon = self.forecast[day:day+7]
                if len(horizon) == 0:
                    horizon = np.array([self.forecast[-1]])
                # Reorder point = forecasted demand over lead time + safety stock
                expected_lt_demand = horizon.mean() * self.lead_time_mean
                safety_stock = 1.65 * np.sqrt(self.lead_time_mean) * horizon.std() if len(horizon) > 1 else 0
                dynamic_rop = expected_lt_demand + safety_stock
                position = self.stock + self.on_order
                if position <= dynamic_rop:
                    order_qty = max(int(horizon.sum() * 2), int(np.mean(self.demand)))
                    self.place_order(order_qty, day)

            # Realise today's demand
            today_demand = int(self.demand[day])
            self.total_demand += today_demand
            fulfilled = min(self.stock, today_demand)
            self.units_fulfilled += fulfilled
            self.stock -= fulfilled
            if today_demand > fulfilled:
                self.stockout_events += 1
                self.stockout_units += (today_demand - fulfilled)

            self.daily_holding.append(self.stock * HOLDING_COST_PER_UNIT_PER_DAY)
            self.daily_stock.append(self.stock)

            yield self.env.timeout(1)

# ============================================================
# 4. RUN SCENARIOS
# ============================================================
def run_scenario(name, demand_multiplier=1.0, lead_time_multiplier=1.0, cost_multiplier=1.0):
    """Run both policies under one scenario, return metrics."""
    global HOLDING_COST_PER_UNIT_PER_DAY
    original_holding = HOLDING_COST_PER_UNIT_PER_DAY

    demand_scen = demand_actual * demand_multiplier
    lead_mean_scen = LEAD_TIME_MEAN * lead_time_multiplier
    HOLDING_COST_PER_UNIT_PER_DAY = original_holding * cost_multiplier

    results = []
    for policy in ["fixed", "ai_dynamic"]:
        env = simpy.Environment()
        sim = InventorySim(
            env, demand_scen, demand_forecast,
            policy=policy,
            lead_time_mean=lead_mean_scen,
            lead_time_std=LEAD_TIME_STD
        )
        env.run(until=N_DAYS)

        fill_rate = sim.units_fulfilled / sim.total_demand if sim.total_demand > 0 else 0
        stockout_freq = sim.stockout_events / N_DAYS
        avg_holding_days = np.mean(sim.daily_stock) / np.mean(demand_scen) if np.mean(demand_scen) > 0 else 0
        total_holding_cost = sum(sim.daily_holding)
        total_stockout_cost = sim.stockout_units * STOCKOUT_COST_PER_UNIT

        results.append({
            "Scenario": name,
            "Policy": "Fixed ROP" if policy == "fixed" else "AI Dynamic",
            "Stockout Frequency (events/day)": round(stockout_freq, 4),
            "Fill Rate": round(fill_rate, 4),
            "Avg Holding Days": round(avg_holding_days, 2),
            "Holding Cost (NGN)": round(total_holding_cost, 0),
            "Stockout Cost (NGN)": round(total_stockout_cost, 0),
            "Total Cost (NGN)": round(total_holding_cost + total_stockout_cost, 0),
        })

    HOLDING_COST_PER_UNIT_PER_DAY = original_holding
    return results

# ============================================================
# 5. EXECUTE ALL SCENARIOS
# ============================================================
all_results = []
all_results += run_scenario("Baseline", demand_multiplier=1.0, lead_time_multiplier=1.0, cost_multiplier=1.0)
all_results += run_scenario("Supplier delay (+3 days)", demand_multiplier=1.0, lead_time_multiplier=(LEAD_TIME_MEAN + 3) / LEAD_TIME_MEAN, cost_multiplier=1.0)
all_results += run_scenario("Demand spike (+40%)", demand_multiplier=1.40, lead_time_multiplier=1.0, cost_multiplier=1.0)
all_results += run_scenario("Transport cost (+20%)", demand_multiplier=1.0, lead_time_multiplier=1.0, cost_multiplier=1.20)

results_df = pd.DataFrame(all_results)

print("\n=== Table 4 (actual numbers) ===")
print(results_df.to_string(index=False))

results_df.to_csv("table4_inventory_results.csv", index=False)
print("\nSaved: table4_inventory_results.csv")

# ============================================================
# 6. SUMMARY: HOW MUCH BETTER IS AI POLICY?
# ============================================================
print("\n=== AI policy improvement over Fixed ROP ===")
for scen in results_df["Scenario"].unique():
    sub = results_df[results_df["Scenario"] == scen]
    fixed = sub[sub["Policy"] == "Fixed ROP"].iloc[0]
    ai    = sub[sub["Policy"] == "AI Dynamic"].iloc[0]
    print(f"\n{scen}:")
    print(f"  Stockout frequency: {fixed['Stockout Frequency (events/day)']:.4f} -> "
          f"{ai['Stockout Frequency (events/day)']:.4f} "
          f"({(1 - ai['Stockout Frequency (events/day)']/max(fixed['Stockout Frequency (events/day)'], 1e-6))*100:.1f}% reduction)")
    print(f"  Fill rate:          {fixed['Fill Rate']:.4f} -> {ai['Fill Rate']:.4f}")
    print(f"  Total cost:         {fixed['Total Cost (NGN)']:.0f} -> {ai['Total Cost (NGN)']:.0f} NGN "
          f"({(1 - ai['Total Cost (NGN)']/max(fixed['Total Cost (NGN)'], 1))*100:.1f}% reduction)")


    # ============================================================
# 8. FORECAST BIAS DURING DEMAND SPIKE
# ============================================================
spike_demand = demand_actual * 1.40
forecast_bias = (spike_demand - demand_forecast).mean()
forecast_bias_pct = forecast_bias / spike_demand.mean() * 100
print(f"\n=== Forecast bias during demand spike (+40%) ===")
print(f"Mean actual demand during spike: {spike_demand.mean():.0f} units/day")
print(f"Mean forecast:                   {demand_forecast.mean():.0f} units/day")
print(f"Mean forecast bias:              {forecast_bias:.0f} units ({forecast_bias_pct:.1f}%)")
print(f"Interpretation: the forecast under-estimates demand by {forecast_bias_pct:.1f}%,")
print(f"which propagates into AI policy under-ordering.")