# Sem-project-work-by-me-
Code as a data analyst topic is  Sales Performance &amp; Revenue Leakage Analysis (Business Intelligence Focus)
# Sales Performance & Revenue Leakage Analysis
# my project for BI class

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- step 1: make some fake sales data ----------
# (if you have a real csv just use: df = pd.read_csv("sales.csv"))
np.random.seed(1)
n = 2000

df = pd.DataFrame()
df["order_id"] = range(1, n + 1)
df["date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 365, n), unit="D")
df["region"] = np.random.choice(["North", "South", "East", "West"], n)
df["product"] = np.random.choice(["Laptop", "Phone", "Tablet", "Headphones"], n)
df["salesperson"] = np.random.choice(["Aman", "Riya", "Karan", "Sneha", "Rohit", "Neha"], n)
df["quantity"] = np.random.randint(1, 20, n)

prices = {"Laptop": 800, "Phone": 500, "Tablet": 300, "Headphones": 80}
df["price"] = df["product"].map(prices)

# discount between 0% and 35%
df["discount"] = np.random.uniform(0, 0.35, n).round(2)

# some orders never got invoiced, some not paid, some returned
df["invoiced"] = np.random.choice([True, False], n, p=[0.96, 0.04])
df["paid"] = np.random.choice([True, False], n, p=[0.92, 0.08])
df["returned"] = np.random.choice([True, False], n, p=[0.05, 0.95])

print("first 5 rows:")
print(df.head())
print()

# ---------- step 2: calculate revenue ----------
df["gross_revenue"] = df["quantity"] * df["price"]
df["discount_amount"] = df["gross_revenue"] * df["discount"]
df["net_revenue"] = df["gross_revenue"] - df["discount_amount"]

# company policy: max discount is 15%
policy_limit = 0.15
df["extra_discount"] = 0.0
for i in range(len(df)):
    if df.loc[i, "discount"] > policy_limit:
        df.loc[i, "extra_discount"] = df.loc[i, "gross_revenue"] * (df.loc[i, "discount"] - policy_limit)

# ---------- step 3: find the leakage ----------
# not invoiced = we lost that money
df["unbilled_loss"] = 0.0
df.loc[df["invoiced"] == False, "unbilled_loss"] = df["net_revenue"]

# returned orders (only if it was invoiced)
df["return_loss"] = 0.0
df.loc[(df["returned"] == True) & (df["invoiced"] == True), "return_loss"] = df["net_revenue"]

# invoiced but customer didnt pay (and not returned, so we dont count twice)
df["unpaid_loss"] = 0.0
df.loc[(df["invoiced"] == True) & (df["paid"] == False) & (df["returned"] == False), "unpaid_loss"] = df["net_revenue"]

df["total_leakage"] = df["discount_amount"] + df["unbilled_loss"] + df["return_loss"] + df["unpaid_loss"]
df["final_revenue"] = df["gross_revenue"] - df["total_leakage"]

# ---------- step 4: overall numbers ----------
gross = df["gross_revenue"].sum()
final = df["final_revenue"].sum()
leak = df["total_leakage"].sum()

print("===== OVERALL =====")
print("Gross revenue:", round(gross))
print("Total discounts:", round(df["discount_amount"].sum()))
print("  (extra discount above policy:", round(df["extra_discount"].sum()), ")")
print("Unbilled loss:", round(df["unbilled_loss"].sum()))
print("Return loss:", round(df["return_loss"].sum()))
print("Unpaid loss:", round(df["unpaid_loss"].sum()))
print("Final revenue:", round(final))
print("Leakage %:", round(leak / gross * 100, 2), "%")
print()

# ---------- step 5: performance by group ----------
print("===== BY REGION =====")
by_region = df.groupby("region")[["gross_revenue", "total_leakage", "final_revenue"]].sum()
by_region["leakage_%"] = (by_region["total_leakage"] / by_region["gross_revenue"] * 100).round(2)
print(by_region)
print()

print("===== BY PRODUCT =====")
by_product = df.groupby("product")[["gross_revenue", "total_leakage", "final_revenue"]].sum()
by_product["leakage_%"] = (by_product["total_leakage"] / by_product["gross_revenue"] * 100).round(2)
print(by_product)
print()

print("===== BY SALESPERSON =====")
by_sp = df.groupby("salesperson").agg(
    orders=("order_id", "count"),
    gross=("gross_revenue", "sum"),
    leakage=("total_leakage", "sum"),
    avg_discount=("discount", "mean"),
    extra_discount=("extra_discount", "sum"),
)
by_sp["leakage_%"] = (by_sp["leakage"] / by_sp["gross"] * 100).round(2)
by_sp = by_sp.sort_values("extra_discount", ascending=False)
print(by_sp.round(2))
print()
print("Salesperson giving most extra discount:", by_sp.index[0])
print()

# ---------- step 6: orders that look suspicious ----------
# discount more than 30% = needs checking
suspicious = df[df["discount"] > 0.30]
print("Orders with discount > 30%:", len(suspicious))
print(suspicious[["order_id", "salesperson", "product", "discount", "extra_discount"]].head(10))
print()

# ---------- step 7: charts ----------
# chart 1 - where does the money go
labels = ["Discounts", "Unbilled", "Returns", "Unpaid"]
values = [df["discount_amount"].sum(), df["unbilled_loss"].sum(),
          df["return_loss"].sum(), df["unpaid_loss"].sum()]

plt.figure(figsize=(7, 5))
plt.bar(labels, values, color=["orange", "red", "purple", "gray"])
plt.title("Revenue Leakage by Type")
plt.ylabel("Amount")
plt.savefig("chart1_leakage_types.png")
plt.show()

# chart 2 - monthly revenue
df["month"] = df["date"].dt.month
monthly = df.groupby("month")[["gross_revenue", "final_revenue"]].sum()

plt.figure(figsize=(8, 5))
plt.plot(monthly.index, monthly["gross_revenue"], marker="o", label="Gross")
plt.plot(monthly.index, monthly["final_revenue"], marker="o", label="Final")
plt.title("Monthly Revenue: Gross vs Final")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.legend()
plt.savefig("chart2_monthly.png")
plt.show()

# chart 3 - leakage % by salesperson
plt.figure(figsize=(8, 5))
plt.bar(by_sp.index, by_sp["leakage_%"], color="teal")
plt.title("Leakage % by Salesperson")
plt.ylabel("Leakage %")
plt.savefig("chart3_salesperson.png")
plt.show()

# ---------- step 8: save results ----------
df.to_csv("sales_with_leakage.csv", index=False)
by_sp.to_csv("salesperson_report.csv")
print("done! files saved.")

# ---------- conclusion (my findings) ----------
# 1. discounts are the biggest source of leakage
# 2. some salespeople go above the 15% policy a lot -> need approval rules
# 3. unbilled orders are pure loss, billing team should check them
# 4. unpaid invoices need follow up
