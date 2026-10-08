import pandas as pd

orders = pd.read_csv("orders.csv")
monthly = pd.read_csv("monthly_revenue.csv")
customers = pd.read_csv("customers.csv")
produk = pd.read_csv("product_summary.csv")
print("ORDERS:", orders.shape)
print("Order ID unik:", orders["order_id"].nunique())
print("Customer ID pada orders:", orders["customer_id"].nunique())
print("MONTHLY:", monthly.shape)
print(monthly[["year", "month", "orders", "revenue_usd"]].head())
januari = orders[(orders["year"] == 2020) & (orders["month"] == 1)]
delivered_januari = januari[januari["order_status"] == "Delivered"]
print("Semua order Januari:", len(januari))
print(januari["order_status"].value_counts())
print("Delivered Januari:", len(delivered_januari))
print(
    "Revenue Januari Delivered:", round(delivered_januari["total_amount_usd"].sum(), 2)
)
print("CUSTOMERS:", customers.shape)
print("Customer ID unik:", customers["customer_id"].nunique())
orders_pelanggan = orders.merge(
    customers[["customer_id", "country", "membership_tier"]],
    on="customer_id",
    how="left",
    validate="many_to_one",
)
print("Setelah join:", orders_pelanggan.shape)
print("Country kosong:", orders_pelanggan["country"].isna().sum())
print("Tier kosong:", orders_pelanggan["membership_tier"].isna().sum())
delivered = orders_pelanggan[orders_pelanggan["order_status"] == "Delivered"]
revenue_country = (
    delivered.groupby("country")["total_amount_usd"].sum().sort_values(ascending=False)
)
print("Revenue Delivered per negara:")
print(revenue_country.head(5))
status = pd.crosstab(
    orders_pelanggan["country"], orders_pelanggan["order_status"]
).reindex(columns=["Delivered", "Processing", "Cancelled", "Returned"], fill_value=0)
status["Total"] = status.sum(axis=1)
status["Persen Delivered"] = (status["Delivered"] / status["Total"] * 100).round(1)
print("Status per negara:")
print(status.sort_values("Total", ascending=False).head(5).to_string())
print("Rating kosong:", orders["customer_rating"].isna().sum())
print("Rating nol:", orders["customer_rating"].eq(0).sum())
print("Rata-rata rating tercatat:", round(orders["customer_rating"].mean(), 2))
print("PRODUCT SUMMARY:", produk.shape)
print("Produk unik:", produk["product_name"].nunique())
Halaman: 16
print(produk[["category", "product_name", "total_orders", "total_revenue_usd"]].head())
