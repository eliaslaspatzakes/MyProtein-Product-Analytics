import streamlit as st
import requests as req
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from itertools import zip_longest
import re
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. CONFIGURATION & THEME
# ------------------------------------------------------------------
st.set_page_config(page_title="MyProtein Analytics", layout="wide")

# GLOBAL PALETTE VARIABLE
PROJECT_PALETTE = "viridis" 
HEATMAP_CMAP = "viridis"
DATAFRAME_CMAP = "Greens" 

# Custom CSS for polished look
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #2e7d32; 
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. HELPER FUNCTIONS (Scraping & Parsing)
# ------------------------------------------------------------------
def fetch_soup(url: str, parser: str = "html.parser"):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = req.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.content, parser)
    except Exception as e:
        st.error(f"Error fetching {url}: {e}")
        return None

def _parse_price(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    m = re.search(r"[-+]?\d[\d.,\s]*", s)
    if not m:
        return None
    num = m.group(0).replace(" ", "")
    if "," in num and "." in num:
        num = num.replace(",", "")
    elif "," in num and "." not in num:
        num = num.replace(".", "") 
        num = num.replace(",", ".")
    else:
        parts = num.split(".")
        if len(parts) > 2:
            num = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(num)
    except ValueError:
        return None

def _parse_rating(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value)
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    token = m.group(0).replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None

def build_products_df(p_title, f_price, f_r, kind_name):
    rows = list(zip_longest(p_title, f_price, f_r, fillvalue=np.nan))
    df = pd.DataFrame(rows, columns=["product", "price", "rating"])
    df["kind"] = kind_name
    return df

# ------------------------------------------------------------------
# 3. DATA LOADING (Cached)
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    urls = {
        "whey": "https://www.myprotein.gr/c/nutrition/protein/whey-protein/",
        "clear": "https://www.myprotein.gr/c/clear-protein/products/",
        "isolate": "https://www.myprotein.gr/c/nutrition/protein/protein-isolate/",
        "blends": "https://www.myprotein.gr/c/nutrition/protein/blends/",
        "milk": "https://www.myprotein.gr/c/nutrition/protein/milk-protein/",
        "diet": "https://www.myprotein.gr/c/nutrition/protein/diet/",
        "vegan": "https://www.myprotein.gr/c/nutrition/protein/vegan-protein/"
    }

    all_dfs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(urls)
    
    for i, (kind, url) in enumerate(urls.items()):
        status_text.text(f"Scraping {kind}...")
        soup = fetch_soup(url)
        if soup:
            titles = [p.text for p in soup.find_all("a", class_="product-item-title mx-auto text-lg my-2 lg:text-[1.25rem] w-full title-font font-bold tracking-wider")]
            prices = [p.text for p in soup.find_all("span", class_="price font-semibold my-auto text-xl")]
            ratings = [r.text for r in soup.find_all("a", class_="reviews pb-2 flex items-center gap-2")]
            
            df = build_products_df(titles, prices, ratings, kind)
            all_dfs.append(df)
        
        progress_bar.progress((i + 1) / total)

    try:
        col_url = "https://www.myprotein.gr/p/sports-nutrition/collagen-protein/12457326/"
        soup_col = fetch_soup(col_url)
        if soup_col:
            t = soup_col.find("h1", class_="text-primary text-xl md:text-2xl font-semibold").text
            p = soup_col.find("span", class_="price font-semibold my-auto text-2xl").text
            r = soup_col.find("div", class_="flex flex-row justify-start items-center gap-2").text
            df_col = pd.DataFrame([{"product": t, "price": p, "rating": r, "kind": "collagen"}])
            all_dfs.append(df_col)
    except:
        pass

    progress_bar.empty()
    status_text.empty()

    if not all_dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df["price"] = full_df["price"].map(_parse_price)
    full_df["rating"] = full_df["rating"].map(_parse_rating)
    return full_df

# ------------------------------------------------------------------
# 4. MAIN APP LOGIC
# ------------------------------------------------------------------
st.title("💪 MyProtein Product Analytics")
st.markdown("Live market analysis of protein supplements from MyProtein GR.")

try:
    with st.spinner('Fetching live data...'):
        df = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if df.empty:
    st.stop()

df_clean = df.dropna(subset=["price", "rating"]).copy()

# ------------------------------------------------------------------
# 5. SIDEBAR FILTERS
# ------------------------------------------------------------------
st.sidebar.header("Filter Options")

with st.sidebar.expander("📂 Select Categories", expanded=True):
    all_kinds = sorted(df_clean["kind"].unique())
    selected_kinds = st.multiselect(
        "Choose types:",
        options=all_kinds,
        default=all_kinds,
        label_visibility="collapsed"
    )

if not df_clean.empty:
    min_p, max_p = float(df_clean["price"].min()), float(df_clean["price"].max())
    price_range = st.sidebar.slider("Price Range (€)", min_value=min_p, max_value=max_p, value=(min_p, max_p))
    min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.0, step=0.1)
else:
    price_range = (0, 100)
    min_rating = 0

filtered_df = df_clean[
    (df_clean["kind"].isin(selected_kinds)) &
    (df_clean["price"].between(price_range[0], price_range[1])) &
    (df_clean["rating"] >= min_rating)
]

# ------------------------------------------------------------------
# 6. DASHBOARD TABS
# ------------------------------------------------------------------

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Products Found", len(filtered_df))
c2.metric("Avg Price", f"€{filtered_df['price'].mean():.2f}")
c3.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f} / 5.0")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "💰 Price Analysis", "⭐ Value for Money", "📋 Raw Data"])

# --- TAB 1: OVERVIEW ---
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Product Count by Category")
        if not filtered_df.empty:
            count_data = filtered_df.groupby("kind").size().reset_index(name="count").sort_values("count", ascending=False)
            
            fig_count, ax = plt.subplots(figsize=(6,4))
            sns.barplot(data=count_data, x="kind", y="count", palette=PROJECT_PALETTE, ax=ax)
            
            for container in ax.containers:
                ax.bar_label(container, fontsize=10, padding=3)
            
            plt.xticks(rotation=45)
            sns.despine()
            st.pyplot(fig_count)

    with col2:
        st.subheader("Median Price per Category")
        if not filtered_df.empty:
            median_data = filtered_df.groupby("kind")["price"].median().reset_index().sort_values("price", ascending=False)
            
            fig_med, ax = plt.subplots(figsize=(6,4))
            sns.barplot(data=median_data, x="kind", y="price", palette=PROJECT_PALETTE, ax=ax)
            
            for container in ax.containers:
                ax.bar_label(container, fmt='€%.2f', fontsize=10, padding=3)
            
            plt.xticks(rotation=45)
            plt.ylabel("Median Price (€)")
            sns.despine()
            st.pyplot(fig_med)
    
    st.subheader("Correlation: Price vs Rating")
    col3, col4 = st.columns([1, 2])
    with col3:
        if not filtered_df.empty:
            fig_corr, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(filtered_df[["price", "rating"]].corr(), annot=True, cmap=HEATMAP_CMAP, vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig_corr)
    with col4:
        st.info("💡 **Insight:** The heatmap shows the relationship between Price and Rating. A value close to 0 means they are not related—paying more doesn't guarantee a higher user rating.")

# --- TAB 2: PRICE ANALYSIS (UPDATED) ---
with tab2:
    st.subheader("Price Distribution (Boxplot)")
    
    if not filtered_df.empty:
        fig_box, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=filtered_df, x="kind", y="price", palette=PROJECT_PALETTE, ax=ax)
        plt.xticks(rotation=45)
        sns.despine()
        st.pyplot(fig_box)

        # --- AUTOMATED INSIGHTS GENERATOR ---
        grp = filtered_df.groupby("kind")["price"]
        median_prices = grp.median().sort_values(ascending=False)
        price_std = grp.std().sort_values(ascending=False)
        
        most_expensive = median_prices.index[0]
        most_expensive_val = median_prices.iloc[0]
        cheapest = median_prices.index[-1]
        cheapest_val = median_prices.iloc[-1]
        most_variable = price_std.index[0] # Highest Standard Deviation

        st.markdown("### 📝 Market Insights Summary")
        st.info(f"""
        Based on the data shown in the box plot above:
        
        * **Premium Segment:** The **{most_expensive}** category commands the highest typical price, with a median cost of **€{most_expensive_val:.2f}**.
        * **Budget Friendly:** For the most affordable option, look at **{cheapest}**, which has a median price of **€{cheapest_val:.2f}**.
        * **Price Variability:** The **{most_variable}** category shows the widest spread in pricing. This indicates a diverse range of products, from budget-friendly items to high-end premium versions.
        * **Outliers:** The  markers (o) in the chart represent outliers—products that are significantly more expensive than the average for their category.
        """)

    else:
        st.info("No data available.")

# --- TAB 3: VALUE FOR MONEY ---
with tab3:
    st.subheader("💎 Value for Money Index")
    st.markdown("We identify hidden gems by calculating: `Score = Rating / Price`. A high score means you get **high ratings for a low price**.")
    
    if not filtered_df.empty:
        medians = filtered_df.groupby("kind")[["price", "rating"]].median().rename(columns={"price": "med_price", "rating": "med_rating"})
        df_vfm = filtered_df.merge(medians, on="kind", how="left")
        df_vfm["value_score"] = df_vfm["rating"] / df_vfm["price"]
        
        vfm_display = df_vfm.sort_values("value_score", ascending=False)[["kind", "product", "price", "rating", "value_score"]]
        
        st.dataframe(
            vfm_display.style
            .format({"price": "€{:.2f}", "rating": "{:.2f}", "value_score": "{:.2%}"})
            .background_gradient(subset=["value_score"], cmap=DATAFRAME_CMAP),
            use_container_width=True,
            height=600
        )
    else:
        st.info("No data available.")

# --- TAB 4: RAW DATA ---
with tab4:
    st.dataframe(filtered_df)