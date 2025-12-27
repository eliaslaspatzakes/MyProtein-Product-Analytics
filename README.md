# 🥤 MyProtein Real-Time Analytics Dashboard

![Status](https://img.shields.io/badge/Status-Live-success)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scraping](https://img.shields.io/badge/Scraper-BeautifulSoup4-green)

> **A data-driven approach to buying supplements.** > *Stop guessing. Start optimizing based on Price-Per-Rating.*

---

## 🧐 The Problem
The supplement market is flooded with hundreds of products, fluctuating prices, and confusing marketing. Users often struggle to answer simple questions:
* *"Is this 'Gold Standard' whey actually worth €20 more than the regular one?"*
* *"Which vegan protein has the highest user rating but the lowest cost?"*

## 💡 The Solution
I built a **Real-Time Analytical Dashboard** that scrapes the MyProtein website on-demand. It cleans the data, performs statistical analysis, and visualizes the market landscape to help users find the **"Best Value"** products instantly.

### 🔗 [View Live Demo](#) *(Add your Streamlit Cloud link here)*

---

## ⚙️ Technical Architecture

The app follows a functional ETL (Extract, Transform, Load) pipeline that runs in real-time.

```mermaid
graph LR
    A[User Visits App] -->|Trigger| B(Scraper Engine)
    B -->|Fetch HTML| C{MyProtein Website}
    C -->|Raw HTML| B
    B -->|Parse with lxml| D[Data Cleaning]
    D -->|Pandas| E{Statistical Engine}
    E -->|Calculates Z-Scores & VFM| F[Streamlit Dashboard]
