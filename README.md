# Thai Dividend Strategy Portfolio

This project is an interactive Streamlit web application for analyzing and constructing high-dividend stock portfolios from the Thai SET market. It provides advanced screening, portfolio optimization, and performance analysis tools for dividend-focused investors.

## Features

- **Upload Thai stock data** (CSV format)
- **Data cleaning and diagnostics** with detailed feedback
- **Stock screening** by market cap, liquidity, dividend yield, volatility, and more
- **Multiple portfolio strategies**:  
  - Modern Portfolio Theory (MPT)
  - Equal Weight
  - Market Cap Weighted
  - Dividend Weighted
  - Low Volatility
  - Quality Factor
  - Momentum
  - Risk Parity
- **Efficient frontier visualization**
- **Portfolio performance metrics**
- **Debugging tools** to help diagnose data and screening issues
- **Large file support** (up to 1GB with config)

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd thai-dividend-strategy-portfolio
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run thai_dividend_app.py
```

#### For large files (>200MB):

- Run with increased upload limit:
  ```bash
  streamlit run thai_dividend_app.py --server.maxUploadSize=1000
  ```
- Or add to `.streamlit/config.toml`:
  ```toml
  [server]
  maxUploadSize = 1000
  ```

## Data Format

Your CSV file should include at least the following columns:

- `PRICINGDATE` (date, e.g. 2024-01-01)
- `CIQ_TICKER` (stock ticker)
- `COMPANYNAME`
- `PRICECLOSE`
- `DIVADJPRICE`
- `MARKETCAP` (in THB)
- `VOLUME`

**Tip:** Make sure there are no missing values in these columns for best results.

## Usage

1. Launch the app.
2. Upload your Thai SET stock data CSV file.
3. Adjust screening and portfolio parameters in the sidebar.
4. Explore the tabs for portfolio construction, efficient frontier, performance analysis, and more.

## Troubleshooting

- If you see "Data Loading Failed - Empty Result", check your CSV for missing values or incorrect column names.
- Use the debug tools in the app to diagnose data issues.

## License

MIT License

---

**Developed for educational