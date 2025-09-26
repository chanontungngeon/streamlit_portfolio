import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page configuration with large file support
st.set_page_config(
    page_title="Thai SET High Dividend Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Streamlit for large files by default
import os
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '1024'  # 1GB limit

# Try to set additional memory configurations
try:
    import streamlit.web.cli as stcli
    # This will help with large file processing
except ImportError:
    pass

# Note: For large file uploads, run Streamlit with:
# streamlit run thai_dividend_app.py --server.maxUploadSize=1000

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .positive-metric {
        color: #28a745;
        font-weight: bold;
    }
    .negative-metric {
        color: #dc3545;
        font-weight: bold;
    }
    .debug-section {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ThaiDividendStrategyApp:
    """Interactive Thai SET High Dividend Strategy Application with Enhanced Debugging"""
    
    def __init__(self):
        self.df = None
        self.filtered_data = None
        self.portfolio = None
        self.performance_data = None
        self.benchmark_df = None
        self.performance_metrics = {}
        
    @st.cache_data
    def load_data(_self, uploaded_file):
        """Load and cache the dataset with error handling"""
        try:
            if uploaded_file is not None:
                # Reset dataframe
                _self.df = None
                
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                # Check if dataframe is empty
                if df.empty:
                    st.error("❌ CSV file is empty!")
                    return False
                
                # Check required columns
                required_columns = ['PRICINGDATE', 'CIQ_TICKER', 'COMPANYNAME', 
                                  'PRICECLOSE', 'DIVADJPRICE', 'MARKETCAP', 'VOLUME']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {missing_columns}")
                    st.write("Available columns:", df.columns.tolist())
                    return False
                
                # Check if we have actual data in key columns
                if df['PRICINGDATE'].isnull().all():
                    st.error("❌ PRICINGDATE column is empty!")
                    return False
                    
                if df['CIQ_TICKER'].isnull().all():
                    st.error("❌ CIQ_TICKER column is empty!")  
                    return False
                
                # Set the dataframe
                _self.df = df
                
                # Prepare data with error handling
                try:
                    _self.prepare_data()
                    
                    # Final validation
                    if _self.df is None or len(_self.df) == 0:
                        st.error("❌ Data preparation resulted in empty dataset!")
                        return False
                        
                    return True
                    
                except Exception as e:
                    st.error(f"❌ Data preparation failed: {str(e)}")
                    _self.df = None
                    return False
                    
        except Exception as e:
            st.error(f"❌ File reading failed: {str(e)}")
            _self.df = None
            return False
            
        return False
    
    def prepare_data(self):
        """Clean and prepare the dataset with comprehensive error handling and duplicate removal"""
        try:
            # Convert date column with error handling
            if 'PRICINGDATE' in self.df.columns:
                try:
                    self.df['PRICINGDATE'] = pd.to_datetime(self.df['PRICINGDATE'])
                except Exception as e:
                    st.error(f"❌ Date conversion failed: {str(e)}")
                    raise ValueError("Could not convert PRICINGDATE to datetime")
            else:
                raise ValueError("PRICINGDATE column not found")
            
            # Check initial data size
            initial_rows = len(self.df)
            
            # Remove rows with missing critical data
            critical_columns = ['PRICECLOSE', 'DIVADJPRICE', 'MARKETCAP', 'VOLUME']
            self.df = self.df.dropna(subset=critical_columns)
            
            # CRITICAL: Remove duplicate ticker-date combinations early
            duplicate_mask = self.df.duplicated(subset=['CIQ_TICKER', 'PRICINGDATE'], keep='first')
            duplicates_count = duplicate_mask.sum()
            
            if duplicates_count > 0:
                st.warning(f"⚠️ Found {duplicates_count} duplicate ticker-date combinations, removing duplicates")
                self.df = self.df[~duplicate_mask].copy()
            
            remaining_rows = len(self.df)
            
            if remaining_rows == 0:
                st.error("❌ No rows remain after removing missing values and duplicates!")
                st.write(f"Started with {initial_rows} rows, all had missing critical data or were duplicates")
                raise ValueError("No valid data rows remaining")
            
            if remaining_rows < initial_rows * 0.1:  # Lost more than 90% of data
                st.warning(f"⚠️ Warning: Only {remaining_rows}/{initial_rows} rows remain after cleaning ({remaining_rows/initial_rows*100:.1f}%)")
            
            # Calculate additional metrics with error handling
            try:
                self.calculate_dividend_metrics()
            except Exception as e:
                st.error(f"❌ Dividend calculations failed: {str(e)}")
                raise ValueError("Could not calculate dividend metrics")
                
        except Exception as e:
            st.error(f"❌ Data preparation failed: {str(e)}")
            self.df = None
            raise
        
    def calculate_dividend_metrics(self):
        """Calculate dividend yields and related metrics with comprehensive duplicate handling"""
        try:
            # Sort data and ensure no duplicates
            self.df = self.df.sort_values(['CIQ_TICKER', 'PRICINGDATE']).reset_index(drop=True)
            
            # Double-check for duplicates after sorting
            duplicate_mask = self.df.duplicated(subset=['CIQ_TICKER', 'PRICINGDATE'], keep='first')
            if duplicate_mask.sum() > 0:
                st.warning(f"⚠️ Removing {duplicate_mask.sum()} remaining duplicates after sorting")
                self.df = self.df[~duplicate_mask].copy()
            
            # Check if we have sufficient data
            if len(self.df) < 60:
                st.warning("⚠️ Warning: Limited data may affect rolling calculations")
            
            # Calculate implied dividend yield
            with np.errstate(divide='ignore', invalid='ignore'):
                self.df['IMPLIED_DIVIDEND_YIELD'] = np.where(
                    self.df['PRICECLOSE'] > 0,
                    (self.df['PRICECLOSE'] - self.df['DIVADJPRICE']) / self.df['PRICECLOSE'],
                    0
                )
            
            # Replace infinite values with 0
            self.df['IMPLIED_DIVIDEND_YIELD'] = self.df['IMPLIED_DIVIDEND_YIELD'].replace([np.inf, -np.inf], 0)
            
            # Calculate annual dividend yield with enhanced error handling
            try:
                # Use transform instead of rolling to avoid index issues
                grouped = self.df.groupby('CIQ_TICKER')['IMPLIED_DIVIDEND_YIELD']
                rolling_sum = grouped.rolling(window=252, min_periods=50).sum()
                
                # Reset index to avoid duplicate issues
                self.df['ANNUAL_DIV_YIELD'] = rolling_sum.reset_index(level=0, drop=True)
                
            except Exception as e:
                st.warning(f"⚠️ Rolling dividend calculation issue: {str(e)}")
                # More robust fallback
                try:
                    # Simple multiplication fallback with groupby
                    annual_yields = []
                    for ticker in self.df['CIQ_TICKER'].unique():
                        ticker_data = self.df[self.df['CIQ_TICKER'] == ticker].copy()
                        ticker_data['ANNUAL_DIV_YIELD'] = ticker_data['IMPLIED_DIVIDEND_YIELD'] * 252
                        annual_yields.append(ticker_data)
                    
                    self.df = pd.concat(annual_yields, ignore_index=True)
                    self.df = self.df.sort_values(['CIQ_TICKER', 'PRICINGDATE']).reset_index(drop=True)
                    
                except Exception as fallback_error:
                    st.error(f"❌ Fallback dividend calculation failed: {str(fallback_error)}")
                    self.df['ANNUAL_DIV_YIELD'] = self.df['IMPLIED_DIVIDEND_YIELD'] * 252
            
            # Calculate returns and volatility with enhanced error handling
            try:
                grouped_prices = self.df.groupby('CIQ_TICKER')['DIVADJPRICE']
                self.df['PRICE_RETURN_1M'] = grouped_prices.pct_change(21)
                self.df['PRICE_RETURN_3M'] = grouped_prices.pct_change(63)
                
                # Calculate volatility using transform to avoid index issues
                grouped_returns = self.df.groupby('CIQ_TICKER')['PRICE_RETURN_1M']
                rolling_std = grouped_returns.rolling(window=63, min_periods=20).std()
                self.df['PRICE_VOLATILITY_3M'] = rolling_std.reset_index(level=0, drop=True) * np.sqrt(252)
                
            except Exception as e:
                st.warning(f"⚠️ Volatility calculation issue: {str(e)}")
                # Set fallback values
                self.df['PRICE_RETURN_1M'] = 0
                self.df['PRICE_RETURN_3M'] = 0
                self.df['PRICE_VOLATILITY_3M'] = 0.2  # 20% default volatility
            
            # Calculate trading value with enhanced error handling
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.df['TRADING_VALUE'] = self.df['VOLUME'] * self.df['PRICECLOSE'] / 1000000
                
                self.df['TRADING_VALUE'] = self.df['TRADING_VALUE'].replace([np.inf, -np.inf], 0)
                
                # Use transform to avoid index issues
                grouped_trading = self.df.groupby('CIQ_TICKER')['TRADING_VALUE']
                rolling_mean = grouped_trading.rolling(window=63, min_periods=20).mean()
                self.df['AVG_TRADING_VALUE_3M'] = rolling_mean.reset_index(level=0, drop=True)
                
            except Exception as e:
                st.warning(f"⚠️ Trading value calculation issue: {str(e)}")
                # Set fallback values
                self.df['TRADING_VALUE'] = 0
                self.df['AVG_TRADING_VALUE_3M'] = 0
            
            # Final data validation and cleanup
            calculated_columns = ['ANNUAL_DIV_YIELD', 'PRICE_VOLATILITY_3M', 'AVG_TRADING_VALUE_3M']
            for col in calculated_columns:
                if col in self.df.columns:
                    # Fill NaN values with reasonable defaults
                    if col == 'ANNUAL_DIV_YIELD':
                        self.df[col] = self.df[col].fillna(0)
                    elif col == 'PRICE_VOLATILITY_3M':
                        self.df[col] = self.df[col].fillna(0.2)  # 20% default
                    elif col == 'AVG_TRADING_VALUE_3M':
                        self.df[col] = self.df[col].fillna(0)
            
            # Final check for any remaining duplicates
            final_duplicates = self.df.duplicated(subset=['CIQ_TICKER', 'PRICINGDATE']).sum()
            if final_duplicates > 0:
                st.warning(f"⚠️ Final cleanup: removing {final_duplicates} remaining duplicates")
                self.df = self.df.drop_duplicates(subset=['CIQ_TICKER', 'PRICINGDATE'], keep='first')
                        
        except Exception as e:
            st.error(f"❌ Dividend metrics calculation failed: {str(e)}")
            st.error("This may be due to data quality issues or duplicate records")
            raise

    def show_data_summary(self):
        """Show comprehensive summary of the loaded data with safety checks"""
        
        # Safety check - ensure data exists
        if self.df is None or len(self.df) == 0:
            st.error("❌ **No data loaded**")
            st.write("Please upload a CSV file first using the sidebar.")
            return
        
        st.markdown('<div class="debug-section">', unsafe_allow_html=True)
        st.markdown("### 📋 **DATA SUMMARY & DIAGNOSTICS**")
        
        try:
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Records", f"{len(self.df):,}")
                st.metric("📅 Trading Days", f"{self.df['PRICINGDATE'].nunique():,}")
            
            with col2:
                st.metric("🏢 Unique Companies", self.df['COMPANYNAME'].nunique())
                st.metric("🎫 Unique Tickers", self.df['CIQ_TICKER'].nunique())
            
            with col3:
                date_range = (self.df['PRICINGDATE'].max() - self.df['PRICINGDATE'].min()).days
                st.metric("📆 Date Range (Days)", f"{date_range}")
                st.metric("🗓️ Data Period", f"{date_range/365.25:.1f} years")
                
            with col4:
                latest_date = self.df['PRICINGDATE'].max()
                latest_stocks = len(self.df[self.df['PRICINGDATE'] == latest_date])
                st.metric("📈 Latest Date Stocks", latest_stocks)
                
                # Check dividend-paying stocks
                dividend_paying = self.df[
                    (self.df['PRICINGDATE'] == latest_date) & 
                    (self.df['ANNUAL_DIV_YIELD'] > 0.01)
                ]
                st.metric("💰 Dividend Stocks (>1%)", len(dividend_paying))
            
            # Data scale row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # Data scale indicator (assuming data comes in millions)
                sample_mcap = self.df[self.df['PRICINGDATE'] == latest_date]['MARKETCAP'].median()
                if pd.isna(sample_mcap):
                    scale_indicator = "📊 No Market Cap Data"
                elif sample_mcap < 1000:  # Less than 1000M = 1B
                    scale_indicator = "📊 Data in Millions (M THB)"
                else:
                    scale_indicator = "📊 Large Cap in Millions (M THB)"
                st.metric("Market Cap Scale", scale_indicator)
                
        except Exception as summary_error:
            st.error(f"❌ Error generating data summary: {str(summary_error)}")
            st.write("This may indicate data quality issues.")
            return
        
        # Data quality check
        st.markdown("#### 🔍 **Data Quality Assessment**")
        
        # Get latest data for quality check
        latest_data = self.df[self.df['PRICINGDATE'] == latest_date].copy()
        
        # Check critical columns
        quality_metrics = {
            'Market Cap': latest_data['MARKETCAP'].isna().sum(),
            'Dividend Yield': latest_data['ANNUAL_DIV_YIELD'].isna().sum(),
            'Trading Volume': latest_data['AVG_TRADING_VALUE_3M'].isna().sum(),
            'Price': latest_data['PRICECLOSE'].isna().sum(),
            'Volatility': latest_data['PRICE_VOLATILITY_3M'].isna().sum()
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Missing Values Count:**")
            for metric, missing_count in quality_metrics.items():
                if missing_count == 0:
                    st.success(f"✅ {metric}: Complete data")
                elif missing_count < len(latest_data) * 0.1:
                    st.warning(f"⚠️ {metric}: {missing_count} missing ({missing_count/len(latest_data)*100:.1f}%)")
                else:
                    st.error(f"❌ {metric}: {missing_count} missing ({missing_count/len(latest_data)*100:.1f}%)")
        
        with col2:
            # Data ranges
            clean_data = latest_data.dropna(subset=['MARKETCAP', 'ANNUAL_DIV_YIELD', 'AVG_TRADING_VALUE_3M'])
            
            if len(clean_data) > 0:
                st.write("**Data Ranges (Latest Date):**")
                
                # Market cap formatting (assuming data comes in millions M THB)
                min_mcap = clean_data['MARKETCAP'].min()
                max_mcap = clean_data['MARKETCAP'].max()
                if max_mcap < 1000:  # Less than 1000M = 1B
                    st.write(f"• Market Cap: {min_mcap:.0f}M - {max_mcap:.0f}M THB")
                else:
                    st.write(f"• Market Cap: {min_mcap/1000:.2f}B - {max_mcap/1000:.1f}B THB")
                
                st.write(f"• Dividend Yield: {clean_data['ANNUAL_DIV_YIELD'].min()*100:.2f}% - {clean_data['ANNUAL_DIV_YIELD'].max()*100:.1f}%")
                
                # Smart trading volume formatting
                min_trading = clean_data['AVG_TRADING_VALUE_3M'].min()
                max_trading = clean_data['AVG_TRADING_VALUE_3M'].max()
                if max_trading < 1.0:
                    st.write(f"• Trading Volume: {min_trading*1000:.0f}K - {max_trading*1000:.0f}K THB")
                else:
                    st.write(f"• Trading Volume: {min_trading:.2f} - {max_trading:.1f}M THB")
                
                # Dividend analysis
                dividend_stocks = clean_data[clean_data['ANNUAL_DIV_YIELD'] > 0]
                st.write(f"• Dividend-paying stocks: {len(dividend_stocks)} ({len(dividend_stocks)/len(clean_data)*100:.1f}%)")
                
                if len(dividend_stocks) > 0:
                    st.write(f"• Median dividend yield: {dividend_stocks['ANNUAL_DIV_YIELD'].median()*100:.2f}%")
                else:
                    st.error("❌ **No dividend-paying stocks found!**")
            else:
                st.error("❌ **No complete data available for latest date!**")
        
        # Sample data preview
        st.markdown("#### 📊 **Sample Data Preview (Latest 10 Records)**")
        
        sample_data = self.df.tail(10)[[
            'PRICINGDATE', 'CIQ_TICKER', 'COMPANYNAME', 'PRICECLOSE', 
            'ANNUAL_DIV_YIELD', 'MARKETCAP', 'AVG_TRADING_VALUE_3M'
        ]].copy()
        
        # Format for display (assuming data comes in millions M THB)
        sample_data['DIV_YIELD_%'] = (sample_data['ANNUAL_DIV_YIELD'] * 100).round(2)
        sample_data['MARKETCAP_M_THB'] = sample_data['MARKETCAP'].round(1)  # Already in millions
        sample_data['TRADING_M_THB'] = sample_data['AVG_TRADING_VALUE_3M'].round(1)
        
        display_columns = ['PRICINGDATE', 'CIQ_TICKER', 'COMPANYNAME', 'PRICECLOSE', 'DIV_YIELD_%', 'MARKETCAP_M_THB', 'TRADING_M_THB']
        st.dataframe(sample_data[display_columns], use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def debug_screening_criteria(self, criteria):
        """Debug screening criteria to see where stocks are being filtered out"""
        
        st.markdown('<div class="debug-section">', unsafe_allow_html=True)
        st.markdown("### 🔍 **SCREENING DEBUG ANALYSIS**")
        
        # Get latest data
        latest_date = self.df['PRICINGDATE'].max()
        latest_data = self.df[self.df['PRICINGDATE'] == latest_date].copy()
        
        st.info(f"**Debug Date**: {latest_date.strftime('%Y-%m-%d')} | **Total Stocks**: {len(latest_data)}")
        
        # Quick data scale detection (assuming data comes in millions M THB)
        if len(latest_data) > 0:
            sample_mcap = latest_data['MARKETCAP'].median()
            if sample_mcap < 1000:  # Less than 1000M = 1B
                scale_note = "📊 **Note**: Market cap values detected in millions (M THB) - suggestions will use appropriate units"
            else:
                scale_note = "📊 **Note**: Market cap values detected in millions (M THB) - suggestions will use billions for large caps"
            st.write(scale_note)
        
        # Remove NaN values for filtering
        clean_data = latest_data.dropna(subset=[
            'MARKETCAP', 'ANNUAL_DIV_YIELD', 'AVG_TRADING_VALUE_3M', 
            'PRICECLOSE', 'PRICE_VOLATILITY_3M'
        ])
        
        st.write(f"**After removing missing values**: {len(clean_data)} stocks remain")
        
        if len(clean_data) == 0:
            st.error("❌ **No stocks have complete data. Your CSV file needs more complete data!**")
            st.markdown('</div>', unsafe_allow_html=True)
            return pd.DataFrame()
        
        # Apply filters step by step with visual feedback
        st.markdown("#### 🔽 **Step-by-Step Filtering Process**")
        
        current_data = clean_data.copy()
        filter_results = []
        
        # Step 1: Market Cap Filter
        step1 = current_data[current_data['MARKETCAP'] >= criteria['min_market_cap']]
        passed = len(step1)
        total = len(current_data)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Step 1: Market Cap ≥ {criteria['min_market_cap']/1000:.1f}B THB**")  # Convert from M to B
        with col2:
            if passed > 0:
                st.success(f"✅ {passed}/{total} pass")
            else:
                st.error(f"❌ {passed}/{total} pass")
        
        filter_results.append(('Market Cap', passed, total))
        
        if passed == 0:
            suggestion_value = current_data['MARKETCAP'].quantile(0.1)  # This is in millions
            
            # Format suggestion (data already in millions)
            if suggestion_value < 1000:  # Less than 1000M = 1B
                suggestion_display = f"{suggestion_value:.0f}M THB"
                suggestion_slider_value = suggestion_value / 1000  # Convert M to B for slider
            else:
                suggestion_display = f"{suggestion_value/1000:.1f}B THB" 
                suggestion_slider_value = suggestion_value / 1000  # Convert M to B for slider
            
            st.error(f"🚨 **Market Cap filter eliminates all stocks!**")
            st.write(f"💡 **Suggestion**: Lower minimum market cap to {suggestion_display}")
            st.write(f"   (Set slider to {suggestion_slider_value:.2f} in the controls)")
            st.markdown('</div>', unsafe_allow_html=True)
            return pd.DataFrame()
        
        # Step 2: Liquidity Filter
        step2 = step1[step1['AVG_TRADING_VALUE_3M'] >= criteria['min_avg_trading_value']]
        passed = len(step2)
        total = len(step1)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Step 2: Daily Trading ≥ {criteria['min_avg_trading_value']}M THB**")
        with col2:
            if passed > 0:
                st.success(f"✅ {passed}/{total} pass")
            else:
                st.error(f"❌ {passed}/{total} pass")
        
        filter_results.append(('Liquidity', passed, total))
        
        if passed == 0:
            suggestion_value = step1['AVG_TRADING_VALUE_3M'].quantile(0.1)
            
            # Smart formatting for trading volume
            if suggestion_value < 1.0:
                suggestion_display = f"{suggestion_value*1000:.0f}K THB"
            else:
                suggestion_display = f"{suggestion_value:.1f}M THB"
            
            st.error(f"🚨 **Liquidity filter eliminates all remaining stocks!**")
            st.write(f"💡 **Suggestion**: Lower minimum trading volume to {suggestion_display}")
            st.write(f"   (Set slider to {suggestion_value:.2f} in the controls)")
            st.markdown('</div>', unsafe_allow_html=True)
            return pd.DataFrame()
        
        # Step 3: Minimum Dividend Yield
        step3 = step2[step2['ANNUAL_DIV_YIELD'] >= criteria['min_dividend_yield']]
        passed = len(step3)
        total = len(step2)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Step 3: Min Dividend Yield ≥ {criteria['min_dividend_yield']*100:.1f}%**")
        with col2:
            if passed > 0:
                st.success(f"✅ {passed}/{total} pass")
            else:
                st.error(f"❌ {passed}/{total} pass")
        
        filter_results.append(('Min Dividend', passed, total))
        
        if passed == 0:
            dividend_paying = step2[step2['ANNUAL_DIV_YIELD'] > 0]
            if len(dividend_paying) > 0:
                suggestion_value = dividend_paying['ANNUAL_DIV_YIELD'].quantile(0.1) * 100
                st.error(f"🚨 **Minimum dividend yield filter eliminates all remaining stocks!**")
                st.write(f"💡 **Suggestion**: Lower minimum dividend yield to {suggestion_value:.2f}%")
                st.write(f"   (Set slider to {suggestion_value:.2f} in the controls)")
                
                # Show current dividend yield distribution
                st.write(f"📊 **Available dividend yields**: {dividend_paying['ANNUAL_DIV_YIELD'].min()*100:.2f}% - {dividend_paying['ANNUAL_DIV_YIELD'].max()*100:.2f}%")
            else:
                st.error(f"🚨 **No stocks have positive dividend yields!**")
                st.write("💡 **Check**: Your dividend calculation might not be working correctly.")
                st.write("   - Verify that DIVADJPRICE < PRICECLOSE for dividend-paying stocks")
                st.write("   - Check if you have sufficient historical data (60+ days per stock)")
            st.markdown('</div>', unsafe_allow_html=True)
            return pd.DataFrame()
        
        # Step 4: Maximum Dividend Yield
        step4 = step3[step3['ANNUAL_DIV_YIELD'] <= criteria['max_dividend_yield']]
        passed = len(step4)
        total = len(step3)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Step 4: Max Dividend Yield ≤ {criteria['max_dividend_yield']*100:.1f}%**")
        with col2:
            if passed > 0:
                st.success(f"✅ {passed}/{total} pass")
            else:
                st.error(f"❌ {passed}/{total} pass")
        
        filter_results.append(('Max Dividend', passed, total))
        
        if passed == 0:
            suggestion_value = step3['ANNUAL_DIV_YIELD'].max() * 100
            st.error(f"🚨 **Maximum dividend yield filter eliminates all remaining stocks!**")
            st.write(f"💡 **Suggestion**: Increase maximum dividend yield to {suggestion_value:.1f}%")
            st.write(f"   (Set slider to {suggestion_value:.1f} in the controls)")
            st.markdown('</div>', unsafe_allow_html=True)
            return pd.DataFrame()
        
        # Step 5: Minimum Price
        step5 = step4[step4['PRICECLOSE'] >= criteria['min_price']]
        passed = len(step5)
        total = len(step4)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Step 5: Min Price ≥ {criteria['min_price']} THB**")
        with col2:
            if passed > 0:
                st.success(f"✅ {passed}/{total} pass")
            else:
                st.error(f"❌ {passed}/{total} pass")
        
        filter_results.append(('Min Price', passed, total))
        
        # Step 6: Maximum Volatility
        final_filtered = step5[step5['PRICE_VOLATILITY_3M'] <= criteria['max_volatility']]
        passed = len(final_filtered)
        total = len(step5)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Step 6: Max Volatility ≤ {criteria['max_volatility']*100:.0f}%**")
        with col2:
            if passed > 0:
                st.success(f"✅ {passed}/{total} pass")
            else:
                st.error(f"❌ {passed}/{total} pass")
        
        filter_results.append(('Max Volatility', passed, total))
        
        if passed == 0:
            suggestion_value = step5['PRICE_VOLATILITY_3M'].quantile(0.9) * 100
            st.error(f"🚨 **Volatility filter eliminates all remaining stocks!**")
            st.write(f"💡 **Suggestion**: Increase maximum volatility to {suggestion_value:.0f}%")
            st.write(f"   (Set slider to {suggestion_value:.0f} in the controls)")
        
        # Final results
        if len(final_filtered) > 0:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success(f"🎉 **SUCCESS! {len(final_filtered)} stocks meet all screening criteria!**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show sample of qualifying stocks
            st.markdown("#### 📈 **Sample Qualifying Stocks**")
            
            sample_stocks = final_filtered.head(10)[[
                'CIQ_TICKER', 'COMPANYNAME', 'ANNUAL_DIV_YIELD', 'MARKETCAP', 'PRICECLOSE', 'AVG_TRADING_VALUE_3M'
            ]].copy()
            
            sample_stocks['Div_Yield_%'] = (sample_stocks['ANNUAL_DIV_YIELD'] * 100).round(2)
            sample_stocks['Market_Cap_M_THB'] = sample_stocks['MARKETCAP'].round(1)  # Already in millions
            sample_stocks['Trading_M_THB'] = sample_stocks['AVG_TRADING_VALUE_3M'].round(1)
            
            st.dataframe(
                sample_stocks[['CIQ_TICKER', 'COMPANYNAME', 'Div_Yield_%', 'Market_Cap_M_THB', 'PRICECLOSE', 'Trading_M_THB']],
                use_container_width=True
            )
        else:
            # Show suggested relaxed filters
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error("❌ **No stocks pass all filters!**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### 💡 **Suggested Relaxed Filters**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Try these more lenient filters:**")
                
                # Market cap suggestion (data already in millions M THB)
                mcap_suggestion = clean_data['MARKETCAP'].quantile(0.1)
                if mcap_suggestion < 1000:  # Less than 1000M = 1B
                    mcap_display = f"{mcap_suggestion:.0f}M THB (slider: {mcap_suggestion/1000:.2f})"
                else:
                    mcap_display = f"{mcap_suggestion/1000:.1f}B THB (slider: {mcap_suggestion/1000:.2f})"
                st.write(f"• **Min Market Cap**: {mcap_display}")
                
                # Smart trading volume suggestion  
                trading_suggestion = clean_data['AVG_TRADING_VALUE_3M'].quantile(0.1)
                if trading_suggestion < 1.0:
                    trading_display = f"{trading_suggestion*1000:.0f}K THB (slider: {trading_suggestion:.2f})"
                else:
                    trading_display = f"{trading_suggestion:.1f}M THB (slider: {trading_suggestion:.1f})"
                st.write(f"• **Min Trading Volume**: {trading_display}")
                
                dividend_stocks = clean_data[clean_data['ANNUAL_DIV_YIELD'] > 0]
                if len(dividend_stocks) > 0:
                    min_div_suggestion = dividend_stocks['ANNUAL_DIV_YIELD'].quantile(0.1)*100
                    max_div_suggestion = dividend_stocks['ANNUAL_DIV_YIELD'].quantile(0.95)*100
                    st.write(f"• **Min Dividend Yield**: {min_div_suggestion:.2f}% (slider: {min_div_suggestion:.2f})")
                    st.write(f"• **Max Dividend Yield**: {max_div_suggestion:.1f}% (slider: {max_div_suggestion:.1f})")
            
            with col2:
                st.write("**Data distribution insights:**")
                
                # Market cap formatting (data already in millions M THB)
                median_mcap = clean_data['MARKETCAP'].median()
                if median_mcap < 1000:  # Less than 1000M = 1B
                    median_mcap_display = f"{median_mcap:.0f}M THB"
                else:
                    median_mcap_display = f"{median_mcap/1000:.1f}B THB"
                st.write(f"• **Median Market Cap**: {median_mcap_display}")
                
                # Smart median trading formatting
                median_trading = clean_data['AVG_TRADING_VALUE_3M'].median()
                if median_trading < 1.0:
                    median_trading_display = f"{median_trading*1000:.0f}K THB"
                else:
                    median_trading_display = f"{median_trading:.1f}M THB"
                st.write(f"• **Median Trading**: {median_trading_display}")
                
                st.write(f"• **Max Volatility (90th percentile)**: {clean_data['PRICE_VOLATILITY_3M'].quantile(0.9)*100:.0f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return final_filtered
    
    def apply_user_filters(self, criteria):
        """Apply user-defined screening criteria"""
        latest_date = self.df['PRICINGDATE'].max()
        latest_data = self.df[self.df['PRICINGDATE'] == latest_date].copy()
        
        # Apply filters
        filtered = latest_data[
            (latest_data['MARKETCAP'] >= criteria['min_market_cap']) &
            (latest_data['AVG_TRADING_VALUE_3M'] >= criteria['min_avg_trading_value']) &
            (latest_data['ANNUAL_DIV_YIELD'] >= criteria['min_dividend_yield']) &
            (latest_data['ANNUAL_DIV_YIELD'] <= criteria['max_dividend_yield']) &
            (latest_data['PRICECLOSE'] >= criteria['min_price']) &
            (latest_data['PRICE_VOLATILITY_3M'] <= criteria['max_volatility'])
        ].dropna()
        
        return filtered
    
    def calculate_returns_matrix(self, tickers, start_date, end_date, min_periods=50):
        """Calculate historical returns matrix for portfolio optimization with enhanced validation and duplicate handling"""
        returns_data = []
        processed_tickers = []
        
        # Remove duplicates while preserving order
        unique_tickers = list(dict.fromkeys(tickers))
        
        for ticker in unique_tickers:
            try:
                ticker_data = self.df[
                    (self.df['CIQ_TICKER'] == ticker) & 
                    (self.df['PRICINGDATE'] >= start_date) & 
                    (self.df['PRICINGDATE'] <= end_date)
                ].sort_values('PRICINGDATE').copy()
                
                if len(ticker_data) >= min_periods:
                    # Remove any duplicates for this ticker (safety check)
                    ticker_data = ticker_data.drop_duplicates(subset=['PRICINGDATE'], keep='first')
                    
                    # Calculate returns if not already available or if they're all NaN
                    if 'PRICE_RETURN_1M' not in ticker_data.columns or ticker_data['PRICE_RETURN_1M'].isna().all():
                        ticker_data['PRICE_RETURN_1M'] = ticker_data['DIVADJPRICE'].pct_change(21)
                    
                    # Set index safely with duplicate handling
                    try:
                        ticker_returns = ticker_data.set_index('PRICINGDATE')['PRICE_RETURN_1M'].dropna()
                    except ValueError as e:
                        if "cannot reindex" in str(e).lower():
                            # Handle duplicate dates by aggregating them (take mean)
                            st.warning(f"⚠️ {ticker}: Found duplicate dates, aggregating by mean")
                            ticker_data_grouped = ticker_data.groupby('PRICINGDATE')['PRICE_RETURN_1M'].mean()
                            ticker_returns = ticker_data_grouped.dropna()
                        else:
                            st.warning(f"⚠️ Skipping {ticker}: Index error - {str(e)}")
                            continue
                    
                    # Check for valid returns (not all zeros or constant)
                    if len(ticker_returns) >= min_periods and ticker_returns.std() > 1e-6:
                        returns_data.append(ticker_returns.rename(ticker))
                        processed_tickers.append(ticker)
                    else:
                        st.warning(f"⚠️ Skipping {ticker}: insufficient return variation (std: {ticker_returns.std():.6f})")
                        
            except Exception as ticker_error:
                st.warning(f"⚠️ Error processing {ticker}: {str(ticker_error)}")
                continue
        
        if len(returns_data) == 0:
            st.error("❌ No valid return series found")
            return pd.DataFrame()
        
        if len(returns_data) < 3:
            st.warning(f"⚠️ Only {len(returns_data)} valid stocks for optimization (minimum 3 required)")
            
        # Combine returns and handle alignment with duplicate handling
        try:
            returns_matrix = pd.concat(returns_data, axis=1, sort=True).dropna()
        except Exception as concat_error:
            st.error(f"❌ Error combining return series: {str(concat_error)}")
            return pd.DataFrame()
        
        # Final validation
        if returns_matrix.empty:
            st.error("❌ Returns matrix is empty after alignment")
            return pd.DataFrame()
            
        if len(returns_matrix) < min_periods:
            st.warning(f"⚠️ Only {len(returns_matrix)} common dates found (minimum {min_periods} recommended)")
            
        # Check for perfect correlations that would cause singular covariance matrix
        if len(returns_matrix.columns) > 1:
            try:
                correlation_matrix = returns_matrix.corr()
                high_corr_pairs = []
                
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.98:  # Very high correlation
                            stock1, stock2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                            high_corr_pairs.append((stock1, stock2, corr_val))
                
                if high_corr_pairs:
                    st.warning(f"⚠️ Found {len(high_corr_pairs)} highly correlated pairs that may cause optimization issues")
                    for stock1, stock2, corr in high_corr_pairs[:3]:  # Show first 3
                        st.write(f"   • {stock1} vs {stock2}: {corr:.3f}")
            except Exception as corr_error:
                st.warning(f"⚠️ Could not calculate correlations: {str(corr_error)}")
        
        st.info(f"✅ Successfully processed {len(processed_tickers)} stocks with {len(returns_matrix)} common dates")
        
        return returns_matrix
    
    def create_covariance_matrix(self, returns_matrix, method='sample', regularization=True):
        """Create covariance matrix with optional regularization and different estimation methods"""
        
        if returns_matrix.empty:
            return np.array([])
            
        if method == 'sample':
            cov_matrix = returns_matrix.cov().values * 252  # Annualized
        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage estimator
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                cov_matrix = lw.fit(returns_matrix.fillna(0)).covariance_ * 252
            except ImportError:
                st.warning("⚠️ sklearn not available, using sample covariance")
                cov_matrix = returns_matrix.cov().values * 252
        else:
            cov_matrix = returns_matrix.cov().values * 252
            
        # Check if matrix is positive definite
        try:
            np.linalg.cholesky(cov_matrix)
            is_positive_definite = True
        except np.linalg.LinAlgError:
            is_positive_definite = False
            
        # Apply regularization if needed
        if not is_positive_definite and regularization:
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            min_eigenval = eigenvals.min()
            
            if min_eigenval <= 0:
                # Add regularization to make positive definite
                regularization_factor = abs(min_eigenval) + 1e-4
                cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * regularization_factor
                st.info(f"ℹ️ Applied regularization factor: {regularization_factor:.6f}")
                
        return cov_matrix
    
    def implement_investment_strategy(self, filtered_data, strategy_name, strategy_params):
        """Implement various investment strategies beyond mean-variance optimization"""
        
        if len(filtered_data) == 0:
            return pd.DataFrame()
            
        portfolio_data = filtered_data.copy()
        
        if strategy_name == 'equal_weight':
            # Equal Weight Strategy
            n_stocks = min(strategy_params.get('max_holdings', 25), len(portfolio_data))
            selected_stocks = portfolio_data.nlargest(n_stocks, 'ANNUAL_DIV_YIELD')
            selected_stocks['PORTFOLIO_WEIGHT'] = 1.0 / len(selected_stocks)
            
        elif strategy_name == 'market_cap_weight':
            # Market Cap Weighted Strategy
            n_stocks = min(strategy_params.get('max_holdings', 25), len(portfolio_data))
            selected_stocks = portfolio_data.nlargest(n_stocks, 'MARKETCAP')
            total_mcap = selected_stocks['MARKETCAP'].sum()
            selected_stocks['PORTFOLIO_WEIGHT'] = selected_stocks['MARKETCAP'] / total_mcap
            
        elif strategy_name == 'dividend_weight':
            # Dividend Yield Weighted Strategy
            n_stocks = min(strategy_params.get('max_holdings', 25), len(portfolio_data))
            # Select top dividend payers
            dividend_stocks = portfolio_data[portfolio_data['ANNUAL_DIV_YIELD'] > 0]
            selected_stocks = dividend_stocks.nlargest(n_stocks, 'ANNUAL_DIV_YIELD')
            total_div_yield = selected_stocks['ANNUAL_DIV_YIELD'].sum()
            selected_stocks['PORTFOLIO_WEIGHT'] = selected_stocks['ANNUAL_DIV_YIELD'] / total_div_yield
            
        elif strategy_name == 'low_volatility':
            # Low Volatility Strategy
            n_stocks = min(strategy_params.get('max_holdings', 25), len(portfolio_data))
            # Select stocks with lowest volatility among dividend payers
            low_vol_stocks = portfolio_data[
                (portfolio_data['ANNUAL_DIV_YIELD'] > 0.01) &
                (portfolio_data['PRICE_VOLATILITY_3M'] > 0)
            ].nsmallest(n_stocks, 'PRICE_VOLATILITY_3M')
            
            # Weight inversely by volatility
            if len(low_vol_stocks) > 0:
                inv_volatility = 1 / low_vol_stocks['PRICE_VOLATILITY_3M']
                selected_stocks = low_vol_stocks.copy()
                selected_stocks['PORTFOLIO_WEIGHT'] = inv_volatility / inv_volatility.sum()
            else:
                selected_stocks = portfolio_data.head(n_stocks).copy()
                selected_stocks['PORTFOLIO_WEIGHT'] = 1.0 / len(selected_stocks)
                
        elif strategy_name == 'quality_factor':
            # Quality Factor Strategy (based on market cap and dividend consistency)
            n_stocks = min(strategy_params.get('max_holdings', 25), len(portfolio_data))
            
            # Create quality score
            portfolio_data['QUALITY_SCORE'] = (
                0.4 * portfolio_data['MARKETCAP'].rank(pct=True) +
                0.3 * portfolio_data['ANNUAL_DIV_YIELD'].rank(pct=True) +
                0.3 * (1 - portfolio_data['PRICE_VOLATILITY_3M'].rank(pct=True))
            )
            
            selected_stocks = portfolio_data.nlargest(n_stocks, 'QUALITY_SCORE')
            # Weight by quality score
            total_quality = selected_stocks['QUALITY_SCORE'].sum()
            selected_stocks['PORTFOLIO_WEIGHT'] = selected_stocks['QUALITY_SCORE'] / total_quality
            
        elif strategy_name == 'momentum':
            # Momentum Strategy
            n_stocks = min(strategy_params.get('max_holdings', 25), len(portfolio_data))
            
            # Create momentum score (3M returns + dividend yield)
            portfolio_data['MOMENTUM_SCORE'] = (
                0.7 * portfolio_data['PRICE_RETURN_3M'].fillna(0) +
                0.3 * portfolio_data['ANNUAL_DIV_YIELD']
            )
            
            selected_stocks = portfolio_data.nlargest(n_stocks, 'MOMENTUM_SCORE')
            # Equal weight for momentum
            selected_stocks['PORTFOLIO_WEIGHT'] = 1.0 / len(selected_stocks)
            
        elif strategy_name == 'risk_parity':
            # Risk Parity Strategy (simplified version)
            n_stocks = min(strategy_params.get('max_holdings', 25), len(portfolio_data))
            
            # Select stocks and weight inversely by volatility
            top_dividend_stocks = portfolio_data[
                portfolio_data['ANNUAL_DIV_YIELD'] > 0.01
            ].nlargest(n_stocks, 'ANNUAL_DIV_YIELD')
            
            if len(top_dividend_stocks) > 0 and not top_dividend_stocks['PRICE_VOLATILITY_3M'].isna().all():
                # Replace zero volatilities with median to avoid division by zero
                volatilities = top_dividend_stocks['PRICE_VOLATILITY_3M'].fillna(
                    top_dividend_stocks['PRICE_VOLATILITY_3M'].median()
                )
                volatilities = volatilities.replace(0, volatilities[volatilities > 0].median())
                
                inv_vol_weights = 1 / volatilities
                selected_stocks = top_dividend_stocks.copy()
                selected_stocks['PORTFOLIO_WEIGHT'] = inv_vol_weights / inv_vol_weights.sum()
            else:
                selected_stocks = portfolio_data.head(n_stocks).copy()
                selected_stocks['PORTFOLIO_WEIGHT'] = 1.0 / len(selected_stocks)
                
        else:
            # Default to equal weight
            n_stocks = min(strategy_params.get('max_holdings', 25), len(portfolio_data))
            selected_stocks = portfolio_data.nlargest(n_stocks, 'ANNUAL_DIV_YIELD')
            selected_stocks['PORTFOLIO_WEIGHT'] = 1.0 / len(selected_stocks)
        
        # Final validation and cleanup
        if len(selected_stocks) == 0:
            st.error("❌ No stocks selected by strategy")
            return pd.DataFrame()
            
        # Remove any duplicate tickers (keep first occurrence)
        selected_stocks = selected_stocks.drop_duplicates(subset=['CIQ_TICKER'], keep='first')
        
        # Ensure weights sum to 1
        total_weight = selected_stocks['PORTFOLIO_WEIGHT'].sum()
        if total_weight > 0:
            selected_stocks['PORTFOLIO_WEIGHT'] = selected_stocks['PORTFOLIO_WEIGHT'] / total_weight
        else:
            selected_stocks['PORTFOLIO_WEIGHT'] = 1.0 / len(selected_stocks)
            
        return selected_stocks.sort_values('PORTFOLIO_WEIGHT', ascending=False)
    
    def optimize_portfolio(self, expected_returns, cov_matrix, target_return=None, target_risk=None, 
                          risk_free_rate=0.02/252, optimization_method='max_sharpe'):
        """Enhanced portfolio optimization with better error handling"""
        try:
            n_assets = len(expected_returns)
            
            # Validation checks
            if n_assets == 0:
                return None, 0, 0, 0, False
            
            if n_assets != len(cov_matrix):
                st.warning("⚠️ Dimension mismatch between returns and covariance matrix")
                return None, 0, 0, 0, False
                
            if n_assets < 3:
                st.warning("⚠️ Too few assets for meaningful optimization")
                return None, 0, 0, 0, False
            
            # Check for invalid values
            if np.any(np.isnan(expected_returns)) or np.any(np.isnan(cov_matrix)):
                st.warning("⚠️ NaN values detected in optimization inputs")
                return None, 0, 0, 0, False
            
            # Enhanced covariance matrix validation
            try:
                eigenvals = np.linalg.eigvals(cov_matrix)
                min_eigenval = eigenvals.min()
                
                if min_eigenval <= 1e-8:
                    st.info(f"ℹ️ Covariance matrix has small eigenvalue: {min_eigenval:.2e}, applying regularization")
                    # More sophisticated regularization
                    regularization_strength = max(1e-6, abs(min_eigenval) * 1.1)
                    cov_matrix = cov_matrix + np.eye(n_assets) * regularization_strength
                    
                # Test for positive definiteness
                np.linalg.cholesky(cov_matrix)
                
            except np.linalg.LinAlgError:
                st.warning("⚠️ Covariance matrix issues detected, applying advanced regularization")
                # Eigenvalue regularization
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                eigenvals = np.maximum(eigenvals, 1e-6)  # Floor eigenvalues
                cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Define objective functions
            def neg_sharpe_ratio(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_std < 1e-10:
                    return -1e-10
                return -(portfolio_return - risk_free_rate) / portfolio_std
            
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            def neg_portfolio_return(weights):
                return -np.sum(weights * expected_returns)
            
            # Select objective function
            objective_functions = {
                'max_sharpe': neg_sharpe_ratio,
                'min_risk': portfolio_volatility,
                'target_return': portfolio_volatility,
                'target_risk': neg_portfolio_return
            }
            
            objective = objective_functions.get(optimization_method, neg_sharpe_ratio)
            
            # Set up constraints
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
            
            if optimization_method == 'target_return' and target_return is not None:
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: np.sum(x * expected_returns) - target_return
                })
            
            if optimization_method == 'target_risk' and target_risk is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))) - target_risk
                })
            
            # Set bounds (minimum 0.5%, maximum 15%)
            bounds = tuple((0.001, 0.80) for _ in range(n_assets))
            
            # Multiple optimization attempts with different starting points
            best_result = None
            best_objective = np.inf
            
            starting_points = [
                np.ones(n_assets) / n_assets,  # Equal weights
                np.random.dirichlet(np.ones(n_assets)),  # Random Dirichlet
                expected_returns / expected_returns.sum()  # Expected return weighted
            ]
            
            for i, x0 in enumerate(starting_points):
                # Ensure starting point satisfies bounds
                x0 = np.clip(x0, 0.005, 0.15)
                x0 = x0 / np.sum(x0)  # Normalize
                
                try:
                    result = minimize(
                        objective, 
                        x0, 
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000}
                    )
                    
                    if result.success and result.fun < best_objective:
                        best_result = result
                        best_objective = result.fun
                        
                except Exception as opt_error:
                    if i == 0:  # Only warn on first attempt
                        st.warning(f"⚠️ Optimization attempt {i + 1} failed: {str(opt_error)}")
                    continue
            
            if best_result is not None and best_result.success:
                optimal_weights = best_result.x
                
                # Validate and clean weights
                optimal_weights = np.clip(optimal_weights, 0, 1)
                weight_sum = np.sum(optimal_weights)
                
                if weight_sum > 0:
                    optimal_weights = optimal_weights / weight_sum
                else:
                    st.warning("⚠️ Invalid weight sum, using equal weights")
                    optimal_weights = np.ones(n_assets) / n_assets
                
                # Calculate portfolio metrics
                optimal_return = np.sum(optimal_weights * expected_returns)
                optimal_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                
                if optimal_std > 0:
                    optimal_sharpe = (optimal_return - risk_free_rate) / optimal_std
                else:
                    optimal_sharpe = 0
                
                return optimal_weights, optimal_sharpe, optimal_return, optimal_std, True
                
            else:
                # Fallback to equal weights
                st.warning("⚠️ Optimization failed, using equal weights fallback")
                equal_weights = np.ones(n_assets) / n_assets
                equal_return = np.sum(equal_weights * expected_returns)
                equal_std = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
                equal_sharpe = (equal_return - risk_free_rate) / equal_std if equal_std > 0 else 0
                
                return equal_weights, equal_sharpe, equal_return, equal_std, True
                
        except Exception as e:
            st.error(f"❌ Portfolio optimization failed: {str(e)}")
            return None, 0, 0, 0, False
    
    def calculate_efficient_frontier(self, expected_returns, cov_matrix, num_points=50):
        """Calculate the efficient frontier with enhanced error handling"""
        try:
            min_ret = expected_returns.min()
            max_ret = expected_returns.max()
            
            # Check if we have valid return range
            if np.isnan(min_ret) or np.isnan(max_ret) or min_ret >= max_ret:
                st.warning("⚠️ Invalid return range for efficient frontier calculation")
                return np.array([]), np.array([]), []
            
            target_returns = np.linspace(min_ret, max_ret, num_points)
            
            frontier_volatility = []
            frontier_returns = []
            frontier_weights = []
            
            successful_calculations = 0
            
            for target_ret in target_returns:
                try:
                    weights, _, ret, vol, success = self.optimize_portfolio(
                        expected_returns, cov_matrix, 
                        target_return=target_ret, 
                        optimization_method='target_return'
                    )
                    if success and not np.isnan(ret) and not np.isnan(vol) and vol > 0:
                        frontier_returns.append(ret)
                        frontier_volatility.append(vol)
                        frontier_weights.append(weights)
                        successful_calculations += 1
                except Exception:
                    # Skip failed optimization points
                    continue
            
            if successful_calculations == 0:
                st.warning("⚠️ No valid efficient frontier points could be calculated")
                return np.array([]), np.array([]), []
            
            if successful_calculations < num_points * 0.1:  # Less than 10% success rate
                st.warning(f"⚠️ Only {successful_calculations}/{num_points} frontier points calculated successfully")
            
            return np.array(frontier_returns), np.array(frontier_volatility), frontier_weights
            
        except Exception as e:
            st.error(f"❌ Efficient frontier calculation failed: {str(e)}")
            return np.array([]), np.array([]), []
    
    def create_portfolio_with_user_params(self, filtered_data, optimization_params):
        """Enhanced portfolio creation with strategy selection and deduplication"""
        if len(filtered_data) == 0:
            return pd.DataFrame()
        
        strategy_method = optimization_params.get('strategy', 'modern_portfolio_theory')
        
        # Remove any duplicate tickers early
        filtered_data = filtered_data.drop_duplicates(subset=['CIQ_TICKER'], keep='first')
        
        if strategy_method != 'modern_portfolio_theory':
            # Use alternative investment strategies
            return self.implement_investment_strategy(
                filtered_data, 
                strategy_method, 
                optimization_params
            )
        
        # Modern Portfolio Theory approach (enhanced)
        max_holdings = min(optimization_params['max_holdings'], len(filtered_data))
        
        # Pre-select candidates with enhanced scoring
        portfolio_data = filtered_data.copy()
        
        # More sophisticated candidate selection
        portfolio_data['DIV_YIELD_RANK'] = portfolio_data['ANNUAL_DIV_YIELD'].rank(ascending=False, pct=True)
        portfolio_data['MARKET_CAP_RANK'] = portfolio_data['MARKETCAP'].rank(ascending=False, pct=True)
        portfolio_data['LIQUIDITY_RANK'] = portfolio_data['AVG_TRADING_VALUE_3M'].rank(ascending=False, pct=True)
        portfolio_data['LOW_VOL_RANK'] = (1 - portfolio_data['PRICE_VOLATILITY_3M'].rank(ascending=True, pct=True))
        
        # Comprehensive quality score
        portfolio_data['QUALITY_SCORE'] = (
            optimization_params.get('dividend_weight', 0.4) * portfolio_data['DIV_YIELD_RANK'] + 
            optimization_params.get('size_weight', 0.3) * portfolio_data['MARKET_CAP_RANK'] +
            0.2 * portfolio_data['LIQUIDITY_RANK'] +
            0.1 * portfolio_data['LOW_VOL_RANK']
        )
        
        candidates = portfolio_data.nlargest(max_holdings, 'QUALITY_SCORE').copy()
        
        # Remove duplicates again (safety check)
        candidates = candidates.drop_duplicates(subset=['CIQ_TICKER'], keep='first')
        candidate_tickers = candidates['CIQ_TICKER'].tolist()
        
        # Calculate returns matrix with enhanced validation
        latest_date = filtered_data['PRICINGDATE'].iloc[0]
        lookback_days = optimization_params.get('lookback_days', 252)
        optimization_start = latest_date - timedelta(days=lookback_days + 30)
        
        returns_matrix = self.calculate_returns_matrix(
            candidate_tickers, optimization_start, latest_date
        )
        
        if returns_matrix.empty or len(returns_matrix.columns) < 3:
            st.warning("Insufficient data for optimization, using dividend-weighted strategy")
            # Fallback to dividend weighting
            total_div = candidates['ANNUAL_DIV_YIELD'].sum()
            if total_div > 0:
                candidates['PORTFOLIO_WEIGHT'] = candidates['ANNUAL_DIV_YIELD'] / total_div
            else:
                candidates['PORTFOLIO_WEIGHT'] = 1.0 / len(candidates)
            return candidates
        
        # Calculate expected returns with multiple methods
        historical_returns = returns_matrix.mean() * 252
        
        # Match candidates to actual returns data - SAFE INDEXING
        valid_candidates = candidates[candidates['CIQ_TICKER'].isin(returns_matrix.columns)].copy()
        
        # Safer dividend yields mapping - avoid set_index issues
        expected_returns = pd.Series(index=returns_matrix.columns, dtype=float)
        
        for ticker in returns_matrix.columns:
            hist_ret = historical_returns.get(ticker, 0)
            
            # Safer dividend yield lookup
            ticker_dividend_data = valid_candidates[valid_candidates['CIQ_TICKER'] == ticker]
            if len(ticker_dividend_data) > 0:
                div_yield = ticker_dividend_data['ANNUAL_DIV_YIELD'].iloc[0]  # Take first if multiple
            else:
                div_yield = 0
                
            expected_returns[ticker] = (
                optimization_params.get('return_weight', 0.4) * hist_ret + 
                (1 - optimization_params.get('return_weight', 0.4)) * div_yield
            )
        
        # Create enhanced covariance matrix
        cov_matrix = self.create_covariance_matrix(
            returns_matrix, 
            method=optimization_params.get('cov_method', 'sample'),
            regularization=True
        )
        
        # Perform optimization
        weights, sharpe, ret, vol, success = self.optimize_portfolio(
            expected_returns.values, 
            cov_matrix,
            target_return=optimization_params.get('target_return'),
            target_risk=optimization_params.get('target_risk'),
            optimization_method=optimization_params.get('method', 'max_sharpe')
        )
        
        if success and weights is not None:
            # Map weights back to candidates (ensuring no duplicates)
            weight_dict = dict(zip(returns_matrix.columns, weights))
            valid_candidates['PORTFOLIO_WEIGHT'] = valid_candidates['CIQ_TICKER'].map(weight_dict)
            valid_candidates['PORTFOLIO_WEIGHT'] = valid_candidates['PORTFOLIO_WEIGHT'].fillna(0)
            
            # Filter out zero weights and ensure no duplicates
            portfolio = valid_candidates[valid_candidates['PORTFOLIO_WEIGHT'] > 1e-4].copy()
            portfolio = portfolio.drop_duplicates(subset=['CIQ_TICKER'], keep='first')
            
            # Final weight normalization
            total_weight = portfolio['PORTFOLIO_WEIGHT'].sum()
            if total_weight > 0:
                portfolio['PORTFOLIO_WEIGHT'] = portfolio['PORTFOLIO_WEIGHT'] / total_weight
                
        else:
            # Enhanced fallback strategy
            st.warning("Optimization failed, using enhanced dividend-weighted fallback")
            valid_candidates = candidates[candidates['CIQ_TICKER'].isin(returns_matrix.columns)].copy()
            
            # Weight by dividend yield with quality adjustment
            quality_div_score = valid_candidates['ANNUAL_DIV_YIELD'] * valid_candidates['QUALITY_SCORE']
            total_score = quality_div_score.sum()
            
            if total_score > 0:
                valid_candidates['PORTFOLIO_WEIGHT'] = quality_div_score / total_score
            else:
                valid_candidates['PORTFOLIO_WEIGHT'] = 1.0 / len(valid_candidates)
                
            portfolio = valid_candidates
        
        return portfolio.sort_values('PORTFOLIO_WEIGHT', ascending=False)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🇹🇭 Thai SET High Dividend Strategy</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Interactive Portfolio Optimization & Analysis with Enhanced Debugging</h3>', unsafe_allow_html=True)
    
    # Initialize the app
    if 'strategy_app' not in st.session_state:
        st.session_state.strategy_app = ThaiDividendStrategyApp()
    
    app = st.session_state.strategy_app
    
    # Sidebar for controls
    st.sidebar.markdown("## 📊 Control Panel")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Thai Stock Data (CSV)", 
        type=['csv'],
        help="Upload your Thai SET stock data file. For files >200MB, see instructions below."
    )
    
    # Large file upload instructions
    with st.sidebar.expander("📁 Large File Upload (>600MB)"):
        st.markdown("""
        **For files larger than 600MB:**
        
        1. **Run with increased limit:**
        ```bash
        streamlit run thai_dividend_app.py --server.maxUploadSize=1000
        ```
        
        2. **Or create `.streamlit/config.toml`:**
        ```toml
        [server]
        maxUploadSize = 1000
        ```
        
        3. **Alternative: Split your data** into smaller monthly/yearly files
        
        **Current upload limit:** ~200MB (default)
        """)
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            if file_size > 200:
                st.warning(f"⚠️ Large file detected: {file_size:.1f}MB")
                st.write("Consider the methods above for better performance")
            else:
                st.success(f"✅ File size OK: {file_size:.1f}MB")
    
    if uploaded_file is not None:
        # Load data
        try:
            # Show progress for large files
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            if file_size_mb > 50:  # Show progress for files > 50MB
                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                
                status_text.text("📁 Reading CSV file...")
                progress_bar.progress(20)
            
            # Attempt to load data
            load_success = app.load_data(uploaded_file)
            
            if file_size_mb > 50:
                status_text.text("⚙️ Processing data...")
                progress_bar.progress(60)
            
            # Check if loading was actually successful AND data exists
            if load_success and app.df is not None and len(app.df) > 0:
                if file_size_mb > 50:
                    status_text.text("✅ Data ready!")
                    progress_bar.progress(100)
                    # Clear progress bar after 1 second
                    import time
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                
                st.sidebar.success("✅ Data loaded successfully!")
                
                # Display basic info - ONLY when app.df is confirmed not None
                st.sidebar.markdown("### Dataset Info")
                
                # Safe access to dataframe with additional None checks
                try:
                    if hasattr(app, 'df') and app.df is not None and not app.df.empty:
                        date_min = app.df['PRICINGDATE'].min()
                        date_max = app.df['PRICINGDATE'].max()
                        st.sidebar.write(f"📅 Date Range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
                        st.sidebar.write(f"🏢 Companies: {app.df['COMPANYNAME'].nunique()}")
                        st.sidebar.write(f"📈 Total Records: {len(app.df):,}")
                        
                        # File and memory info
                        st.sidebar.write(f"📁 File Size: {file_size_mb:.1f} MB")
                        
                        # Memory usage estimate
                        memory_usage = app.df.memory_usage(deep=True).sum() / (1024 * 1024)
                        st.sidebar.write(f"🧠 Memory Usage: ~{memory_usage:.1f} MB")
                        
                        if memory_usage > 500:  # > 500MB in memory
                            st.sidebar.warning("⚠️ High memory usage - app may be slow")
                    else:
                        st.sidebar.error("❌ Data is empty or invalid")
                        st.error("**Data Validation Failed**")
                        st.write("The uploaded file was processed but resulted in empty data.")
                        return
                        
                except Exception as sidebar_error:
                    st.sidebar.error(f"❌ Error displaying data info: {str(sidebar_error)}")
                    st.error("**Data Display Error**")
                    st.write(f"Data loaded but display failed: {str(sidebar_error)}")
                    return
                
                # Main tabs - only show if data is properly loaded
                if hasattr(app, 'df') and app.df is not None and len(app.df) > 0:
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "🎯 Portfolio Builder", 
                        "📈 Efficient Frontier", 
                        "📊 Performance Analysis",
                        "🔍 Stock Screener",
                        "📋 Strategy Report"
                    ])
                    
                    with tab1:
                        st.markdown('<div class="sub-header">🎯 Interactive Portfolio Optimization</div>', unsafe_allow_html=True)
                        
                        # Show data summary first
                        app.show_data_summary()
                        
                        st.markdown("---")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("### 🔧 Optimization Parameters")
                            
                            # Screening criteria
                            st.markdown("#### Stock Screening")
                            min_market_cap = st.slider(
                                "Minimum Market Cap (Billion THB)", 
                                0.01, 50.0, 1.0, 0.01,  # Lowered default from 5.0 to 1.0
                                help="Filter stocks by minimum market capitalization"
                            )
                            
                            min_div_yield = st.slider(
                                "Minimum Dividend Yield (%)", 
                                0.0, 10.0, 1.0, 0.1,  # Lowered default from 2.0 to 1.0
                                help="Minimum required dividend yield"
                            ) / 100
                            
                            max_div_yield = st.slider(
                                "Maximum Dividend Yield (%)", 
                                5.0, 25.0, 20.0, 0.5,  # Increased default from 15.0 to 20.0
                                help="Maximum dividend yield (avoid dividend traps)"
                            ) / 100
                            
                            min_liquidity = st.slider(
                                "Minimum Daily Trading (Million THB)", 
                                0.1, 100.0, 1.0, 0.1,
                                help="Minimum average daily trading volume"
                            )
                            
                            max_volatility = st.slider(
                                "Maximum Volatility (%)", 
                                20.0, 200.0, 100.0, 5.0,
                                help="Maximum annual volatility"
                            ) / 100
                            
                            # Portfolio construction
                            st.markdown("#### Portfolio Construction")
                            max_holdings = st.slider("Maximum Holdings", 10, 50, 25, 1)
                            
                            # Strategy Selection
                            investment_strategy = st.selectbox(
                                "Investment Strategy",
                                [
                                    'modern_portfolio_theory',
                                    'equal_weight', 
                                    'market_cap_weight',
                                    'dividend_weight',
                                    'low_volatility',
                                    'quality_factor',
                                    'momentum',
                                    'risk_parity'
                                ],
                                format_func=lambda x: {
                                    'modern_portfolio_theory': '🧮 Modern Portfolio Theory (MPT)',
                                    'equal_weight': '⚖️ Equal Weight',
                                    'market_cap_weight': '🏢 Market Cap Weighted',
                                    'dividend_weight': '💰 Dividend Weighted',
                                    'low_volatility': '🛡️ Low Volatility',
                                    'quality_factor': '⭐ Quality Factor',
                                    'momentum': '🚀 Momentum',
                                    'risk_parity': '📊 Risk Parity'
                                }[x],
                                help="Choose your investment strategy approach"
                            )
                            
                            # Show strategy description
                            strategy_descriptions = {
                                'modern_portfolio_theory': 'Uses mean-variance optimization to find optimal risk-return trade-offs',
                                'equal_weight': 'Assigns equal weight to all selected stocks',
                                'market_cap_weight': 'Weights stocks by market capitalization',
                                'dividend_weight': 'Weights stocks by dividend yield',
                                'low_volatility': 'Focuses on stocks with lowest volatility',
                                'quality_factor': 'Weights by quality metrics (size, dividends, stability)',
                                'momentum': 'Selects stocks with strong recent performance',
                                'risk_parity': 'Equal risk contribution from each stock'
                            }
                            
                            st.info(f"📖 **{strategy_descriptions[investment_strategy]}**")
                            
                            # MPT-specific options
                            if investment_strategy == 'modern_portfolio_theory':
                                optimization_method = st.selectbox(
                                    "MPT Optimization Method",
                                    ['max_sharpe', 'min_risk', 'target_return', 'target_risk'],
                                    format_func=lambda x: {
                                        'max_sharpe': '📈 Maximum Sharpe Ratio',
                                        'min_risk': '🛡️ Minimum Risk',
                                        'target_return': '🎯 Target Return',
                                        'target_risk': '⚖️ Target Risk'
                                    }[x]
                                )
                                
                                # Covariance estimation method
                                cov_method = st.selectbox(
                                    "Covariance Estimation",
                                    ['sample', 'shrinkage'],
                                    format_func=lambda x: {
                                        'sample': '📊 Sample Covariance',
                                        'shrinkage': '🎯 Shrinkage (Ledoit-Wolf)'
                                    }[x],
                                    help="Shrinkage can help with estimation errors"
                                )
                            else:
                                optimization_method = 'max_sharpe'  # Default for other strategies
                                cov_method = 'sample'
                            
                            # Additional parameters based on method
                            target_return = None
                            target_risk = None
                            
                            if optimization_method == 'target_return':
                                target_return = st.slider(
                                    "Target Annual Return (%)", 
                                    0.0, 30.0, 8.0, 0.5
                                ) / 100
                            
                            if optimization_method == 'target_risk':
                                target_risk = st.slider(
                                    "Target Annual Risk (%)", 
                                    5.0, 50.0, 20.0, 1.0
                                ) / 100
                            
                            # Advanced Parameters
                            with st.expander("🔧 Advanced Parameters"):
                                dividend_weight = st.slider(
                                    "Dividend vs Size Weight", 
                                    0.0, 1.0, 0.4, 0.1,
                                    help="Weight for dividend yield vs market cap in candidate selection"
                                )
                                
                                return_weight = st.slider(
                                    "Historical vs Dividend Return Weight", 
                                    0.0, 1.0, 0.4, 0.1,
                                    help="Weight for historical returns vs dividend yield in MPT"
                                ) if investment_strategy == 'modern_portfolio_theory' else 0.4
                                
                                size_weight = st.slider(
                                    "Size Factor Weight",
                                    0.0, 1.0, 0.3, 0.1,
                                    help="Weight for market cap in quality scoring"
                                )
                                
                                lookback_days = st.slider(
                                    "Lookback Period (Days)", 
                                    120, 500, 252, 10,
                                    help="Historical data period for calculations"
                                )
                            
                            # Build portfolio button
                            if st.button("🚀 Build Portfolio", type="primary"):
                                with st.spinner("Building optimized portfolio..."):
                                    # Apply filters
                                    criteria = {
                                        'min_market_cap': min_market_cap * 1e3,
                                        'min_avg_trading_value': min_liquidity,
                                        'min_dividend_yield': min_div_yield,
                                        'max_dividend_yield': max_div_yield,
                                        'min_price': 1.0,
                                        'max_volatility': max_volatility
                                    }
                                    
                                    filtered_data = app.apply_user_filters(criteria)
                                    
                                    if len(filtered_data) > 0:
                                        # Optimization parameters
                                        # opt_params = {
                                        #     'max_holdings': max_holdings,
                                        #     'method': optimization_method,
                                        #     'target_return': target_return,
                                        #     'target_risk': target_risk,
                                        #     'dividend_weight': dividend_weight,
                                        #     'return_weight': return_weight,
                                        #     'lookback_days': lookback_days
                                        # }

                                        # Should be:
                                        opt_params = {
                                            'max_holdings': max_holdings,
                                            'strategy': investment_strategy,  # MISSING
                                            'method': optimization_method,
                                            'target_return': target_return,
                                            'target_risk': target_risk,
                                            'dividend_weight': dividend_weight,
                                            'return_weight': return_weight,
                                            'size_weight': size_weight,      # MISSING
                                            'lookback_days': lookback_days,
                                            'cov_method': cov_method         # MISSING
                                        }
                                        
                                        portfolio = app.create_portfolio_with_user_params(filtered_data, opt_params)
                                        app.portfolio = portfolio
                                        
                                        st.success(f"✅ Portfolio built with {len(portfolio)} stocks!")
                                    else:
                                        st.error("❌ No stocks meet the screening criteria. Please adjust filters.")
                                        
                                        # Enhanced debugging section
                                        with st.expander("🔍 **DEBUG: Why are no stocks qualifying?**", expanded=True):
                                            app.debug_screening_criteria(criteria)
                            
                            # Debug button for troubleshooting
                            if st.button("🔍 Debug Current Filters", type="secondary"):
                                criteria = {
                                    'min_market_cap': min_market_cap * 1000,  # Convert B to M (since data is in millions)
                                    'min_avg_trading_value': min_liquidity,
                                    'min_dividend_yield': min_div_yield,
                                    'max_dividend_yield': max_div_yield,
                                    'min_price': 1.0,
                                    'max_volatility': max_volatility
                                }
                                app.debug_screening_criteria(criteria)
                        
                        with col2:
                            if hasattr(app, 'portfolio') and app.portfolio is not None and len(app.portfolio) > 0:
                                st.markdown("### 📊 Portfolio Composition")
                                
                                # Portfolio metrics
                                port_metrics = {
                                    "Holdings": len(app.portfolio),
                                    "Avg Dividend Yield": f"{app.portfolio['ANNUAL_DIV_YIELD'].mean()*100:.2f}%",
                                    "Weighted Avg Div Yield": f"{(app.portfolio['ANNUAL_DIV_YIELD'] * app.portfolio['PORTFOLIO_WEIGHT']).sum()*100:.2f}%",
                                    "Total Market Cap": f"{app.portfolio['MARKETCAP'].sum()/1000:.2f}B THB",  # Convert M to B
                                    "Largest Position": f"{app.portfolio['PORTFOLIO_WEIGHT'].max()*100:.1f}%",
                                    "Portfolio HHI": f"{(app.portfolio['PORTFOLIO_WEIGHT']**2).sum():.3f}"
                                }
                                
                                # Display metrics in columns
                                metric_cols = st.columns(3)
                                for i, (key, value) in enumerate(port_metrics.items()):
                                    metric_cols[i % 3].metric(key, value)
                                
                                # Holdings table
                                st.markdown("#### Top 10 Holdings")
                                holdings_display = app.portfolio.head(10)[
                                    ['CIQ_TICKER', 'COMPANYNAME', 'PORTFOLIO_WEIGHT', 'ANNUAL_DIV_YIELD', 'MARKETCAP']
                                ].copy()
                                holdings_display['Weight (%)'] = holdings_display['PORTFOLIO_WEIGHT'] * 100
                                holdings_display['Div Yield (%)'] = holdings_display['ANNUAL_DIV_YIELD'] * 100
                                holdings_display['Market Cap (M THB)'] = holdings_display['MARKETCAP'].round(1)  # Already in millions
                                
                                st.dataframe(
                                    holdings_display[['CIQ_TICKER', 'Weight (%)', 'Div Yield (%)', 'Market Cap (M THB)']],
                                    use_container_width=True
                                )
                                
                                # Portfolio visualization
                                top_10 = app.portfolio.head(10)
                                others_weight = app.portfolio.iloc[10:]['PORTFOLIO_WEIGHT'].sum() if len(app.portfolio) > 10 else 0
                                
                                labels = list(top_10['CIQ_TICKER']) + (['Others'] if others_weight > 0 else [])
                                values = list(top_10['PORTFOLIO_WEIGHT'] * 100) + ([others_weight * 100] if others_weight > 0 else [])
                                
                                fig = px.pie(
                                    values=values, 
                                    names=labels, 
                                    title="Portfolio Allocation",
                                    color_discrete_sequence=px.colors.qualitative.Set3
                                )
                                fig.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            else:
                                st.info("👆 Configure parameters and click 'Build Portfolio' to see results")
                                
                                # Show helpful tips
                                st.markdown("### 💡 **Getting Started Tips**")
                                st.markdown("""
                                1. **Check the data summary above** to understand your data ranges
                                2. **Start with relaxed filters** (lower minimums, higher maximums)  
                                3. **Use the Debug button** to see exactly where stocks are filtered out
                                4. **Gradually tighten filters** once you see stocks passing
                                
                                **Common Issues:**
                                - Market cap too high → Lower to 1B THB
                                - Dividend yield too high → Lower to 1% 
                                - Trading volume too high → Lower to 1M THB
                                """)
                    
                    with tab2:
                        st.markdown('<div class="sub-header">📈 Efficient Frontier Analysis</div>', unsafe_allow_html=True)
                        
                        if hasattr(app, 'portfolio') and app.portfolio is not None and len(app.portfolio) > 0:
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.markdown("### 🎛️ Frontier Parameters")
                                
                                frontier_points = st.slider("Number of Points", 20, 100, 50, 10)
                                risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0, 0.1) / 100
                                
                                if st.button("📊 Calculate Efficient Frontier"):
                                    with st.spinner("Calculating efficient frontier..."):
                                        # Get portfolio data for frontier calculation
                                        portfolio_tickers = app.portfolio['CIQ_TICKER'].tolist()
                                        latest_date = app.df['PRICINGDATE'].max()
                                        lookback_start = latest_date - timedelta(days=365)
                                        
                                        returns_matrix = app.calculate_returns_matrix(
                                            portfolio_tickers, lookback_start, latest_date
                                        )
                                        
                                        if not returns_matrix.empty and len(returns_matrix.columns) >= 5:
                                            expected_returns = returns_matrix.mean() * 252
                                            cov_matrix = returns_matrix.cov() * 252
                                            
                                            # Calculate efficient frontier
                                            frontier_returns, frontier_vols, _ = app.calculate_efficient_frontier(
                                                expected_returns.values, cov_matrix.values, frontier_points
                                            )
                                            
                                            # Store in session state
                                            st.session_state.frontier_data = {
                                                'returns': frontier_returns,
                                                'volatilities': frontier_vols,
                                                'expected_returns': expected_returns,
                                                'portfolio_tickers': portfolio_tickers,
                                                'cov_matrix': cov_matrix,
                                                'risk_free_rate': risk_free_rate
                                            }
                                            
                                            st.success("✅ Efficient frontier calculated!")
                                        else:
                                            st.error("❌ Insufficient data for frontier calculation")
                            
                            with col2:
                                if 'frontier_data' in st.session_state:
                                    st.markdown("### 📈 Efficient Frontier Plot")
                                    
                                    frontier_data = st.session_state.frontier_data
                                    
                                    # Create frontier plot
                                    fig = go.Figure()
                                    
                                    # Efficient frontier
                                    fig.add_trace(go.Scatter(
                                        x=frontier_data['volatilities'] * 100,
                                        y=frontier_data['returns'] * 100,
                                        mode='lines',
                                        name='Efficient Frontier',
                                        line=dict(color='blue', width=3)
                                    ))
                                    
                                    # Individual assets
                                    individual_returns = frontier_data['expected_returns'] * 100
                                    individual_vols = np.sqrt(np.diag(frontier_data['cov_matrix'])) * 100
                                    
                                    fig.add_trace(go.Scatter(
                                        x=individual_vols,
                                        y=individual_returns,
                                        mode='markers',
                                        name='Individual Assets',
                                        marker=dict(size=8, color='red', symbol='circle'),
                                        text=frontier_data['portfolio_tickers'],
                                        textposition="top center"
                                    ))
                                    
                                    # Current portfolio point
                                    if hasattr(app, 'portfolio'):
                                        portfolio_return = (app.portfolio['ANNUAL_DIV_YIELD'] * app.portfolio['PORTFOLIO_WEIGHT']).sum()
                                        # Estimate portfolio volatility (simplified)
                                        portfolio_vol = np.sqrt(np.mean(individual_vols**2)) / 100  # Rough estimate
                                        
                                        fig.add_trace(go.Scatter(
                                            x=[portfolio_vol * 100],
                                            y=[portfolio_return * 100],
                                            mode='markers',
                                            name='Current Portfolio',
                                            marker=dict(size=15, color='green', symbol='star')
                                        ))
                                    
                                    # Capital allocation line (if risk-free rate > 0)
                                    if frontier_data['risk_free_rate'] > 0 and len(frontier_data['volatilities']) > 0:
                                        try:
                                            # Find maximum Sharpe ratio point
                                            sharpe_ratios = (frontier_data['returns'] - frontier_data['risk_free_rate']) / frontier_data['volatilities']
                                            
                                            # Check if we have valid Sharpe ratios
                                            valid_sharpe = sharpe_ratios[~np.isnan(sharpe_ratios) & ~np.isinf(sharpe_ratios)]
                                            
                                            if len(valid_sharpe) > 0:
                                                max_sharpe_idx = np.argmax(sharpe_ratios)
                                                max_sharpe_vol = frontier_data['volatilities'][max_sharpe_idx]
                                                max_sharpe_ret = frontier_data['returns'][max_sharpe_idx]
                                                
                                                # Capital allocation line
                                                cal_vols = np.linspace(0, max(frontier_data['volatilities']) * 1.2, 50)
                                                cal_returns = frontier_data['risk_free_rate'] + sharpe_ratios[max_sharpe_idx] * cal_vols
                                                
                                                fig.add_trace(go.Scatter(
                                                    x=cal_vols * 100,
                                                    y=cal_returns * 100,
                                                    mode='lines',
                                                    name='Capital Allocation Line',
                                                    line=dict(color='orange', dash='dash', width=2)
                                                ))
                                                
                                                # Maximum Sharpe ratio point
                                                fig.add_trace(go.Scatter(
                                                    x=[max_sharpe_vol * 100],
                                                    y=[max_sharpe_ret * 100],
                                                    mode='markers',
                                                    name='Max Sharpe Portfolio',
                                                    marker=dict(size=12, color='purple', symbol='diamond')
                                                ))
                                            else:
                                                st.warning("⚠️ Unable to calculate valid Sharpe ratios for capital allocation line")
                                                
                                        except Exception as sharpe_error:
                                            st.warning(f"⚠️ Capital allocation line calculation failed: {str(sharpe_error)}")
                                    
                                    fig.update_layout(
                                        title='Efficient Frontier & Portfolio Analysis',
                                        xaxis_title='Risk (Annual Volatility %)',
                                        yaxis_title='Expected Return (Annual %)',
                                        width=800,
                                        height=600,
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Frontier statistics
                                    st.markdown("### 📊 Frontier Statistics")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    # Check if we have valid frontier data before calculating statistics
                                    if len(frontier_data['volatilities']) > 0 and len(frontier_data['returns']) > 0:
                                        with col1:
                                            min_vol_idx = np.argmin(frontier_data['volatilities'])
                                            st.metric("Min Risk Portfolio", 
                                                     f"{frontier_data['volatilities'][min_vol_idx]*100:.1f}%",
                                                     f"Return: {frontier_data['returns'][min_vol_idx]*100:.1f}%")
                                        
                                        with col2:
                                            max_ret_idx = np.argmax(frontier_data['returns'])
                                            st.metric("Max Return Portfolio", 
                                                     f"{frontier_data['returns'][max_ret_idx]*100:.1f}%",
                                                     f"Risk: {frontier_data['volatilities'][max_ret_idx]*100:.1f}%")
                                        
                                        with col3:
                                            if frontier_data['risk_free_rate'] > 0:
                                                try:
                                                    sharpe_ratios = (frontier_data['returns'] - frontier_data['risk_free_rate']) / frontier_data['volatilities']
                                                    valid_sharpe = sharpe_ratios[~np.isnan(sharpe_ratios) & ~np.isinf(sharpe_ratios)]
                                                    
                                                    if len(valid_sharpe) > 0:
                                                        max_sharpe_idx = np.argmax(sharpe_ratios)
                                                        st.metric("Max Sharpe Ratio", 
                                                                 f"{sharpe_ratios[max_sharpe_idx]:.3f}",
                                                                 f"Risk: {frontier_data['volatilities'][max_sharpe_idx]*100:.1f}%")
                                                    else:
                                                        st.metric("Max Sharpe Ratio", "N/A", "Invalid data")
                                                except:
                                                    st.metric("Max Sharpe Ratio", "N/A", "Calculation failed")
                                        
                                        with col4:
                                            st.metric("Risk-Free Rate", 
                                                     f"{frontier_data['risk_free_rate']*100:.1f}%",
                                                     "Annual")
                                    else:
                                        st.error("❌ No valid frontier data available for statistics")
                                
                                else:
                                    st.info("👆 Click 'Calculate Efficient Frontier' to see the analysis")
                        
                        else:
                            st.warning("⚠️ Please build a portfolio first in the 'Portfolio Builder' tab")
                    
                    with tab3:
                        st.markdown('<div class="sub-header">📊 Performance Analysis</div>', unsafe_allow_html=True)
                        st.info("🚧 Performance backtesting feature coming soon! This will include historical performance analysis, risk metrics, and benchmark comparison.")
                        
                    with tab4:
                        st.markdown('<div class="sub-header">🔍 Stock Screener Results</div>', unsafe_allow_html=True)
                        
                        # Quick screener with current parameters
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("### 🎛️ Quick Screen")
                            
                            screen_div_yield = st.slider("Min Dividend Yield (%)", 0.0, 15.0, 1.0, 0.5, key="screen_div") / 100
                            screen_market_cap = st.slider("Min Market Cap (B THB)", 0.01, 100.0, 0.5, 0.01, key="screen_mcap")
                            screen_liquidity = st.slider("Min Trading Volume (M THB)", 0.1, 200.0, 1.0, 0.1, key="screen_liq")
                            
                            if st.button("🔍 Screen Stocks"):
                                criteria = {
                                    'min_market_cap': screen_market_cap * 1000,  # Convert B to M (since data is in millions)
                                    'min_avg_trading_value': screen_liquidity,
                                    'min_dividend_yield': screen_div_yield,
                                    'max_dividend_yield': 0.30,  # 30% max
                                    'min_price': 0.1,
                                    'max_volatility': 2.0  # 200% max
                                }
                                
                                screened_stocks = app.apply_user_filters(criteria)
                                st.session_state.screened_stocks = screened_stocks
                        
                        with col2:
                            if 'screened_stocks' in st.session_state and len(st.session_state.screened_stocks) > 0:
                                stocks = st.session_state.screened_stocks
                                
                                st.markdown(f"### 📈 Found {len(stocks)} Stocks")
                                
                                # Display table
                                display_stocks = stocks[[
                                    'CIQ_TICKER', 'COMPANYNAME', 'ANNUAL_DIV_YIELD', 'MARKETCAP', 
                                    'PRICECLOSE', 'AVG_TRADING_VALUE_3M', 'PRICE_VOLATILITY_3M'
                                ]].copy()
                                
                                display_stocks['Div Yield (%)'] = display_stocks['ANNUAL_DIV_YIELD'] * 100
                                display_stocks['Market Cap (M THB)'] = display_stocks['MARKETCAP'].round(1)  # Already in millions
                                display_stocks['Price (THB)'] = display_stocks['PRICECLOSE']
                                display_stocks['Avg Trading (M THB)'] = display_stocks['AVG_TRADING_VALUE_3M']
                                display_stocks['Volatility (%)'] = display_stocks['PRICE_VOLATILITY_3M'] * 100
                                
                                st.dataframe(
                                    display_stocks[['CIQ_TICKER', 'COMPANYNAME', 'Div Yield (%)', 'Market Cap (M THB)', 
                                                  'Price (THB)', 'Avg Trading (M THB)', 'Volatility (%)']].round(2),
                                    use_container_width=True,
                                    height=400
                                )
                                
                                # Summary statistics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Avg Div Yield", f"{stocks['ANNUAL_DIV_YIELD'].mean()*100:.2f}%")
                                with col2:
                                    st.metric("Median Market Cap", f"{stocks['MARKETCAP'].median():.1f}M THB")  # Already in millions
                                with col3:
                                    st.metric("Avg Volatility", f"{stocks['PRICE_VOLATILITY_3M'].mean()*100:.1f}%")
                                with col4:
                                    st.metric("Total Market Cap", f"{stocks['MARKETCAP'].sum()/1000:.2f}B THB")  # Convert M to B
                            
                            else:
                                st.info("👆 Use the screening controls to find stocks matching your criteria")
                    
                    with tab5:
                        st.markdown('<div class="sub-header">📋 Strategy Report</div>', unsafe_allow_html=True)
                        
                        if hasattr(app, 'portfolio') and app.portfolio is not None and len(app.portfolio) > 0:
                            
                            # Executive Summary
                            st.markdown("## Executive Summary")
                            
                            summary_cols = st.columns(4)
                            
                            with summary_cols[0]:
                                st.metric("Portfolio Holdings", len(app.portfolio))
                            
                            with summary_cols[1]:
                                avg_yield = app.portfolio['ANNUAL_DIV_YIELD'].mean() * 100
                                st.metric("Average Dividend Yield", f"{avg_yield:.2f}%")
                            
                            with summary_cols[2]:
                                weighted_yield = (app.portfolio['ANNUAL_DIV_YIELD'] * app.portfolio['PORTFOLIO_WEIGHT']).sum() * 100
                                st.metric("Weighted Dividend Yield", f"{weighted_yield:.2f}%")
                            
                            with summary_cols[3]:
                                total_mcap = app.portfolio['MARKETCAP'].sum() / 1000  # Convert M to B
                                st.metric("Total Market Cap", f"{total_mcap:.2f}B THB")
                            
                            # Investment Thesis
                            st.markdown("## Investment Thesis")
                            st.markdown("""
                            **Thai SET High Dividend Strategy** employs quantitative optimization to construct portfolios 
                            that maximize risk-adjusted returns from Thailand's dividend-paying companies. The strategy 
                            combines systematic screening with Modern Portfolio Theory to deliver:
                            
                            - **Enhanced Returns**: Target companies with sustainable dividend yields
                            - **Risk Management**: Comprehensive volatility and liquidity controls  
                            - **Optimization**: Maximum Sharpe ratio portfolio construction
                            - **Diversification**: Optimal correlation-based position sizing
                            """)
                            
                            # Portfolio Details
                            st.markdown("## Portfolio Composition")
                            
                            # Sector analysis (simplified)
                            st.markdown("### Top Holdings Analysis")
                            
                            top_5 = app.portfolio.head(5)
                            for idx, stock in top_5.iterrows():
                                with st.expander(f"{stock['CIQ_TICKER']} - {stock['PORTFOLIO_WEIGHT']*100:.1f}% allocation"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**Company**: {stock['COMPANYNAME']}")
                                        st.write(f"**Dividend Yield**: {stock['ANNUAL_DIV_YIELD']*100:.2f}%")
                                        st.write(f"**Market Cap**: {stock['MARKETCAP']:.1f}M THB")  # Already in millions
                                    with col2:
                                        st.write(f"**Price**: {stock['PRICECLOSE']:.2f} THB")
                                        st.write(f"**Volatility**: {stock['PRICE_VOLATILITY_3M']*100:.1f}%")
                                        st.write(f"**Avg Trading**: {stock['AVG_TRADING_VALUE_3M']:.1f}M THB")
                            
                            # Risk Metrics
                            st.markdown("## Risk Analysis")
                            
                            risk_cols = st.columns(3)
                            
                            with risk_cols[0]:
                                concentration = (app.portfolio['PORTFOLIO_WEIGHT']**2).sum()
                                st.metric("Portfolio Concentration (HHI)", f"{concentration:.3f}")
                                if concentration > 0.15:
                                    st.warning("⚠️ High concentration risk")
                                else:
                                    st.success("✅ Well diversified")
                            
                            with risk_cols[1]:
                                max_position = app.portfolio['PORTFOLIO_WEIGHT'].max() * 100
                                st.metric("Largest Position", f"{max_position:.1f}%")
                                if max_position > 10:
                                    st.warning("⚠️ Large single position")
                                else:
                                    st.success("✅ Position size controlled")
                            
                            with risk_cols[2]:
                                avg_vol = app.portfolio['PRICE_VOLATILITY_3M'].mean() * 100
                                st.metric("Average Volatility", f"{avg_vol:.1f}%")
                                if avg_vol > 40:
                                    st.warning("⚠️ High volatility portfolio")
                                else:
                                    st.success("✅ Moderate risk level")
                            
                            # Download report
                            st.markdown("## 📥 Export Results")
                            
                            if st.button("📊 Generate Excel Report"):
                                # Create downloadable Excel file
                                from io import BytesIO
                                
                                buffer = BytesIO()
                                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                    app.portfolio.to_excel(writer, sheet_name='Portfolio', index=False)
                                    
                                    if 'screened_stocks' in st.session_state:
                                        st.session_state.screened_stocks.to_excel(writer, sheet_name='Screened_Stocks', index=False)
                                
                                st.download_button(
                                    label="📥 Download Excel Report",
                                    data=buffer.getvalue(),
                                    file_name=f"thai_dividend_strategy_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        
                        else:
                            st.info("📊 Build a portfolio first to generate the strategy report")
                        
            else:
                # Clean up progress bar if shown
                if file_size_mb > 50:
                    progress_bar.empty()
                    status_text.empty()
                    
                if load_success and (app.df is None or len(app.df) == 0):
                    st.sidebar.error("❌ Data loading resulted in empty dataset!")
                    st.error("**Data Loading Failed - Empty Result**")
                    st.write("The file was read but no valid data was found. Possible issues:")
                    st.write("- All rows had missing critical data")
                    st.write("- Date conversion failed")
                    st.write("- Data format issues")
                else:
                    st.sidebar.error("❌ Data loading failed!")
                    st.error("**Data Loading Failed**")
                    st.write("Please check that your CSV file contains the required columns:")
                    st.write("- PRICINGDATE, CIQ_TICKER, COMPANYNAME")  
                    st.write("- PRICECLOSE, DIVADJPRICE, MARKETCAP")
                    st.write("- VOLUME, VWAP")
                return
                
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")
            st.error("**Data Loading Error**")
            st.write(f"Error details: {str(e)}")
            
            # Provide specific guidance based on error type
            error_str = str(e).lower()
            if "memory" in error_str:
                st.write("**Memory Error - File too large:**")
                st.write("- Try splitting your data into smaller files")
                st.write("- Increase system RAM")
                st.write("- Use data sampling for testing")
            elif "encoding" in error_str:
                st.write("**Encoding Error:**") 
                st.write("- Save CSV with UTF-8 encoding")
                st.write("- Check for special characters")
            elif "nonetype" in error_str:
                st.write("**Data Structure Error:**")
                st.write("- Check CSV file format")
                st.write("- Ensure all required columns exist")
                st.write("- Verify data is not corrupted")
            else:
                st.write("**General troubleshooting:**")
                st.write("- File is a valid CSV format")
                st.write("- File is not corrupted")
                st.write("- File contains required columns")
                st.write("- File size is within limits")
            return
    
    else:
        # Welcome screen - no file uploaded
        st.markdown("""
        ## 🚀 Welcome to Thai SET High Dividend Strategy
        
        This interactive application allows you to:
        
        ### 📊 **Portfolio Optimization Features**
        - **Interactive Stock Screening** with customizable criteria and **enhanced debugging**
        - **Modern Portfolio Theory** optimization (Maximum Sharpe, Minimum Risk, Target Return/Risk)
        - **Efficient Frontier Analysis** with real-time calculations
        - **Dynamic Parameter Adjustment** for risk and return preferences
        
        ### 🎯 **Advanced Analytics & Debugging**
        - **Real-time Portfolio Construction** based on your preferences
        - **Enhanced Filter Debugging** to understand why stocks don't qualify
        - **Comprehensive Data Quality Checks** and diagnostic information
        - **Step-by-Step Filter Analysis** to optimize your screening criteria
        - **Professional Reporting** with downloadable results
        
        ### 📈 **Getting Started**
        1. **Upload your Thai SET stock data** (CSV format) using the sidebar
        2. **Review the data summary** to understand your dataset
        3. **Configure screening parameters** (start with relaxed filters!)
        4. **Use the debug tools** to optimize your filtering criteria
        5. **Build your optimized portfolio** with custom risk/return targets
        6. **Analyze the efficient frontier** and explore trade-offs
        7. **Generate professional reports** for presentation
        
        ---
        
        ### 📋 **Required Data Format**
        Your CSV file should contain columns for:
        - `PRICINGDATE`, `CIQ_TICKER`, `COMPANYNAME`  
        - `PRICECLOSE`, `DIVADJPRICE`, `MARKETCAP`
        - `VOLUME`, `VWAP`
        
        ### 🔍 **New Debugging Features**
        - **Data Quality Assessment**: Check for missing values and data ranges
        - **Step-by-Step Filter Analysis**: See exactly where stocks are eliminated
        - **Suggested Filter Adjustments**: Automatic recommendations for better results
        - **Visual Filter Progress**: Color-coded success/failure indicators
        
        **Ready to start? Upload your data file in the sidebar! 👈**
        """)
        
        # Show sample data format
        with st.expander("📋 Sample Data Format"):
            sample_data = pd.DataFrame({
                'PRICINGDATE': ['2023-01-01', '2023-01-02'],
                'CIQ_TICKER': ['PTTEP', 'CPALL'],
                'COMPANYNAME': ['PTT Exploration', 'CP All'],
                'PRICECLOSE': [100.0, 50.0],
                'DIVADJPRICE': [95.0, 48.0],
                'MARKETCAP': [500000000000, 300000000000],
                'VOLUME': [1000000, 2000000],
                'VWAP': [99.5, 49.5]
            })
            st.dataframe(sample_data)

if __name__ == "__main__":
    main()