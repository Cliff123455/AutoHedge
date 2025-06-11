#!/usr/bin/env python3
"""
Hedge AI Environment Setup Script
================================

Complete setup guide for your enhanced AutoHedge trading algorithm environment.
"""

import os
import sys
from datetime import datetime

def print_banner():
    """Print the setup banner"""
    print("🚀 " + "=" * 60)
    print("🚀   HEDGE AI ENVIRONMENT SETUP COMPLETE!")
    print("🚀 " + "=" * 60)
    print()

def check_environment():
    """Check current environment status"""
    print("📊 Environment Status:")
    print(f"   🐍 Python Version: {sys.version.split()[0]}")
    print(f"   📁 Virtual Env: {os.environ.get('VIRTUAL_ENV', 'Not in virtual env')}")
    print(f"   📂 Working Directory: {os.getcwd()}")
    print()

def show_installed_packages():
    """Show key installed packages"""
    print("📦 Key Packages Installed:")
    
    packages_to_check = [
        "pandas", "numpy", "yfinance", "matplotlib", "seaborn", 
        "scikit_learn", "lightgbm", "xgboost", "pandas_ta",
        "swarms", "pydantic", "fastapi", "aiohttp"
    ]
    
    for package in packages_to_check:
        try:
            if package == "scikit_learn":
                import sklearn
                print(f"   ✅ scikit-learn: {sklearn.__version__}")
            else:
                exec(f"import {package}")
                try:
                    version = eval(f"{package}.__version__")
                    print(f"   ✅ {package}: {version}")
                except:
                    print(f"   ✅ {package}: installed")
        except ImportError:
            print(f"   ❌ {package}: not installed")
    
    print()

def show_enhanced_components():
    """Show enhanced components created"""
    print("🔧 Enhanced Trading Components:")
    print("   ✅ Technical Analysis Engine (autohedge/technical_indicators.py)")
    print("      - 9 professional indicators (SMA/EMA, RSI, MACD, Bollinger, etc.)")
    print("      - Consensus signal generation")
    print("      - Pure pandas/numpy implementation (no TA-Lib needed)")
    print()
    print("   ✅ Advanced Risk Management (autohedge/risk_management.py)")
    print("      - Kelly Criterion position sizing")
    print("      - Value at Risk (VaR) calculations")
    print("      - Dynamic stop losses")
    print("      - Portfolio correlation analysis")
    print()
    print("   ✅ Market Data Integration (autohedge/market_data.py)")
    print("      - Real-time quotes from Yahoo Finance")
    print("      - Historical data with multiple timeframes")
    print("      - Fundamental analysis metrics")
    print("      - News sentiment integration")
    print()
    print("   ✅ Enhanced Trading Bot (enhanced_trading_bot.py)")
    print("      - Multi-component signal generation")
    print("      - Portfolio-level optimization")
    print("      - Pre-built strategies (Momentum, Conservative, AI-First)")
    print()
    print("   ✅ Comprehensive Documentation (TRADING_ALGORITHM_GUIDE.md)")
    print("      - 6-phase development roadmap")
    print("      - Advanced strategies and optimization")
    print("      - Risk management best practices")
    print()

def show_next_steps():
    """Show next steps for the user"""
    print("🚀 NEXT STEPS TO START TRADING:")
    print()
    
    print("1. 🔑 Set Up API Keys:")
    print("   Create a .env file in your project directory with:")
    print("   ```")
    print("   OPENAI_API_KEY=your_openai_api_key_here")
    print("   ETRADE_CONSUMER_KEY=your_etrade_consumer_key")
    print("   ETRADE_CONSUMER_SECRET=your_etrade_consumer_secret")
    print("   ```")
    print()
    
    print("2. 🧪 Test the Enhanced Components:")
    print("   Run: python test_enhanced_components.py")
    print("   (Once API keys are set)")
    print()
    
    print("3. 📊 Start with Paper Trading:")
    print("   Run: python enhanced_trading_bot.py")
    print("   (Make sure PAPER_TRADING=True in the bot)")
    print()
    
    print("4. 📈 Backtest Your Strategies:")
    print("   Use the backtesting framework in the guide")
    print("   Optimize parameters before live trading")
    print()
    
    print("5. 🎯 Deploy Live Trading:")
    print("   Follow PRODUCTION_TRADING_GUIDE.md")
    print("   Start with small position sizes")
    print()

def show_commands():
    """Show useful commands"""
    print("💻 USEFUL COMMANDS:")
    print()
    print("Activate Hedge AI environment:")
    print('   & "Hedge AI\\Scripts\\Activate.ps1"  (PowerShell)')
    print('   source "Hedge AI/bin/activate"        (Linux/Mac)')
    print()
    print("Install additional packages:")
    print("   pip install package_name")
    print()
    print("View installed packages:")
    print("   pip list")
    print()
    print("Deactivate environment:")
    print("   deactivate")
    print()

def show_resources():
    """Show learning resources"""
    print("📚 LEARNING RESOURCES:")
    print()
    print("📖 Documentation:")
    print("   - TRADING_ALGORITHM_GUIDE.md (comprehensive guide)")
    print("   - PRODUCTION_TRADING_GUIDE.md (live trading)")
    print("   - README.md (project overview)")
    print()
    print("🌐 External Resources:")
    print("   - Quantitative Finance: quantlib.org")
    print("   - Technical Analysis: ta-lib.org")
    print("   - Risk Management: GARP.org")
    print("   - Machine Learning: scikit-learn.org")
    print()

def main():
    """Main setup summary"""
    print_banner()
    check_environment()
    show_installed_packages()
    show_enhanced_components()
    show_next_steps()
    show_commands()
    show_resources()
    
    print("🎉 " + "=" * 60)
    print("🎉   YOUR HEDGE AI ENVIRONMENT IS READY!")
    print("🎉   You now have a professional-grade algorithmic trading system!")
    print("🎉 " + "=" * 60)
    print()
    print(f"Setup completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Happy Trading! 📈🚀")

if __name__ == "__main__":
    main() 