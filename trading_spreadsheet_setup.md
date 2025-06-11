# Google Sheets Trading Tracker Setup

## 📊 **Copy-Paste Google Sheets Templates**

### **Sheet 1: Daily Trading Log**

Copy this into Google Sheets (Row 1 = Headers):

```
Date	Symbol	Entry Price	Exit Price	Shares	Entry Time	Exit Time	P&L	% Gain/Loss	Stop Loss	Target	AI Recommendation	Trade Reason	Risk Amount	Account Balance
=TODAY()	NVDA	=C2	=D2	=E2	=NOW()	=NOW()	=(D2-C2)*E2	=(D2-C2)/C2	=C2*0.93	=C2*1.04	BUY	Breakout	=C2*E2*0.07	=25000
```

### **Key Formulas to Use:**

**P&L Calculation (Column H):**
```
=(D2-C2)*E2
```

**Percentage Gain/Loss (Column I):**
```
=(D2-C2)/C2
```

**Auto Stop Loss -7% (Column J):**
```
=C2*0.93
```

**Auto Target +4% (Column K):**
```
=C2*1.04
```

**Risk Amount (Column M):**
```
=C2*E2*0.07
```

**Running Account Balance (Column N):**
```
=N1+H2
```

### **Sheet 2: Performance Dashboard**

Copy this setup:

```
Metric	Formula	Value
Total Trades	=COUNTA(A:A)-1	
Win Rate	=COUNTIF(H:H,">0")/(COUNTA(H:H)-1)	
Average Win	=AVERAGEIF(H:H,">0",H:H)	
Average Loss	=AVERAGEIF(H:H,"<0",H:H)	
Largest Win	=MAX(H:H)	
Largest Loss	=MIN(H:H)	
Total P&L	=SUM(H:H)	
Return %	=(N2-25000)/25000	
Risk Per Trade	=AVERAGE(M:M)	
Profit Factor	=SUMIF(H:H,">0",H:H)/ABS(SUMIF(H:H,"<0",H:H))	
```

### **Sheet 3: AI Recommendations Tracker**

```
Date	Symbol	AI Action	Confidence	Entry Price	Current Price	AI Reasoning	Followed?	Outcome	Notes
=TODAY()	=B2	BUY	85%	=E2	=GOOGLEFINANCE(B2,"price")	Breakout pattern	YES	=IF(G2="YES",H2,"N/A")	
```

**Formula for Current Price:**
```
=GOOGLEFINANCE(B2,"price")
```

**Formula for Price Change:**
```
=GOOGLEFINANCE(B2,"changepct")
```

## 🎯 **Best Ways to Find Intraday Trading Winners**

### **1. Pre-Market Scanning (8:00-9:30 AM)**

#### **🔥 Gap Scanners - Look for:**
- **Gaps >3%** with high volume
- **News catalysts** (earnings, FDA approvals, analyst upgrades)
- **Sector momentum** (if tech is hot, scan all tech)

#### **Volume Filters:**
- **Volume >500K** in pre-market
- **Relative volume >2x** average
- **Price >$10** (avoid penny stocks)

### **2. Technical Setups That Work**

#### **🚀 Breakout Patterns:**
- **Bull flags** after strong morning move
- **Consolidation** near HOD (High of Day)
- **Support/resistance** breaks with volume

#### **📈 Momentum Patterns:**
- **Higher highs, higher lows**
- **VWAP reclaim** after pullback
- **Moving average** bounces (20/50 EMA)

#### **⚡ Volume Patterns:**
- **Volume spike** on breakout
- **Decreasing volume** on pullback (healthy)
- **Heavy volume** at key levels

### **3. Scanning Tools & Websites**

#### **Free Scanners:**
- **Finviz.com** - Gap scanner, volume leaders
- **TradingView** - Custom screeners
- **Yahoo Finance** - Most active, gainers/losers
- **MarketWatch** - Movers and shakers

#### **Paid Tools (Worth It):**
- **Trade Ideas** - AI-powered scanning
- **Benzinga Pro** - News + scanning
- **TC2000** - Professional charting

### **4. Time-Based Strategies**

#### **🌅 9:30-10:30 AM - "Power Hour"**
- **Gap and go** strategies
- **Breakout plays** from overnight ranges
- **News reaction** trades

#### **🕐 10:30-11:30 AM - "Grind Time"**
- **VWAP bounces**
- **Flag breakouts**
- **Failed breakdown** reversals

#### **🌆 3:30-4:00 PM - "Power Hour"**
- **EOD momentum** plays
- **Institutional buying**
- **Squeeze into close**

### **5. Google Sheets Formulas for Scanning**

#### **Price Change Formula:**
```
=GOOGLEFINANCE("NVDA","changepct")
```

#### **Volume Formula:**
```
=GOOGLEFINANCE("NVDA","volume")
```

#### **Multi-Stock Scanner Setup:**
```
Symbol	Price	Change %	Volume	50-Day Avg Vol	Rel Volume
NVDA	=GOOGLEFINANCE(A2,"price")	=GOOGLEFINANCE(A2,"changepct")	=GOOGLEFINANCE(A2,"volume")	=GOOGLEFINANCE(A2,"volume",TODAY()-50,TODAY())	=C2/E2
TSLA	=GOOGLEFINANCE(A3,"price")	=GOOGLEFINANCE(A3,"changepct")	=GOOGLEFINANCE(A3,"volume")	=GOOGLEFINANCE(A3,"volume",TODAY()-50,TODAY())	=C3/E3
```

#### **Conditional Formatting Rules:**
- **Green** if Change % > 3%
- **Red** if Change % < -3%
- **Bold** if Rel Volume > 2

### **6. News-Based Trading**

#### **🗞️ Key News Sources:**
- **Benzinga** - Real-time news alerts
- **MarketWatch** - Breaking news
- **SEC filings** - 8K forms for material events
- **Earnings whispers** - Estimate revisions

#### **📱 Twitter Follows:**
- @DeItaone - Breaking news
- @FirstSquawk - Market alerts
- @LiveSquawk - Real-time updates
- Sector-specific accounts

### **7. Sector Rotation Plays**

#### **🔥 Hot Sectors to Watch:**
- **AI/Tech** - NVDA, AMD, GOOGL
- **Biotech** - on FDA news
- **Energy** - on oil price moves
- **Financials** - on rate changes

#### **Rotation Indicators:**
- **Sector ETF** performance (XLK, XLF, XLV)
- **Relative strength** vs SPY
- **Volume flow** between sectors

### **8. Risk Management Integration**

#### **Position Sizing Formula:**
```
=MIN(2500, (Account_Balance * 0.1))
```

#### **Shares to Buy:**
```
=FLOOR(Position_Size / Entry_Price, 1)
```

#### **Risk Amount:**
```
=Shares * Entry_Price * 0.07
```

### **9. AI Integration Formula**

#### **AI Confidence Filter:**
```
=IF(AI_Confidence > 0.7, "TRADE", "PASS")
```

#### **Combined Score:**
```
=(Technical_Score * 0.4) + (AI_Confidence * 0.3) + (Volume_Score * 0.3)
```

## 🎯 **Daily Workflow Integration**

### **8:00 AM - Pre-Market Setup:**
1. Update Google Sheet with gap scanner results
2. Run AI advisor: `python ai_advisor_mode.py`
3. Cross-reference AI picks with gap/volume criteria
4. Set alerts for key levels

### **9:30 AM - Market Open:**
1. Monitor flagged stocks
2. Wait for volume confirmation
3. Enter positions only with clear setups
4. Log everything in Google Sheets

### **4:00 PM - Post-Market Review:**
1. Update P&L in spreadsheet
2. Review AI vs actual performance
3. Note lessons learned
4. Plan tomorrow's watchlist

## 📋 **Copy-Paste Watchlist Template**

```
MEGA CAPS (>$500B)	LARGE CAPS ($100-500B)	MID CAPS ($10-100B)	HIGH BETA (>1.5)
AAPL	NVDA	PLTR	TSLA
MSFT	AMD	SNOW	RIVN
GOOGL	CRM	COIN	LCID
AMZN	NFLX	ROKU	ARKK
META	ADBE	ZM	SOXL
```

Want me to create any specific formulas or expand on any of these strategies? 