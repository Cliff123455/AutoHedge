# Google Sheets Stock Scanner - Updated Formulas (GOOGLEFINANCE Version)

## Sheet Structure
- **Scanner Tab**: Main data with columns A-L
- **Settings Tab**: API key in cell D2, configuration parameters

## Updated Formulas (Using GOOGLEFINANCE for reliability)

### Column A: Ticker
```
=A2
```
(Manual entry: AAPL, TSLA, NVDA, etc.)

### Column B: Company Name
```
=GOOGLEFINANCE(A2,"name")
```

### Column C: Current Price
```
=GOOGLEFINANCE(A2,"price")
```

### Column D: Day Change %
```
=GOOGLEFINANCE(A2,"changepct")
```

### Column E: Volume
```
=GOOGLEFINANCE(A2,"volume")
```

### Column F: 50 EMA (Approximation using 50-day average)
```
=AVERAGE(GOOGLEFINANCE(A2,"close",TODAY()-50,TODAY()))
```

### Column G: 200 EMA (Approximation using 200-day average)
```
=AVERAGE(GOOGLEFINANCE(A2,"close",TODAY()-200,TODAY()))
```

### Column H: RSI(7) - Simplified momentum indicator
```
=IF(C2>F2,IF(D2>2,"Overbought",IF(D2>0,"Bullish","Neutral")),IF(D2<-2,"Oversold",IF(D2<0,"Bearish","Neutral")))
```

### Column I: VWAP Indicator (Price vs 20-day average)
```
=IF(C2>AVERAGE(GOOGLEFINANCE(A2,"close",TODAY()-20,TODAY())),"Above VWAP","Below VWAP")
```

### Column J: ROC(9) - Rate of Change indicator
```
=IF(ISERROR(GOOGLEFINANCE(A2,"close",TODAY()-9,TODAY()-9)),0,(C2-GOOGLEFINANCE(A2,"close",TODAY()-9,TODAY()-9))/GOOGLEFINANCE(A2,"close",TODAY()-9,TODAY()-9)*100)
```

### Column K: Volume Spike Detection
```
=IF(E2>AVERAGE(GOOGLEFINANCE(A2,"volume",TODAY()-20,TODAY()))*1.5,"High Volume","Normal")
```

### Column L: Buy/Sell Signal
```
=IF(AND(D2>3,K2="High Volume",C2>F2),"STRONG BUY",IF(AND(D2>1.5,C2>F2),"BUY",IF(AND(D2<-3,K2="High Volume"),"STRONG SELL",IF(D2<-1.5,"SELL","HOLD"))))
```

## Alternative: Yahoo Finance API Approach (More reliable than Alpha Vantage)

If you want real-time data with technical indicators, use this IMPORTXML approach with Yahoo Finance:

### Column C: Current Price (Yahoo Finance)
```
=IMPORTXML("https://finance.yahoo.com/quote/"&A2,"//fin-streamer[@data-field='regularMarketPrice']/@value")
```

### Column D: Day Change % (Yahoo Finance)
```
=IMPORTXML("https://finance.yahoo.com/quote/"&A2,"//fin-streamer[@data-field='regularMarketChangePercent']/@value")
```

### Column E: Volume (Yahoo Finance)
```
=IMPORTXML("https://finance.yahoo.com/quote/"&A2,"//fin-streamer[@data-field='regularMarketVolume']/@value")
```

## Apps Script Alternative (Most Reliable)

For the most reliable solution, create a Google Apps Script function:

```javascript
function getStockData(ticker) {
  try {
    const response = UrlFetchApp.fetch(`https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${ticker}&apikey=YOUR_API_KEY`);
    const data = JSON.parse(response.getContentText());
    const quote = data['Global Quote'];
    
    return [
      quote['05. price'],
      quote['10. change percent'].replace('%', ''),
      quote['06. volume']
    ];
  } catch (error) {
    return ['Error', 'Error', 'Error'];
  }
}
```

Then use in sheets: `=getStockData(A2)`

## Settings Tab Setup

Put these values in your Settings tab:
- **A1**: "API Key"
- **B1**: "Alpha Vantage"
- **A2**: "Key:"
- **B2**: [Your API Key]
- **A4**: "Refresh Rate"
- **B4**: "5 minutes"
- **A6**: "Volume Threshold"
- **B6**: "1.5x average"

## Recommended Approach

1. **Start with GOOGLEFINANCE** - Most reliable, updates every 15-20 minutes
2. **Add Yahoo Finance IMPORTXML** for real-time prices (if needed)
3. **Use Apps Script** for advanced technical indicators

The GOOGLEFINANCE approach will be much more stable and won't give you parsing errors. 