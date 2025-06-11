#!/usr/bin/env python3
"""
Enhanced AutoHedge Trading Bot
==============================

Advanced algorithmic trading bot that combines:
- AI-driven market analysis
- Technical indicator signals
- Advanced risk management
- Real-time market data
- Multi-strategy execution
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import schedule
from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Import enhanced components
from autohedge.main import AutoHedge
from autohedge.technical_indicators import TechnicalAnalyzer, TechnicalSignal
from autohedge.risk_management import AdvancedRiskManager, RiskLevel, RiskMetrics, PositionSizing
from autohedge.market_data import MarketDataProvider, MarketData
from autohedge.tools.e_trade_wrapper import ETradeClient
from utils.price_fetcher import price_fetcher

# Load environment variables
load_dotenv()

# Configure logging
logger.add(
    "logs/enhanced_trading_bot_{time}.log",
    rotation="100 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

@dataclass
class TradingSignal:
    """Comprehensive trading signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    ai_recommendation: str
    technical_consensus: TechnicalSignal
    risk_assessment: RiskMetrics
    position_sizing: PositionSizing
    reasoning: str
    timestamp: datetime

@dataclass
class AdvancedStrategy:
    """Enhanced trading strategy configuration"""
    name: str
    description: str
    stocks: List[str]
    max_position_size: float
    risk_level: RiskLevel
    rebalance_frequency: str
    use_technical_analysis: bool = True
    use_ai_analysis: bool = True
    use_sentiment_analysis: bool = True
    max_correlation: float = 0.8
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15

class EnhancedTradingBot:
    """
    Advanced trading bot with multi-component analysis
    
    Features:
    - AI-powered market analysis
    - Technical indicator integration
    - Advanced risk management
    - Real-time market data
    - Portfolio optimization
    - Multi-strategy execution
    """
    
    def __init__(self, 
                 strategy: AdvancedStrategy, 
                 initial_capital: float = 100000,
                 paper_trading: bool = True):
        """
        Initialize enhanced trading bot
        
        Args:
            strategy: Trading strategy configuration
            initial_capital: Starting capital
            paper_trading: Use paper trading or real money
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.paper_trading = paper_trading
        
        # Initialize components
        self.etrade_client = ETradeClient(use_sandbox=paper_trading)
        self.autohedge = AutoHedge(
            name=f"Enhanced-{strategy.name}",
            description=strategy.description,
            stocks=strategy.stocks
        )
        self.technical_analyzer = TechnicalAnalyzer()
        self.risk_manager = AdvancedRiskManager(
            initial_capital=initial_capital,
            risk_level=strategy.risk_level,
            max_portfolio_risk=0.02,
            max_single_position=0.1
        )
        self.market_data = MarketDataProvider()
        
        # Performance tracking
        self.signals_history: List[TradingSignal] = []
        self.performance_metrics: Dict = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_capital': initial_capital
        }
        
        # Market state tracking
        self.market_regime = 'NORMAL'  # BULL, BEAR, NORMAL, VOLATILE
        self.sector_performance: Dict[str, float] = {}
        
        logger.info(f"Initialized enhanced trading bot: {strategy.name}")
        logger.info(f"Strategy: {len(strategy.stocks)} stocks, "
                   f"Risk level: {strategy.risk_level.name}")
    
    async def run_trading_cycle(self) -> Dict[str, TradingSignal]:
        """
        Execute complete enhanced trading cycle
        
        Returns:
            Dictionary of trading signals by symbol
        """
        logger.info("🚀 Starting enhanced trading cycle...")
        
        try:
            # 1. Market Environment Analysis
            await self._analyze_market_environment()
            
            # 2. Get real-time market data
            market_quotes = self.market_data.get_multiple_quotes(self.strategy.stocks)
            
            # 3. Generate signals for each stock
            trading_signals = {}
            
            for symbol in self.strategy.stocks:
                if symbol in market_quotes:
                    signal = await self._generate_comprehensive_signal(
                        symbol, market_quotes[symbol]
                    )
                    trading_signals[symbol] = signal
                    
            # 4. Portfolio-level risk check
            portfolio_signals = self._portfolio_optimization(trading_signals)
            
            # 5. Execute trades
            await self._execute_signals(portfolio_signals)
            
            # 6. Update performance metrics
            self._update_performance_metrics()
            
            logger.success(f"✅ Trading cycle completed. Generated {len(portfolio_signals)} signals")
            return portfolio_signals
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
            return {}
    
    async def _analyze_market_environment(self):
        """Analyze overall market conditions"""
        try:
            # Get market indices
            indices = self.market_data.get_market_indices()
            
            # Get sector performance
            self.sector_performance = self.market_data.get_sector_performance()
            
            # Determine market regime
            if 'VIX' in indices:
                vix_level = indices['VIX'].current_price
                if vix_level > 30:
                    self.market_regime = 'VOLATILE'
                elif vix_level < 15:
                    self.market_regime = 'BULL'
                else:
                    self.market_regime = 'NORMAL'
            
            # Log market state
            logger.info(f"📊 Market Environment: {self.market_regime}")
            if 'S&P 500' in indices:
                sp500_change = indices['S&P 500'].change_percent
                logger.info(f"📈 S&P 500: {sp500_change:+.2f}%")
                
        except Exception as e:
            logger.error(f"Error analyzing market environment: {e}")
    
    async def _generate_comprehensive_signal(self, 
                                           symbol: str, 
                                           market_data: MarketData) -> TradingSignal:
        """
        Generate comprehensive trading signal combining all analysis methods
        
        Args:
            symbol: Stock symbol
            market_data: Real-time market data
            
        Returns:
            Comprehensive trading signal
        """
        try:
            logger.info(f"🔍 Analyzing {symbol}...")
            
            # Get historical data for technical analysis
            hist_data = self.market_data.get_historical_data(symbol, period="6mo")
            
            # 1. AI Analysis (if enabled)
            ai_recommendation = "HOLD"
            ai_reasoning = "No AI analysis"
            
            if self.strategy.use_ai_analysis:
                ai_analysis = await self._get_ai_analysis(symbol, market_data, hist_data)
                ai_recommendation = self._extract_ai_recommendation(ai_analysis)
                ai_reasoning = ai_analysis[:200] + "..." if len(ai_analysis) > 200 else ai_analysis
            
            # 2. Technical Analysis (if enabled)
            technical_signals = {}
            technical_consensus = TechnicalSignal('None', 'NEUTRAL', 0.0, 0.0)
            
            if self.strategy.use_technical_analysis and not hist_data.empty:
                technical_signals = self.technical_analyzer.analyze_stock(hist_data, symbol)
                technical_consensus = self.technical_analyzer.get_consensus_signal(technical_signals)
            
            # 3. Risk Assessment
            risk_metrics = self.risk_manager.assess_stock_risk(symbol, hist_data)
            
            # 4. Position Sizing
            signal_strength = self._calculate_signal_strength(
                ai_recommendation, technical_consensus, market_data
            )
            
            win_probability = self._estimate_win_probability(
                technical_signals, market_data, risk_metrics
            )
            
            position_sizing = self.risk_manager.calculate_position_size(
                symbol=symbol,
                current_price=market_data.current_price,
                signal_strength=signal_strength,
                win_probability=win_probability,
                risk_metrics=risk_metrics
            )
            
            # 5. Generate final signal
            final_action, confidence = self._synthesize_signals(
                ai_recommendation, technical_consensus, risk_metrics, market_data
            )
            
            # 6. Create comprehensive signal
            signal = TradingSignal(
                symbol=symbol,
                action=final_action,
                confidence=confidence,
                ai_recommendation=ai_recommendation,
                technical_consensus=technical_consensus,
                risk_assessment=risk_metrics,
                position_sizing=position_sizing,
                reasoning=self._generate_reasoning(
                    ai_reasoning, technical_consensus, risk_metrics, final_action
                ),
                timestamp=datetime.now()
            )
            
            self.signals_history.append(signal)
            
            logger.info(f"📊 {symbol}: {final_action} (confidence: {confidence:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            # Return neutral signal
            return TradingSignal(
                symbol, 'HOLD', 0.1, 'HOLD', 
                TechnicalSignal('Error', 'NEUTRAL', 0.0, 0.0),
                RiskMetrics(0.05, 0.2, 0.0, 0.3, 1.0, 0.0, 0.5, 0.5),
                PositionSizing(symbol, 1, market_data.current_price, 
                              market_data.current_price * 0.95, 
                              market_data.current_price * 1.05, 0.01, 0.1),
                f"Error in analysis: {e}", datetime.now()
            )
    
    async def _get_ai_analysis(self, 
                              symbol: str, 
                              market_data: MarketData, 
                              hist_data: pd.DataFrame) -> str:
        """Get AI-powered analysis from AutoHedge"""
        try:
            # Enhanced prompt with market context
            task = f"""
            Comprehensive Analysis for {symbol}:
            
            Current Market Data:
            - Price: ${market_data.current_price:.2f}
            - Daily Change: {market_data.change_percent:+.2f}%
            - Volume: {market_data.volume:,}
            - Market Cap: ${market_data.market_cap:,.0f}
            - P/E Ratio: {market_data.pe_ratio}
            
            Market Environment: {self.market_regime}
            
            Please analyze:
            1. Technical price action and trends
            2. Fundamental valuation metrics
            3. Market sentiment and news impact
            4. Sector performance context
            5. Risk factors and volatility assessment
            6. Entry/exit recommendations with specific price levels
            7. Position sizing guidance for risk level: {self.strategy.risk_level.name}
            
            Provide a clear BUY/SELL/HOLD recommendation with reasoning.
            """
            
            analysis = self.autohedge.run(task=task)
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {symbol}: {e}")
            return f"AI analysis unavailable: {e}"
    
    def _extract_ai_recommendation(self, analysis: str) -> str:
        """Extract recommendation from AI analysis"""
        analysis_lower = analysis.lower()
        
        # Enhanced keyword detection
        buy_keywords = ['strong buy', 'buy', 'bullish', 'long', 'accumulate', 'positive']
        sell_keywords = ['strong sell', 'sell', 'bearish', 'short', 'avoid', 'negative']
        
        buy_score = sum(1 for word in buy_keywords if word in analysis_lower)
        sell_score = sum(1 for word in sell_keywords if word in analysis_lower)
        
        if buy_score > sell_score and buy_score >= 2:
            return 'BUY'
        elif sell_score > buy_score and sell_score >= 2:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_signal_strength(self, 
                                  ai_rec: str, 
                                  tech_consensus: TechnicalSignal, 
                                  market_data: MarketData) -> float:
        """Calculate overall signal strength"""
        strength = 0.0
        
        # AI component (40% weight)
        if ai_rec == 'BUY':
            strength += 0.4
        elif ai_rec == 'SELL':
            strength += 0.4
        else:
            strength += 0.1
        
        # Technical component (40% weight)
        if tech_consensus.signal in ['BUY', 'SELL']:
            strength += 0.4 * tech_consensus.strength
        else:
            strength += 0.1
        
        # Volume confirmation (20% weight)
        if market_data.volume > market_data.avg_volume * 1.5:
            strength += 0.2
        else:
            strength += 0.1
        
        return min(1.0, strength)
    
    def _estimate_win_probability(self, 
                                 technical_signals: Dict[str, TechnicalSignal],
                                 market_data: MarketData, 
                                 risk_metrics: RiskMetrics) -> float:
        """Estimate probability of winning trade"""
        base_prob = 0.5  # Base 50% probability
        
        # Adjust based on technical signals
        if technical_signals:
            consensus = self.technical_analyzer.get_consensus_signal(technical_signals)
            if consensus.signal in ['BUY', 'SELL']:
                base_prob += 0.1 * consensus.strength
        
        # Adjust based on Sharpe ratio
        if risk_metrics.sharpe_ratio > 1.0:
            base_prob += 0.1
        elif risk_metrics.sharpe_ratio < 0:
            base_prob -= 0.1
        
        # Adjust based on market environment
        if self.market_regime == 'BULL':
            base_prob += 0.05
        elif self.market_regime == 'VOLATILE':
            base_prob -= 0.05
        
        return max(0.3, min(0.8, base_prob))
    
    def _synthesize_signals(self, 
                           ai_rec: str,
                           tech_consensus: TechnicalSignal,
                           risk_metrics: RiskMetrics,
                           market_data: MarketData) -> Tuple[str, float]:
        """Synthesize all signals into final recommendation"""
        
        # Scoring system
        buy_score = 0.0
        sell_score = 0.0
        
        # AI recommendation (weight: 0.4)
        if ai_rec == 'BUY':
            buy_score += 0.4
        elif ai_rec == 'SELL':
            sell_score += 0.4
        
        # Technical consensus (weight: 0.3)
        if tech_consensus.signal == 'BUY':
            buy_score += 0.3 * tech_consensus.strength
        elif tech_consensus.signal == 'SELL':
            sell_score += 0.3 * tech_consensus.strength
        
        # Risk factors (weight: 0.2)
        if risk_metrics.volatility > 0.4:  # High volatility = avoid
            sell_score += 0.1
        if risk_metrics.sharpe_ratio > 1.0:  # Good risk-adjusted returns
            buy_score += 0.1
        elif risk_metrics.sharpe_ratio < 0:
            sell_score += 0.1
        
        # Market environment (weight: 0.1)
        if self.market_regime == 'BULL':
            buy_score += 0.05
        elif self.market_regime == 'BEAR':
            sell_score += 0.05
        elif self.market_regime == 'VOLATILE':
            sell_score += 0.03  # Be more cautious
        
        # Determine final action and confidence
        if buy_score > sell_score and buy_score > 0.5:
            return 'BUY', buy_score
        elif sell_score > buy_score and sell_score > 0.5:
            return 'SELL', sell_score
        else:
            return 'HOLD', max(buy_score, sell_score, 0.3)
    
    def _generate_reasoning(self, 
                           ai_reasoning: str,
                           tech_consensus: TechnicalSignal,
                           risk_metrics: RiskMetrics,
                           action: str) -> str:
        """Generate human-readable reasoning for the signal"""
        reasoning = f"Action: {action}\n\n"
        
        reasoning += f"AI Analysis: {ai_reasoning[:100]}...\n\n"
        
        reasoning += f"Technical Analysis: {tech_consensus.indicator} shows {tech_consensus.signal} "
        reasoning += f"with {tech_consensus.strength:.2f} strength\n\n"
        
        reasoning += f"Risk Assessment:\n"
        reasoning += f"- Volatility: {risk_metrics.volatility:.2f}\n"
        reasoning += f"- Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}\n"
        reasoning += f"- Max Drawdown: {risk_metrics.max_drawdown:.2f}\n\n"
        
        reasoning += f"Market Environment: {self.market_regime}"
        
        return reasoning
    
    def _portfolio_optimization(self, 
                               signals: Dict[str, TradingSignal]) -> Dict[str, TradingSignal]:
        """Optimize signals at portfolio level"""
        # Filter signals based on portfolio constraints
        optimized_signals = {}
        
        # Sort by confidence
        sorted_signals = sorted(
            signals.items(), 
            key=lambda x: x[1].confidence, 
            reverse=True
        )
        
        buy_signals = []
        sell_signals = []
        
        for symbol, signal in sorted_signals:
            if signal.action == 'BUY':
                buy_signals.append((symbol, signal))
            elif signal.action == 'SELL':
                sell_signals.append((symbol, signal))
            else:
                optimized_signals[symbol] = signal
        
        # Apply position limits (max positions per strategy risk level)
        max_positions = self.risk_manager.risk_params[self.strategy.risk_level]['max_positions']
        
        # Select top buy signals
        for symbol, signal in buy_signals[:max_positions//2]:
            optimized_signals[symbol] = signal
        
        # Select top sell signals
        for symbol, signal in sell_signals[:max_positions//2]:
            optimized_signals[symbol] = signal
        
        logger.info(f"📊 Portfolio optimization: {len(optimized_signals)} final signals")
        return optimized_signals
    
    async def _execute_signals(self, signals: Dict[str, TradingSignal]):
        """Execute trading signals"""
        for symbol, signal in signals.items():
            if signal.action in ['BUY', 'SELL']:
                try:
                    await self._place_order(signal)
                except Exception as e:
                    logger.error(f"Failed to execute {signal.action} for {symbol}: {e}")
    
    async def _place_order(self, signal: TradingSignal):
        """Place order based on signal"""
        try:
            if signal.confidence < 0.3:
                logger.info(f"Skipping {signal.symbol}: Low confidence ({signal.confidence:.2f})")
                return
            
            quantity = signal.position_sizing.recommended_size
            current_price = signal.position_sizing.max_position_value / quantity
            
            logger.info(f"💰 Placing {signal.action} order: {quantity} shares of {signal.symbol}")
            
            # Place the order
            order_response = self.etrade_client.place_order(
                account_id=self.etrade_client.account_id,
                symbol=signal.symbol,
                quantity=quantity,
                action=signal.action,
                price=current_price
            )
            
            # Update risk manager positions
            self.risk_manager.update_position(
                signal.symbol, signal.action, quantity, current_price
            )
            
            # Log trade
            self._log_trade(signal, order_response)
            
        except Exception as e:
            logger.error(f"Error placing order for {signal.symbol}: {e}")
    
    def _log_trade(self, signal: TradingSignal, order_response: Dict):
        """Log trade details"""
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'symbol': signal.symbol,
            'action': signal.action,
            'quantity': signal.position_sizing.recommended_size,
            'confidence': signal.confidence,
            'reasoning': signal.reasoning[:100] + "...",
            'order_response': order_response
        }
        
        logger.info(f"🔄 Trade executed: {json.dumps(trade_log, indent=2)}")
    
    def _update_performance_metrics(self):
        """Update performance tracking"""
        # Get current portfolio value
        risk_report = self.risk_manager.get_risk_report()
        current_capital = risk_report['total_capital']
        
        self.performance_metrics['current_capital'] = current_capital
        
        # Calculate returns
        total_return = (current_capital - self.initial_capital) / self.initial_capital
        
        logger.info(f"📈 Portfolio Performance: {total_return:+.2%}")
    
    def get_status_report(self) -> Dict:
        """Generate comprehensive status report"""
        risk_report = self.risk_manager.get_risk_report()
        
        recent_signals = [
            {
                'symbol': s.symbol,
                'action': s.action,
                'confidence': s.confidence,
                'timestamp': s.timestamp.isoformat()
            }
            for s in self.signals_history[-10:]  # Last 10 signals
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy.name,
            'market_regime': self.market_regime,
            'performance_metrics': self.performance_metrics,
            'risk_report': risk_report,
            'recent_signals': recent_signals,
            'sector_performance': self.sector_performance
        }

# Strategy Examples
def create_momentum_strategy() -> AdvancedStrategy:
    """Create a momentum-based strategy"""
    return AdvancedStrategy(
        name="AI_Momentum",
        description="AI-driven momentum strategy with technical confirmation",
        stocks=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        max_position_size=10000,
        risk_level=RiskLevel.MODERATE,
        rebalance_frequency="daily",
        use_technical_analysis=True,
        use_ai_analysis=True,
        max_correlation=0.7,
        stop_loss_pct=0.05,
        take_profit_pct=0.15
    )

def create_conservative_strategy() -> AdvancedStrategy:
    """Create a conservative dividend strategy"""
    return AdvancedStrategy(
        name="Conservative_AI",
        description="Conservative strategy focusing on quality stocks",
        stocks=["JNJ", "PG", "KO", "PFE", "VZ"],
        max_position_size=5000,
        risk_level=RiskLevel.CONSERVATIVE,
        rebalance_frequency="weekly",
        use_technical_analysis=True,
        use_ai_analysis=True,
        max_correlation=0.8,
        stop_loss_pct=0.03,
        take_profit_pct=0.10
    )

async def main():
    """Main execution function"""
    logger.info("🚀 Starting Enhanced AutoHedge Trading Bot")
    
    # Create strategy
    strategy = create_momentum_strategy()
    
    # Initialize bot
    bot = EnhancedTradingBot(
        strategy=strategy,
        initial_capital=100000,
        paper_trading=True
    )
    
    # Run trading cycle
    try:
        signals = await bot.run_trading_cycle()
        
        # Print status report
        status = bot.get_status_report()
        print(json.dumps(status, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 