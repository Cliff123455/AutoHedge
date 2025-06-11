import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math
from loguru import logger
from datetime import datetime, timedelta

class RiskLevel(Enum):
    """Risk level enumeration"""
    CONSERVATIVE = 1
    MODERATE = 2
    AGGRESSIVE = 3

@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    var_95: float  # Value at Risk (95%)
    max_drawdown: float  # Maximum historical drawdown
    sharpe_ratio: float  # Risk-adjusted returns
    volatility: float  # Price volatility
    beta: float  # Market correlation
    skewness: float  # Return distribution skew
    correlation_score: float  # Portfolio correlation
    liquidity_score: float  # Liquidity assessment

@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    symbol: str
    recommended_size: int  # Number of shares
    max_position_value: float  # Dollar amount
    stop_loss_price: float  # Stop loss level
    take_profit_price: float  # Take profit level
    position_risk: float  # Risk per trade (%)
    confidence_score: float  # 0-1 confidence in sizing

class AdvancedRiskManager:
    """
    Sophisticated risk management system for trading algorithms
    
    Implements multiple risk models including:
    - Kelly Criterion position sizing
    - Value at Risk (VaR) calculations
    - Maximum drawdown protection
    - Portfolio correlation analysis
    - Dynamic stop losses
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 max_portfolio_risk: float = 0.02,  # 2% max risk per trade
                 max_single_position: float = 0.1,   # 10% max in single stock
                 risk_level: RiskLevel = RiskLevel.MODERATE):
        """
        Initialize risk management system
        
        Args:
            initial_capital: Starting portfolio value
            max_portfolio_risk: Maximum risk per trade as % of portfolio
            max_single_position: Maximum single position as % of portfolio
            risk_level: Overall risk appetite
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_single_position = max_single_position
        self.risk_level = risk_level
        
        # Risk tracking
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.daily_returns: List[float] = []
        
        # Risk parameters by level
        self.risk_params = {
            RiskLevel.CONSERVATIVE: {
                'max_positions': 5,
                'correlation_limit': 0.7,
                'volatility_limit': 0.3,
                'kelly_fraction': 0.25
            },
            RiskLevel.MODERATE: {
                'max_positions': 8,
                'correlation_limit': 0.8,
                'volatility_limit': 0.4,
                'kelly_fraction': 0.5
            },
            RiskLevel.AGGRESSIVE: {
                'max_positions': 12,
                'correlation_limit': 0.9,
                'volatility_limit': 0.6,
                'kelly_fraction': 0.75
            }
        }
        
        logger.info(f"Initialized risk manager with {risk_level.name} profile")
    
    def assess_stock_risk(self, 
                         symbol: str, 
                         price_data: pd.DataFrame,
                         market_data: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """
        Comprehensive risk assessment for a single stock
        
        Args:
            symbol: Stock symbol
            price_data: Historical OHLCV data
            market_data: Market index data for beta calculation
            
        Returns:
            RiskMetrics object with comprehensive risk assessment
        """
        try:
            returns = price_data['Close'].pct_change().dropna()
            
            # Calculate risk metrics
            var_95 = self._calculate_var(returns, confidence=0.95)
            max_drawdown = self._calculate_max_drawdown(price_data['Close'])
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            beta = self._calculate_beta(returns, market_data) if market_data is not None else 1.0
            skewness = returns.skew()
            correlation_score = self._calculate_portfolio_correlation(symbol, returns)
            liquidity_score = self._calculate_liquidity_score(price_data)
            
            metrics = RiskMetrics(
                var_95=var_95,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                beta=beta,
                skewness=skewness,
                correlation_score=correlation_score,
                liquidity_score=liquidity_score
            )
            
            logger.info(f"Risk assessment for {symbol}: VaR={var_95:.3f}, Vol={volatility:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error assessing risk for {symbol}: {e}")
            # Return conservative default metrics
            return RiskMetrics(0.05, 0.2, 0.0, 0.3, 1.0, 0.0, 0.5, 0.5)
    
    def calculate_position_size(self,
                               symbol: str,
                               current_price: float,
                               signal_strength: float,
                               win_probability: float,
                               risk_metrics: RiskMetrics,
                               technical_stop: Optional[float] = None) -> PositionSizing:
        """
        Calculate optimal position size using multiple methods
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            signal_strength: Trading signal strength (0-1)
            win_probability: Estimated probability of winning trade
            risk_metrics: Risk assessment for the stock
            technical_stop: Technical stop loss level
            
        Returns:
            PositionSizing recommendation
        """
        try:
            # Kelly Criterion sizing
            kelly_size = self._kelly_position_size(win_probability, signal_strength, risk_metrics)
            
            # Risk-based sizing
            risk_size = self._risk_based_position_size(current_price, risk_metrics)
            
            # Volatility-adjusted sizing
            vol_size = self._volatility_adjusted_size(current_price, risk_metrics)
            
            # Take the most conservative size
            recommended_shares = min(kelly_size, risk_size, vol_size)
            
            # Apply position limits
            max_position_value = self.current_capital * self.max_single_position
            max_shares_by_value = int(max_position_value / current_price)
            recommended_shares = min(recommended_shares, max_shares_by_value)
            
            # Ensure minimum viable position
            recommended_shares = max(1, recommended_shares)
            
            # Calculate stop loss and take profit
            stop_loss = self._calculate_stop_loss(current_price, risk_metrics, technical_stop)
            take_profit = self._calculate_take_profit(current_price, signal_strength, risk_metrics)
            
            # Calculate position risk
            position_value = recommended_shares * current_price
            risk_per_share = current_price - stop_loss
            total_risk = recommended_shares * risk_per_share
            position_risk = total_risk / self.current_capital
            
            # Confidence score based on multiple factors
            confidence = self._calculate_confidence_score(
                signal_strength, risk_metrics, position_risk
            )
            
            sizing = PositionSizing(
                symbol=symbol,
                recommended_size=recommended_shares,
                max_position_value=position_value,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                position_risk=position_risk,
                confidence_score=confidence
            )
            
            logger.info(f"Position sizing for {symbol}: {recommended_shares} shares, "
                       f"risk={position_risk:.3f}, confidence={confidence:.3f}")
            
            return sizing
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            # Return minimal position
            return PositionSizing(symbol, 1, current_price, current_price * 0.95, 
                                current_price * 1.05, 0.01, 0.1)
    
    def _kelly_position_size(self, win_prob: float, signal_strength: float, 
                           risk_metrics: RiskMetrics) -> int:
        """Calculate position size using Kelly Criterion"""
        # Estimate average win/loss based on signal strength and volatility
        avg_win = signal_strength * 0.1  # Assume up to 10% gains
        avg_loss = risk_metrics.volatility * 0.5  # Conservative loss estimate
        
        # Kelly fraction calculation
        if avg_loss > 0:
            kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        else:
            kelly_fraction = 0.1
        
        # Apply risk level adjustment
        kelly_fraction *= self.risk_params[self.risk_level]['kelly_fraction']
        
        # Convert to position size
        risk_capital = self.current_capital * self.max_portfolio_risk
        kelly_capital = self.current_capital * max(0.01, min(0.1, kelly_fraction))
        
        return int(kelly_capital / 100)  # Approximate shares (simplified)
    
    def _risk_based_position_size(self, price: float, risk_metrics: RiskMetrics) -> int:
        """Calculate position size based on maximum risk tolerance"""
        max_risk_capital = self.current_capital * self.max_portfolio_risk
        estimated_risk_per_share = price * risk_metrics.volatility * 0.5  # Estimate
        
        if estimated_risk_per_share > 0:
            max_shares = int(max_risk_capital / estimated_risk_per_share)
        else:
            max_shares = int(max_risk_capital / price * 0.1)
        
        return max(1, max_shares)
    
    def _volatility_adjusted_size(self, price: float, risk_metrics: RiskMetrics) -> int:
        """Adjust position size based on volatility"""
        base_size = int(self.current_capital * 0.05 / price)  # 5% base allocation
        
        # Reduce size for high volatility
        vol_adjustment = 1.0 / (1.0 + risk_metrics.volatility * 2)
        adjusted_size = int(base_size * vol_adjustment)
        
        return max(1, adjusted_size)
    
    def _calculate_stop_loss(self, price: float, risk_metrics: RiskMetrics, 
                           technical_stop: Optional[float]) -> float:
        """Calculate dynamic stop loss level"""
        # ATR-based stop (2x volatility)
        atr_stop = price * (1 - risk_metrics.volatility * 0.5)
        
        # Risk-based stop (max 2% loss)
        risk_stop = price * (1 - self.max_portfolio_risk * 2)
        
        # Technical stop if provided
        if technical_stop:
            stops = [atr_stop, risk_stop, technical_stop]
        else:
            stops = [atr_stop, risk_stop]
        
        # Use the highest (most conservative) stop
        return max(stops)
    
    def _calculate_take_profit(self, price: float, signal_strength: float, 
                             risk_metrics: RiskMetrics) -> float:
        """Calculate take profit level"""
        # Base take profit at 1.5x risk (risk/reward ratio)
        base_target = price * (1 + signal_strength * 0.1)
        
        # Adjust for volatility - higher vol = higher targets
        vol_adjustment = 1 + risk_metrics.volatility * 0.5
        
        return base_target * vol_adjustment
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 30:
            return 0.05  # Default 5% VaR
        
        return abs(returns.quantile(1 - confidence))
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
        
        excess_return = returns.mean() * 252 - risk_free_rate  # Annualized
        volatility = returns.std() * np.sqrt(252)
        
        return excess_return / volatility if volatility > 0 else 0.0
    
    def _calculate_beta(self, stock_returns: pd.Series, 
                       market_data: pd.DataFrame) -> float:
        """Calculate stock beta relative to market"""
        if market_data is None or len(market_data) < len(stock_returns):
            return 1.0
        
        market_returns = market_data['Close'].pct_change().dropna()
        
        # Align data
        min_length = min(len(stock_returns), len(market_returns))
        stock_aligned = stock_returns.tail(min_length)
        market_aligned = market_returns.tail(min_length)
        
        covariance = np.cov(stock_aligned, market_aligned)[0][1]
        market_variance = np.var(market_aligned)
        
        return covariance / market_variance if market_variance > 0 else 1.0
    
    def _calculate_portfolio_correlation(self, symbol: str, returns: pd.Series) -> float:
        """Calculate correlation with existing portfolio"""
        if not self.positions:
            return 0.0  # No existing positions
        
        # Simplified correlation calculation
        # In practice, would calculate with actual position returns
        return 0.5  # Placeholder
    
    def _calculate_liquidity_score(self, price_data: pd.DataFrame) -> float:
        """Calculate liquidity score based on volume and price stability"""
        if 'Volume' not in price_data.columns:
            return 0.5  # Default neutral score
        
        avg_volume = price_data['Volume'].mean()
        volume_consistency = 1 - price_data['Volume'].std() / avg_volume
        
        # Higher volume and consistency = higher liquidity
        return min(1.0, max(0.1, volume_consistency))
    
    def _calculate_confidence_score(self, signal_strength: float, 
                                  risk_metrics: RiskMetrics, 
                                  position_risk: float) -> float:
        """Calculate overall confidence in position sizing"""
        # Factors that increase confidence
        signal_factor = signal_strength
        sharpe_factor = min(1.0, max(0.0, (risk_metrics.sharpe_ratio + 1) / 3))
        liquidity_factor = risk_metrics.liquidity_score
        
        # Factors that decrease confidence
        volatility_penalty = max(0.0, 1.0 - risk_metrics.volatility)
        risk_penalty = max(0.0, 1.0 - position_risk / self.max_portfolio_risk)
        
        confidence = (signal_factor + sharpe_factor + liquidity_factor + 
                     volatility_penalty + risk_penalty) / 5
        
        return min(1.0, max(0.1, confidence))
    
    def portfolio_risk_check(self) -> Dict[str, bool]:
        """Check if portfolio meets risk constraints"""
        checks = {
            'max_positions': len(self.positions) <= self.risk_params[self.risk_level]['max_positions'],
            'total_exposure': sum(pos.get('value', 0) for pos in self.positions.values()) <= self.current_capital * 0.95,
            'correlation_limit': True,  # Simplified
            'var_limit': True  # Simplified
        }
        
        return checks
    
    def update_position(self, symbol: str, action: str, quantity: int, price: float):
        """Update position tracking"""
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'value': 0, 'avg_price': 0}
        
        pos = self.positions[symbol]
        
        if action.upper() == 'BUY':
            new_quantity = pos['quantity'] + quantity
            total_cost = pos['value'] + (quantity * price)
            pos['avg_price'] = total_cost / new_quantity if new_quantity > 0 else price
            pos['quantity'] = new_quantity
            pos['value'] = total_cost
        elif action.upper() == 'SELL':
            pos['quantity'] = max(0, pos['quantity'] - quantity)
            pos['value'] = pos['quantity'] * pos['avg_price']
            
            if pos['quantity'] == 0:
                del self.positions[symbol]
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        total_value = sum(pos.get('value', 0) for pos in self.positions.values())
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_capital': self.current_capital,
            'total_invested': total_value,
            'cash_available': self.current_capital - total_value,
            'number_of_positions': len(self.positions),
            'portfolio_utilization': total_value / self.current_capital,
            'risk_level': self.risk_level.name,
            'positions': dict(self.positions),
            'risk_checks': self.portfolio_risk_check()
        }
        
        return report 