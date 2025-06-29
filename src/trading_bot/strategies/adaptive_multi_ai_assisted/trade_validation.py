"""
Pydantic models for trade validation responses
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class ValidationDecision(str, Enum):
    """AI validation decisions"""
    APPROVE = "APPROVE"
    REJECT = "REJECT" 
    CONDITIONAL = "CONDITIONAL"


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class MarketSentiment(str, Enum):
    """Market sentiment assessment"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNCERTAIN = "UNCERTAIN"


class TechnicalQuality(str, Enum):
    """Technical setup quality"""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


class TradeValidationResponse(BaseModel):
    """Structured response from AI trade validation"""
    
    # Core decision
    decision: ValidationDecision = Field(
        description="Whether to approve, reject, or conditionally approve the trade"
    )
    
    confidence: int = Field(
        ge=1, le=10,
        description="Confidence level in the decision (1-10 scale)"
    )
    
    # Assessment breakdown
    technical_quality: TechnicalQuality = Field(
        description="Quality of technical setup"
    )
    
    risk_level: RiskLevel = Field(
        description="Overall risk assessment"
    )
    
    market_sentiment: MarketSentiment = Field(
        description="Current market sentiment assessment"
    )
    
    # Detailed reasoning
    primary_reasoning: str = Field(
        max_length=300,
        description="Main reason for the decision in 1-2 sentences"
    )
    
    risk_factors: List[str] = Field(
        max_items=3,
        description="Up to 3 key risk factors identified"
    )
    
    supporting_factors: List[str] = Field(
        max_items=3,
        description="Up to 3 factors supporting the trade"
    )
    
    # Modifications (if conditional)
    suggested_modifications: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Specific modifications if decision is CONDITIONAL"
    )
    
    position_size_adjustment: Optional[float] = Field(
        default=None,
        ge=0.1, le=2.0,
        description="Position size multiplier (0.1 = 10% size, 2.0 = 200% size)"
    )
    
    stop_loss_adjustment: Optional[float] = Field(
        default=None,
        ge=0.5, le=3.0,
        description="Stop loss multiplier (0.5 = tighter, 3.0 = wider)"
    )
    
    # Market context
    timing_assessment: str = Field(
        max_length=150,
        description="Assessment of trade timing"
    )
    
    similar_setup_history: Optional[str] = Field(
        default=None,
        max_length=150,
        description="How similar setups have performed historically"
    )


class ValidationStats(BaseModel):
    """Statistics for validation performance tracking"""
    
    total_validations: int
    approvals: int
    rejections: int
    conditionals: int
    
    average_confidence: float
    
    # Performance metrics (if available)
    approved_trades_win_rate: Optional[float] = None
    rejected_trades_saved_loss: Optional[float] = None
    
    # Common reasons
    top_rejection_reasons: List[str] = Field(max_items=3)
    top_approval_factors: List[str] = Field(max_items=3)


class ValidationHistory(BaseModel):
    """Historical validation record"""
    
    timestamp: datetime
    symbol: str
    strategy_signal: str
    ai_decision: ValidationDecision
    confidence: int
    
    # Outcome tracking (filled in later)
    actual_outcome: Optional[str] = None  # "WIN", "LOSS", "BREAK_EVEN"
    actual_return: Optional[float] = None
    trade_duration_minutes: Optional[int] = None
    
    # Was the AI right?
    ai_accuracy: Optional[bool] = None


class ContextSummary(BaseModel):
    """Condensed market context for validation"""
    
    current_price: float
    price_change_1h: float
    price_change_24h: float
    
    rsi_level: float
    volume_vs_average: float
    volatility_percentile: float
    
    detected_regime: str
    volatility_mood: str
    
    key_technical_levels: Dict[str, float] = Field(
        description="Support/resistance levels"
    )
    
    market_hours: str
    day_of_week: str


class CandleData(BaseModel):
    """Individual candle data for price action history"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    volume_ratio: Optional[float] = None
    rsi: Optional[float] = None
    change_pct: Optional[float] = None


class ValidationRequest(BaseModel):
    """Complete validation request package"""
    
    # Trade signal details
    symbol: str
    timeframe: str
    signal_reason: str
    proposed_entry_price: float
    proposed_stop_loss: float
    proposed_position_size: float
    
    # Market context
    context: ContextSummary
    
    # Recent price action history
    recent_candles: List[CandleData] = Field(
        max_items=20,
        description="Recent candle data showing price action history"
    )
    
    # Recent performance
    recent_similar_trades: List[Dict[str, Any]] = Field(
        max_items=5,
        description="Recent trades with similar setup"
    )
    
    # Strategy state
    strategy_confidence: float = Field(
        ge=0, le=1,
        description="Algorithm's confidence in the signal"
    )
    
    current_portfolio_exposure: float = Field(
        description="Current portfolio exposure percentage"
    )