"""
AI-Enhanced Trading Strategy Integration
Combines algorithmic signals with AI validation
"""

import asyncio
import pandas as pd
import structlog
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from trading_bot.strategies.adaptive_multi_ai_assisted.trade_context import ValidationRequestBuilder
from trading_bot.strategies.adaptive_multi_ai_assisted.trade_validation import TradeValidationResponse, ValidationDecision
from trading_bot.strategies.adaptive_multi_ai_assisted.trade_validation_agent import TradeValidationAgent


logger = structlog.get_logger(__name__)


class AIEnhancedStrategy:
    """
    Wrapper that enhances any trading strategy with AI validation
    
    Usage:
        base_strategy = AdaptiveMultiStrategy(...)
        ai_strategy = AIEnhancedStrategy(base_strategy, use_ai=True)
        
        # Use like normal strategy but with AI validation
        should_buy, reason, stop_loss = await ai_strategy.buy_condition(row, prev_row, df)
    """
    
    def __init__(self, 
                 base_strategy,
                 use_ai_validation: bool = True,
                 ai_threshold_confidence: int = 6,
                 max_validation_time: float = 90):
        """
        Initialize AI-enhanced strategy
        
        Args:
            base_strategy: The underlying trading strategy
            use_ai_validation: Whether to use AI validation
            ai_threshold_confidence: Minimum AI confidence to approve trades
            max_validation_time: Max time to wait for AI response (seconds)
        """
        self.base_strategy = base_strategy
        self.use_ai_validation = use_ai_validation
        self.ai_threshold_confidence = ai_threshold_confidence
        self.max_validation_time = max_validation_time
        
        # Initialize AI components
        if use_ai_validation:
            self.ai_validator = TradeValidationAgent()
            self.request_builder = ValidationRequestBuilder()
        else:
            self.ai_validator = None
            self.request_builder = None
        
        # Performance tracking
        self.stats = {
            'total_algorithm_signals': 0,
            'ai_approved': 0,
            'ai_rejected': 0,
            'ai_conditional': 0,
            'ai_timeouts': 0,
            'validation_errors': 0
        }
        
        logger.info(
            "AI Enhanced Strategy initialized",
            ai_enabled=use_ai_validation,
            confidence_threshold=ai_threshold_confidence
        )
    
    async def buy_condition(self, 
                           row: pd.Series, 
                           prev_row: pd.Series, 
                           df: pd.DataFrame,
                           current_idx: Optional[int] = None) -> Tuple[bool, Optional[str], float]:
        """
        Enhanced buy condition with AI validation
        
        Returns:
            (should_buy, reason, stop_loss) - same interface as base strategy
        """
        
        # Get base algorithm signal
        should_buy, reason, stop_loss = self.base_strategy.buy_condition(row, prev_row)
        self.stats['total_algorithm_signals'] += 1
        
        # If no signal or AI disabled, return base result
        if not should_buy or not self.use_ai_validation:
            return should_buy, reason, stop_loss
        
        logger.info(
            "Algorithm signal detected, validating with AI",
            signal=reason,
            entry_price=row['close']
        )
        
        try:
            # Get AI validation with timeout
            validation = await asyncio.wait_for(
                self._get_ai_validation(row, df, current_idx, reason, stop_loss),
                timeout=self.max_validation_time
            )
            
            return self._process_ai_decision(validation, reason, stop_loss)
            
        except asyncio.TimeoutError:
            logger.warning("AI validation timeout, using algorithm signal")
            self.stats['ai_timeouts'] += 1
            return should_buy, f"AI-TIMEOUT: {reason}", stop_loss
            
        except Exception as e:
            logger.error("AI validation error", error=str(e))
            self.stats['validation_errors'] += 1
            return should_buy, f"AI-ERROR: {reason}", stop_loss
    
    async def _get_ai_validation(self, 
                                row: pd.Series,
                                df: pd.DataFrame,
                                current_idx: Optional[int],
                                reason: str,
                                stop_loss: float) -> TradeValidationResponse:
        """Get AI validation for the trade signal"""
        
        # Determine current index if not provided
        if current_idx is None:
            if hasattr(row, 'name') and hasattr(df.index, 'get_loc'):
                try:
                    current_idx = df.index.get_loc(row.name)
                except KeyError:
                    current_idx = len(df) - 1
            else:
                current_idx = len(df) - 1
        
        # Build validation request
        validation_request = self.request_builder.build_request(
            symbol=getattr(row, 'symbol', 'UNKNOWN'),
            timeframe='5m',  # Could be made configurable
            signal_reason=reason,
            entry_price=float(row['close']),
            stop_loss=float(row['close'] * (1 - stop_loss)),
            position_size=1000.0,  # Placeholder - could be calculated
            row=row,
            df=df,
            current_idx=current_idx,
            strategy_confidence=0.8,  # Could be from base strategy
            portfolio_exposure=0.1   # Could be tracked
        )
        
        # Get AI validation
        return await self.ai_validator.validate_trade_signal(validation_request)
    
    def _process_ai_decision(self, 
                           validation: TradeValidationResponse,
                           original_reason: str,
                           original_stop_loss: float) -> Tuple[bool, str, float]:
        """Process AI validation decision and return appropriate result"""
        
        decision = validation.decision
        confidence = validation.confidence
        
        # Update stats
        if decision == ValidationDecision.APPROVE:
            self.stats['ai_approved'] += 1
        elif decision == ValidationDecision.REJECT:
            self.stats['ai_rejected'] += 1
        else:
            self.stats['ai_conditional'] += 1
        
        # Log the decision
        logger.info(
            "AI validation completed",
            decision=decision.value,
            confidence=confidence,
            reasoning=validation.primary_reasoning
        )
        
        # Process decision
        if decision == ValidationDecision.APPROVE:
            # AI approves - execute trade
            enhanced_reason = f"AI-APPROVED({confidence}/10): {original_reason}"
            return True, enhanced_reason, original_stop_loss
            
        elif decision == ValidationDecision.REJECT:
            # AI rejects - skip trade
            reject_reason = f"AI-REJECTED({confidence}/10): {validation.primary_reasoning}"
            return False, reject_reason, original_stop_loss
            
        else:  # CONDITIONAL
            # AI suggests modifications
            if confidence >= self.ai_threshold_confidence:
                # Accept with modifications
                modified_stop = original_stop_loss
                if validation.stop_loss_adjustment:
                    modified_stop *= validation.stop_loss_adjustment
                
                conditional_reason = f"AI-CONDITIONAL({confidence}/10): {original_reason}"
                if validation.suggested_modifications:
                    conditional_reason += f" | MOD: {validation.suggested_modifications}"
                
                return True, conditional_reason, modified_stop
            else:
                # Confidence too low even for conditional
                reject_reason = f"AI-LOW-CONFIDENCE({confidence}/10): {validation.primary_reasoning}"
                return False, reject_reason, original_stop_loss
    
    def sell_condition(self, position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Delegate sell condition to base strategy (could be enhanced later)"""
        return self.base_strategy.sell_condition(position, row)
    
    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Calculate position size with potential AI adjustments"""
        base_size = self.base_strategy.calculate_position_size(available_balance, current_price, row)
        
        # Could be enhanced with AI position sizing recommendations
        # For now, delegate to base strategy
        return base_size
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Delegate data preparation to base strategy"""
        return self.base_strategy.prepare_data(df)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get combined strategy info including AI stats"""
        base_info = self.base_strategy.get_strategy_info()
        
        ai_info = {
            'ai_validation_enabled': self.use_ai_validation,
            'ai_stats': self.stats.copy()
        }
        
        if self.use_ai_validation and self.ai_validator:
            ai_stats = self.ai_validator.get_validation_stats()
            ai_info['ai_performance'] = {
                'total_validations': ai_stats.total_validations,
                'approval_rate': ai_stats.approvals / max(ai_stats.total_validations, 1),
                'avg_confidence': ai_stats.average_confidence
            }
        
        return {**base_info, **ai_info}
    
    def update_trade_outcome(self, 
                           validation_timestamp: datetime,
                           outcome: str,
                           return_pct: float,
                           duration_minutes: int):
        """Update AI validator with trade outcomes for learning"""
        if self.ai_validator:
            self.ai_validator.update_trade_outcome(
                validation_timestamp,
                outcome,
                return_pct,
                duration_minutes
            )
        
        if self.request_builder:
            self.request_builder.add_trade_outcome(
                signal_reason="trade_outcome",  # Could be more specific
                entry_price=0,  # Not used for outcome tracking
                exit_price=0,   # Not used for outcome tracking
                duration_minutes=duration_minutes,
                outcome=outcome
            )
    
    async def quick_validate_signal(self, 
                                  symbol: str,
                                  signal: str,
                                  price: float,
                                  rsi: float,
                                  volume_ratio: float) -> TradeValidationResponse:
        """Quick validation method for testing"""
        if not self.ai_validator:
            raise ValueError("AI validation not enabled")
        
        return await self.ai_validator.quick_validate(
            symbol=symbol,
            signal_reason=signal,
            current_price=price,
            rsi=rsi,
            volume_ratio=volume_ratio
        )
    
    def print_ai_stats(self):
        """Print AI validation statistics"""
        print("\n🤖 AI VALIDATION STATISTICS")
        print("=" * 40)
        
        total = self.stats['total_algorithm_signals']
        if total == 0:
            print("No algorithm signals processed yet")
            return
        
        print(f"📊 Total Algorithm Signals: {total}")
        print(f"✅ AI Approved: {self.stats['ai_approved']} ({self.stats['ai_approved']/total*100:.1f}%)")
        print(f"❌ AI Rejected: {self.stats['ai_rejected']} ({self.stats['ai_rejected']/total*100:.1f}%)")
        print(f"⚠️  AI Conditional: {self.stats['ai_conditional']} ({self.stats['ai_conditional']/total*100:.1f}%)")
        print(f"⏰ Timeouts: {self.stats['ai_timeouts']}")
        print(f"🚫 Errors: {self.stats['validation_errors']}")
        
        if self.ai_validator:
            recent_stats = self.ai_validator.get_validation_stats()
            if recent_stats.approved_trades_win_rate is not None:
                print(f"🎯 AI Approved Win Rate: {recent_stats.approved_trades_win_rate*100:.1f}%")
            print(f"📈 Avg AI Confidence: {recent_stats.average_confidence:.1f}/10")


# Example usage and testing
async def test_ai_enhanced_strategy():
    """Test the AI enhanced strategy system"""
    
    print("🧪 Testing AI Enhanced Strategy System")
    print("=" * 50)
    
    # Mock base strategy for testing
    class MockStrategy:
        def buy_condition(self, row, prev_row):
            # Simulate algorithm finding a signal
            if row.get('rsi', 50) < 30:
                return True, "[RANGING] Mean Reversion Oversold", 0.015
            return False, None, 0.0
        
        def get_strategy_info(self):
            return {'current_regime': 'ranging', 'current_mood': 'normal'}
    
    # Create AI enhanced strategy
    base_strategy = MockStrategy()
    ai_strategy = AIEnhancedStrategy(base_strategy, use_ai_validation=True)
    
    # Create mock data
    mock_row = pd.Series({
        'close': 107250.0,
        'rsi': 28.5,
        'volume_ratio': 1.8,
        'symbol': 'BTCUSDT'
    })
    
    mock_prev_row = pd.Series({'close': 107300.0})
    
    # Mock dataframe
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    mock_df = pd.DataFrame({
        'close': [107000 + i * 2.5 for i in range(100)],
        'rsi': [50 + (i % 40 - 20) for i in range(100)],
        'volume_ratio': [1.0 + (i % 10 - 5) * 0.1 for i in range(100)]
    }, index=dates)
    
    try:
        # Test AI validation
        should_buy, reason, stop_loss = await ai_strategy.buy_condition(
            mock_row, mock_prev_row, mock_df, current_idx=99
        )
        
        print(f"📊 Algorithm + AI Result:")
        print(f"   Should Buy: {should_buy}")
        print(f"   Reason: {reason}")
        print(f"   Stop Loss: {stop_loss:.3f}")
        
        # Print AI stats
        ai_strategy.print_ai_stats()
        
        print(f"\n✅ AI Enhanced Strategy test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ai_enhanced_strategy())