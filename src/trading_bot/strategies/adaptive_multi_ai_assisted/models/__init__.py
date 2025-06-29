"""
Trade Validation Agent using BaseAgent with Instructor
Professional AI-powered trade signal validation
"""

import os
import structlog
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.agents.base_agent import BaseAgent
from models.trade_validation import (
    TradeValidationResponse, 
    ValidationStats, 
    ValidationHistory,
    ValidationRequest,
    ValidationDecision
)
from context.trade_context import ValidationRequestBuilder

logger = structlog.get_logger(__name__)


class TradeValidationAgent(BaseAgent):
    """Professional trade validation agent using Claude with structured outputs"""
    
    def __init__(self):
        agent_dir = os.path.dirname(__file__)
        
        super().__init__(
            name="Trade_Validation_Expert",
            template_dir=os.path.join(agent_dir, "templates"),
            system_message_template="trade_validation_system.j2",
            system_message_kwargs={
                "expertise_level": "senior_trader",
                "focus": "risk_management_and_market_context"
            }
        )
        
        # Initialize context builder
        self.request_builder = ValidationRequestBuilder()
        
        # Validation history tracking
        self.validation_history: List[ValidationHistory] = []
        self.validation_count = 0
        
        logger.info("Trade Validation Agent initialized")
    
    async def validate_trade_signal(self,
                                   validation_request: ValidationRequest) -> TradeValidationResponse:
        """Main validation method - analyzes trade signal and returns structured decision"""
        
        self.validation_count += 1
        
        logger.info(
            "Validating trade signal",
            symbol=validation_request.symbol,
            signal=validation_request.signal_reason,
            validation_id=self.validation_count
        )
        
        try:
            # Generate structured validation response
            response = await self.generate_reply(
                prompt_template="trade_validation.j2",
                response_model=TradeValidationResponse,
                request=validation_request,
                validation_id=self.validation_count,
                timestamp=datetime.now().isoformat()
            )
            
            # Store validation history
            history_record = ValidationHistory(
                timestamp=datetime.now(),
                symbol=validation_request.symbol,
                strategy_signal=validation_request.signal_reason,
                ai_decision=response.decision,
                confidence=response.confidence
            )
            self.validation_history.append(history_record)
            
            logger.info(
                "Trade validation completed",
                decision=response.decision.value,
                confidence=response.confidence,
                validation_id=self.validation_count
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Trade validation failed",
                error=str(e),
                validation_id=self.validation_count
            )
            raise
    
    async def quick_validate(self,
                            symbol: str,
                            signal_reason: str,
                            current_price: float,
                            rsi: float,
                            volume_ratio: float,
                            regime: str = "unknown") -> TradeValidationResponse:
        """Quick validation with minimal context (for testing/simple cases)"""
        
        # Build minimal request
        from models.trade_validation import ContextSummary
        
        simple_context = ContextSummary(
            current_price=current_price,
            price_change_1h=0.0,
            price_change_24h=0.0,
            rsi_level=rsi,
            volume_vs_average=volume_ratio,
            volatility_percentile=0.5,
            detected_regime=regime,
            volatility_mood="normal",
            key_technical_levels={
                "support": current_price * 0.99,
                "resistance": current_price * 1.01,
                "recent_high": current_price * 1.02,
                "recent_low": current_price * 0.98,
                "pivot": current_price
            },
            market_hours="unknown",
            day_of_week="unknown"
        )
        
        simple_request = ValidationRequest(
            symbol=symbol,
            timeframe="5m",
            signal_reason=signal_reason,
            proposed_entry_price=current_price,
            proposed_stop_loss=current_price * 0.985,
            proposed_position_size=1000.0,
            context=simple_context,
            recent_similar_trades=[],
            strategy_confidence=0.7,
            current_portfolio_exposure=0.1
        )
        
        return await self.validate_trade_signal(simple_request)
    
    async def batch_validate(self, 
                            validation_requests: List[ValidationRequest]) -> List[TradeValidationResponse]:
        """Validate multiple trade signals (useful for backtesting)"""
        
        logger.info(f"Starting batch validation of {len(validation_requests)} signals")
        
        responses = []
        for i, request in enumerate(validation_requests):
            try:
                response = await self.validate_trade_signal(request)
                responses.append(response)
                
                # Log progress for large batches
                if len(validation_requests) > 10 and (i + 1) % 5 == 0:
                    logger.info(f"Batch validation progress: {i + 1}/{len(validation_requests)}")
                    
            except Exception as e:
                logger.error(f"Failed to validate signal {i + 1}: {e}")
                # Continue with other validations
                continue
        
        logger.info(f"Batch validation completed: {len(responses)}/{len(validation_requests)} successful")
        return responses
    
    def update_trade_outcome(self,
                           validation_timestamp: datetime,
                           actual_outcome: str,
                           actual_return: float,
                           duration_minutes: int):
        """Update validation history with actual trade outcomes for performance tracking"""
        
        # Find matching validation record
        for record in self.validation_history:
            if record.timestamp == validation_timestamp:
                record.actual_outcome = actual_outcome
                record.actual_return = actual_return
                record.trade_duration_minutes = duration_minutes
                
                # Determine if AI was right
                if actual_outcome == "WIN" and record.ai_decision == ValidationDecision.APPROVE:
                    record.ai_accuracy = True
                elif actual_outcome == "LOSS" and record.ai_decision == ValidationDecision.REJECT:
                    record.ai_accuracy = True
                else:
                    record.ai_accuracy = False
                
                logger.info(
                    "Updated validation outcome",
                    ai_decision=record.ai_decision.value,
                    actual_outcome=actual_outcome,
                    ai_correct=record.ai_accuracy
                )
                break
    
    def get_validation_stats(self, lookback_days: int = 7) -> ValidationStats:
        """Get validation performance statistics"""
        
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        
        recent_validations = [
            v for v in self.validation_history 
            if v.timestamp >= cutoff_time
        ]
        
        if not recent_validations:
            return ValidationStats(
                total_validations=0,
                approvals=0,
                rejections=0,
                conditionals=0,
                average_confidence=0.0,
                top_rejection_reasons=[],
                top_approval_factors=[]
            )
        
        # Count decisions
        approvals = len([v for v in recent_validations if v.ai_decision == ValidationDecision.APPROVE])
        rejections = len([v for v in recent_validations if v.ai_decision == ValidationDecision.REJECT])
        conditionals = len([v for v in recent_validations if v.ai_decision == ValidationDecision.CONDITIONAL])
        
        # Calculate metrics
        avg_confidence = sum(v.confidence for v in recent_validations) / len(recent_validations)
        
        # Performance metrics (if outcome data available)
        completed_trades = [v for v in recent_validations if v.actual_outcome is not None]
        approved_wins = len([
            v for v in completed_trades 
            if v.ai_decision == ValidationDecision.APPROVE and v.actual_outcome == "WIN"
        ])
        approved_total = len([v for v in completed_trades if v.ai_decision == ValidationDecision.APPROVE])
        
        return ValidationStats(
            total_validations=len(recent_validations),
            approvals=approvals,
            rejections=rejections,
            conditionals=conditionals,
            average_confidence=avg_confidence,
            approved_trades_win_rate=approved_wins / approved_total if approved_total > 0 else None,
            top_rejection_reasons=["Low volume", "Poor timing", "Weak setup"],  # Could be extracted from data
            top_approval_factors=["Strong technicals", "Good volume", "Clear trend"]
        )
    
    def get_recent_decisions(self, limit: int = 10) -> List[ValidationHistory]:
        """Get recent validation decisions for monitoring"""
        return self.validation_history[-limit:]
    
    async def analyze_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Analyze current market regime for context (bonus feature)"""
        
        response = await self.generate_reply_with_raw_prompt(
            prompt=f"""
            Analyze the current market regime based on this data:
            
            Price: ${market_data.get('price', 0):,.2f}
            24h Change: {market_data.get('change_24h', 0):.2f}%
            Volume: {market_data.get('volume_ratio', 1.0):.1f}x average
            Volatility: {market_data.get('volatility_percentile', 50):.0f}th percentile
            
            Respond with a single word: TRENDING, RANGING, or BREAKOUT
            """,
            response_model=str
        )
        
        return response