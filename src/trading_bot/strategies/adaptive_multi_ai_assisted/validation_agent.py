"""
Updated Trade Validation Agent using Structured Actions
File: src/trading_bot/strategies/adaptive_multi_ai_assisted/structured_validation_agent.py

Uses Instructor to get structured action-based responses instead of text parsing
"""

import os
import structlog
from typing import Dict, List, Any
from datetime import datetime

from trading_bot.strategies.adaptive_multi_ai_assisted.base_agent import BaseAgent
from trading_bot.strategies.adaptive_multi_ai_assisted.trade_context import ValidationRequestBuilder
from trading_bot.strategies.adaptive_multi_ai_assisted.trade_validation import ValidationRequest
from trading_bot.strategies.adaptive_multi_ai_assisted.trade_actions import (
    AITradeRecommendation, TradeActionProcessor, TradeAction,
    ActionType, StopLossModification, PositionSizeModification,
    TradeExecution, TradeRejection, WaitCondition
)

logger = structlog.get_logger(__name__)


class TradeValidationAgent(BaseAgent):
    """AI Trade Validation using structured action responses"""
    
    def __init__(self):
        agent_dir = os.path.dirname(__file__)
        
        super().__init__(
            name="Trade_Validation_Expert",
            template_dir=os.path.join(agent_dir, "templates"),
            system_message_template="trade_validation_system.j2",
            system_message_kwargs={
                "expertise_level": "senior_trader",
                "focus": "structured_action_recommendations"
            }
        )
        
        # Initialize processors
        self.request_builder = ValidationRequestBuilder()
        self.action_processor = TradeActionProcessor()
        
        # Statistics tracking
        self.validation_count = 0
        self.action_stats = {action_type: 0 for action_type in ActionType}
        self.successful_modifications = 0
        
        logger.info("Structured Trade Validation Agent initialized")
    
    async def validate_trade_with_actions(self,
                                        validation_request: ValidationRequest) -> Dict[str, Any]:
        """
        Validate trade and get structured action recommendations
        
        Returns:
            Complete trade recommendation with applied modifications
        """
        
        self.validation_count += 1
        
        logger.info(
            "Validating trade signal with structured actions",
            symbol=validation_request.symbol,
            signal=validation_request.signal_reason,
            validation_id=self.validation_count
        )
        
        try:
            # Get structured AI recommendation
            ai_recommendation = await self.generate_reply(
                prompt_template="trade_validation.j2",
                response_model=AITradeRecommendation,
                request=validation_request,
                validation_id=self.validation_count,
                timestamp=datetime.now().isoformat()
            )
            
            # Process and apply the AI actions
            modified_params = self.action_processor.apply_recommendation(
                recommendation=ai_recommendation,
                original_entry_price=validation_request.proposed_entry_price,
                original_stop_loss_pct=validation_request.proposed_stop_loss,
                original_position_size=validation_request.proposed_position_size
            )
            
            # Update statistics
            self._update_action_stats(ai_recommendation.recommended_actions)
            
            # Combine everything into final result
            result = {
                'ai_recommendation': ai_recommendation,
                'modified_params': modified_params,
                'validation_id': self.validation_count,
                'processing_successful': True,
                'original_params': {
                    'entry_price': validation_request.proposed_entry_price,
                    'stop_loss_pct': validation_request.proposed_stop_loss, 
                    'position_size': validation_request.proposed_position_size
                }
            }
            
            logger.info(
                "Structured validation completed",
                decision=ai_recommendation.decision,
                confidence=ai_recommendation.confidence,
                actions_count=len(ai_recommendation.recommended_actions),
                should_execute=modified_params['should_execute'],
                validation_id=self.validation_count
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Structured validation failed",
                error=str(e),
                validation_id=self.validation_count
            )
            
            # Return safe default
            return {
                'ai_recommendation': self._create_rejection_recommendation(str(e)),
                'modified_params': {'should_execute': False, 'error': str(e)},
                'validation_id': self.validation_count,
                'processing_successful': False
            }
    
    async def quick_validate_with_actions(self,
                                        symbol: str,
                                        signal_reason: str,
                                        current_price: float,
                                        rsi: float,
                                        volume_ratio: float,
                                        regime: str = "unknown") -> Dict[str, Any]:
        """Quick validation with structured actions"""
        
        # Create validation request (same as before)
        validation_request = self._create_quick_validation_request(
            symbol, signal_reason, current_price, rsi, volume_ratio, regime
        )
        
        return await self.validate_trade_with_actions(validation_request)
    
    def _create_quick_validation_request(self, symbol, signal_reason, current_price, rsi, volume_ratio, regime):
        """Create ValidationRequest for quick testing (helper method)"""
        # Implementation same as before...
        from .trade_validation import ContextSummary, CandleData
        
        # Create dummy recent candles
        recent_candles = []
        for i in range(5):
            price_var = current_price * (1 + (i - 2) * 0.001)
            candle = CandleData(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                open=price_var * 0.999,
                high=price_var * 1.001,
                low=price_var * 0.998,
                close=price_var,
                volume=1000.0 * volume_ratio,
                volume_ratio=volume_ratio,
                rsi=rsi,
                change_pct=0.1 * (i - 2)
            )
            recent_candles.append(candle)
        
        context = ContextSummary(
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
        
        from .trade_validation import ValidationRequest
        return ValidationRequest(
            symbol=symbol,
            timeframe="5m",
            signal_reason=signal_reason,
            proposed_entry_price=current_price,
            proposed_stop_loss=0.006,  # 0.6%
            proposed_position_size=1000.0,
            context=context,
            recent_candles=recent_candles,
            recent_similar_trades=[],
            strategy_confidence=0.7,
            current_portfolio_exposure=0.1
        )
    
    def _update_action_stats(self, actions: List[TradeAction]):
        """Update statistics for recommended actions"""
        for action in actions:
            action_type = ActionType(action.action_type)
            self.action_stats[action_type] += 1
        
        self.successful_modifications += 1
    
    def _create_rejection_recommendation(self, error_msg: str) -> AITradeRecommendation:
        """Create rejection recommendation for errors"""
        return AITradeRecommendation(
            decision="REJECT",
            confidence=1,
            risk_level="HIGH",
            market_sentiment="UNCERTAIN",
            recommended_actions=[
                TradeRejection(
                    primary_concern="market_conditions",
                    reasoning=f"Validation failed: {error_msg}"
                )
            ],
            key_factors=["Validation error", "System failure", "Safety measure"],
            risk_factors=["Technical failure", "Uncertain conditions", "Risk management"],
            timing_assessment="Cannot assess timing due to validation failure"
        )
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics on AI actions recommended"""
        total_actions = sum(self.action_stats.values())
        
        return {
            'total_validations': self.validation_count,
            'total_actions_recommended': total_actions,
            'successful_modifications': self.successful_modifications,
            'action_breakdown': {
                action_type.value: {
                    'count': count,
                    'percentage': (count / total_actions * 100) if total_actions > 0 else 0
                }
                for action_type, count in self.action_stats.items()
            },
            'most_common_action': max(self.action_stats.items(), key=lambda x: x[1])[0].value if total_actions > 0 else None
        }
    
    def simulate_action_application(self, result: Dict[str, Any]) -> str:
        """Simulate applying the AI actions and return description"""
        
        if not result['processing_successful']:
            return "❌ Processing failed - trade rejected"
        
        modified_params = result['modified_params']
        ai_rec = result['ai_recommendation']
        
        if not modified_params['should_execute']:
            return f"🚫 Trade REJECTED: {ai_rec.recommended_actions[0].reasoning if ai_rec.recommended_actions else 'Unknown reason'}"
        
        description_parts = []
        description_parts.append(f"✅ Trade APPROVED with {len(ai_rec.recommended_actions)} modifications:")
        
        for i, action in enumerate(ai_rec.recommended_actions, 1):
            if isinstance(action, StopLossModification):
                new_sl = modified_params.get('stop_loss_pct', 0) * 100
                description_parts.append(f"   {i}. Stop Loss → {new_sl:.2f}%")
            
            elif isinstance(action, PositionSizeModification):
                new_size = modified_params.get('position_size', 0)
                description_parts.append(f"   {i}. Position Size → ${new_size:.2f}")
            
            elif isinstance(action, WaitCondition):
                description_parts.append(f"   {i}. Wait for: {action.condition_type}")
            
            else:
                description_parts.append(f"   {i}. {action.action_type.value}")
        
        return "\n".join(description_parts)


# Example usage function
async def test_structured_validation():
    """Test the structured validation system"""
    
    agent = StructuredTradeValidationAgent()
    
    # Test validation
    result = await agent.quick_validate_with_actions(
        symbol="SOLUSDT",
        signal_reason="[RANGING] Mean Reversion Oversold",
        current_price=149.65,
        rsi=33.2,
        volume_ratio=0.9,
        regime="ranging"
    )
    
    # Show results
    print("🧠 AI Structured Recommendation:")
    ai_rec = result['ai_recommendation']
    print(f"   Decision: {ai_rec.decision}")
    print(f"   Confidence: {ai_rec.confidence}/10")
    print(f"   Risk Level: {ai_rec.risk_level}")
    print(f"   Actions: {len(ai_rec.recommended_actions)}")
    
    for i, action in enumerate(ai_rec.recommended_actions, 1):
        print(f"      {i}. {action.action_type}: {action.reasoning}")
    
    print("\n📊 Applied Modifications:")
    print(agent.simulate_action_application(result))
    
    print("\n📈 Action Statistics:")
    stats = agent.get_action_statistics()
    for action_type, data in stats['action_breakdown'].items():
        if data['count'] > 0:
            print(f"   {action_type}: {data['count']} times ({data['percentage']:.1f}%)")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_structured_validation())