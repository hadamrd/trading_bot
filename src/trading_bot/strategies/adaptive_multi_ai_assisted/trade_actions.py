"""
Trade Actions - Fixed Pydantic Issues
File: src/trading_bot/strategies/adaptive_multi_ai_assisted/trade_actions.py

Simpler action model that's easier for AI to generate correctly
"""

from typing import List, Optional, Union, Literal, Any, Dict
from enum import Enum
from pydantic import BaseModel, Field, validator


class ActionType(str, Enum):
    """Available action types the AI can recommend"""
    EXECUTE_TRADE = "execute_trade"
    REJECT_TRADE = "reject_trade"
    MODIFY_STOP_LOSS = "modify_stop_loss" 
    MODIFY_POSITION_SIZE = "modify_position_size"
    MODIFY_ENTRY_PRICE = "modify_entry_price"
    WAIT_FOR_CONDITION = "wait_for_condition"
    USE_LIMIT_ORDER = "use_limit_order"
    SCALE_OUT_TRADE = "scale_out_trade"
    SET_TAKE_PROFIT = "set_take_profit"


class TradeAction(BaseModel):
    """Simplified unified action model"""
    
    action_type: ActionType = Field(description="Type of action to take")
    
    # Stop loss modifications
    new_stop_loss_price: Optional[float] = Field(None, description="New absolute stop loss price")
    new_stop_loss_percentage: Optional[float] = Field(None, ge=0.001, le=0.1, description="New stop loss percentage from entry")
    stop_loss_multiplier: Optional[float] = Field(None, ge=0.5, le=3.0, description="Multiply current stop loss by this factor")
    
    # Position size modifications  
    size_multiplier: Optional[float] = Field(None, ge=0.1, le=2.0, description="Multiply position size by this factor")
    size_percentage: Optional[float] = Field(None, ge=0.1, le=2.0, description="Set position to this percentage of original")
    absolute_size_dollars: Optional[float] = Field(None, gt=0, description="Set absolute position size in dollars")
    
    # Entry price modifications
    new_entry_price: Optional[float] = Field(None, description="New entry price for limit orders")
    use_limit_order: Optional[bool] = Field(None, description="Whether to use limit order")
    
    # Wait conditions
    condition_type: Optional[Literal["rsi_level", "price_level", "volume_spike", "reversal_candle", "time_delay"]] = Field(None)
    rsi_threshold: Optional[float] = Field(None, ge=0, le=100, description="RSI level to wait for")
    price_threshold: Optional[float] = Field(None, description="Price level to wait for")
    volume_multiplier: Optional[float] = Field(None, ge=1.0, description="Volume multiplier to wait for")
    wait_minutes: Optional[int] = Field(None, ge=1, le=60, description="Minutes to wait")
    
    # Take profit
    take_profit_price: Optional[float] = Field(None, description="Take profit price level")
    take_profit_percentage: Optional[float] = Field(None, ge=0.005, le=0.5, description="Take profit percentage")
    
    # Scale out
    scale_levels: Optional[List[Dict[str, Any]]] = Field(None, description="Scale out levels with price and percentage")
    
    # Execution parameters
    confidence_level: Optional[int] = Field(None, ge=1, le=10, description="Confidence in trade execution")
    primary_concern: Optional[Literal["poor_timing", "weak_setup", "high_risk", "low_volume", "market_conditions"]] = Field(None)
    
    # Always required
    reasoning: str = Field(description="Detailed reasoning for this action")
    
    @validator('reasoning')
    def validate_action_parameters(cls, v, values):
        """Ensure required parameters are provided for each action type"""
        action_type = values.get('action_type')
        
        if action_type == ActionType.MODIFY_STOP_LOSS:
            has_stop_param = any([
                values.get('new_stop_loss_price'),
                values.get('new_stop_loss_percentage'), 
                values.get('stop_loss_multiplier')
            ])
            if not has_stop_param:
                raise ValueError("Stop loss modification requires at least one: new_stop_loss_price, new_stop_loss_percentage, or stop_loss_multiplier")
        
        elif action_type == ActionType.MODIFY_POSITION_SIZE:
            has_size_param = any([
                values.get('size_multiplier'),
                values.get('size_percentage'),
                values.get('absolute_size_dollars')
            ])
            if not has_size_param:
                raise ValueError("Position size modification requires at least one: size_multiplier, size_percentage, or absolute_size_dollars")
        
        elif action_type == ActionType.MODIFY_ENTRY_PRICE:
            if not values.get('new_entry_price'):
                raise ValueError("Entry price modification requires new_entry_price")
        
        elif action_type == ActionType.WAIT_FOR_CONDITION:
            if not values.get('condition_type'):
                raise ValueError("Wait condition requires condition_type")
        
        elif action_type == ActionType.EXECUTE_TRADE:
            if not values.get('confidence_level'):
                raise ValueError("Trade execution requires confidence_level")
        
        elif action_type == ActionType.REJECT_TRADE:
            if not values.get('primary_concern'):
                raise ValueError("Trade rejection requires primary_concern")
        
        return v


class AITradeRecommendation(BaseModel):
    """Complete AI recommendation with simplified actions"""
    
    # Overall assessment
    decision: Literal["APPROVE", "REJECT", "CONDITIONAL"] = Field(
        description="Overall trade decision"
    )
    
    confidence: int = Field(
        ge=1, le=10, 
        description="Overall confidence in the analysis (1-10)"
    )
    
    # Risk assessment
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "EXTREME"] = Field(
        description="Assessed risk level of this trade"
    )
    
    # Market context
    market_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL", "UNCERTAIN"] = Field(
        description="Current market sentiment assessment"
    )
    
    # Simplified actions list
    recommended_actions: List[TradeAction] = Field(
        min_items=1,
        max_items=5,
        description="Specific actions to implement this recommendation"
    )
    
    # Key insights
    key_factors: List[str] = Field(
        max_items=3,
        description="3 most important factors influencing this decision"
    )
    
    risk_factors: List[str] = Field(
        max_items=3, 
        description="3 main risk factors identified"
    )
    
    # Timing assessment
    timing_assessment: str = Field(
        max_length=150,
        description="Brief assessment of trade timing"
    )


class TradeActionProcessor:
    """Process trade actions"""
    
    def __init__(self):
        self.actions_applied = []
        self.modifications_log = []
    
    def apply_recommendation(self, 
                           recommendation: AITradeRecommendation,
                           original_entry_price: float,
                           original_stop_loss_pct: float,
                           original_position_size: float) -> dict:
        """Apply AI recommendation actions to trade parameters"""
        
        modified_params = {
            'entry_price': original_entry_price,
            'stop_loss_pct': original_stop_loss_pct,
            'position_size': original_position_size,
            'use_limit_order': False,
            'take_profit_levels': [],
            'scale_out_levels': [],
            'wait_conditions': [],
            'should_execute': recommendation.decision != "REJECT",
            'modifications_applied': [],
            'ai_confidence': recommendation.confidence,
            'risk_level': recommendation.risk_level
        }
        
        # Apply each recommended action
        for action in recommendation.recommended_actions:
            self._apply_single_action(action, modified_params, original_entry_price)
        
        return modified_params
    
    def _apply_single_action(self, action: TradeAction, params: dict, original_entry: float):
        """Apply a single trade action"""
        
        if action.action_type == ActionType.REJECT_TRADE:
            params['should_execute'] = False
            params['modifications_applied'].append(f"REJECTED: {action.primary_concern}")
        
        elif action.action_type == ActionType.MODIFY_STOP_LOSS:
            if action.new_stop_loss_price:
                params['stop_loss_pct'] = abs(action.new_stop_loss_price - original_entry) / original_entry
                params['modifications_applied'].append(f"STOP_LOSS_PRICE: ${action.new_stop_loss_price:.2f}")
            elif action.new_stop_loss_percentage:
                params['stop_loss_pct'] = action.new_stop_loss_percentage
                params['modifications_applied'].append(f"STOP_LOSS_PCT: {action.new_stop_loss_percentage:.3f}")
            elif action.stop_loss_multiplier:
                params['stop_loss_pct'] *= action.stop_loss_multiplier
                params['modifications_applied'].append(f"STOP_LOSS_MULT: {action.stop_loss_multiplier:.2f}")
        
        elif action.action_type == ActionType.MODIFY_POSITION_SIZE:
            if action.size_multiplier:
                params['position_size'] *= action.size_multiplier
                params['modifications_applied'].append(f"SIZE_MULT: {action.size_multiplier:.2f}")
            elif action.size_percentage:
                params['position_size'] *= action.size_percentage
                params['modifications_applied'].append(f"SIZE_PCT: {action.size_percentage:.2f}")
            elif action.absolute_size_dollars:
                params['position_size'] = action.absolute_size_dollars
                params['modifications_applied'].append(f"SIZE_ABS: ${action.absolute_size_dollars:.2f}")
        
        elif action.action_type == ActionType.MODIFY_ENTRY_PRICE:
            params['entry_price'] = action.new_entry_price
            if action.use_limit_order:
                params['use_limit_order'] = True
            params['modifications_applied'].append(f"ENTRY_PRICE: ${action.new_entry_price:.2f}")
        
        elif action.action_type == ActionType.WAIT_FOR_CONDITION:
            wait_condition = {
                'type': action.condition_type,
                'rsi_threshold': action.rsi_threshold,
                'price_threshold': action.price_threshold,
                'volume_multiplier': action.volume_multiplier,
                'wait_minutes': action.wait_minutes
            }
            params['wait_conditions'].append(wait_condition)
            params['modifications_applied'].append(f"WAIT: {action.condition_type}")
        
        elif action.action_type == ActionType.SET_TAKE_PROFIT:
            if action.take_profit_price:
                params['take_profit_levels'].append(action.take_profit_price)
                params['modifications_applied'].append(f"TP_PRICE: ${action.take_profit_price:.2f}")
            elif action.take_profit_percentage:
                profit_price = original_entry * (1 + action.take_profit_percentage)
                params['take_profit_levels'].append(profit_price)
                params['modifications_applied'].append(f"TP_PCT: {action.take_profit_percentage:.3f}")
        
        elif action.action_type == ActionType.SCALE_OUT_TRADE:
            if action.scale_levels:
                params['scale_out_levels'] = action.scale_levels
                params['modifications_applied'].append(f"SCALE_OUT: {len(action.scale_levels)} levels")
        
        elif action.action_type == ActionType.EXECUTE_TRADE:
            params['modifications_applied'].append(f"EXECUTE: confidence_{action.confidence_level}")
