#!/usr/bin/env python3
"""
Test AI Trade Validation with Structured Actions
Shows how to use the new structured AI validation system
"""
from trading_bot.strategies.adaptive_multi_ai_assisted.validation_agent import TradeValidationAgent
from trading_bot.strategies.adaptive_multi_ai_assisted.trade_validation import ValidationRequest, ContextSummary, CandleData


class AIEnhancedStrategy:
    """AI-Enhanced Strategy using structured actions"""
    
    def __init__(self, base_strategy, ai_confidence_threshold=6):
        self.base_strategy = base_strategy
        self.ai_agent = TradeValidationAgent()
        self.ai_confidence_threshold = ai_confidence_threshold
        
        # Statistics tracking
        self.ai_stats = {
            'total_validations': 0,
            'approved': 0,
            'rejected': 0,
            'conditional': 0,
            'errors': 0,
            'avg_confidence': 0.0,
            'modifications_applied': []
        }
        
        print(f"🧠 AI Enhanced Strategy initialized with structured actions")
        print(f"   AI confidence threshold: {ai_confidence_threshold}")
    
    async def buy_condition_with_ai(self, row, prev_row, df, current_index):
        """Enhanced buy condition with structured AI validation"""
        
        # First check if base strategy generates a signal
        base_signal = self.base_strategy.buy_condition(row, prev_row)
        
        if not base_signal[0]:  # No base signal
            return False, "No algorithm signal", 0.0
        
        # Base strategy found a signal - validate with AI
        algorithm_reason = base_signal[1]
        stop_loss_pct = base_signal[2]
        
        print(f"🎯 Algorithm signal detected: {algorithm_reason}")
        print(f"   Entry: ${row['close']:.2f}, Stop: {stop_loss_pct:.3f}")
        
        try:
            # Create validation request
            validation_request = self._create_validation_request(
                row, df, current_index, algorithm_reason, stop_loss_pct
            )
            
            # Get structured AI recommendation
            result = await self.ai_agent.validate_trade_with_actions(validation_request)
            
            self.ai_stats['total_validations'] += 1
            
            if not result['processing_successful']:
                self.ai_stats['errors'] += 1
                return False, f"AI-ERROR: {algorithm_reason}", 0.0
            
            ai_rec = result['ai_recommendation']
            modified_params = result['modified_params']
            
            # Update statistics
            self.ai_stats['avg_confidence'] = (
                (self.ai_stats['avg_confidence'] * (self.ai_stats['total_validations'] - 1) + ai_rec.confidence) 
                / self.ai_stats['total_validations']
            )
            
            # Apply AI decision
            if ai_rec.decision == "REJECT":
                self.ai_stats['rejected'] += 1
                return False, f"AI-REJECTED: {algorithm_reason}", 0.0
            
            elif ai_rec.decision == "CONDITIONAL":
                self.ai_stats['conditional'] += 1
                
                # Apply modifications from structured actions
                modifications_desc = self._describe_modifications(result)
                self.ai_stats['modifications_applied'].extend(modified_params.get('modifications_applied', []))
                
                # Check if AI confidence meets our threshold
                if ai_rec.confidence >= self.ai_confidence_threshold:
                    # Use modified parameters
                    modified_stop = modified_params.get('stop_loss_pct', stop_loss_pct)
                    return True, f"AI-CONDITIONAL({ai_rec.confidence}/10): {algorithm_reason} | MOD: {modifications_desc}", modified_stop
                else:
                    return False, f"AI-LOW_CONFIDENCE({ai_rec.confidence}/10): {algorithm_reason}", 0.0
            
            else:  # APPROVE
                self.ai_stats['approved'] += 1
                return True, f"AI-APPROVED({ai_rec.confidence}/10): {algorithm_reason}", stop_loss_pct
                
        except Exception as e:
            self.ai_stats['errors'] += 1
            print(f"❌ AI validation error: {e}")
            return False, f"AI-ERROR: {algorithm_reason}", 0.0
    
    def _create_validation_request(self, row, df, current_index, signal_reason, stop_loss_pct):
        """Create validation request from current market data"""
        
        # Get recent candles
        recent_candles = []
        for i in range(max(0, current_index-4), current_index+1):
            if i < len(df):
                candle_row = df.iloc[i]
                candle = CandleData(
                    timestamp=candle_row.name.strftime("%Y-%m-%d %H:%M:%S"),
                    open=float(candle_row['open']),
                    high=float(candle_row['high']),
                    low=float(candle_row['low']),
                    close=float(candle_row['close']),
                    volume=float(candle_row['volume']),
                    volume_ratio=float(candle_row.get('volume_ratio', 1.0)),
                    rsi=float(candle_row.get('rsi', 50)),
                    change_pct=float(candle_row['close'] / candle_row['open'] - 1) * 100
                )
                recent_candles.append(candle)
        
        # Create context
        context = ContextSummary(
            current_price=float(row['close']),
            price_change_1h=0.0,  # Could calculate from recent candles
            price_change_24h=0.0, # Could calculate from recent candles
            rsi_level=float(row.get('rsi', 50)),
            volume_vs_average=float(row.get('volume_ratio', 1.0)),
            volatility_percentile=0.5,  # Could calculate
            detected_regime=getattr(self.base_strategy, 'current_regime', 'unknown').value if hasattr(getattr(self.base_strategy, 'current_regime', None), 'value') else str(getattr(self.base_strategy, 'current_regime', 'unknown')),
            volatility_mood=getattr(self.base_strategy, 'current_mood', 'normal').value if hasattr(getattr(self.base_strategy, 'current_mood', None), 'value') else str(getattr(self.base_strategy, 'current_mood', 'normal')),
            key_technical_levels={
                "support": float(row['close']) * 0.99,
                "resistance": float(row['close']) * 1.01,
                "recent_high": float(row['close']) * 1.02,
                "recent_low": float(row['close']) * 0.98,
                "pivot": float(row['close'])
            },
            market_hours="unknown",
            day_of_week=row.name.strftime("%A") if hasattr(row.name, 'strftime') else "unknown"
        )
        
        # Calculate position size
        available_balance = 10000.0  # Mock balance
        position_size = self.base_strategy.calculate_position_size(
            available_balance, float(row['close']), row
        )
        
        return ValidationRequest(
            symbol="SOLUSDT",  # You can parameterize this
            timeframe="5m",
            signal_reason=signal_reason,
            proposed_entry_price=float(row['close']),
            proposed_stop_loss=stop_loss_pct,
            proposed_position_size=position_size,
            context=context,
            recent_candles=recent_candles,
            recent_similar_trades=[],
            strategy_confidence=0.7,
            current_portfolio_exposure=0.1
        )
    
    def _describe_modifications(self, result):
        """Create human-readable description of AI modifications"""
        modified_params = result['modified_params']
        modifications = modified_params.get('modifications_applied', [])
        
        if not modifications:
            return "No modifications"
        
        # Create readable description
        descriptions = []
        for mod in modifications:
            if "STOP_LOSS_MODIFIED" in mod:
                descriptions.append(f"Stop adjusted to {mod.split(':')[1].strip()}")
            elif "POSITION_SIZE_MODIFIED" in mod:
                descriptions.append(f"Size adjusted to {mod.split(':')[1].strip()}")
            elif "WAIT_CONDITION" in mod:
                descriptions.append(f"Wait for {mod.split(':')[1].strip()}")
            else:
                descriptions.append(mod.replace('_', ' ').title())
        
        return ", ".join(descriptions[:2])  # Show max 2 modifications
    
    def get_action_stats(self):
        """Get AI action statistics"""
        return self.ai_agent.get_action_statistics()
    
    def print_ai_stats(self):
        """Print AI validation statistics"""
        print(f"\n🤖 AI VALIDATION STATISTICS")
        print("=" * 40)
        
        total = self.ai_stats['total_validations']
        if total == 0:
            print("No algorithm signals processed yet")
            return
        
        print(f"📊 Total Algorithm Signals: {total}")
        print(f"✅ AI Approved: {self.ai_stats['approved']} ({self.ai_stats['approved']/total*100:.1f}%)")
        print(f"❌ AI Rejected: {self.ai_stats['rejected']} ({self.ai_stats['rejected']/total*100:.1f}%)")
        print(f"⚠️  AI Conditional: {self.ai_stats['conditional']} ({self.ai_stats['conditional']/total*100:.1f}%)")
        print(f"⏰ Timeouts: 0")
        print(f"🚫 Errors: {self.ai_stats['errors']}")
        print(f"📈 Avg AI Confidence: {self.ai_stats['avg_confidence']:.1f}/10")
        
        # Show action statistics
        action_stats = self.get_action_stats()
        if action_stats['total_actions_recommended'] > 0:
            print(f"\n🎯 AI ACTIONS RECOMMENDED:")
            for action_type, data in action_stats['action_breakdown'].items():
                if data['count'] > 0:
                    print(f"   {action_type.replace('_', ' ').title()}: {data['count']} times ({data['percentage']:.1f}%)")
