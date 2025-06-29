#!/usr/bin/env python3
"""
Simple Claude API Test
Tests if Claude API is working with your setup
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import anthropic
    print("✅ anthropic package installed")
except ImportError:
    print("❌ anthropic package not installed")
    print("💡 Install with: pip install anthropic")
    sys.exit(1)

try:
    import instructor
    print("✅ instructor package installed")
except ImportError:
    print("❌ instructor package not installed") 
    print("💡 Install with: pip install instructor")
    sys.exit(1)

from pydantic import BaseModel


class SimpleResponse(BaseModel):
    """Simple response model for testing"""
    decision: str
    reasoning: str
    confidence: int


async def test_claude_api():
    """Test basic Claude API functionality"""
    
    print("\n🧪 Testing Claude API Connection")
    print("=" * 40)
    
    # Check for API key
    api_key = os.getenv("CLAUDE_API_KEY")
    
    if not api_key:
        print("❌ CLAUDE_API_KEY not found in environment")
        print("💡 Set it with: export CLAUDE_API_KEY=sk-ant-...")
        print("💡 Or add to .env file: CLAUDE_API_KEY=sk-ant-...")
        return False
    
    if not api_key.startswith("sk-ant-"):
        print("❌ Invalid Claude API key format")
        print("💡 Should start with 'sk-ant-'")
        return False
    
    print(f"✅ API key found: {api_key[:12]}...")
    
    try:
        # Test basic Anthropic client
        print("\n🔗 Testing basic Anthropic connection...")
        
        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=1000)
        
        # Simple message test
        response = await client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=50,
            messages=[{"role": "user", "content": "Hello! Just respond with 'API working'"}]
        )
        
        print(f"✅ Basic API call successful")
        print(f"📝 Response: {response.content[0].text}")
        
    except Exception as e:
        print(f"❌ Basic API call failed: {e}")
        return False
    
    try:
        # Test Instructor integration
        print("\n🎯 Testing Instructor integration...")
        
        instructor_client = instructor.from_anthropic(
            client,
            mode=instructor.Mode.ANTHROPIC_TOOLS
        )
        
        # Structured response test
        structured_response = await instructor_client.chat.completions.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=200,
            messages=[{
                "role": "user", 
                "content": "Analyze this trade: BUY BTCUSDT at $50000, RSI=25. Should I approve this trade?"
            }],
            response_model=SimpleResponse
        )
        
        print(f"✅ Structured response successful")
        print(f"📝 Decision: {structured_response.decision}")
        print(f"📝 Reasoning: {structured_response.reasoning}")
        print(f"📝 Confidence: {structured_response.confidence}")
        
    except Exception as e:
        print(f"❌ Instructor integration failed: {e}")
        return False
    
    print(f"\n🎉 All tests passed! Claude API is working correctly.")
    return True


async def test_trade_validation_prompt():
    """Test a simple trade validation prompt"""
    
    print("\n📊 Testing Trade Validation Prompt")
    print("=" * 40)
    
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        print("❌ Need API key for this test")
        return
    
    try:
        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=1000)
        instructor_client = instructor.from_anthropic(client, mode=instructor.Mode.ANTHROPIC_TOOLS)
        
        # Trade validation prompt
        prompt = """
        You are a senior crypto trader. Analyze this trade signal:
        
        SIGNAL: Mean Reversion Oversold
        Symbol: BTCUSDT
        Price: $107,250
        RSI: 28.5 (oversold)
        Volume: 1.8x average
        Recent: Down 2% in last hour
        
        Should I take this trade? Respond with decision (APPROVE/REJECT/CONDITIONAL), 
        reasoning (1 sentence), and confidence (1-10).
        """
        
        response = await instructor_client.chat.completions.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
            response_model=SimpleResponse
        )
        
        print(f"✅ Trade validation test successful")
        print(f"🎯 Decision: {response.decision}")
        print(f"🧠 Reasoning: {response.reasoning}")
        print(f"📊 Confidence: {response.confidence}/10")
        
        # This is what we expect to see working!
        return True
        
    except Exception as e:
        print(f"❌ Trade validation test failed: {e}")
        return False


async def main():
    """Run all tests"""
    
    print("🚀 Claude API Testing Suite")
    print("=" * 50)
    
    # Test 1: Basic API connection
    basic_works = await test_claude_api()
    
    if basic_works:
        # Test 2: Trade validation example
        await test_trade_validation_prompt()
        
        print(f"\n✅ SUCCESS! Your Claude API is ready for trade validation.")
        print(f"💡 You can now use the AI-enhanced trading strategy.")
        
    else:
        print(f"\n❌ Setup incomplete. Fix the issues above and try again.")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Get Claude API key from: https://console.anthropic.com/")
        print(f"   2. Set environment: export CLAUDE_API_KEY=sk-ant-...")
        print(f"   3. Or add to .env file: CLAUDE_API_KEY=sk-ant-...")
        print(f"   4. Install packages: pip install anthropic instructor")


if __name__ == "__main__":
    asyncio.run(main())