import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import { TrendingUp, TrendingDown, Activity, DollarSign } from 'lucide-react';
import './App.css';

// Price Display Component
const PriceDisplay = ({ symbol, price, change24h, isConnected }) => {
  return (
    <div className="bg-gray-800 rounded-lg p-4 mb-6">
      <div className="flex items-center justify-center gap-3 mb-2">
        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
        <span className="text-yellow-400 font-bold text-xl">{symbol}</span>
      </div>
      <div className="text-center">
        <div className="text-3xl font-bold text-white mb-2">
          ${price?.toFixed(4) || 'Loading...'}
        </div>
        {change24h !== null && (
          <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
            change24h >= 0 ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
          }`}>
            {change24h >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
            {change24h >= 0 ? '+' : ''}{change24h.toFixed(2)}%
          </div>
        )}
      </div>
    </div>
  );
};

// Side Selection Component
const SideSelector = ({ selectedSide, onSideChange }) => {
  return (
    <div className="grid grid-cols-2 gap-4 mb-6">
      <button
        onClick={() => onSideChange('Buy')}
        className={`p-4 rounded-lg font-bold text-lg uppercase transition-all border-2 ${
          selectedSide === 'Buy'
            ? 'bg-green-600 border-green-400 text-white scale-95'
            : 'bg-green-900 border-green-600 text-green-300 hover:bg-green-800'
        }`}
      >
        🚀 Long
      </button>
      <button
        onClick={() => onSideChange('Sell')}
        className={`p-4 rounded-lg font-bold text-lg uppercase transition-all border-2 ${
          selectedSide === 'Sell'
            ? 'bg-red-600 border-red-400 text-white scale-95'
            : 'bg-red-900 border-red-600 text-red-300 hover:bg-red-800'
        }`}
      >
        📉 Short
      </button>
    </div>
  );
};

// Risk Calculator Component
const RiskCalculator = ({ price, selectedSide, leverage, slPercent, tpPercent, riskPercent, balance }) => {
  const calculateRisk = () => {
    if (!price || !selectedSide) return null;

    let slPrice, tpPrice;
    if (selectedSide === 'Buy') {
      slPrice = price * (1 - slPercent / 100);
      tpPrice = price * (1 + tpPercent / 100);
    } else {
      slPrice = price * (1 + slPercent / 100);
      tpPrice = price * (1 - tpPercent / 100);
    }

    const riskDistance = Math.abs(price - slPrice);
    const rewardDistance = Math.abs(tpPrice - price);
    const rrRatio = rewardDistance / riskDistance;

    const riskAmount = balance * (riskPercent / 100);
    const priceChange = Math.abs(price - slPrice) / price;
    const positionValue = riskAmount / priceChange;
    const positionSize = positionValue / price;

    return {
      rrRatio,
      riskAmount,
      positionSize,
      positionValue,
      slPrice,
      tpPrice
    };
  };

  const risk = calculateRisk();

  if (!risk) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 mb-6">
        <div className="text-center text-gray-400">
          Select a side to calculate risk
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 mb-6 border-l-4 border-cyan-400">
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-400">Risk/Reward:</span>
          <span className={`font-bold ${risk.rrRatio >= 2 ? 'text-green-400' : 'text-red-400'}`}>
            1:{risk.rrRatio.toFixed(2)}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Risk Amount:</span>
          <span className="text-white font-bold">${risk.riskAmount.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Position Size:</span>
          <span className="text-white font-bold">{risk.positionSize.toFixed(6)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Position Value:</span>
          <span className="text-white font-bold">${risk.positionValue.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
};

// Trading Parameters Component
const TradingParams = ({ leverage, slPercent, tpPercent, riskPercent, onChange }) => {
  return (
    <div className="space-y-4 mb-6">
      {/* Leverage Slider */}
      <div>
        <label className="block text-gray-300 text-sm font-bold mb-2 uppercase">
          Leverage: <span className="text-yellow-400 text-lg">{leverage}x</span>
        </label>
        <input
          type="range"
          min="1"
          max="50"
          value={leverage}
          onChange={(e) => onChange('leverage', parseInt(e.target.value))}
          className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
        />
      </div>
      
      {/* Parameters Grid */}
      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="block text-gray-300 text-xs font-bold mb-1 uppercase">Stop Loss %</label>
          <input
            type="number"
            value={slPercent}
            onChange={(e) => onChange('slPercent', parseFloat(e.target.value))}
            step="0.1"
            className="w-full p-2 bg-gray-700 border border-gray-600 rounded text-center text-white focus:border-cyan-400 focus:outline-none"
          />
        </div>
        <div>
          <label className="block text-gray-300 text-xs font-bold mb-1 uppercase">Take Profit %</label>
          <input
            type="number"
            value={tpPercent}
            onChange={(e) => onChange('tpPercent', parseFloat(e.target.value))}
            step="0.1"
            className="w-full p-2 bg-gray-700 border border-gray-600 rounded text-center text-white focus:border-cyan-400 focus:outline-none"
          />
        </div>
        <div>
          <label className="block text-gray-300 text-xs font-bold mb-1 uppercase">Risk %</label>
          <input
            type="number"
            value={riskPercent}
            onChange={(e) => onChange('riskPercent', parseFloat(e.target.value))}
            step="0.1"
            min="0.1"
            max="10"
            className="w-full p-2 bg-gray-700 border border-gray-600 rounded text-center text-white focus:border-cyan-400 focus:outline-none"
          />
        </div>
      </div>
    </div>
  );
};

// Main Trading App
function App() {
  // State
  const [symbol, setSymbol] = useState('APTUSDT');
  const [price, setPrice] = useState(null);
  const [change24h, setChange24h] = useState(null);
  const [selectedSide, setSelectedSide] = useState('Sell');
  const [leverage, setLeverage] = useState(10);
  const [slPercent, setSlPercent] = useState(2);
  const [tpPercent, setTpPercent] = useState(4);
  const [riskPercent, setRiskPercent] = useState(2);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [balance, setBalance] = useState(5000);
  const [socket, setSocket] = useState(null);

  // WebSocket connection
  useEffect(() => {
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Connected to price feed');
      setIsConnected(true);
      // Subscribe to initial ticker
      newSocket.emit('subscribe_ticker', { symbol: symbol });
    });

    newSocket.on('price_update', (data) => {
      if (data.symbol === symbol) {
        setPrice(data.price);
        setChange24h(data.change_24h);
      }
    });

    newSocket.on('subscribed', (data) => {
      console.log('Subscribed to', data.symbol);
    });

    newSocket.on('disconnect', () => {
      setIsConnected(false);
    });

    return () => newSocket.close();
  }, []);

  // Subscribe to new symbol when changed
  useEffect(() => {
    if (socket && isConnected) {
      socket.emit('subscribe_ticker', { symbol: symbol });
    }
  }, [symbol, socket, isConnected]);

  const handleParamChange = (param, value) => {
    switch (param) {
      case 'leverage':
        setLeverage(value);
        break;
      case 'slPercent':
        setSlPercent(value);
        break;
      case 'tpPercent':
        setTpPercent(value);
        break;
      case 'riskPercent':
        setRiskPercent(value);
        break;
    }
  };

  const handleExecuteTrade = async () => {
    if (!selectedSide) {
      setStatus({ type: 'error', message: 'Please select Long or Short' });
      return;
    }

    setIsLoading(true);
    
    const tradeData = {
      symbol: symbol,
      side: selectedSide,
      leverage: leverage,
      sl_percent: slPercent,
      tp_percent: tpPercent,
      risk_percent: riskPercent
    };

    try {
      const response = await fetch('/place_trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tradeData)
      });

      const result = await response.json();

      if (result.success) {
        setStatus({
          type: 'success',
          message: `✅ TRADE EXECUTED!\n${selectedSide} ${symbol} ${leverage}x\nEntry: $${price?.toFixed(4)}\nRisk: $${(balance * riskPercent / 100).toFixed(2)}`
        });
      } else {
        setStatus({ type: 'error', message: `❌ ERROR: ${result.error}` });
      }
    } catch (error) {
      setStatus({ type: 'error', message: `❌ ERROR: ${error.message}` });
    }

    setIsLoading(false);
  };

  const handleSymbolChange = (newSymbol) => {
    setSymbol(newSymbol.toUpperCase());
    setPrice(null);
    setChange24h(null);
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-xl p-6 max-w-md w-full border border-gray-700 shadow-2xl">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-cyan-400 mb-2 flex items-center justify-center gap-2">
            <Activity className="w-6 h-6" />
            MARGIN RISK MANAGER
          </h1>
          <div className="text-green-400 font-bold flex items-center justify-center gap-1">
            <DollarSign className="w-4 h-4" />
            Balance: ${balance.toFixed(2)} USDT
          </div>
        </div>

        {/* Symbol Input */}
        <div className="mb-6">
          <label className="block text-gray-300 text-sm font-bold mb-2 uppercase">
            Ticker Symbol
          </label>
          <input
            type="text"
            value={symbol}
            onChange={(e) => handleSymbolChange(e.target.value)}
            className="w-full p-3 bg-gray-700 border-2 border-gray-600 rounded text-center text-white text-lg font-bold uppercase focus:border-cyan-400 focus:outline-none"
            placeholder="APTUSDT"
          />
        </div>

        {/* Price Display */}
        <PriceDisplay 
          symbol={symbol}
          price={price}
          change24h={change24h}
          isConnected={isConnected}
        />

        {/* Side Selection */}
        <SideSelector 
          selectedSide={selectedSide}
          onSideChange={setSelectedSide}
        />

        {/* Trading Parameters */}
        <TradingParams
          leverage={leverage}
          slPercent={slPercent}
          tpPercent={tpPercent}
          riskPercent={riskPercent}
          onChange={handleParamChange}
        />

        {/* Risk Calculator */}
        <RiskCalculator
          price={price}
          selectedSide={selectedSide}
          leverage={leverage}
          slPercent={slPercent}
          tpPercent={tpPercent}
          riskPercent={riskPercent}
          balance={balance}
        />

        {/* Execute Button */}
        <button
          onClick={handleExecuteTrade}
          disabled={!selectedSide || isLoading}
          className="w-full p-4 bg-gradient-to-r from-orange-600 to-orange-500 hover:from-orange-500 hover:to-orange-400 text-white font-bold text-lg uppercase rounded-lg transition-all transform hover:-translate-y-1 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
        >
          {isLoading ? (
            <span className="flex items-center justify-center gap-2">
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              EXECUTING...
            </span>
          ) : (
            'EXECUTE TRADE'
          )}
        </button>

        {/* Status */}
        {status && (
          <div className={`mt-4 p-3 rounded-lg text-center font-bold ${
            status.type === 'success' 
              ? 'bg-green-900 border border-green-600 text-green-300'
              : 'bg-red-900 border border-red-600 text-red-300'
          }`}>
            <pre className="whitespace-pre-line text-sm">{status.message}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;