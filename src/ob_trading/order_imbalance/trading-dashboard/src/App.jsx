// src/App.jsx - Main component  
import React, { useState } from 'react';
import { useMarketData } from './hooks/useMarketData';
import { ConnectionStatus } from './components/ConnectionStatus';
import { OverviewTab } from './components/tabs/OverviewTab';

const TradingDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const { 
    connectionStatus, 
    marketData, 
    signals, 
    deltaHistory, 
    config, 
    updateConfig 
  } = useMarketData();

  const tabs = [
    { id: 'overview', name: 'Overview', icon: '📊' },
    { id: 'orderbook', name: 'Order Book', icon: '📋' },
    { id: 'delta', name: 'Delta Analysis', icon: '📈' },
    { id: 'signals', name: 'Signals', icon: '🚨' },
    { id: 'config', name: 'Config', icon: '⚙️' }
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-6">
        <ConnectionStatus 
          status={connectionStatus} 
          marketData={marketData} 
          signals={signals} 
        />
        
        {/* Tab Navigation */}
        <div className="flex space-x-1 mb-6 bg-gray-800 p-1 rounded-lg">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md transition-colors ${
                activeTab === tab.id 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <span>{tab.icon}</span>
              <span className="hidden sm:inline">{tab.name}</span>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {activeTab === 'overview' && (
            <OverviewTab 
              marketData={marketData} 
              deltaHistory={deltaHistory} 
            />
          )}
          {/* Add other tabs here */}
        </div>
      </div>
    </div>
  );
};

export default TradingDashboard;