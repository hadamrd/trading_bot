import React from 'react';

export const ConnectionStatus = ({ status, marketData, signals }) => (
  <div className="flex items-center gap-2 mb-4 bg-gray-800 p-3 rounded-lg justify-between">
    <div className="flex items-center gap-2">
      <div className={`w-3 h-3 rounded-full ${
        status === 'connected' ? 'bg-green-500 shadow-green-500/50' : 
        status === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
      } shadow-lg`}></div>
      <span className="text-sm text-green-400 font-bold">
        {status === 'connected' ? 'Connected - Live Data' : 
         status === 'connecting' ? 'Connecting...' : 'Connection Error'}
      </span>
    </div>
    <div className="text-xs text-gray-400">
      Updates: {marketData.update_count?.toLocaleString() || 0} | 
      Signals: {signals.length} | 
      Price: ${marketData.latest_metrics?.mid_price?.toFixed(4) || '0.0000'} |
      Delta Confirmations: {(marketData.statistics?.delta_confirmation_rate || 0).toFixed(1)}%
    </div>
  </div>
);