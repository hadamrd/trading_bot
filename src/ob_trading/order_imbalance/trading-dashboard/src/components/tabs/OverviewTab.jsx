// src/components/tabs/OverviewTab.jsx
import React from 'react';
import { MetricCard } from '../MetricCard';
import { CumulativeDeltaChart } from '../charts/CumulativeDeltaChart';
import { PriceDeltaChart } from '../charts/PriceDeltaChart';

export const OverviewTab = ({ marketData, deltaHistory }) => {
  const metrics = marketData.latest_metrics || {};
  const deltaStats = marketData.statistics?.delta_stats || {};
  
  return (
    <div className="space-y-6">
      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard 
          title="Price" 
          value={`$${metrics.mid_price?.toFixed(4) || '0.0000'}`}
          color="blue"
        />
        <MetricCard 
          title="Spread" 
          value={`${(metrics.spread_pct || 0).toFixed(3)}%`}
          color="yellow"
        />
        <MetricCard 
          title="Bid Ratio" 
          value={`${((metrics.bid_ratio || 0.5) * 100).toFixed(1)}%`}
          color={metrics.bid_ratio > 0.6 ? "green" : metrics.bid_ratio < 0.4 ? "red" : "yellow"}
        />
        <MetricCard 
          title="Total Value" 
          value={`$${((metrics.total_bid_value || 0) + (metrics.total_ask_value || 0)).toLocaleString()}`}
          color="purple"
        />
      </div>

      {/* Delta Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard 
          title="Cumulative Delta" 
          value={`$${deltaStats.cumulative_delta?.toLocaleString() || '0'}`}
          color={deltaStats.cumulative_delta > 1000 ? "green" : deltaStats.cumulative_delta < -1000 ? "red" : "orange"}
        />
        <MetricCard 
          title="Current Period" 
          value={`$${deltaStats.current_minute_delta?.toLocaleString() || '0'}`}
          color={deltaStats.current_minute_delta > 100 ? "green" : deltaStats.current_minute_delta < -100 ? "red" : "orange"}
        />
        <MetricCard 
          title="Buy Volume" 
          value={`${deltaStats.buy_percentage?.toFixed(1) || '50.0'}%`}
          color={deltaStats.buy_percentage > 60 ? "green" : deltaStats.buy_percentage < 40 ? "red" : "orange"}
        />
        <MetricCard 
          title="Delta Trend" 
          value={deltaStats.delta_trend || 'NEUTRAL'}
          color={deltaStats.delta_trend === 'BULLISH' ? "green" : deltaStats.delta_trend === 'BEARISH' ? "red" : "yellow"}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 border border-gray-600 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4 text-orange-400">Cumulative Delta</h3>
          <CumulativeDeltaChart data={deltaHistory} />
        </div>

        <div className="bg-gray-800 border border-gray-600 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4 text-green-400">Price vs Delta</h3>
          <PriceDeltaChart data={deltaHistory} />
        </div>
      </div>

      {/* Divergence Alert */}
      {deltaStats.divergence_detected && (
        <div className="bg-yellow-900/50 border border-yellow-600 rounded-lg p-4 text-center">
          <div className="text-yellow-400 font-bold text-lg">⚠️ PRICE/DELTA DIVERGENCE DETECTED</div>
          <div className="text-sm text-yellow-200 mt-1">
            Price and volume delta are moving in opposite directions - potential reversal signal
          </div>
        </div>
      )}
    </div>
  );
};