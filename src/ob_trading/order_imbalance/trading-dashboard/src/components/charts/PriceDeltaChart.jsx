// src/components/charts/PriceDeltaChart.jsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export const PriceDeltaChart = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-400">
        No data available - waiting for trades...
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="time" stroke="#9CA3AF" fontSize={10} />
        <YAxis yAxisId="price" orientation="left" stroke="#10B981" />
        <YAxis yAxisId="delta" orientation="right" stroke="#F59E0B" />
        <Tooltip 
          contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
        />
        <Line 
          yAxisId="price"
          type="monotone" 
          dataKey="price" 
          stroke="#10B981" 
          strokeWidth={2}
          dot={false}
          name="Price"
        />
        <Line 
          yAxisId="delta"
          type="monotone" 
          dataKey="cumulativeDelta" 
          stroke="#F59E0B" 
          strokeWidth={2}
          dot={false}
          name="Delta"
        />
      </LineChart>
    </ResponsiveContainer>
  );
};