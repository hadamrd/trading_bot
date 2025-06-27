import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

export const CumulativeDeltaChart = ({ data }) => {
  console.log('📈 Chart data:', data); // DEBUG
  
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
        <XAxis 
          dataKey="time" 
          stroke="#9CA3AF" 
          fontSize={10} 
          tick={{ fill: '#9CA3AF' }}
        />
        <YAxis 
          stroke="#9CA3AF" 
          tick={{ fill: '#9CA3AF' }}
          tickFormatter={(value) => `$${value.toLocaleString()}`}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1F2937', 
            border: '1px solid #374151',
            color: '#F3F4F6'
          }}
          formatter={(value) => [`$${value.toLocaleString()}`, 'Cumulative Delta']}
        />
        <Line 
          type="monotone" 
          dataKey="cumulativeDelta" 
          stroke="#F59E0B" 
          strokeWidth={2}
          dot={false}
          name="Cumulative Delta"
        />
        <ReferenceLine y={0} stroke="#6B7280" strokeDasharray="2 2" />
      </LineChart>
    </ResponsiveContainer>
  );
};