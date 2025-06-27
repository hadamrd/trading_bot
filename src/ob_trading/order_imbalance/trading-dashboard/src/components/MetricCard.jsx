import React from 'react';

export const MetricCard = ({ title, value, color = "green", subtitle }) => (
  <div className="bg-gray-800 border border-gray-600 rounded-lg p-4 text-center">
    <div className={`text-2xl font-bold text-${color}-400`}>{value}</div>
    <div className="text-sm text-gray-400">{title}</div>
    {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
  </div>
);