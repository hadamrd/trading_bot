import { useState, useEffect, useRef } from 'react';

export const useMarketData = () => {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [marketData, setMarketData] = useState({});
  const [signals, setSignals] = useState([]);
  const [deltaHistory, setDeltaHistory] = useState([]);
  const [config, setConfig] = useState({});
  const eventSourceRef = useRef(null);

  useEffect(() => {
    const connectToStream = () => {
      eventSourceRef.current = new EventSource('http://localhost:5000/stream');
      
      eventSourceRef.current.onopen = () => {
        setConnectionStatus('connected');
      };
      
      eventSourceRef.current.onerror = () => {
        setConnectionStatus('error');
        setTimeout(connectToStream, 5000);
      };
      
      eventSourceRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type !== 'heartbeat') {
            console.log('📊 Received data:', data); // DEBUG
            setMarketData(data);
            
            // Update delta history for charts - FIX THE PATH
            const deltaStats = data.statistics?.delta_stats;
            const metrics = data.latest_metrics;
            
            console.log('📈 Delta stats:', deltaStats); // DEBUG
            console.log('💰 Metrics:', metrics); // DEBUG
            
            if (deltaStats && metrics?.mid_price) {
              console.log('✅ Adding to chart history!'); // DEBUG
              const now = new Date();
              const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${Math.floor(now.getMilliseconds()/100)}`;
              setDeltaHistory(prev => {
                const newPoint = {
                  time: timestamp,
                  cumulativeDelta: deltaStats.cumulative_delta || 0,
                  price: metrics.mid_price || 0,
                  buyVolume: deltaStats.buy_percentage || 50,
                  currentDelta: deltaStats.current_minute_delta || 0
                };
                console.log('📊 New data point:', newPoint); // DEBUG
                const newHistory = [...prev, newPoint];
                return newHistory.slice(-100);
              });
            } else {
              console.log('❌ Missing data for charts:', { deltaStats, midPrice: metrics?.mid_price });
            }
            
            if (data.signals) {
              setSignals(data.signals);
            }
          }
        } catch (e) {
          console.error('Error parsing data:', e);
        }
      };
    };

    connectToStream();
    
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Load configuration
  useEffect(() => {
    fetch('http://localhost:5000/api/config')
      .then(response => response.json())
      .then(setConfig)
      .catch(console.error);
  }, []);

  const updateConfig = (newConfig) => {
    fetch('http://localhost:5000/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newConfig)
    })
    .then(response => response.json())
    .then(() => setConfig(prev => ({ ...prev, ...newConfig })))
    .catch(console.error);
  };

  return {
    connectionStatus,
    marketData,
    signals,
    deltaHistory,
    config,
    updateConfig
  };
};