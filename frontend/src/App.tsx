import React, { useState, useEffect } from 'react';
import './index.css';

interface MatchState {
  match_id: string;
  home_score: number;
  away_score: number;
  elapsed_seconds: number;
  is_active: bool;
}

export function App() {
  const [homeScore, setHomeScore] = useState(2);
  const [awayScore, setAwayScore] = useState(1);
  const [clock, setClock] = useState("34:12");
  const [period, setPeriod] = useState("1ST HALF");
  const [activeTab, setActiveTab] = useState<'live' | 'analytics' | 'radar'>('live');

  const [events, setEvents] = useState([
    { id: '1', type: 'GOAL', team: 'HOME', time: '12:04', desc: 'Goal scored by #10 (Speed: 78 km/h)' },
    { id: '2', type: 'SHOT', team: 'AWAY', time: '21:30', desc: 'Shot on target saved by Goalkeeper' },
    { id: '3', type: 'GOAL', team: 'AWAY', time: '28:15', desc: 'Goal scored by #7 (Signed distance: +1.2m)' },
    { id: '4', type: 'GOAL', team: 'HOME', time: '32:50', desc: 'Goal scored by #9 (Distance: 18.5m)' },
  ]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Header & Live Scoreboard Bar */}
      <header className="glass-panel sticky top-0 z-50 px-6 py-4 flex items-center justify-between border-b border-slate-800">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-emerald-500 flex items-center justify-center font-bold text-slate-950 text-xl accent-glow-green">
            ⚽
          </div>
          <div>
            <h1 className="font-extrabold text-xl tracking-tight text-white">TURFVISION <span className="text-emerald-400">AI</span></h1>
            <p className="text-xs text-slate-400">Commercial Sports Computer Vision & Telemetry SaaS</p>
          </div>
        </div>

        {/* Live Broadcast Scoreboard Widget */}
        <div className="bg-slate-900 border border-slate-700/80 rounded-xl px-6 py-2 flex items-center space-x-6 shadow-2xl">
          <div className="flex items-center space-x-3">
            <span className="font-black text-lg text-emerald-400">HOME FC</span>
            <span className="font-black text-2xl text-white">{homeScore}</span>
          </div>
          <span className="text-slate-500 font-bold text-lg">:</span>
          <div className="flex items-center space-x-3">
            <span className="font-black text-2xl text-white">{awayScore}</span>
            <span className="font-black text-lg text-cyan-400">AWAY UT</span>
          </div>
          <div className="h-8 w-px bg-slate-700"></div>
          <div className="flex flex-col items-center">
            <span className="text-xs font-bold text-emerald-400 tracking-wider flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> {period}
            </span>
            <span className="font-mono font-extrabold text-lg text-white">{clock}</span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex bg-slate-900/80 p-1 rounded-lg border border-slate-800">
          <button 
            onClick={() => setActiveTab('live')}
            className={`px-4 py-1.5 rounded-md text-sm font-semibold transition ${activeTab === 'live' ? 'bg-emerald-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'}`}
          >
            Live Feed
          </button>
          <button 
            onClick={() => setActiveTab('radar')}
            className={`px-4 py-1.5 rounded-md text-sm font-semibold transition ${activeTab === 'radar' ? 'bg-emerald-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'}`}
          >
            2D Tactical Radar
          </button>
          <button 
            onClick={() => setActiveTab('analytics')}
            className={`px-4 py-1.5 rounded-md text-sm font-semibold transition ${activeTab === 'analytics' ? 'bg-emerald-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'}`}
          >
            Match Report
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 p-6 grid grid-cols-12 gap-6 max-w-7xl mx-auto w-full">
        {/* Left Column: Video Feed / Tactical Pitch */}
        <section className="col-span-8 flex flex-col space-y-6">
          <div className="glass-panel rounded-2xl overflow-hidden border border-slate-800 shadow-2xl relative aspect-video flex items-center justify-center bg-slate-900">
            {activeTab === 'live' && (
              <div className="w-full h-full relative flex items-center justify-center bg-gradient-to-b from-slate-900 to-slate-950">
                {/* Simulated Computer Vision Overlay Feed */}
                <div className="absolute inset-0 flex items-center justify-center opacity-30">
                  <div className="w-full h-full border-4 border-dashed border-emerald-500/20 m-4 rounded-xl flex items-center justify-center">
                    <span className="text-emerald-400 font-mono text-sm">[ YOLOv8 Multi-Object Player & Ball Detector Stream ]</span>
                  </div>
                </div>

                {/* Bounding Box Demonstrations */}
                <div className="absolute top-1/3 left-1/4 border-2 border-emerald-400 bg-emerald-500/10 rounded px-2 py-1 flex flex-col items-center">
                  <span className="text-[10px] font-bold bg-emerald-400 text-slate-950 px-1 rounded">#10 Player</span>
                  <div className="w-1.5 h-1.5 bg-red-500 rounded-full mt-8"></div>
                </div>

                <div className="absolute top-1/2 right-1/3 border-2 border-cyan-400 bg-cyan-500/10 rounded px-2 py-1 flex flex-col items-center">
                  <span className="text-[10px] font-bold bg-cyan-400 text-slate-950 px-1 rounded">#7 Player</span>
                  <div className="w-1.5 h-1.5 bg-red-500 rounded-full mt-8"></div>
                </div>

                {/* Ball Highlight */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 border-2 border-yellow-400 rounded-full p-2 flex items-center justify-center animate-ping">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                </div>
              </div>
            )}

            {activeTab === 'radar' && (
              <div className="w-full h-full p-6 flex flex-col items-center justify-center bg-emerald-950/40">
                <h3 className="text-emerald-400 font-bold text-sm mb-2">2D BIRD'S EYE VIEW TACTICAL RADAR</h3>
                <div className="w-full h-64 border-2 border-emerald-500/60 rounded-xl relative bg-emerald-900/60 flex items-center justify-center">
                  <div className="w-px h-full bg-emerald-500/60"></div>
                  <div className="w-24 h-24 border-2 border-emerald-500/60 rounded-full absolute"></div>
                  {/* Player Dots */}
                  <div className="w-3 h-3 bg-emerald-400 rounded-full absolute top-1/4 left-1/3 shadow-lg"></div>
                  <div className="w-3 h-3 bg-emerald-400 rounded-full absolute bottom-1/3 left-1/4 shadow-lg"></div>
                  <div className="w-3 h-3 bg-cyan-400 rounded-full absolute top-1/2 right-1/4 shadow-lg"></div>
                  <div className="w-3 h-3 bg-yellow-400 rounded-full absolute top-1/2 left-1/2 shadow-lg"></div>
                </div>
              </div>
            )}
          </div>

          {/* Quick Metrics Bar */}
          <div className="grid grid-cols-3 gap-4">
            <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col">
              <span className="text-xs font-semibold text-slate-400">Possession Split</span>
              <div className="flex items-center justify-between mt-2">
                <span className="font-extrabold text-lg text-emerald-400">54%</span>
                <div className="w-24 h-2 bg-slate-800 rounded-full overflow-hidden flex">
                  <div className="w-[54%] bg-emerald-500 h-full"></div>
                  <div className="w-[46%] bg-cyan-500 h-full"></div>
                </div>
                <span className="font-extrabold text-lg text-cyan-400">46%</span>
              </div>
            </div>

            <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col">
              <span className="text-xs font-semibold text-slate-400">Total Distance Run</span>
              <div className="flex items-baseline space-x-2 mt-1">
                <span className="font-extrabold text-2xl text-white">48.2</span>
                <span className="text-xs text-slate-400 font-semibold">km</span>
              </div>
            </div>

            <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col">
              <span className="text-xs font-semibold text-slate-400">YOLO Model FPS</span>
              <div className="flex items-baseline space-x-2 mt-1">
                <span className="font-extrabold text-2xl text-emerald-400">62.4</span>
                <span className="text-xs text-slate-400 font-semibold">FPS (TensorRT)</span>
              </div>
            </div>
          </div>
        </section>

        {/* Right Column: Live Event Stream & Logs */}
        <section className="col-span-4 flex flex-col space-y-6">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex-1 flex flex-col">
            <h3 className="font-bold text-md text-white mb-4 flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span> Live Match Event Stream
            </h3>

            <div className="flex-1 space-y-3 overflow-y-auto max-h-[440px] pr-1">
              {events.map((evt) => (
                <div key={evt.id} className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex flex-col space-y-1">
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-extrabold px-2 py-0.5 rounded ${evt.type === 'GOAL' ? 'bg-amber-400 text-slate-950' : 'bg-slate-800 text-slate-300'}`}>
                      {evt.type}
                    </span>
                    <span className="text-xs font-mono text-slate-400">{evt.time}</span>
                  </div>
                  <p className="text-xs text-slate-200 font-medium mt-1">{evt.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
