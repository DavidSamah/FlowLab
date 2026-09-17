"use client";

import { useMemo, useState } from "react";

function makeField(viscosity: number, steps: number) {
  const rows = 18, cols = 34;
  const a = Math.min(0.24, viscosity * 0.1);
  let f = Array.from({ length: rows }, (_, y) => Array.from({ length: cols }, (_, x) =>
    y >= 8 && y <= 9 && x >= 3 && x <= 6 ? 1 : 0));
  for (let s = 0; s < steps; s++) {
    const previous = f;
    f = previous.map((row, y) => row.map((value, x) => {
      const l = previous[y][Math.max(0, x - 1)], r = previous[y][Math.min(cols - 1, x + 1)];
      const u = previous[Math.max(0, y - 1)][x], d = previous[Math.min(rows - 1, y + 1)][x];
      return value + a * (l + r + u + d - 4 * value);
    }));
  }
  return f;
}

function channelBenchmark(viscosity: number, height = 20) {
  const pressureGradient = 1;
  const H = height - 1;
  return Array.from({ length: height }, (_, y) =>
    pressureGradient * y * (H - y) / (2 * Math.max(viscosity, 0.01))
  );
}

// Independent finite-difference/Jacobi approximation of steady planar
// Poiseuille flow: nu * u'' = -G, with no-slip u(0)=u(H)=0.
function numericalChannel(viscosity: number, iterations: number, height = 20) {
  const pressureGradient = 1;
  const nu = Math.max(viscosity, 0.01);
  let u = Array(height).fill(0) as number[];
  for (let k = 0; k < iterations; k++) {
    const previous = u;
    u = previous.slice();
    for (let y = 1; y < height - 1; y++) {
      u[y] = 0.5 * (previous[y - 1] + previous[y + 1] + pressureGradient / nu);
    }
  }
  return u;
}

function normalizedRmse(reference: number[], numerical: number[]) {
  const mse = reference.reduce((sum, value, i) => {
    const d = numerical[i] - value;
    return sum + d * d;
  }, 0) / reference.length;
  const scale = Math.max(...reference.map(Math.abs), Number.EPSILON);
  return Math.sqrt(mse) / scale;
}

export default function Home() {
  const [viscosity, setViscosity] = useState(0.1);
  const [steps, setSteps] = useState(100);
  const field = useMemo(() => makeField(viscosity, steps), [viscosity, steps]);
  const peak = Math.max(...field.flat());
  const total = field.flat().reduce((a, b) => a + b, 0);
  const channel = useMemo(() => channelBenchmark(viscosity), [viscosity]);
  const numericalChannelResult = useMemo(() => numericalChannel(viscosity, steps), [viscosity, steps]);
  const benchmarkError = normalizedRmse(channel, numericalChannelResult);
  const channelPeak = Math.max(...channel, Number.EPSILON);

  return <main className="min-h-screen bg-[#07151d] text-slate-100">
    <header className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
      <div><p className="text-xs font-semibold uppercase tracking-[0.25em] text-cyan-300">Flow Lab / 01</p><h1 className="mt-1 text-2xl font-semibold tracking-tight">A transparent fluid laboratory</h1></div>
      <div className="rounded-full border border-cyan-900 bg-cyan-950/40 px-4 py-2 text-xs text-cyan-200">Research prototype · 2D</div>
    </header>
    <section className="mx-auto grid max-w-7xl gap-6 px-6 pb-10 lg:grid-cols-[1fr_320px]">
      <div className="rounded-3xl border border-slate-700 bg-[#0b222d] p-5 shadow-2xl shadow-cyan-950/20">
        <div className="mb-5 flex items-end justify-between"><div><p className="text-sm text-slate-400">Viscosity diffusion experiment</p><h2 className="text-xl font-medium">Localized disturbance</h2></div><div className="text-right text-xs text-slate-400">Grid 34 × 18<br/><span className="text-cyan-300">live recalculation</span></div></div>
        <div className="grid gap-[3px] rounded-2xl bg-[#061017] p-3" style={{gridTemplateColumns:`repeat(34,minmax(0,1fr))`}} aria-label="Flow field visualization">{field.flat().map((v,i)=><div key={i} className="aspect-square rounded-[2px]" style={{backgroundColor:`hsl(${190-v*35} ${55+v*35}% ${9+v*62}%)`}} />)}</div>
        <div className="mt-4 flex items-center justify-between text-xs text-slate-400"><span>low energy</span><div className="h-2 w-40 rounded-full bg-gradient-to-r from-[#123340] via-cyan-400 to-white"/><span>peak</span></div>
      </div>
      <aside className="space-y-4">
        <div className="rounded-3xl border border-slate-700 bg-[#0b222d] p-5"><p className="text-xs uppercase tracking-widest text-slate-400">Controls</p><label className="mt-5 block text-sm">Viscosity <span className="float-right text-cyan-300">{viscosity.toFixed(2)}</span></label><input className="mt-3 w-full accent-cyan-400" type="range" min="0.01" max="0.24" step="0.01" value={viscosity} onChange={e=>setViscosity(Number(e.target.value))}/><label className="mt-6 block text-sm">Time / solver steps <span className="float-right text-cyan-300">{steps}</span></label><input className="mt-3 w-full accent-cyan-400" type="range" min="0" max="300" step="10" value={steps} onChange={e=>setSteps(Number(e.target.value))}/></div>
        <div className="grid grid-cols-2 gap-3"><div className="rounded-2xl border border-slate-700 bg-[#0b222d] p-4"><p className="text-xs text-slate-400">Peak</p><p className="mt-1 text-2xl font-semibold text-cyan-200">{peak.toFixed(3)}</p></div><div className="rounded-2xl border border-slate-700 bg-[#0b222d] p-4"><p className="text-xs text-slate-400">Total</p><p className="mt-1 text-2xl font-semibold text-cyan-200">{total.toFixed(2)}</p></div></div>
        <div className="rounded-3xl border border-amber-900/60 bg-amber-950/30 p-5 text-sm leading-6 text-amber-100"><p className="font-semibold">Model boundary</p><p className="mt-2 text-amber-200/80">This is the diffusion component only—not a complete Navier–Stokes solver. Every parameter is visible so the result can be questioned.</p></div>
        <div className="rounded-3xl border border-slate-700 bg-[#0b222d] p-5"><p className="text-xs uppercase tracking-widest text-slate-400">Validation</p><p className="mt-3 text-sm text-slate-300">Poiseuille: analytical vs finite-difference</p><div className="mt-3 flex h-20 items-end gap-1">{numericalChannelResult.map((v,i)=><div key={i} className="flex-1 rounded-t bg-cyan-400/70" title={`numerical ${v.toFixed(3)} / analytical ${channel[i].toFixed(3)}`} style={{height:`${Math.max(4,Math.min(100,v/channelPeak*100))}%`}} />)}</div><p className={`mt-3 text-xs ${benchmarkError < 0.05 ? "text-emerald-300" : "text-amber-300"}`}>Normalized RMSE · {(benchmarkError * 100).toFixed(2)}%</p></div>
      </aside>
    </section>
    <section className="mx-auto grid max-w-7xl gap-6 px-6 pb-16 md:grid-cols-3"><div className="md:col-span-2 rounded-3xl border border-slate-700 bg-[#0b222d] p-6"><p className="text-xs uppercase tracking-widest text-cyan-300">Evidence note</p><p className="mt-3 max-w-3xl text-lg leading-8 text-slate-200">The diffusion visualization is separate from the validation experiment. The channel-flow panel independently compares a finite-difference numerical solution against the analytical Poiseuille profile and reports normalized RMSE.</p></div><div className="rounded-3xl border border-slate-700 bg-[#0b222d] p-6"><p className="text-xs uppercase tracking-widest text-slate-400">Practical use case</p><p className="mt-3 text-slate-200">Ventilation studies can compare how viscosity and geometry alter mixing before expensive CFD runs.</p></div></section>
  </main>;
}
