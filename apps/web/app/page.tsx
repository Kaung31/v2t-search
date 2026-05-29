"use client";
import { useState } from "react";

const API = "http://localhost:8000";

export default function Home() {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  async function search() {
    if (!q.trim()) return;
    setLoading(true);
    const r = await fetch(`${API}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
    });
    const data = await r.json();
    setResults(data.results || []);
    setLoading(false);
  }

  return (
    <main className="min-h-screen bg-gray-50 p-8">
      <h1 className="text-2xl font-bold mb-4">V2T-Search (vertical slice)</h1>
      <div className="flex gap-2 mb-6">
        <input
          className="flex-1 border rounded px-3 py-2"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && search()}
          placeholder="Try: 'a car', 'a person', 'a road at night'"
        />
        <button onClick={search} className="bg-black text-white rounded px-4">
          {loading ? "..." : "Search"}
        </button>
      </div>
      <div className="grid grid-cols-3 gap-4">
        {results.map((r, i) => (
          <div key={i} className="bg-white rounded shadow p-2">
            <img src={`${API}${r.thumbnail}`} className="w-full rounded" alt="" />
            <div className="text-sm mt-2 flex justify-between">
              <span>t = {r.timestamp.toFixed(2)}s</span>
              <span className="text-gray-500">{r.score.toFixed(2)}</span>
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}