"use client";

import { useEffect, useState } from "react";
import { api, egp } from "@/lib/api";

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);

export default function Insights() {
  const [d, setD] = useState<Record<string, any> | null>(null);
  const [days, setDays] = useState(30);
  const [error, setError] = useState(false);

  useEffect(() => {
    setError(false);
    api.insights(days).then(setD).catch(() => setError(true));
  }, [days]);

  const k = d?.kpis;
  return (
    <main className="shell shell-wide">
      <header className="top">
        <div>
          <p className="kicker">Search analytics</p>
          <h1>Insights</h1>
        </div>
        <nav>
          <select value={days} onChange={(e) => setDays(Number(e.target.value))} aria-label="Window">
            <option value={1}>24 hours</option>
            <option value={7}>7 days</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
          </select>
          <a className="btn-ghost" href="/">
            Back to search
          </a>
        </nav>
      </header>

      {error && <p className="msg msg-error">Could not load insights.</p>}
      {!d && !error && <p className="muted">Loading…</p>}

      {k && (
        <>
          <section className="kpis" data-testid="kpis">
            {[
              ["Turns", k.turns],
              ["Sessions", k.sessions],
              ["Zero-result rate", pct(k.zero_result_rate)],
              ["Clarification rate", pct(k.clarification_rate)],
              ["Click-through", pct(k.ctr)],
              ["Inquiry rate", pct(k.inquiry_rate)],
              ["Latency p50", `${k.latency_p50_ms} ms`],
              ["Latency p95", `${k.latency_p95_ms} ms`],
              ["Cost / query", `$${k.cost_per_query_usd}`],
              ["LLM fallback", pct(k.llm_fallback_rate)],
            ].map(([name, value]) => (
              <div key={name} className="kpi">
                <span>{name}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </section>

          <section className="panel">
            <h2>Demand vs supply by area</h2>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Area</th>
                    <th>Searches</th>
                    <th>Demand share</th>
                    <th>Supply share</th>
                    <th>Median budget</th>
                    <th>Median price</th>
                    <th>Zero-result</th>
                    <th>Inquiry</th>
                  </tr>
                </thead>
                <tbody>
                  {d.areas.map((a: any) => (
                    <tr key={a.area}>
                      <td>{a.name_en}</td>
                      <td>{a.searches}</td>
                      <td>{pct(a.share_of_demand)}</td>
                      <td>{pct(a.share_of_supply)}</td>
                      <td>{a.median_budget ? egp(a.median_budget, "en") : "—"}</td>
                      <td>{a.median_price ? egp(a.median_price, "en") : "—"}</td>
                      <td>{pct(a.zero_result_rate)}</td>
                      <td>{pct(a.inquiry_rate)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <div className="grid-2">
            <section className="panel">
              <h2>Unmet demand</h2>
              {d.unmet_demand.length === 0 && <p className="muted">No zero-result searches in this window.</p>}
              <ul className="list">
                {d.unmet_demand.map((u: any, i: number) => (
                  <li key={i}>
                    <strong>
                      {u.type_names} · {u.area_names}
                      {u.bedrooms ? ` · ${u.bedrooms} bd` : ""} · {u.budget ?? "any budget"}
                    </strong>
                    <span className="muted">
                      {u.searches} searches · blocked by {u.binding_constraint ?? "—"}
                      {u.median_unlock ? `, relaxing it unlocks ~${u.median_unlock} units` : ""}
                    </span>
                    {u.examples.map((e: string) => (
                      <q key={e} dir="auto">
                        {e}
                      </q>
                    ))}
                  </li>
                ))}
              </ul>
            </section>

            <section className="panel">
              <h2>Synonym review queue</h2>
              <p className="muted">Low-confidence or zero-result queries with terms the parser did not know.</p>
              {d.review_queue.length === 0 && <p className="muted">Nothing to review.</p>}
              <ul className="list">
                {d.review_queue.map((r: any, i: number) => (
                  <li key={i}>
                    <q dir="auto">{r.query}</q>
                    <span className="muted">
                      unknown: {r.unknown_terms.join(", ")} · confidence {r.confidence}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          </div>

          <div className="grid-2">
            <section className="panel">
              <h2>Language mix</h2>
              <ul className="bars">
                {Object.entries(d.language_mix as Record<string, number>).map(([lang, n]) => (
                  <li key={lang}>
                    <span>{lang}</span>
                    <i style={{ width: `${(n / Math.max(1, k.turns)) * 100}%` }} />
                    <b>{n}</b>
                  </li>
                ))}
              </ul>
            </section>
            <section className="panel">
              <h2>Latency p95 by stage</h2>
              <ul className="bars">
                {Object.entries(d.latency_p95_by_stage as Record<string, number>).map(([stage, ms]) => (
                  <li key={stage}>
                    <span>{stage}</span>
                    <i style={{ width: `${(ms / Math.max(1, k.latency_p95_ms)) * 100}%` }} />
                    <b>{ms} ms</b>
                  </li>
                ))}
              </ul>
            </section>
          </div>
        </>
      )}
    </main>
  );
}
