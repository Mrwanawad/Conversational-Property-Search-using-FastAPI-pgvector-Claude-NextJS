"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { UnitCard } from "@/components/UnitCard";
import { UnitDrawer } from "@/components/UnitDrawer";
import { api, filterPills, type Lang, type Meta, type SearchTurn } from "@/lib/api";

type Msg = { role: "user"; text: string } | { role: "assistant"; turn: SearchTurn } | { role: "error"; text: string };

const isArabic = (s: string) => /[؀-ۿ]/.test(s);

export default function Home() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [slow, setSlow] = useState(false);
  const [open, setOpen] = useState<{ id: number; lang: Lang } | null>(null);
  const [inquired, setInquired] = useState<Set<number>>(new Set());
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.meta().then(setMeta).catch(() => null);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [msgs, busy]);

  const send = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message || busy) return;
      setInput("");
      setMsgs((m) => [...m, { role: "user", text: message }]);
      setBusy(true);
      // Free hosting sleeps when idle; tell the user instead of looking frozen.
      const timer = setTimeout(() => setSlow(true), 4000);
      try {
        const turn = await api.search(message, sessionId);
        setSessionId(turn.session_id);
        setMsgs((m) => [...m, { role: "assistant", turn }]);
      } catch {
        setMsgs((m) => [
          ...m,
          {
            role: "error",
            text: isArabic(message)
              ? "حصلت مشكلة في الاتصال بالسيرفر. جرب تاني."
              : "Could not reach the search service. Try again.",
          },
        ]);
      } finally {
        clearTimeout(timer);
        setSlow(false);
        setBusy(false);
      }
    },
    [busy, sessionId],
  );

  const reset = () => {
    setMsgs([]);
    setSessionId(null);
    setInquired(new Set());
  };

  const openUnit = (turn: SearchTurn, unitId: number, position: number) => {
    setOpen({ id: unitId, lang: turn.reply_language });
    api.event({ session_id: turn.session_id, turn_id: turn.turn_id, unit_id: unitId, kind: "click", position });
  };

  const inquire = (turn: SearchTurn, unitId: number, position: number) => {
    setInquired((s) => new Set(s).add(unitId));
    api.event({ session_id: turn.session_id, turn_id: turn.turn_id, unit_id: unitId, kind: "inquiry", position });
  };

  return (
    <main className="shell">
      <header className="top">
        <div>
          <p className="kicker">Arabic · Franco-Arabic · English</p>
          <h1>Property Search</h1>
        </div>
        <nav>
          {msgs.length > 0 && (
            <button type="button" className="btn-ghost" onClick={reset}>
              New search · بحث جديد
            </button>
          )}
          <a className="btn-ghost" href="/insights">
            Insights
          </a>
        </nav>
      </header>

      <section className="thread" aria-live="polite">
        {msgs.length === 0 && (
          <div className="empty">
            <p className="lede">
              Describe the home you want the way you would say it.
              <br />
              <span dir="rtl">اكتب اللي بتدور عليه زي ما بتتكلم.</span>
            </p>
            <div className="examples">
              {(meta?.examples ?? []).map((ex) => (
                <button key={ex} type="button" className="example" dir="auto" onClick={() => send(ex)}>
                  {ex}
                </button>
              ))}
            </div>
            <p className="note">
              Demo on synthetic inventory: compounds, developers and prices are generated, not real listings.
            </p>
          </div>
        )}

        {msgs.map((m, i) => {
          if (m.role === "user")
            return (
              <div key={i} className="msg msg-user" dir="auto" data-testid="user-msg">
                {m.text}
              </div>
            );
          if (m.role === "error")
            return (
              <div key={i} className="msg msg-error" dir="auto">
                {m.text}
              </div>
            );
          const t = m.turn;
          const lang = t.reply_language;
          const en = lang === "en";
          const pills = filterPills(t.parsed.filters, meta, lang);
          const last = i === msgs.length - 1;
          return (
            <div key={i} className="msg msg-assistant" dir={en ? "ltr" : "rtl"} data-testid="assistant-msg">
              {pills.length > 0 && (
                <div className="understood" data-testid="filters">
                  <span className="understood-label">{en ? "Understood" : "فهمت"}</span>
                  {pills.map((p) => (
                    <span key={p} className="pill">
                      {p}
                    </span>
                  ))}
                  {t.parsed.soft_preferences && (
                    <span className="pill pill-soft" dir="auto">
                      “{t.parsed.soft_preferences}”
                    </span>
                  )}
                </div>
              )}

              <p className="reply" data-testid="reply">
                {t.message}
              </p>

              {t.relaxations.length > 0 && (
                <ul className="relaxed" data-testid="relaxations">
                  {t.relaxations.map((r, k) => (
                    <li key={k}>{en ? r.en : r.ar}</li>
                  ))}
                </ul>
              )}

              {t.clarification && (
                <div className="chips" data-testid="clarification">
                  {t.clarification.options.map((o) => (
                    <button
                      key={o.value}
                      type="button"
                      className="chip"
                      disabled={!last || busy}
                      onClick={() => send(o.value)}
                    >
                      {en ? o.label_en : o.label_ar}
                    </button>
                  ))}
                </div>
              )}

              {t.units.length > 0 && (
                <div className="cards">
                  {t.units.map((u, k) => (
                    <UnitCard
                      key={u.id}
                      unit={u}
                      lang={lang}
                      inquired={inquired.has(u.id)}
                      onOpen={() => openUnit(t, u.id, k + 1)}
                      onInquire={() => inquire(t, u.id, k + 1)}
                    />
                  ))}
                </div>
              )}

              {last && t.suggestions.length > 0 && (
                <div className="chips" data-testid="suggestions">
                  {t.suggestions.map((s) => (
                    <button key={s.value} type="button" className="chip" disabled={busy} onClick={() => send(s.value)}>
                      {s.label}
                    </button>
                  ))}
                </div>
              )}

              <p className="meta-line">
                {t.parsed.parser} · {Math.round(t.timings_ms.total ?? 0)} ms
                {t.total_matches > 0 && ` · ${t.total_matches} ${en ? "matches" : "نتيجة"}`}
              </p>
            </div>
          );
        })}

        {busy && (
          <div className="msg msg-assistant typing" data-testid="busy">
            <span className="dots">
              <i />
              <i />
              <i />
            </span>
            {slow && <span className="muted"> Waking the server up, first search can take up to a minute.</span>}
          </div>
        )}
        <div ref={endRef} className="end" />
      </section>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <input
          dir="auto"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Describe the home you want · اكتب اللي بتدور عليه"
          maxLength={500}
          aria-label="Search message"
          data-testid="composer"
        />
        <button type="submit" className="btn" disabled={busy || !input.trim()}>
          Search
        </button>
      </form>

      {open && <UnitDrawer id={open.id} lang={open.lang} onClose={() => setOpen(null)} />}
    </main>
  );
}
