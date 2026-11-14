"use client";

import { egp, label, type Lang, type UnitHit } from "@/lib/api";

const T = {
  en: { beds: "bd", baths: "ba", m2: "m²", perM2: "/m²", down: "down", mo: "/mo", yrs: "yrs", inquire: "Inquire", sent: "Request sent", details: "Details", delivery: "Delivery" },
  ar: { beds: "غرف", baths: "حمام", m2: "م²", perM2: "/م²", down: "مقدم", mo: "/شهر", yrs: "سنين", inquire: "استفسر", sent: "تم الإرسال", details: "التفاصيل", delivery: "الاستلام" },
};

export function UnitCard({
  unit,
  lang,
  inquired,
  onOpen,
  onInquire,
}: {
  unit: UnitHit;
  lang: Lang;
  inquired: boolean;
  onOpen: () => void;
  onInquire: () => void;
}) {
  const t = T[lang];
  const en = lang === "en";
  const plan = unit.best_plan;
  return (
    <article className="card" data-testid="unit-card" dir={en ? "ltr" : "rtl"}>
      <header className="card-head">
        <div>
          <h3 className="card-title">{en ? unit.title_en : unit.title_ar}</h3>
          <p className="card-sub">
            {en ? unit.area_en : unit.area_ar} · {unit.developer} · <span className="code">{unit.code}</span>
          </p>
        </div>
        <div className="price">
          <strong>{egp(unit.price, lang)}</strong>
          <span>
            {Math.round(unit.price_per_m2).toLocaleString("en-US")} {t.perM2}
          </span>
        </div>
      </header>

      <dl className="facts">
        <div>
          <dt>{label(unit.unit_type, lang)}</dt>
          <dd>
            {unit.bedrooms} {t.beds} · {unit.bathrooms} {t.baths}
          </dd>
        </div>
        <div>
          <dt>{t.m2}</dt>
          <dd>
            {unit.bua_m2}
            {unit.garden_m2 > 0 ? ` + ${unit.garden_m2}` : ""}
          </dd>
        </div>
        <div>
          <dt>{t.delivery}</dt>
          <dd>{unit.delivery}</dd>
        </div>
        {plan && (
          <div>
            <dt>
              {plan.down_payment_pct}% {t.down} · {plan.years} {t.yrs}
            </dt>
            <dd>
              {egp(plan.monthly, lang)}
              {t.mo}
            </dd>
          </div>
        )}
      </dl>

      <ul className="reasons">
        {unit.reasons.map((r, i) => (
          <li key={i} className={`reason reason-${r.kind}`}>
            {en ? r.en : r.ar}
          </li>
        ))}
      </ul>

      <footer className="card-actions">
        <button type="button" className="btn-ghost" onClick={onOpen}>
          {t.details}
        </button>
        <button type="button" className="btn" onClick={onInquire} disabled={inquired}>
          {inquired ? t.sent : t.inquire}
        </button>
      </footer>
    </article>
  );
}
