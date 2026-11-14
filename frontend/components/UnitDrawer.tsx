"use client";

import { useEffect, useState } from "react";
import { api, egp, label, type Lang } from "@/lib/api";

export function UnitDrawer({ id, lang, onClose }: { id: number; lang: Lang; onClose: () => void }) {
  const [u, setU] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState(false);
  const en = lang === "en";

  useEffect(() => {
    api.unit(id).then(setU).catch(() => setError(true));
  }, [id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside
        className="drawer"
        dir={en ? "ltr" : "rtl"}
        role="dialog"
        aria-modal="true"
        data-testid="unit-drawer"
        onClick={(e) => e.stopPropagation()}
      >
        <button type="button" className="drawer-close" onClick={onClose} aria-label="Close">
          ×
        </button>
        {error && <p>{en ? "Could not load this unit." : "مش قادر أحمل الوحدة دي."}</p>}
        {!u && !error && <p className="muted">…</p>}
        {u && (
          <>
            <p className="kicker">{u.code}</p>
            <h2>
              {label(u.unit_type, lang)} · {en ? u.compound.name_en : u.compound.name_ar}
            </h2>
            <p className="muted">
              {u.compound.developer} ·{" "}
              {en
                ? `${Math.round(u.compound.developer_on_time_rate * 100)}% on-time delivery record`
                : `${Math.round(u.compound.developer_on_time_rate * 100)}% التزام بمواعيد التسليم`}
            </p>
            <p className="drawer-price">{egp(u.price, lang)}</p>
            <dl className="facts facts-wide">
              <div>
                <dt>{en ? "Bedrooms" : "غرف"}</dt>
                <dd>{u.bedrooms}</dd>
              </div>
              <div>
                <dt>{en ? "Bathrooms" : "حمامات"}</dt>
                <dd>{u.bathrooms}</dd>
              </div>
              <div>
                <dt>{en ? "Built-up" : "المساحة"}</dt>
                <dd>{u.bua_m2} m²</dd>
              </div>
              {u.garden_m2 > 0 && (
                <div>
                  <dt>{en ? "Garden" : "جنينة"}</dt>
                  <dd>{u.garden_m2} m²</dd>
                </div>
              )}
              {u.floor !== null && (
                <div>
                  <dt>{en ? "Floor" : "الدور"}</dt>
                  <dd>{u.floor}</dd>
                </div>
              )}
              <div>
                <dt>{en ? "Finishing" : "التشطيب"}</dt>
                <dd>{label(u.finishing, lang)}</dd>
              </div>
              <div>
                <dt>{en ? "Delivery" : "الاستلام"}</dt>
                <dd>{u.is_ready ? (en ? "Ready" : "فوري") : u.delivery_year}</dd>
              </div>
            </dl>
            <p className="desc">{en ? u.description_en : u.description_ar}</p>
            <h3>{en ? "Payment plans" : "أنظمة السداد"}</h3>
            <table className="plans">
              <thead>
                <tr>
                  <th>{en ? "Plan" : "النظام"}</th>
                  <th>{en ? "Down" : "المقدم"}</th>
                  <th>{en ? "Years" : "سنين"}</th>
                  <th>{en ? "Cash discount" : "خصم الكاش"}</th>
                </tr>
              </thead>
              <tbody>
                {u.plans.map((p: any) => (
                  <tr key={p.name}>
                    <td>{p.name}</td>
                    <td>
                      {p.down_payment_pct}% · {egp((u.price * p.down_payment_pct) / 100, lang)}
                    </td>
                    <td>{p.years}</td>
                    <td>{p.cash_discount_pct ? `${p.cash_discount_pct}%` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h3>{en ? "Amenities" : "المميزات"}</h3>
            <p className="pills">
              {u.amenities.map((a: string) => (
                <span key={a} className="pill">
                  {label(a, lang)}
                </span>
              ))}
            </p>
          </>
        )}
      </aside>
    </div>
  );
}
