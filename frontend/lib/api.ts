// Types mirror propsearch.domain (FastAPI). Keep in sync with SearchTurnResponse.

export type Lang = "ar" | "en";

export interface Reason {
  kind: "match" | "near" | "info" | "relaxed";
  en: string;
  ar: string;
}

export interface PlanSummary {
  name: string;
  down_payment_pct: number;
  down_payment: number;
  years: number;
  monthly: number;
  cash_discount_pct: number;
}

export interface UnitHit {
  id: number;
  code: string;
  title_en: string;
  title_ar: string;
  compound: string;
  compound_ar: string;
  developer: string;
  area: string;
  area_en: string;
  area_ar: string;
  unit_type: string;
  bedrooms: number;
  bathrooms: number;
  bua_m2: number;
  garden_m2: number;
  floor: number | null;
  finishing: string;
  price: number;
  price_per_m2: number;
  delivery: string;
  status: string;
  amenities: string[];
  best_plan: PlanSummary | null;
  distance_km: number | null;
  score: number;
  reasons: Reason[];
  description_en: string;
  description_ar: string;
}

export interface Clarification {
  question_en: string;
  question_ar: string;
  options: { label_en: string; label_ar: string; value: string }[];
}

export interface Relaxation {
  field: string;
  from_value: string;
  to_value: string;
  en: string;
  ar: string;
  unlocked: number;
}

export type Filters = Record<string, string | number | boolean | string[] | null>;

export interface SearchTurn {
  session_id: string;
  turn_id: number;
  language: "ar" | "en" | "franco" | "mixed";
  reply_language: Lang;
  parsed: {
    filters: Filters;
    soft_preferences: string;
    sort: string;
    intent: string;
    parser: string;
    confidence: number;
    unknown_terms: string[];
  };
  message: string;
  clarification: Clarification | null;
  relaxations: Relaxation[];
  total_matches: number;
  units: UnitHit[];
  suggestions: { label: string; value: string }[];
  timings_ms: Record<string, number>;
}

export interface Meta {
  areas: { slug: string; name_en: string; name_ar: string }[];
  landmarks: { slug: string; name_en: string; name_ar: string }[];
  examples: string[];
  parser: string;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => call<Meta>("/api/meta"),
  search: (message: string, session_id: string | null) =>
    call<SearchTurn>("/api/search", { method: "POST", body: JSON.stringify({ message, session_id }) }),
  unit: (id: number) => call<Record<string, any>>(`/api/units/${id}`),
  event: (body: {
    session_id: string;
    turn_id: number;
    unit_id: number;
    kind: "click" | "inquiry";
    position: number;
  }) => call<{ ok: boolean }>("/api/events", { method: "POST", body: JSON.stringify(body) }).catch(() => null),
  insights: (days = 30) => call<Record<string, any>>(`/api/insights?days=${days}`),
};

// ---- formatting -----------------------------------------------------------

export function egp(v: number, lang: Lang): string {
  if (v >= 1_000_000) {
    const m = (v / 1_000_000).toFixed(2).replace(/\.?0+$/, "");
    return lang === "en" ? `${m}M EGP` : `${m} مليون جنيه`;
  }
  const k = Math.round(v / 1000).toLocaleString("en-US");
  return lang === "en" ? `${k}K EGP` : `${k} ألف جنيه`;
}

const LABELS: Record<string, [string, string]> = {
  apartment: ["Apartment", "شقة"],
  duplex: ["Duplex", "دوبلكس"],
  penthouse: ["Penthouse", "بنتهاوس"],
  studio: ["Studio", "ستوديو"],
  townhouse: ["Townhouse", "تاون هاوس"],
  twin_house: ["Twin house", "توين هاوس"],
  villa: ["Villa", "فيلا"],
  chalet: ["Chalet", "شاليه"],
  fully_finished: ["Fully finished", "متشطب بالكامل"],
  semi_finished: ["Semi-finished", "نص تشطيب"],
  core_shell: ["Core & shell", "طوب أحمر"],
  garden: ["Garden", "جنينة"],
  roof: ["Roof", "روف"],
  private_pool: ["Private pool", "حمام سباحة خاص"],
  sea_view: ["Sea view", "فيو بحر"],
  maids_room: ["Maid's room", "غرفة دادة"],
  swimming_pool: ["Pool", "حمام سباحة"],
  clubhouse: ["Clubhouse", "كلوب هاوس"],
  gym: ["Gym", "جيم"],
  kids_area: ["Kids area", "منطقة أطفال"],
  crystal_lagoon: ["Lagoon", "لاجون"],
  beach_access: ["Beach access", "على البحر"],
  golf: ["Golf", "جولف"],
  security: ["Security", "أمن"],
  parking: ["Parking", "جراج"],
};

export const label = (key: string, lang: Lang) => (LABELS[key] ? LABELS[key][lang === "en" ? 0 : 1] : key);

/** Active filters as short bilingual pills: "what the system understood". */
export function filterPills(f: Filters, meta: Meta | null, lang: Lang): string[] {
  const en = lang === "en";
  const place = (slug: string, list: Meta["areas"] | undefined) => {
    const hit = list?.find((a) => a.slug === slug);
    return hit ? (en ? hit.name_en : hit.name_ar) : slug;
  };
  const out: string[] = [];
  const arr = (k: string) => (Array.isArray(f[k]) ? (f[k] as string[]) : []);
  const num = (k: string) => (typeof f[k] === "number" ? (f[k] as number) : null);

  arr("areas").forEach((a) => out.push(place(a, meta?.areas)));
  arr("unit_types").forEach((t) => out.push(label(t, lang)));
  const bmin = num("bedrooms_min");
  const bmax = num("bedrooms_max");
  if (bmin !== null || bmax !== null) {
    const r = bmin === bmax ? `${bmin}` : bmax === null ? `${bmin}+` : bmin === null ? `≤${bmax}` : `${bmin}–${bmax}`;
    out.push(en ? `${r} bedrooms` : `${r} غرف`);
  }
  const lo = num("budget_min");
  const hi = num("budget_max");
  if (hi !== null) out.push(en ? `Under ${egp(hi, lang)}` : `تحت ${egp(hi, lang)}`);
  if (lo !== null) out.push(en ? `Over ${egp(lo, lang)}` : `فوق ${egp(lo, lang)}`);
  const dp = num("down_payment_max_pct");
  if (dp !== null) out.push(en ? `Down ≤ ${dp}%` : `مقدم ≤ ${dp}%`);
  const dpa = num("down_payment_max_amount");
  if (dpa !== null) out.push(en ? `Down ≤ ${egp(dpa, lang)}` : `مقدم ≤ ${egp(dpa, lang)}`);
  const yrs = num("installment_years_min");
  if (yrs !== null) out.push(en ? `${yrs}+ yrs installments` : `تقسيط ${yrs}+ سنين`);
  const mo = num("monthly_installment_max");
  if (mo !== null) out.push(en ? `≤ ${egp(mo, lang)}/mo` : `≤ ${egp(mo, lang)} في الشهر`);
  if (f.ready_to_move === true) out.push(en ? "Ready to move" : "استلام فوري");
  const dy = num("delivery_year_max");
  if (dy !== null) out.push(en ? `Delivery by ${dy}` : `استلام قبل ${dy}`);
  arr("finishing").forEach((x) => out.push(label(x, lang)));
  arr("amenities").forEach((x) => out.push(label(x, lang)));
  if (typeof f.near_landmark === "string") {
    const lm = place(f.near_landmark, meta?.landmarks);
    const km = num("max_distance_km");
    out.push(en ? `Near ${lm}${km ? ` (${km} km)` : ""}` : `قريب من ${lm}${km ? ` (${km} كم)` : ""}`);
  }
  if (typeof f.unit_code === "string") out.push(f.unit_code);
  return out;
}
