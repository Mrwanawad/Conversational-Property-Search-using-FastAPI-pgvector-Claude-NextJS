import { expect, test, type Page } from "@playwright/test";

async function ask(page: Page, text: string) {
  const before = await page.getByTestId("assistant-msg").count();
  await page.getByTestId("composer").fill(text);
  await page.getByTestId("composer").press("Enter");
  await expect(page.getByTestId("assistant-msg")).toHaveCount(before + 1);
  return page.getByTestId("assistant-msg").last();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
});

test("Arabic query: filters understood, units returned in Arabic, RTL", async ({ page }) => {
  const turn = await ask(page, "3 غرف في التجمع تحت 8 مليون تقسيط 8 سنين");
  await expect(turn).toHaveAttribute("dir", "rtl");
  const filters = turn.getByTestId("filters");
  await expect(filters).toContainText("القاهرة الجديدة");
  await expect(filters).toContainText("3 غرف");
  await expect(turn.getByTestId("unit-card").first()).toBeVisible();
  await expect(turn.getByTestId("reply")).toContainText("لقيت");
});

test("English query with soft preference returns grounded cards", async ({ page }) => {
  const turn = await ask(page, "villa in Sheikh Zayed, ready to move, garden");
  await expect(turn).toHaveAttribute("dir", "ltr");
  await expect(turn.getByTestId("filters")).toContainText("Sheikh Zayed");
  await expect(turn.getByTestId("filters")).toContainText("Villa");
  const card = turn.getByTestId("unit-card").first();
  await expect(card).toBeVisible();
  await expect(card).toContainText("EGP");
});

test("Franco-Arabic query is parsed", async ({ page }) => {
  const turn = await ask(page, "3ayez sha2a fel tagamo3 ta7t 7 mlyon");
  await expect(turn.getByTestId("filters")).toContainText(/New Cairo|القاهرة الجديدة/);
  await expect(turn.getByTestId("unit-card").first()).toBeVisible();
});

test("ambiguous area asks a clarifying question, picking an option searches", async ({ page }) => {
  const turn = await ask(page, "apartment in zayed");
  const chips = turn.getByTestId("clarification");
  await expect(chips).toBeVisible();
  await chips.getByRole("button", { name: "Sheikh Zayed" }).click();
  const next = page.getByTestId("assistant-msg").last();
  await expect(page.getByTestId("assistant-msg")).toHaveCount(2);
  await expect(next.getByTestId("filters")).toContainText("Sheikh Zayed");
});

test("follow-up refinement keeps context", async ({ page }) => {
  await ask(page, "apartment in New Cairo under 10 million");
  const turn = await ask(page, "cheaper");
  await expect(turn.getByTestId("filters")).toContainText("New Cairo");
  await expect(turn.getByTestId("filters")).toContainText("Apartment");
});

test("impossible query explains instead of returning nothing", async ({ page }) => {
  const turn = await ask(page, "villa in new cairo under 2 million with private pool");
  await expect(turn.getByTestId("reply")).not.toBeEmpty();
  const relaxed = await turn.getByTestId("relaxations").count();
  const cards = await turn.getByTestId("unit-card").count();
  const explained = /cheapest|No units|أرخص/.test(await turn.getByTestId("reply").innerText());
  expect(relaxed > 0 || cards > 0 || explained).toBeTruthy();
});

test("unit details drawer and inquiry", async ({ page }) => {
  const turn = await ask(page, "apartment in New Cairo");
  const card = turn.getByTestId("unit-card").first();
  await card.getByRole("button", { name: "Details" }).click();
  const drawer = page.getByTestId("unit-drawer");
  await expect(drawer).toBeVisible();
  await expect(drawer).toContainText("Payment plans");
  await page.keyboard.press("Escape");
  await expect(drawer).toBeHidden();
  await card.getByRole("button", { name: "Inquire" }).click();
  await expect(card.getByRole("button", { name: "Request sent" })).toBeDisabled();
});

test("insights page renders KPIs", async ({ page }) => {
  await ask(page, "apartment in New Cairo");
  await page.goto("/insights");
  await expect(page.getByTestId("kpis")).toContainText("Zero-result rate");
});
