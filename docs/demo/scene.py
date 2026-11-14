"""Demo GIF scene for the README. Needs the stack running on localhost:3000 (docker compose up).

    python docs/demo/scene.py
    python ~/.claude/skills/demo-gif/scripts/compose.py docs/demo/build --out docs/demo.gif
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/.claude/skills/demo-gif/scripts"))

from director import Director, run  # noqa: E402

URL = os.environ.get("DEMO_URL", "http://localhost:3000")

# PW_CHANNEL=chrome records with the installed Chrome instead of Playwright's Chromium.
if os.environ.get("PW_CHANNEL"):
    from playwright.async_api._generated import BrowserType

    _launch = BrowserType.launch

    async def _launch_channel(self, **kw):
        kw.setdefault("channel", os.environ["PW_CHANNEL"])
        return await _launch(self, **kw)

    BrowserType.launch = _launch_channel


async def scene(d: Director):
    p = d.page
    turns = p.get_by_test_id("assistant-msg")
    await d.goto(URL, settle=1.0)
    await p.get_by_role("button", name="villa in Zayed, ready to move, garden").wait_for()

    await d.caption("Ask the way buyers talk", step=1)
    await d.type(p.get_by_test_id("composer"), "3 غرف في التجمع تحت 8 مليون تقسيط 8 سنين", cps=26)
    await d.press("Enter", pause=0.1)
    await d.fast(turns.first.get_by_test_id("unit-card").first.wait_for(), label="Searching")
    await d.hold(0.5)
    # Bring the parsed filters and the first row of cards into one frame.
    await p.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    await d.hold(0.7)

    await d.caption("Dialect to exact filters, with reasons", step=2)
    await d.spotlight(turns.first.get_by_test_id("filters"), pad=8)
    await d.hold(1.1)
    await d.clear_spotlight()
    await d.focus(turns.first.get_by_test_id("unit-card").first, zoom=1.45)
    await d.hold(1.4)
    await d.unfocus(dur=0.6)

    await d.caption("Refine in one tap", step=3)
    await d.click(p.get_by_test_id("suggestions").get_by_role("button", name="أرخص شوية"))
    await d.fast(turns.nth(1).get_by_test_id("unit-card").first.wait_for(), label="Refining")
    await d.hold(0.6)

    await d.caption("Check payment plans, then inquire", step=4)
    card = turns.nth(1).get_by_test_id("unit-card").first
    await d.click(card.get_by_role("button", name="التفاصيل"))
    await p.get_by_test_id("unit-drawer").locator("table").wait_for()
    await d.hold(1.5)
    await d.click(p.get_by_role("button", name="Close"), pause=0.2)
    await d.click(card.get_by_role("button", name="استفسر"))
    await d.hold(1.6)


if __name__ == "__main__":
    run(
        scene,
        out="docs/demo/build",
        title="Conversational Property Search",
        subtitle="Arabic, Franco-Arabic and English to ranked, explained units",
        url_label="localhost:3000",
    )
