from __future__ import annotations


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["embedder"]


def test_meta_lists_areas_and_examples(client):
    body = client.get("/api/meta").json()
    assert any(a["slug"] == "new_cairo" for a in body["areas"])
    assert body["examples"]


def test_search_and_session(client):
    r = client.post("/api/search", json={"message": "apartment in New Cairo under 8 million"})
    assert r.status_code == 200
    body = r.json()
    assert body["units"] and body["session_id"]
    r2 = client.post("/api/search", json={"message": "cheaper", "session_id": body["session_id"]})
    assert r2.json()["parsed"]["intent"] == "refine"


def test_search_validates_input(client):
    assert client.post("/api/search", json={"message": ""}).status_code == 422
    assert client.post("/api/search", json={"message": "x" * 501}).status_code == 422


def test_unit_detail_and_404(client):
    hit = client.post("/api/search", json={"message": "villa in October"}).json()["units"][0]
    u = client.get(f"/api/units/{hit['id']}").json()
    assert u["code"] == hit["code"] and u["price"] == hit["price"] and u["plans"]
    assert client.get("/api/units/999999").status_code == 404


def test_events_check_turn_ownership(client):
    body = client.post("/api/search", json={"message": "villa in October"}).json()
    ev = {
        "session_id": body["session_id"],
        "turn_id": body["turn_id"],
        "unit_id": body["units"][0]["id"],
        "kind": "click",
        "position": 1,
    }
    assert client.post("/api/events", json=ev).status_code == 201
    assert client.post("/api/events", json={**ev, "session_id": "someone-else"}).status_code == 404


def test_insights(client):
    body = client.get("/api/insights?days=7").json()
    assert isinstance(body, dict) and body


def test_hosted_postgres_url_gets_psycopg_driver():
    from propsearch.config import Settings

    for raw in ("postgres://u:p@h:5432/db", "postgresql://u:p@h:5432/db"):
        assert Settings(database_url=raw).database_url == "postgresql+psycopg://u:p@h:5432/db"
    already = "postgresql+psycopg://u:p@h/db"
    assert Settings(database_url=already).database_url == already
