"""
tests/test_api.py — End-to-end API tests via FastAPI's TestClient
"""
import csv
import io

import pytest


class TestPrediction:

    def test_predict_returns_enriched_payload(self, client, healthy_telemetry):
        r = client.post("/api/models/predict", json=healthy_telemetry)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["risk_level"] in {"low", "medium", "high", "critical"}
        assert body["rul_days"] <= body["ttf_days"]
        assert body["id"]
        assert body["is_novel"] is False
        assert body["out_of_range_features"] == []

    def test_critical_drive_scores_lower_than_healthy(self, client, healthy_telemetry, critical_telemetry):
        healthy = client.post("/api/models/predict", json=healthy_telemetry).json()
        critical = client.post("/api/models/predict", json=critical_telemetry).json()
        assert critical["ttf_days"] < healthy["ttf_days"]

    def test_ood_input_is_flagged_and_alerts(self, client, ood_telemetry):
        r = client.post("/api/models/predict", json=ood_telemetry)
        assert r.status_code == 200
        body = r.json()
        assert body["is_novel"] is True
        assert body["novelty_score"] > 1.0
        assert set(body["out_of_range_features"]) == {
            "smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw",
        }
        assert len(body["fired_alert_ids"]) == 1

    def test_repeated_ood_does_not_create_unbounded_alerts(self, client, ood_telemetry):
        ids = set()
        for _ in range(5):
            ids.update(client.post("/api/models/predict", json=ood_telemetry).json()["fired_alert_ids"])
        # Repeat firings update the single active LATENT_NOVELTY alert
        assert len(ids) == 1
        alerts = client.get("/api/models/alerts").json()
        novelty = [a for a in alerts if a["alert_type"] == "LATENT_NOVELTY"]
        assert len(novelty) == 1
        assert novelty[0]["occurrences"] >= 5
        assert novelty[0]["severity"] == "CRITICAL"

    @pytest.mark.parametrize("bad", [
        {"smart_5_raw": -1},
        {"smart_5_raw": 70000},
        {"smart_5_raw": "lots"},
        {"serial_number": "=HYPERLINK(\"x\")"},
        {"serial_number": "x" * 100},
    ])
    def test_invalid_input_rejected(self, client, bad):
        r = client.post("/api/models/predict", json=bad)
        assert r.status_code == 422

    def test_validation_errors_do_not_echo_input(self, client):
        r = client.post("/api/models/predict", json={"smart_5_raw": "<script>alert(1)</script>"})
        assert r.status_code == 422
        assert "<script>" not in r.text

    def test_legacy_predict_endpoint_works(self, client, healthy_telemetry):
        r = client.post("/predict", json=healthy_telemetry)
        assert r.status_code == 200, r.text
        assert set(r.json()) == {"ttf_days", "rul_days", "risk_level", "log_time"}


class TestBatchAndHistory:

    def test_batch_predict(self, client, healthy_telemetry, critical_telemetry):
        drives = [
            {**healthy_telemetry, "serial_number": "BATCH-A"},
            {**critical_telemetry, "serial_number": "BATCH-B"},
        ]
        r = client.post("/api/models/predict/batch", json={"drives": drives})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["count"] == 2
        assert sum(body["risk_counts"].values()) == 2
        assert [p["serial_number"] for p in body["predictions"]] == ["BATCH-A", "BATCH-B"]

    def test_batch_rejects_oversized_and_empty(self, client, healthy_telemetry):
        assert client.post("/api/models/predict/batch", json={"drives": []}).status_code == 422
        big = {"drives": [healthy_telemetry] * 101}
        assert client.post("/api/models/predict/batch", json=big).status_code == 413

    def test_drive_history(self, client, healthy_telemetry, critical_telemetry):
        serial = "HIST-001"
        client.post("/api/models/predict", json={**healthy_telemetry, "serial_number": serial, "drive_model": "ST4000DM000"})
        client.post("/api/models/predict", json={**critical_telemetry, "serial_number": serial})
        r = client.get(f"/api/drives/{serial}/history")
        assert r.status_code == 200
        body = r.json()
        assert body["n_predictions"] == 2
        assert body["drive_model"] == "ST4000DM000"
        assert body["rul_trend_days"] < 0  # degrading
        assert client.get("/api/drives/UNKNOWN-1/history").status_code == 404

    def test_predictions_filters_and_bounds(self, client):
        assert client.get("/api/models/predictions?limit=0").status_code == 422
        assert client.get("/api/models/predictions?limit=501").status_code == 422
        assert client.get("/api/models/predictions?risk_level=bogus").status_code == 422
        r = client.get("/api/models/predictions?serial_number=HIST-001")
        assert r.status_code == 200
        assert all(p["serial_number"] == "HIST-001" for p in r.json())

    def test_summary(self, client):
        r = client.get("/api/models/predictions/summary")
        assert r.status_code == 200
        body = r.json()
        assert body["total_predictions"] == sum(body["risk_counts"].values())
        assert 0.0 <= body["novelty_rate"] <= 1.0

    def test_csv_export(self, client):
        r = client.get("/api/models/predictions/export?limit=50")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        rows = list(csv.reader(io.StringIO(r.text)))
        assert rows[0][0] == "id"
        assert len(rows) > 1


class TestMonitoring:

    def test_health(self, client):
        body = client.get("/api/models/health").json()
        assert body["model_loaded"] is True
        assert body["faiss"]["is_built"] is True
        assert body["fingerprint_id"]

    def test_alert_status_validated(self, client):
        assert client.get("/api/models/alerts?status=everything").status_code == 422
        assert client.get("/api/models/alerts?status=all").status_code == 200

    def test_drift_detected_and_summary_returned(self, client, critical_telemetry):
        """Drift summary must be returned on the drift-check cycle, including when drift IS detected."""
        summaries = []
        for _ in range(60):
            body = client.post("/api/models/predict", json=critical_telemetry).json()
            if body["drift_summary"]:
                summaries.append(body["drift_summary"])
        assert summaries, "expected at least one drift summary in 60 inferences"
        assert summaries[-1]["overall_drift_detected"] is True
        history = client.get("/api/models/drift/history").json()
        assert len(history) >= 1

    def test_healthz_and_meta(self, client):
        assert client.get("/healthz").json() == {"status": "ok", "model_loaded": True}
        meta = client.get("/api").json()
        assert meta["admin_enabled"] is True
        assert meta["model_sha256"]


class TestSecurity:

    def test_probe_requires_admin_key(self, client):
        assert client.post("/api/models/probe").status_code == 401
        assert client.post("/api/models/probe", headers={"X-API-Key": "wrong"}).status_code == 401

    def test_probe_with_admin_key(self, client, admin_headers):
        r = client.post("/api/models/probe?n_samples=200", headers=admin_headers)
        assert r.status_code == 200, r.text
        assert r.json()["wasserstein_to_previous"] is not None

    def test_probe_sample_bounds(self, client, admin_headers):
        assert client.post("/api/models/probe?n_samples=99999", headers=admin_headers).status_code == 422

    def test_resolve_alert_requires_admin(self, client, ood_telemetry, admin_headers):
        alert_id = client.post("/api/models/predict", json=ood_telemetry).json()["fired_alert_ids"][0]
        assert client.delete(f"/api/models/alerts/{alert_id}").status_code == 401
        r = client.delete(f"/api/models/alerts/{alert_id}", headers=admin_headers)
        assert r.status_code == 200
        active_ids = [a["id"] for a in client.get("/api/models/alerts").json()]
        assert alert_id not in active_ids
        assert client.delete("/api/models/alerts/not-a-uuid", headers=admin_headers).status_code == 422

    def test_admin_disabled_when_key_unset(self, client, monkeypatch):
        import src.api.security as security
        monkeypatch.setattr(security, "_configured_admin_key", lambda: "")
        r = client.post("/api/models/probe", headers={"X-API-Key": ""})
        assert r.status_code == 503

    def test_security_headers(self, client):
        r = client.get("/api/models/health")
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
        assert r.headers["Cache-Control"] == "no-store"
        assert r.headers["X-Request-ID"]

    def test_request_id_is_sanitised(self, client):
        r = client.get("/api", headers={"X-Request-ID": "bad id\r\nX-Injected: 1"})
        assert r.headers["X-Request-ID"] != "bad id\r\nX-Injected: 1"
        assert "X-Injected" not in r.headers
        good = client.get("/api", headers={"X-Request-ID": "abc-123"})
        assert good.headers["X-Request-ID"] == "abc-123"

    def test_cors_rejects_unknown_origin(self, client):
        r = client.options(
            "/api/models/predict",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}

    def test_csv_formula_injection_neutralised(self):
        from src.api.routes_inference import _csv_safe
        assert _csv_safe("=cmd()") == "'=cmd()"
        assert _csv_safe("@SUM(A1)") == "'@SUM(A1)"
        assert _csv_safe("-12.5") == "-12.5"
        assert _csv_safe(None) == ""
