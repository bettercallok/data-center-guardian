"""
tests/test_alert_engine.py — Alert lifecycle, de-duplication and memory bounds
"""
import threading

from src.monitoring.alert_engine import AlertEngine, AlertSeverity, AlertStatus, AlertType


def _fire(engine, n=1):
    return [engine.fire_novelty(novelty_score=2.0, distance=1.0, threshold=0.5) for _ in range(n)]


def test_repeat_firings_deduplicate_and_escalate():
    engine = AlertEngine()
    alerts = _fire(engine, 3)
    assert len({a.id for a in alerts}) == 1
    alert = alerts[-1]
    assert alert.occurrences == 3
    assert alert.severity == AlertSeverity.CRITICAL
    assert len(engine.get_active_alerts()) == 1


def test_active_alert_is_never_deescalated():
    engine = AlertEngine()
    _fire(engine, 3)
    engine.resolve_novelty()  # streak reset by a normal inference
    alert = _fire(engine, 1)[0]
    assert alert.severity == AlertSeverity.CRITICAL


def test_resolving_starts_a_fresh_alert():
    engine = AlertEngine()
    first = _fire(engine, 1)[0]
    resolved = engine.resolve_alert(first.id)
    assert resolved.status == AlertStatus.RESOLVED
    second = _fire(engine, 1)[0]
    assert second.id != first.id
    assert second.severity == AlertSeverity.WARNING


def test_types_are_tracked_independently():
    engine = AlertEngine()
    _fire(engine, 1)
    engine.fire_drift(["smart_5_raw"], [])
    types = {a.alert_type for a in engine.get_active_alerts()}
    assert types == {AlertType.LATENT_NOVELTY, AlertType.FEATURE_DRIFT}


def test_store_is_bounded():
    engine = AlertEngine(max_alerts=5)
    for _ in range(50):
        alert = _fire(engine, 1)[0]
        engine.resolve_alert(alert.id)
    assert len(engine) <= 5


def test_resolve_unknown_returns_none():
    assert AlertEngine().resolve_alert("missing") is None


def test_concurrent_firing_is_safe():
    engine = AlertEngine()
    threads = [threading.Thread(target=_fire, args=(engine, 50)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    active = engine.get_active_alerts()
    assert len(active) == 1
    assert active[0].occurrences == 400


def test_to_dict_is_json_friendly():
    import json
    alert = _fire(AlertEngine(), 1)[0]
    payload = json.dumps(alert.to_dict())
    assert "LATENT_NOVELTY" in payload
