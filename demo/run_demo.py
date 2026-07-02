#!/usr/bin/env python3
"""
demo/run_demo.py — Data Center Guardian Demo

Demonstrates the full monitoring stack by:
  1. Sending 50 normal, healthy-drive payloads
  2. Sending 50 severely degraded, drifted payloads
  3. Polling /api/models/alerts and /api/models/health in real time
  4. Printing a live summary to the terminal

Usage:
  python demo/run_demo.py [--url http://localhost:8000]

Watch the alerts fire in your terminal as the drifted data crosses the
novelty and drift thresholds.
"""
import argparse
import json
import random
import time
import sys
import os

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is not installed. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# ANSI colors for terminal output
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def color_risk(risk: str) -> str:
    colors = {"low": GREEN, "medium": YELLOW, "high": RED, "critical": RED + BOLD}
    return colors.get(risk, RESET) + risk.upper() + RESET


def color_novelty(is_novel: bool, score: float) -> str:
    if is_novel:
        return f"{RED}{BOLD}NOVEL ({score:.2f}){RESET}"
    return f"{GREEN}normal ({score:.2f}){RESET}"


def color_alert_type(alert_type: str) -> str:
    if alert_type == "LATENT_NOVELTY":
        return f"{CYAN}{alert_type}{RESET}"
    return f"{YELLOW}{alert_type}{RESET}"


def color_severity(severity: str) -> str:
    if severity == "CRITICAL":
        return f"{RED}{BOLD}{severity}{RESET}"
    return f"{YELLOW}{severity}{RESET}"


# ---------------------------------------------------------------------------
# Payload generators
# ---------------------------------------------------------------------------

def healthy_payload(i: int) -> dict:
    """Simulate a healthy drive: all SMART values low."""
    return {
        "smart_5_raw":   random.randint(0, 5),
        "smart_187_raw": random.randint(0, 2),
        "smart_188_raw": random.randint(0, 3),
        "smart_197_raw": random.randint(0, 4),
        "smart_198_raw": random.randint(0, 3),
    }


def drifted_payload(i: int) -> dict:
    """Simulate a dying drive: all SMART values severely degraded."""
    return {
        "smart_5_raw":   random.randint(200, 500),
        "smart_187_raw": random.randint(80, 200),
        "smart_188_raw": random.randint(500, 1000),
        "smart_197_raw": random.randint(100, 500),
        "smart_198_raw": random.randint(100, 500),
    }


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(base_url: str, n_normal: int = 50, n_drifted: int = 50, delay: float = 0.1):
    predict_url = f"{base_url}/api/models/predict"
    health_url  = f"{base_url}/api/models/health"
    alerts_url  = f"{base_url}/api/models/alerts"

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  DATA CENTER GUARDIAN — LIVE DRIFT DEMO{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  Backend: {CYAN}{base_url}{RESET}")
    print(f"  Normal payloads:  {n_normal}")
    print(f"  Drifted payloads: {n_drifted}")
    print(f"{'=' * 60}\n")

    # ---------------------------------------------------------------------------
    # Phase 1: Healthy inferences
    # ---------------------------------------------------------------------------
    print(f"{GREEN}{BOLD}[PHASE 1] Sending {n_normal} healthy drive telemetry payloads...{RESET}\n")
    total_alerts_before = 0

    for i in range(n_normal):
        payload = healthy_payload(i)
        try:
            resp = requests.post(predict_url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            novel_str = color_novelty(
                data.get("is_novel", False),
                data.get("novelty_score") or 0.0
            )
            print(
                f"  [{i+1:02d}/{n_normal}] TTF={data['ttf_days']:>7.1f}d "
                f"risk={color_risk(data['risk_level']):<20} "
                f"novelty={novel_str}"
            )
        except requests.RequestException as e:
            print(f"  {RED}[{i+1}] Request failed: {e}{RESET}")
        time.sleep(delay)

    # ---------------------------------------------------------------------------
    # Phase 2: Drifted inferences
    # ---------------------------------------------------------------------------
    print(f"\n{RED}{BOLD}[PHASE 2] Injecting {n_drifted} SEVERELY DRIFTED payloads...{RESET}\n")

    fired_alert_ids = set()

    for i in range(n_drifted):
        payload = drifted_payload(i)
        try:
            resp = requests.post(predict_url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            novel_str = color_novelty(
                data.get("is_novel", False),
                data.get("novelty_score") or 0.0
            )
            new_alerts = data.get("fired_alert_ids", [])
            for aid in new_alerts:
                fired_alert_ids.add(aid)

            alert_str = ""
            if new_alerts:
                alert_str = f" {RED}{BOLD}⚠ ALERT FIRED{RESET}"

            print(
                f"  [{i+1:02d}/{n_drifted}] TTF={data['ttf_days']:>7.1f}d "
                f"risk={color_risk(data['risk_level']):<20} "
                f"novelty={novel_str}{alert_str}"
            )
        except requests.RequestException as e:
            print(f"  {RED}[{i+1}] Request failed: {e}{RESET}")
        time.sleep(delay)

    # ---------------------------------------------------------------------------
    # Final: Poll health and alerts
    # ---------------------------------------------------------------------------
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  FINAL STATE{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    try:
        health = requests.get(health_url, timeout=10).json()
        print(f"\n  Model Status: {BOLD}{health.get('status', 'unknown').upper()}{RESET}")
        print(f"  Total Inferences: {health.get('total_inferences', 0)}")

        faiss = health.get("faiss", {})
        print(f"\n  FAISS Index:")
        print(f"    Built:     {GREEN if faiss.get('is_built') else RED}{faiss.get('is_built')}{RESET}")
        print(f"    Threshold: {faiss.get('threshold', 'N/A')}")

        alerts_summary = health.get("alerts", {})
        print(f"\n  Active Alerts:")
        print(f"    Total:           {alerts_summary.get('total_active', 0)}")
        print(f"    LATENT_NOVELTY:  {alerts_summary.get('latent_novelty', 0)}")
        print(f"    FEATURE_DRIFT:   {alerts_summary.get('feature_drift', 0)}")
        print(f"    CRITICAL:        {alerts_summary.get('critical_count', 0)}")

        drift = health.get("drift")
        if drift:
            print(f"\n  Drift Report (window={drift.get('window_size')} samples):")
            for fs in drift.get("feature_scores", []):
                sev_color = RED if fs["severity"] == "major" else (YELLOW if fs["severity"] == "minor" else GREEN)
                print(
                    f"    {fs['feature']:<20} PSI={fs['psi']:.4f}  "
                    f"KS={fs['ks_statistic']:.4f}  "
                    f"severity={sev_color}{fs['severity']}{RESET}"
                )
    except requests.RequestException as e:
        print(f"  {RED}Health check failed: {e}{RESET}")

    # Fetch full active alerts
    try:
        alerts = requests.get(alerts_url, timeout=10).json()
        if alerts:
            print(f"\n  Active Alert Log:")
            for a in alerts[:10]:  # show first 10
                print(
                    f"    [{color_alert_type(a['alert_type'])}] "
                    f"[{color_severity(a['severity'])}] "
                    f"{DIM}{a['message'][:80]}...{RESET}"
                )
        else:
            print(f"\n  {GREEN}No active alerts.{RESET}")
    except requests.RequestException as e:
        print(f"  {RED}Alert fetch failed: {e}{RESET}")

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"  Demo complete. {len(fired_alert_ids)} alert(s) fired during demo.")
    print(f"  View full docs at: {CYAN}{base_url}/docs{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Center Guardian drift demo")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--normal", type=int, default=50, help="Number of normal payloads")
    parser.add_argument("--drifted", type=int, default=50, help="Number of drifted payloads")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between requests (seconds)")
    args = parser.parse_args()

    run_demo(
        base_url=args.url,
        n_normal=args.normal,
        n_drifted=args.drifted,
        delay=args.delay
    )
