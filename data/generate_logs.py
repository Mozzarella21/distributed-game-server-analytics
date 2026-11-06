#!/usr/bin/env python3
"""Reproducible, streaming synthetic multiplayer game-server event logs."""
import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

FIELDS = ["timestamp", "server_id", "region", "player_id", "session_id", "event_type", "latency_ms", "packet_loss_pct", "jitter_ms", "game_mode", "server_load_pct"]
REGIONS = {"asia": (65, 19), "europe": (45, 12), "na": (58, 17), "sa": (105, 30), "oceania": (82, 23)}
MODES = ["BATTLE_ROYALE", "TEAM_DEATHMATCH", "RANKED", "CO_OP"]
HOUR_WEIGHTS = [2, 1, 1, 1, 1, 2, 3, 5, 6, 6, 7, 8, 8, 9, 10, 12, 15, 20, 28, 34, 38, 35, 23, 10]
START = datetime(2026, 1, 1)
DAYS = 90


def clipped(value, low, high):
    return max(low, min(high, value))


def generate(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob("part-*.csv"))
    if existing:
        raise SystemExit(f"Output already contains {len(existing)} part files: {out}. Use a fresh directory.")
    rng = random.Random(args.seed)
    regions = list(REGIONS)
    files = []
    writers = []
    try:
        for i in range(args.parts):
            f = (out / f"part-{i:05d}.csv").open("w", newline="", encoding="utf-8", buffering=1024 * 1024)
            files.append(f)
            writer = csv.writer(f, lineterminator="\n")
            if args.header:
                writer.writerow(FIELDS)
            writers.append(writer)

        emitted = 0
        session_no = 0
        disconnected = 0
        while emitted < args.records:
            session_no += 1
            region = rng.choices(regions, weights=[32, 25, 25, 12, 6], k=1)[0]
            server_num = regions.index(region) * 20 + rng.randint(1, 20)
            server_id = f"S{server_num:03d}"
            player_id = f"P{rng.randint(1, args.players):08d}"
            session_id = f"SS{session_no:010d}"
            mode = rng.choice(MODES)
            day = rng.randrange(DAYS)
            hour = rng.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]
            start = START + timedelta(days=day, hours=hour, minutes=rng.randrange(60), seconds=rng.randrange(60))
            peak = 18 <= hour <= 22
            load = clipped(rng.gauss(78 if peak else 43, 11), 10, 99)
            degraded = rng.random() < (0.10 if peak else 0.035)
            base, spread = REGIONS[region]
            # Session-level network conditions persist across related events.
            session_penalty = (rng.uniform(75, 155) if degraded else 0) + max(0, load - 65) * 1.1
            loss_baseline = 0.15 + (rng.uniform(2.0, 5.0) if degraded else 0) + max(0, load - 75) * 0.035
            # Each session has LOGIN, MATCH_START, optional activity, and one terminal event.
            n = min(rng.randint(4, 9), args.records - emitted)
            if n == 1:
                events = ["LOGOUT"]
            elif n == 2:
                events = ["LOGIN", "LOGOUT"]
            else:
                events = ["LOGIN", "MATCH_START"]
                events += [rng.choice(["KILL", "DEATH", "HEARTBEAT", "MATCH_EVENT"]) for _ in range(n - 3)]
                events += ["LOGOUT"]
            # Higher packet loss, latency and load increase disconnection probability.
            disconnect_probability = clipped(0.008 + loss_baseline * 0.035 + max(0, session_penalty - 50) * 0.00035 + max(0, load - 85) * 0.001, 0, 0.45)
            if n >= 2 and rng.random() < disconnect_probability:
                events[-1] = "DISCONNECT"
                disconnected += 1
            writer = writers[(session_no - 1) % args.parts]
            step = rng.randint(30, 120)
            for idx, event in enumerate(events):
                timestamp = start + timedelta(seconds=idx * step)
                latency = clipped(rng.gauss(base + session_penalty, spread), 5, 1000)
                loss = clipped(rng.gauss(loss_baseline, 0.20 + loss_baseline * 0.12), 0, 30)
                jitter = clipped(rng.gauss(5 + loss * 5 + latency * 0.07, 4), 0, 300)
                writer.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S"), server_id, region.upper(), player_id,
                                 session_id, event, round(latency, 2), round(loss, 2), round(jitter, 2), mode, round(load, 2)])
            emitted += n
            if args.progress and (emitted // args.progress) != ((emitted - n) // args.progress):
                print(f"Generated {emitted:,}/{args.records:,} records", flush=True)
        print(f"Complete: {emitted:,} records, {session_no:,} sessions, {disconnected:,} disconnects, {args.parts} files in {out}")
    finally:
        for f in files:
            f.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=int, default=10_000_000)
    p.add_argument("--parts", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--players", type=int, default=250_000)
    p.add_argument("--output", default="game_logs")
    p.add_argument("--header", action="store_true", help="Include header in each part (off by default for Hadoop Streaming)")
    p.add_argument("--progress", type=int, default=1_000_000)
    args = p.parse_args()
    if args.records < 1 or args.parts < 1 or args.players < 1 or args.progress < 1:
        p.error("records, parts, players and progress must be positive")
    generate(args)


if __name__ == "__main__":
    main()
