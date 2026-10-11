#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Load test for Espetero's HTTP server (src/http). Python 3 stdlib only.

Starts the test server (`http_test --serve 0`, built from tests/http_test.cpp)
unless --port points at a running one, then measures:

  1. 64 concurrent keep-alive clients x 2,000 small requests each
     (alternating GET /hello and POST /echo with a small JSON body):
     requests/s and p50/p99/max latency, every response checked.
  2. 32 concurrent SSE streams of 200 events 5 ms apart: every event arrives
     in order followed by [DONE]; time to first event, stream duration.

Python threads share one GIL, so a single process would mostly measure the
interpreter: the clients are spread over worker processes (--procs), each
running its share of client threads.

  python3 tests/http_load.py --server build/http_test
  python3 tests/http_load.py --port 8080
"""
import argparse
import http.client
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time

HOST = "127.0.0.1"
ECHO_BODY = b'{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}'


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = round(p / 100.0 * (len(sorted_vals) - 1))
    return sorted_vals[min(len(sorted_vals) - 1, max(0, k))]


def cpu_seconds(pid):
    """CPU time (user + system) of a process, from ps; None if unavailable."""
    try:
        out = subprocess.run(["ps", "-o", "cputime=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        secs = 0.0
        for part in out.split(":"):  # [[hh:]mm:]ss.xx
            secs = secs * 60 + float(part)
        return secs
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- small requests

def _client(port, n_requests, connected, start, results):
    """One keep-alive client: n_requests sequential requests on one connection."""
    conn = http.client.HTTPConnection(HOST, port, timeout=30)
    conn.connect()
    connected.wait()
    start.wait()
    lat, errors, reconnects = [], 0, 0
    for i in range(n_requests):
        t0 = time.perf_counter()
        try:
            if i & 1:
                conn.request("POST", "/echo", body=ECHO_BODY, headers={"Content-Type": "application/json"})
            else:
                conn.request("GET", "/hello")
            resp = conn.getresponse()
            data = resp.read()
            ok = resp.status == 200 and (data.startswith(b'{"method":"POST"') if i & 1 else data == b"hello")
            if resp.will_close:  # keep-alive must hold for the whole run
                ok = False
        except (OSError, http.client.HTTPException):
            ok = False
            conn.close()
            conn = http.client.HTTPConnection(HOST, port, timeout=30)
            reconnects += 1
        lat.append(time.perf_counter() - t0)
        errors += not ok
    conn.close()
    results.append((lat, errors, reconnects, time.monotonic()))


def _worker(port, n_threads, n_requests, ready, go, out):
    """A worker process running n_threads clients; reports their latencies."""
    connected = threading.Barrier(n_threads + 1)
    start = threading.Event()
    results = []
    threads = [threading.Thread(target=_client, args=(port, n_requests, connected, start, results))
               for _ in range(n_threads)]
    for t in threads:
        t.start()
    connected.wait()  # every connection of this process is open
    ready.wait()      # ...and of every other process
    go.wait()
    start.set()
    for t in threads:
        t.join()
    lat = [x for r in results for x in r[0]]
    out.put((lat, sum(r[1] for r in results), sum(r[2] for r in results), max(r[3] for r in results)))


def run_small(port, clients, requests, procs, server_pid):
    procs = max(1, min(procs, clients))
    shares = [clients // procs + (1 if i < clients % procs else 0) for i in range(procs)]
    ctx = mp.get_context("spawn")
    ready, go, out = ctx.Barrier(procs + 1), ctx.Event(), ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(port, n, requests, ready, go, out)) for n in shares]
    for w in workers:
        w.start()
    ready.wait()
    cpu0 = cpu_seconds(server_pid) if server_pid else None
    t0 = time.monotonic()
    go.set()
    parts = [out.get() for _ in workers]
    for w in workers:
        w.join()
    wall = max(p[3] for p in parts) - t0
    cpu1 = cpu_seconds(server_pid) if server_pid else None
    lat = sorted(x for p in parts for x in p[0])
    errors = sum(p[1] for p in parts)
    reconnects = sum(p[2] for p in parts)
    total = clients * requests
    print(f"small requests: {clients} keep-alive clients x {requests} = {total} requests "
          f"({procs} client processes)")
    print(f"  {total / wall:,.0f} req/s over {wall:.2f} s; latency p50 {percentile(lat, 50) * 1e3:.3f} ms, "
          f"p99 {percentile(lat, 99) * 1e3:.3f} ms, max {lat[-1] * 1e3:.2f} ms")
    if cpu0 is not None and cpu1 is not None:
        used = cpu1 - cpu0
        print(f"  server CPU {used:.2f} s = {used / total * 1e6:.1f} us/request "
              f"({used / wall * 100:.0f}% of one core; the Python clients are the bottleneck)")
    print(f"  errors {errors}, reconnects {reconnects}")
    return errors == 0 and len(lat) == total


# ---------------------------------------------------------------- SSE streams

def _stream(port, n_events, interval_ms, results):
    t0 = time.perf_counter()
    first, events, done, in_order = None, 0, False, True
    try:
        conn = http.client.HTTPConnection(HOST, port, timeout=30)
        conn.request("GET", f"/sse?n={n_events}&ms={interval_ms}")
        resp = conn.getresponse()
        in_order = (resp.status == 200 and resp.getheader("Content-Type") == "text/event-stream"
                    and resp.getheader("Transfer-Encoding") == "chunked")
        while True:
            line = resp.readline()  # http.client undoes the chunked encoding
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            if first is None:
                first = time.perf_counter() - t0
            if line == b"data: [DONE]\n":
                done = True
            else:
                in_order &= line == b'data: {"i":%d}\n' % events
                events += 1
        conn.close()
    except (OSError, http.client.HTTPException):
        in_order = False
    results.append((in_order and done and events == n_events, events, first, time.perf_counter() - t0))


def run_sse(port, streams, n_events, interval_ms):
    results = []
    threads = [threading.Thread(target=_stream, args=(port, n_events, interval_ms, results)) for _ in range(streams)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    complete = sum(r[0] for r in results)
    events = sum(r[1] for r in results)
    firsts = sorted(r[2] for r in results if r[2] is not None)
    durations = sorted(r[3] for r in results)
    print(f"SSE: {streams} concurrent streams x {n_events} events, {interval_ms} ms apart")
    print(f"  {complete}/{streams} complete and in order, {events} events in {wall:.2f} s "
          f"({events / wall:,.0f} events/s)")
    if firsts:
        print(f"  first event p50 {percentile(firsts, 50) * 1e3:.2f} ms, max {firsts[-1] * 1e3:.2f} ms; "
              f"stream duration p50 {percentile(durations, 50):.3f} s, max {durations[-1]:.3f} s "
              f"(handler sleeps alone: >= {(n_events - 1) * interval_ms / 1e3:.3f} s)")
    return complete == streams


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", help="http_test binary to start with --serve 0")
    ap.add_argument("--port", type=int, help="port of an already running `http_test --serve`")
    ap.add_argument("--clients", type=int, default=64)
    ap.add_argument("--requests", type=int, default=2000, help="requests per client")
    ap.add_argument("--procs", type=int, default=min(8, os.cpu_count() or 1), help="client processes")
    ap.add_argument("--streams", type=int, default=32)
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--interval-ms", type=int, default=5)
    args = ap.parse_args()

    server = None
    if args.port:
        port = args.port
    elif args.server:
        server = subprocess.Popen([args.server, "--serve", "0"], stdout=subprocess.PIPE, text=True)
        line = server.stdout.readline()  # "listening on <port>"
        if not line.startswith("listening on "):
            server.kill()
            sys.exit(f"server did not start: {line!r}")
        port = int(line.split()[-1])
    else:
        ap.error("give --server <http_test binary> or --port <port>")

    try:
        ok_small = run_small(port, args.clients, args.requests, args.procs, server.pid if server else None)
        ok_sse = run_sse(port, args.streams, args.events, args.interval_ms)
    finally:
        if server:
            server.terminate()  # SIGTERM -> Server::stop() -> graceful exit
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    ok = ok_small and ok_sse
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
