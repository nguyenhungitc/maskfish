#!/usr/bin/env python3
"""Smoke test of go ponder, ponderhit, go infinite and go mate over UCI.

    python3 tests/ponder_smoke.py build/maskfish [NET]

Timings are loose (a loaded machine passes); the test fails only on a missing or an early answer.
"""
import queue
import subprocess
import sys
import threading
import time


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    p = subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put((time.perf_counter(), l.strip())) for l in p.stdout], daemon=True).start()

    def send(s):
        p.stdin.write(s + "\n")
        p.stdin.flush()

    def bestmove(timeout):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            try:
                t, line = q.get(timeout=0.05)
            except queue.Empty:
                continue
            if line.startswith("bestmove"):
                return t, line
        return None, None

    failures = []

    def check(name, cond, extra=""):
        print(("pass " if cond else "FAIL ") + name + (" (" + extra + ")" if extra else ""))
        if not cond:
            failures.append(name)

    if len(sys.argv) > 2:
        send("setoption name EvalFile value " + sys.argv[2])
    send("setoption name Ponder value true")
    send("isready")
    while q.get()[1] != "readyok":
        pass

    send("position startpos")
    send("go ponder wtime 2000 btime 2000 winc 0 binc 0")
    _, line = bestmove(3.0)
    check("no answer while pondering, past the clock", line is None)
    hit = time.perf_counter()
    send("ponderhit")
    t, line = bestmove(10)
    check("answer after ponderhit, on a clock that starts at the hit", line is not None and t - hit > 0.02,
          f"{(t - hit) * 1000:.0f} ms" if line else "")

    send("position startpos")
    send("go ponder wtime 2000 btime 2000")
    time.sleep(0.3)
    send("stop")
    _, line = bestmove(10)
    check("stop ends pondering", line is not None)

    send("position startpos")
    hit = time.perf_counter()
    send("go ponder movetime 400")
    send("ponderhit")  # may arrive before the search thread starts
    t, line = bestmove(10)
    check("a ponderhit sent at once is not lost", line is not None and t - hit > 0.2, f"{(t - hit) * 1000:.0f} ms" if line else "")

    send("position fen 4k4/9/9/9/9/9/9/9/4A4/3RK4 w - 0 1")  # a mate found at once: the search runs out of depth
    send("go infinite")
    _, line = bestmove(3.0)
    check("go infinite waits for stop", line is None)
    send("stop")
    _, line = bestmove(10)
    check("stop ends go infinite", line is not None)
    check("bestmove names a move to ponder on", line is not None and " ponder " in line, line or "")

    send("position fen 4k4/9/9/9/9/9/9/9/4A4/3RK4 w - 0 1")
    send("go mate 3")
    _, line = bestmove(60)
    check("go mate answers", line is not None, line or "")

    send("quit")
    p.wait()
    if failures:
        sys.exit("ponder_smoke: FAIL: " + "; ".join(failures))
    print("ponder_smoke: pass")


if __name__ == "__main__":
    main()
