import queue
import threading

from qwenserve.kv_cache import free
from qwenserve.model import add_request, step

# Runs step() forever in its own thread. Requests come in through engine["rail"],
# tokens go back through each request's mailbox.


def start(engine):
    engine["rail"] = queue.Queue()  # new requests from the server
    # daemon: stops when the server stops
    threading.Thread(target=cook, args=(engine,), daemon=True).start()


def cook(engine):
    rail = engine["rail"]
    while True:
        # nothing to do: wait here for a request
        if not engine["running"] and not engine["waiting"]:
            take(engine, rail.get())
        # take the rest that came in meanwhile
        while not rail.empty():
            take(engine, rail.get())
        try:
            stepped = step(engine)
        except Exception as e:
            # a crash here would kill this thread and every request would hang
            fail_running(engine, e)
            continue
        # send each new token right away, for streaming
        for r in stepped:
            r["mailbox"].put((r["text"][-1], r["finish_reason"]))


def fail_running(engine, error):
    # We can't tell which user broke the step, so all of them get the error.
    for r in engine["running"]:
        if r["finish_reason"] is None:  # finished ones already gave their pages back
            free(engine["pool"], r["user"])
            engine["reserved"] -= r["need"]
        r["mailbox"].put(error)
    engine["running"] = []


def take(engine, order):
    try:
        r = add_request(engine, order["ids"], order["max_new_tokens"], order["temperature"], order["top_k"], order["top_p"], order["seed"])
    except ValueError as e:
        order["mailbox"].put(e)  # too long for the pool: send the error back, keep running
        return
    r["mailbox"] = order["mailbox"]

