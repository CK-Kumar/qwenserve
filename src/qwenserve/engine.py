import queue
import threading

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
        # send each new token right away, for streaming
        for r in step(engine):
            r["mailbox"].put((r["text"][-1], r["finish_reason"]))


def take(engine, order):
    try:
        r = add_request(engine, order["ids"], order["max_new_tokens"], order["temperature"], order["top_k"], order["top_p"], order["seed"])
    except ValueError as e:
        order["mailbox"].put(e)  # too long for the pool: send the error back, keep running
        return
    r["mailbox"] = order["mailbox"]
