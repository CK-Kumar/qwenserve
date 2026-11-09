import queue
import threading

from qwenserve.model import add_request, step

# The cook: one thread that runs step() forever, so requests that come in at
# different times still share the same rounds. The web side only talks to it
# through queues, so only this thread ever touches the engine.


def start(engine):
    engine["rail"] = queue.Queue()  # new orders from the web side
    # daemon: the thread stops when the server stops
    threading.Thread(target=cook, args=(engine,), daemon=True).start()


def cook(engine):
    rail = engine["rail"]
    while True:
        # kitchen empty: sleep here until an order comes, instead of spinning
        if not engine["running"] and not engine["waiting"]:
            take(engine, rail.get())
        # and take every other order that came in meanwhile
        while not rail.empty():
            take(engine, rail.get())
        # every request gets its new token right away, so the web side can stream it
        for r in step(engine):
            r["mailbox"].put((r["text"][-1], r["finish_reason"]))


def take(engine, order):
    try:
        r = add_request(engine, order["ids"], order["max_new_tokens"], order["temperature"], order["top_k"], order["top_p"], order["seed"])
    except ValueError as e:
        order["mailbox"].put(e)  # too big for the pool: tell the web side, don't let the cook die
        return
    r["mailbox"] = order["mailbox"]
