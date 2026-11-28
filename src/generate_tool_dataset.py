"""Synthetic tool calling, in the same {q_str, a_str} format as the arithmetic datasets.

Every prompt lists four tool signatures and one user request; the answer is the
call, arguments in signature order:

    Tools: get_weather(city); book_train(origin, destination); set_timer(minutes); translate(word, language)
    Request: I want to take the train to Lima from Oslo.
    Call:                                        -> " book_train(origin: Lima ... )"  (gold: origin: Oslo, destination: Lima)

What a model has to get right, and where a small one fails:
  - which tool: the request never names it, and listed tools share parameter
    types (get_weather / get_time both take a city);
  - which value goes in which slot: templates state the values in a different
    order from the signature (origin/destination, amount/percent), and some add a
    distractor of the same type ("I'm in Boston but heading to Denver").
Values are copied verbatim from the request, so no outside knowledge is needed.

Two datasets are written:
  <name>           30 tools; train and test share the tools (new requests in test).
  <name>_unseen    10 held-out tools never seen in training (distractor tools drawn
                   from all 40), the generalisation axis: the call has to be read off
                   the listed signature, not recalled.
Both carry meta.json answer_type "call": eval_model grades by normalised exact match
of the first line (utils.extract_text_answer) and decodes up to CALL_EVAL_TOKENS.

The ": " argument separator is deliberate: with "=" the BPE pre-tokeniser glues the
separator to the value ("=Oslo"), and the first token of a value would no longer be
a clean decision token for the graph (experiments/graph_distance.py --decision-tokens).

    python generate_tool_dataset.py --name toolcall --train 5000 --test 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DIR_ROOT  # noqa: E402

POOLS = {
    "city": "Paris Tokyo Denver Boston Madrid Cairo Lima Oslo Delhi Seoul Rome Austin Dublin Nairobi Lisbon Toronto "
            "Sydney Berlin Chicago Miami Vienna Prague Bangkok Havana Quito Athens Zurich Manila Dallas Phoenix".split(),
    "person": "Alice Bob Carol David Emma Frank Grace Henry Irene Jack Kara Leo Maya Nina Omar Paul Quinn Rosa Sam "
              "Tara Uma Victor Wendy Yusuf Zoe".split(),
    "num": [str(n) for n in range(2, 100)],
    "day": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
    "time": [f"{h}{m}" for h in range(1, 13) for m in ("am", "pm")],
    "currency": "USD EUR JPY GBP INR CAD AUD CHF MXN BRL".split(),
    "item": "apples bread milk eggs coffee rice butter cheese pasta tea soap batteries onions yogurt".split(),
    "color": "red blue green white purple orange yellow".split(),
    "room": "kitchen bedroom garage office hallway basement attic patio".split(),
    "artist": "Adele Drake Shakira Coldplay Queen Beyonce Metallica Rihanna Eminem Sade".split(),
    "language": "French Spanish German Japanese Hindi Arabic Korean Italian Russian Swahili".split(),
    "word": "hello thanks goodbye water friend house morning book train music".split(),
    "ticker": "AAPL MSFT TSLA AMZN NVDA GOOG META IBM ORCL INTC".split(),
    "subject": "budget lunch invoice vacation report contract schedule party".split(),
    "cuisine": "Thai Mexican Italian Indian Greek Korean Ethiopian Vietnamese".split(),
    "topic": "sports politics science business health technology".split(),
    "task": "stretch hydrate meditate journal vacuum water-plants".split(),
}

# name: ([(param, pool), ...], [request templates]). "{_param}" is a distractor value
# from the same pool as param, different from every value in the call.
TRAIN_TOOLS = {
    "get_weather": ([("city", "city")], [
        "What's the weather in {city}?", "Is it going to rain in {city} today?",
        "I'm in {_city} but heading to {city} - how's the weather there?"]),
    "get_time": ([("city", "city")], [
        "What time is it in {city}?", "Current local time in {city}, please.",
        "I live in {_city}; my friend lives in {city}. What time is it for my friend?"]),
    "book_flight": ([("origin", "city"), ("destination", "city"), ("day", "day")], [
        "Book a flight from {origin} to {destination} on {day}.",
        "I need to fly to {destination} on {day}, leaving from {origin}.",
        "On {day}, get me from {origin} to {destination} by plane."]),
    "book_train": ([("origin", "city"), ("destination", "city")], [
        "Get me a train ticket from {origin} to {destination}.",
        "I want to take the train to {destination} from {origin}."]),
    "book_hotel": ([("city", "city"), ("nights", "num")], [
        "Reserve a hotel in {city} for {nights} nights.", "I need a room for {nights} nights in {city}."]),
    "send_email": ([("to", "person"), ("subject", "subject")], [
        "Email {to} about the {subject}.", "Write to {to}, subject {subject}.",
        "Send the {subject} email to {to}, not {_to}."]),
    "call_contact": ([("name", "person")], [
        "Call {name}.", "Ring {name} for me.", "Can you phone {name}? Not {_name}, {name}."]),
    "set_timer": ([("minutes", "num")], [
        "Set a timer for {minutes} minutes.", "Remind me in {minutes} minutes.", "Start a {minutes} minute countdown."]),
    "set_alarm": ([("time", "time"), ("day", "day")], [
        "Wake me up at {time} on {day}.", "Set an alarm for {day} at {time}."]),
    "convert_currency": ([("amount", "num"), ("source", "currency"), ("target", "currency")], [
        "Convert {amount} {source} to {target}.", "How much is {amount} {source} in {target}?",
        "What do I get in {target} for {amount} {source}?"]),
    "transfer_money": ([("amount", "num"), ("recipient", "person")], [
        "Send {amount} dollars to {recipient}.", "Pay {recipient} {amount} dollars.",
        "Transfer {amount} dollars to {recipient}, not {_recipient}."]),
    "add_to_cart": ([("item", "item"), ("quantity", "num")], [
        "Add {quantity} {item} to my cart.", "Put {item} in the cart, {quantity} of them."]),
    "play_music": ([("artist", "artist")], [
        "Play some {artist}.", "I want to hear {artist}.", "Put on {artist} instead of {_artist}."]),
    "set_volume": ([("level", "num")], [
        "Set the volume to {level}.", "Turn the volume to {level} percent."]),
    "set_light": ([("room", "room"), ("color", "color")], [
        "Make the {room} lights {color}.", "Turn the lights in the {room} {color}.",
        "I want {color} light in the {room}."]),
    "set_thermostat": ([("room", "room"), ("temp", "num")], [
        "Set the {room} to {temp} degrees.", "Make it {temp} degrees in the {room}."]),
    "lock_door": ([("room", "room")], [
        "Lock the {room} door.", "Is the {room} locked? Lock it."]),
    "translate": ([("word", "word"), ("language", "language")], [
        "Translate {word} into {language}.", "How do you say {word} in {language}?",
        "What's the {language} word for {word}?"]),
    "define_word": ([("word", "word")], [
        "Define {word}.", "What does {word} mean?"]),
    "stock_price": ([("ticker", "ticker")], [
        "What's {ticker} trading at?", "Get the price of {ticker}.",
        "Forget {_ticker}; how is {ticker} doing?"]),
    "buy_stock": ([("ticker", "ticker"), ("shares", "num")], [
        "Buy {shares} shares of {ticker}.", "Get me {ticker}, {shares} shares."]),
    "schedule_meeting": ([("person", "person"), ("day", "day"), ("time", "time")], [
        "Schedule a meeting with {person} on {day} at {time}.",
        "Set up time with {person} at {time} on {day}.",
        "On {day} at {time} I'm meeting {person}; put it on the calendar."]),
    "cancel_meeting": ([("person", "person"), ("day", "day")], [
        "Cancel my meeting with {person} on {day}.", "Call off the {day} meeting with {person}."]),
    "get_directions": ([("origin", "city"), ("destination", "city")], [
        "How do I get from {origin} to {destination}?", "Directions to {destination} starting at {origin}."]),
    "find_restaurant": ([("city", "city"), ("cuisine", "cuisine")], [
        "Find a {cuisine} restaurant in {city}.", "Where can I get {cuisine} food in {city}?"]),
    "order_taxi": ([("destination", "city"), ("time", "time")], [
        "Get me a taxi to {destination} at {time}.", "At {time} I need a cab to {destination}."]),
    "get_news": ([("topic", "topic")], [
        "Show me the latest {topic} news.", "Any {topic} headlines today?"]),
    "calculate_tip": ([("amount", "num"), ("percent", "num")], [
        "What's a {percent} percent tip on {amount} dollars?", "Tip {percent} percent on a {amount} dollar bill.",
        "The bill is {amount} dollars and I want to leave {percent} percent."]),
    "split_bill": ([("amount", "num"), ("people", "num")], [
        "Split {amount} dollars between {people} people.",
        "{people} of us owe {amount} dollars in total; how much each?"]),
    "send_sms": ([("to", "person"), ("text", "word")], [
        "Text {to} the word {text}.", "Send {text} to {to} by text message."]),
}

UNSEEN_TOOLS = {
    "get_population": ([("city", "city")], [
        "How many people live in {city}?", "What is the population of {city}?"]),
    "rent_car": ([("city", "city"), ("days", "num")], [
        "Rent a car in {city} for {days} days.", "I need a rental car for {days} days in {city}."]),
    "send_gift": ([("recipient", "person"), ("item", "item")], [
        "Send {recipient} some {item} as a gift.", "Gift {item} to {recipient}."]),
    "ship_package": ([("origin", "city"), ("destination", "city"), ("weight", "num")], [
        "Ship a {weight} kilo package from {origin} to {destination}.",
        "Send a box to {destination} from {origin}; it weighs {weight} kilos."]),
    "set_reminder": ([("minutes", "num"), ("task", "task")], [
        "Remind me to {task} in {minutes} minutes.", "In {minutes} minutes, tell me to {task}."]),
    "dim_lights": ([("room", "room"), ("level", "num")], [
        "Dim the {room} lights to {level} percent.", "Set the {room} brightness to {level}."]),
    "count_calories": ([("item", "item"), ("quantity", "num")], [
        "How many calories are in {quantity} servings of {item}?", "Calories for {item}, {quantity} servings."]),
    "invite_guest": ([("person", "person"), ("day", "day")], [
        "Invite {person} over on {day}.", "Ask {person} to come by on {day}."]),
    "sell_stock": ([("ticker", "ticker"), ("shares", "num")], [
        "Sell {shares} shares of {ticker}.", "Dump {ticker}, {shares} shares."]),
    "weather_forecast": ([("city", "city"), ("day", "day")], [
        "What's the forecast for {city} on {day}?", "Will it be sunny in {city} on {day}?"]),
}

N_LISTED = 4

# --context: argument values are looked up in a list of facts instead of read off the
# request. kind -> (fact, reference to the fact's value). "num" is left literal
# ("Ava's lucky number minutes" stops being English).
FACTS = {
    "city": ("{o} lives in {v}.", "{o}'s city"),
    "day": ("{o} is free on {v}.", "the day {o} is free"),
    "time": ("{o} wakes up at {v}.", "{o}'s wake-up time"),
    "currency": ("{o} is paid in {v}.", "{o}'s currency"),
    "item": ("{o}'s favourite food is {v}.", "{o}'s favourite food"),
    "color": ("{o}'s favourite color is {v}.", "{o}'s favourite color"),
    "artist": ("{o}'s favourite artist is {v}.", "{o}'s favourite artist"),
    "language": ("{o} speaks {v}.", "{o}'s language"),
    "ticker": ("{o} owns {v} stock.", "{o}'s stock"),
    "person": ("{o}'s manager is {v}.", "{o}'s manager"),
}
# Owners of facts; disjoint from the "person" pool so a fact's subject is never a call value.
OWNERS = "Ava Ben Cleo Dev Eli Fay Gus Hal Ivy Jon Kim Lou Mei Ned Ola Pia Raj Sol Ted Val Wes Xan Yan Zed".split()


def signature(name: str, tools: dict) -> str:
    return f"{name}({', '.join(p for p, _ in tools[name][0])})"


def context_facts(rng: random.Random, params, values: dict, fill: dict, p_ref: float, p_two: float) -> list[str]:
    """Rewrite ``fill`` so referable values are named through facts, and return the shuffled facts.

    One hop: "Ava lives in Oslo." + request "... Ava's city ..."; a distractor fact of
    the same kind about another owner. Two hops (non-person kinds): "Ava's manager is
    Ben. Ben lives in Oslo." + "... Ava's manager's city ...", plus Ava's own city
    (the one-hop shortcut, wrong) and another owner's manager chain. Every value in a
    fact is distinct from the call's values and from the request's literal distractors.
    """
    owners = rng.sample(OWNERS, len(OWNERS))
    taken = {kind: {v for p, k in params for v in (values[p], fill["_" + p]) if k == kind} for _, kind in params}

    def draw(kind):
        v = rng.choice([x for x in POOLS[kind] if x not in taken.setdefault(kind, set())])
        taken[kind].add(v)
        return v

    manager = FACTS["person"][0]
    facts = []
    for p, kind in params:
        if kind not in FACTS or rng.random() >= p_ref:
            continue
        fact, ref = FACTS[kind]
        o = owners.pop()
        if kind != "person" and rng.random() < p_two:
            m, o2, m2 = owners.pop(), owners.pop(), owners.pop()
            facts += [manager.format(o=o, v=m), fact.format(o=m, v=values[p]), fact.format(o=o, v=draw(kind)),
                      manager.format(o=o2, v=m2), fact.format(o=m2, v=draw(kind))]
            fill[p] = ref.format(o=f"{o}'s manager")
        else:
            facts += [fact.format(o=o, v=values[p]), fact.format(o=owners.pop(), v=draw(kind))]
            fill[p] = ref.format(o=o)
    while len(facts) < 2:  # every prompt has a context line
        kind = rng.choice(sorted(FACTS))
        facts.append(FACTS[kind][0].format(o=owners.pop(), v=draw(kind)))
    rng.shuffle(facts)
    return facts


def sample(rng: random.Random, targets: dict, pool: dict, context: bool = False, p_ref: float = 0.8,
           p_two: float = 0.3):
    """One (prompt, call) with the target drawn from ``targets`` and the other listed tools from ``pool``."""
    name = rng.choice(sorted(targets))
    params, templates = targets[name]
    template = rng.choice(templates)
    by_pool: dict[str, list[str]] = {}
    for _, kind in params:
        by_pool.setdefault(kind, [])
    for kind in by_pool:  # distinct values within a pool, so slot binding is never ambiguous
        n = sum(k == kind for _, k in params) + 1
        by_pool[kind] = rng.sample(POOLS[kind], n)
    values, used = {}, {k: 0 for k in by_pool}
    for p, kind in params:
        values[p] = by_pool[kind][used[kind]]
        used[kind] += 1
    fill = dict(values)
    for p, kind in params:
        fill["_" + p] = by_pool[kind][-1]  # the spare value: never in the call
    others = rng.sample([t for t in sorted(pool) if t != name], N_LISTED - 1)
    listed = others + [name]
    rng.shuffle(listed)
    every = {**TRAIN_TOOLS, **UNSEEN_TOOLS}
    ctx = f"Context: {' '.join(context_facts(rng, params, values, fill, p_ref, p_two))}\n" if context else ""
    prompt = (f"Tools: {'; '.join(signature(t, every) for t in listed)}\n"
              f"{ctx}Request: {template.format(**fill)}\nCall:")
    call = f" {name}({', '.join(f'{p}: {values[p]}' for p, _ in params)})"
    return prompt, call


def build(rng: random.Random, targets: dict, pool: dict, n: int, seen: set, **kw) -> list[dict]:
    rows = []
    while len(rows) < n:
        q, a = sample(rng, targets, pool, **kw)
        if q not in seen:
            seen.add(q)
            rows.append({"q_str": q, "a_str": a})
    return rows


def write(name: str, parts: dict, meta: dict) -> str:
    out = os.path.join(DIR_ROOT, "datasets", name)
    os.makedirs(out, exist_ok=True)
    for split, rows in parts.items():
        with open(os.path.join(out, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="toolcall")
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context", action="store_true",
                    help="argument values named through a context of facts (one or two hops) instead of literally")
    ap.add_argument("--p-ref", type=float, default=0.8, help="--context: chance a referable value is looked up")
    ap.add_argument("--p-two-hop", type=float, default=0.3, help="--context: chance a lookup takes two hops")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seen: set = set()
    kw = dict(context=args.context, p_ref=args.p_ref, p_two=args.p_two_hop)
    train = build(rng, TRAIN_TOOLS, TRAIN_TOOLS, args.train, seen, **kw)
    test = build(rng, TRAIN_TOOLS, TRAIN_TOOLS, args.test, seen, **kw)
    every = {**TRAIN_TOOLS, **UNSEEN_TOOLS}
    u_train = build(rng, UNSEEN_TOOLS, every, args.test, seen, **kw)  # never trained on; load_data needs a train split
    u_test = build(rng, UNSEEN_TOOLS, every, args.test, seen, **kw)
    meta = {"answer_type": "call", "train_tools": sorted(TRAIN_TOOLS), "unseen_tools": sorted(UNSEEN_TOOLS),
            **({"context": True, "p_ref": args.p_ref, "p_two_hop": args.p_two_hop} if args.context else {})}
    a = write(args.name, {"train": train, "test": test, "all": train + test}, {**meta, "split": "seen tools"})
    b = write(args.name + "_unseen", {"train": u_train, "test": u_test, "all": u_train + u_test},
              {**meta, "split": "held-out tools only (targets); distractors from all tools"})
    print(f"{args.name}: {len(train)} train + {len(test)} test -> {a}")
    print(f"{args.name}_unseen: {len(u_train)} + {len(u_test)} -> {b}")
    for r in train[:2] + u_test[:2]:
        print(f"---\n{r['q_str']}{r['a_str']}")


if __name__ == "__main__":
    main()
