#a consideration I made is to only crawl markets with yes or no for classification. The main reason being it would be weird to make news titles out of markets with more than 2 outcomes.
#I chose to only look at markets that are currently unresolved, as resolved markets all collapses into a single price of 1 for the winning outcome and 0 for the losing outcomes, which doesn't help with our project as we would just learn to predict the winning outcome.
#I also chose to only look at markets with a certain amount of liquidity, mainly as a methodology to filter our noise.

import requests
import time
import json

BASE_URL = "https://gamma-api.polymarket.com/markets"
PAGE_LIMIT = 500
SLEEP = 0.15

FETCH_LIMIT = 100000
LIQUIDITY_MIN = 1000


def fetch_markets_capped(cap=FETCH_LIMIT, liquidity_min=LIQUIDITY_MIN):
    markets = []
    offset = 0

    while True:
        if len(markets) >= cap:
            print(f"Reached fetch cap of {cap}. Stopping.")
            break
        url = (
            f"{BASE_URL}"
            f"?closed=false"
            f"&liquidity_num_min={liquidity_min}"
            f"&limit={PAGE_LIMIT}"
            f"&offset={offset}"
        )
        print(f"Fetching: offset={offset}")
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            print("Empty page. Stopping.")
            break
        markets.extend(data)
        offset += PAGE_LIMIT
        time.sleep(SLEEP)
    return markets[:cap]


def is_yes_no_market(market):
    raw = market.get("outcomes")
    if raw is None:
        return False
    if isinstance(raw, str):
        try:
            outcomes = json.loads(raw)
        except json.JSONDecodeError:
            return False
    else:
        outcomes = raw
    if not isinstance(outcomes, list):
        return False
    norm = [o.strip().lower() for o in outcomes]
    return sorted(norm) == ["no", "yes"]


def extract_market_record(market):
    market_id = market.get("id")
    name = market.get("question")
    if not name:
        events = market.get("events") or []
        if events:
            name = events[0].get("title", f"Market {market_id}")
    if not name:
        name = f"Market {market_id}"
    raw_outcomes = market.get("outcomes")
    if isinstance(raw_outcomes, str):
        try:
            outcomes = json.loads(raw_outcomes)
        except:
            outcomes = raw_outcomes
    else:
        outcomes = raw_outcomes
    raw_prices = market.get("outcomePrices")
    if isinstance(raw_prices, str):
        try:
            prices = json.loads(raw_prices)
        except:
            prices = raw_prices
    else:
        prices = raw_prices
    return {
        "market_id": market_id,
        "name": name,
        "outcomes": outcomes,
        "outcomePrices": prices,
    }


if __name__ == "__main__":
    print("Fetching unresolved + high-liquidity markets via API filters…")
    all_markets = fetch_markets_capped(FETCH_LIMIT, LIQUIDITY_MIN)
    print(f"Fetched {len(all_markets)} markets (already unresolved + liquidity filtered).")
    print("Filtering for Yes/No markets…")
    filtered = []
    for m in all_markets:
        if not is_yes_no_market(m):
            continue
        filtered.append(extract_market_record(m))
    print(f"Remaining Yes/No unresolved high-liquidity markets: {len(filtered)}")
    out_file = "test_markets_unresolved_yes_no.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Saved to {out_file}")