def main():
    ticker = "AAPL"

    # List Aggregates (Bars)
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="minute", from_="2024-01-01", to="2024-06-13", limit=50000):
        aggs.append(a)


if __name__ == "__main__":
    main()