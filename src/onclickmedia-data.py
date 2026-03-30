# Option Data Downloader, OnclickMedia to Parquet
# This script downloads option chain data from OnclickMedia and saves it as a `.parquet` file.
# It is designed for large datasets such as `SPX` or `SPY`, where Parquet is much better than CSV for size and speed.
#
# What the script does:
# - gets all available quote dates for a ticker
# - selects dates from the last `N` years
# - downloads the option chain for each date
# - adds a `quote_date` column
# - saves everything into one Parquet file
# - can skip already saved dates with `--resume`
#
# Basic usage:
# python "src/onclickmedia-data.py" --ticker SPY --years 100 --resume --output "data/raw/spy-options-onclickmedia.parquet"

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

API_BASE = 'https://api.onclickmedia.com/'
USER_AGENT = 'Mozilla/5.0'


def fetch_json(params: dict, retries: int = 4, pause: float = 1.0):
    url = f'{API_BASE}?{urlencode(params)}'
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={'User-Agent': USER_AGENT})
            with urlopen(req, timeout=90) as response:
                return json.load(response)

        except HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(f'No dataset for URL: {url}') from exc
            last_err = exc
            if attempt < retries:
                time.sleep(pause * attempt)

        except (URLError, TimeoutError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(pause * attempt)

    raise RuntimeError(f'Failed request after {retries} attempts: {url}') from last_err


def get_available_dates(ticker: str) -> list[str]:
    payload = fetch_json({'ticker': ticker, 'list': 'date', 'output': 'json-v1'})
    dates = payload.get(ticker, [])
    if not dates:
        raise RuntimeError(f'No available dates returned for ticker {ticker}')
    return dates


def fetch_chain_for_date(ticker: str, quote_date: str) -> list[dict]:
    params = {
        'ticker': ticker,
        'date': quote_date,
        'output': 'json-v1',
        'data': 'greeks',
    }

    rows = fetch_json(params)

    if not isinstance(rows, list):
        raise RuntimeError(f'Unexpected payload for {ticker} {quote_date}: {type(rows)}')

    for row in rows:
        row['quote_date'] = quote_date

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch OnclickMedia option chain data into Parquet.')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol, default: SPY')
    parser.add_argument('--years', type=int, default=5, help='Trailing years to fetch, default: 5')
    parser.add_argument(
        '--pause-seconds',
        type=float,
        default=0.25,
        help='Pause between API calls in seconds, default: 0.25',
    )
    parser.add_argument(
        '--output',
        default='',
        help='Optional output parquet path',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip quote dates already present in the output parquet file',
    )
    return parser.parse_args()


def select_dates_for_years(available_dates: list[str], years: int) -> list[str]:
    if years <= 0:
        raise ValueError('--years must be a positive integer.')

    latest = date.fromisoformat(available_dates[-1])

    try:
        cutoff = latest.replace(year=latest.year - years)
    except ValueError:
        cutoff = latest - timedelta(days=365 * years)

    return [d for d in available_dates if date.fromisoformat(d) >= cutoff]


def read_existing_quote_dates(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    df_existing = pd.read_parquet(output_path)

    if 'quote_date' not in df_existing.columns:
        return set()

    return set(df_existing['quote_date'].dropna().astype(str).unique())


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f'{hours}h {minutes}m {secs}s'
    if minutes > 0:
        return f'{minutes}m {secs}s'
    return f'{secs}s'


def main():
    total_start_time = time.time()

    args = parse_args()
    ticker = args.ticker.upper()

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(__file__).resolve().parents[1] / 'data' / 'raw' / f'{ticker.lower()}_options_onclickmedia_last{args.years}y.parquet'
    )

    available_dates = get_available_dates(ticker)
    selected_dates = select_dates_for_years(available_dates, args.years)

    if not selected_dates:
        raise RuntimeError(f'No dates selected for {ticker} in last {args.years} year(s).')

    existing_dates = read_existing_quote_dates(output_path) if args.resume else set()
    pending_dates = [d for d in selected_dates if d not in existing_dates]

    if not pending_dates:
        print(f'No pending dates. Output already up to date: {output_path}', flush=True)
        return

    rows_all = []
    missing_dates = []
    download_start_time = time.time()
    total_pending = len(pending_dates)

    for i, quote_date in enumerate(pending_dates, start=1):
        try:
            date_rows = fetch_chain_for_date(ticker, quote_date)
            rows_all.extend(date_rows)

            if i == 1 or i % 20 == 0 or i == total_pending:
                elapsed = time.time() - download_start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total_pending - i) / rate if rate > 0 else 0

                print(
                    f'[{i:04d}/{total_pending}] {quote_date}: {len(date_rows)} rows '
                    f'(collected {len(rows_all)}, skipped {len(missing_dates)}) | '
                    f'elapsed {format_seconds(elapsed)} | '
                    f'eta {format_seconds(remaining)}',
                    flush=True,
                )

        except FileNotFoundError:
            missing_dates.append(quote_date)

            if i == 1 or i % 20 == 0 or i == total_pending:
                elapsed = time.time() - download_start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total_pending - i) / rate if rate > 0 else 0

                print(
                    f'[{i:04d}/{total_pending}] {quote_date}: missing (404), skipped '
                    f'| elapsed {format_seconds(elapsed)} '
                    f'| eta {format_seconds(remaining)}',
                    flush=True,
                )

        time.sleep(args.pause_seconds)

    if not rows_all:
        print('No new rows fetched.', flush=True)
        return

    df_new = pd.DataFrame(rows_all)

    if output_path.exists() and args.resume:
        df_existing = pd.read_parquet(output_path)
        df = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
    else:
        df = df_new

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    total_elapsed = time.time() - total_start_time

    print(f'\nSaved {len(df_new)} new rows to {output_path}', flush=True)
    print(f'Total rows in file: {len(df)}', flush=True)
    print(f'Total runtime: {format_seconds(total_elapsed)}', flush=True)

    if missing_dates:
        print(f'Skipped {len(missing_dates)} missing dates: {", ".join(missing_dates)}', flush=True)


if __name__ == '__main__':
    main()