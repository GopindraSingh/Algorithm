"""
NIFTY 500 FALLING STOCK + INTRADAY SHORT SCREEN

============================================================
DAILY SCREEN
============================================================

For EVERY trading day inside START_DATE -> END_DATE:

    Previous day (T-1) must fall >= 1%
    AND
    Current day (T0) must fall >= 1%

    AND

    T0 Volume >= 1.30 x previous 20-session Volume SMA

IMPORTANT:
    Every qualifying day is kept.

Example:

If you enter:

    2026-09-01
    2026-09-04

and ABC.NS qualifies on:

    2026-09-01
    2026-09-03
    2026-09-04

then ALL THREE dates are displayed.

============================================================
INTRADAY SCREEN
============================================================

For each qualifying T0 that is within Yahoo's intraday window:

    First candle:
        09:15 - 09:30

    morning_low = first 15m candle Low

Then from 09:30 onward:

    Close < morning_low
    RSI(14) > 25
    MACD(12,26) < Signal(9)

The first matching candle is recorded.

============================================================
OUTPUT
============================================================

While scanning, ONLY:

    Remaining: 3m 42s

At the end:

    Ticker       Date          T-1 Fall    T0 Fall    Vol Ratio
    -------------------------------------------------------------
    ABC.NS       2026-09-01    -1.32%      -2.11%      1.54x
    ABC.NS       2026-09-03    -1.51%      -1.87%      1.42x
    XYZ.NS       2026-09-04    -2.03%      -3.21%      1.91x

No CSV is created.
"""


from __future__ import annotations

import io
import os
import sys
import time
import urllib.request
import contextlib

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# SETTINGS
# ============================================================

NIFTY_500_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
)

NIFTY_50_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
)

# BOTH days must fall by at least 1%
MIN_DAILY_FALL = 0.01

# Volume
VOLUME_SMA_PERIOD = 20
VOLUME_MULTIPLIER = 1.30

# Yahoo intraday limitation
INTRADAY_MAX_AGE_DAYS = 58

# RSI
RSI_PERIOD = 14

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# NSE timezone
MARKET_TIMEZONE = "Asia/Kolkata"

MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

REQUEST_DELAY_SECONDS = 0.08


# ============================================================
# DATE INPUT
# ============================================================

def get_date_input(prompt):

    while True:

        value = input(prompt).strip()

        try:

            return pd.to_datetime(
                value,
                format="%Y-%m-%d"
            ).normalize()

        except Exception:

            print(
                "Invalid date. Please use YYYY-MM-DD."
            )


def get_date_range():

    start_date = get_date_input(
        "START_DATE (YYYY-MM-DD): "
    )

    end_date = get_date_input(
        "END_DATE   (YYYY-MM-DD): "
    )

    if end_date < start_date:

        raise ValueError(
            "END_DATE cannot be earlier than START_DATE."
        )

    return start_date, end_date


# ============================================================
# NSE UNIVERSE
# ============================================================

def download_nse_csv(url, name):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }

    request = urllib.request.Request(
        url,
        headers=headers
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        raw = response.read()

    return pd.read_csv(
        io.BytesIO(raw)
    )


def find_symbol_column(df):

    for column in [
        "Symbol",
        "SYMBOL",
        "symbol",
        "Ticker",
        "TICKER"
    ]:

        if column in df.columns:

            return column

    raise ValueError(
        "Could not find symbol column."
    )


def get_index_symbols(url, name):

    df = download_nse_csv(
        url,
        name
    )

    symbol_column = find_symbol_column(
        df
    )

    symbols = (
        df[symbol_column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return set(symbols)


def build_universe():

    print(
        "\nLoading Nifty 500 universe..."
    )

    nifty500 = get_index_symbols(
        NIFTY_500_URL,
        "Nifty 500"
    )

    nifty50 = get_index_symbols(
        NIFTY_50_URL,
        "Nifty 50"
    )

    symbols = sorted(
        nifty500 - nifty50
    )

    tickers = []

    for symbol in symbols:

        if symbol.endswith(".NS"):

            ticker = symbol

        else:

            ticker = (
                f"{symbol}.NS"
            )

        tickers.append(
            ticker
        )

    print(
        f"Universe loaded: {len(tickers)} stocks"
    )

    return tickers


# ============================================================
# SILENCE YFINANCE CONSOLE NOISE
# ============================================================

def quiet_yfinance_download(
    *args,
    **kwargs
):
    """
    yfinance may print:

        HTTP Error 404
        possibly delisted
        no timezone found
        1 Failed download

    directly to stderr/stdout instead of raising a normal
    exception.

    This wrapper suppresses those messages.

    We then determine whether usable data was actually returned.
    """

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with contextlib.redirect_stdout(
        stdout_buffer
    ), contextlib.redirect_stderr(
        stderr_buffer
    ):

        try:

            result = yf.download(
                *args,
                **kwargs
            )

            return result

        except Exception:

            return pd.DataFrame()


# ============================================================
# COLUMN FLATTENER
# ============================================================

def flatten_columns(df):

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    return df


# ============================================================
# DAILY DATA
# ============================================================

def download_daily_data(
    ticker,
    start_date,
    end_date
):

    buffered_start = (
        start_date
        - pd.Timedelta(days=60)
    )

    download_end = (
        end_date
        + pd.Timedelta(days=1)
    )

    df = quiet_yfinance_download(

        ticker,

        start=buffered_start.strftime(
            "%Y-%m-%d"
        ),

        end=download_end.strftime(
            "%Y-%m-%d"
        ),

        interval="1d",

        auto_adjust=False,

        progress=False,

        threads=False,

        timeout=15,

        # Prevent timezone confusion for daily data.
        ignore_tz=True,
    )

    if df is None or df.empty:

        return pd.DataFrame()

    df = flatten_columns(df)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in required:

        if column not in df.columns:

            return pd.DataFrame()

    df = df[
        required
    ].copy()

    # --------------------------------------------------------
    # DAILY INDEX
    # --------------------------------------------------------

    df.index = pd.to_datetime(
        df.index
    )

    if getattr(
        df.index,
        "tz",
        None
    ) is not None:

        df.index = (
            df.index
            .tz_localize(None)
        )

    df.index = (
        df.index
        .normalize()
    )

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Close",
            "Volume"
        ]
    )

    if df.empty:

        return pd.DataFrame()

    return df


# ============================================================
# DAILY CANDIDATES
# ============================================================

def find_daily_candidates(
    ticker,
    df,
    start_date,
    end_date
):

    if df.empty:

        return []

    df = df.copy()

    # --------------------------------------------------------
    # DAILY RETURN
    # --------------------------------------------------------

    df["Day_Drop"] = (
        df["Close"]
        .pct_change()
    )

    # --------------------------------------------------------
    # VOLUME SMA
    #
    # Shift by 1 so today's volume is compared against the
    # previous 20 completed sessions.
    # --------------------------------------------------------

    df["Vol_MA20"] = (
        df["Volume"]
        .rolling(
            VOLUME_SMA_PERIOD,
            min_periods=VOLUME_SMA_PERIOD
        )
        .mean()
        .shift(1)
    )

    df["Vol_Ratio"] = (
        df["Volume"]
        / df["Vol_MA20"]
    )

    # --------------------------------------------------------
    # ONLY THE USER'S DATE RANGE
    # --------------------------------------------------------

    positions = np.flatnonzero(
        np.asarray(
            (
                (df.index >= start_date)
                &
                (df.index <= end_date)
            )
        )
    )

    candidates = []

    # ========================================================
    # IMPORTANT:
    #
    # We check EVERY date in the range.
    #
    # We do NOT stop after finding one candidate.
    # ========================================================

    for position in positions:

        if position < 1:

            continue

        t1 = df.iloc[
            position - 1
        ]

        t0 = df.iloc[
            position
        ]

        t1_drop = t1[
            "Day_Drop"
        ]

        t0_drop = t0[
            "Day_Drop"
        ]

        vol_ratio = t0[
            "Vol_Ratio"
        ]

        if pd.isna(t1_drop):
            continue

        if pd.isna(t0_drop):
            continue

        if pd.isna(vol_ratio):
            continue

        # ----------------------------------------------------
        # BOTH DAYS MUST FALL >= 1%
        # ----------------------------------------------------

        t1_ok = (
            float(t1_drop)
            <= -MIN_DAILY_FALL
        )

        t0_ok = (
            float(t0_drop)
            <= -MIN_DAILY_FALL
        )

        volume_ok = (
            float(vol_ratio)
            >= VOLUME_MULTIPLIER
        )

        if (
            t1_ok
            and t0_ok
            and volume_ok
        ):

            candidates.append(
                {
                    "Ticker": ticker,

                    "Date": df.index[
                        position
                    ],

                    "T-1 Fall %": (
                        float(t1_drop)
                        * 100
                    ),

                    "T0 Fall %": (
                        float(t0_drop)
                        * 100
                    ),

                    "Vol Ratio": (
                        float(vol_ratio)
                    )
                }
            )

    return candidates


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    close,
    period=14
):

    close = pd.to_numeric(
        close,
        errors="coerce"
    )

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    rs = (
        avg_gain
        / avg_loss
    )

    rsi = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    rsi = rsi.where(
        avg_loss != 0,
        100
    )

    return rsi


# ============================================================
# MACD
# ============================================================

def calculate_macd(close):

    close = pd.to_numeric(
        close,
        errors="coerce"
    )

    ema_fast = (
        close
        .ewm(
            span=MACD_FAST,
            adjust=False,
            min_periods=MACD_FAST
        )
        .mean()
    )

    ema_slow = (
        close
        .ewm(
            span=MACD_SLOW,
            adjust=False,
            min_periods=MACD_SLOW
        )
        .mean()
    )

    macd = (
        ema_fast
        - ema_slow
    )

    signal = (
        macd
        .ewm(
            span=MACD_SIGNAL,
            adjust=False,
            min_periods=MACD_SIGNAL
        )
        .mean()
    )

    return macd, signal


# ============================================================
# 15-MINUTE DATA
# ============================================================

def download_15m_data(
    ticker,
    target_date
):

    start = target_date.normalize()

    end = (
        start
        + pd.Timedelta(days=1)
    )

    df = quiet_yfinance_download(

        ticker,

        start=start.strftime(
            "%Y-%m-%d"
        ),

        end=end.strftime(
            "%Y-%m-%d"
        ),

        interval="15m",

        auto_adjust=False,

        prepost=False,

        progress=False,

        threads=False,

        timeout=15,

        # Intraday timestamps need their timezone.
        ignore_tz=False,
    )

    if df is None or df.empty:

        return pd.DataFrame()

    df = flatten_columns(df)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in required:

        if column not in df.columns:

            return pd.DataFrame()

    df = df[
        required
    ].copy()

    df.index = pd.to_datetime(
        df.index
    )

    # --------------------------------------------------------
    # CONVERT TO INDIA TIME
    # --------------------------------------------------------

    if getattr(
        df.index,
        "tz",
        None
    ) is not None:

        df.index = (
            df.index
            .tz_convert(
                MARKET_TIMEZONE
            )
        )

    else:

        df.index = (
            df.index
            .tz_localize(
                MARKET_TIMEZONE
            )
        )

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    df = df.sort_index()

    # --------------------------------------------------------
    # TARGET DATE
    # --------------------------------------------------------

    df = df[
        df.index.date
        == target_date.date()
    ]

    # --------------------------------------------------------
    # NSE SESSION
    # --------------------------------------------------------

    session_open = pd.Timestamp(
        f"{target_date:%Y-%m-%d} "
        f"{MARKET_OPEN}",
        tz=MARKET_TIMEZONE
    )

    session_close = pd.Timestamp(
        f"{target_date:%Y-%m-%d} "
        f"{MARKET_CLOSE}",
        tz=MARKET_TIMEZONE
    )

    df = df[
        (df.index >= session_open)
        &
        (df.index < session_close)
    ]

    return df


# ============================================================
# INTRADAY ANALYSIS
# ============================================================

def analyze_intraday(
    ticker,
    target_date
):

    df = download_15m_data(
        ticker,
        target_date
    )

    if df.empty:

        return None

    if len(df) < 2:

        return None

    # --------------------------------------------------------
    # FIRST CANDLE
    # 09:15 - 09:30
    # --------------------------------------------------------

    morning_low = (
        df.iloc[0]["Low"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI"] = calculate_rsi(
        df["Close"],
        RSI_PERIOD
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    (
        df["MACD"],
        df["MACD_Signal"]
    ) = calculate_macd(
        df["Close"]
    )

    # --------------------------------------------------------
    # SCAN FROM 09:30
    # --------------------------------------------------------

    for i in range(
        1,
        len(df)
    ):

        candle = df.iloc[i]

        close = candle["Close"]

        rsi = candle["RSI"]

        macd = candle["MACD"]

        signal = candle[
            "MACD_Signal"
        ]

        if pd.isna(close):
            continue

        if pd.isna(rsi):
            continue

        if pd.isna(macd):
            continue

        if pd.isna(signal):
            continue

        condition_1 = (
            float(close)
            < float(morning_low)
        )

        condition_2 = (
            float(rsi)
            > 25
        )

        condition_3 = (
            float(macd)
            < float(signal)
        )

        if (
            condition_1
            and condition_2
            and condition_3
        ):

            return {
                "Trigger Time":
                    df.index[i].strftime(
                        "%H:%M"
                    ),

                "Breakout Price":
                    float(close),

                "RSI":
                    float(rsi)
            }

    return None


# ============================================================
# TIME FORMAT
# ============================================================

def format_time(seconds):

    seconds = max(
        0,
        int(seconds)
    )

    hours = (
        seconds // 3600
    )

    minutes = (
        seconds % 3600
    ) // 60

    secs = (
        seconds % 60
    )

    if hours:

        return (
            f"{hours}h "
            f"{minutes}m "
            f"{secs}s"
        )

    if minutes:

        return (
            f"{minutes}m "
            f"{secs}s"
        )

    return f"{secs}s"


# ============================================================
# SINGLE-LINE REMAINING TIME
# ============================================================

def update_remaining(
    scanned,
    total,
    start_time
):

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if scanned > 0:

        average = (
            elapsed
            / scanned
        )

        remaining = (
            total
            - scanned
        )

        eta = (
            average
            * remaining
        )

    else:

        eta = 0

    text = (
        f"Remaining: "
        f"{format_time(eta)}"
    )

    # Clear complete line.
    sys.stdout.write(
        "\r\033[2K"
        + text
    )

    sys.stdout.flush()


# ============================================================
# MAIN SCANNER
# ============================================================

def run_scanner(
    start_date,
    end_date
):

    tickers = build_universe()

    total = len(tickers)

    today = (
        pd.Timestamp.now()
        .normalize()
    )

    # --------------------------------------------------------
    # ALL DAILY RESULTS
    # --------------------------------------------------------

    daily_results = []

    # --------------------------------------------------------
    # VERIFIED INTRADAY RESULTS
    # --------------------------------------------------------

    intraday_results = []

    scanned = 0

    start_time = (
        time.perf_counter()
    )

    print()
    print(
        "Scanning..."
    )

    # ========================================================
    # STOCK LOOP
    # ========================================================

    for ticker in tickers:

        try:

            # ------------------------------------------------
            # DAILY DOWNLOAD
            # ------------------------------------------------

            daily = download_daily_data(
                ticker,
                start_date,
                end_date
            )

            if not daily.empty:

                candidates = (
                    find_daily_candidates(
                        ticker,
                        daily,
                        start_date,
                        end_date
                    )
                )

                # ========================================================
                # IMPORTANT:
                #
                # ALL candidates are retained.
                #
                # If the same stock qualifies on 4 different dates,
                # all 4 dates appear in the final output.
                # ========================================================

                for candidate in candidates:

                    daily_results.append(
                        candidate
                    )

                    target_date = (
                        pd.Timestamp(
                            candidate["Date"]
                        )
                        .normalize()
                    )

                    # ------------------------------------------------
                    # AGE OF T0
                    # ------------------------------------------------

                    age_days = (
                        today
                        - target_date
                    ).days

                    # ------------------------------------------------
                    # TOO OLD FOR 15M
                    # ------------------------------------------------

                    if (
                        age_days
                        > INTRADAY_MAX_AGE_DAYS
                    ):

                        continue

                    # ------------------------------------------------
                    # 15M CHECK
                    # ------------------------------------------------

                    try:

                        intraday = (
                            analyze_intraday(
                                ticker,
                                target_date
                            )
                        )

                    except Exception:

                        intraday = None

                    # ------------------------------------------------
                    # VERIFIED SETUP
                    # ------------------------------------------------

                    if intraday is not None:

                        intraday_results.append(
                            {
                                "Ticker":
                                    ticker,

                                "Date":
                                    target_date,

                                "T-1 Fall %":
                                    candidate[
                                        "T-1 Fall %"
                                    ],

                                "T0 Fall %":
                                    candidate[
                                        "T0 Fall %"
                                    ],

                                "Vol Ratio":
                                    candidate[
                                        "Vol Ratio"
                                    ],

                                "Trigger Time":
                                    intraday[
                                        "Trigger Time"
                                    ],

                                "Breakout Price":
                                    intraday[
                                        "Breakout Price"
                                    ],

                                "RSI":
                                    intraday[
                                        "RSI"
                                    ]
                            }
                        )

        except KeyboardInterrupt:

            print(
                "\nScan stopped."
            )

            raise

        except Exception:

            # Invalid Yahoo symbols such as DUMMYHEG.NS
            # are silently skipped.
            pass

        # ------------------------------------------------
        # PROGRESS
        # ------------------------------------------------

        scanned += 1

        update_remaining(
            scanned,
            total,
            start_time
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    # ========================================================
    # FINISHED
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # Move to next line once.
    print()

    # ========================================================
    # DAILY RESULTS
    # ========================================================

    print()
    print("=" * 90)
    print(
        "ALL QUALIFYING FALLING DAYS"
    )
    print("=" * 90)

    if not daily_results:

        print(
            "No qualifying stocks found."
        )

    else:

        daily_df = pd.DataFrame(
            daily_results
        )

        daily_df["Date"] = (
            pd.to_datetime(
                daily_df["Date"]
            )
        )

        daily_df = (
            daily_df
            .sort_values(
                [
                    "Date",
                    "Ticker"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        display = daily_df[
            [
                "Ticker",
                "Date",
                "T-1 Fall %",
                "T0 Fall %",
                "Vol Ratio"
            ]
        ].copy()

        display["Date"] = (
            display["Date"]
            .dt.strftime(
                "%Y-%m-%d"
            )
        )

        display["T-1 Fall %"] = (
            display["T-1 Fall %"]
            .map(
                lambda x:
                f"{x:.2f}%"
            )
        )

        display["T0 Fall %"] = (
            display["T0 Fall %"]
            .map(
                lambda x:
                f"{x:.2f}%"
            )
        )

        display["Vol Ratio"] = (
            display["Vol Ratio"]
            .map(
                lambda x:
                f"{x:.2f}x"
            )
        )

        print(
            display.to_string(
                index=False
            )
        )

    # ========================================================
    # INTRADAY RESULTS
    # ========================================================

    print()
    print("=" * 90)
    print(
        "VERIFIED INTRADAY SETUPS"
    )
    print("=" * 90)

    if not intraday_results:

        print(
            "No verified intraday setups found."
        )

    else:

        intraday_df = pd.DataFrame(
            intraday_results
        )

        intraday_df["Date"] = (
            pd.to_datetime(
                intraday_df["Date"]
            )
        )

        intraday_df = (
            intraday_df
            .sort_values(
                [
                    "Date",
                    "Ticker"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        display = intraday_df[
            [
                "Ticker",
                "Date",
                "T-1 Fall %",
                "T0 Fall %",
                "Vol Ratio",
                "Trigger Time",
                "Breakout Price",
                "RSI"
            ]
        ].copy()

        display["Date"] = (
            display["Date"]
            .dt.strftime(
                "%Y-%m-%d"
            )
        )

        display["T-1 Fall %"] = (
            display["T-1 Fall %"]
            .map(
                lambda x:
                f"{x:.2f}%"
            )
        )

        display["T0 Fall %"] = (
            display["T0 Fall %"]
            .map(
                lambda x:
                f"{x:.2f}%"
            )
        )

        display["Vol Ratio"] = (
            display["Vol Ratio"]
            .map(
                lambda x:
                f"{x:.2f}x"
            )
        )

        display["Breakout Price"] = (
            display["Breakout Price"]
            .map(
                lambda x:
                f"{x:.2f}"
            )
        )

        display["RSI"] = (
            display["RSI"]
            .map(
                lambda x:
                f"{x:.2f}"
            )
        )

        print(
            display.to_string(
                index=False
            )
        )

    # ========================================================
    # SMALL FINAL SUMMARY
    # ========================================================

    print()
    print(
        f"Finished in {format_time(elapsed)}."
    )

    print(
        f"Qualifying falling days: "
        f"{len(daily_results)}"
    )

    print(
        f"Verified intraday setups: "
        f"{len(intraday_results)}"
    )

    return (
        daily_results,
        intraday_results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print(
        "NIFTY 500 FALLING STOCK SCANNER"
    )
    print("=" * 90)

    print(
        "T-1 fall >= 1%"
    )

    print(
        "T0  fall >= 1%"
    )

    print(
        "T0 volume >= 1.30x 20-day volume SMA"
    )

    print(
        "Every qualifying date in the selected range is retained."
    )

    print("=" * 90)

    try:

        start_date, end_date = (
            get_date_range()
        )

        run_scanner(
            start_date,
            end_date
        )

    except KeyboardInterrupt:

        print(
            "\nProgram stopped."
        )

        sys.exit(1)

    except Exception as exc:

        print(
            f"\nFatal error: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":

    main()

