"""
============================================================
NIFTY 500 DATE-WISE FALLING STOCK SCANNER
============================================================

UNIVERSE
--------
Nifty 500 stocks
EXCLUDING Nifty 50 stocks

DAILY CONDITIONS
----------------
For EVERY trading day in the user-selected date range:

1. Previous trading day (T-1) fell >= 1%
2. Current day (T0) fell >= 1%
3. T0 volume >= 1.30 x previous 20-session volume SMA

IMPORTANT:
-----------
Every qualifying date is retained.

If a stock qualifies on 4 different dates, all 4 dates
appear in the final date-wise table.

INTRADAY CONDITIONS
-------------------
For qualifying T0 dates that are within Yahoo's intraday
history window:

First 15-minute candle:
    09:15 - 09:30

morning_low = first candle Low

From 09:30 onward:

    Close < morning_low
    RSI(14) > 25
    MACD(12,26) < Signal(9)

The FIRST matching candle is recorded.

OUTPUT
------
During scanning, ONLY:

    Remaining: 3m 42s

At the end:

    DATE-WISE FALLING STOCKS

    Ticker       2026-09-07       2026-09-08       2026-09-09
    ----------------------------------------------------------------
    ABC.NS       -2.31% | 1.5x    -                -3.21% | 2.1x

No CSV is created.
============================================================
"""

from __future__ import annotations

import io
import sys
import time
import contextlib
import urllib.request

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

# Both T-1 and T0 must fall by at least this percentage.
MIN_DAILY_FALL = 0.01

# T0 volume requirement.
VOLUME_SMA_PERIOD = 20
VOLUME_MULTIPLIER = 1.30

# Yahoo Finance intraday protection.
INTRADAY_MAX_AGE_DAYS = 58

# RSI.
RSI_PERIOD = 14

# MACD.
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# NSE timezone.
MARKET_TIMEZONE = "Asia/Kolkata"

MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

# Small delay between stocks.
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

    # --------------------------------------------------------
    # Show number of calendar days.
    # --------------------------------------------------------

    calendar_days = (
        end_date - start_date
    ).days + 1

    print()
    print(
        f"Date range: "
        f"{start_date:%Y-%m-%d} -> "
        f"{end_date:%Y-%m-%d}"
    )

    print(
        f"Calendar days: {calendar_days}"
    )

    if calendar_days > 5:

        print(
            "Warning: You selected more than 5 calendar days."
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

    possible_columns = [
        "Symbol",
        "SYMBOL",
        "symbol",
        "Ticker",
        "TICKER"
    ]

    for column in possible_columns:

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

    print()
    print(
        "Loading Nifty 500 universe..."
    )

    nifty500 = get_index_symbols(
        NIFTY_500_URL,
        "Nifty 500"
    )

    nifty50 = get_index_symbols(
        NIFTY_50_URL,
        "Nifty 50"
    )

    # Remove Nifty 50.
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
# QUIET YFINANCE DOWNLOAD
# ============================================================

def quiet_yfinance_download(
    *args,
    **kwargs
):
    """
    yfinance can print errors such as:

        HTTP Error 404
        possibly delisted
        no timezone found
        1 Failed download

    directly to the console.

    This function suppresses those messages.

    Invalid Yahoo symbols are simply skipped.
    """

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:

        with contextlib.redirect_stdout(
            stdout_buffer
        ), contextlib.redirect_stderr(
            stderr_buffer
        ):

            result = yf.download(
                *args,
                **kwargs
            )

        return result

    except Exception:

        return pd.DataFrame()


# ============================================================
# FLATTEN YFINANCE COLUMNS
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

    # --------------------------------------------------------
    # 60 calendar-day buffer.
    #
    # This gives enough history for the 20-day volume SMA.
    # --------------------------------------------------------

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

        ignore_tz=True,
    )

    if df is None or df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Flatten MultiIndex.
    # --------------------------------------------------------

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
    # Normalize daily dates.
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
    # Convert columns to numeric.
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

    return df


# ============================================================
# FIND ALL DAILY CANDIDATES
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
    # Daily percentage change.
    # --------------------------------------------------------

    df["Day_Drop"] = (
        df["Close"]
        .pct_change()
    )

    # --------------------------------------------------------
    # Previous 20 completed sessions' volume average.
    #
    # Shift(1) is important:
    #
    # Today's volume is NOT included in today's average.
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
    # Get every trading day inside requested range.
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
    # CHECK EVERY DAY
    # ========================================================

    for position in positions:

        # Need one previous trading day.
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
        # T-1 FALL >= 1%
        # ----------------------------------------------------

        t1_ok = (
            float(t1_drop)
            <= -MIN_DAILY_FALL
        )

        # ----------------------------------------------------
        # T0 FALL >= 1%
        # ----------------------------------------------------

        t0_ok = (
            float(t0_drop)
            <= -MIN_DAILY_FALL
        )

        # ----------------------------------------------------
        # VOLUME >= 1.30x
        # ----------------------------------------------------

        volume_ok = (
            float(vol_ratio)
            >= VOLUME_MULTIPLIER
        )

        # ----------------------------------------------------
        # FINAL DAILY CONDITION
        # ----------------------------------------------------

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
    period=RSI_PERIOD
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

    # If average loss is zero,
    # RSI is 100.
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

        ignore_tz=False,
    )

    if df is None or df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Flatten MultiIndex.
    # --------------------------------------------------------

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
    # Datetime.
    # --------------------------------------------------------

    df.index = pd.to_datetime(
        df.index
    )

    # --------------------------------------------------------
    # Fix timezone.
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
    # Numeric.
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
    # Target date only.
    # --------------------------------------------------------

    df = df[
        df.index.date
        == target_date.date()
    ]

    # --------------------------------------------------------
    # NSE session.
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
    # First 15-minute candle.
    #
    # REQUIRED:
    # df.iloc[0]["Low"]
    # --------------------------------------------------------

    morning_low = (
        df.iloc[0]["Low"]
    )

    # --------------------------------------------------------
    # RSI.
    # --------------------------------------------------------

    df["RSI"] = calculate_rsi(
        df["Close"],
        RSI_PERIOD
    )

    # --------------------------------------------------------
    # MACD.
    # --------------------------------------------------------

    (
        df["MACD"],
        df["MACD_Signal"]
    ) = calculate_macd(
        df["Close"]
    )

    # --------------------------------------------------------
    # Scan every subsequent candle.
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

        # ----------------------------------------------------
        # PRICE BREAK
        # ----------------------------------------------------

        price_break = (
            float(close)
            < float(morning_low)
        )

        # ----------------------------------------------------
        # RSI FILTER
        # ----------------------------------------------------

        rsi_ok = (
            float(rsi)
            > 25
        )

        # ----------------------------------------------------
        # MACD BEARISH
        # ----------------------------------------------------

        macd_bearish = (
            float(macd)
            < float(signal)
        )

        # ----------------------------------------------------
        # VERIFIED TRIGGER
        # ----------------------------------------------------

        if (
            price_break
            and rsi_ok
            and macd_bearish
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

    if hours > 0:

        return (
            f"{hours}h "
            f"{minutes}m "
            f"{secs}s"
        )

    if minutes > 0:

        return (
            f"{minutes}m "
            f"{secs}s"
        )

    return f"{secs}s"


# ============================================================
# REMAINING TIME
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

        average_time = (
            elapsed
            / scanned
        )

        remaining_stocks = (
            total
            - scanned
        )

        eta = (
            average_time
            * remaining_stocks
        )

    else:

        eta = 0

    text = (
        f"Remaining: "
        f"{format_time(eta)}"
    )

    # Clear the current terminal line.
    sys.stdout.write(
        "\r\033[2K"
        + text
    )

    sys.stdout.flush()


# ============================================================
# PRINT DATE-WISE RESULTS
# ============================================================

def print_date_wise_results(
    daily_results,
    start_date,
    end_date
):

    print()
    print("=" * 110)
    print(
        "DATE-WISE FALLING STOCKS"
    )
    print("=" * 110)

    if not daily_results:

        print(
            "No qualifying stocks found."
        )

        return

    daily_df = pd.DataFrame(
        daily_results
    )

    daily_df["Date"] = pd.to_datetime(
        daily_df["Date"]
    )

    # --------------------------------------------------------
    # Create cell content.
    #
    # Example:
    #
    # -3.94% | Vol 1.51x
    #
    # This represents T0 fall and volume.
    # --------------------------------------------------------

    daily_df["Display"] = (
        daily_df["T0 Fall %"].map(
            lambda x:
            f"{x:.2f}%"
        )
        + " | Vol "
        + daily_df["Vol Ratio"].map(
            lambda x:
            f"{x:.2f}x"
        )
    )

    # --------------------------------------------------------
    # Pivot:
    #
    # Rows    = stocks
    # Columns = dates
    # --------------------------------------------------------

    pivot_df = daily_df.pivot_table(
        index="Ticker",
        columns="Date",
        values="Display",
        aggfunc="first"
    )

    # --------------------------------------------------------
    # Create ALL requested calendar-date columns.
    #
    # If user enters:
    #
    # 2026-09-07 -> 2026-09-10
    #
    # the output will have:
    #
    # 2026-09-07
    # 2026-09-08
    # 2026-09-09
    # 2026-09-10
    # --------------------------------------------------------

    requested_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    pivot_df = pivot_df.reindex(
        columns=requested_dates
    )

    # --------------------------------------------------------
    # Date headings.
    # --------------------------------------------------------

    pivot_df.columns = [
        date.strftime("%Y-%m-%d")
        for date in pivot_df.columns
    ]

    # --------------------------------------------------------
    # Missing = "-"
    # --------------------------------------------------------

    pivot_df = pivot_df.fillna(
        "-"
    )

    # --------------------------------------------------------
    # Sort stocks alphabetically.
    # --------------------------------------------------------

    pivot_df = pivot_df.sort_index()

    # --------------------------------------------------------
    # Print table.
    # --------------------------------------------------------

    print(
        pivot_df.to_string()
    )


# ============================================================
# PRINT VERIFIED INTRADAY RESULTS
# ============================================================

def print_intraday_results(
    intraday_results
):

    print()
    print("=" * 110)
    print(
        "VERIFIED INTRADAY SETUPS"
    )
    print("=" * 110)

    if not intraday_results:

        print(
            "No verified intraday setups found."
        )

        return

    df = pd.DataFrame(
        intraday_results
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df
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

    display = df[
        [
            "Ticker",
            "Date",
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


# ============================================================
# MAIN SCANNER
# ============================================================

def run_scanner(
    start_date,
    end_date
):

    tickers = build_universe()

    total_stocks = len(
        tickers
    )

    # --------------------------------------------------------
    # Current date.
    # --------------------------------------------------------

    today = (
        pd.Timestamp.now()
        .normalize()
    )

    # --------------------------------------------------------
    # Results.
    # --------------------------------------------------------

    daily_results = []

    intraday_results = []

    scanned = 0

    start_time = (
        time.perf_counter()
    )

    print()
    print(
        f"Scanning {total_stocks} stocks..."
    )

    # ========================================================
    # STOCK LOOP
    # ========================================================

    for ticker in tickers:

        try:

            # ------------------------------------------------
            # DAILY DATA
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

                # ------------------------------------------------
                # EVERY CANDIDATE DATE IS KEPT
                # ------------------------------------------------

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
                    # AGE FROM TODAY
                    # ------------------------------------------------

                    age_days = (
                        today
                        - target_date
                    ).days

                    # ------------------------------------------------
                    # If older than 58 days:
                    #
                    # Do NOT request 15m data.
                    # ------------------------------------------------

                    if (
                        age_days
                        > INTRADAY_MAX_AGE_DAYS
                    ):

                        continue

                    # ------------------------------------------------
                    # INTRADAY
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
                    # VERIFIED TRIGGER
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
                "\n\nScan stopped by user."
            )

            raise

        except Exception:

            # ------------------------------------------------
            # Bad Yahoo symbols such as DUMMYHEG.NS are
            # silently skipped.
            # ------------------------------------------------

            pass

        # ----------------------------------------------------
        # UPDATE ONLY REMAINING TIME
        # ----------------------------------------------------

        scanned += 1

        update_remaining(
            scanned,
            total_stocks,
            start_time
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    # ========================================================
    # SCAN COMPLETE
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # Move below the progress line.
    print()

    # ========================================================
    # DATE-WISE DAILY RESULTS
    # ========================================================

    print_date_wise_results(
        daily_results,
        start_date,
        end_date
    )

    # ========================================================
    # INTRADAY RESULTS
    # ========================================================

    print_intraday_results(
        intraday_results
    )

    # ========================================================
    # FINAL SHORT SUMMARY
    # ========================================================

    print()
    print("=" * 110)

    print(
        f"Stocks scanned: "
        f"{total_stocks}"
    )

    print(
        f"Qualifying stock-days: "
        f"{len(daily_results)}"
    )

    print(
        f"Verified intraday setups: "
        f"{len(intraday_results)}"
    )

    print(
        f"Total runtime: "
        f"{format_time(elapsed)}"
    )

    print("=" * 110)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print(
        "NIFTY 500 DATE-WISE FALLING STOCK SCANNER"
    )
    print("=" * 110)

    print(
        "Daily rule:"
    )

    print(
        "  T-1 must fall >= 1%"
    )

    print(
        "  T0 must fall >= 1%"
    )

    print(
        "  T0 volume must be >= 1.30x previous 20-session average"
    )

    print()
    print(
        "Every qualifying date in your input range is retained."
    )

    print(
        "Maximum recommended input range: 5 calendar days."
    )

    print("=" * 110)

    try:

        # ----------------------------------------------------
        # USER DATE INPUT
        # ----------------------------------------------------

        start_date, end_date = (
            get_date_range()
        )

        # ----------------------------------------------------
        # RUN
        # ----------------------------------------------------

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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

