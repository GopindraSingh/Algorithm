"""
====================================================================
NIFTY 500 MOMENTUM REVERSAL / INTRADAY SHORT SCANNER
TKINTER GUI VERSION
====================================================================

DAILY CONDITIONS
----------------
T-1 fall >= 1%
T0  fall >= 1%
T0 volume >= 1.30x previous 20-session average volume

INTRADAY CONDITIONS
-------------------
First 15-minute candle:
    09:15 - 09:30

morning_low:
    Low of first 15-minute candle

Trigger:
    Close < morning_low
    RSI(14) > 25
    MACD(12,26,9) < Signal

Only the FIRST qualifying intraday candle is recorded.

EFV
---
Estimated Fair Value is a market-based estimate.

It uses:
    20-day VWAP
    20-day SMA
    50-day SMA
    100-day SMA
    ATR-adjusted price

Expected Fall:
    (Current Price - EFV) / Current Price * 100

IMPORTANT:
EFV is NOT guaranteed intrinsic/fundamental value.

====================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import threading
import time
import io
import urllib.request
import contextlib

import numpy as np
import pandas as pd
import yfinance as yf


# ====================================================================
# CONFIGURATION
# ====================================================================

NIFTY_500_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
)

NIFTY_50_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
)

MIN_DAILY_FALL = 0.01
VOLUME_SMA_PERIOD = 20
VOLUME_MULTIPLIER = 1.30

INTRADAY_MAX_AGE_DAYS = 58

RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

SMA_SHORT = 20
SMA_MEDIUM = 50
SMA_LONG = 100

VWAP_PERIOD = 20
ATR_PERIOD = 14

MARKET_TIMEZONE = "Asia/Kolkata"

MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

REQUEST_DELAY_SECONDS = 0.05


# ====================================================================
# QUIET YFINANCE
# ====================================================================

def quiet_yfinance_download(*args, **kwargs):
    """
    Suppress yfinance's noisy terminal output.
    """

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            with contextlib.redirect_stderr(stderr_buffer):

                data = yf.download(
                    *args,
                    **kwargs
                )

        return data

    except Exception:
        return pd.DataFrame()


# ====================================================================
# FLATTEN YFINANCE MULTI-INDEX
# ====================================================================

def flatten_columns(df):

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


# ====================================================================
# NSE CSV
# ====================================================================

def download_nse_csv(url):

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

    possible = [
        "Symbol",
        "SYMBOL",
        "symbol",
        "Ticker",
        "TICKER"
    ]

    for column in possible:

        if column in df.columns:
            return column

    raise ValueError(
        "Symbol column was not found in NSE file."
    )


def get_index_symbols(url):

    df = download_nse_csv(url)

    column = find_symbol_column(df)

    symbols = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return set(symbols)


def build_universe():

    nifty500 = get_index_symbols(
        NIFTY_500_URL
    )

    nifty50 = get_index_symbols(
        NIFTY_50_URL
    )

    symbols = sorted(
        nifty500 - nifty50
    )

    tickers = []

    for symbol in symbols:

        if symbol.endswith(".NS"):
            ticker = symbol
        else:
            ticker = symbol + ".NS"

        tickers.append(ticker)

    return tickers


# ====================================================================
# DAILY DATA
# ====================================================================

def download_daily_data(
    ticker,
    start_date,
    end_date
):

    buffered_start = (
        start_date
        - pd.Timedelta(days=180)
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

        ignore_tz=True
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


# ====================================================================
# DAILY INDICATORS
# ====================================================================

def calculate_daily_indicators(df):

    df = df.copy()

    # ---------------------------------------------------------------
    # Daily percentage change
    # ---------------------------------------------------------------

    df["Day_Drop"] = (
        df["Close"]
        .pct_change()
    )

    # ---------------------------------------------------------------
    # Previous 20 completed sessions volume average
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Moving averages
    # ---------------------------------------------------------------

    df["SMA20"] = (
        df["Close"]
        .rolling(
            SMA_SHORT,
            min_periods=SMA_SHORT
        )
        .mean()
    )

    df["SMA50"] = (
        df["Close"]
        .rolling(
            SMA_MEDIUM,
            min_periods=SMA_MEDIUM
        )
        .mean()
    )

    df["SMA100"] = (
        df["Close"]
        .rolling(
            SMA_LONG,
            min_periods=SMA_LONG
        )
        .mean()
    )

    # ---------------------------------------------------------------
    # True Range
    # ---------------------------------------------------------------

    previous_close = (
        df["Close"]
        .shift(1)
    )

    tr1 = (
        df["High"]
        - df["Low"]
    )

    tr2 = (
        df["High"]
        - previous_close
    ).abs()

    tr3 = (
        df["Low"]
        - previous_close
    ).abs()

    df["TR"] = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(
        axis=1
    )

    # ---------------------------------------------------------------
    # ATR
    # ---------------------------------------------------------------

    df["ATR14"] = (
        df["TR"]
        .rolling(
            ATR_PERIOD,
            min_periods=ATR_PERIOD
        )
        .mean()
    )

    # ---------------------------------------------------------------
    # Typical price
    # ---------------------------------------------------------------

    df["Typical_Price"] = (
        df["High"]
        + df["Low"]
        + df["Close"]
    ) / 3.0

    # ---------------------------------------------------------------
    # Rolling 20-day VWAP
    # ---------------------------------------------------------------

    df["PV"] = (
        df["Typical_Price"]
        * df["Volume"]
    )

    rolling_pv = (
        df["PV"]
        .rolling(
            VWAP_PERIOD,
            min_periods=VWAP_PERIOD
        )
        .sum()
    )

    rolling_volume = (
        df["Volume"]
        .rolling(
            VWAP_PERIOD,
            min_periods=VWAP_PERIOD
        )
        .sum()
    )

    df["VWAP20"] = (
        rolling_pv
        / rolling_volume
    )

    # ---------------------------------------------------------------
    # Volatility-adjusted anchor
    # ---------------------------------------------------------------

    df["Volatility_Anchor"] = (
        df["Close"]
        - df["ATR14"]
    )

    # ---------------------------------------------------------------
    # EFV
    #
    # 30% 20-day VWAP
    # 25% 20-day SMA
    # 20% 50-day SMA
    # 15% 100-day SMA
    # 10% ATR-adjusted anchor
    # ---------------------------------------------------------------

    df["EFV"] = (

        0.30 * df["VWAP20"]

        + 0.25 * df["SMA20"]

        + 0.20 * df["SMA50"]

        + 0.15 * df["SMA100"]

        + 0.10 * df["Volatility_Anchor"]
    )

    # ---------------------------------------------------------------
    # Expected Fall
    # ---------------------------------------------------------------

    df["Expected_Fall"] = (
        (
            df["Close"]
            - df["EFV"]
        )
        / df["Close"]
    ) * 100.0

    df["Expected_Fall"] = (
        df["Expected_Fall"]
        .clip(lower=0)
    )

    return df


# ====================================================================
# DAILY CANDIDATES
# ====================================================================

def find_daily_candidates(
    ticker,
    df,
    start_date,
    end_date
):

    if df.empty:
        return []

    df = calculate_daily_indicators(
        df
    )

    mask = (
        (df.index >= start_date)
        &
        (df.index <= end_date)
    )

    positions = np.flatnonzero(
        np.asarray(mask)
    )

    candidates = []

    for position in positions:

        if position < 1:
            continue

        previous_day = df.iloc[
            position - 1
        ]

        current_day = df.iloc[
            position
        ]

        t1_drop = previous_day[
            "Day_Drop"
        ]

        t0_drop = current_day[
            "Day_Drop"
        ]

        volume_ratio = current_day[
            "Vol_Ratio"
        ]

        current_price = current_day[
            "Close"
        ]

        efv = current_day[
            "EFV"
        ]

        expected_fall = current_day[
            "Expected_Fall"
        ]

        if pd.isna(t1_drop):
            continue

        if pd.isna(t0_drop):
            continue

        if pd.isna(volume_ratio):
            continue

        if pd.isna(current_price):
            continue

        if pd.isna(efv):
            continue

        if pd.isna(expected_fall):
            continue

        # T-1 must fall >= 1%
        condition_t1 = (
            float(t1_drop)
            <= -MIN_DAILY_FALL
        )

        # T0 must fall >= 1%
        condition_t0 = (
            float(t0_drop)
            <= -MIN_DAILY_FALL
        )

        # T0 volume >= 1.30x
        condition_volume = (
            float(volume_ratio)
            >= VOLUME_MULTIPLIER
        )

        if (
            condition_t1
            and condition_t0
            and condition_volume
        ):

            candidates.append(
                {
                    "Ticker": ticker,

                    "Date": df.index[
                        position
                    ],

                    "T1_Fall": (
                        float(t1_drop)
                        * 100.0
                    ),

                    "T0_Fall": (
                        float(t0_drop)
                        * 100.0
                    ),

                    "Vol_Ratio": (
                        float(volume_ratio)
                    ),

                    "Current_Price": (
                        float(current_price)
                    ),

                    "EFV": (
                        float(efv)
                    ),

                    "Expected_Fall": (
                        float(expected_fall)
                    )
                }
            )

    return candidates


# ====================================================================
# RSI
# ====================================================================

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

    average_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    average_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    rs = (
        average_gain
        / average_loss
    )

    rsi = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    rsi = rsi.where(
        average_loss != 0,
        100
    )

    return rsi


# ====================================================================
# MACD
# ====================================================================

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


# ====================================================================
# 15-MINUTE DATA
# ====================================================================

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

        ignore_tz=False
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

    # ---------------------------------------------------------------
    # Correct timezone handling
    # ---------------------------------------------------------------

    try:

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

    except Exception:

        return pd.DataFrame()

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

    # ---------------------------------------------------------------
    # Keep requested date only
    # ---------------------------------------------------------------

    df = df[
        df.index.date
        == target_date.date()
    ]

    # ---------------------------------------------------------------
    # NSE session
    # ---------------------------------------------------------------

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


# ====================================================================
# INTRADAY ANALYSIS
# ====================================================================

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

    # ================================================================
    # FIRST 15-MINUTE CANDLE
    # ================================================================

    # Explicit integer-location extraction.
    morning_low = df.iloc[0]["Low"]

    # ================================================================
    # RSI
    # ================================================================

    df["RSI"] = calculate_rsi(
        df["Close"],
        RSI_PERIOD
    )

    # ================================================================
    # MACD
    # ================================================================

    (
        df["MACD"],
        df["MACD_Signal"]
    ) = calculate_macd(
        df["Close"]
    )

    # ================================================================
    # SCAN FROM SECOND CANDLE
    # ================================================================

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

        price_break = (
            float(close)
            < float(morning_low)
        )

        rsi_ok = (
            float(rsi)
            > 25
        )

        macd_bearish = (
            float(macd)
            < float(signal)
        )

        if (
            price_break
            and rsi_ok
            and macd_bearish
        ):

            return {
                "Trigger_Time":
                    df.index[i].strftime(
                        "%H:%M"
                    ),

                "Breakout_Price":
                    float(close),

                "Morning_Low":
                    float(morning_low),

                "RSI":
                    float(rsi)
            }

    return None


# ====================================================================
# FORMAT TIME
# ====================================================================

def format_seconds(seconds):

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


# ====================================================================
# GUI APPLICATION
# ====================================================================

class NiftyScannerGUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Nifty 500 Momentum Reversal Scanner"
        )

        self.root.geometry(
            "1500x850"
        )

        self.root.minsize(
            1100,
            650
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.exit_application
        )

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------

        self.scanning = False

        self.stop_requested = False

        self.scan_thread = None

        self.start_time = None

        self.total_stocks = 0

        self.scanned_stocks = 0

        self.daily_results = []

        self.intraday_results = []

        # ------------------------------------------------------------
        # Variables
        # ------------------------------------------------------------

        self.start_date_var = tk.StringVar()

        self.end_date_var = tk.StringVar()

        self.status_var = tk.StringVar(
            value="Ready"
        )

        self.progress_var = tk.DoubleVar(
            value=0
        )

        self.scanned_var = tk.StringVar(
            value="Stocks scanned: 0 / 0"
        )

        self.candidate_var = tk.StringVar(
            value="Candidates: 0"
        )

        self.trigger_var = tk.StringVar(
            value="Triggers: 0"
        )

        self.remaining_var = tk.StringVar(
            value="Remaining: --"
        )

        # ------------------------------------------------------------
        # Style
        # ------------------------------------------------------------

        self.setup_style()

        # ------------------------------------------------------------
        # Build GUI
        # ------------------------------------------------------------

        self.build_gui()

    # =================================================================
    # STYLE
    # =================================================================

    def setup_style(self):

        style = ttk.Style()

        try:
            style.theme_use(
                "clam"
            )
        except Exception:
            pass

        style.configure(
            "Title.TLabel",
            font=(
                "Segoe UI",
                18,
                "bold"
            )
        )

        style.configure(
            "Header.TLabel",
            font=(
                "Segoe UI",
                11,
                "bold"
            )
        )

        style.configure(
            "Status.TLabel",
            font=(
                "Segoe UI",
                11
            )
        )

        style.configure(
            "Treeview",
            rowheight=28,
            font=(
                "Segoe UI",
                9
            )
        )

        style.configure(
            "Treeview.Heading",
            font=(
                "Segoe UI",
                9,
                "bold"
            )
        )

    # =================================================================
    # BUILD GUI
    # =================================================================

    def build_gui(self):

        # ------------------------------------------------------------
        # Main container
        # ------------------------------------------------------------

        main = ttk.Frame(
            self.root,
            padding=12
        )

        main.pack(
            fill="both",
            expand=True
        )

        # ------------------------------------------------------------
        # Title
        # ------------------------------------------------------------

        title = ttk.Label(
            main,
            text=(
                "Nifty 500 Momentum Reversal "
                "Research Scanner"
            ),
            style="Title.TLabel"
        )

        title.pack(
            anchor="w",
            pady=(0, 12)
        )

        # ------------------------------------------------------------
        # Input frame
        # ------------------------------------------------------------

        input_frame = ttk.LabelFrame(
            main,
            text="Scan Settings",
            padding=10
        )

        input_frame.pack(
            fill="x",
            pady=(0, 10)
        )

        ttk.Label(
            input_frame,
            text="Start Date:"
        ).grid(
            row=0,
            column=0,
            padx=(0, 6),
            pady=5,
            sticky="w"
        )

        self.start_entry = ttk.Entry(
            input_frame,
            textvariable=self.start_date_var,
            width=15
        )

        self.start_entry.grid(
            row=0,
            column=1,
            padx=5,
            pady=5
        )

        ttk.Label(
            input_frame,
            text="End Date:"
        ).grid(
            row=0,
            column=2,
            padx=(20, 6),
            pady=5,
            sticky="w"
        )

        self.end_entry = ttk.Entry(
            input_frame,
            textvariable=self.end_date_var,
            width=15
        )

        self.end_entry.grid(
            row=0,
            column=3,
            padx=5,
            pady=5
        )

        ttk.Label(
            input_frame,
            text="Format: YYYY-MM-DD"
        ).grid(
            row=0,
            column=4,
            padx=10,
            pady=5,
            sticky="w"
        )

        # ------------------------------------------------------------
        # Buttons
        # ------------------------------------------------------------

        self.start_button = ttk.Button(
            input_frame,
            text="START SCAN",
            command=self.start_scan
        )

        self.start_button.grid(
            row=0,
            column=5,
            padx=(25, 5),
            pady=5
        )

        self.stop_button = ttk.Button(
            input_frame,
            text="STOP",
            command=self.stop_scan,
            state="disabled"
        )

        self.stop_button.grid(
            row=0,
            column=6,
            padx=5,
            pady=5
        )

        self.exit_button = ttk.Button(
            input_frame,
            text="EXIT",
            command=self.exit_application
        )

        self.exit_button.grid(
            row=0,
            column=7,
            padx=(5, 0),
            pady=5
        )

        # ------------------------------------------------------------
        # Information frame
        # ------------------------------------------------------------

        info_frame = ttk.Frame(
            main
        )

        info_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        ttk.Label(
            info_frame,
            textvariable=self.scanned_var,
            style="Status.TLabel"
        ).pack(
            side="left",
            padx=(0, 25)
        )

        ttk.Label(
            info_frame,
            textvariable=self.candidate_var,
            style="Status.TLabel"
        ).pack(
            side="left",
            padx=(0, 25)
        )

        ttk.Label(
            info_frame,
            textvariable=self.trigger_var,
            style="Status.TLabel"
        ).pack(
            side="left",
            padx=(0, 25)
        )

        ttk.Label(
            info_frame,
            textvariable=self.remaining_var,
            style="Status.TLabel"
        ).pack(
            side="right"
        )

        # ------------------------------------------------------------
        # Progress bar
        # ------------------------------------------------------------

        self.progress = ttk.Progressbar(
            main,
            variable=self.progress_var,
            maximum=100,
            mode="determinate"
        )

        self.progress.pack(
            fill="x",
            pady=(0, 5)
        )

        # ------------------------------------------------------------
        # Current status
        # ------------------------------------------------------------

        status_frame = ttk.Frame(
            main
        )

        status_frame.pack(
            fill="x",
            pady=(0, 8)
        )

        ttk.Label(
            status_frame,
            text="Status:"
        ).pack(
            side="left"
        )

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="Status.TLabel"
        ).pack(
            side="left",
            padx=8
        )

        # ------------------------------------------------------------
        # Notebook
        # ------------------------------------------------------------

        self.notebook = ttk.Notebook(
            main
        )

        self.notebook.pack(
            fill="both",
            expand=True
        )

        # ------------------------------------------------------------
        # Daily results tab
        # ------------------------------------------------------------

        daily_frame = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            daily_frame,
            text="Qualifying Falling Days"
        )

        daily_columns = (
            "Ticker",
            "Date",
            "T1 Fall %",
            "T0 Fall %",
            "Volume",
            "Price",
            "EFV",
            "Expected Fall"
        )

        self.daily_tree = ttk.Treeview(
            daily_frame,
            columns=daily_columns,
            show="headings"
        )

        daily_widths = {
            "Ticker": 130,
            "Date": 110,
            "T1 Fall %": 100,
            "T0 Fall %": 100,
            "Volume": 90,
            "Price": 100,
            "EFV": 100,
            "Expected Fall": 120
        }

        for column in daily_columns:

            self.daily_tree.heading(
                column,
                text=column
            )

            self.daily_tree.column(
                column,
                width=daily_widths[column],
                anchor="center"
            )

        daily_scroll_y = ttk.Scrollbar(
            daily_frame,
            orient="vertical",
            command=self.daily_tree.yview
        )

        daily_scroll_x = ttk.Scrollbar(
            daily_frame,
            orient="horizontal",
            command=self.daily_tree.xview
        )

        self.daily_tree.configure(
            yscrollcommand=daily_scroll_y.set,
            xscrollcommand=daily_scroll_x.set
        )

        self.daily_tree.pack(
            side="left",
            fill="both",
            expand=True
        )

        daily_scroll_y.pack(
            side="right",
            fill="y"
        )

        daily_scroll_x.pack(
            side="bottom",
            fill="x"
        )

        # ------------------------------------------------------------
        # Intraday results tab
        # ------------------------------------------------------------

        intraday_frame = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            intraday_frame,
            text="Verified Intraday Setups"
        )

        intraday_columns = (
            "Ticker",
            "Date",
            "Trigger Time",
            "Morning Low",
            "Breakout Price",
            "RSI",
            "T0 Fall %",
            "Volume",
            "EFV",
            "Expected Fall"
        )

        self.intraday_tree = ttk.Treeview(
            intraday_frame,
            columns=intraday_columns,
            show="headings"
        )

        intraday_widths = {
            "Ticker": 120,
            "Date": 105,
            "Trigger Time": 105,
            "Morning Low": 110,
            "Breakout Price": 120,
            "RSI": 80,
            "T0 Fall %": 100,
            "Volume": 90,
            "EFV": 100,
            "Expected Fall": 120
        }

        for column in intraday_columns:

            self.intraday_tree.heading(
                column,
                text=column
            )

            self.intraday_tree.column(
                column,
                width=intraday_widths[column],
                anchor="center"
            )

        intraday_scroll_y = ttk.Scrollbar(
            intraday_frame,
            orient="vertical",
            command=self.intraday_tree.yview
        )

        intraday_scroll_x = ttk.Scrollbar(
            intraday_frame,
            orient="horizontal",
            command=self.intraday_tree.xview
        )

        self.intraday_tree.configure(
            yscrollcommand=intraday_scroll_y.set,
            xscrollcommand=intraday_scroll_x.set
        )

        self.intraday_tree.pack(
            side="left",
            fill="both",
            expand=True
        )

        intraday_scroll_y.pack(
            side="right",
            fill="y"
        )

        intraday_scroll_x.pack(
            side="bottom",
            fill="x"
        )

        # ------------------------------------------------------------
        # Summary tab
        # ------------------------------------------------------------

        summary_frame = ttk.Frame(
            self.notebook,
            padding=20
        )

        self.notebook.add(
            summary_frame,
            text="Summary"
        )

        self.summary_text = tk.Text(
            summary_frame,
            wrap="word",
            font=(
                "Consolas",
                11
            ),
            state="disabled"
        )

        self.summary_text.pack(
            fill="both",
            expand=True
        )

        # ------------------------------------------------------------
        # Default dates
        # ------------------------------------------------------------

        today = datetime.now()

        self.end_date_var.set(
            today.strftime(
                "%Y-%m-%d"
            )
        )

        self.start_date_var.set(
            today.strftime(
                "%Y-%m-%d"
            )
        )

    # =================================================================
    # DATE VALIDATION
    # =================================================================

    def get_dates(self):

        start_text = (
            self.start_date_var
            .get()
            .strip()
        )

        end_text = (
            self.end_date_var
            .get()
            .strip()
        )

        try:

            start_date = pd.to_datetime(
                start_text,
                format="%Y-%m-%d"
            ).normalize()

            end_date = pd.to_datetime(
                end_text,
                format="%Y-%m-%d"
            ).normalize()

        except Exception:

            raise ValueError(
                "Dates must use YYYY-MM-DD format."
            )

        if end_date < start_date:

            raise ValueError(
                "End date cannot be earlier than start date."
            )

        days = (
            end_date
            - start_date
        ).days + 1

        if days > 5:

            raise ValueError(
                "Please use a maximum of 5 calendar days."
            )

        return start_date, end_date

    # =================================================================
    # START SCAN
    # =================================================================

    def start_scan(self):

        if self.scanning:

            return

        try:

            start_date, end_date = (
                self.get_dates()
            )

        except ValueError as error:

            messagebox.showerror(
                "Invalid Date",
                str(error)
            )

            return

        # ------------------------------------------------------------
        # Reset state
        # ------------------------------------------------------------

        self.scanning = True

        self.stop_requested = False

        self.daily_results = []

        self.intraday_results = []

        self.scanned_stocks = 0

        self.total_stocks = 0

        self.start_time = time.perf_counter()

        # ------------------------------------------------------------
        # Clear tables
        # ------------------------------------------------------------

        self.clear_results()

        # ------------------------------------------------------------
        # Buttons
        # ------------------------------------------------------------

        self.start_button.configure(
            state="disabled"
        )

        self.stop_button.configure(
            state="normal"
        )

        # ------------------------------------------------------------
        # Status
        # ------------------------------------------------------------

        self.status_var.set(
            "Loading Nifty 500 universe..."
        )

        self.scanned_var.set(
            "Stocks scanned: 0 / 0"
        )

        self.candidate_var.set(
            "Candidates: 0"
        )

        self.trigger_var.set(
            "Triggers: 0"
        )

        self.remaining_var.set(
            "Remaining: calculating..."
        )

        self.progress_var.set(
            0
        )

        # ------------------------------------------------------------
        # Background thread
        # ------------------------------------------------------------

        self.scan_thread = threading.Thread(
            target=self.scan_worker,
            args=(
                start_date,
                end_date
            ),
            daemon=True
        )

        self.scan_thread.start()

        self.root.after(
            250,
            self.update_gui_progress
        )

    # =================================================================
    # SCAN WORKER
    # =================================================================

    def scan_worker(
        self,
        start_date,
        end_date
    ):

        try:

            tickers = build_universe()

            self.total_stocks = len(
                tickers
            )

            if self.total_stocks == 0:

                raise RuntimeError(
                    "No Nifty 500 stocks were loaded."
                )

            for ticker in tickers:

                if self.stop_requested:

                    break

                try:

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

                        for candidate in candidates:

                            if self.stop_requested:
                                break

                            self.daily_results.append(
                                candidate
                            )

                            target_date = (
                                pd.Timestamp(
                                    candidate["Date"]
                                ).normalize()
                            )

                            today = (
                                pd.Timestamp.now()
                                .normalize()
                            )

                            age_days = (
                                today
                                - target_date
                            ).days

                            # ------------------------------------------------
                            # Yahoo 58-day protection
                            # ------------------------------------------------

                            if (
                                age_days
                                > INTRADAY_MAX_AGE_DAYS
                            ):

                                continue

                            # ------------------------------------------------
                            # Intraday
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

                            if intraday is not None:

                                self.intraday_results.append(
                                    {
                                        "Ticker":
                                            ticker,

                                        "Date":
                                            target_date,

                                        "Trigger_Time":
                                            intraday[
                                                "Trigger_Time"
                                            ],

                                        "Morning_Low":
                                            intraday[
                                                "Morning_Low"
                                            ],

                                        "Breakout_Price":
                                            intraday[
                                                "Breakout_Price"
                                            ],

                                        "RSI":
                                            intraday[
                                                "RSI"
                                            ],

                                        "T0_Fall":
                                            candidate[
                                                "T0_Fall"
                                            ],

                                        "Vol_Ratio":
                                            candidate[
                                                "Vol_Ratio"
                                            ],

                                        "EFV":
                                            candidate[
                                                "EFV"
                                            ],

                                        "Expected_Fall":
                                            candidate[
                                                "Expected_Fall"
                                            ]
                                    }
                                )

                except Exception:

                    # ------------------------------------------------
                    # Invalid/delisted/time-out symbol.
                    # Do not crash entire scan.
                    # ------------------------------------------------

                    pass

                self.scanned_stocks += 1

                time.sleep(
                    REQUEST_DELAY_SECONDS
                )

            # --------------------------------------------------------
            # Finished
            # --------------------------------------------------------

            self.root.after(
                0,
                self.scan_finished
            )

        except Exception as error:

            self.root.after(
                0,
                lambda e=str(error):
                self.scan_failed(e)
            )

    # =================================================================
    # UPDATE GUI
    # =================================================================

    def update_gui_progress(self):

        if not self.scanning:

            return

        scanned = self.scanned_stocks

        total = self.total_stocks

        if total > 0:

            percentage = (
                scanned
                / total
                * 100
            )

            self.progress_var.set(
                percentage
            )

        self.scanned_var.set(
            f"Stocks scanned: "
            f"{scanned} / {total}"
        )

        self.candidate_var.set(
            f"Candidates: "
            f"{len(self.daily_results)}"
        )

        self.trigger_var.set(
            f"Triggers: "
            f"{len(self.intraday_results)}"
        )

        # ------------------------------------------------------------
        # ETA
        # ------------------------------------------------------------

        if (
            scanned > 0
            and total > scanned
            and self.start_time is not None
        ):

            elapsed = (
                time.perf_counter()
                - self.start_time
            )

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

            self.remaining_var.set(
                "Remaining: "
                + format_seconds(eta)
            )

        elif total > 0 and scanned >= total:

            self.remaining_var.set(
                "Remaining: 0s"
            )

        # ------------------------------------------------------------
        # Current status
        # ------------------------------------------------------------

        if total > 0 and scanned < total:

            self.status_var.set(
                "Scanning stocks..."
            )

        # ------------------------------------------------------------
        # Continue timer
        # ------------------------------------------------------------

        self.root.after(
            250,
            self.update_gui_progress
        )

    # =================================================================
    # STOP
    # =================================================================

    def stop_scan(self):

        if not self.scanning:

            return

        self.stop_requested = True

        self.status_var.set(
            "Stopping scan after current request..."
        )

        self.stop_button.configure(
            state="disabled"
        )

    # =================================================================
    # SCAN FINISHED
    # =================================================================

    def scan_finished(self):

        self.scanning = False

        self.start_button.configure(
            state="normal"
        )

        self.stop_button.configure(
            state="disabled"
        )

        self.progress_var.set(
            100
            if (
                self.total_stocks > 0
                and self.scanned_stocks
                >= self.total_stocks
            )
            else self.progress_var.get()
        )

        elapsed = (
            time.perf_counter()
            - self.start_time
        )

        if self.stop_requested:

            self.status_var.set(
                "Scan stopped."
            )

        else:

            self.status_var.set(
                "Scan complete."
            )

        self.remaining_var.set(
            "Remaining: 0s"
        )

        self.update_daily_table()

        self.update_intraday_table()

        self.update_summary(
            elapsed
        )

        # Automatically show results
        self.notebook.select(
            0
        )

    # =================================================================
    # SCAN FAILED
    # =================================================================

    def scan_failed(
        self,
        error
    ):

        self.scanning = False

        self.start_button.configure(
            state="normal"
        )

        self.stop_button.configure(
            state="disabled"
        )

        self.status_var.set(
            "Scan failed."
        )

        messagebox.showerror(
            "Scanner Error",
            error
        )

    # =================================================================
    # CLEAR RESULTS
    # =================================================================

    def clear_results(self):

        for item in self.daily_tree.get_children():

            self.daily_tree.delete(
                item
            )

        for item in self.intraday_tree.get_children():

            self.intraday_tree.delete(
                item
            )

        self.summary_text.configure(
            state="normal"
        )

        self.summary_text.delete(
            "1.0",
            "end"
        )

        self.summary_text.configure(
            state="disabled"
        )

    # =================================================================
    # DAILY TABLE
    # =================================================================

    def update_daily_table(self):

        for item in self.daily_tree.get_children():

            self.daily_tree.delete(
                item
            )

        if not self.daily_results:

            return

        results = sorted(
            self.daily_results,
            key=lambda x: (
                x["Date"],
                x["Ticker"]
            )
        )

        for row in results:

            self.daily_tree.insert(
                "",
                "end",
                values=(
                    row["Ticker"],

                    pd.Timestamp(
                        row["Date"]
                    ).strftime(
                        "%Y-%m-%d"
                    ),

                    f'{row["T1_Fall"]:.2f}%',

                    f'{row["T0_Fall"]:.2f}%',

                    f'{row["Vol_Ratio"]:.2f}x',

                    f'₹{row["Current_Price"]:.2f}',

                    f'₹{row["EFV"]:.2f}',

                    f'{row["Expected_Fall"]:.2f}%'
                )
            )

    # =================================================================
    # INTRADAY TABLE
    # =================================================================

    def update_intraday_table(self):

        for item in self.intraday_tree.get_children():

            self.intraday_tree.delete(
                item
            )

        if not self.intraday_results:

            return

        results = sorted(
            self.intraday_results,
            key=lambda x: (
                x["Date"],
                x["Ticker"]
            )
        )

        for row in results:

            self.intraday_tree.insert(
                "",
                "end",
                values=(
                    row["Ticker"],

                    pd.Timestamp(
                        row["Date"]
                    ).strftime(
                        "%Y-%m-%d"
                    ),

                    row["Trigger_Time"],

                    f'₹{row["Morning_Low"]:.2f}',

                    f'₹{row["Breakout_Price"]:.2f}',

                    f'{row["RSI"]:.2f}',

                    f'{row["T0_Fall"]:.2f}%',

                    f'{row["Vol_Ratio"]:.2f}x',

                    f'₹{row["EFV"]:.2f}',

                    f'{row["Expected_Fall"]:.2f}%'
                )
            )

    # =================================================================
    # SUMMARY
    # =================================================================

    def update_summary(
        self,
        elapsed
    ):

        self.summary_text.configure(
            state="normal"
        )

        self.summary_text.delete(
            "1.0",
            "end"
        )

        status = (
            "STOPPED"
            if self.stop_requested
            else "COMPLETE"
        )

        text = f"""
======================================================================
SCAN {status}
======================================================================

Stocks scanned:
    {self.scanned_stocks} / {self.total_stocks}

Qualifying stock-days:
    {len(self.daily_results)}

Verified 15-minute setups:
    {len(self.intraday_results)}

Runtime:
    {format_seconds(elapsed)}

======================================================================
DAILY CONDITIONS
======================================================================

T-1 fall:
    >= 1%

T0 fall:
    >= 1%

T0 volume:
    >= 1.30x previous 20-session average

======================================================================
INTRADAY CONDITIONS
======================================================================

Opening candle:
    09:15 - 09:30

Breakout:
    Close < Morning Low

RSI:
    RSI(14) > 25

MACD:
    MACD(12,26,9) < Signal

Only the first qualifying intraday candle is recorded.

======================================================================
EFV
======================================================================

EFV combines:

    20-day VWAP
    20-day SMA
    50-day SMA
    100-day SMA
    ATR-adjusted price

Expected Fall:

    (Current Price - EFV)
    ----------------------
        Current Price

    x 100

IMPORTANT:
EFV is a market-based estimate.
It is NOT guaranteed intrinsic value.

======================================================================
"""

        self.summary_text.insert(
            "1.0",
            text
        )

        self.summary_text.configure(
            state="disabled"
        )

    # =================================================================
    # EXIT
    # =================================================================

    def exit_application(self):

        if self.scanning:

            answer = messagebox.askyesno(
                "Exit",
                (
                    "A scan is currently running.\n\n"
                    "Do you want to stop it and exit?"
                )
            )

            if not answer:

                return

            self.stop_requested = True

        self.root.destroy()


# ====================================================================
# MAIN
# ====================================================================

def main():

    root = tk.Tk()

    app = NiftyScannerGUI(
        root
    )

    root.mainloop()


if __name__ == "__main__":

    main()

