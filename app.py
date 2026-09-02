import json
import os
from datetime import datetime, timedelta, timezone
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

st.set_page_config(
    page_title="Thesis Tester",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    .metric-card {background: #f7f8fa; border: 1px solid #e5e7eb;
                  border-radius: 12px; padding: 14px 16px;}
    .small-note {color: #667085; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_number(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(value):
    value = clean_number(value)
    return None if value is None else round(value * 100, 2)


def compact(value):
    value = clean_number(value)
    if value is None:
        return "N/A"
    absolute = abs(value)
    if absolute >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,.2f}"


def stooq_history(symbol):
    """Return one year of daily closes from Stooq as a Yahoo fallback."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=370)
    stooq_symbol = f"{symbol.lower()}.us"
    url = (
        "https://stooq.com/q/d/l/"
        f"?s={stooq_symbol}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
    )
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 ThesisTester/1.0"},
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if frame.empty or "Date" not in frame or "Close" not in frame:
        raise ValueError("backup source returned no price history")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True)
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    close = frame.dropna(subset=["Date", "Close"]).set_index("Date")["Close"]
    if close.empty:
        raise ValueError("backup source returned no usable closing prices")
    return close.sort_index()


def price_changes(close):
    current = clean_number(close.iloc[-1])
    change_1m = (current / clean_number(close.iloc[-22])) - 1 if len(close) > 21 else None
    change_3m = (current / clean_number(close.iloc[-64])) - 1 if len(close) > 63 else None
    change_1y = (current / clean_number(close.iloc[0])) - 1 if len(close) > 1 else None
    return current, change_1m, change_3m, change_1y


@st.cache_data(ttl=900, show_spinner=False)
def collect_market_data(ticker_symbols):
    records = []
    histories = {}
    news_items = []
    errors = []

    for symbol in ticker_symbols:
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period="1y", auto_adjust=True)
            info = ticker.info or {}

            if history.empty:
                errors.append(f"{symbol}: no price history returned")
                continue

            close = history["Close"].dropna()
            current, change_1m, change_3m, change_1y = price_changes(close)

            record = {
                "ticker": symbol,
                "company": info.get("longName") or info.get("shortName") or symbol,
                "price": round(current, 2) if current is not None else None,
                "currency": info.get("currency"),
                "market_cap": clean_number(info.get("marketCap")),
                "trailing_pe": clean_number(info.get("trailingPE")),
                "forward_pe": clean_number(info.get("forwardPE")),
                "price_to_sales": clean_number(info.get("priceToSalesTrailing12Months")),
                "revenue_growth_pct": pct(info.get("revenueGrowth")),
                "earnings_growth_pct": pct(info.get("earningsGrowth")),
                "gross_margin_pct": pct(info.get("grossMargins")),
                "operating_margin_pct": pct(info.get("operatingMargins")),
                "free_cash_flow": clean_number(info.get("freeCashflow")),
                "debt_to_equity": clean_number(info.get("debtToEquity")),
                "analyst_target_mean": clean_number(info.get("targetMeanPrice")),
                "analyst_count": clean_number(info.get("numberOfAnalystOpinions")),
                "recommendation": info.get("recommendationKey"),
                "change_1m_pct": pct(change_1m),
                "change_3m_pct": pct(change_3m),
                "change_1y_pct": pct(change_1y),
                "data_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            records.append(record)
            histories[symbol] = close.rename(symbol)

            for item in (ticker.news or [])[:4]:
                content = item.get("content", item)
                canonical = content.get("canonicalUrl") or {}
                clickthrough = content.get("clickThroughUrl") or {}
                url = canonical.get("url") or clickthrough.get("url") or item.get("link")
                title = content.get("title") or item.get("title")
                provider = content.get("provider") or {}
                publisher = provider.get("displayName") or item.get("publisher")
                published = content.get("pubDate") or item.get("providerPublishTime")
                if title:
                    news_items.append(
                        {
                            "ticker": symbol,
                            "title": title,
                            "publisher": publisher,
                            "published": published,
                            "url": url,
                        }
                    )
        except Exception as exc:
            yahoo_error = str(exc)
            try:
                close = stooq_history(symbol)
                current, change_1m, change_3m, change_1y = price_changes(close)
                records.append(
                    {
                        "ticker": symbol,
                        "company": symbol,
                        "price": round(current, 2) if current is not None else None,
                        "currency": "USD",
                        "market_cap": None,
                        "trailing_pe": None,
                        "forward_pe": None,
                        "price_to_sales": None,
                        "revenue_growth_pct": None,
                        "earnings_growth_pct": None,
                        "gross_margin_pct": None,
                        "operating_margin_pct": None,
                        "free_cash_flow": None,
                        "debt_to_equity": None,
                        "analyst_target_mean": None,
                        "analyst_count": None,
                        "recommendation": None,
                        "change_1m_pct": pct(change_1m),
                        "change_3m_pct": pct(change_3m),
                        "change_1y_pct": pct(change_1y),
                        "data_source": "Stooq fallback",
                        "data_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
                )
                histories[symbol] = close.rename(symbol)
                errors.append(
                    f"{symbol}: Yahoo unavailable ({yahoo_error}); using Stooq price-history fallback. "
                    "Fundamentals and news are unavailable for this run."
                )
            except Exception as backup_exc:
                errors.append(
                    f"{symbol}: Yahoo unavailable ({yahoo_error}); "
                    f"backup also failed ({backup_exc})"
                )

    return records, histories, news_items, errors


def build_prompt(thesis, tickers, action, horizon, weights, records, news_items):
    evidence = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thesis": thesis,
        "tickers": tickers,
        "proposed_action": action,
        "time_horizon": horizon,
        "portfolio_weights": weights,
        "market_and_fundamental_data": records,
        "recent_news": news_items,
    }

    return f"""
You are an evidence-led investment committee analyst. Test the user's thesis;
do not merely agree with it. Use only the supplied evidence for company-specific
facts. If evidence is absent, explicitly label it as missing. Do not invent
figures, events, quotations, forecasts, probabilities, or sources.

This is decision support, not personalised financial advice. Separate observed
facts from interpretations. Treat analyst targets and recommendations as weak
secondary evidence. Note that the market data may be delayed or incomplete.

Write a concise but serious report in Markdown with these exact sections:

# Verdict
Give one of: SUPPORT ACTION, MODIFY ACTION, REJECT ACTION, or INSUFFICIENT EVIDENCE.
Then give a plain-English recommendation. Do not recommend an exact trade size
unless the user supplied portfolio weights. Give confidence as Low, Medium, or
High and explain why; do not invent a numerical probability.

# Thesis decomposed
List the testable claims hidden inside the thesis.

# Evidence dashboard
Use a Markdown table. For every important point label it Fact, Interpretation,
or Missing. Explain whether it supports or challenges the thesis.

# Bull case
Steelman the strongest case against trimming/exiting or in favour of adding/holding.

# Bear case
Steelman the strongest case in favour of trimming/exiting or against adding/holding.

# Risk/reward scenarios
Describe bull, base, and bear scenarios. State drivers and direction of impact.
Do not fabricate price targets.

# Portfolio risk
Discuss concentration, correlated exposures, and opportunity cost. If weights
are missing, say what cannot be quantified.

# What would prove this wrong?
Give specific observable invalidation conditions and upcoming evidence to monitor.

# Missing evidence
List the most important missing data that could change the verdict.

# Sources
List only the supplied working news URLs and identify market/fundamental metrics
as sourced from Yahoo Finance or the Stooq fallback, according to the supplied
record. Do not create URLs.

SUPPLIED EVIDENCE:
{json.dumps(evidence, indent=2, default=str)}
"""


def call_analyst(api_key, model, prompt):
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "developer",
                "content": "Be sceptical, balanced, source-disciplined, and explicit about uncertainty.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.output_text


st.title("⚖️ Thesis Tester")
st.caption("Challenge an investment thesis with market evidence, opposing cases, scenarios and invalidation tests.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Stored only for this running session unless you place it in a local .env file.",
    )
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    st.markdown(
        '<p class="small-note">API usage may incur charges. Never publish or share your API key.</p>',
        unsafe_allow_html=True,
    )

with st.form("thesis_form"):
    thesis = st.text_area(
        "Investment thesis",
        value="The Mythos software trade is exhausted and I should trim my positions.",
        height=120,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        tickers_text = st.text_input("Tickers", value="NOW, CRWD, DDOG")
    with col2:
        action = st.selectbox("Proposed action", ["Trim", "Hold", "Add", "Exit"])
    with col3:
        horizon = st.selectbox("Time horizon", ["3–6 months", "6–12 months", "1–3 years", "3+ years"], index=1)

    weights_text = st.text_input(
        "Optional current weights",
        placeholder="Example: NOW 12%, CRWD 6%, DDOG 4%",
    )
    submitted = st.form_submit_button("Test thesis", type="primary", use_container_width=True)

if submitted:
    tickers = list(dict.fromkeys(x.strip().upper() for x in tickers_text.split(",") if x.strip()))
    if not thesis.strip():
        st.error("Enter an investment thesis.")
        st.stop()
    if not tickers:
        st.error("Enter at least one ticker.")
        st.stop()
    if len(tickers) > 10:
        st.error("Use no more than 10 tickers in one test.")
        st.stop()
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar before running the test.")
        st.stop()

    with st.spinner("Collecting market evidence…"):
        records, histories, news_items, errors = collect_market_data(tickers)

    if errors:
        with st.expander("Data warnings"):
            for error in errors:
                st.warning(error)
    if not records:
        st.error("No usable market data was returned. Check the ticker symbols and try again.")
        st.stop()

    st.subheader("Market snapshot")
    display_rows = []
    for row in records:
        display_rows.append(
            {
                "Ticker": row["ticker"],
                "Price": row["price"],
                "Market cap": compact(row["market_cap"]),
                "Forward P/E": row["forward_pe"],
                "P/S": row["price_to_sales"],
                "Revenue growth %": row["revenue_growth_pct"],
                "Operating margin %": row["operating_margin_pct"],
                "1M %": row["change_1m_pct"],
                "3M %": row["change_3m_pct"],
                "1Y %": row["change_1y_pct"],
            }
        )
    st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)

    if histories:
        combined = pd.concat(histories.values(), axis=1).dropna(how="all")
        normalized = combined.div(combined.ffill().iloc[0]).mul(100)
        fig = go.Figure()
        for symbol in normalized.columns:
            fig.add_trace(go.Scatter(x=normalized.index, y=normalized[symbol], name=symbol))
        fig.update_layout(
            title="One-year relative performance (start = 100)",
            yaxis_title="Indexed price",
            xaxis_title="",
            legend_title="Ticker",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    prompt = build_prompt(
        thesis.strip(), tickers, action, horizon, weights_text.strip() or "Not supplied", records, news_items
    )
    with st.spinner("Challenging the thesis…"):
        try:
            report = call_analyst(api_key, model.strip(), prompt)
        except Exception as exc:
            st.error(f"The AI analysis could not run: {exc}")
            st.stop()

    st.subheader("Investment committee report")
    st.markdown(report)
    report_header = (
        f"# Thesis Tester Report\n\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"**Thesis:** {thesis.strip()}\n\n**Tickers:** {', '.join(tickers)}\n\n"
    )
    st.download_button(
        "Download report (.md)",
        data=report_header + report,
        file_name=f"thesis-test-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Research prototype only—not financial advice. Verify all figures against primary filings before making decisions. "
    "Market data may be delayed, incomplete or inaccurate. No trades are executed."
)
