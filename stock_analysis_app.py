#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票分析評估表 (美股：FMP 股價資料 + Gemini AI 分析／台股：yfinance + Gemini 白話解說)
=================================================

首頁可以選擇要分析「美股」還是「台股」，兩種市場各自獨立運作，共用同一組
Gemini API 金鑰（美股另外還需要 FMP／可選 Alpha Vantage 金鑰）：

【美股模式】以「單一美股個股」為分析對象（輸入一個股票代號即可分析）：
    1. 向 Financial Modeling Prep (FMP) 取得該股票的歷史股價資料
    2. 即時報價：另外呼叫 FMP 的即時報價 API，顯示目前價格、漲跌、今日區間、成交量與更新時間，
       讓你看到的「現在」價格跟技術指標所依據的「歷史收盤」資料能互相對照
    3. 計算技術指標：成交量、RSI、MACD、均線(MA5/20/60)、布林通道、KD（皆以歷史每日收盤計算）
    4. 型態分析：趨勢方向、支撐壓力、突破/跌破、K線型態辨識（十字星/鎚子線/吞噬型態等）
    5. 訊號回測（歷史勝率）：回測各技術訊號出現後「隔日上漲機率」，並與基準比較，
       讓你直接看到訊號的歷史可靠度，而不是只看一個「偏多/偏空」的字面結論
    6. 顯示「資料更新至」日期，避免誤把落後的歷史資料當成即時報價
    7. 視覺化：技術指標圖（均線/布林/KD/RSI/MACD/OBV）、訊號回測圖
    8. 規則式投資建議與風險評估（不需呼叫AI即可看到）
    9. 將整理後的數據交給 Google Gemini，產生具體、可執行的深度分析建議
    10. 新聞情緒分析：向 Alpha Vantage NEWS_SENTIMENT API 取得該股票的最新新聞與情緒評分，
        計算情緒分布、來源分布、主題分析，並交給 Gemini 產生市場情緒總結與深度分析
    11. CNN Business「恐懼與貪婪指數（Fear & Greed Index）」：顯示目前整體美股市場情緒
        （此功能串接的是非官方資料端點，並非 CNN 官方公開 API，可能隨時失效，詳見程式內說明）

安裝套件：
    pip install streamlit pandas numpy requests plotly

執行方式：
    streamlit run stock_analysis_app.py
    （若出現「'streamlit' 不是內部或外部命令」，改用：python -m streamlit run stock_analysis_app.py）

執行後瀏覽器會開啟一個網頁，畫面上有欄位可以自行輸入：
    - FMP API 金鑰 (https://site.financialmodelingprep.com/ 註冊取得)
    - Gemini API 金鑰 (https://aistudio.google.com/apikey 註冊取得)
    - Alpha Vantage API 金鑰（選用，新聞情緒分析用；https://www.alphavantage.co/support/#api-key 免費申請）
    - 欲分析的股票代號

API 金鑰只會保存在你本機瀏覽器開啟的這個 session 記憶體中，不會被寫入檔案或上傳。

關於「即時報價」的重要說明：RSI、MACD、KD、均線、布林通道這些技術指標的定義本身就是
「根據一段時間的歷史每日收盤價」計算出來的（例如RSI14要14天的收盤價），單一個當下的報價
無法算出這些指標，所以無法把整套技術分析改成「純即時」。本工具能做、也已經做的，是額外
抓取即時報價獨立顯示在分析最上方，讓你隨時能對照「現在的價格」跟「技術指標所依據的歷史
資料」兩者是否一致，而不是讓你誤以為技術指標本身是即時更新的。

重要提醒：RSI/MACD/KD/均線等都是「落後指標」，反映的是過去趨勢，不是對下一個交易日
漲跌的預言；本工具的「訊號回測」就是要讓你看到這些訊號過去實際的命中率（通常落在五、
六成左右），藉此管理預期，而不是把任何單一訊號當成保證獲利的買賣依據。所有內容（含 AI
分析）僅供參考，不構成投資建議。

【台股模式】給完全新手用，設計理念不同於美股模式：不需要使用者自己申請/輸入任何
API 金鑰（Gemini 金鑰由部署者透過 st.secrets 事先設定好）、不顯示 RSI/MACD/KD 等
進階技術指標（避免畫面太複雜嚇跑新手），改用 K線＋均線＋成交量＋支撐壓力＋K線型態
辨識＋與大盤（加權指數）的相對強弱比較，並把結果濃縮成一句「偏多／中性／偏空」的
綜合燈號，再交給 Gemini 用生活化白話文解釋給完全不懂技術分析的人聽。股價資料來源
是 Yahoo Finance（透過 yfinance 套件，不需要申請帳號或金鑰），台股代號會自動嘗試
補上 .TW（上市）／.TWO（上櫃）。
"""

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# 常數設定
# ----------------------------------------------------------------------------

FMP_STABLE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
FMP_V3_URL_TMPL = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
FMP_QUOTE_STABLE_URL = "https://financialmodelingprep.com/stable/quote"
FMP_QUOTE_V3_URL_TMPL = "https://financialmodelingprep.com/api/v3/quote/{symbol}"
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# CNN Business「恐懼與貪婪指數」沒有官方公開 API；這是外界長期觀察到、CNN 網站前端
# 自己在用的一個「未正式紀錄」資料端點，並非官方合約保證的介面，隨時可能改版或封鎖。
FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0  # Sharpe Ratio 固定以 0% 無風險利率計算（即年化報酬／年化波動度）
# 原本固定抓365天（約52週），改成420天（約60週），多留一點緩衝，確保下面
# 「整體核心持倉防禦停損」需要用到的「週線50週均線」在資料剛好卡在邊界時
# 也能算出有效數值（50週均線至少需要50根週K棒才有值）。
HISTORY_LOOKBACK_DAYS = 420

STOCK_COLOR = "#2563eb"

# --- 使用者個人持倉停損／停利機制（取代原本「距目前收盤價的固定百分比」算法）---
# 依使用者提供的《NVDA 持倉管理與交易規則範本》第三節「出場、停損與追蹤停利機制」，
# 這是使用者自己的量化紀律，並非依風險等級（保守/穩健/積極）浮動調整的通用估算值，
# 所以底下改成單一套固定規則，取代原本按風險等級給不同百分比的做法。
NEW_POSITION_STOP_LOSS_PCT = 0.07      # 新加碼部位：買入成本價（此處以目前收盤價為進場參考價）跌破 -7% 無條件停損
WEEKLY_DEFENSE_MA_WINDOW = 50          # 整體核心持倉防禦停損：週線的 50 週均線
WEEKLY_DEFENSE_CONFIRM_DAYS = 3        # 跌破後，最近 N 個交易日收盤都未收復才視為「確認」
STAGE1_TAKE_PROFIT_SELL_PCT = 0.20     # 第一階段（鎖定獲利）：達目標價時最多賣出的比例，其餘續抱
TRAILING_STOP_MA_WINDOW = 20           # 第二階段（追蹤停利）：只要收盤守住這條均線之上就全程持有
TRAILING_STOP_SELL_PCT = 0.30          # 第二階段：收盤實體跌破上述均線時，調節的持股比例

# --- 使用者「持倉風控與紀律」機制（範本第一、二、四節）---
# 這幾項規則需要知道使用者的「實際持倉狀態」（成本、股數、帳戶總資產、目標倉位），
# 單靠這支工具原本抓的股價資料無法算出來，所以額外設計成「選填」區塊：使用者上傳
# 自己的交易紀錄 CSV，並手動輸入帳戶總資產／現金水位／目標總倉位這幾個工具無法
# 自動得知的數字，才會顯示這個區塊的檢核結果；不提供的話，完全不影響上方原本的
# 個股技術面分析。
SINGLE_STOCK_MAX_EXPOSURE_PCT = 0.35      # 單一股票最大曝險：帳戶總資產上限 35%
SINGLE_ADDON_MAX_PCT_OF_TARGET = 0.25     # 單次加碼上限：不得超過目標總倉位 25%
CASH_RESERVE_MIN_PCT = 0.15               # 預留現金防禦率：常態維持 15%~20%
CASH_RESERVE_MAX_PCT = 0.20
PYRAMID_A_RSI_MAX = 45                    # A. 逢低回測買點：日線 RSI <= 45
PYRAMID_A_MA_TOLERANCE_PCT = 0.03         # A. 「回測20MA或50MA止穩」的貼近容忍度（正負3%）
PYRAMID_B_VOLUME_MULTIPLE = 1.5           # B. 突破確認買點：成交量須高於20日均量的1.5倍
PYRAMID_B_LOOKBACK_WIN = 20               # B. 「前高壓力帶」用近幾日高點判斷
ANTI_FOMO_WINDOW_DAYS = 3                 # 反FOMO回補檢查：賣出後幾個自然日內的買回都列入檢查

# 新聞情緒五級分類配色（Bearish=深紅 ～ Bullish=深綠，符合金融情緒分析慣例配色）
SENTIMENT_LABELS_ORDER = ["Bearish", "Somewhat-Bearish", "Neutral", "Somewhat-Bullish", "Bullish"]
SENTIMENT_COLORS = {
    "Bearish": "#DC143C",
    "Somewhat-Bearish": "#FF6B6B",
    "Neutral": "#95A5A6",
    "Somewhat-Bullish": "#90EE90",
    "Bullish": "#2ECC71",
}

# 情緒標籤對應表情符號：Bullish=笑臉／Bearish=哭臉／Neutral=不哭不笑；
# Somewhat-Bullish/Bearish 用較淺的表情銜接兩端，方便一眼辨識情緒強弱。
SENTIMENT_EMOJI = {
    "Bearish": "😢",
    "Somewhat-Bearish": "🙁",
    "Neutral": "😐",
    "Somewhat-Bullish": "🙂",
    "Bullish": "😊",
}


def _sentiment_label_emoji(label: str) -> str:
    """在情緒標籤前加上對應表情符號；非五級標準標籤（例如 N/A）則原樣傳回，不加表情。"""
    emoji = SENTIMENT_EMOJI.get(label)
    return f"{emoji} {label}" if emoji else label


FEAR_GREED_RATING_ZH = {
    "extreme fear": "極度恐懼",
    "fear": "恐懼",
    "neutral": "中性",
    "greed": "貪婪",
    "extreme greed": "極度貪婪",
}


# ----------------------------------------------------------------------------
# 1. 資料抓取 (FMP)
# ----------------------------------------------------------------------------

def _normalize_fmp_df(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["close"])
    df = df.sort_values("date").reset_index(drop=True)
    df["symbol"] = symbol
    return df


def fetch_fmp_history(symbol: str, start_date: date, end_date: date, api_key: str) -> pd.DataFrame:
    """向 FMP 取得指定股票在區間內的歷史日線資料（含 open/high/low/close/volume）。

    先嘗試新版 `stable` 端點，失敗則回退到舊版 `v3` 端點，盡量提高相容性。
    """
    symbol = symbol.strip().upper()
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    try:
        resp = requests.get(
            FMP_STABLE_URL,
            params={"symbol": symbol, "from": start_str, "to": end_str, "apikey": api_key},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0 and "close" in data[0]:
                return _normalize_fmp_df(pd.DataFrame(data), symbol)
            if isinstance(data, dict) and "Error Message" in data:
                raise ValueError(f"FMP API 錯誤（{symbol}）：{data['Error Message']}")
    except requests.RequestException:
        pass  # 交給下面的 v3 端點重試

    resp = requests.get(
        FMP_V3_URL_TMPL.format(symbol=symbol),
        params={"from": start_str, "to": end_str, "apikey": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "Error Message" in data:
        raise ValueError(f"FMP API 錯誤（{symbol}）：{data['Error Message']}")
    if isinstance(data, dict) and "historical" in data:
        hist = data["historical"]
        if not hist:
            raise ValueError(f"FMP 未回傳 {symbol} 在 {start_str} ~ {end_str} 的資料，請確認股票代號與日期區間。")
        return _normalize_fmp_df(pd.DataFrame(hist), symbol)

    raise ValueError(f"無法解析 FMP 回傳資料（{symbol}）：{str(data)[:300]}")


def fetch_fmp_quote(symbol: str, api_key: str) -> dict:
    """向 FMP 取得即時報價（目前價格、漲跌、今日區間、成交量、更新時間）。

    即時報價與歷史技術指標是兩個獨立的資料來源：技術指標（RSI/MACD/KD/均線等）
    本質上需要一段期間的「歷史每日收盤價」才能計算，無法用單一即時報價取代；
    這裡抓到的即時報價只用於獨立顯示「現在」的價格，供你對照參考。

    是否能取得即時報價、是否有延遲，依你的 FMP 方案而定；失敗時會拋出例外，
    呼叫端應自行 try/except，避免因為即時報價抓取失敗而讓整個分析中斷。
    """
    symbol = symbol.strip().upper()
    data = None

    try:
        resp = requests.get(
            FMP_QUOTE_STABLE_URL, params={"symbol": symbol, "apikey": api_key}, timeout=15,
        )
        if resp.status_code == 200:
            j = resp.json()
            if isinstance(j, list) and j:
                data = j[0]
    except requests.RequestException:
        pass

    if data is None:
        resp = requests.get(
            FMP_QUOTE_V3_URL_TMPL.format(symbol=symbol), params={"apikey": api_key}, timeout=15,
        )
        resp.raise_for_status()
        j = resp.json()
        if isinstance(j, dict) and "Error Message" in j:
            raise ValueError(f"FMP 即時報價錯誤（{symbol}）：{j['Error Message']}")
        if isinstance(j, list) and j:
            data = j[0]

    if not data:
        raise ValueError(f"無法取得 {symbol} 的即時報價（可能是股票代號有誤，或目前 API 方案不含此功能）。")

    ts = data.get("timestamp")
    updated_at = None
    if ts:
        try:
            updated_at = datetime.fromtimestamp(ts)
        except (OSError, OverflowError, ValueError):
            updated_at = None

    return {
        "price": data.get("price"),
        "change": data.get("change"),
        "change_pct": data.get("changesPercentage", data.get("changePercentage")),
        "day_low": data.get("dayLow"),
        "day_high": data.get("dayHigh"),
        "open": data.get("open"),
        "previous_close": data.get("previousClose"),
        "volume": data.get("volume"),
        "avg_volume": data.get("avgVolume"),
        "updated_at": updated_at,
    }


# ----------------------------------------------------------------------------
# 1b. 資料抓取：Alpha Vantage 新聞情緒 + CNN Business Fear & Greed Index
# ----------------------------------------------------------------------------

def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fetch_alpha_vantage_news(symbol: str, api_key: str, limit: int = 50) -> list:
    """向 Alpha Vantage NEWS_SENTIMENT API 取得指定股票的最新新聞與情緒資料。

    回傳整理後的新聞清單（list of dict），每筆包含標題、連結、發布時間、來源、摘要、
    整篇文章情緒分數／標籤，以及該篇文章「針對此股票」的相關性分數與情緒分數／標籤
    （從 API 回傳的 ticker_sentiment 陣列中，篩出符合此股票代號的那一筆）。

    Alpha Vantage 免費版帳號每日限 25 次 API 請求，超過額度或金鑰錯誤時，API 會用
    HTTP 200 回傳一段包含 "Note" 或 "Information" 的錯誤說明文字，因此這裡會額外
    檢查這兩個欄位並轉成明確的例外，而不是誤判成「沒有新聞」。
    """
    symbol = symbol.strip().upper()
    resp = requests.get(
        ALPHA_VANTAGE_URL,
        params={"function": "NEWS_SENTIMENT", "tickers": symbol, "limit": limit, "apikey": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage API 錯誤（{symbol}）：{data['Error Message']}")
    if "Note" in data:
        raise ValueError(f"Alpha Vantage API 額度限制：{data['Note']}（免費版每日限 25 次請求，請稍後再試）")
    if "Information" in data:
        raise ValueError(f"Alpha Vantage API 訊息：{data['Information']}（請確認金鑰是否正確）")

    feed = data.get("feed")
    if not feed:
        raise ValueError(f"Alpha Vantage 未回傳 {symbol} 的新聞資料，請確認股票代號是否正確。")

    news_list = []
    for item in feed:
        ticker_info = None
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == symbol:
                ticker_info = ts
                break

        try:
            published_at = datetime.strptime(item.get("time_published", ""), "%Y%m%dT%H%M%S")
        except ValueError:
            published_at = None

        news_list.append({
            "title": item.get("title") or "（無標題）",
            "url": item.get("url", ""),
            "published_at": published_at,
            "source": item.get("source") or "未知來源",
            "summary": item.get("summary", ""),
            "overall_sentiment_score": _safe_float(item.get("overall_sentiment_score")),
            "overall_sentiment_label": item.get("overall_sentiment_label", "N/A"),
            "topics": item.get("topics", []),
            "ticker_relevance": _safe_float(ticker_info.get("relevance_score")) if ticker_info else None,
            "ticker_sentiment_score": _safe_float(ticker_info.get("ticker_sentiment_score")) if ticker_info else None,
            "ticker_sentiment_label": ticker_info.get("ticker_sentiment_label", "N/A") if ticker_info else "N/A",
        })

    news_list.sort(key=lambda n: n["published_at"] or datetime.min, reverse=True)
    return news_list


def compute_news_sentiment_stats(news_list: list) -> dict:
    """將原始新聞清單整理成情緒分布、來源分布、主題分析等統計資料。"""
    ticker_labels = [x["ticker_sentiment_label"] for x in news_list if x["ticker_sentiment_label"] in SENTIMENT_LABELS_ORDER]
    article_labels = [x["overall_sentiment_label"] for x in news_list if x["overall_sentiment_label"] in SENTIMENT_LABELS_ORDER]

    def dist(labels):
        total = len(labels) or 1
        return {lab: {"count": labels.count(lab), "pct": labels.count(lab) / total} for lab in SENTIMENT_LABELS_ORDER}

    ticker_scores = [x["ticker_sentiment_score"] for x in news_list if x["ticker_sentiment_score"] is not None]
    article_scores = [x["overall_sentiment_score"] for x in news_list if x["overall_sentiment_score"] is not None]
    relevance_scores = [x["ticker_relevance"] for x in news_list if x["ticker_relevance"] is not None]

    source_counts = {}
    for x in news_list:
        source_counts[x["source"]] = source_counts.get(x["source"], 0) + 1
    source_counts = dict(sorted(source_counts.items(), key=lambda kv: kv[1], reverse=True))

    topic_scores = {}
    for x in news_list:
        for t in x.get("topics", []):
            name = t.get("topic") or "未知主題"
            w = _safe_float(t.get("relevance_score")) or 0.0
            topic_scores[name] = topic_scores.get(name, 0.0) + w
    topic_scores = dict(sorted(topic_scores.items(), key=lambda kv: kv[1], reverse=True)[:8])

    ticker_dist = dist(ticker_labels)
    dominant_label = max(ticker_dist.items(), key=lambda kv: kv[1]["count"])[0] if ticker_labels else "N/A"

    return {
        "n_news": len(news_list),
        "ticker_sentiment_dist": ticker_dist,
        "article_sentiment_dist": dist(article_labels),
        "avg_ticker_score": float(np.mean(ticker_scores)) if ticker_scores else None,
        "avg_article_score": float(np.mean(article_scores)) if article_scores else None,
        "avg_relevance": float(np.mean(relevance_scores)) if relevance_scores else None,
        "source_counts": source_counts,
        "topic_scores": topic_scores,
        "dominant_ticker_label": dominant_label,
    }


def fetch_fear_greed_index() -> dict:
    """嘗試取得 CNN Business 的 Fear & Greed Index（恐懼與貪婪指數，反映整體美股市場情緒）。

    重要說明：CNN 並未提供這項指數的官方公開 API。這裡串接的是外界長期觀察到、CNN
    網站前端自己在使用的一個「未正式紀錄」資料端點，並非官方合約保證的介面：CNN
    隨時可能改版、調整回應格式，或封鎖非瀏覽器來源的請求。一旦發生，這個功能會直接
    拋出例外並在畫面上顯示清楚的錯誤訊息，但不會影響股價、技術指標等其他分析功能。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    resp = requests.get(FEAR_GREED_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    fg = data.get("fear_and_greed")
    if not fg or fg.get("score") is None:
        raise ValueError("CNN Fear & Greed Index 回傳格式不如預期（非官方端點可能已變更），暫時無法解析。")

    rating_raw = (fg.get("rating") or "").lower()
    return {
        "score": _safe_float(fg.get("score")),
        "rating_raw": fg.get("rating", "N/A"),
        "rating_zh": FEAR_GREED_RATING_ZH.get(rating_raw, fg.get("rating", "N/A")),
        "previous_close": _safe_float(fg.get("previous_close")),
        "previous_1_week": _safe_float(fg.get("previous_1_week")),
        "previous_1_month": _safe_float(fg.get("previous_1_month")),
        "previous_1_year": _safe_float(fg.get("previous_1_year")),
        "timestamp": fg.get("timestamp"),
    }


# ----------------------------------------------------------------------------
# 2. 技術指標：RSI / MACD / 均線 / 布林通道 / KD / 成交量
# ----------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)  # 完全沒有下跌時 RSI = 100


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = close.rolling(window, min_periods=1).mean()
    std = close.rolling(window, min_periods=1).std().fillna(0)
    return mid + num_std * std, mid, mid - num_std * std


def compute_kd(df: pd.DataFrame, n: int = 9) -> tuple:
    """傳統式 KD 隨機指標（RSV 依 9 日高低，K/D 以 2/3、1/3 遞迴平滑，初始值 50）。"""
    low_n = df["low"].rolling(n, min_periods=1).min()
    high_n = df["high"].rolling(n, min_periods=1).max()
    rsv = ((df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100).fillna(50)

    k_values, d_values = [], []
    k_prev, d_prev = 50.0, 50.0
    for rsv_t in rsv:
        k_t = (2 / 3) * k_prev + (1 / 3) * rsv_t
        d_t = (2 / 3) * d_prev + (1 / 3) * k_t
        k_values.append(k_t)
        d_values.append(d_t)
        k_prev, d_prev = k_t, d_t
    return pd.Series(k_values, index=df.index), pd.Series(d_values, index=df.index)


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """OBV（On-Balance Volume，能量潮）：收盤上漲當天加計成交量、下跌當天減計成交量，
    用累積起來的量能方向搭配價格走勢，判斷買賣力道是否與價格趨勢一致（量價同步／背離）。"""
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).cumsum()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["daily_return"] = df["close"].pct_change()
    df["rsi14"] = compute_rsi(df["close"], 14)
    macd, signal, hist = compute_macd(df["close"])
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist
    df["volume_ma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["ma5"] = df["close"].rolling(5, min_periods=1).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    df["ma50"] = df["close"].rolling(50, min_periods=1).mean()
    df["ma60"] = df["close"].rolling(60, min_periods=1).mean()
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger_bands(df["close"], 20, 2.0)
    df["kd_k"], df["kd_d"] = compute_kd(df, 9)
    df["obv"] = compute_obv(df)
    df["obv_ma20"] = df["obv"].rolling(20, min_periods=1).mean()
    return df


def compute_stats(df: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> dict:
    daily_ret = df["daily_return"].dropna()
    total_return = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    ann_return = daily_ret.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol not in (0, np.nan) else np.nan

    cum = (1 + daily_ret).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min() if len(drawdown) else np.nan

    latest_rsi = df["rsi14"].iloc[-1]
    latest_macd = df["macd"].iloc[-1]
    latest_signal = df["macd_signal"].iloc[-1]
    # MACD 本質上是短期動能指標（12/26日EMA差值），跟均線排列（尤其MA60）這種偏長期的
    # 結構性指標本來就可能不同步，明確加上「短期」二字，避免讓人誤以為兩者矛盾。
    macd_cross = "黃金交叉（短期偏多訊號）" if latest_macd > latest_signal else "死亡交叉（短期偏空訊號）"

    latest_volume = df["volume"].iloc[-1]
    avg_volume20 = df["volume_ma20"].iloc[-1]
    if avg_volume20 and latest_volume > avg_volume20 * 1.2:
        vol_signal = "放量（高於20日均量20%以上）"
    elif avg_volume20 and latest_volume < avg_volume20 * 0.8:
        vol_signal = "縮量（低於20日均量20%以上）"
    else:
        vol_signal = "量能持平"

    if latest_rsi >= 70:
        rsi_signal = "超買區間"
    elif latest_rsi <= 30:
        rsi_signal = "超賣區間"
    else:
        rsi_signal = "中性區間"

    ma5, ma20, ma60 = df["ma5"].iloc[-1], df["ma20"].iloc[-1], df["ma60"].iloc[-1]
    ma_cross = "均線黃金交叉（MA5 > MA20，偏多）" if ma5 > ma20 else "均線死亡交叉（MA5 < MA20，偏空）"
    # MA60 涵蓋60個交易日，屬於長期結構指標，加上「長期」二字跟上面MACD的
    # 「短期」做出時間尺度上的區隔——兩者本來就可能同時成立、不代表矛盾。
    if ma5 > ma20 > ma60:
        ma_alignment = "長期多頭排列（MA5 > MA20 > MA60）"
    elif ma5 < ma20 < ma60:
        ma_alignment = "長期空頭排列（MA5 < MA20 < MA60）"
    else:
        ma_alignment = "均線糾結／排列不明確"

    latest_close = df["close"].iloc[-1]
    bb_upper, bb_mid, bb_lower = df["bb_upper"].iloc[-1], df["bb_mid"].iloc[-1], df["bb_lower"].iloc[-1]
    if latest_close > bb_upper:
        bb_position = "價格突破布林上軌（可能過熱或強勢噴出）"
    elif latest_close < bb_lower:
        bb_position = "價格跌破布林下軌（可能超跌或弱勢破底）"
    elif latest_close > bb_mid:
        bb_position = "價格位於中軌與上軌之間（偏多）"
    else:
        bb_position = "價格位於中軌與下軌之間（偏空）"

    kd_k, kd_d = df["kd_k"].iloc[-1], df["kd_d"].iloc[-1]
    kd_cross = "KD 黃金交叉（K > D，偏多）" if kd_k > kd_d else "KD 死亡交叉（K < D，偏空）"
    if kd_k >= 80:
        kd_signal = "超買區間"
    elif kd_k <= 20:
        kd_signal = "超賣區間"
    else:
        kd_signal = "中性區間"

    # --- OBV（能量潮）：判斷量能方向是否與價格趨勢一致 ---
    latest_obv = df["obv"].iloc[-1]
    lookback = min(20, len(df) - 1)
    obv_trend = "上升" if latest_obv > df["obv"].iloc[-1 - lookback] else "下降"
    price_trend_20 = "上升" if latest_close > df["close"].iloc[-1 - lookback] else "下降"
    if obv_trend == price_trend_20:
        obv_signal = f"量價同步（近{lookback}日OBV與價格同為{obv_trend}，趨勢有成交量佐證，動能較可信）"
    else:
        obv_signal = f"量價背離（近{lookback}日OBV{obv_trend}但價格{price_trend_20}，趨勢動能可能不足，需留意）"

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "latest_date": df["date"].iloc[-1],
        "latest_close": latest_close,
        "latest_rsi": latest_rsi,
        "rsi_signal": rsi_signal,
        "latest_macd": latest_macd,
        "latest_macd_signal": latest_signal,
        "macd_cross": macd_cross,
        "latest_volume": latest_volume,
        "avg_volume20": avg_volume20,
        "vol_signal": vol_signal,
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "ma_cross": ma_cross,
        "ma_alignment": ma_alignment,
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "bb_position": bb_position,
        "latest_kd_k": kd_k,
        "latest_kd_d": kd_d,
        "kd_cross": kd_cross,
        "kd_signal": kd_signal,
        "latest_obv": latest_obv,
        "obv_signal": obv_signal,
    }


# ----------------------------------------------------------------------------
# 2b. 訊號回測（歷史勝率）
# ----------------------------------------------------------------------------

def backtest_signals(df: pd.DataFrame) -> dict:
    """回測整段歷史資料中，幾種常見技術訊號出現當天，「隔日」股價上漲的機率，
    並與基準（所有交易日隔日上漲的機率）比較，藉此量化訊號的歷史可靠度。

    這是描述「過去」的統計結果，不保證未來會維持相同機率，僅供參考。
    """
    work = df.copy()
    work["next_return"] = work["daily_return"].shift(-1)
    valid = work.dropna(subset=["next_return"])

    def hit_rate(mask):
        subset = valid[mask]
        n = len(subset)
        if n == 0:
            return None, 0
        return float((subset["next_return"] > 0).mean()), n

    baseline_wr, baseline_n = hit_rate(pd.Series(True, index=valid.index))

    signal_defs = {
        "MACD黃金交叉": valid["macd"] > valid["macd_signal"],
        "RSI超賣(<=30)": valid["rsi14"] <= 30,
        "RSI超買(>=70)": valid["rsi14"] >= 70,
        "KD黃金交叉": valid["kd_k"] > valid["kd_d"],
        "均線多頭(MA5>MA20)": valid["ma5"] > valid["ma20"],
    }

    signal_results = {}
    for name, mask in signal_defs.items():
        wr, n = hit_rate(mask)
        signal_results[name] = {"win_rate": wr, "n": n}

    return {"baseline": {"win_rate": baseline_wr, "n": baseline_n}, "signals": signal_results}


# ----------------------------------------------------------------------------
# 2c. 投資建議與風險評估（規則式，直接由技術指標計算，不需呼叫 AI）
# ----------------------------------------------------------------------------

def assess_risk_level(stats: dict):
    """依年化波動度與最大回撤，將個股分成 低/中/高 風險三級。

    門檻為一般常見的經驗法則，非精確科學分類，僅供快速參考。
    """
    ann_vol = stats["ann_vol"]
    max_dd = stats["max_drawdown"]

    vol_score = 0 if ann_vol < 0.25 else (1 if ann_vol < 0.45 else 2)
    dd_score = 0 if max_dd > -0.15 else (1 if max_dd > -0.30 else 2)
    total = vol_score + dd_score

    if total <= 1:
        return "低風險", "low"
    elif total <= 2:
        return "中風險", "mid"
    else:
        return "高風險", "high"


def check_weekly_ma50_defense_stop(df: pd.DataFrame) -> dict:
    """判斷「整體核心持倉防禦停損」是否觸發：週線收盤跌破 50 週均線，且最近
    WEEKLY_DEFENSE_CONFIRM_DAYS 個交易日的收盤都未收復回這條均線之上。

    這裡的「3日未收復」解讀為：50週均線的位置變動很慢，所以拿「最新一根週K棒
    算出來的50週均線水位」當基準線，檢查最近3個交易日的『日收盤價』是否都還在
    這條線之下──只要有任何一天收盤收回到線之上，就視為尚未confirmed（仍在
    觀察，不代表規則失效），這是對文字規則的具體化解讀，並非唯一可能的解讀方式。

    50週均線至少需要50根有效的週K棒才算得出來；資料不足時回傳
    status='insufficient_data'，呼叫端應顯示提醒文字，不中斷其他分析。
    """
    weekly_close = df.set_index("date")["close"].resample("W-FRI").last().dropna()
    if len(weekly_close) < WEEKLY_DEFENSE_MA_WINDOW:
        return {"status": "insufficient_data", "weeks_available": int(len(weekly_close))}

    weekly_ma = weekly_close.rolling(WEEKLY_DEFENSE_MA_WINDOW, min_periods=WEEKLY_DEFENSE_MA_WINDOW).mean()
    latest_weekly_ma = weekly_ma.iloc[-1]
    if pd.isna(latest_weekly_ma):
        return {"status": "insufficient_data", "weeks_available": int(len(weekly_close))}

    latest_weekly_close = weekly_close.iloc[-1]
    weekly_broken = latest_weekly_close < latest_weekly_ma

    last_n_daily_close = df["close"].tail(WEEKLY_DEFENSE_CONFIRM_DAYS)
    has_enough_days = len(last_n_daily_close) >= WEEKLY_DEFENSE_CONFIRM_DAYS
    reclaimed = bool((last_n_daily_close > latest_weekly_ma).any())
    confirmed = bool(weekly_broken and has_enough_days and not reclaimed)

    return {
        "status": "ok",
        "weekly_ma": float(latest_weekly_ma),
        "latest_weekly_close": float(latest_weekly_close),
        "weekly_broken": bool(weekly_broken),
        "confirmed": confirmed,
        "weeks_available": int(len(weekly_close)),
    }


def compute_playbook_stop_plan(df: pd.DataFrame, trend: dict, custom_target_price: float = None) -> dict:
    """依使用者自己的持倉停損／停利機制（取代原本「距目前收盤價的固定百分比」算法），
    直接換算出實際股價與目前是否已觸發，而不是只給一組通用的風險等級百分比：

    停損（兩條防線，分別對應不同身份，兩者都會顯示）：
      1. 新加碼部位：進場參考價（此處以目前收盤價代入）跌破 NEW_POSITION_STOP_LOSS_PCT
         （-7%）無條件停損。
      2. 整體核心持倉：週線跌破50週均線，且最近3個交易日收盤都未收復
         （見 check_weekly_ma50_defense_stop）。「若基本面未變」這個前提屬於質化判斷，
         本工具無法量化，僅在文字說明中提醒使用者自行確認。

    停利（分兩階段，非固定金額）：
      第一階段：股價達到「階段目標價」（可由使用者自訂；留空則以近期壓力價／前高為準）
                時，最多賣出 STAGE1_TAKE_PROFIT_SELL_PCT（20%），其餘續抱。
      第二階段（追蹤停利）：剩餘部位只要收盤守住 TRAILING_STOP_MA_WINDOW 日均線
                （20MA）之上就全程持有、不設停利上限；收盤跌破才調節
                TRAILING_STOP_SELL_PCT（30%）。
    """
    latest_close = float(df["close"].iloc[-1])
    ma_col = f"ma{TRAILING_STOP_MA_WINDOW}"
    trailing_ma = float(df[ma_col].iloc[-1]) if ma_col in df.columns else None

    new_position_stop_loss_price = latest_close * (1 - NEW_POSITION_STOP_LOSS_PCT)

    weekly_defense = check_weekly_ma50_defense_stop(df)

    resistance = (trend or {}).get("resistance")
    if custom_target_price is not None and custom_target_price > 0:
        stage1_target = float(custom_target_price)
        stage1_target_source = "使用者自訂目標價"
    elif resistance is not None and not pd.isna(resistance):
        stage1_target = float(resistance)
        stage1_target_source = "近期壓力價（近期高點，自動估算，非使用者自訂）"
    else:
        stage1_target = None
        stage1_target_source = None
    stage1_reached = bool(stage1_target is not None and latest_close >= stage1_target)

    trailing_broken = bool(trailing_ma is not None and not pd.isna(trailing_ma) and latest_close < trailing_ma)

    return {
        "entry_ref_price": latest_close,
        "new_position_stop_loss_pct": NEW_POSITION_STOP_LOSS_PCT,
        "new_position_stop_loss_price": new_position_stop_loss_price,
        "weekly_defense": weekly_defense,
        "stage1_target": stage1_target,
        "stage1_target_source": stage1_target_source,
        "stage1_sell_pct": STAGE1_TAKE_PROFIT_SELL_PCT,
        "stage1_reached": stage1_reached,
        "trailing_ma": trailing_ma,
        "trailing_ma_window": TRAILING_STOP_MA_WINDOW,
        "trailing_broken": trailing_broken,
        "trailing_sell_pct": TRAILING_STOP_SELL_PCT,
    }


def _fmt_stop_plan_lines(stop_plan: dict) -> str:
    """把 compute_playbook_stop_plan() 的結果，整理成給 Gemini 提示語與
    Markdown 報告共用的條列文字，確保 AI 看到的數字跟畫面上顯示的完全一致。"""
    sp = stop_plan
    wd = sp["weekly_defense"]

    if wd["status"] == "insufficient_data":
        weekly_line = f"歷史資料僅約 {wd['weeks_available']} 週，尚不足50週，暫無法計算"
    elif wd["confirmed"]:
        weekly_line = (f"已觸發（週收盤 {wd['latest_weekly_close']:.2f} 跌破50週均線 "
                        f"{wd['weekly_ma']:.2f}，且最近{WEEKLY_DEFENSE_CONFIRM_DAYS}個交易日收盤都未收復）")
    elif wd["weekly_broken"]:
        weekly_line = (f"觀察中（週收盤 {wd['latest_weekly_close']:.2f} 已跌破50週均線 "
                        f"{wd['weekly_ma']:.2f}，但尚未滿足3日未收復的確認條件）")
    else:
        weekly_line = f"未觸發（週收盤 {wd['latest_weekly_close']:.2f} 仍在50週均線 {wd['weekly_ma']:.2f} 之上）"

    if sp["stage1_target"] is not None:
        stage1_line = (f"{sp['stage1_target']:.2f}（來源：{sp['stage1_target_source']}；"
                        f"{'已達標' if sp['stage1_reached'] else '尚未達標'}，"
                        f"達標時最多賣出{sp['stage1_sell_pct']:.0%}，其餘續抱）")
    else:
        stage1_line = "尚無法自動估算（近期高低點資料不足，使用者也未自訂目標價）"

    if sp["trailing_ma"] is not None:
        trailing_line = (f"{TRAILING_STOP_MA_WINDOW}日均線 {sp['trailing_ma']:.2f}，"
                          f"目前{'已跌破，達調節條件' if sp['trailing_broken'] else '收盤仍守住均線之上，維持持有'}，"
                          f"跌破時調節{sp['trailing_sell_pct']:.0%}，未跌破則不設停利上限")
    else:
        trailing_line = "均線資料不足，暫無法計算"

    return (
        f"- 停損①新加碼部位（進場參考價 {sp['entry_ref_price']:.2f} 跌破 -{sp['new_position_stop_loss_pct']:.0%}）："
        f"{sp['new_position_stop_loss_price']:.2f}\n"
        f"- 停損②整體核心持倉防禦（週線50週均線+3日未收復，前提為基本面未變）：{weekly_line}\n"
        f"- 停利①第一階段目標價：{stage1_line}\n"
        f"- 停利②第二階段追蹤停利：{trailing_line}"
    )


# ----------------------------------------------------------------------------
# 2.5 持倉風控與紀律檢核（範本第一、二、四節：曝險/現金上限、分批建倉型態、反FOMO回補）
# ----------------------------------------------------------------------------

def parse_trade_log(raw_df: pd.DataFrame):
    """把使用者上傳的交易紀錄 CSV（欄位：Date,Type,Symbol,Name,Price,Quantity,Reason）
    整理成標準化格式。看不懂的列（日期/價格/股數/買賣別格式錯誤）會被跳過並回傳警告文字，
    而不是讓整支工具直接報錯中斷。回傳 (整理後的DataFrame, 警告文字list)。
    """
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "date":
            col_map[c] = "date"
        elif lc == "type":
            col_map[c] = "type"
        elif lc == "symbol":
            col_map[c] = "symbol"
        elif lc == "price":
            col_map[c] = "price"
        elif lc in ("quantity", "qty", "shares"):
            col_map[c] = "quantity"
        elif lc == "reason":
            col_map[c] = "reason"
    df = df.rename(columns=col_map)

    required = {"date", "type", "price", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"交易紀錄缺少必要欄位：{'、'.join(sorted(missing))}（需要 Date, Type, Price, Quantity）")

    warnings = []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["type_norm"] = df["type"].astype(str).str.strip().str.lower()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    if "reason" not in df.columns:
        df["reason"] = ""
    df["reason"] = df["reason"].fillna("")
    if "symbol" not in df.columns:
        df["symbol"] = ""

    valid_mask = (
        df["date"].notna() & df["price"].notna() & df["quantity"].notna()
        & df["type_norm"].isin(["buy", "sell"])
    )
    n_bad = int((~valid_mask).sum())
    if n_bad > 0:
        warnings.append(f"有 {n_bad} 筆資料格式看不懂（日期／價格／股數／買賣別），已略過這幾筆。")

    df = df[valid_mask].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["type_zh"] = df["type_norm"].map({"buy": "買進", "sell": "賣出"})
    return df, warnings


def compute_position_from_trades(trades: pd.DataFrame) -> dict:
    """依「移動平均成本法」逐筆計算目前持股數與平均成本：買進時用加權平均更新成本，
    賣出時不改變剩餘部位的平均成本，只實現損益。僅適用單一股票的交易紀錄。
    """
    shares = 0.0
    avg_cost = 0.0
    realized_pnl = 0.0
    rows = []
    for _, r in trades.iterrows():
        shares_before = shares
        if r["type_norm"] == "buy":
            new_shares = shares + r["quantity"]
            avg_cost = ((shares * avg_cost) + (r["quantity"] * r["price"])) / new_shares if new_shares > 0 else 0.0
            shares = new_shares
        else:  # sell
            sell_qty = min(r["quantity"], shares) if shares > 0 else r["quantity"]
            realized_pnl += (r["price"] - avg_cost) * sell_qty
            shares = max(shares - r["quantity"], 0.0)
        rows.append({
            "date": r["date"], "type_zh": r["type_zh"], "price": float(r["price"]),
            "quantity": float(r["quantity"]), "reason": r.get("reason", ""),
            "shares_before": shares_before, "shares_after": shares,
        })
    return {
        "current_shares": shares,
        "avg_cost": avg_cost,
        "realized_pnl": realized_pnl,
        "trade_rows": rows,
    }


def check_portfolio_exposure(current_shares: float, latest_close: float,
                              total_account_assets: float, cash_amount: float) -> dict:
    """對照「單一股票最大曝險35%」與「現金防禦水位15%~20%」這兩項帳戶總體風控。
    total_account_assets / cash_amount 為0時（使用者未填）視為「未提供」，不計算比例。
    """
    position_value = current_shares * latest_close
    exposure_pct = (position_value / total_account_assets) if total_account_assets > 0 else None
    cash_pct = (cash_amount / total_account_assets) if total_account_assets > 0 else None
    return {
        "position_value": position_value,
        "exposure_pct": exposure_pct,
        "exposure_over_cap": bool(exposure_pct is not None and exposure_pct > SINGLE_STOCK_MAX_EXPOSURE_PCT),
        "cash_pct": cash_pct,
        "cash_below_min": bool(cash_pct is not None and cash_pct < CASH_RESERVE_MIN_PCT),
        "cash_in_band": bool(cash_pct is not None and CASH_RESERVE_MIN_PCT <= cash_pct <= CASH_RESERVE_MAX_PCT),
        "cash_above_max": bool(cash_pct is not None and cash_pct > CASH_RESERVE_MAX_PCT),
    }


def classify_pyramid_buys(trade_rows: list, price_df: pd.DataFrame, target_total_position: float) -> list:
    """針對交易紀錄中每一筆「買進」，比對當天（或最接近的前一個交易日）的技術指標，
    嘗試歸類到使用者持倉規則的 A（逢低回測）／B（突破確認）型態，並檢查該筆加碼股數
    是否超過「目標總倉位」的25%。

    C（財報跳空缺口）型態需要「財報公布日期」資料，這支工具目前沒有串接財報行事曆，
    無法自動判斷，一律標記為「無法自動判斷」，請使用者自行對照。
    """
    results = []
    dates = price_df["date"]
    for row in trade_rows:
        if row["type_zh"] != "買進":
            continue
        buy_date = row["date"]
        idx_candidates = price_df.index[dates <= buy_date]
        if len(idx_candidates) == 0:
            results.append({
                **row, "match_status": "no_data",
                "classification": "無資料（早於工具抓取的歷史資料範圍，無法比對當天技術指標）",
                "addon_pct_of_target": (row["quantity"] / target_total_position) if target_total_position > 0 else None,
                "addon_over_cap": bool(target_total_position > 0 and row["quantity"] / target_total_position > SINGLE_ADDON_MAX_PCT_OF_TARGET),
            })
            continue

        idx = idx_candidates[-1]
        prow = price_df.loc[idx]
        close, open_ = prow["close"], prow["open"]
        rsi = prow.get("rsi14")
        ma20, ma50 = prow.get("ma20"), prow.get("ma50")
        volume, vol_ma20 = prow.get("volume"), prow.get("volume_ma20")

        near_ma20 = pd.notna(ma20) and ma20 > 0 and abs(close - ma20) / ma20 <= PYRAMID_A_MA_TOLERANCE_PCT
        near_ma50 = pd.notna(ma50) and ma50 > 0 and abs(close - ma50) / ma50 <= PYRAMID_A_MA_TOLERANCE_PCT
        type_a = bool((near_ma20 or near_ma50) and pd.notna(rsi) and rsi <= PYRAMID_A_RSI_MAX)

        window_start = max(idx - PYRAMID_B_LOOKBACK_WIN, 0)
        prior_high = price_df["high"].iloc[window_start:idx].max() if idx > window_start else np.nan
        is_red_candle = close > open_
        breakout = pd.notna(prior_high) and close > prior_high
        volume_surge = pd.notna(vol_ma20) and vol_ma20 > 0 and pd.notna(volume) and volume > PYRAMID_B_VOLUME_MULTIPLE * vol_ma20
        type_b = bool(is_red_candle and breakout and volume_surge)

        if type_a and type_b:
            classification = "同時符合 A（逢低回測）與 B（突破確認）"
        elif type_a:
            classification = "疑似符合 A：逢低回測買點"
        elif type_b:
            classification = "疑似符合 B：突破確認買點"
        else:
            classification = "未明顯符合 A 或 B（也可能是 C 財報型買點，此工具無法自動判斷財報日期，請自行確認）"

        addon_pct_of_target = (row["quantity"] / target_total_position) if target_total_position > 0 else None
        addon_pct_of_prior_shares = (row["quantity"] / row["shares_before"]) if row["shares_before"] > 0 else None

        results.append({
            **row,
            "match_status": "ok",
            "matched_date": prow["date"],
            "classification": classification,
            "type_a": type_a, "type_b": type_b,
            "rsi_at_buy": rsi, "ma20_at_buy": ma20, "ma50_at_buy": ma50,
            "addon_pct_of_target": addon_pct_of_target,
            "addon_over_cap": bool(addon_pct_of_target is not None and addon_pct_of_target > SINGLE_ADDON_MAX_PCT_OF_TARGET),
            "addon_pct_of_prior_shares": addon_pct_of_prior_shares,
        })
    return results


def check_anti_fomo_buybacks(trade_rows: list) -> list:
    """簡化版「反FOMO回補」檢查：只要賣出後 ANTI_FOMO_WINDOW_DAYS 天內，又用「更高的價格」
    買回，就標記為「疑似追高回補」。

    這是簡化版的通用判斷（賣出後短期內、價格更高的回補先一律標記），並非使用者原本規則裡
    「允許執行的客觀回補條件」完整表格的精確還原（該表格內容這次沒有完整帶到文字裡）。
    """
    flags = []
    sells = [r for r in trade_rows if r["type_zh"] == "賣出"]
    buys = [r for r in trade_rows if r["type_zh"] == "買進"]
    for s in sells:
        window_end = s["date"] + pd.Timedelta(days=ANTI_FOMO_WINDOW_DAYS)
        for b in buys:
            if s["date"] < b["date"] <= window_end and b["price"] >= s["price"]:
                flags.append({
                    "sell_date": s["date"], "sell_price": s["price"], "sell_qty": s["quantity"],
                    "buy_date": b["date"], "buy_price": b["price"], "buy_qty": b["quantity"],
                    "days_between": (b["date"] - s["date"]).days,
                    "price_diff_pct": ((b["price"] - s["price"]) / s["price"]) if s["price"] else None,
                })
    return flags


def _fmt_portfolio_check_lines(portfolio_check: dict) -> str:
    """把持倉風控與紀律檢核的結果，整理成給 Gemini 提示語與 Markdown 報告共用的條列文字。"""
    pc = portfolio_check
    lines = [
        f"- 目前持股（依交易紀錄推算，移動平均成本法）：{pc['current_shares']:.0f} 股，"
        f"平均成本 {pc['avg_cost']:.2f}，已實現損益 {pc['realized_pnl']:+.2f}",
    ]

    exp = pc.get("exposure_check")
    if exp and exp.get("exposure_pct") is not None:
        lines.append(
            f"- 單一股票曝險：持股市值 {exp['position_value']:,.0f}，"
            f"佔帳戶總資產 {exp['exposure_pct']:.1%}（上限35%）："
            f"{'已超過上限' if exp['exposure_over_cap'] else '在上限之內'}"
        )
    else:
        lines.append("- 單一股票曝險：使用者未提供帳戶總資產，無法計算")

    if exp and exp.get("cash_pct") is not None:
        lines.append(
            f"- 現金防禦水位：佔帳戶總資產 {exp['cash_pct']:.1%}（常態目標15%~20%）："
            f"{'低於下限' if exp['cash_below_min'] else ('高於上限' if exp['cash_above_max'] else '在常態區間內')}"
        )
    else:
        lines.append("- 現金防禦水位：使用者未提供現金水位，無法計算")

    pyramid = pc.get("pyramid_buys") or []
    if pyramid:
        lines.append(f"- 分批建倉型態比對（共 {len(pyramid)} 筆買進紀錄）：")
        for p in pyramid:
            over_cap_note = "；⚠️超過單次加碼25%上限" if p.get("addon_over_cap") else ""
            lines.append(
                f"  - {p['date'].date().isoformat()} 買進 {p['quantity']:.0f} 股 @ {p['price']:.2f}："
                f"{p['classification']}{over_cap_note}"
            )
    else:
        lines.append("- 分批建倉型態比對：無買進紀錄可比對")

    fomo = pc.get("fomo_flags") or []
    if fomo:
        lines.append(f"- 反FOMO回補檢查：偵測到 {len(fomo)} 筆疑似「賣出後短期內追高回補」：")
        for f in fomo:
            lines.append(
                f"  - {f['sell_date'].date().isoformat()} 賣出 @ {f['sell_price']:.2f} → "
                f"{f['buy_date'].date().isoformat()}（{f['days_between']}天後）買回 @ {f['buy_price']:.2f}"
                f"（{f['price_diff_pct']:+.1%}）"
            )
    else:
        lines.append("- 反FOMO回補檢查：未偵測到賣出後短期內追高回補的紀錄")

    return "\n".join(lines)


def build_quick_recommendation(stats: dict):
    """依 RSI、MACD、均線、KD、量能訊號組合出一個即時、規則式的操作方向建議（非 AI 生成）。"""
    reasons = []
    score = 0

    if stats["latest_macd"] > stats["latest_macd_signal"]:
        reasons.append("MACD 呈現黃金交叉，短期動能偏多")
        score += 1
    else:
        reasons.append("MACD 呈現死亡交叉，短期動能偏空")
        score -= 1

    if stats["latest_rsi"] >= 70:
        reasons.append(f"RSI（{stats['latest_rsi']:.1f}）進入超買區間，短線追高風險增加")
        score -= 1
    elif stats["latest_rsi"] <= 30:
        reasons.append(f"RSI（{stats['latest_rsi']:.1f}）進入超賣區間，留意反彈機會")
        score += 1
    else:
        reasons.append(f"RSI（{stats['latest_rsi']:.1f}）處於中性區間，未出現極端訊號")

    if "放量" in stats["vol_signal"]:
        reasons.append("成交量放大，可作為趨勢是否延續的驗證訊號")
    elif "縮量" in stats["vol_signal"]:
        reasons.append("成交量縮減，動能可能不足，訊號可信度需打折扣")

    if "黃金交叉" in stats.get("ma_cross", ""):
        reasons.append("均線MA5站上MA20，短期趨勢轉強")
        score += 1
    else:
        reasons.append("均線MA5跌破MA20，短期趨勢轉弱")
        score -= 1

    if stats.get("kd_signal") == "超買區間":
        reasons.append(f"KD（K={stats['latest_kd_k']:.1f}）進入超買區間，追高風險增加")
        score -= 1
    elif stats.get("kd_signal") == "超賣區間":
        reasons.append(f"KD（K={stats['latest_kd_k']:.1f}）進入超賣區間，留意反彈機會")
        score += 1

    obv_signal = stats.get("obv_signal", "")
    if "量價同步" in obv_signal:
        reasons.append(f"OBV：{obv_signal}")
        score += 1 if "上升" in obv_signal else -1
    elif "量價背離" in obv_signal:
        reasons.append(f"OBV：{obv_signal}，訊號可信度需打折扣")

    if score >= 2:
        action = "偏多／可留意分批進場"
    elif score <= -2:
        action = "偏空／建議觀望待訊號轉強"
    else:
        action = "中性／訊號不一致，建議續觀察"

    return action, reasons, score


# ----------------------------------------------------------------------------
# 2d. 型態分析：K線型態辨識 + 趨勢／突破型態
# ----------------------------------------------------------------------------

def detect_candlestick_patterns(df: pd.DataFrame, lookback: int = 5) -> list:
    """掃描最近 lookback 天的 K 線，辨識幾種常見的單根／雙根蠟燭型態。

    回傳 [(日期, 型態名稱, 意涵), ...]，僅偵測型態明顯（比例夠極端）的情況，
    避免雜訊過多。這是簡化版規則判斷，非嚴謹的專業技術分析軟體演算法。
    """
    patterns = []
    recent = df.tail(lookback + 1).reset_index(drop=True)

    for i in range(1, len(recent)):
        row = recent.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        rng = (h - l) if h != l else 1e-9
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        date_str = row["date"].date().isoformat()

        if body / rng < 0.1:
            patterns.append((date_str, "十字星（Doji）", "多空拉鋸，可能為反轉前兆"))

        if lower_shadow > body * 2 and upper_shadow < body * 0.5 and body / rng < 0.4:
            prev_close = recent.iloc[i - 1]["close"]
            if c < prev_close:
                patterns.append((date_str, "鎚子線（Hammer）", "跌勢中出現，留意止跌反彈"))
            else:
                patterns.append((date_str, "吊人線（Hanging Man）", "漲勢中出現，留意反轉向下"))

        prev = recent.iloc[i - 1]
        prev_top, prev_bottom = max(prev["open"], prev["close"]), min(prev["open"], prev["close"])
        curr_top, curr_bottom = max(o, c), min(o, c)
        if c > o and prev["close"] < prev["open"] and curr_top >= prev_top and curr_bottom <= prev_bottom:
            patterns.append((date_str, "看漲吞噬（Bullish Engulfing）", "偏多反轉訊號"))
        elif c < o and prev["close"] > prev["open"] and curr_top >= prev_top and curr_bottom <= prev_bottom:
            patterns.append((date_str, "看跌吞噬（Bearish Engulfing）", "偏空反轉訊號"))

    return patterns


def detect_trend_pattern(df: pd.DataFrame, short_win: int = 20, breakout_win: int = 20) -> dict:
    """用線性回歸斜率判斷近期趨勢方向，並檢查是否出現近 N 日高低點的突破／跌破。"""
    close = df["close"]
    recent_short = close.tail(short_win)
    x = np.arange(len(recent_short))
    slope = np.polyfit(x, recent_short.values, 1)[0] if len(recent_short) > 1 else 0
    norm_slope = slope / recent_short.mean() if recent_short.mean() else 0

    if norm_slope > 0.001:
        trend = "上升趨勢"
    elif norm_slope < -0.001:
        trend = "下降趨勢"
    else:
        trend = "區間盤整"

    if len(df) > breakout_win:
        prior_high = df["high"].iloc[-(breakout_win + 1):-1].max()
        prior_low = df["low"].iloc[-(breakout_win + 1):-1].min()
    else:
        prior_high = df["high"].max()
        prior_low = df["low"].min()

    latest_close = close.iloc[-1]
    breakout = None
    if latest_close > prior_high:
        breakout = f"向上突破近{breakout_win}日高點（{prior_high:.2f}）"
    elif latest_close < prior_low:
        breakout = f"向下跌破近{breakout_win}日低點（{prior_low:.2f}）"

    return {
        "trend": trend,
        "breakout": breakout,
        "support": prior_low,
        "resistance": prior_high,
    }


# ----------------------------------------------------------------------------
# 3. 視覺化圖表
# ----------------------------------------------------------------------------

def chart_technical(df, symbol, color):
    # 這張圖用 make_subplots 把5個子圖（價格/OBV/KD/RSI/MACD）疊在同一張圖裡，
    # 但 Plotly 的圖例（legend）是「整張圖共用一份」，不會照子圖分開，所以底下
    # 每一條線都刻意指定成「整張圖裡獨一無二」的顏色，即使兩條線位於不同子圖、
    # 用途完全不同（例如 MA60 跟 RSI14、KD-K 跟 MACD線、KD-D 跟訊號線），
    # 之前分別共用同一色碼，會讓人誤以為看錯圖或顏色設定重複，這裡全部改開。
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.32, 0.17, 0.17, 0.17, 0.17], vertical_spacing=0.045,
        subplot_titles=(f"{symbol}：K線、均線、布林通道與成交量", "OBV（能量潮）",
                         "KD (9)", "RSI (14)", "MACD (12,26,9)"),
        specs=[[{"secondary_y": True}], [{}], [{}], [{}], [{}]],
    )

    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"], name="布林上軌",
                              line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
                              showlegend=False), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"], name="布林通道",
                              line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(150,150,150,0.12)"),
                  row=1, col=1, secondary_y=False)
    # 股價改成蠟燭圖（K線）呈現，而不是只畫收盤價的折線，可以同時看到當天開高低收；
    # 漲用紅、跌用綠沿用下方 MACD 柱狀圖同一套配色語意，整份報表的漲跌顏色保持一致。
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="K線", increasing=dict(line=dict(color="#dc2626")), decreasing=dict(line=dict(color="#16a34a")),
    ), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma5"], name="MA5",
                              line=dict(color="#eab308", width=1.2)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA20",
                              line=dict(color="#0ea5e9", width=1.2)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma60"], name="MA60",
                              line=dict(color="#7c3aed", width=1.2)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量",
                          marker=dict(color="rgba(150,150,150,0.35)")), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df["date"], y=df["obv"], name="OBV",
                              line=dict(color="#06b6d4", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["obv_ma20"], name="OBV 20日均線",
                              line=dict(color="#94a3b8", width=1, dash="dot")), row=2, col=1)

    fig.add_trace(go.Scatter(x=df["date"], y=df["kd_k"], name="K",
                              line=dict(color="#2563eb", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["kd_d"], name="D",
                              line=dict(color="#f97316", width=1.5)), row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="green", row=3, col=1)

    # RSI14 原本跟 MA60 同色（都是#7c3aed），改成洋紅色以區隔。
    fig.add_trace(go.Scatter(x=df["date"], y=df["rsi14"], name="RSI(14)",
                              line=dict(color="#db2777", width=1.5)), row=4, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=4, col=1)

    hist_colors = np.where(df["macd_hist"] >= 0, "#16a34a", "#dc2626")
    fig.add_trace(go.Bar(x=df["date"], y=df["macd_hist"], name="MACD 柱狀圖",
                          marker=dict(color=hist_colors)), row=5, col=1)
    # MACD線原本跟KD的K同色、訊號線原本跟KD的D同色（都各自共用#2563eb／#f97316），
    # 改成不重複的顏色，避免在共用圖例裡誤以為是同一條線。
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd"], name="MACD 線",
                              line=dict(color="#0d9488", width=1.5)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="訊號線",
                              line=dict(color="#92400e", width=1.5)), row=5, col=1)

    # 蠟燭圖預設會在最下方多出一條 range slider，在這種多子圖疊在一起的版面
    # 會跟下面的 OBV/KD/RSI/MACD 子圖重疊，這裡關掉。
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    fig.update_yaxes(title_text="價格", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="成交量", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="OBV", row=2, col=1)
    fig.update_yaxes(title_text="KD", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="RSI", row=4, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=5, col=1)

    fig.update_layout(template="plotly_white", height=1020, showlegend=True,
                       margin=dict(l=40, r=20, t=60, b=40))
    return fig


def chart_backtest(backtest: dict, symbol: str, color: str):
    signals = backtest["signals"]
    names = list(signals.keys())
    win_rates = [(signals[n]["win_rate"] * 100 if signals[n]["win_rate"] is not None else 0) for n in names]
    counts = [signals[n]["n"] for n in names]
    baseline_wr = backtest["baseline"]["win_rate"] or 0

    # n=0 的訊號本身就沒有勝率可算，長條高度會是 0，若只在柱子頂端（也就是 y=0 貼著
    # x 軸的地方）標「n=0」小字，視覺上幾乎看不到、容易被誤以為是程式空白或錯誤。
    # 這裡改成：這種欄位不畫柱內文字，另外在該欄位固定高度處放一個醒目的「無樣本」
    # 註記框，讓使用者一眼就看出「這不是bug，是歷史上這個訊號真的沒出現過」。
    bar_texts = ["" if n == 0 else f"n={n}" for n in counts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=win_rates, name="訊號出現後隔日上漲機率",
        marker=dict(color=color),
        text=bar_texts, textposition="outside",
    ))

    for name, n in zip(names, counts):
        if n == 0:
            fig.add_annotation(
                x=name, y=8, text="⚠️ 無樣本<br>（歷史上未出現過）", showarrow=False,
                font=dict(color="#6b7280", size=12),
                bgcolor="rgba(107,114,128,0.12)", bordercolor="#6b7280", borderwidth=1,
                borderpad=4,
            )

    fig.add_hline(y=baseline_wr * 100, line_dash="dash", line_color="gray",
                  annotation_text=f"基準（全期間任一交易日隔日上漲機率）{baseline_wr:.1%}",
                  annotation_position="top left")
    fig.update_layout(
        title=f"{symbol}：訊號歷史勝率回測（隔日上漲機率 vs 基準）",
        yaxis_title="隔日上漲機率 (%)", yaxis_range=[0, 100],
        template="plotly_white", height=380,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def chart_sentiment_pie(news_stats: dict, symbol: str):
    """個股情緒分布 + 整體文章情緒分布，並排的兩個環狀圖（配色依五級情緒標準配色）。

    兩張圖用的是完全相同的五級標籤與配色對應（labels / colors 陣列一致），
    差異只在於底層資料來源不同：
      - 左圖「個股情緒分布」只計算每則新聞中「明確標記為 {symbol}」的情緒判斷
        （Alpha Vantage 的 ticker_sentiment，僅限有對應到這支股票的部分）。
      - 右圖「整體文章情緒分布」則是整篇文章的情緒（overall_sentiment），涵蓋
        文章討論的所有主題，不限於 {symbol}。
    因此兩張圖看起來顏色分布不同，並非配色錯誤，而是兩者統計的資料範圍本來就不同；
    若某個情緒類別在其中一張圖裡則數為 0，該顏色的扇形會消失（角度為 0），也會讓
    兩張圖「看起來用到的顏色種類」不一樣。
    """
    labels = SENTIMENT_LABELS_ORDER
    display_labels = [_sentiment_label_emoji(l) for l in labels]
    colors = [SENTIMENT_COLORS[l] for l in labels]
    ticker_values = [news_stats["ticker_sentiment_dist"][l]["count"] for l in labels]
    article_values = [news_stats["article_sentiment_dist"][l]["count"] for l in labels]

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
                         subplot_titles=(f"{symbol} 個股情緒分布", "整體文章情緒分布"))

    # Plotly 的 Pie 預設會依數值大小重新排序扇形（sort=True），這會打亂我們刻意
    # 排好的「Bearish→Somewhat-Bearish→Neutral→Somewhat-Bullish→Bullish」情緒光譜順序，
    # 導致同樣偏多方向的 Bullish 跟 Somewhat-Bullish 被中間插入的其他類別隔開、
    # 看起來不連續。加上 sort=False 讓扇形照 labels 陣列的順序排列，保持光譜連續。
    fig.add_trace(go.Pie(labels=display_labels, values=ticker_values, marker=dict(colors=colors),
                          hole=0.4, name="個股情緒", sort=False, direction="clockwise"), row=1, col=1)
    fig.add_trace(go.Pie(labels=display_labels, values=article_values, marker=dict(colors=colors),
                          hole=0.4, name="文章情緒", showlegend=False, sort=False, direction="clockwise"),
                  row=1, col=2)

    fig.update_layout(template="plotly_white", height=380, showlegend=True,
                       margin=dict(l=20, r=20, t=60, b=20))
    return fig


def chart_sentiment_time_series(news_list: list, symbol: str):
    """個股新聞情緒分數隨時間變化的散佈／折線圖，並標示 Bullish／Bearish 分類門檻。"""
    dated = sorted(
        ((x["published_at"], x["ticker_sentiment_score"]) for x in news_list
         if x["published_at"] is not None and x["ticker_sentiment_score"] is not None),
        key=lambda t: t[0],
    )

    fig = go.Figure()
    if dated:
        xs, ys = zip(*dated)
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="markers+lines", name="個股情緒分數",
                                  line=dict(color="#2563eb", width=1.5), marker=dict(size=6)))
    fig.add_hline(y=0.35, line_dash="dot", line_color="#2ECC71", annotation_text="Bullish 門檻 0.35")
    fig.add_hline(y=-0.35, line_dash="dot", line_color="#DC143C", annotation_text="Bearish 門檻 -0.35")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"{symbol}：新聞情緒分數隨時間變化",
        yaxis_title="情緒分數（-1 極度看空 ～ +1 極度看多）", yaxis_range=[-1, 1],
        template="plotly_white", height=380, margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def chart_source_distribution(news_stats: dict):
    """新聞來源分布橫條圖（前10大來源）。"""
    sources = list(news_stats["source_counts"].keys())[:10]
    counts = [news_stats["source_counts"][s] for s in sources]
    fig = go.Figure(go.Bar(x=counts, y=sources, orientation="h", marker=dict(color="#2563eb")))
    fig.update_layout(
        title="新聞來源分布（前10大來源）", xaxis_title="新聞則數",
        template="plotly_white", height=max(300, 34 * max(len(sources), 1)),
        margin=dict(l=140, r=20, t=60, b=40), yaxis=dict(autorange="reversed"),
    )
    return fig


def chart_fear_greed_gauge(fg: dict):
    """CNN Fear & Greed Index 儀表圖（0=極度恐懼 ～ 100=極度貪婪）。"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fg["score"],
        title={"text": f"CNN Fear & Greed Index：{fg['rating_zh']}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#111827"},
            "steps": [
                {"range": [0, 25], "color": "#DC143C"},
                {"range": [25, 45], "color": "#FF6B6B"},
                {"range": [45, 55], "color": "#95A5A6"},
                {"range": [55, 75], "color": "#90EE90"},
                {"range": [75, 100], "color": "#2ECC71"},
            ],
        },
    ))
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=30, r=30, t=60, b=10))
    return fig


# ----------------------------------------------------------------------------
# 4. Gemini AI 分析
# ----------------------------------------------------------------------------

def _fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2%}"


def _fmt_num(x, digits=2):
    return "N/A" if pd.isna(x) else f"{x:,.{digits}f}"


def _fmt_patterns(patterns: list) -> str:
    if not patterns:
        return "近期未偵測到明顯的典型K線型態"
    return "；".join(f"{d} 出現{name}（{meaning}）" for d, name, meaning in patterns)


def _fmt_backtest(backtest: dict) -> str:
    baseline = backtest["baseline"]
    lines = [f"基準（全期間任一交易日隔日上漲機率）：{_fmt_pct(baseline['win_rate'])}（樣本數 {baseline['n']}）"]
    for name, r in backtest["signals"].items():
        if r["win_rate"] is None or r["n"] < 5:
            lines.append(f"- {name}：樣本數過少（n={r['n']}），不具統計參考性")
        else:
            lines.append(f"- {name}：出現後隔日上漲機率 {_fmt_pct(r['win_rate'])}（樣本數 {r['n']}）")
    return "\n".join(lines)


def _fmt_quote(quote) -> str:
    if not quote or quote.get("price") is None:
        return "目前無即時報價資料"
    change = quote.get("change")
    change_pct = quote.get("change_pct")
    updated = quote.get("updated_at")
    updated_str = updated.strftime("%Y-%m-%d %H:%M:%S") if updated else "N/A"
    change_str = f"，漲跌 {change:+.2f}（{change_pct:+.2f}%）" if change is not None and change_pct is not None else ""
    return f"{_fmt_num(quote.get('price'))}{change_str}（更新時間 {updated_str}，本機時區）"


def build_prompt(symbol, stats, start_date, end_date, trend=None, patterns=None,
                  backtest=None, quote=None, stop_plan=None, portfolio_check=None) -> str:
    """建構送給 Gemini 的分析提示語（單一個股獨立分析）。

    設計重點：
      1. 明確角色設定（量化分析師）與適用範圍限制（僅根據歷史技術指標，非投資建議）。
      2. 提供結構化數據，避免模型自行臆測或幻覺產生數字。
      3. 明確要求引用「訊號回測」的歷史勝率數字，避免對單一訊號過度自信地斷言多空方向。
      4. 停損／停利直接引用 compute_playbook_stop_plan() 算出的實際數字（使用者自己的
         持倉規則），不再讓 AI 自行發明百分比，避免畫面上的規則式數字跟 AI 文字互相矛盾。
    """
    trend = trend or {}
    patterns = patterns or []
    stop_plan_block = _fmt_stop_plan_lines(stop_plan) if stop_plan else "（本次未提供停損／停利機制資料）"
    portfolio_block = (
        _fmt_portfolio_check_lines(portfolio_check) if portfolio_check
        else "（本次未上傳交易紀錄，無持倉風控與紀律檢核資料）"
    )

    stock_block = f"""【{symbol} 統計數據（歷史技術指標資料更新至 {stats['latest_date'].date().isoformat()}）】
- 即時報價（另一獨立來源，僅供對照現在價格，非技術指標依據）：{_fmt_quote(quote)}
- 期間總報酬率：{_fmt_pct(stats['total_return'])}
- 年化報酬率：{_fmt_pct(stats['ann_return'])}
- 年化波動度：{_fmt_pct(stats['ann_vol'])}
- Sharpe Ratio（以0%無風險利率計算）：{_fmt_num(stats['sharpe'])}
- 最大回撤：{_fmt_pct(stats['max_drawdown'])}
- 最新收盤價：{_fmt_num(stats['latest_close'])}
- 最新 RSI(14)：{_fmt_num(stats['latest_rsi'], 1)}（{stats['rsi_signal']}）
- 最新 MACD：{_fmt_num(stats['latest_macd'], 3)} / 訊號線：{_fmt_num(stats['latest_macd_signal'], 3)}（{stats['macd_cross']}）
- 最新成交量：{_fmt_num(stats['latest_volume'], 0)}，20日均量：{_fmt_num(stats['avg_volume20'], 0)}（{stats['vol_signal']}）
- 均線：MA5={_fmt_num(stats['ma5'])}／MA20={_fmt_num(stats['ma20'])}／MA60={_fmt_num(stats['ma60'])}（{stats['ma_cross']}；{stats['ma_alignment']}）
- 布林通道：上軌={_fmt_num(stats['bb_upper'])}／中軌={_fmt_num(stats['bb_mid'])}／下軌={_fmt_num(stats['bb_lower'])}（{stats['bb_position']}）
- KD：K={_fmt_num(stats['latest_kd_k'],1)}／D={_fmt_num(stats['latest_kd_d'],1)}（{stats['kd_cross']}，{stats['kd_signal']}）
- OBV（能量潮，代表量能方向）：{_fmt_num(stats['latest_obv'], 0)}（{stats['obv_signal']}）
- 趨勢型態：{trend.get('trend', 'N/A')}；支撐 {_fmt_num(trend.get('support'))} ／壓力 {_fmt_num(trend.get('resistance'))}
  {('；' + trend['breakout']) if trend.get('breakout') else ''}
- 近期K線型態：{_fmt_patterns(patterns)}
- 訊號歷史回測（隔日上漲機率）：
{_fmt_backtest(backtest)}
- 使用者自己的停損／停利機制（規則式計算，非AI生成，以下數字為畫面上實際顯示的數字，
  分析時必須直接引用這些數字，不可自行另外發明其他百分比或金額）：
{stop_plan_block}
- 使用者的持倉風控與紀律檢核（規則式計算，非AI生成，若顯示「未上傳交易紀錄」則代表
  這次沒有這份資料，不要臆測數字）：
{portfolio_block}
"""

    prompt = f"""你是一位專業的量化投資分析師，擅長技術面、型態面分析，並且非常重視統計證據而非空泛斷言。
以下是 {symbol} 這檔股票在 {start_date.isoformat()} 至 {end_date.isoformat()} 期間，根據歷史股價計算出的
技術指標、型態辨識、統計數據，以及各項訊號的「歷史回測勝率」。請你完全根據這些數據分析，不要引用你自己
對這家公司其他方面（如未提供的財報、新聞）的既有印象。

{stock_block}

請嚴格依照以下框架，用繁體中文輸出分析報告（使用 Markdown 標題，勿使用表格）：

## 1. 技術面與型態面分析
整合解讀 RSI、MACD、KD、均線排列、布林通道位置，以及偵測到的K線型態與趨勢/突破狀況，判斷目前多空
力道與可信度（例如指標是否互相驗證，或出現分歧訊號），並根據 OBV 與價格走勢是否同步，判斷目前量能
是否真的支撐價格趨勢（量價同步代表動能可信，量價背離則需提高警覺）。

## 2. 訊號可靠度（依歷史回測）
針對目前顯示的主要訊號（例如MACD黃金交叉、RSI超賣等），對照上方提供的「歷史回測勝率」數字，說明這個
訊號過去的實際命中率是否明顯優於基準勝率、樣本數是否足夠，藉此評估目前訊號的可信度高低，避免對任何
單一訊號做出過度自信的斷言。

## 3. 停損／停利機制解讀
上方「使用者自己的停損／停利機制」已經是規則式算好的實際數字，這一段只需要針對這些
「已知數字」做解讀與情境說明，禁止另外發明任何其他百分比、金額或風險等級（不要出現
「保守型／穩健型／積極型」這類分級）：
- 逐項說明目前四個狀態（新加碼部位停損、整體核心持倉防禦停損、第一階段停利目標、
  第二階段追蹤停利）距離目前股價各自還有多少空間（用百分比描述即可，數字必須跟上方
  提供的金額換算一致），以及目前分別處於「未觸發／觀察中／已觸發」的哪個狀態。
- 若「整體核心持倉防禦停損」目前是「觀察中」或「已觸發」，明確提醒使用者這條規則的前提
  是「基本面未變（ROE／營收維持強勁）」，本分析無法評估基本面，請使用者自行確認。
- 若第一階段停利目標的來源是「近期壓力價（自動估算）」而非使用者自訂，明確提醒這個目標
  是用近期高低點估算出來的，並非使用者原本設定的實際目標價，僅供參考。
- 結合上方 RSI／MACD／KD／均線等技術訊號，說明目前比較接近觸發哪一條規則、值得優先
  留意哪個價位，但仍只能描述「距離」與「情境」，不可給出「建議買進／賣出」之類的操作
  指令，也不可承諾價格一定會到達或跌破某個價位。

## 4. 持倉風控與紀律檢核解讀
若上方顯示「未上傳交易紀錄」，這一段只需要寫一句話說明本次未提供持倉資料、故略過，不要
臆測或編造任何持股/曝險數字。若有提供資料，則針對「使用者的持倉風控與紀律檢核」裡已經算好
的項目（單一股票曝險是否超過35%、現金水位是否在15%~20%常態區間、每筆加碼是否超過目標倉位
25%上限、每筆加碼比較接近A/B哪個型態或屬於無法判斷、是否有疑似反FOMO回補的紀錄）逐項做
情境說明，一樣禁止另外發明任何新的百分比或判斷標準；若曝險超過35%是因為股價上漲造成，提醒
使用者這種情況依規則是「僅以追蹤停利調節、不主動猜頂」，不等於要立刻出清。

## 5. 風險提醒
列出根據目前數據看到的主要風險因子（如波動度偏高、回撤過大、RSI 已達極端區間、訊號樣本數過少等），
並註明：本分析僅根據歷史技術指標、型態辨識與統計回測產生，技術指標為落後指標，不代表對未來（尤其是
單一交易日）漲跌的預測，不構成投資建議，投資人應自行評估風險並諮詢專業意見。

請維持專業、精簡的語氣，避免重複同樣的話術，每一段落盡量在 150 字以內。
"""
    return prompt


def call_gemini(prompt: str, api_key: str, model: str = "gemini-3.5-flash-lite") -> str:
    url = GEMINI_URL_TMPL.format(model=model)
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, params={"key": api_key}, json=body, timeout=90)

    if resp.status_code != 200:
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        raise RuntimeError(
            f"Gemini API 呼叫失敗（HTTP {resp.status_code}）：{msg}\n"
            f"若是 404 / model not found，請確認 model 名稱是否仍然有效"
            f"（可至 https://ai.google.dev/gemini-api/docs/models 查詢目前可用模型）。"
        )

    data = resp.json()
    try:
        candidate = data["candidates"][0]
        if "content" not in candidate:
            reason = candidate.get("finishReason", "未知原因")
            raise RuntimeError(f"Gemini 未回傳內容（finishReason: {reason}），請調整輸入內容後再試一次。")
        parts = candidate["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"無法解析 Gemini 回傳內容：{json.dumps(data, ensure_ascii=False)[:500]}") from exc


def build_news_summary_prompt(symbol: str, news_stats: dict, news_list: list, fg: dict = None) -> str:
    """建構送給 Gemini 的「市場情緒總結」提示語：只要求 3-5 句話的簡短客觀總結。"""
    top_titles = "\n".join(
        f"- {x['title']}（{x['source']}，{x['ticker_sentiment_label']}）" for x in news_list[:15]
    )
    fg_line = (f"目前 CNN Fear & Greed Index：{fg['score']:.0f} 分（{fg['rating_zh']}）"
               if fg and fg.get("score") is not None else "（本次未取得 Fear & Greed Index 資料）")

    return f"""你是一位客觀中立的市場情緒分析師。請根據以下 {symbol} 的新聞情緒統計資料，用 3 到 5 句話、
以繁體中文簡短總結目前市場對這檔股票的新聞情緒狀態。只描述「數據顯示」的歷史情緒統計現象，不要給出
任何投資建議、預測或操作暗示，也不要使用「應該」「建議」「預期」等字眼。

{fg_line}
個股情緒分布（則數）：{ {k: v['count'] for k, v in news_stats['ticker_sentiment_dist'].items()} }
文章情緒分布（則數）：{ {k: v['count'] for k, v in news_stats['article_sentiment_dist'].items()} }
平均個股情緒分數：{_fmt_num(news_stats['avg_ticker_score'], 3)}
平均相關性分數：{_fmt_num(news_stats['avg_relevance'], 3)}
主要新聞來源：{list(news_stats['source_counts'].keys())[:5]}

近期新聞標題（節錄）：
{top_titles}
"""


def build_news_ai_prompt(symbol: str, stats: dict, news_stats: dict, news_list: list, fg: dict = None) -> str:
    """建構送給 Gemini 的「新聞情緒深度分析」提示語：結合股價技術面資料與新聞情緒統計，
    要求輸出「分析總結／媒體情緒觀察／綜合分析結果」三段式結構化報告。

    設計原則沿用規格說明書：角色設定為客觀中立的情緒分析師、僅描述歷史統計現象、
    禁止暗示性的建議用詞、全程繁體中文、並附上明確的教育性免責聲明。
    """
    fg_block = (
        f"- CNN Fear & Greed Index：{fg['score']:.0f} 分（{fg['rating_zh']}），"
        f"前一交易日 {_fmt_num(fg['previous_close'], 0)} 分／一週前 {_fmt_num(fg['previous_1_week'], 0)} 分／"
        f"一個月前 {_fmt_num(fg['previous_1_month'], 0)} 分"
        if fg and fg.get("score") is not None else "- 本次未取得 CNN Fear & Greed Index 資料"
    )

    news_detail = "\n".join(
        f"{i+1}. [{x['published_at'].strftime('%Y-%m-%d %H:%M') if x['published_at'] else 'N/A'}] "
        f"{x['title']}（來源：{x['source']}／個股情緒：{x['ticker_sentiment_label']}"
        f"（{_fmt_num(x['ticker_sentiment_score'], 3)}）／相關性：{_fmt_num(x['ticker_relevance'], 3)}）"
        for i, x in enumerate(news_list[:20])
    )

    return f"""你是一位專業的市場情緒分析師，專精於新聞情緒解讀與投資心理分析。你的職責包括：
1. 客觀分析新聞情緒數據的分布和趨勢
2. 解讀市場情緒對投資決策的潛在影響（僅描述歷史現象，非預測）
3. 識別新聞來源的可信度和影響力
4. 提供純教育性的情緒分析知識

重要原則：
- 僅提供歷史新聞情緒分析和市場心理解讀，絕不提供任何投資建議或預測
- 保持完全客觀中立的分析態度；使用專業術語但保持易懂；全程使用繁體中文
- 使用「數據顯示」「情緒指標反映」「歷史統計呈現」等客觀描述
- 避免「可能性」「預期」「建議」「應該」等暗示性用詞
- 禁用「如果...則...」的假設句型，改用「歷史上當...時，曾出現...現象」
- 強調「情緒數據僅供參考，不代表投資結果」

### {symbol} 基本資訊
- 分析新聞則數：{news_stats['n_news']}
- 最新收盤價（技術指標資料，更新至 {stats['latest_date'].date().isoformat()}）：{_fmt_num(stats['latest_close'])}
{fg_block}

### 情緒統計數據
- 個股情緒分布（則數）：{ {k: v['count'] for k, v in news_stats['ticker_sentiment_dist'].items()} }
- 文章情緒分布（則數）：{ {k: v['count'] for k, v in news_stats['article_sentiment_dist'].items()} }
- 平均個股情緒分數：{_fmt_num(news_stats['avg_ticker_score'], 3)}（-1 極度看空 ～ +1 極度看多）
- 平均相關性分數：{_fmt_num(news_stats['avg_relevance'], 3)}（0 無關 ～ 1 高度相關）

### 新聞來源分布
{news_stats['source_counts']}

### 主要主題分析（依加權相關性排序）
{news_stats['topic_scores']}

### 新聞情緒明細（依時間排序，最新在前）
{news_detail}

請嚴格依照以下結構，用繁體中文輸出（使用 Markdown 標題，勿使用表格）：

## 分析總結
以 3-5 句話總結目前新聞情緒的整體樣貌與強度。

## 媒體情緒觀察
依據上述新聞標題與摘要，具體指出情緒分布是否集中或分歧、是否有特定來源立場明顯偏多或偏空，
並比較 Fear & Greed Index（若有資料，屬於大盤層級的整體市場情緒）與個股新聞情緒（僅反映這支
股票本身的新聞語氣）兩者方向是否一致；若不一致，請說明這是「大盤氣氛」與「個股消息面」本來
就衡量不同範圍、可能出現分歧的正常現象，不要暗示其中一方有誤。

## 綜合分析結果（短期與中長期）
分別就「短期（新聞驅動的情緒波動）」與「中長期（基本面／趨勢層面）」，描述歷史上類似情緒組合
出現後的一般觀察現象，並列出至少 3 項需要留意的風險與限制（含情緒數據本身的局限性、樣本可能
的偏誤等）。全文避免給出具體的買賣操作建議。

免責聲明：本分析僅基於新聞情緒與 Fear & Greed Index 的統計解讀，純供教育與研究參考，不構成
任何投資建議或未來走勢預測。
"""


# ----------------------------------------------------------------------------
# 5. Streamlit 主流程
# ----------------------------------------------------------------------------

def render_performance_summary(stats: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("期間總報酬率", _fmt_pct(stats["total_return"]))
    c2.metric("年化波動度", _fmt_pct(stats["ann_vol"]))
    c3.metric("Sharpe Ratio", _fmt_num(stats["sharpe"]))
    c4.metric("最大回撤", _fmt_pct(stats["max_drawdown"]))
    c1b, c2b, c3b = st.columns(3)
    c1b.metric("RSI(14)", f"{_fmt_num(stats['latest_rsi'], 1)}（{stats['rsi_signal']}）")
    c2b.metric("MACD 訊號", stats["macd_cross"])
    c3b.metric("量能訊號", stats["vol_signal"])


def render_realtime_quote(symbol, quote, error_msg=None):
    st.markdown("##### ⚡ 即時報價")
    if quote is None:
        st.warning(f"目前無法取得 {symbol} 的即時報價" + (f"：{error_msg}" if error_msg else "") +
                   "。下方技術指標分析仍正常，僅根據歷史每日收盤資料計算。")
        return

    change = quote.get("change")
    change_pct = quote.get("change_pct")
    delta_str = None
    if change is not None and change_pct is not None:
        delta_str = f"{change:+.2f} ({change_pct:+.2f}%)"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目前價格", _fmt_num(quote.get("price")), delta_str)
    c2.metric("今日區間", f"{_fmt_num(quote.get('day_low'))} ~ {_fmt_num(quote.get('day_high'))}")
    c3.metric("成交量", _fmt_num(quote.get("volume"), 0))
    updated_str = quote["updated_at"].strftime("%Y-%m-%d %H:%M:%S") if quote.get("updated_at") else "N/A"
    c4.metric("報價更新時間（本機時區）", updated_str)
    st.caption("即時報價是否即時、有無延遲，依 FMP 帳號方案而定；下方 RSI／MACD／KD／均線等技術指標"
               "仍是根據歷史每日收盤價計算，兩者為互補的獨立資訊，並非同一組數字。")


def render_playbook_stop_plan(stop_plan: dict):
    """顯示依使用者持倉規則算出的停損／停利機制（取代原本的固定百分比區塊）。"""
    st.markdown("##### 📋 停損／停利機制（依你的持倉規則計算，不需呼叫 AI）")

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**🛑 停損**")
        st.metric(
            "① 新加碼部位停損（-7%）",
            f"${stop_plan['new_position_stop_loss_price']:.2f}",
            f"-{stop_plan['new_position_stop_loss_pct']:.0%}",
        )
        st.caption(f"以目前收盤價 ${stop_plan['entry_ref_price']:.2f} 作為進場參考價換算；"
                   "若這是你這次新加碼的部位，成本價跌破這個價位就無條件停損。")

        wd = stop_plan["weekly_defense"]
        if wd["status"] == "insufficient_data":
            st.info(f"② 整體核心持倉防禦停損：歷史資料僅約 {wd['weeks_available']} 週，"
                    "還不足 50 週，暫時無法計算週線 50 週均線，累積更多資料後會自動出現。")
        else:
            if wd["confirmed"]:
                st.error(f"② 整體核心持倉防禦停損：**已觸發**（週收盤 ${wd['latest_weekly_close']:.2f} "
                          f"跌破 50 週均線 ${wd['weekly_ma']:.2f}，且最近 {WEEKLY_DEFENSE_CONFIRM_DAYS} "
                          "個交易日收盤都未收復）")
            elif wd["weekly_broken"]:
                st.warning(f"② 整體核心持倉防禦停損：**觀察中**（週收盤 ${wd['latest_weekly_close']:.2f} "
                            f"已跌破 50 週均線 ${wd['weekly_ma']:.2f}，但最近幾個交易日收盤尚有收復，"
                            "未滿足「3日未收復」的確認條件）")
            else:
                st.success(f"② 整體核心持倉防禦停損：**未觸發**（週收盤 ${wd['latest_weekly_close']:.2f} "
                            f"仍在 50 週均線 ${wd['weekly_ma']:.2f} 之上）")
            st.caption("這條防線的前提是「基本面未變（ROE／營收維持強勁）」，屬於質化判斷，"
                       "本工具無法自動評估，觸發前請自行確認基本面是否仍然穩健。")

    with sc2:
        st.markdown("**🎯 停利（分階段，非固定金額）**")
        if stop_plan["stage1_target"] is not None:
            reached = "✅ 已達標" if stop_plan["stage1_reached"] else "尚未達標"
            st.metric(f"① 第一階段目標價（{reached}）",
                       f"${stop_plan['stage1_target']:.2f}",
                       f"最多賣出 {stop_plan['stage1_sell_pct']:.0%}，其餘續抱")
            st.caption(f"目標價來源：{stop_plan['stage1_target_source']}。")
        else:
            st.info("① 第一階段目標價：尚無法自動估算（近期高低點資料不足），可在上方欄位自行填入目標價。")

        if stop_plan["trailing_ma"] is not None:
            status = "⚠️ 已跌破，建議調節" if stop_plan["trailing_broken"] else "✅ 持有中（守住均線之上）"
            st.metric(f"② 追蹤停利（{TRAILING_STOP_MA_WINDOW}日均線，{status}）",
                       f"${stop_plan['trailing_ma']:.2f}",
                       f"跌破時調節 {stop_plan['trailing_sell_pct']:.0%}")
            st.caption(f"適用於達第一階段目標價後的剩餘部位：只要收盤守住 {TRAILING_STOP_MA_WINDOW}日均線"
                       "之上就不設停利上限、全程持有；收盤實體跌破才調節部位，不是達標就立刻全數出場。")
        else:
            st.info("② 追蹤停利：均線資料不足，暫時無法計算。")

    st.caption("以上為依你提供的持倉規則直接換算的量化結果，屬於規則式計算，不是 AI 生成，"
               "也不構成投資建議；實際執行仍請自行確認部位身份（新加碼／核心持倉）與基本面狀況。")


def render_portfolio_discipline_section(symbol: str, price_df: pd.DataFrame):
    """選填區塊：使用者上傳自己的交易紀錄 CSV，並手動輸入帳戶總資產／現金水位／目標總倉位，
    才會顯示「單一股票最大曝險35%」「現金防禦率15%~20%」「單次加碼上限25%」「分批建倉A/B/C型態」
    「反FOMO回補紀律」這幾項需要知道實際持倉狀態才能判斷的規則（範本第一、二、四節）。

    不上傳交易紀錄的話，這個區塊只顯示說明文字，不影響上方原本的個股技術面分析。
    回傳 portfolio_check dict（供 AI 分析與下載報告引用）；未上傳交易紀錄時回傳 None。
    """
    st.markdown("##### 📊 持倉風控與紀律檢核（選填功能）")
    st.caption(
        f"以下需要你上傳自己的 {symbol} 交易紀錄 CSV，才能算出「單一股票曝險」「加碼比例」"
        "「分批建倉型態」「反FOMO回補」這幾項——這支工具本身不記得你的持倉狀態，"
        "每次分析都需要重新上傳（不會存到雲端）。不上傳的話完全不影響上方的技術面分析。"
    )

    trade_file = st.file_uploader(
        f"上傳 {symbol} 的交易紀錄 CSV（欄位：Date, Type, Symbol, Name, Price, Quantity, Reason）",
        type=["csv"], key="trade_log_uploader",
    )

    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        total_account_assets = st.number_input(
            "帳戶總資產（所有持股＋現金，選填）", min_value=0.0, value=0.0, step=1000.0,
            help="用於「單一股票最大曝險35%」與「現金防禦水位15%~20%」這兩項的計算基準；留空或填0則不計算這兩項。",
        )
    with colp2:
        cash_amount = st.number_input(
            "目前現金／高流動資產水位（選填）", min_value=0.0, value=0.0, step=1000.0,
        )
    with colp3:
        target_total_position = st.number_input(
            f"{symbol} 目標總倉位（股數，選填）", min_value=0.0, value=0.0, step=1.0,
            help="用於「單次加碼上限不得超過目標總倉位25%」的計算基準；留空或填0則不計算這一項。",
        )

    if trade_file is None:
        st.info("尚未上傳交易紀錄，以上四個欄位可先不用理會。")
        return None

    try:
        raw = pd.read_csv(trade_file)
        trades, warns = parse_trade_log(raw)
    except Exception as exc:
        st.error(f"交易紀錄讀取失敗：{exc}")
        return None

    for w in warns:
        st.warning(w)
    if len(trades) == 0:
        st.error("交易紀錄裡沒有看得懂的買賣紀錄，請確認欄位格式（Date/Type/Price/Quantity 是否正確）。")
        return None

    position = compute_position_from_trades(trades)
    latest_close = float(price_df["close"].iloc[-1])
    exposure_check = check_portfolio_exposure(
        position["current_shares"], latest_close, total_account_assets, cash_amount
    )
    pyramid_buys = classify_pyramid_buys(position["trade_rows"], price_df, target_total_position)
    fomo_flags = check_anti_fomo_buybacks(position["trade_rows"])

    st.markdown("**目前持股（依交易紀錄推算，移動平均成本法）**")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("目前股數", f"{position['current_shares']:.0f}")
    pc2.metric("平均成本", f"${position['avg_cost']:.2f}")
    pc3.metric("持股市值（依最新收盤價）", f"${exposure_check['position_value']:,.0f}")
    pc4.metric("已實現損益", f"${position['realized_pnl']:+,.0f}")

    st.markdown("**① 單一股票曝險 ／ 現金防禦水位**")
    ec1, ec2 = st.columns(2)
    with ec1:
        if exposure_check["exposure_pct"] is not None:
            over = exposure_check["exposure_over_cap"]
            st.metric("佔帳戶總資產比重（上限35%）", f"{exposure_check['exposure_pct']:.1%}")
            if over:
                st.warning("已超過35%上限。若是股價上漲導致市值變大，依你的規則「僅以追蹤停利調節，不主動猜頂」，"
                           "不代表要馬上出清；若是加碼買進造成的，則要留意是否過度集中。")
            else:
                st.success("在35%上限之內。")
        else:
            st.info("未提供帳戶總資產，無法計算單一股票曝險比重。")
    with ec2:
        if exposure_check["cash_pct"] is not None:
            st.metric("現金佔帳戶總資產比重（常態目標15%~20%）", f"{exposure_check['cash_pct']:.1%}")
            if exposure_check["cash_below_min"]:
                st.warning("低於15%下限，可留意的防禦資金可能偏少。")
            elif exposure_check["cash_above_max"]:
                st.info("高於20%，防禦資金水位偏寬鬆。")
            else:
                st.success("在15%~20%常態區間內。")
        else:
            st.info("未提供現金水位，無法計算現金防禦率。")

    st.markdown("**② 分批建倉型態比對（依買進當天技術指標，回溯判斷）**")
    if pyramid_buys:
        for p in pyramid_buys:
            over_cap = p.get("addon_over_cap")
            addon_pct = p.get("addon_pct_of_target")
            label = f"{p['date'].date().isoformat()}　買進 {p['quantity']:.0f} 股 @ ${p['price']:.2f}"
            detail = p["classification"]
            if addon_pct is not None:
                detail += f"；佔目標總倉位 {addon_pct:.1%}"
            if over_cap:
                st.warning(f"{label}\n\n{detail}　⚠️超過單次加碼25%上限")
            else:
                st.info(f"{label}\n\n{detail}")
        st.caption("C（財報跳空缺口）型態需要財報公布日期資料，此工具尚未串接財報行事曆，"
                   "無法自動判斷，「未明顯符合A或B」的買進也可能屬於C型態，請自行對照。")
    else:
        st.caption("交易紀錄中沒有買進紀錄可比對。")

    st.markdown("**④ 反FOMO回補檢查（簡化版）**")
    if fomo_flags:
        for f in fomo_flags:
            st.warning(
                f"{f['sell_date'].date().isoformat()} 賣出 @ ${f['sell_price']:.2f} → "
                f"{f['buy_date'].date().isoformat()}（{f['days_between']}天後）用更高價格 "
                f"${f['buy_price']:.2f}（{f['price_diff_pct']:+.1%}）買回，疑似追高回補。"
            )
    else:
        st.success("未偵測到「賣出後短期內用更高價格買回」的紀錄。")
    st.caption("這是簡化版的通用判斷（賣出後3天內、價格更高的回補先一律標記出來），"
               "並非你原本規則裡「允許執行的客觀回補條件」完整表格的精確還原——如果你想要更精準比對，"
               "麻煩把那份表格的實際內容貼給我，我再調整判斷邏輯。")

    return {
        "current_shares": position["current_shares"],
        "avg_cost": position["avg_cost"],
        "realized_pnl": position["realized_pnl"],
        "exposure_check": exposure_check,
        "pyramid_buys": pyramid_buys,
        "fomo_flags": fomo_flags,
    }


def render_stock_section(symbol, color, df, stats, trend, patterns, backtest, stop_plan,
                          quote=None, quote_error=None):
    """單一個股的完整分析區塊：即時報價 → 技術指標圖 → 型態分析 → 訊號回測 → 投資建議與風險評估。"""
    render_realtime_quote(symbol, quote, quote_error)

    st.plotly_chart(chart_technical(df, symbol, color), use_container_width=True)

    st.markdown("##### 🔍 型態分析")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.metric("趨勢方向（近20日回歸斜率）", trend["trend"])
        st.metric("均線排列", stats["ma_alignment"])
        st.metric("支撐 ／ 壓力", f"{_fmt_num(trend['support'])} ／ {_fmt_num(trend['resistance'])}")
        if trend["breakout"]:
            st.warning(trend["breakout"])
        st.caption(f"布林通道：{stats['bb_position']}")
        st.caption(f"OBV（能量潮）：{stats['obv_signal']}")
    with tc2:
        st.markdown("**近期K線型態辨識**")
        if patterns:
            for d, name, meaning in patterns:
                st.markdown(f"- **{d}｜{name}**：{meaning}")
        else:
            st.caption(
                "已檢查最近5天的K線資料，但沒有偵測到本工具支援辨識的型態"
                "（十字星／鎚子線／吊人線／吞噬型態）——這是「有資料、但這幾天"
                "沒出現這幾種型態」，不是資料抓取失敗，屬於正常情況。"
            )
        st.caption("型態辨識為簡化版規則演算法，非專業技術分析軟體等級的完整判斷，僅供輔助參考。")

    st.markdown("##### 🧪 訊號回測（歷史勝率）")
    st.plotly_chart(chart_backtest(backtest, symbol, color), use_container_width=True)
    low_n_signals = [n for n, r in backtest["signals"].items() if r["n"] < 5]
    if low_n_signals:
        st.caption(f"樣本數過少（n<5）的訊號：{'、'.join(low_n_signals)}，其勝率數字不具統計參考性。")
    st.caption("此為根據這段歷史區間回測的結果，反映的是「過去」訊號出現後隔日上漲的機率，"
               "不保證未來會維持相同機率，也不是對下一個交易日的預測。")

    st.markdown("##### 📋 快速操作方向（規則式即時計算，不需呼叫 AI）")
    level, level_kind = assess_risk_level(stats)
    action, reasons, _ = build_quick_recommendation(stats)

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        if level_kind == "low":
            st.success(f"波動風險等級：{level}")
        elif level_kind == "mid":
            st.warning(f"波動風險等級：{level}")
        else:
            st.error(f"波動風險等級：{level}")
        st.caption("依年化波動度與最大回撤估算，僅描述股價波動大小，跟下方停損／停利機制是各自"
                   "獨立的計算，並非用來決定停損／停利百分比。")
    with rc2:
        st.markdown(f"**快速操作方向：{action}**")
        for r in reasons:
            st.markdown(f"- {r}")

    render_playbook_stop_plan(stop_plan)


def build_markdown_report(symbol, stats, start_date, end_date, ai_analysis: str,
                           trend=None, patterns=None, backtest=None, quote=None, stop_plan=None,
                           portfolio_check=None) -> str:
    trend = trend or {}
    patterns = patterns or []

    level, level_kind = assess_risk_level(stats)
    action, reasons, _ = build_quick_recommendation(stats)
    stop_plan_block = _fmt_stop_plan_lines(stop_plan) if stop_plan else "（本次未提供停損／停利機制資料）"
    portfolio_block = (
        _fmt_portfolio_check_lines(portfolio_check) if portfolio_check
        else "（本次未上傳交易紀錄，無持倉風控與紀律檢核資料）"
    )

    lines = [
        f"# 股票分析評估報告：{symbol}",
        f"分析區間：{start_date.isoformat()} ~ {end_date.isoformat()}",
        f"歷史技術指標資料更新至：{stats['latest_date'].date().isoformat()}；即時報價：{_fmt_quote(quote)}\n",
        f"## 快速風險評估與操作方向（規則式計算）\n",
        f"- 波動風險等級：{level}；快速操作方向：{action}",
        "".join(f"\n  - {r}" for r in reasons),
        f"\n\n## 停損／停利機制（依你的持倉規則計算）\n",
        stop_plan_block,
        f"\n\n## 持倉風控與紀律檢核（依你的交易紀錄計算）\n",
        portfolio_block,
        f"\n\n## 趨勢與型態\n",
        f"- 趨勢：{trend.get('trend', 'N/A')}；均線排列：{stats['ma_alignment']}；"
        f"布林通道：{stats['bb_position']}；KD：{stats['kd_cross']}（{stats['kd_signal']}）"
        + (f"；{trend['breakout']}" if trend.get('breakout') else ""),
        f"- OBV（能量潮）：{stats['obv_signal']}",
        f"- 近期K線型態：{_fmt_patterns(patterns)}\n",
        f"## 訊號回測（隔日上漲機率）\n",
        _fmt_backtest(backtest),
        f"\n\n## 統計摘要\n",
        f"- 期間總報酬率：{_fmt_pct(stats['total_return'])}",
        f"- 年化報酬率：{_fmt_pct(stats['ann_return'])}",
        f"- 年化波動度：{_fmt_pct(stats['ann_vol'])}",
        f"- Sharpe Ratio：{_fmt_num(stats['sharpe'])}",
        f"- 最大回撤：{_fmt_pct(stats['max_drawdown'])}",
        f"\n## Gemini AI 分析\n",
        ai_analysis,
    ]
    return "\n".join(lines)


def _secret_default(key: str) -> str:
    """部署到雲端後，若有在 Streamlit 的 Secrets 裡先設定好金鑰，就直接拿來
    當輸入框的預設值，不用每次重新貼一次；本機測試若沒有設定對應的 secret，
    則預設空白，維持原本手動輸入的操作方式，兩種情況都還是可以在畫面上
    手動覆蓋這個預設值。"""
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def check_password() -> bool:
    """簡單的密碼保護：這支工具會用掉你自己申請的 FMP / Gemini / Alpha Vantage
    API 額度，部署成公開網址後，如果網址不小心外流，陌生人也能打開來用，
    會白白消耗你的額度。這裡在 Streamlit 的 Secrets 設定一組 APP_PASSWORD，
    使用者要先輸入正確密碼才能看到裡面的分析功能。

    本機測試（沒有設定 APP_PASSWORD 這個 secret）時，直接放行不擋密碼，
    維持原本開發測試的方便性。
    """
    try:
        correct_password = st.secrets["APP_PASSWORD"]
    except Exception:
        return True

    if st.session_state.get("password_correct"):
        return True

    st.set_page_config(page_title="股票分析評估表", layout="centered", page_icon="🔒")
    st.markdown("### 🔒 這是私人工具，請輸入密碼")
    pw = st.text_input("密碼", type="password", label_visibility="collapsed",
                        placeholder="請輸入密碼")
    if st.button("進入", type="primary"):
        if pw == correct_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("密碼錯誤，請再試一次。")
    return False


def run_us_app(shared_gemini_key: str = ""):
    """美股模式主流程：單一美股個股的完整技術分析（FMP股價 + Gemini AI）。"""
    st.caption("以單一美股個股為分析對象：即時報價、技術指標、型態分析、訊號歷史回測、"
               "規則式風險評估與 Gemini AI 深度分析。")

    with st.sidebar:
        st.header("🔑 API 金鑰設定")
        st.caption("已在部署設定裡填好金鑰的話，下面會自動帶出來，不用每次重打；"
                   "要換一組金鑰時，直接在這裡覆蓋掉就可以。")
        fmp_key = st.text_input("FMP API 金鑰", type="password", value=_secret_default("FMP_API_KEY"),
                                 help="至 https://site.financialmodelingprep.com/ 註冊取得")
        gemini_key = shared_gemini_key
        st.caption("Gemini API 金鑰已統一在首頁上方輸入，美股、台股分析共用同一組金鑰，這裡不用再輸入一次。")
        gemini_model = st.text_input("Gemini 模型名稱", value="gemini-3.5-flash-lite",
                                      help="若出現 404 model not found，請至官方文件確認目前可用模型名稱")

        st.markdown("---")
        st.subheader("📰 新聞情緒分析設定（選用）")
        enable_news = st.checkbox("啟用新聞情緒分析（Alpha Vantage + Fear & Greed Index）", value=True)
        alpha_vantage_key = st.text_input(
            "Alpha Vantage API 金鑰", type="password", value=_secret_default("ALPHA_VANTAGE_API_KEY"),
            disabled=not enable_news,
            help="至 https://www.alphavantage.co/support/#api-key 免費申請",
        )
        news_limit = st.slider("擷取新聞則數", min_value=10, max_value=200, value=50, step=10,
                                disabled=not enable_news)
        st.caption("Alpha Vantage 免費版每日限 25 次 API 請求，請節制重新分析的次數；"
                   "CNN Fear & Greed Index 串接的是非官方資料端點，可能隨時失效，失效時"
                   "不影響其餘分析功能。")

        st.markdown("---")
        st.caption("金鑰僅保存於本次瀏覽器 session 記憶體中，不會寫入檔案或上傳。")
        st.markdown("### 📢 免責聲明")
        st.caption(
            "本系統（含技術指標、規則式建議、新聞情緒分析與 Gemini AI 分析）僅供學術研究與教育用途，"
            "所有分析結果僅供參考，**不構成投資建議或財務建議**。新聞情緒與 Fear & Greed Index 反映"
            "的是歷史市場觀點，不代表未來股價走勢。請使用者自行判斷投資決策並承擔相關風險，本系統"
            "作者不對任何投資行為負責，亦不承擔任何損失責任。"
        )

    st.markdown(
        """
        <div style="background-color:#eff6ff;border:2px solid #2563eb;border-radius:10px;
                    padding:16px 20px;margin-top:8px;margin-bottom:14px;">
            <div style="font-size:1.1rem;font-weight:700;color:#1e3a8a;margin-bottom:4px;">
                🔍 請輸入欲分析的美股股票代號
            </div>
            <div style="font-size:0.88rem;color:#334155;">
                例如：AAPL（蘋果）、NVDA（輝達）、TSLA（特斯拉）。輸入後按下方「開始分析」按鈕即可。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    input_col, target_col, _spacer_col = st.columns([1, 1, 1])
    with input_col:
        symbol = st.text_input(
            "股票代號", value="NVDA", placeholder="輸入股票代號，例如 NVDA",
            label_visibility="collapsed",
        )
    with target_col:
        custom_target_price = st.number_input(
            "🎯 第一階段停利目標價（選填）", min_value=0.0, value=0.0, step=1.0, format="%.2f",
            help="對應你的持倉規則裡「階段目標價（如前高 $230~$235）」。留空或填 0，"
                 "則自動以下方型態分析算出的「近期壓力價」代入。",
        )
    st.caption(f"分析區間會自動抓取最近 {HISTORY_LOOKBACK_DAYS} 天的歷史資料，結束日期以下方"
               "「⚡ 即時報價」實際取得的日期為準（若無法取得即時報價，則以今天的日期為準），"
               "不需要另外選擇開始／結束日期。")

    run = st.button("🚀 開始分析", type="primary", use_container_width=False)

    if not run:
        st.info("填好左側 API 金鑰與上方股票代號後，按下「開始分析」即可。")
        return

    if not symbol.strip():
        st.error("請輸入股票代號。")
        return
    if not fmp_key:
        st.error("請先在左側輸入 FMP API 金鑰。")
        return

    symbol = symbol.strip().upper()

    with st.spinner(f"正在向 FMP 取得 {symbol} 的即時報價..."):
        try:
            quote, quote_err = fetch_fmp_quote(symbol, fmp_key), None
        except Exception as exc:
            quote, quote_err = None, str(exc)

    end_date = quote["updated_at"].date() if (quote and quote.get("updated_at")) else date.today()
    start_date = end_date - timedelta(days=HISTORY_LOOKBACK_DAYS)

    with st.spinner(f"正在向 FMP 取得 {symbol} 的歷史股價..."):
        try:
            df_raw = fetch_fmp_history(symbol, start_date, end_date, fmp_key)
        except Exception as exc:
            st.error(f"取得股價資料失敗：{exc}")
            return

    df = add_indicators(df_raw)
    stats = compute_stats(df)
    trend = detect_trend_pattern(df)
    patterns = detect_candlestick_patterns(df)
    backtest = backtest_signals(df)
    stop_plan = compute_playbook_stop_plan(df, trend, custom_target_price=custom_target_price)

    date_str = stats["latest_date"].date().isoformat()
    st.info(f"📅 分析區間：{start_date.isoformat()} ~ {end_date.isoformat()}"
            f"（結束日期取自即時報價日期，往回抓 {HISTORY_LOOKBACK_DAYS} 天歷史資料）；"
            f"歷史技術指標資料實際更新至：{date_str}。"
            "RSI/MACD/KD/均線/OBV 等技術指標都是根據這份歷史資料算出來的，本身無法做成即時更新；"
            "下方的「⚡ 即時報價」則是另外呼叫即時報價 API 取得的目前價格，可用來對照現在的價格與"
            "技術指標所依據的歷史資料是否一致。")
    st.success("分析完成！以下由上到下依序顯示：績效摘要 → 即時報價／技術指標／型態分析／"
               "訊號回測／投資建議 → AI 分析，往下捲動即可看到全部內容。")

    st.markdown("---")
    st.header(f"🔎 {symbol} 個股完整分析")
    st.markdown("##### 📊 績效摘要")
    render_performance_summary(stats)
    render_stock_section(symbol, STOCK_COLOR, df, stats, trend, patterns, backtest, stop_plan,
                          quote=quote, quote_error=quote_err)

    st.markdown("---")
    portfolio_check = render_portfolio_discipline_section(symbol, df)

    st.markdown("---")
    st.header("🤖 AI 分析")
    if not gemini_key:
        st.warning("請先在左側輸入 Gemini API 金鑰才能產生 AI 分析。")
    else:
        with st.spinner("正在請 Gemini 進行分析，請稍候..."):
            prompt = build_prompt(symbol, stats, start_date, end_date,
                                   trend=trend, patterns=patterns, backtest=backtest, quote=quote,
                                   stop_plan=stop_plan, portfolio_check=portfolio_check)
            try:
                analysis = call_gemini(prompt, gemini_key, gemini_model)
            except Exception as exc:
                st.error(str(exc))
                analysis = None

        if analysis:
            st.markdown(analysis, unsafe_allow_html=True)
            report_md = build_markdown_report(symbol, stats, start_date, end_date, analysis,
                                               trend=trend, patterns=patterns, backtest=backtest,
                                               quote=quote, stop_plan=stop_plan, portfolio_check=portfolio_check)
            st.download_button(
                "📥 下載完整分析報告（Markdown）",
                data=report_md.encode("utf-8"),
                file_name=f"{symbol}_分析報告.md",
                mime="text/markdown",
            )
            with st.expander("查看送給 Gemini 的完整提示語（prompt）"):
                st.code(prompt, language="text")

    st.markdown("---")
    st.header("📰 新聞情緒分析")
    st.caption("以下結合 Alpha Vantage 新聞情緒資料與 CNN Fear & Greed Index，"
               "從「新聞與市場心理」的角度補充上方的技術面分析。")

    if not enable_news:
        st.info("新聞情緒分析功能未啟用（可在左側勾選「啟用新聞情緒分析」）。")
    elif not alpha_vantage_key:
        st.warning("請先在左側輸入 Alpha Vantage API 金鑰，才能取得新聞情緒資料。")
    else:
        st.markdown("##### 😨😊 CNN Fear & Greed Index（恐懼與貪婪指數）")
        try:
            fg = fetch_fear_greed_index()
        except Exception as exc:
            fg = None
            st.warning(f"目前無法取得 CNN Fear & Greed Index：{exc}\n\n"
                       "（此為非官方資料端點，CNN 隨時可能調整或封鎖，不影響其他分析功能。）")

        if fg:
            st.plotly_chart(chart_fear_greed_gauge(fg), use_container_width=True)
            fgc1, fgc2, fgc3 = st.columns(3)
            fgc1.metric("前一交易日", _fmt_num(fg["previous_close"], 0))
            fgc2.metric("一週前", _fmt_num(fg["previous_1_week"], 0))
            fgc3.metric("一個月前", _fmt_num(fg["previous_1_month"], 0))
            st.caption("資料來源：CNN Business（非官方端點，僅供參考）。指數範圍 0（極度恐懼）～ "
                       "100（極度貪婪），反映整體美股市場情緒，並非個股專屬指標。")
            st.caption(
                "⚠️ 這個指數跟下方「新聞情緒統計」是兩組互相獨立的資料，衡量的東西本來就不同："
                "F&G 綜合的是大盤層級的技術面訊號（如市場動能、避險資金流向、選擇權評價等），"
                "下方新聞情緒則是單獨統計這支股票的新聞文字語氣。兩者方向不一致（例如大盤恐懼、"
                "但個股新聞中性偏多）是常見且合理的現象，不代表其中一邊算錯，可以把兩者當成"
                "「大盤氣氛」與「個股消息面」兩個互補的觀察角度來看。"
            )

        st.markdown("##### 📊 新聞情緒統計")
        with st.spinner(f"正在向 Alpha Vantage 取得 {symbol} 的新聞情緒資料..."):
            try:
                news_list = fetch_alpha_vantage_news(symbol, alpha_vantage_key, news_limit)
                news_err = None
            except Exception as exc:
                news_list, news_err = [], str(exc)

        if news_err:
            st.error(f"取得新聞情緒資料失敗：{news_err}")
        elif not news_list:
            st.info("目前沒有可分析的新聞資料。")
        else:
            news_stats = compute_news_sentiment_stats(news_list)

            nc1, nc2, nc3, nc4 = st.columns(4)
            nc1.metric("總新聞數量", news_stats["n_news"])
            if news_stats["n_news"] != news_limit:
                # 這裡的則數是 Alpha Vantage 實際回傳的則數，不一定會等於左側滑桿設定的
                # 上限（news_limit）：滑桿只是「最多抓幾則」的上限，若該股票在查詢範圍內
                # 實際能匹配到的相關新聞本來就比較少，回傳則數就會低於設定值，這是正常情況、
                # 不是程式沒有把滑桿數值送出去（實際送給API的請求上限就是 news_limit）。
                nc1.caption(f"（已設定上限 {news_limit} 則，Alpha Vantage 實際回傳 {news_stats['n_news']} 則）")
            nc2.metric("平均情緒評分", _fmt_num(news_stats["avg_article_score"], 3))
            nc3.metric("平均相關性評分", _fmt_num(news_stats["avg_relevance"], 3))
            with nc4:
                # 用自訂樣式取代 st.metric：st.metric 的數值字級較大，
                # 「Somewhat-Bullish」這類較長的標籤在四欄窄版面下容易被裁切看不全，
                # 這裡改用較小字級並允許換行，確保完整顯示。
                st.markdown(
                    f"""
                    <div style="display:flex;flex-direction:column;gap:2px;">
                        <div style="font-size:0.8rem;color:#6b7280;">主要情緒傾向</div>
                        <div style="font-size:1.1rem;font-weight:600;color:#111827;
                                    line-height:1.3;overflow-wrap:break-word;">
                            {_sentiment_label_emoji(news_stats['dominant_ticker_label'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.plotly_chart(chart_sentiment_pie(news_stats, symbol), use_container_width=True)
            st.caption(
                f"左圖只計算新聞中「明確標記與 {symbol} 相關」的情緒判斷；右圖則是整篇文章的情緒，"
                "涵蓋文章討論的所有主題、不限於這支股票，兩者統計範圍不同，比例與顯示出來的顏色"
                "種類也可能不同（某類情緒則數為 0 時，該顏色的扇形就不會出現），這是資料本身的"
                "差異，配色定義本身兩張圖是一致的。"
            )

            with st.expander("ℹ️ 情緒分數／相關性分數說明"):
                st.markdown(
                    "**情緒分數**範圍 -1（極度看空）～ +1（極度看多）：\n"
                    "- 😊 Bullish（看漲）：分數 ≥ 0.35\n"
                    "- 🙂 Somewhat-Bullish（偏看漲）：0.15 ～ 0.35\n"
                    "- 😐 Neutral（中性）：-0.15 ～ 0.15\n"
                    "- 🙁 Somewhat-Bearish（偏看跌）：-0.35 ～ -0.15\n"
                    "- 😢 Bearish（看跌）：分數 ≤ -0.35\n\n"
                    "**相關性分數**範圍 0（無關）～ 1（高度相關）：\n"
                    "- 高度相關：≥ 0.8　- 相關：0.5 ～ 0.8　- 中度相關：0.3 ～ 0.5　- 低度相關：< 0.3"
                )

            ntc1, ntc2 = st.columns(2)
            with ntc1:
                st.plotly_chart(chart_sentiment_time_series(news_list, symbol), use_container_width=True)
            with ntc2:
                st.plotly_chart(chart_source_distribution(news_stats), use_container_width=True)

            st.markdown("##### 📰 相關新聞列表（依相關性排序，取前10則）")
            top_news = sorted(news_list, key=lambda x: x["ticker_relevance"] or 0, reverse=True)[:10]
            for i, item in enumerate(top_news, 1):
                with st.expander(f"{i}. 📰 {item['title']} ｜ 🎯 {_sentiment_label_emoji(item['ticker_sentiment_label'])}"):
                    ec1, ec2, ec3 = st.columns(3)
                    with ec1:
                        pub = item["published_at"].strftime("%Y-%m-%d %H:%M") if item["published_at"] else "N/A"
                        st.markdown(f"**發布時間**：{pub}")
                        st.markdown(f"**股票情緒**：{_sentiment_label_emoji(item['ticker_sentiment_label'])}"
                                    f"（{_fmt_num(item['ticker_sentiment_score'], 3)}）")
                        st.markdown(f"**文章情緒**：{_sentiment_label_emoji(item['overall_sentiment_label'])}"
                                    f"（{_fmt_num(item['overall_sentiment_score'], 3)}）")
                        st.markdown(f"**相關性**：{_fmt_num(item['ticker_relevance'], 3)}")
                    with ec2:
                        st.markdown("**相關主題**")
                        topics = sorted(
                            item.get("topics", []),
                            key=lambda t: _safe_float(t.get("relevance_score")) or 0, reverse=True,
                        )[:3]
                        if topics:
                            for t in topics:
                                st.markdown(f"- {t.get('topic', 'N/A')}"
                                            f"（權重 {_fmt_num(_safe_float(t.get('relevance_score')), 3)}）")
                        else:
                            st.caption("無主題資料")
                    with ec3:
                        summary = item["summary"] or ""
                        st.markdown(f"**摘要**：{summary[:200]}{'...' if len(summary) > 200 else ''}")
                        if item["url"]:
                            st.markdown(f"[閱讀原文]({item['url']})")

            st.markdown("##### 🧠 市場情緒總結（Gemini AI）")
            if not gemini_key:
                st.warning("請先在左側輸入 Gemini API 金鑰才能產生市場情緒總結。")
            else:
                with st.spinner("正在請 Gemini 分析市場情緒..."):
                    try:
                        news_summary = call_gemini(
                            build_news_summary_prompt(symbol, news_stats, news_list, fg),
                            gemini_key, gemini_model,
                        )
                    except Exception as exc:
                        news_summary = None
                        st.error(str(exc))
                if news_summary:
                    st.markdown(news_summary)

            st.markdown("##### 🤖 新聞情緒 AI 深度分析（Gemini AI）")
            if not gemini_key:
                st.warning("請先在左側輸入 Gemini API 金鑰才能產生新聞情緒深度分析。")
            else:
                with st.spinner("正在請 Gemini 產生新聞情緒深度分析報告..."):
                    news_ai_prompt = build_news_ai_prompt(symbol, stats, news_stats, news_list, fg)
                    try:
                        news_analysis = call_gemini(news_ai_prompt, gemini_key, gemini_model)
                    except Exception as exc:
                        news_analysis = None
                        st.error(str(exc))
                if news_analysis:
                    st.markdown(news_analysis)
                    with st.expander("查看送給 Gemini 的新聞情緒分析提示語（prompt）"):
                        st.code(news_ai_prompt, language="text")

        st.caption("📢 免責聲明：新聞情緒分析與 Fear & Greed Index 僅反映歷史新聞與市場情緒統計，"
                   "不構成投資建議或未來走勢預測，請自行判斷並承擔投資風險。")


# ----------------------------------------------------------------------------
# 6. 台股模式（Yahoo Finance／yfinance，給完全新手用，不需自己申請金鑰）
#    以下函式搬自 taiwan_stock_pattern_app.py，僅重新命名 main()→run_taiwan_app()，
#    並移除其中重複的 st.set_page_config()（改由最外層共用的 main() 統一呼叫一次）。
#    detect_trend_pattern／detect_candlestick_patterns／_fmt_patterns／_fmt_num／
#    call_gemini／STOCK_COLOR／GEMINI_URL_TMPL 皆與美股模式共用同一份，不重複定義。
# ----------------------------------------------------------------------------

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 在某些版本／某些查詢方式下，會回傳多層欄位（MultiIndex），
    這裡統一攤平成單層，避免後面存取 df['Close'] 時出錯。"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_tw_history(symbol_input: str, lookback_days: int = 365):
    """抓取台股歷史日線資料（open/high/low/close/volume）。

    使用者只要輸入數字代號（例如 2330），這裡會自動依序嘗試：
        1. 使用者輸入的原始字串（若已經自己打了 .TW / .TWO 等後綴）
        2. 補上 .TW（上市股票，例如 2330.TW = 台積電）
        3. 補上 .TWO（上櫃股票）
    第一個抓得到資料的就採用，讓完全不懂上市/上櫃差異的人也能直接用。

    回傳 (df, 實際採用的代號)；抓不到資料時丟出例外，訊息用中文說明。
    """
    raw = symbol_input.strip().upper()
    if not raw:
        raise ValueError("請輸入股票代號。")

    candidates = [raw] if "." in raw else [f"{raw}.TW", f"{raw}.TWO"]

    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    last_err = None
    for candidate in candidates:
        try:
            df = yf.download(
                candidate,
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - 對外都轉成中文訊息
            last_err = exc
            continue

        if df is None or df.empty:
            continue

        df = _flatten_yf_columns(df)
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep_cols].dropna(subset=["close"])
        df = df.reset_index().rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if len(df) >= 30:
            return df, candidate

    raise ValueError(
        f"查不到股票代號「{raw}」的資料。已經自動嘗試過"
        f"{'、'.join(candidates)}，請確認代號是否正確"
        "（台股代號通常是4位數字，例如台積電是2330）。"
        + (f" 詳細錯誤：{last_err}" if last_err else "")
    )


def fetch_tw_realtime_quote(resolved_symbol: str):
    """嘗試取得目前即時（或最近）報價；失敗時回傳 None，不中斷整體分析。"""
    try:
        ticker = yf.Ticker(resolved_symbol)
        fast_info = getattr(ticker, "fast_info", None)
        if not fast_info:
            return None
        price = fast_info.get("last_price")
        prev_close = fast_info.get("previous_close")
        day_low = fast_info.get("day_low")
        day_high = fast_info.get("day_high")
        volume = fast_info.get("last_volume")
        if price is None:
            return None
        change = (price - prev_close) if prev_close else None
        change_pct = (change / prev_close * 100) if (change is not None and prev_close) else None
        return {
            "price": price, "previous_close": prev_close,
            "change": change, "change_pct": change_pct,
            "day_low": day_low, "day_high": day_high, "volume": volume,
        }
    except Exception:
        return None


def fetch_index_history(lookback_days: int = 365):
    """抓取台股加權指數（^TWII，也就是新聞常講的「大盤」）歷史資料，
    用來跟個股做「相對強弱」比較。抓不到時回傳 None，不影響其他分析。"""
    try:
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        df = yf.download(
            "^TWII",
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )
        if df is None or df.empty:
            return None
        df = _flatten_yf_columns(df)
        df = df.rename(columns={"Close": "close"})
        df = df.reset_index().rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def get_tw_company_info(resolved_symbol: str) -> dict:
    """嘗試取得公司基本資料（產業別、本益比、殖利率、市值），讓新手對這家
    公司「在幹嘛」有基本概念，不只是看線圖。yfinance的公司資料有時抓不到
    或抓得比較慢，失敗時回傳空 dict，不影響其他分析結果。"""
    try:
        ticker = yf.Ticker(resolved_symbol)
        info = ticker.get_info() if hasattr(ticker, "get_info") else ticker.info
        if not info:
            return {}
        dividend_yield = info.get("dividendYield")
        if dividend_yield is not None and dividend_yield < 1:
            dividend_yield *= 100  # 有些版本回傳的是小數（0.02），統一換成百分比
        return {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "pe": info.get("trailingPE"),
            "dividend_yield": dividend_yield,
            "market_cap": info.get("marketCap"),
        }
    except Exception:
        return {}


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma5"] = df["close"].rolling(5, min_periods=1).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    df["ma60"] = df["close"].rolling(60, min_periods=1).mean()
    return df


def compute_52week_stats(df: pd.DataFrame, latest_close: float) -> dict:
    """用抓到的歷史資料（最長約一年）估算52週高低點，以及目前價位落在
    這段區間的百分比位置（0%＝貼近最低點，100%＝貼近最高點）。"""
    high_52w = df["high"].max()
    low_52w = df["low"].min()
    rng = (high_52w - low_52w) if high_52w != low_52w else 1e-9
    position_pct = (latest_close - low_52w) / rng * 100
    return {"high": high_52w, "low": low_52w, "position_pct": position_pct}


def compute_relative_strength(df: pd.DataFrame, index_df, window: int = 60) -> dict:
    """比較個股與大盤（加權指數）在近 window 個交易日的漲跌幅，
    判斷這支股票最近是比大盤強還是比大盤弱。任一邊資料不足時回傳 None，
    不影響其他分析。"""
    if index_df is None or len(df) < 2:
        return None

    stock_recent = df.tail(min(window, len(df)))
    if stock_recent["close"].iloc[0] == 0:
        return None
    stock_return = (stock_recent["close"].iloc[-1] / stock_recent["close"].iloc[0] - 1) * 100

    idx_recent = index_df.tail(min(window, len(index_df)))
    if len(idx_recent) < 2 or idx_recent["close"].iloc[0] == 0:
        return None
    index_return = (idx_recent["close"].iloc[-1] / idx_recent["close"].iloc[0] - 1) * 100

    return {
        "days": len(stock_recent),
        "stock_return": stock_return,
        "index_return": index_return,
        "diff": stock_return - index_return,
    }


def compute_outlook(trend: dict, patterns: list, relative_strength) -> dict:
    """把趨勢方向、突破/跌破、K線型態、與大盤的相對強弱，綜合整理成一個
    給新手看的「偏多／中性／偏空」燈號，方便一眼掌握大方向，細節再往下看
    各項分析。這只是把上面已經算出的訊號做簡單加總，不是額外的獨立指標，
    也不構成投資建議。"""
    score = 0.0
    if trend.get("trend") == "上升趨勢":
        score += 1
    elif trend.get("trend") == "下降趨勢":
        score -= 1

    breakout = trend.get("breakout") or ""
    if "向上突破" in breakout:
        score += 1
    elif "向下跌破" in breakout:
        score -= 1

    for _, name, _ in patterns:
        if "看漲" in name or "鎚子線" in name:
            score += 0.5
        elif "看跌" in name or "吊人線" in name:
            score -= 0.5

    if relative_strength is not None:
        if relative_strength["diff"] > 1:
            score += 0.5
        elif relative_strength["diff"] < -1:
            score -= 0.5

    # 台股慣例：紅色代表漲／偏多，綠色代表跌／偏空，跟K線圖顏色保持一致。
    if score >= 1.5:
        return {"label": "偏多", "emoji": "🔴", "color": "#dc2626",
                "detail": "多項訊號偏向多方，但這只是歷史資料的統計整理，僅供學習參考，不是投資建議。"}
    if score <= -1.5:
        return {"label": "偏空", "emoji": "🟢", "color": "#16a34a",
                "detail": "多項訊號偏向空方，但這只是歷史資料的統計整理，僅供學習參考，不是投資建議。"}
    return {"label": "中性 ／ 觀望", "emoji": "⚪", "color": "#6b7280",
            "detail": "目前多空訊號不明顯，屬於觀望階段，可以持續留意後續變化。"}


def chart_tw_pattern(df: pd.DataFrame, symbol: str, trend: dict):
    """台股K線圖：紅漲綠跌（台股慣例，與美股模式的技術指標圖顏色邏輯相反，
    因此獨立成自己的繪圖函式，不與美股模式的 chart_technical() 共用）。"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
    )

    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="K線",
        increasing=dict(line=dict(color="#dc2626")),
        decreasing=dict(line=dict(color="#16a34a")),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma5"], name="MA5",
                              line=dict(color="#eab308", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA20",
                              line=dict(color="#0ea5e9", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma60"], name="MA60",
                              line=dict(color="#7c3aed", width=1.2)), row=1, col=1)

    if trend.get("resistance") is not None and not pd.isna(trend["resistance"]):
        fig.add_hline(y=trend["resistance"], line_dash="dot", line_color="#dc2626",
                       annotation_text=f"近期壓力 {trend['resistance']:.2f}", row=1, col=1)
    if trend.get("support") is not None and not pd.isna(trend["support"]):
        fig.add_hline(y=trend["support"], line_dash="dot", line_color="#16a34a",
                       annotation_text=f"近期支撐 {trend['support']:.2f}", row=1, col=1)

    if "volume" in df.columns:
        vol_colors = ["#dc2626" if c >= o else "#16a34a"
                      for o, c in zip(df["open"], df["close"])]
        fig.add_trace(go.Bar(
            x=df["date"], y=df["volume"], name="成交量",
            marker_color=vol_colors, showlegend=False,
        ), row=2, col=1)

    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
    fig.update_yaxes(title_text="價格（新台幣）", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_layout(
        title=f"{symbol}：K線、均線、成交量與支撐壓力",
        template="plotly_white", height=680, showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def build_tw_beginner_prompt(symbol: str, resolved_symbol: str, latest_close: float,
                              latest_date: str, trend: dict, patterns: list, quote=None,
                              stats52=None, relative_strength=None, company_info=None) -> str:
    quote_text = "（本次未取得即時報價，以下以最新收盤價為準）"
    if quote and quote.get("price") is not None:
        chg = quote.get("change")
        chg_pct = quote.get("change_pct")
        chg_text = f"，較前一交易日{'上漲' if (chg or 0) >= 0 else '下跌'} {abs(chg):.2f} 元（{abs(chg_pct):.2f}%）" \
            if chg is not None and chg_pct is not None else ""
        quote_text = f"目前價格 {quote['price']:.2f} 元{chg_text}"

    extra_lines = []
    if stats52:
        extra_lines.append(
            f"- 近一年（約52週）最高 {stats52['high']:.2f} 元／最低 {stats52['low']:.2f} 元，"
            f"目前價位落在這段區間的 {stats52['position_pct']:.0f}% 位置"
            "（0%接近最低點，100%接近最高點）"
        )
    if relative_strength:
        extra_lines.append(
            f"- 近{relative_strength['days']}個交易日：這支股票漲跌 {relative_strength['stock_return']:+.1f}%，"
            f"同期大盤（加權指數）漲跌 {relative_strength['index_return']:+.1f}%，"
            f"相對大盤{'強勢' if relative_strength['diff'] > 0 else '弱勢' if relative_strength['diff'] < 0 else '持平'}"
        )
    if company_info and (company_info.get("industry") or company_info.get("sector")):
        info_bits = []
        if company_info.get("industry"):
            info_bits.append(f"產業別：{company_info['industry']}")
        if company_info.get("pe"):
            info_bits.append(f"本益比：{company_info['pe']:.1f}")
        if company_info.get("dividend_yield"):
            info_bits.append(f"殖利率：{company_info['dividend_yield']:.1f}%")
        if info_bits:
            extra_lines.append(f"- 公司基本資料：{'／'.join(info_bits)}")
    extra_block = ("\n".join(extra_lines) + "\n") if extra_lines else ""

    data_block = f"""
【{symbol}（實際查詢代號：{resolved_symbol}）資料，更新至 {latest_date}】
- {quote_text}
- 最新收盤價：{latest_close:.2f} 元
- 近期趨勢：{trend.get('trend', 'N/A')}
- 近期支撐價：{trend.get('support'):.2f} 元／近期壓力價：{trend.get('resistance'):.2f} 元
- 是否剛突破或跌破：{trend.get('breakout') or '沒有'}
- 最近5天K線型態：{_fmt_patterns(patterns)}
{extra_block}"""

    prompt = f"""你是一位很會教學、有耐心的股市入門老師，現在要跟一位「完全沒學過技術分析」的
初學者解釋這支股票最近的走勢型態。這位讀者不知道什麼是支撐、壓力、K線型態，請你完全
用生活化的白話文解釋，禁止只丟術語不解釋。

{data_block}

請用繁體中文，依照以下架構回答（使用 Markdown 標題，勿使用表格，語氣像在跟朋友聊天
但保持專業正確，不要浮誇）：

## 1. 現在股價在做什麼（白話說明）
用一般人聽得懂的方式，解釋目前是漲勢、跌勢還是盤整，並簡單解釋「支撐」跟「壓力」
分別是什麼意思（例如：支撐可以想像成地板，壓力可以想像成天花板），再套用到這支股票
目前的實際價位上。

## 2. 最近K線型態代表什麼
如果上面有偵測到K線型態，請用白話解釋這個型態通常代表市場心理上發生了什麼事
（例如十字星代表多空雙方勢均力敵，猶豫不決）；如果沒有偵測到型態，就直接說明
「這幾天沒有出現特別值得注意的型態，屬於正常情況」，不要硬掰一個出來。

## 3. 新手可以怎麼解讀（僅供學習參考）
用「如果...歷史上這類情況通常...」的方式舉例說明，避免使用「建議買進/賣出」這類
明確操作指令。務必提醒：這只是根據過去股價統計出的型態參考，不是對未來漲跌的保證，
也不構成投資建議。

## 4. 給新手的小提醒
用1-2句話提醒新手技術分析的限制（例如：型態分析是「落後」於已發生的價格，看到型態
時價格可能已經反應一部分了），並鼓勵他如果想認真投資，除了型態，也要多了解基本面
與風險控管，不要只看單一圖表就做決定。

請全程使用簡單易懂的白話文，避免使用讀者可能看不懂的專有名詞；若必須使用專有名詞
（例如支撐、壓力、突破），第一次出現時務必附上白話解釋。全文控制在500字以內。
"""
    return prompt


def get_tw_gemini_key() -> str:
    """台股模式優先從 Streamlit 的 st.secrets 讀取（部署時由管理者設定，使用者看不到、
    也不用輸入）；本機測試若沒有設定 secrets，才退回讓開發者自己在旁邊輸入。"""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


def run_taiwan_app(shared_gemini_key: str = ""):
    """台股模式主流程：給完全新手用，輸入代號按一下就能看到型態分析＋白話AI解說。"""
    st.markdown(
        """
        <div style="text-align:center; padding: 12px 0 4px 0;">
            <h1 style="margin-bottom:4px;">📈 台股型態分析小幫手</h1>
            <p style="color:#6b7280; font-size:1.05rem;">
                輸入股票代號，馬上看懂最近的走勢型態（附白話解說，新手也看得懂）
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    date_placeholder = st.empty()
    date_placeholder.markdown(
        "<p style='text-align:center; color:#9ca3af; font-size:0.9rem;'>"
        "尚未查詢，輸入股票代號後即可看到資料日期</p>",
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        symbol_input = st.text_input(
            "請輸入台股代號",
            placeholder="例如：2330（台積電）、2317（鴻海）、0050（元大台灣50）",
            label_visibility="visible",
            key="tw_symbol_input",
        )
        run = st.button("🔍 開始分析", type="primary", use_container_width=True, key="tw_run_button")

    if not run:
        st.info("💡 輸入股票代號後按下「開始分析」即可，不需要輸入任何密碼或帳號。")
        return

    if not symbol_input.strip():
        st.error("請先輸入股票代號。")
        return

    gemini_key = shared_gemini_key or get_tw_gemini_key()

    with st.spinner("正在抓取股價資料..."):
        try:
            df_raw, resolved_symbol = fetch_tw_history(symbol_input)
        except Exception as exc:
            st.error(str(exc))
            return

    df = add_moving_averages(df_raw)
    trend = detect_trend_pattern(df)
    patterns = detect_candlestick_patterns(df)
    quote = fetch_tw_realtime_quote(resolved_symbol)

    latest_close = float(df["close"].iloc[-1])
    latest_date = df["date"].iloc[-1].date().isoformat()

    stats52 = compute_52week_stats(df, latest_close)
    company_info = get_tw_company_info(resolved_symbol)
    with st.spinner("正在比對大盤資料..."):
        index_df = fetch_index_history()
    relative_strength = compute_relative_strength(df, index_df)
    outlook = compute_outlook(trend, patterns, relative_strength)

    today_str = datetime.now().date().isoformat()
    if latest_date == today_str:
        date_note = "✅ 含今日最新收盤資料"
    else:
        date_note = "⚠️ 非今日資料（可能是假日、尚未收盤，或資料來源更新延遲）"
    date_placeholder.markdown(
        f"<p style='text-align:center; color:#374151; font-size:0.95rem;'>"
        f"📅 資料日期：<b>{latest_date}</b>　{date_note}</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if quote and quote.get("price") is not None:
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("目前價格", f"{quote['price']:.2f}",
                    f"{quote['change']:+.2f} ({quote['change_pct']:+.2f}%)" if quote.get("change") is not None else None)
        qc2.metric("今日區間", f"{_fmt_num(quote.get('day_low'))} ~ {_fmt_num(quote.get('day_high'))}")
        qc3.metric("成交量", _fmt_num(quote.get("volume"), 0))
        qc4.metric("資料代號", resolved_symbol)
    else:
        st.caption(f"（本次未取得即時報價，以下技術分析以最新收盤價 {latest_close:.2f} 為準，"
                   f"資料代號：{resolved_symbol}）")

    st.markdown(
        f"""
        <div style="border:1px solid {outlook['color']}; border-radius:10px;
                    padding:14px 18px; margin:10px 0; background:{outlook['color']}0d;">
            <span style="font-size:1.3rem; font-weight:700; color:{outlook['color']};">
                {outlook['emoji']} 綜合研判：{outlook['label']}
            </span>
            <p style="margin:6px 0 0 0; color:#4b5563; font-size:0.9rem;">{outlook['detail']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if company_info and (company_info.get("industry") or company_info.get("sector") or company_info.get("pe")):
        st.markdown("##### 🏢 公司基本資料")
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("產業別", company_info.get("industry") or company_info.get("sector") or "N/A")
        ic2.metric("本益比（PE）", f"{company_info['pe']:.1f}" if company_info.get("pe") else "N/A")
        ic3.metric("殖利率", f"{company_info['dividend_yield']:.1f}%" if company_info.get("dividend_yield") else "N/A")
        ic4.metric("市值（億元）", f"{company_info['market_cap'] / 1e8:,.0f}" if company_info.get("market_cap") else "N/A")
        st.caption("本益比、殖利率、市值僅供參考，資料可能有延遲或缺漏，正式投資決策請以官方公開資訊為準。")

    st.plotly_chart(chart_tw_pattern(df, symbol_input.strip().upper(), trend), use_container_width=True)

    st.markdown("##### 🔍 型態分析結果")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.metric("趨勢方向", trend["trend"])
        st.metric("支撐 ／ 壓力", f"{_fmt_num(trend['support'])} ／ {_fmt_num(trend['resistance'])}")
        if trend["breakout"]:
            st.warning(trend["breakout"])
        st.metric("近一年（52週）高／低", f"{_fmt_num(stats52['high'])} ／ {_fmt_num(stats52['low'])}")
        st.caption(f"目前價位落在近一年區間的 {stats52['position_pct']:.0f}% 位置"
                   "（0%接近最低點，100%接近最高點）")
    with tc2:
        st.markdown("**最近5天K線型態**")
        if patterns:
            for d, name, meaning in patterns:
                st.markdown(f"- **{d}｜{name}**：{meaning}")
        else:
            st.caption("已檢查最近5天的K線，但沒有出現十字星/鎚子線/吞噬型態等明顯型態，這是正常情況。")

    if relative_strength:
        diff = relative_strength["diff"]
        verdict = "比大盤強勢" if diff > 1 else "比大盤弱勢" if diff < -1 else "跟大盤差不多"
        st.markdown("##### ⚖️ 與大盤比較")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric(f"近{relative_strength['days']}個交易日漲跌", f"{relative_strength['stock_return']:+.1f}%")
        rc2.metric("同期大盤（加權指數）漲跌", f"{relative_strength['index_return']:+.1f}%")
        rc3.metric("相對表現", verdict, f"{diff:+.1f}%")
        st.caption("「相對表現」是這支股票漲跌幅減掉大盤漲跌幅，數字越大代表比大盤強勢。")

    st.markdown("---")
    st.markdown("##### 🤖 AI 白話解說")

    if not gemini_key:
        st.info(
            "目前這個網站還沒有設定AI解說功能（需要管理者在後台設定金鑰），"
            "但上面的型態分析結果已經可以直接看囉！"
        )
    else:
        with st.spinner("AI 正在整理白話解說，請稍候..."):
            try:
                prompt = build_tw_beginner_prompt(
                    symbol_input.strip().upper(), resolved_symbol, latest_close,
                    latest_date, trend, patterns, quote=quote,
                    stats52=stats52, relative_strength=relative_strength,
                    company_info=company_info,
                )
                explanation = call_gemini(prompt, gemini_key)
                st.markdown(explanation)
            except Exception as exc:
                st.warning(f"AI解說暫時無法產生：{exc}\n\n（不影響上面的型態分析結果）")

    st.markdown("---")
    st.caption(
        "📢 免責聲明：本工具僅根據歷史股價資料做規則式型態辨識與AI文字整理，"
        "屬於學習與教育用途，不構成投資建議，也不保證未來走勢；股票交易有風險，"
        "投資前應自行判斷或諮詢專業意見。"
    )


# ----------------------------------------------------------------------------
# 7. 共用主入口：先驗證密碼，再讓使用者選擇市場（美股／台股），兩種模式互相獨立
# ----------------------------------------------------------------------------

def main():
    if not check_password():
        return

    st.set_page_config(page_title="股票分析評估表", layout="wide", page_icon="📊")
    st.markdown(
        """
        <div style="text-align:center; padding: 4px 0 8px 0;">
            <h1 style="margin-bottom:4px;">📊 股票分析評估表</h1>
            <p style="color:#6b7280;">美股：技術指標＋Gemini AI＋持倉風控｜台股：型態分析＋白話AI解說</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 🔑 Gemini AI 金鑰")
    shared_gemini_key = st.text_input(
        "Gemini API 金鑰（美股、台股分析共用；已在後台設定好的話會自動帶入，也可以直接在這裡輸入）",
        type="password",
        value=_secret_default("GEMINI_API_KEY"),
        key="shared_gemini_key_input",
        help="至 https://aistudio.google.com/apikey 免費註冊取得",
    )
    st.caption("金鑰僅保存於本次瀏覽器 session 記憶體中，不會寫入檔案或上傳。輸入好金鑰後，"
               "下方選擇市場、輸入股票代號即可直接得到 AI 分析結果。")

    st.markdown(
        """
        <style>
        div[data-testid="stButton"] button {
            font-size: 1.25rem;
            font-weight: 700;
            padding: 0.9rem 0.6rem;
            border-radius: 12px;
            height: auto;
            white-space: normal;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**請選擇要分析的市場**")
    if "market_selector" not in st.session_state:
        st.session_state["market_selector"] = "us"

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        if st.button(
            "🇺🇸 美股\n（進階：技術指標＋停損停利＋持倉風控）",
            key="market_btn_us",
            type="primary" if st.session_state["market_selector"] == "us" else "secondary",
            use_container_width=True,
        ):
            st.session_state["market_selector"] = "us"
    with mcol2:
        if st.button(
            "🇹🇼 台股\n（新手：型態分析＋白話AI解說）",
            key="market_btn_tw",
            type="primary" if st.session_state["market_selector"] == "tw" else "secondary",
            use_container_width=True,
        ):
            st.session_state["market_selector"] = "tw"

    st.markdown("---")

    if st.session_state["market_selector"] == "us":
        run_us_app(shared_gemini_key=shared_gemini_key)
    else:
        run_taiwan_app(shared_gemini_key=shared_gemini_key)


if __name__ == "__main__":
    main()
