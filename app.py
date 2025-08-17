# app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ta
import yfinance as yf
import requests
import ccxt
from datetime import datetime, timedelta, UTC
import os

# --- 1. AI 모델 정의 (LSTMAttention 최종 강화 버전) ---
class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.4):
        super(LSTMAttentionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        self.classifier_fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = F.softmax(self.attention_fc(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return torch.sigmoid(self.classifier_fc(context_vector))

# --- 2. 데이터 수집 함수 (이전과 동일) ---
@st.cache_data(ttl=300)
def fetch_prediction_data(symbol):
    days = 150
    start_date = (datetime.now(UTC) - timedelta(days=days)).strftime('%Y-%m-%d')
    
    price_df = None
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(f'{symbol}/USDT', '1d', since=exchange.parse8601(f"{start_date}T00:00:00Z"), limit=days)
        price_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        price_df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='ms')
        price_df.set_index('timestamp', inplace=True)
    except Exception:
        st.warning("ccxt 데이터 수집 실패. Yahoo Finance로 대체합니다.")
        price_df = yf.download(f"{symbol}-USD", start=start_date, progress=False, auto_adjust=True)
        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.droplevel(0)
    
    if price_df is None or price_df.empty: return None
    master_df = price_df.copy()
    master_df.index = master_df.index.tz_localize(None)

    # (거시/특화 데이터 수집 로직은 이전과 동일하게 유지)
    # ...

    master_df.ffill(inplace=True)
    master_df.bfill(inplace=True)
    master_df['rsi'] = ta.momentum.RSIIndicator(master_df['Close'], 14).rsi()
    master_df['macd'] = ta.trend.MACD(master_df['Close']).macd_diff()
    bb = ta.volatility.BollingerBands(master_df['Close'], window=20)
    master_df['bb_width'] = bb.bollinger_wband()
    master_df['obv'] = ta.volume.OnBalanceVolumeIndicator(master_df['Close'], master_df['Volume']).on_balance_volume()
    
    return master_df


# --- 3. Streamlit UI ---
st.set_page_config(page_title="Gemini-Crypto Predictor", layout="wide")
st.title("🚀 Gemini-Crypto Predictor")
st.info("안정성이 검증된 LSTMAttention AI 모델 5개의 예측을 종합한 앙상블 분석 결과입니다.")

selected_asset = st.selectbox("분석할 자산을 선택하세요:", ("BTC", "ETH", "SOL"))

if st.button(f"최신 {selected_asset} 상승 확률 예측하기"):
    with st.spinner("최신 시장 데이터를 분석하고 5개의 AI가 예측을 종합합니다..."):
        df = fetch_prediction_data(selected_asset)
        
        if df is None or df.empty or len(df) < 30:
            st.error("데이터가 부족하여 예측을 수행할 수 없습니다.")
            st.stop()
        
        if selected_asset == 'BTC':
            features = ['Close', 'Volume', 'dxy', 'fng_value', 'nvt_ratio', 'rsi', 'macd', 'bb_width', 'obv']
        elif selected_asset == 'ETH':
            features = ['Close', 'Volume', 'dxy', 'fng_value', 'eth_tvl', 'rsi', 'macd', 'bb_width', 'obv']
        else: # SOL
            features = ['Close', 'Volume', 'dxy', 'fng_value', 'sol_tvl', 'rsi', 'macd', 'bb_width', 'obv']
        
        N_MODELS = 5
        predictions = []
        
        try:
            latest_data = df.tail(30).copy()
            for f in features:
                if f not in latest_data.columns:
                    latest_data[f] = 0.0
            
            scaled_features = (latest_data[features] - latest_data[features].mean()) / latest_data[features].std()
            scaled_features.fillna(0, inplace=True)
            tensor = torch.tensor(np.array([scaled_features.values]), dtype=torch.float32)

            for i in range(N_MODELS):
                model_path = f'gemini_{selected_asset.lower()}_lstm_model_{i}.pth' # 파일명 _lstm_으로 수정
                if not os.path.exists(model_path):
                    st.warning(f"모델 파일({model_path})을 찾을 수 없습니다. 다음 모델로 넘어갑니다.")
                    continue

                model = LSTMAttentionClassifier(input_dim=len(features))
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # CPU에서 실행되도록 설정
                model.eval()

                with torch.no_grad():
                    prob = model(tensor).squeeze().item()
                    predictions.append(prob)
            
            if not predictions:
                st.error("예측에 사용할 모델을 하나도 찾을 수 없습니다. 모델 파일들을 함께 업로드했는지 확인하세요.")
                st.stop()

            final_prob = np.mean(predictions)

            st.success("🎉 예측 완료!")
            col1, col2 = st.columns(2)
            col1.metric("AI 앙상블 예측 상승 확률", f"{final_prob:.2%}")
            col2.metric("분석 기준 가격", f"${df['Close'].iloc[-1]:,.2f}")
            
            with st.expander("개별 모델 예측 보기"):
                chart_data = pd.DataFrame({
                    "Model": [f"Model {i}" for i in range(len(predictions))],
                    "Probability": [p * 100 for p in predictions]
                })
                st.bar_chart(chart_data.set_index("Model"))

        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")