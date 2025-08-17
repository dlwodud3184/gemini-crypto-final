# app.py (LSTMAttention 앙상블 모델용 최종 앱)

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

# --- 1. AI 모델 정의 (학습 스크립트와 동일한 LSTMAttention 모델) ---
class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.4):
        super(LSTMAttentionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention_fc = nn.Linear(hidden_dim * 2, 1) # 양방향이라 *2
        self.classifier_fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = F.softmax(self.attention_fc(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return torch.sigmoid(self.classifier_fc(context_vector))

# --- 2. 데이터 수집 함수 ---
@st.cache_data(ttl=300) # 5분 동안 결과 캐싱
def fetch_prediction_data(symbol):
    days = 150 # 예측에는 최근 150일 데이터만 사용
    start_date = (datetime.now(UTC) - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 가격 데이터 (ccxt 우선, yf fallback)
    price_df = None
    try:
        # Streamlit Cloud의 Secrets에서 API 키를 가져와 ccxt에 인증
        exchange = ccxt.binance({
            'apiKey': st.secrets.get('BINANCE_API_KEY'),
            'secret': st.secrets.get('BINANCE_SECRET_KEY'),
        })
        ohlcv = exchange.fetch_ohlcv(f'{symbol}/USDT', '1d', since=exchange.parse8601(f"{start_date}T00:00:00Z"), limit=days)
        price_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        price_df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='ms')
        price_df.set_index('timestamp', inplace=True)
    except Exception as e:
        st.warning(f"ccxt 데이터 수집 실패 ({e}). Yahoo Finance로 대체합니다.")
        try:
            price_df = yf.download(f"{symbol}-USD", start=start_date, progress=False, auto_adjust=True)
            if isinstance(price_df.columns, pd.MultiIndex):
                price_df.columns = price_df.columns.droplevel(0)
            if price_df.empty:
                st.error(f"{symbol}에 대한 데이터를 Yahoo Finance에서도 가져올 수 없습니다.")
                return None
        except Exception as ex:
            st.error(f"Yahoo Finance 데이터 수집 중 오류 발생: {ex}")
            return None
    
    if price_df is None or price_df.empty:
        st.error("모든 데이터 소스에서 가격 정보를 가져오지 못했습니다.")
        return None

    master_df = price_df.copy()
    master_df.index = master_df.index.tz_localize(None)

    # 거시 경제 데이터 (FRED, FNG)
    try:
        fred_api_key = st.secrets['FRED_API_KEY']
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DTWEXBGS&api_key={fred_api_key}&file_type=json&observation_start={start_date}"
        res = requests.get(url); res.raise_for_status()
        dxy_data = res.json()['observations']
        dxy_df = pd.DataFrame(dxy_data)[['date', 'value']].rename(columns={'date': 'timestamp', 'value': 'dxy'})
        dxy_df['timestamp'] = pd.to_datetime(dxy_df['timestamp'])
        dxy_df.set_index('timestamp', inplace=True)
        dxy_df['dxy'] = pd.to_numeric(dxy_df['dxy'], errors='coerce')
        master_df = master_df.join(dxy_df, how='left')
    except Exception as e:
        st.warning(f"DXY 데이터 수집 실패: {e}")

    try:
        fng_res = requests.get(f"https://api.alternative.me/fng/?limit={days}"); fng_res.raise_for_status()
        fng_df = pd.DataFrame(fng_res.json()['data'])
        fng_df['timestamp'] = pd.to_datetime(pd.to_numeric(fng_df['timestamp']), unit='s')
        fng_df.set_index('timestamp', inplace=True)
        fng_df['fng_value'] = pd.to_numeric(fng_df['value'], errors='coerce')
        master_df = master_df.join(fng_df[['fng_value']], how='left')
    except Exception as e:
        st.warning(f"FNG 데이터 수집 실패: {e}")

    # 자산별 특화 데이터 (온체인, TVL)
    try:
        if symbol == 'BTC':
            res_cm = requests.get(f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics=CapMrktCurUSD,TxTfrValAdjUSD&start_time={start_date}&frequency=1d")
            res_cm.raise_for_status()
            onchain_df = pd.DataFrame(res_cm.json().get('data', []))
            if not onchain_df.empty:
                onchain_df['time'] = pd.to_datetime(onchain_df['time']).dt.tz_localize(None)
                onchain_df.set_index('time', inplace=True)
                onchain_df['market_cap'] = pd.to_numeric(onchain_df['CapMrktCurUSD'], errors='coerce')
                onchain_df['trade_volume'] = pd.to_numeric(onchain_df['TxTfrValAdjUSD'], errors='coerce')
                onchain_df['nvt_ratio'] = onchain_df['market_cap'] / (onchain_df['trade_volume'] + 1e-8)
                master_df = master_df.join(onchain_df[['nvt_ratio']], how='left')
        elif symbol in ['ETH', 'SOL']:
            asset_name = 'Ethereum' if symbol == 'ETH' else 'Solana'
            res_tvl = requests.get(f"https://api.llama.fi/charts/{asset_name}"); res_tvl.raise_for_status()
            tvl_df = pd.DataFrame(res_tvl.json())
            tvl_df['date'] = pd.to_datetime(tvl_df['date'], unit='s')
            tvl_df.set_index('date', inplace=True)
            tvl_col_name = f'{symbol.lower()}_tvl'
            tvl_df[tvl_col_name] = pd.to_numeric(tvl_df['totalLiquidityUSD'], errors='coerce')
            master_df = master_df.join(tvl_df[[tvl_col_name]], how='left')
    except Exception as e:
        st.warning(f"{symbol} 특화 데이터 수집 실패: {e}")

    # 최종 데이터 가공
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
            
            # 표준화(StandardScaler)로 변경
            scaled_features = (latest_data[features] - latest_data[features].mean()) / latest_data[features].std()
            scaled_features.fillna(0, inplace=True)
            tensor = torch.tensor(np.array([scaled_features.values]), dtype=torch.float32)

            for i in range(N_MODELS):
                # 최종 강화 LSTM 모델 파일명을 불러오도록 수정
                model_path = f'gemini_{selected_asset.lower()}_lstm_model_{i}.pth'
                if not os.path.exists(model_path):
                    st.warning(f"모델 파일({model_path})을 찾을 수 없습니다. 다음 모델로 넘어갑니다.")
                    continue

                model = LSTMAttentionClassifier(input_dim=len(features))
                # 클라우드 환경(CPU)에서 실행되도록 map_location 설정
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()

                with torch.no_grad():
                    prob = model(tensor).squeeze().item()
                    predictions.append(prob)
            
            if not predictions:
                st.error("예측에 사용할 모델을 하나도 찾을 수 없습니다. 모델 파일들을 GitHub에 함께 업로드했는지 확인하세요.")
                st.stop()

            # 소프트 보팅: 예측 확률의 평균 계산
            final_prob = np.mean(predictions)

            st.success("🎉 예측 완료!")
            col1, col2 = st.columns(2)
            col1.metric("AI 앙상블 예측 상승 확률", f"{final_prob:.2%}")
            col2.metric("분석 기준 가격", f"${df['Close'].iloc[-1]:,.2f}")
            
            with st.expander("개별 모델 예측 보기"):
                chart_data = pd.DataFrame({
                    "Model": [f"Model {i+1}" for i in range(len(predictions))],
                    "Probability (%)": [p * 100 for p in predictions]
                })
                st.bar_chart(chart_data.set_index("Model"))

        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")
