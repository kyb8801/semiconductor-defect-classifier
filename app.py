
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# 페이지 설정
st.set_page_config(
    page_title="반도체 불량 분류 AI",
    page_icon="🔬",
    layout="wide"
)

# 모델 로드 (캐싱으로 빠르게)
@st.cache_resource
def load_models():
    imputer      = joblib.load('models/imputer.pkl')
    top_features = joblib.load('models/top_features.pkl')
    scaler       = joblib.load('models/scaler.pkl')
    xgb_model    = joblib.load('models/xgb_best.pkl')
    calibrator   = joblib.load('models/calibrator.pkl')
    return imputer, top_features, scaler, xgb_model, calibrator

imputer, top_features, scaler, xgb_model, calibrator = load_models()

# 핵심 센서 이름 + 정상 범위 (상관분석 Top 5)
SENSOR_INFO = {
    59:  {"name": "센서_59번",  "min": 2700, "max": 3500, "default": 3000},
    103: {"name": "센서_103번", "min": 2200, "max": 2800, "default": 2500},
    510: {"name": "센서_510번", "min": 2050, "max": 2300, "default": 2200},
    21:  {"name": "센서_21번",  "min": 1.2,  "max": 1.6,  "default": 1.4},
    28:  {"name": "센서_28번",  "min": 0.5,  "max": 2.0,  "default": 1.0},
}

# 제목
st.title("🔬 반도체 불량 분류 AI 데모")
st.markdown("**SECOM 공정 센서 데이터 기반 실시간 불량 판정 시스템**")
st.markdown("---")

# 사이드바: 센서값 입력
st.sidebar.title("⚙️ 센서값 입력")
st.sidebar.markdown("핵심 센서 5개 값을 조정하세요")

sensor_values = {}
for sensor_idx, info in SENSOR_INFO.items():
    sensor_values[sensor_idx] = st.sidebar.slider(
        info["name"],
        min_value=float(info["min"]),
        max_value=float(info["max"]),
        value=float(info["default"]),
        step=float((info["max"] - info["min"]) / 100)
    )

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 불량 판정", type="primary", use_container_width=True)

# 메인 화면
col1, col2, col3 = st.columns(3)

if predict_btn:
    # 기본값 데이터 생성 (590개 센서)
    X_input = np.zeros(590)
    
    # 핵심 센서 5개 값 설정
    for sensor_idx, value in sensor_values.items():
        if sensor_idx < 590:
            X_input[sensor_idx] = value

    # 예측
    X_imputed = imputer.transform(X_input.reshape(1, -1))
    prob_raw  = xgb_model.predict_proba(X_imputed)[:, 1][0]
    prob_cal  = float(calibrator.predict([prob_raw])[0])

    threshold = 0.257
    is_defect = prob_cal >= threshold
    prediction = "불량" if is_defect else "정상"
    risk = "높음" if prob_cal >= 0.5 else "중간" if prob_cal >= 0.3 else "낮음"
    color = "#E24B4A" if is_defect else "#1D9E75"

    # 결과 표시
    with col1:
        st.metric("판정 결과", prediction)
        st.metric("불량 확률", f"{prob_cal:.1%}")

    with col2:
        st.metric("위험도", risk)
        st.metric("Threshold", f"{threshold}")

    with col3:
        if is_defect:
            st.error(f"⚠️ 불량 판정! 확률 {prob_cal:.1%}")
        else:
            st.success(f"✅ 정상 판정! 확률 {1-prob_cal:.1%}")

    st.markdown("---")

    # 확률 게이지
    st.subheader("📊 불량 확률 게이지")
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(["불량 확률"], [prob_cal], color=color, height=0.4)
    ax.barh(["불량 확률"], [1-prob_cal], left=[prob_cal],
            color="#E8E8E8", height=0.4)
    ax.axvline(x=threshold, color='black', linestyle='--',
               linewidth=2, label=f'Threshold={threshold}')
    ax.set_xlim(0, 1)
    ax.set_xlabel("확률")
    ax.legend()
    ax.set_title(f"불량 확률: {prob_cal:.1%}")
    st.pyplot(fig)
    plt.close()

    # Threshold Trade-off 테이블
    st.subheader("📋 운영 모드별 Trade-off")
    tradeoff_data = {
        "모드": ["엄격 모드", "균형 모드 (현재)", "여유 모드"],
        "Threshold": [0.20, 0.257, 0.30],
        "예상 탐지율": ["93%", "80%", "53%"],
        "예상 오탐률": ["높음", "중간", "낮음"],
        "권장 상황": ["중요 공정", "일반 공정", "오탐 비용 높을 때"]
    }
    df = pd.DataFrame(tradeoff_data)
    st.dataframe(df, use_container_width=True)

else:
    st.info("👈 왼쪽 사이드바에서 센서값을 조정하고 **불량 판정** 버튼을 누르세요!")

    # 모델 성능 요약
    st.subheader("📈 모델 성능 요약")
    perf_data = {
        "모델": ["MLP Optuna", "XGBoost Optuna", "Random Forest", "SVM"],
        "K-fold F1": [0.291, 0.288, 0.261, 0.178],
        "ROC-AUC": [0.853, 0.787, 0.779, 0.742],
        "비고": ["최고 성능!", "안정적", "무난함", "고차원 약함"]
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

# 하단 정보
st.markdown("---")
st.markdown("**Author:** YoungBum Kim (김용범) PhD | **GitHub:** kyb8801 | **Dataset:** UCI SECOM")
