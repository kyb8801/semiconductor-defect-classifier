# 🔬 Semiconductor Defect Classifier

반도체 공정 센서 데이터(SECOM)를 활용한 불량 자동 분류 AI 포트폴리오

**Author:** YoungBum Kim (김용범) | PhD in Energy Science  
**Period:** 2026.03 ~ 2026.04  
**Target:** ASML / KLA / Samsung Electronics AI Engineer Position

---

## 🎯 Project Overview

반도체 공정에서 발생하는 불량을 센서 데이터로 자동 탐지하는 딥러닝 모델 개발.  
도메인 지식(나노광학, 계측)과 AI를 결합한 **도메인 전문가형 AI 엔지니어링** 접근법.

---

## 📊 Dataset

- **SECOM Dataset** (UCI Machine Learning Repository)
- 1,567개 웨이퍼 × 590개 공정 센서
- 클래스 불균형: 정상 93.4% / 불량 6.6%

---

## 🧪 전체 실험 결과 (K-fold 기준)

| 모델 | K-fold F1 | ROC-AUC | 비고 |
|------|-----------|---------|------|
| **MLP Optuna** | **0.291** | **0.853** | 최고 성능! |
| XGBoost Optuna | 0.288 | 0.787 | |
| Random Forest | 0.261 | 0.779 | |
| XGBoost 기본 | 0.252 | 0.787 | 안정적 |
| Ensemble MLP+XGB | 0.247 | - | |
| LightGBM | 0.222 | 0.765 | |
| SVM | 0.178 | 0.742 | 고차원 약함 |

---

## 🔑 Key Techniques

**불균형 처리:**
- Class Weight, SMOTE, Threshold 조정, Focal Loss

**Feature Engineering:**
- 상관분석 Top 20 센서 선택
- PCA 차원 축소

**모델:**
- MLP, ResMLP (Skip Connection), XGBoost, LightGBM, RF, SVM
- Optuna 하이퍼파라미터 자동 튜닝
- K-fold Cross Validation

**해석:**
- Grad-CAM (센서 중요도)
- PDP (센서 임계값 시각화)
- PR Curve + ROC-AUC
- Error Analysis (놓친 불량 심층 분석)
- Calibration (확률 보정)

**앙상블:**
- Weighted Ensemble, Stacking

---

## 💡 핵심 인사이트

1. **센서_59번**이 상관분석/Grad-CAM/XGBoost 모두에서 상위권 → 가장 신뢰할 수 있는 핵심 센서
2. **Optuna 적용** 후 성능 향상: MLP 0.252 → 0.291 (+15%), XGBoost 0.252 → 0.288 (+14%)
3. **K-fold 기준** 단순 Val 평가(F1=0.36)는 과대평가 → 실제 성능(F1=0.252)
4. **PDP 분석**: 센서_21번 -6000, 센서_59번 10 이상이면 불량 위험!
5. **Error Analysis**: 놓친 불량 3개 중 1개는 정상과 구분 불가 → 추가 센서 필요

---

## 🏗️ Model Architecture
```
MLP Baseline:      562 → 256 → 64 → 2
ResMLP:            562 → [256+skip] → [64+skip] → 2
Correlation Top20: 20  → 64  → 32 → 2
MLP Optuna:        20  → 154 → 42 → 2  ← Best Model
```

---

## 🚀 Getting Started
```bash
git clone https://github.com/kyb8801/semiconductor-defect-classifier
cd semiconductor-defect-classifier
conda activate sem-defect
jupyter notebook
```

**Streamlit 웹앱 실행:**
```bash
streamlit run app.py
```

→ http://localhost:8501 에서 실시간 불량 판정 데모!

---

## 📁 Project Structure
```
semiconductor-defect-classifier/
├── data/
│   ├── uci-secom.csv
│   ├── X_train.npy / X_val.npy / X_test.npy
├── models/
│   ├── xgb_best.pkl       # XGBoost Optuna 최적 모델
│   ├── mlp_best.pth       # MLP Optuna 최적 모델
│   ├── calibrator.pkl     # Calibration 모델
│   ├── imputer.pkl        # 결측값 처리
│   ├── scaler.pkl         # 정규화
│   └── top_features.pkl   # Top 20 센서 목록
├── app.py                 # Streamlit 웹앱
├── week2_mlp_secom.ipynb  # MLP + 불균형 처리
├── week3_advanced.ipynb   # XGBoost + 고급 분석
└── README.md
```

---

## 📈 Next Steps

- [ ] NSOM MoSe₂ 하이퍼스펙트럴 데이터 적용
- [ ] 논문 Draft (Ultramicroscopy / MST)
- [ ] Streamlit Cloud 배포

---

## 👨‍🔬 About Author

- **전문:** 나노광학 계측, NSOM/TERS, KrF 리소그래피
- **학위:** 성균관대학교 에너지과학과 박사
- **GitHub:** [kyb8801](https://github.com/kyb8801)
- **Kaggle:** [aioptic](https://kaggle.com/aioptic)
