# 🔬 Semiconductor Defect Classifier

반도체 공정 센서 데이터(SECOM)를 활용한 불량 자동 분류 AI 포트폴리오

**Author:** YoungBum Kim (김용범) | PhD in Energy Science  
**Period:** 2026.03 ~ 2026.05  
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
- 전처리: 결측값 50% 이상 센서 제거 (590 → 562개)

---

## 🧪 Experiments & Results

| 방법 | 불량 F1 | 불량 탐지 | 비고 |
|------|---------|----------|------|
| 기본 MLP (562개 센서) | 0.00 | 0/15 | 베이스라인 |
| + Class Weight (T=0.5) | 0.09 | 1/15 | 불균형 보정 |
| + Class Weight (T=0.2) | 0.28 | 8/15 | Threshold 조정 |
| + SMOTE (T=0.2) | 0.25 | 7/15 | 오버샘플링 |
| ResMLP + Skip Connection | 0.26 | 7/15 | ResNet 아이디어 적용 |
| Focal Loss | 0.23 | 5/15 | 어려운 샘플 집중 |
| PCA 50개 + Class Weight | 0.25 | 9/15 | 차원 축소 |
| **상관분석 Top 20 + Class Weight** | **0.29** | **11/15** | **최고 성능** |

### 핵심 인사이트
> 562개 센서 전부 사용하는 것보다 **불량과 상관관계 높은 센서 20개만 선택**하는 것이 더 효과적.  
> 도메인 지식 기반 Feature Selection → 노이즈 제거 → 성능 향상

---

## 🏗️ Model Architecture
```
MLP Baseline:      562 → 256 → 64 → 2
ResMLP:            562 → [256+skip] → [64+skip] → 2
Correlation Top20: 20  → 64  → 32 → 2  ← Best Model
```

---

## 🔑 Key Techniques

- **Class Weight** (1.0 / 5.0): 불량 클래스 벌점 5배
- **Threshold 조정** (0.5 → 0.3): 탐지 민감도 향상
- **Early Stopping** (patience=10): 과적합 방지
- **SMOTE**: 불량 데이터 보간 증강 (73 → 1,023개)
- **Skip Connection**: Vanishing Gradient 방지
- **Correlation Analysis**: 불량 연관 센서 Top 20 선택
- **PCA**: 50개 주성분으로 차원 축소 (분산 62.1% 보존)

---

## 📁 Project Structure
```
semiconductor-defect-classifier/
├── data/
│   ├── uci-secom.csv          # 원본 데이터
│   ├── X_train.npy            # 전처리된 학습 데이터
│   ├── X_val.npy              # 검증 데이터
│   └── X_test.npy             # 테스트 데이터
├── week2_mlp_secom.ipynb      # MLP + Class Weight + SMOTE
├── week2_cnn_basic.ipynb      # CNN 기초 이론
└── README.md
```

---

## 🚀 Getting Started
```bash
git clone https://github.com/kyb8801/semiconductor-defect-classifier
cd semiconductor-defect-classifier
conda activate sem-defect
jupyter notebook
```

---

## 📈 Next Steps

- [ ] Grad-CAM 시각화: 어떤 센서가 불량 판단에 중요한가?
- [ ] Streamlit 웹앱 배포 (실시간 불량 분류 데모)
- [ ] 논문 Draft (Ultramicroscopy / MST)

---

## 👨‍🔬 About Author

- **전문:** 나노광학 계측, NSOM/TERS, KrF 리소그래피
- **학위:** 성균관대학교 에너지과학과 박사
- **GitHub:** [kyb8801](https://github.com/kyb8801)
- **Kaggle:** [aioptic](https://kaggle.com/aioptic)
