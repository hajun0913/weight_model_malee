import streamlit as st
import pandas as pd
import joblib

# 1. 페이지 제목 및 설명
st.title("신체 정보를 이용한 몸무게 예측 머신러닝 모델")
st.write("남자 신체 정보를 입력하면 몸무게를 예측합니다.")

# 2. 사이드바 구성
st.sidebar.header("머신러닝 모델 설계 실습 (다중회귀)")

# 3. 사용자 입력 받기 (첫 번째 코드의 변수 반영: 키, 허리둘레, 가슴둘레)
height = st.slider("키 (cm)", 140.0, 190.0, 173.0)      # 남성 평균 데이터 근처로 기본값 변경
waist = st.slider("허리 둘레 (cm)", 50.0, 120.0, 83.0)   # 엉덩이 -> 허리
chest = st.slider("가슴 둘레 (cm)", 70.0, 130.0, 95.0)   # 새로운 가슴둘레 슬라이더 추가

# 4. 모델 로드 (남자 모델)
@st.cache_resource  # 모델을 매번 새로 로드하지 않도록 캐싱하여 성능 최적화
def load_model():
    return joblib.load("weight_model_male.pkl")

model = load_model()

# 5. 예측 실행 버튼
if st.button("몸무게 예측하기"):
    # 첫 번째 코드와 동일한 컬럼 순서 및 DataFrame 형태로 입력 데이터 생성
    input_data = pd.DataFrame([[waist, chest, height]], columns=['허리둘레', '가슴둘레', '키'])
    
    # 예측 및 결과 출력
    prediction = model.predict(input_data)
    st.success(f"▶ 모델이 예측한 몸무게: {prediction[0]:.1f} kg")