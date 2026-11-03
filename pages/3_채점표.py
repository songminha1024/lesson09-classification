
import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. 기본 설정
# ==========================================

st.title("📋 모델 채점표")

st.write(
    "테스트 데이터를 이용하여 두 분류 모델의 예측 결과를 평가합니다. "
    "혼동행렬과 평가 지표를 비교하고, "
    "잘못 예측한 사람들도 직접 확인할 수 있습니다."
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부",
    "bmi": "체질량지수"
}


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.write(error)
    st.stop()


# 분류 모델 페이지와 동일한 데이터 정리
df["stroke"] = pd.to_numeric(
    df["stroke"], errors="coerce"
)

df = df.dropna(subset=["stroke"]).copy()
df["stroke"] = df["stroke"].astype(int)

df = df.sort_values("id").reset_index(drop=True)


# ==========================================
# 3. 데이터 분리
# ==========================================

st.header("1. 데이터 분리")

# 10명씩 묶어 앞의 3명을 테스트 데이터로 사용
test_mask = df.index % 10 < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

st.write(
    f"훈련 데이터: **{len(train_df):,}명**"
    " | "
    f"테스트 데이터: **{len(test_df):,}명**"
)

# 정답 데이터
y_train = train_df["stroke"]
y_test = test_df["stroke"]

# 입력 데이터
X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

# 숫자형 변환
for column in FEATURES:
    X_train[column] = pd.to_numeric(
        X_train[column], errors="coerce"
    )

    X_test[column] = pd.to_numeric(
        X_test[column], errors="coerce"
    )


# ==========================================
# 4. BMI 결측값 처리
# ==========================================

# 이 페이지의 기본 속성에는 BMI가 포함되지 않으므로
# BMI 결측값은 처리하지 않는다.

st.info(
    "분류 모델 페이지의 기본 선택 속성과 동일하게 "
    "나이, 평균 혈당, 고혈압, 심장병을 사용합니다. "
    "따라서 체질량지수의 결측값은 처리하지 않습니다."
)

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("입력 속성에 결측값이 있습니다.")
    st.stop()


# ==========================================
# 5. 모델 학습
# ==========================================

st.header("2. 분류 모델 학습")

# ------------------------------------------
# 로지스틱 회귀
# ------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

log_model.fit(X_train_scaled, y_train)

log_train_pred = log_model.predict(X_train_scaled)
log_test_pred = log_model.predict(X_test_scaled)


# ------------------------------------------
# 의사결정트리
# ------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)


# ------------------------------------------
# 기준 모델
# ------------------------------------------

# 훈련 데이터에서 더 많은 클래스를 선택
majority_class = y_train.value_counts().idxmax()

baseline_train_pred = [
    majority_class
] * len(y_train)

baseline_test_pred = [
    majority_class
] * len(y_test)


# ==========================================
# 6. 혼동행렬 계산 함수
# ==========================================

def calculate_confusion(y_true, y_pred):

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    # 양성: stroke = 1
    TP = int(((y_true == 1) & (y_pred == 1)).sum())

    # 실제 음성, 예측 양성
    FP = int(((y_true == 0) & (y_pred == 1)).sum())

    # 실제 양성, 예측 음성
    FN = int(((y_true == 1) & (y_pred == 0)).sum())

    # 실제 음성, 예측 음성
    TN = int(((y_true == 0) & (y_pred == 0)).sum())

    return TP, FP, FN, TN


# ==========================================
# 7. 평가 지표 계산 함수
# ==========================================

def calculate_metrics(TP, FP, FN, TN):

    total = TP + FP + FN + TN

    # 정확도
    accuracy = (
        (TP + TN) / total
        if total > 0 else None
    )

    # 재현율
    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0 else None
    )

    # 정밀도
    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0 else None
    )

    # F1
    if precision is None or recall is None:
        f1 = None

    elif precision + recall == 0:
        f1 = 0

    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def format_metric(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value:.2%}"


# ==========================================
# 8. 세 모델의 혼동행렬 계산
# ==========================================

log_TP, log_FP, log_FN, log_TN = calculate_confusion(
    y_test, log_test_pred
)

tree_TP, tree_FP, tree_FN, tree_TN = calculate_confusion(
    y_test, tree_test_pred
)

base_TP, base_FP, base_FN, base_TN = calculate_confusion(
    y_test, baseline_test_pred
)


# ==========================================
# 9. 테스트 데이터 정보
# ==========================================

st.header("3. 테스트 데이터 확인")

test_positive = int((y_test == 1).sum())

st.write(
    f"테스트용 사람 수: **{len(y_test):,}명** "
    "| "
    f"그중 실제 뇌졸중인 사람: **{test_positive:,}명**"
)


# ==========================================
# 10. 혼동행렬 표시 함수
# ==========================================

def show_confusion_matrix(TP, FP, FN, TN):

    matrix = pd.DataFrame(
        [
            [
                f"TN: {TN}명",
                f"FP: {FP}명"
            ],
            [
                f"FN: {FN}명",
                f"TP: {TP}명"
            ]
        ],
        index=[
            "실제 뇌졸중 아님",
            "실제 뇌졸중 있음"
        ],
        columns=[
            "예측 뇌졸중 아님",
            "예측 뇌졸중 있음"
        ]
    )

    matrix["행 합계"] = [
        TN + FP,
        FN + TP
    ]

    matrix.loc["열 합계"] = [
        TN + FN,
        FP + TP,
        TN + FP + FN + TP
    ]

    st.dataframe(
        matrix,
        use_container_width=True
    )


# ==========================================
# 11. 로지스틱 회귀 혼동행렬
# ==========================================

st.header("4. 모델별 혼동행렬")

st.subheader(
    "로지스틱 회귀(확률로 답하는 모델)"
)

show_confusion_matrix(
    log_TP,
    log_FP,
    log_FN,
    log_TN
)

st.caption(
    "TP: 실제 뇌졸중을 뇌졸중으로 예측 | "
    "FP: 실제 뇌졸중이 아닌데 뇌졸중으로 예측 | "
    "FN: 실제 뇌졸중인데 아니라고 예측 | "
    "TN: 실제 뇌졸중이 아닌 사람을 아니라고 예측"
)


# ==========================================
# 12. 의사결정트리 혼동행렬
# ==========================================

st.subheader(
    "의사결정트리(질문으로 답하는 모델)"
)

show_confusion_matrix(
    tree_TP,
    tree_FP,
    tree_FN,
    tree_TN
)


# ==========================================
# 13. 기준 모델 혼동행렬
# ==========================================

st.subheader(
    "훈련용에서 많은 쪽으로만 답하는 모델"
)

show_confusion_matrix(
    base_TP,
    base_FP,
    base_FN,
    base_TN
)


# ==========================================
# 14. 평가 지표 비교
# ==========================================

st.header("5. 세 모델의 평가 지표 비교")

log_metrics = calculate_metrics(
    log_TP, log_FP, log_FN, log_TN
)

tree_metrics = calculate_metrics(
    tree_TP, tree_FP, tree_FN, tree_TN
)

base_metrics = calculate_metrics(
    base_TP, base_FP, base_FN, base_TN
)

metrics_table = pd.DataFrame(
    [
        {
            "모델": "로지스틱 회귀(확률로 답하는 모델)",
            **{
                key: format_metric(value)
                for key, value in log_metrics.items()
            }
        },
        {
            "모델": "의사결정트리(질문으로 답하는 모델)",
            **{
                key: format_metric(value)
                for key, value in tree_metrics.items()
            }
        },
        {
            "모델": "훈련용에서 많은 쪽으로만 답하는 모델",
            **{
                key: format_metric(value)
                for key, value in base_metrics.items()
            }
        }
    ]
)

st.dataframe(
    metrics_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "정확도 = (TP + TN) / 전체 인원 | "
    "재현율 = TP / (TP + FN) | "
    "정밀도 = TP / (TP + FP) | "
    "F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)"
)


# ==========================================
# 15. 잘못 예측한 사람 목록 생성
# ==========================================

def get_error_people(predictions):

    result = test_df.copy()

    result["예측"] = predictions
    result["실제"] = y_test.to_numpy()

    # 실제 뇌졸중인데 아니라고 예측한 사람
    missed = result[
        (result["실제"] == 1)
        & (result["예측"] == 0)
    ].copy()

    # 실제 뇌졸중이 아닌데 뇌졸중이라고 예측한 사람
    false_alarm = result[
        (result["실제"] == 0)
        & (result["예측"] == 1)
    ].copy()

    # 나이가 많은 순서로 정렬
    missed = missed.sort_values(
        "age", ascending=False
    )

    false_alarm = false_alarm.sort_values(
        "age", ascending=False
    )

    return missed, false_alarm


# ==========================================
# 16. 모델 선택
# ==========================================

st.header("6. 잘못 예측한 사람 확인")

model_choice = st.selectbox(
    "확인할 모델을 선택하세요.",
    [
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
        "훈련용에서 많은 쪽으로만 답하는 모델"
    ]
)

if model_choice == "로지스틱 회귀(확률로 답하는 모델)":

    missed, false_alarm = get_error_people(
        log_test_pred
    )

    expected_FN = log_FN
    expected_FP = log_FP

elif model_choice == "의사결정트리(질문으로 답하는 모델)":

    missed, false_alarm = get_error_people(
        tree_test_pred
    )

    expected_FN = tree_FN
    expected_FP = tree_FP

else:

    missed, false_alarm = get_error_people(
        baseline_test_pred
    )

    expected_FN = base_FN
    expected_FP = base_FP


# ==========================================
# 17. 목록 표시 함수
# ==========================================

def prepare_display(data):

    display = data[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease"
        ]
    ].copy()

    display = display.rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부"
        }
    )

    display["고혈압 여부"] = display[
        "고혈압 여부"
    ].map({
        0: "없음",
        1: "있음"
    })

    display["심장병 여부"] = display[
        "심장병 여부"
    ].map({
        0: "없음",
        1: "있음"
    })

    display["나이"] = display["나이"].round(1)

    display["평균 혈당"] = display[
        "평균 혈당"
    ].round(2)

    return display


# ==========================================
# 18. 놓친 환자와 헛짚은 사람
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        f"🚨 놓친 환자 ({len(missed):,}명)"
    )

    st.write(
        "실제로 뇌졸중이 발생했지만 "
        "모델이 발생하지 않았다고 예측한 사람입니다."
    )

    if len(missed) > 0:

        st.dataframe(
            prepare_display(missed.head(20)),
            use_container_width=True,
            hide_index=True
        )

        if len(missed) > 20:
            st.caption(
                f"전체 {len(missed):,}명 중 "
                "나이가 많은 상위 20명만 표시했습니다."
            )

    else:
        st.success("놓친 환자가 없습니다.")


with col2:

    st.subheader(
        f"⚠️ 헛짚은 사람 ({len(false_alarm):,}명)"
    )

    st.write(
        "실제로 뇌졸중이 발생하지 않았지만 "
        "모델이 발생했다고 예측한 사람입니다."
    )

    if len(false_alarm) > 0:

        st.dataframe(
            prepare_display(false_alarm.head(20)),
            use_container_width=True,
            hide_index=True
        )

        if len(false_alarm) > 20:
            st.caption(
                f"전체 {len(false_alarm):,}명 중 "
                "나이가 많은 상위 20명만 표시했습니다."
            )

    else:
        st.success("헛짚은 사람이 없습니다.")


# ==========================================
# 19. 혼동행렬과 목록 비교
# ==========================================

st.divider()

st.subheader("목록과 혼동행렬의 일치 여부")

st.write(
    f"놓친 환자 목록: **{len(missed):,}명** "
    f"= 혼동행렬의 FN: **{expected_FN:,}명**"
)

st.write(
    f"헛짚은 사람 목록: **{len(false_alarm):,}명** "
    f"= 혼동행렬의 FP: **{expected_FP:,}명**"
)

if (
    len(missed) == expected_FN
    and len(false_alarm) == expected_FP
):
    st.success(
        "두 목록의 인원수가 혼동행렬의 FN, FP와 일치합니다."
    )
else:
    st.error(
        "목록과 혼동행렬의 인원수가 일치하지 않습니다."
    )


# ==========================================
# 20. 마무리
# ==========================================

st.divider()

st.caption(
    "양성 클래스는 stroke = 1이며, "
    "모든 평가 지표는 테스트 데이터를 기준으로 계산했습니다."
)

st.caption(
    "데이터 출처: "
    "https://github.com/greatsong/modudata"
)
