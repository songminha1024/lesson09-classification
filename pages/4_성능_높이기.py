import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기",
    page_icon="⚖️",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
    "bmi": "체질량지수"
}

LABEL_TO_COLUMN = {
    value: key for key, value in FEATURE_LABELS.items()
}

DEFAULT_FEATURES = [
    "나이",
    "평균 혈당",
    "고혈압",
    "심장병"
]


# --------------------------------------------------
# 제목
# --------------------------------------------------

st.title("⚖️ 성능 높이기")
st.write(
    "뇌졸중 환자는 전체 데이터에서 차지하는 비율이 낮습니다. "
    "이 때문에 모델이 뇌졸중이 아닌 사람을 주로 예측하는 문제가 생길 수 있습니다."
)

st.write(
    "이번 페이지에서는 두 모델에 **가중치를 적용하기 전과 후**를 비교합니다. "
    "가중치를 조정하면 뇌졸중 환자를 더 잘 찾아낼 수 있는지 확인해 봅시다."
)


# --------------------------------------------------
# 입력 속성 선택
# --------------------------------------------------

st.subheader("1. 입력으로 사용할 속성 선택")

selected_features = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요.",
    options=list(FEATURE_LABELS.values()),
    default=DEFAULT_FEATURES
)

if len(selected_features) < 2:
    st.warning("산점도와 경계선을 표시하려면 속성을 2개 이상 선택해야 합니다.")
    st.stop()

selected_columns = [
    LABEL_TO_COLUMN[feature]
    for feature in selected_features
]

st.caption(
    "선택한 속성: " + ", ".join(selected_features)
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    data["stroke"] = pd.to_numeric(
        data["stroke"],
        errors="coerce"
    )

    data = data.dropna(subset=["stroke"])
    data["stroke"] = data["stroke"].astype(int)

    data = data.sort_values("id").reset_index(drop=True)

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 훈련용 / 시험용 데이터 분리
# 채점표 페이지와 동일한 방식
# --------------------------------------------------

test_mask = df.index % 10 < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

X_train = train_df[selected_columns].copy()
X_test = test_df[selected_columns].copy()

y_train = train_df["stroke"].copy()
y_test = test_df["stroke"].copy()


# --------------------------------------------------
# 결측치 처리
# BMI는 훈련용 중앙값으로 채우기
# --------------------------------------------------

if "bmi" in selected_columns:

    bmi_median = X_train["bmi"].median()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.info(
        f"체질량지수의 빈 값은 훈련용 데이터의 중앙값 "
        f"({bmi_median:.2f})으로 채웠습니다."
    )


# 나머지 선택 속성의 결측치 확인
if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
    st.error(
        "선택한 속성에 BMI 이외의 빈 값이 있습니다. "
        "해당 속성의 결측치를 확인해 주세요."
    )
    st.stop()


# 숫자형 데이터로 변환
try:
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

except Exception:
    st.error("선택한 속성에 숫자로 변환할 수 없는 값이 있습니다.")
    st.stop()


# --------------------------------------------------
# 데이터 정보
# --------------------------------------------------

st.divider()
st.subheader("2. 데이터 분리 결과")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(df):,}명")
col2.metric("훈련용 데이터", f"{len(train_df):,}명")
col3.metric("시험용 데이터", f"{len(test_df):,}명")

actual_positive = int(y_test.sum())

st.write(
    f"시험용 데이터 {len(test_df):,}명 중 실제 뇌졸중 환자는 "
    f"**{actual_positive:,}명**입니다."
)


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------

if y_train.nunique() < 2:
    st.error("훈련용 데이터에 두 종류의 결과가 모두 있어야 모델을 학습할 수 있습니다.")
    st.stop()


# 로지스틱 회귀용 표준화
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 가중치 조정 전 모델
logistic_before = LogisticRegression(
    max_iter=2000,
    random_state=42
)

tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)


# 가중치 조정 후 모델
logistic_after = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)


# 학습
logistic_before.fit(X_train_scaled, y_train)
logistic_after.fit(X_train_scaled, y_train)

tree_before.fit(X_train, y_train)
tree_after.fit(X_train, y_train)


# 시험용 데이터 예측
predictions = {
    "logistic_before": logistic_before.predict(X_test_scaled),
    "logistic_after": logistic_after.predict(X_test_scaled),
    "tree_before": tree_before.predict(X_test),
    "tree_after": tree_after.predict(X_test)
}


# --------------------------------------------------
# 혼동 행렬과 성능 지표 계산
# --------------------------------------------------

def calculate_result(y_true, y_pred):

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1 점수": f1
    }


results = {
    "logistic_before": calculate_result(
        y_test, predictions["logistic_before"]
    ),
    "logistic_after": calculate_result(
        y_test, predictions["logistic_after"]
    ),
    "tree_before": calculate_result(
        y_test, predictions["tree_before"]
    ),
    "tree_after": calculate_result(
        y_test, predictions["tree_after"]
    )
}


# --------------------------------------------------
# 표 출력 함수
# --------------------------------------------------

def format_metric(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


def make_comparison_table(before, after):

    rows = [
        ["실제 음성 / 예측 음성 (TN)", before["TN"], after["TN"]],
        ["실제 음성 / 예측 양성 (FP)", before["FP"], after["FP"]],
        ["실제 양성 / 예측 음성 (FN)", before["FN"], after["FN"]],
        ["실제 양성 / 예측 양성 (TP)", before["TP"], after["TP"]],
        ["정확도", format_metric(before["정확도"]), format_metric(after["정확도"])],
        ["재현율", format_metric(before["재현율"]), format_metric(after["재현율"])],
        ["정밀도", format_metric(before["정밀도"]), format_metric(after["정밀도"])],
        ["F1 점수", format_metric(before["F1 점수"]), format_metric(after["F1 점수"])]
    ]

    return pd.DataFrame(
        rows,
        columns=["구분", "가중치 조정 전", "가중치 조정 후"]
    )


# --------------------------------------------------
# 모델별 결과 비교
# --------------------------------------------------

st.divider()
st.subheader("3. 가중치 조정 전후 성능 비교")

st.caption(
    "TN: 정상인을 정상으로 예측한 수 / "
    "FP: 정상인을 환자로 잘못 예측한 수 / "
    "FN: 환자를 놓친 수 / "
    "TP: 환자를 환자로 찾아낸 수"
)

st.caption(
    "재현율은 실제 환자 중 찾아낸 비율, "
    "정밀도는 환자라고 예측한 사람 중 실제 환자의 비율입니다."
)


# 로지스틱 회귀
st.markdown("### 로지스틱 회귀(확률로 답하는 모델)")

logistic_table = make_comparison_table(
    results["logistic_before"],
    results["logistic_after"]
)

st.dataframe(
    logistic_table,
    use_container_width=True,
    hide_index=True
)

logistic_before_result = results["logistic_before"]
logistic_after_result = results["logistic_after"]

st.write(
    f"찾아낸 환자 수: **{logistic_before_result['TP']}명 → "
    f"{logistic_after_result['TP']}명** | "
    f"놓친 환자 수: **{logistic_before_result['FN']}명 → "
    f"{logistic_after_result['FN']}명** | "
    f"헛짚은 사람 수: **{logistic_before_result['FP']}명 → "
    f"{logistic_after_result['FP']}명**"
)


st.divider()


# 의사결정트리
st.markdown("### 의사결정트리(질문으로 답하는 모델)")

tree_table = make_comparison_table(
    results["tree_before"],
    results["tree_after"]
)

st.dataframe(
    tree_table,
    use_container_width=True,
    hide_index=True
)

tree_before_result = results["tree_before"]
tree_after_result = results["tree_after"]

st.write(
    f"찾아낸 환자 수: **{tree_before_result['TP']}명 → "
    f"{tree_after_result['TP']}명** | "
    f"놓친 환자 수: **{tree_before_result['FN']}명 → "
    f"{tree_after_result['FN']}명** | "
    f"헛짚은 사람 수: **{tree_before_result['FP']}명 → "
    f"{tree_after_result['FP']}명**"
)


# --------------------------------------------------
# 산점도에 사용할 실제 뇌졸중 여부
# --------------------------------------------------

st.divider()
st.subheader("4. 실제 환자 분포와 0.5 경계선 비교")

st.write(
    "점의 색은 시험용 데이터의 실제 뇌졸중 여부를 나타냅니다. "
    "경계선은 모델이 뇌졸중일 확률을 0.5로 판단하는 지점입니다."
)

st.caption(
    "선택한 속성이 3개를 초과하면 산점도에는 앞의 3개 속성을 사용합니다. "
    "나머지 속성은 시험용 데이터의 중앙값으로 고정합니다."
)


# 실제 여부에 따른 색상
stroke_colors = {
    0: "#2878B5",
    1: "#E45756"
}

stroke_names = {
    0: "실제 뇌졸중 없음",
    1: "실제 뇌졸중 있음"
}


# --------------------------------------------------
# 2차원 산점도 및 경계선
# --------------------------------------------------

def make_2d_figure(
    model_before,
    model_after,
    model_type,
    title
):

    x_column = selected_columns[0]
    y_column = selected_columns[1]

    x_label = FEATURE_LABELS[x_column]
    y_label = FEATURE_LABELS[y_column]

    # 나머지 속성은 시험용 중앙값으로 고정
    fixed_values = X_test.median()

    x_min = float(X_test[x_column].min())
    x_max = float(X_test[x_column].max())
    y_min = float(X_test[y_column].min())
    y_max = float(X_test[y_column].max())

    x_padding = (x_max - x_min) * 0.08
    y_padding = (y_max - y_min) * 0.08

    if x_padding == 0:
        x_padding = 1

    if y_padding == 0:
        y_padding = 1

    x_range = [
        x_min - x_padding,
        x_max + x_padding
    ]

    y_range = [
        y_min - y_padding,
        y_max + y_padding
    ]

    x_values = pd.Series(
        [
            x_range[0] + i * (x_range[1] - x_range[0]) / 99
            for i in range(100)
        ]
    )

    y_values = pd.Series(
        [
            y_range[0] + i * (y_range[1] - y_range[0]) / 99
            for i in range(100)
        ]
    )

    xx, yy = pd.Series(x_values), pd.Series(y_values)

    # 2차원 예측용 격자 생성
    grid_x, grid_y = [], []

    for y_value in y_values:
        for x_value in x_values:
            grid_x.append(x_value)
            grid_y.append(y_value)

    grid = pd.DataFrame({
        column: [fixed_values[column]] * len(grid_x)
        for column in selected_columns
    })

    grid[x_column] = grid_x
    grid[y_column] = grid_y

    # 모델별 확률 계산
    if model_type == "logistic":

        grid_scaled = scaler.transform(grid)
        probability_before = model_before.predict_proba(grid_scaled)[:, 1]
        probability_after = model_after.predict_proba(grid_scaled)[:, 1]

    else:

        probability_before = model_before.predict_proba(grid)[:, 1]
        probability_after = model_after.predict_proba(grid)[:, 1]

    z_before = probability_before.reshape(100, 100)
    z_after = probability_after.reshape(100, 100)

    fig = go.Figure()

    # 실제 데이터 산점도
    for actual_value in [0, 1]:

        mask = y_test == actual_value

        fig.add_trace(
            go.Scatter(
                x=X_test.loc[mask, x_column],
                y=X_test.loc[mask, y_column],
                mode="markers",
                name=stroke_names[actual_value],
                marker=dict(
                    color=stroke_colors[actual_value],
                    size=7,
                    opacity=0.65
                )
            )
        )

    # 가중치 조정 전 경계선
    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=z_before,
            name="가중치 조정 전",
            showscale=False,
            contours=dict(
                start=0.5,
                end=0.5,
                size=0.01,
                coloring="lines"
            ),
            line=dict(
                color="#F39C12",
                width=4
            ),
            hoverinfo="skip"
        )
    )

    # 가중치 조정 후 경계선
    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=z_after,
            name="가중치 조정 후",
            showscale=False,
            contours=dict(
                start=0.5,
                end=0.5,
                size=0.01,
                coloring="lines"
            ),
            line=dict(
                color="#9B59B6",
                width=4
            ),
            hoverinfo="skip"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        legend_title="구분",
        hovermode="closest"
    )

    # 경계선 존재 여부
    before_exists = (
        probability_before.min() <= 0.5
        and probability_before.max() >= 0.5
    )

    after_exists = (
        probability_after.min() <= 0.5
        and probability_after.max() >= 0.5
    )

    return fig, before_exists, after_exists


# --------------------------------------------------
# 3차원 산점도 및 경계면
# --------------------------------------------------

def make_3d_figure(
    model_before,
    model_after,
    model_type,
    title
):

    x_column = selected_columns[0]
    y_column = selected_columns[1]
    z_column = selected_columns[2]

    x_label = FEATURE_LABELS[x_column]
    y_label = FEATURE_LABELS[y_column]
    z_label = FEATURE_LABELS[z_column]

    fixed_values = X_test.median()

    # 각 축 범위
    ranges = []

    for column in [x_column, y_column, z_column]:

        minimum = float(X_test[column].min())
        maximum = float(X_test[column].max())

        padding = (maximum - minimum) * 0.08

        if padding == 0:
            padding = 1

        ranges.append([
            minimum - padding,
            maximum + padding
        ])

    x_values = [
        ranges[0][0] + i * (ranges[0][1] - ranges[0][0]) / 17
        for i in range(18)
    ]

    y_values = [
        ranges[1][0] + i * (ranges[1][1] - ranges[1][0]) / 17
        for i in range(18)
    ]

    z_values = [
        ranges[2][0] + i * (ranges[2][1] - ranges[2][0]) / 17
        for i in range(18)
    ]

    grid_x = []
    grid_y = []
    grid_z = []

    for z_value in z_values:
        for y_value in y_values:
            for x_value in x_values:
                grid_x.append(x_value)
                grid_y.append(y_value)
                grid_z.append(z_value)

    grid = pd.DataFrame({
        column: [fixed_values[column]] * len(grid_x)
        for column in selected_columns
    })

    grid[x_column] = grid_x
    grid[y_column] = grid_y
    grid[z_column] = grid_z

    if model_type == "logistic":

        grid_scaled = scaler.transform(grid)
        probability_before = model_before.predict_proba(grid_scaled)[:, 1]
        probability_after = model_after.predict_proba(grid_scaled)[:, 1]

    else:

        probability_before = model_before.predict_proba(grid)[:, 1]
        probability_after = model_after.predict_proba(grid)[:, 1]

    fig = go.Figure()

    # 실제 데이터 산점도
    for actual_value in [0, 1]:

        mask = y_test == actual_value

        fig.add_trace(
            go.Scatter3d(
                x=X_test.loc[mask, x_column],
                y=X_test.loc[mask, y_column],
                z=X_test.loc[mask, z_column],
                mode="markers",
                name=stroke_names[actual_value],
                marker=dict(
                    color=stroke_colors[actual_value],
                    size=4,
                    opacity=0.7
                )
            )
        )

    # 가중치 조정 전 0.5 경계면
    fig.add_trace(
        go.Isosurface(
            x=grid_x,
            y=grid_y,
            z=grid_z,
            value=probability_before,
            isomin=0.499,
            isomax=0.501,
            surface_count=1,
            opacity=0.45,
            colorscale=[
                [0, "#F39C12"],
                [1, "#F39C12"]
            ],
            showscale=False,
            name="가중치 조정 전 경계면",
            caps=dict(
                x_show=False,
                y_show=False,
                z_show=False
            )
        )
    )

    # 가중치 조정 후 0.5 경계면
    fig.add_trace(
        go.Isosurface(
            x=grid_x,
            y=grid_y,
            z=grid_z,
            value=probability_after,
            isomin=0.499,
            isomax=0.501,
            surface_count=1,
            opacity=0.45,
            colorscale=[
                [0, "#9B59B6"],
                [1, "#9B59B6"]
            ],
            showscale=False,
            name="가중치 조정 후 경계면",
            caps=dict(
                x_show=False,
                y_show=False,
                z_show=False
            )
        )
    )

    fig.update_layout(
        title=title,
        height=700,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label
        ),
        legend_title="구분"
    )

    before_exists = (
        probability_before.min() <= 0.5
        and probability_before.max() >= 0.5
    )

    after_exists = (
        probability_after.min() <= 0.5
        and probability_after.max() >= 0.5
    )

    return fig, before_exists, after_exists


# --------------------------------------------------
# 그래프 출력
# --------------------------------------------------

if len(selected_features) == 2:

    st.write(
        "가로축과 세로축은 선택한 첫 두 속성입니다. "
        "다른 속성을 추가로 선택했다면 해당 속성은 중앙값으로 고정됩니다."
    )

    logistic_fig, logistic_before_exists, logistic_after_exists = make_2d_figure(
        logistic_before,
        logistic_after,
        "logistic",
        "로지스틱 회귀: 가중치 조정 전후"
    )

    tree_fig, tree_before_exists, tree_after_exists = make_2d_figure(
        tree_before,
        tree_after,
        "tree",
        "의사결정트리: 가중치 조정 전후"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 로지스틱 회귀(확률로 답하는 모델)")
        st.plotly_chart(logistic_fig, use_container_width=True)

        if not logistic_before_exists:
            st.caption("가중치 조정 전 0.5 경계선은 그림 범위 밖에 있습니다.")

        if not logistic_after_exists:
            st.caption("가중치 조정 후 0.5 경계선은 그림 범위 밖에 있습니다.")

    with col2:
        st.markdown("#### 의사결정트리(질문으로 답하는 모델)")
        st.plotly_chart(tree_fig, use_container_width=True)

        if not tree_before_exists:
            st.caption("가중치 조정 전 0.5 경계선은 그림 범위 밖에 있습니다.")

        if not tree_after_exists:
            st.caption("가중치 조정 후 0.5 경계선은 그림 범위 밖에 있습니다.")


else:

    st.write(
        "3차원 그래프의 가로축, 세로축, 높이축은 "
        "선택한 속성 중 앞의 세 가지입니다. "
        "마우스로 그래프를 움직여 여러 방향에서 살펴볼 수 있습니다."
    )

    logistic_fig, logistic_before_exists, logistic_after_exists = make_3d_figure(
        logistic_before,
        logistic_after,
        "logistic",
        "로지스틱 회귀: 가중치 조정 전후"
    )

    tree_fig, tree_before_exists, tree_after_exists = make_3d_figure(
        tree_before,
        tree_after,
        "tree",
        "의사결정트리: 가중치 조정 전후"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 로지스틱 회귀(확률로 답하는 모델)")
        st.plotly_chart(logistic_fig, use_container_width=True)

        if not logistic_before_exists:
            st.caption("가중치 조정 전 0.5 경계면은 그림 범위 밖에 있습니다.")

        if not logistic_after_exists:
            st.caption("가중치 조정 후 0.5 경계면은 그림 범위 밖에 있습니다.")

    with col2:
        st.markdown("#### 의사결정트리(질문으로 답하는 모델)")
        st.plotly_chart(tree_fig, use_container_width=True)

        if not tree_before_exists:
            st.caption("가중치 조정 전 0.5 경계면은 그림 범위 밖에 있습니다.")

        if not tree_after_exists:
            st.caption("가중치 조정 후 0.5 경계면은 그림 범위 밖에 있습니다.")


# --------------------------------------------------
# 결과 해석 안내
# --------------------------------------------------

st.divider()
st.subheader("5. 결과 해석하기")

st.write(
    """
    - **찾아낸 환자 수(TP)**가 증가하면 실제 뇌졸중 환자를 더 많이 발견했다는 뜻입니다.
    - **놓친 환자 수(FN)**가 감소하면 실제 환자를 정상으로 잘못 판단한 경우가 줄었다는 뜻입니다.
    - **헛짚은 사람 수(FP)**가 증가하면 실제로는 뇌졸중이 아닌 사람을 환자로 예측한 경우가 늘었다는 뜻입니다.
    - 가중치를 조정한 모델은 소수 집단인 뇌졸중 환자의 중요도를 높여 학습합니다. 따라서 환자를 더 많이 찾아낼 수 있지만, 정상인을 환자로 잘못 판단하는 경우도 늘어날 수 있습니다.
    """
)

st.warning(
    "이 결과는 학습용 데이터와 시험용 데이터로 수행한 교육용 실습입니다. "
    "실제 환자의 진단이나 치료 판단에 사용할 수 있는 의료 모델이 아닙니다."
)
