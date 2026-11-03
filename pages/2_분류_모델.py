
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ==========================================
# 1. 기본 설정
# ==========================================

st.title("🌳 분류 모델")
st.write(
    "건강 정보를 이용하여 뇌졸중 발생 여부를 예측하는 "
    "분류 모델을 만들고 성능을 비교합니다."
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURE_NAMES = {
    "나이": "age",
    "평균 혈당": "avg_glucose_level",
    "체질량지수": "bmi",
    "고혈압": "hypertension",
    "심장병": "heart_disease"
}

LABEL_NAMES = {
    value: key for key, value in FEATURE_NAMES.items()
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


# 데이터 정리
df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")
df = df.dropna(subset=["stroke"]).copy()
df["stroke"] = df["stroke"].astype(int)

df = df.sort_values("id").reset_index(drop=True)


# ==========================================
# 3. 입력 속성 선택
# ==========================================

st.header("1. 예측에 사용할 속성 선택")

st.write(
    "뇌졸중 예측에 사용할 속성을 선택하세요. "
    "처음에는 체질량지수를 제외한 네 가지 속성이 선택되어 있습니다."
)

selected_labels = st.multiselect(
    "입력 속성",
    options=list(FEATURE_NAMES.keys()),
    default=["나이", "평균 혈당", "고혈압", "심장병"]
)

if len(selected_labels) < 2:
    st.warning("예측에 사용할 속성을 두 개 이상 선택해 주세요.")
    st.stop()

selected_features = [
    FEATURE_NAMES[label]
    for label in selected_labels
]

st.write(
    "**선택한 속성:** "
    + ", ".join(selected_labels)
)


# ==========================================
# 4. 훈련 데이터와 테스트 데이터 분리
# ==========================================

st.header("2. 훈련 데이터와 테스트 데이터 분리")

st.write(
    "번호 순으로 정렬한 뒤 10명씩 묶고, "
    "각 묶음의 앞 3명을 테스트 데이터로 사용합니다. "
    "나머지 7명은 훈련 데이터로 사용합니다."
)

# 각 10명 묶음에서 앞 3명은 테스트 데이터
test_mask = df.index % 10 < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

st.write(f"훈련 데이터: **{len(train_df):,}명**")
st.write(f"테스트 데이터: **{len(test_df):,}명**")

if len(test_df) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(test_df):,}명입니다. "
        "데이터 전체 인원에 따라 요청한 1,533명과 달라질 수 있습니다."
    )

# 정답 데이터
y_train = train_df["stroke"]
y_test = test_df["stroke"]

# 입력 데이터
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

# 선택된 속성을 숫자로 변환
for column in selected_features:
    X_train[column] = pd.to_numeric(
        X_train[column], errors="coerce"
    )
    X_test[column] = pd.to_numeric(
        X_test[column], errors="coerce"
    )


# ==========================================
# 5. BMI 결측값 처리
# ==========================================

st.header("3. 결측값 처리")

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련 데이터에서 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    missing_train = X_train["bmi"].isna().sum()
    missing_test = X_test["bmi"].isna().sum()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.info(
        f"체질량지수의 빈 값을 훈련 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )

    st.caption(
        f"훈련 데이터 결측값: {missing_train}명 / "
        f"테스트 데이터 결측값: {missing_test}명"
    )

else:
    st.write(
        "체질량지수를 선택하지 않았으므로 "
        "BMI 결측값을 처리하지 않았습니다."
    )


# BMI 이외에 결측값이 있는지 확인
other_missing = [
    column
    for column in selected_features
    if X_train[column].isna().any()
    or X_test[column].isna().any()
]

if other_missing:
    st.error(
        "선택한 속성 중 BMI 이외에 결측값이 발견되었습니다: "
        + ", ".join(LABEL_NAMES[column] for column in other_missing)
    )
    st.stop()


# ==========================================
# 6. 모델 학습
# ==========================================

st.header("4. 분류 모델 학습")

# ------------------------------------------
# 로지스틱 회귀
# ------------------------------------------

# 훈련 데이터만 이용하여 표준화 기준 계산
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

log_model.fit(X_train_scaled, y_train)

# 예측
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


# ==========================================
# 7. 정확도 계산
# ==========================================

log_train_acc = accuracy_score(y_train, log_train_pred)
log_test_acc = accuracy_score(y_test, log_test_pred)

tree_train_acc = accuracy_score(y_train, tree_train_pred)
tree_test_acc = accuracy_score(y_test, tree_test_pred)


# ------------------------------------------
# 기준 모델
# ------------------------------------------

# 입력 속성을 하나도 사용하지 않고
# 훈련 데이터에서 더 많은 클래스를 예측하는 모델

majority_class = y_train.value_counts().idxmax()

baseline_train_pred = [majority_class] * len(y_train)
baseline_test_pred = [majority_class] * len(y_test)

baseline_train_acc = accuracy_score(
    y_train, baseline_train_pred
)

baseline_test_acc = accuracy_score(
    y_test, baseline_test_pred
)


# ==========================================
# 8. 정확도 카드
# ==========================================

st.header("5. 모델별 정확도 비교")

st.write(
    "테스트 데이터의 정확도를 크게 표시하고, "
    "훈련 데이터와 테스트 데이터의 정확도를 함께 비교합니다."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{log_test_acc:.2%}"
    )

    st.caption(
        f"훈련 데이터 정확도: {log_train_acc:.2%}\n\n"
        f"테스트 데이터 정확도: {log_test_acc:.2%}"
    )

with col2:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{tree_test_acc:.2%}"
    )

    st.caption(
        f"훈련 데이터 정확도: {tree_train_acc:.2%}\n\n"
        f"테스트 데이터 정확도: {tree_test_acc:.2%}"
    )

with col3:
    st.metric(
        label="입력 없이 다수 클래스로 예측하는 모델",
        value=f"{baseline_test_acc:.2%}"
    )

    st.caption(
        f"훈련 데이터 정확도: {baseline_train_acc:.2%}\n\n"
        f"테스트 데이터 정확도: {baseline_test_acc:.2%}"
    )


# ==========================================
# 9. 두 속성으로 예측 영역 시각화
# ==========================================

st.header("6. 두 속성에 따른 예측 영역")

st.write(
    "선택한 속성 중 두 개를 가로축과 세로축으로 지정합니다. "
    "점의 색은 실제 뇌졸중 발생 여부를 나타냅니다."
)

col1, col2 = st.columns(2)

with col1:
    x_label = st.selectbox(
        "가로축 속성",
        selected_labels,
        index=0
    )

with col2:
    y_options = [
        label for label in selected_labels
        if label != x_label
    ]

    y_label = st.selectbox(
        "세로축 속성",
        y_options,
        index=0
    )

x_feature = FEATURE_NAMES[x_label]
y_feature = FEATURE_NAMES[y_label]


# ------------------------------------------
# 나머지 속성은 테스트 데이터 중앙값으로 고정
# ------------------------------------------

fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = X_test[feature].median()

st.markdown("#### 나머지 속성의 고정값")

if fixed_values:
    fixed_table = pd.DataFrame({
        "속성": [
            LABEL_NAMES[column]
            for column in fixed_values
        ],
        "고정한 값": [
            round(value, 3)
            for value in fixed_values.values()
        ]
    })

    st.dataframe(
        fixed_table,
        hide_index=True,
        use_container_width=True
    )

else:
    st.write("선택한 속성이 두 개뿐이므로 고정할 속성이 없습니다.")


# ------------------------------------------
# 예측 영역을 계산할 격자 생성
# ------------------------------------------

x_min = X_test[x_feature].min()
x_max = X_test[x_feature].max()

y_min = X_test[y_feature].min()
y_max = X_test[y_feature].max()

# 같은 값만 존재할 경우 범위 확장
if x_min == x_max:
    x_min -= 1
    x_max += 1

if y_min == y_max:
    y_min -= 1
    y_max += 1

grid_size = 100

x_values = pd.Series([
    x_min + (x_max - x_min) * i / (grid_size - 1)
    for i in range(grid_size)
])

y_values = pd.Series([
    y_min + (y_max - y_min) * i / (grid_size - 1)
    for i in range(grid_size)
])

grid_rows = []

for y_value in y_values:
    for x_value in x_values:

        row = {}

        for feature in selected_features:

            if feature == x_feature:
                row[feature] = x_value

            elif feature == y_feature:
                row[feature] = y_value

            else:
                row[feature] = fixed_values[feature]

        grid_rows.append(row)

grid_df = pd.DataFrame(
    grid_rows,
    columns=selected_features
)


# ------------------------------------------
# 두 모델의 예측 영역 계산
# ------------------------------------------

# 로지스틱 회귀의 뇌졸중 확률
grid_scaled = scaler.transform(grid_df)

grid_probability = log_model.predict_proba(
    grid_scaled
)[:, 1]

probability_grid = pd.Series(
    grid_probability
).to_numpy().reshape(grid_size, grid_size)


# 의사결정트리의 예측 결과
grid_tree_pred = tree_model.predict(grid_df)

tree_grid = pd.Series(
    grid_tree_pred
).to_numpy().reshape(grid_size, grid_size)


# ------------------------------------------
# 그래프 작성
# ------------------------------------------

fig = go.Figure()


# 의사결정트리 예측 영역
fig.add_trace(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=tree_grid,
        colorscale=[
            [0, "rgba(66, 133, 244, 0.35)"],
            [1, "rgba(234, 67, 53, 0.35)"]
        ],
        zmin=0,
        zmax=1,
        showscale=False,
        opacity=0.45,
        hoverinfo="skip",
        name="의사결정트리 예측 영역"
    )
)


# 로지스틱 회귀의 0.5 결정 경계
prob_min = grid_probability.min()
prob_max = grid_probability.max()

boundary_inside = prob_min <= 0.5 <= prob_max

if boundary_inside:

    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=probability_grid,
            contours=dict(
                start=0.5,
                end=0.5,
                size=0.01,
                coloring="lines"
            ),
            line=dict(
                color="black",
                width=3,
                dash="dash"
            ),
            showscale=False,
            hoverinfo="skip",
            name="로지스틱 회귀 0.5 결정 경계"
        )
    )


# ------------------------------------------
# 테스트 데이터 실제값 표시
# ------------------------------------------

for actual_value, group_name, marker_color in [
    (0, "실제 뇌졸중 없음", "blue"),
    (1, "실제 뇌졸중 있음", "red")
]:

    mask = y_test == actual_value

    fig.add_trace(
        go.Scatter(
            x=X_test.loc[mask, x_feature],
            y=X_test.loc[mask, y_feature],
            mode="markers",
            name=group_name,
            marker=dict(
                color=marker_color,
                size=7,
                opacity=0.7,
                line=dict(width=0.5, color="white")
            )
        )
    )


fig.update_layout(
    title="테스트 데이터와 두 분류 모델의 예측 영역",
    xaxis_title=x_label,
    yaxis_title=y_label,
    legend_title="실제 뇌졸중 여부",
    height=650
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ------------------------------------------
# 결정 경계 안내
# ------------------------------------------

st.markdown("#### 로지스틱 회귀 결정 경계")

if boundary_inside:
    st.write(
        "검은색 점선은 로지스틱 회귀 모델이 "
        "뇌졸중 확률을 0.5로 판단하는 경계입니다."
    )

else:
    st.write(
        "현재 테스트 데이터의 두 축 범위 안에서는 "
        "확률 0.5의 결정 경계가 나타나지 않습니다. "
        "결정 경계가 그림 밖에 있거나 해당 범위에 존재하지 않습니다."
    )

st.markdown("#### 의사결정트리 예측 영역")

st.write(
    "옅은 파란색 영역은 뇌졸중이 없다고 예측하는 영역이고, "
    "옅은 빨간색 영역은 뇌졸중이 있다고 예측하는 영역입니다."
)


# ==========================================
# 10. 의사결정트리 질문 구조
# ==========================================

st.header("7. 의사결정트리의 질문 구조")

st.write(
    "의사결정트리가 훈련 데이터에 어떤 질문을 던졌는지 "
    "가지가 갈라지는 구조로 살펴봅니다."
)

tree_structure = tree_model.tree_


# 훈련 데이터가 각 마디에 도달한 횟수 계산
train_path = tree_model.decision_path(X_train)

node_counts = [
    int(value)
    for value in train_path.sum(axis=0).tolist()[0]
]

positive_counts = [
    int(value)
    for value in train_path.T.dot(y_train.to_numpy()).tolist()
]


# ------------------------------------------
# DOT 문자열 생성
# ------------------------------------------

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white"];',
    'node [shape=box, style="rounded,filled", '
    'fontname="Malgun Gothic", fontsize=11];',
    'edge [fontname="Malgun Gothic", fontsize=10];'
]


leaf_nodes = []
asked_features = []


def add_tree_node(node_id):

    samples = node_counts[node_id]
    positives = positive_counts[node_id]

    negative = samples - positives

    ratio = (
        positives / samples * 100
        if samples > 0 else 0
    )

    is_leaf = (
        tree_structure.children_left[node_id]
        == tree_structure.children_right[node_id]
    )

    if is_leaf:

        leaf_nodes.append(node_id)

        # 다수 클래스 기준으로 최종 답 결정
        predicted_class = 1 if positives > negative else 0

        if predicted_class == 1:
            answer = "뇌졸중 있음"
            fill_color = "#FCE4EC"
        else:
            answer = "뇌졸중 아님"
            fill_color = "#E3F2FD"

        label = (
            f"최종 답: {answer}\\n"
            f"훈련 인원: {samples}명\\n"
            f"실제 뇌졸중: {positives}명\\n"
            f"뇌졸중 비율: {ratio:.1f}%"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", '
            f'fillcolor="{fill_color}"];'
        )

    else:

        feature_index = tree_structure.feature[node_id]
        feature = selected_features[feature_index]

        feature_label = LABEL_NAMES[feature]

        threshold = tree_structure.threshold[node_id]

        if feature_label not in asked_features:
            asked_features.append(feature_label)

        label = (
            f"질문: {feature_label} ≤ {threshold:.2f}?\\n"
            f"훈련 인원: {samples}명\\n"
            f"실제 뇌졸중: {positives}명\\n"
            f"뇌졸중 비율: {ratio:.1f}%"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", '
            'fillcolor="#F3F4F6"];'
        )

        left_child = tree_structure.children_left[node_id]
        right_child = tree_structure.children_right[node_id]

        dot_lines.append(
            f'{node_id} -> {left_child} [label="예"];'
        )

        dot_lines.append(
            f'{node_id} -> {right_child} [label="아니요"];'
        )

        add_tree_node(left_child)
        add_tree_node(right_child)


add_tree_node(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)


# ------------------------------------------
# 의사결정트리 시각화
# ------------------------------------------

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# ==========================================
# 11. 의사결정트리 분석 결과
# ==========================================

st.subheader("의사결정트리 구조 분석")

total_leaves = len(leaf_nodes)

negative_leaves = 0

for node_id in leaf_nodes:

    positive = positive_counts[node_id]
    negative = node_counts[node_id] - positive

    if positive <= negative:
        negative_leaves += 1


st.write(
    f"**① 최종 답을 내리는 마디:** "
    f"{total_leaves}개"
)

st.write(
    f"**② 그중 뇌졸중이 아니라고 답하는 마디:** "
    f"{negative_leaves}개"
)

st.write(
    "**③ 실제로 질문에 사용한 속성:** "
    + (
        ", ".join(asked_features)
        if asked_features
        else "질문에 사용한 속성이 없습니다."
    )
)


# ==========================================
# 12. 마무리
# ==========================================

st.divider()

st.caption(
    "분석 데이터: "
    "https://github.com/greatsong/modudata"
)

st.caption(
    "훈련 데이터만으로 모델을 학습하고 "
    "테스트 데이터는 최종 성능 평가에 사용했습니다."
)
