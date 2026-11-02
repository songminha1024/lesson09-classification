
import streamlit as st
import pandas as pd
import plotly.express as px


# -----------------------------
# 기본 설정
# -----------------------------
st.title("🔎 데이터 탐색")
st.write(
    "뇌졸중 데이터의 분포와 여러 건강 요인에 따른 "
    "뇌졸중 발생 비율을 탐색해 봅시다."
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.write(error)
    st.stop()


# 숫자형 데이터 변환
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["avg_glucose_level"] = pd.to_numeric(
    df["avg_glucose_level"], errors="coerce"
)
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")


# -----------------------------
# 1. 나이와 평균 혈당 분포
# -----------------------------
st.header("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        }
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_age, use_container_width=True)


with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        }
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_glucose, use_container_width=True)


# -----------------------------
# 2. 뇌졸중 발생 여부에 따른 비교
# -----------------------------
st.header("2. 뇌졸중 발생 여부에 따른 비교")

st.write(
    "뇌졸중을 겪은 사람과 겪지 않은 사람의 "
    "나이와 평균 혈당을 상자그림으로 비교합니다."
)

df_compare = df[df["stroke"].isin([0, 1])].copy()

df_compare["뇌졸중 여부"] = df_compare["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

col1, col2 = st.columns(2)

with col1:
    fig_age_box = px.box(
        df_compare,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "age": "나이",
            "뇌졸중 여부": "그룹"
        },
        category_orders={
            "뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]
        }
    )

    st.plotly_chart(fig_age_box, use_container_width=True)


with col2:
    fig_glucose_box = px.box(
        df_compare,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "avg_glucose_level": "평균 혈당",
            "뇌졸중 여부": "그룹"
        },
        category_orders={
            "뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]
        }
    )

    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 두 그룹의 평균값
mean_table = (
    df_compare.groupby("뇌졸중 여부")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .reset_index()
)

mean_table = mean_table.rename(columns={
    "뇌졸중 여부": "구분",
    "age": "평균 나이",
    "avg_glucose_level": "평균 혈당"
})

mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.subheader("두 그룹의 평균값 비교")
st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# 3. 고혈압과 심장병에 따른 뇌졸중 비율
# -----------------------------
st.header("3. 질병 여부에 따른 뇌졸중 비율")

st.write(
    "고혈압과 심장병이 있는 집단과 없는 집단의 "
    "뇌졸중 발생 비율을 비교합니다."
)


def make_rate_table(data, column):
    temp = data[data[column].isin([0, 1])].copy()

    result = (
        temp.groupby(column)
        .agg(
            전체_인원=("stroke", "count"),
            뇌졸중_인원=("stroke", "sum")
        )
        .reset_index()
    )

    result["뇌졸중_비율"] = (
        result["뇌졸중_인원"] / result["전체_인원"] * 100
    )

    result["구분"] = result[column].map({
        0: "없음",
        1: "있음"
    })

    return result


df_valid_stroke = df[df["stroke"].isin([0, 1])].copy()

col1, col2 = st.columns(2)

with col1:
    hypertension_table = make_rate_table(
        df_valid_stroke,
        "hypertension"
    )

    fig_hypertension = px.bar(
        hypertension_table,
        x="구분",
        y="뇌졸중_비율",
        text=hypertension_table["뇌졸중_비율"].round(2),
        title="고혈압 여부에 따른 뇌졸중 비율",
        labels={
            "구분": "고혈압 여부",
            "뇌졸중_비율": "뇌졸중 비율 (%)"
        }
    )

    fig_hypertension.update_traces(texttemplate="%{text}%")
    st.plotly_chart(fig_hypertension, use_container_width=True)


with col2:
    heart_table = make_rate_table(
        df_valid_stroke,
        "heart_disease"
    )

    fig_heart = px.bar(
        heart_table,
        x="구분",
        y="뇌졸중_비율",
        text=heart_table["뇌졸중_비율"].round(2),
        title="심장병 여부에 따른 뇌졸중 비율",
        labels={
            "구분": "심장병 여부",
            "뇌졸중_비율": "뇌졸중 비율 (%)"
        }
    )

    fig_heart.update_traces(texttemplate="%{text}%")
    st.plotly_chart(fig_heart, use_container_width=True)


# -----------------------------
# 4. BMI 결측값과 뇌졸중 비율
# -----------------------------
st.header("4. BMI 빈 값과 뇌졸중의 관계")

st.write(
    "BMI 값이 비어 있는 사람들을 따로 구분하여 "
    "전체 데이터의 뇌졸중 비율과 비교합니다."
)

missing_bmi = df_valid_stroke[df_valid_stroke["bmi"].isna()]

total_stroke_ratio = (
    df_valid_stroke["stroke"].mean() * 100
    if len(df_valid_stroke) > 0 else 0
)

missing_bmi_ratio = (
    missing_bmi["stroke"].mean() * 100
    if len(missing_bmi) > 0 else 0
)

bmi_table = pd.DataFrame({
    "구분": ["BMI 빈 값인 사람", "전체 사람"],
    "사람 수": [
        len(missing_bmi),
        len(df_valid_stroke)
    ],
    "뇌졸중 발생 인원": [
        int(missing_bmi["stroke"].sum()),
        int(df_valid_stroke["stroke"].sum())
    ],
    "뇌졸중 비율 (%)": [
        round(missing_bmi_ratio, 2),
        round(total_stroke_ratio, 2)
    ]
})

st.dataframe(
    bmi_table,
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# 5. 흡연 상태별 사람 수
# -----------------------------
st.header("5. 흡연 상태별 사람 수")

st.write(
    "흡연 상태의 값마다 해당하는 사람 수를 계산합니다."
)

smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

smoking_table["흡연 상태"] = smoking_table[
    "흡연 상태"
].fillna("빈 값")

st.dataframe(
    smoking_table,
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# 마무리
# -----------------------------
st.divider()

st.caption(
    "데이터 출처: "
    "https://github.com/greatsong/modudata"
)
