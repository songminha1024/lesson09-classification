
import streamlit as st
import pandas as pd
import json
from pathlib import Path

# -------------------------
# 기본 설정
# -------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

SAVE_DIR = Path("내_기록")
SAVE_FILE = SAVE_DIR / "stroke_data_notes.json"


# -------------------------
# 기록 불러오기
# -------------------------
def load_notes():
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return {"meanings": {}, "source": ""}

    return {"meanings": {}, "source": ""}


# -------------------------
# 기록 저장하기
# -------------------------
def save_notes(meanings, source):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "meanings": meanings,
        "source": source
    }

    with open(SAVE_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# -------------------------
# 데이터 불러오기
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.write(error)
    st.stop()


# -------------------------
# 제목 및 소개
# -------------------------
st.title("🧠 뇌졸중 예측 실습실")

st.subheader("데이터 살펴보기")

st.write(
    "이 데이터는 여러 사람의 건강 및 생활 관련 정보를 담고 있으며, "
    "각 사람에게 뇌졸중이 발생했는지를 나타내는 stroke 열을 포함합니다. "
    "데이터의 크기와 각 열의 특징을 살펴보겠습니다."
)

st.info(
    "stroke 값이 1이면 뇌졸중이 발생한 사람, "
    "0이면 발생하지 않은 사람을 의미합니다."
)


# -------------------------
# 데이터 계산
# -------------------------
total_people = len(df)
column_count = len(df.columns)

stroke_count = (
    pd.to_numeric(df["stroke"], errors="coerce") == 1
).sum()

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0 else 0
)


# -------------------------
# 숫자 카드
# -------------------------
st.markdown("### 📊 데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 사람 수", f"{total_people:,}명")
col2.metric("열 개수", f"{column_count}개")
col3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
col4.metric("stroke 비율", f"{stroke_ratio:.2f}%")


# -------------------------
# 열 정보 표
# -------------------------
st.markdown("### 📋 열 이름과 데이터 특징")

st.write(
    "우리말 뜻은 교재를 참고하여 직접 입력할 수 있습니다."
)

saved_notes = load_notes()
saved_meanings = saved_notes.get("meanings", {})

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [
        saved_meanings.get(column, "")
        for column in df.columns
    ],
    "값의 종류": [
        df[column].nunique(dropna=True)
        for column in df.columns
    ],
    "빈 값 개수": [
        df[column].isna().sum()
        for column in df.columns
    ]
})


# -------------------------
# 입력 및 저장
# -------------------------
with st.form("notes_form"):

    edited_info = st.data_editor(
        column_info,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=[
            "열 이름",
            "값의 종류",
            "빈 값 개수"
        ],
        column_config={
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻",
                help="교재를 참고하여 직접 입력하세요."
            )
        }
    )

    st.markdown("### 📚 데이터 출처")

    source_text = st.text_area(
        "교재에 적힌 데이터 출처를 입력하세요.",
        value=saved_notes.get("source", ""),
        height=100
    )

    save_button = st.form_submit_button(
        "💾 우리말 뜻과 출처 저장"
    )


if save_button:

    meanings_to_save = {
        row["열 이름"]: row["우리말 뜻"]
        for _, row in edited_info.iterrows()
    }

    try:
        save_notes(meanings_to_save, source_text)
        st.success("기록이 성공적으로 저장되었습니다.")

    except OSError as error:
        st.error(f"저장에 실패했습니다: {error}")


# -------------------------
# 데이터 처음 다섯 줄
# -------------------------
st.markdown("### 🔍 데이터 처음 다섯 줄")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


# -------------------------
# 데이터 출처 안내
# -------------------------
st.markdown("### 📌 데이터 출처")

st.write(
    "교재에 제시된 데이터 출처를 위 입력란에 작성하고 저장하세요."
)

st.caption(f"데이터 주소: {DATA_URL}")
st.caption(f"기록 저장 위치: {SAVE_FILE}")
