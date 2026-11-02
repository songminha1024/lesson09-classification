# -----------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.subheader("데이터 살펴보기")

st.write(
    "이 데이터는 여러 사람의 건강 및 생활 관련 정보를 담고 있으며, "
    "각 사람에게 뇌졸중이 발생했는지를 나타내는 `stroke` 열을 포함합니다. "
    "먼저 데이터의 크기와 각 열의 특징을 확인해 봅시다."
)

st.info(
    "`stroke` 값이 1이면 데이터에서 뇌졸중 발생으로 표시된 사람이고, "
    "0이면 발생하지 않은 사람입니다."
)


# -----------------------------
# 핵심 숫자 카드 4개
# -----------------------------
total_people = len(df)
column_count = len(df.columns)

if "stroke" in df.columns:
    stroke_count = int((pd.to_numeric(df["stroke"], errors="coerce") == 1).sum())
else:
    stroke_count = 0

stroke_ratio = (stroke_count / total_people * 100) if total_people else 0

st.markdown("### 데이터 한눈에 보기")

card1, card2, card3, card4 = st.columns(4)

card1.metric("전체 사람 수", f"{total_people:,}명")
card2.metric("열 개수", f"{column_count}개")
card3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
card4.metric("stroke 비율", f"{stroke_ratio:.2f}%")


# -----------------------------
# 열 정보 표와 우리말 뜻 입력
# -----------------------------
st.markdown("### 열 이름과 데이터 특징")
st.write(
    "각 열의 우리말 뜻을 교재에서 찾아 입력해 보세요. "
    "값의 종류는 빈 값을 제외한 고유한 값의 개수입니다."
)

saved_notes = load_notes()
saved_meanings = saved_notes["meanings"]

column_info = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [saved_meanings.get(column, "") for column in df.columns],
        "값의 종류": [int(df[column].nunique(dropna=True)) for column in df.columns],
        "빈 값 개수": [int(df[column].isna().sum()) for column in df.columns],
    }
)

with st.form("notes_form"):
    edited_info = st.data_editor(
        column_info,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn("열 이름"),
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻",
                help="교재를 참고해 각 열의 뜻을 직접 입력하세요.",
            ),
            "값의 종류": st.column_config.NumberColumn("값의 종류"),
            "빈 값 개수": st.column_config.NumberColumn("빈 값 개수"),
        },
        key="column_meanings_editor",
    )

    st.markdown("### 데이터 출처 기록")
    source_text = st.text_area(
        "교재에 제시된 데이터 출처를 입력하세요.",
        value=saved_notes["source"],
        height=100,
        key="source_text_area",
        placeholder="예: 교재에 적힌 데이터 출처를 여기에 입력",
    )

    save_button = st.form_submit_button("💾 우리말 뜻과 출처 저장")

if save_button:
    meanings_to_save = {
        row["열 이름"]: row["우리말 뜻"]
        for _, row in edited_info.iterrows()
    }

    try:
        save_notes(meanings_to_save, source_text)
        st.success(f"기록을 저장했습니다. 저장 위치: {SAVE_FILE}")
    except OSError as error:
        st.error(f"기록 파일을 저장하지 못했습니다: {error}")


# -----------------------------
# 데이터 미리보기
# -----------------------------
st.markdown("### 데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

st.caption("위 표는 원본 데이터의 첫 다섯 행을 그대로 보여 줍니다.")


# -----------------------------
# 데이터 출처 안내
# -----------------------------
st.markdown("### 데이터 출처")
st.write("교재에 제시된 데이터 출처를 위의 입력란에 적고 저장하세요.")
st.caption(f"데이터 파일 주소: {DATA_URL}")
st.caption(f"기록 파일: {SAVE_FILE}")
