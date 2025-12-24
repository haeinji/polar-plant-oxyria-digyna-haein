import io
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# -----------------------------
# Streamlit page config & fonts
# -----------------------------
st.set_page_config(page_title="극지식물 최적 EC 농도 연구", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_FONT_FAMILY = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"

# -----------------------------
# Constants (no f-string filename building)
# -----------------------------
SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]
SCHOOL_LABEL_ALL = "전체"

# EC targets (given)
EC_TARGET_BY_SCHOOL = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적
    "아라고": 4.0,
    "동산고": 8.0,
}

# Colors
COLOR_BY_SCHOOL = {
    "송도고": "#1f77b4",
    "하늘고": "#2ca02c",
    "아라고": "#ff7f0e",
    "동산고": "#d62728",
}

ENV_CSV_LOGICAL_NAMES = [
    "송도고_환경데이터.csv",
    "하늘고_환경데이터.csv",
    "아라고_환경데이터.csv",
    "동산고_환경데이터.csv",
]
GROWTH_XLSX_LOGICAL_NAME = "4개교_생육결과데이터.xlsx"

DATA_DIR = Path(__file__).resolve().parent / "data"


# -----------------------------
# Helpers: NFC/NFD robust matching
# -----------------------------
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def _same_name(a: str, b: str) -> bool:
    a0 = str(a).strip()
    b0 = str(b).strip()
    if a0 == b0:
        return True
    return (
        _nfc(a0) == _nfc(b0)
        or _nfd(a0) == _nfd(b0)
        or _nfc(a0) == _nfd(b0)
        or _nfd(a0) == _nfc(b0)
    )


def find_file_by_logical_name(directory: Path, logical_name: str) -> Path | None:
    """
    Must use Path.iterdir(), and NFC/NFD bidirectional comparison.
    """
    if not directory.exists():
        return None

    for p in directory.iterdir():
        if p.is_file() and _same_name(p.name, logical_name):
            return p

    # fallback (case-insensitive, extra-safe)
    for p in directory.iterdir():
        if p.is_file() and _nfc(p.name).lower() == _nfc(logical_name).lower():
            return p

    return None


def infer_school_from_name(name: str) -> str:
    n = _nfc(str(name))
    for s in SCHOOLS:
        if _nfc(s) in n:
            return s
    return str(name)


def ensure_env_schema(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]

    required = ["time", "temperature", "humidity", "ph", "ec"]
    missing = [c for c in required if c not in df2.columns]
    if missing:
        raise ValueError("환경 데이터에 필요한 컬럼이 누락되었습니다: " + ", ".join(missing))

    df2["time"] = pd.to_datetime(df2["time"], errors="coerce")

    for c in ["temperature", "humidity", "ph", "ec"]:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    df2 = df2.dropna(subset=["time"])
    return df2


def ensure_growth_schema(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]

    required = ["개체번호", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
    missing = [c for c in required if c not in df2.columns]
    if missing:
        raise ValueError("생육 결과 데이터에 필요한 컬럼이 누락되었습니다: " + ", ".join(missing))

    df2["개체번호"] = df2["개체번호"].astype(str)
    for c in ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    return df2


# -----------------------------
# Data loading (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_environment_data(data_dir: Path) -> pd.DataFrame:
    rows = []
    for logical in ENV_CSV_LOGICAL_NAMES:
        p = find_file_by_logical_name(data_dir, logical)
        if p is None:
            continue

        df = pd.read_csv(p, encoding="utf-8", engine="python")
        df = ensure_env_schema(df)
        df["school"] = infer_school_from_name(p.stem)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    env = pd.concat(rows, ignore_index=True)
    env["school"] = env["school"].apply(infer_school_from_name)
    env = env[env["school"].isin(SCHOOLS)]
    return env


@st.cache_data(show_spinner=False)
def load_growth_data(data_dir: Path) -> pd.DataFrame:
    p = find_file_by_logical_name(data_dir, GROWTH_XLSX_LOGICAL_NAME)
    if p is None:
        return pd.DataFrame()

    all_sheets = pd.read_excel(p, sheet_name=None, engine="openpyxl")

    rows = []
    for sheet_name, df in all_sheets.items():
        if df is None or len(df) == 0:
            continue
        df2 = ensure_growth_schema(df)
        df2["school"] = infer_school_from_name(sheet_name)
        rows.append(df2)

    if not rows:
        return pd.DataFrame()

    growth = pd.concat(rows, ignore_index=True)
    growth["school"] = growth["school"].apply(infer_school_from_name)
    growth = growth[growth["school"].isin(SCHOOLS)]
    return growth


def to_download_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_download_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# Sidebar
# -----------------------------
st.title("🌱 극지식물 최적 EC 농도 연구")

with st.sidebar:
    st.header("설정")
    selected_school = st.selectbox("학교 선택", [SCHOOL_LABEL_ALL] + SCHOOLS, index=0)


# -----------------------------
# Load data
# -----------------------------
with st.spinner("데이터 로딩 중..."):
    env_df = load_environment_data(DATA_DIR)
    growth_df = load_growth_data(DATA_DIR)

if env_df.empty:
    st.error("환경 데이터(CSV)를 찾거나 읽을 수 없습니다. data/ 폴더의 파일명을 확인하세요.")
if growth_df.empty:
    st.error("생육 결과 데이터(XLSX)를 찾거나 읽을 수 없습니다. data/ 폴더의 파일명을 확인하세요.")


def filter_by_school(df: pd.DataFrame, school_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if school_choice == SCHOOL_LABEL_ALL:
        return df.copy()
    return df[df["school"] == school_choice].copy()


env_sel = filter_by_school(env_df, selected_school)
growth_sel = filter_by_school(growth_df, selected_school)

# Summaries
env_summary = pd.DataFrame()
if not env_df.empty:
    env_summary = (
        env_df.groupby("school", as_index=False)[["temperature", "humidity", "ph", "ec"]]
        .mean(numeric_only=True)
        .rename(
            columns={
                "temperature": "평균 온도(°C)",
                "humidity": "평균 습도(%)",
                "ph": "평균 pH",
                "ec": "실측 평균 EC",
            }
        )
    )
    env_summary["목표 EC"] = env_summary["school"].map(EC_TARGET_BY_SCHOOL)
    env_summary["EC 오차(실측-목표)"] = env_summary["실측 평균 EC"] - env_summary["목표 EC"]

growth_summary = pd.DataFrame()
if not growth_df.empty:
    g = growth_df.copy()
    g["목표 EC"] = g["school"].map(EC_TARGET_BY_SCHOOL)
    growth_summary = (
        g.groupby(["school", "목표 EC"], as_index=False)
        .agg(
            평균_생중량=("생중량(g)", "mean"),
            평균_잎수=("잎 수(장)", "mean"),
            평균_지상부=("지상부 길이(mm)", "mean"),
            개체수=("개체번호", "count"),
        )
        .sort_values("목표 EC")
    )

total_individuals = int(growth_df["개체번호"].count()) if not growth_df.empty else 0
avg_temp = float(env_sel["temperature"].mean()) if not env_sel.empty else float("nan")
avg_hum = float(env_sel["humidity"].mean()) if not env_sel.empty else float("nan")
optimal_ec = 2.0  # requirement: highlight Hanulgo (EC 2.0)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# =============================
# Tab 1
# =============================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
본 대시보드는 **극지식물의 최적 EC(양액 농도) 조건**을 찾기 위해,
4개 학교에서 수집한 **환경 데이터(온도/습도/pH/EC)** 및 **생육 결과(생중량/잎 수/길이)**를
한 화면에서 비교·분석하도록 설계되었습니다.

핵심 목표:
- 학교별 **환경 조건 차이** 비교
- EC 조건별 **생육 지표 차이** 비교
- 평균 생중량 기준 **최적 EC 농도 도출**
"""
    )

    st.subheader("학교별 EC 조건")
    counts_by_school = growth_df["school"].value_counts().to_dict() if not growth_df.empty else {}

    cond_rows = []
    for s in SCHOOLS:
        cond_rows.append(
            {
                "학교명": s,
                "EC 목표": EC_TARGET_BY_SCHOOL.get(s),
                "개체수": int(counts_by_school.get(s, 0)),
                "색상": COLOR_BY_SCHOOL.get(s),
            }
        )
    st.dataframe(pd.DataFrame(cond_rows), use_container_width=True, hide_index=True)

    st.subheader("주요 지표")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", f"{total_individuals:,}")
    c2.metric("평균 온도(선택 범위)", "-" if pd.isna(avg_temp) else f"{avg_temp:.2f} °C")
    c3.metric("평균 습도(선택 범위)", "-" if pd.isna(avg_hum) else f"{avg_hum:.2f} %")
    c4.metric("최적 EC", f"{optimal_ec:.1f}", help="요구사항: 하늘고 EC 2.0 최적값 강조")


# =============================
# Tab 2
# =============================
with tab2:
    st.subheader("학교별 환경 평균 비교")

    if env_summary.empty:
        st.error("환경 요약 그래프를 생성할 데이터가 없습니다.")
    else:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 평균 EC"),
            horizontal_spacing=0.12,
            vertical_spacing=0.18,
        )

        env_s = env_summary.copy()
        env_s["목표 EC"] = env_s["school"].map(EC_TARGET_BY_SCHOOL)
        env_s = env_s.sort_values("목표 EC")

        fig.add_trace(
            go.Bar(
                x=env_s["school"],
                y=env_s["평균 온도(°C)"],
                name="평균 온도(°C)",
                marker_color=[COLOR_BY_SCHOOL.get(s, "#888888") for s in env_s["school"]],
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=env_s["school"],
                y=env_s["평균 습도(%)"],
                name="평균 습도(%)",
                marker_color=[COLOR_BY_SCHOOL.get(s, "#888888") for s in env_s["school"]],
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                x=env_s["school"],
                y=env_s["평균 pH"],
                name="평균 pH",
                marker_color=[COLOR_BY_SCHOOL.get(s, "#888888") for s in env_s["school"]],
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=env_s["school"],
                y=env_s["목표 EC"],
                name="목표 EC",
                marker_color="rgba(150,150,150,0.6)",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=env_s["school"],
                y=env_s["실측 평균 EC"],
                name="실측 평균 EC",
                marker_color=[COLOR_BY_SCHOOL.get(s, "#888888") for s in env_s["school"]],
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            barmode="group",
            height=720,
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            font=dict(family=PLOTLY_FONT_FAMILY),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("선택한 학교 시계열")

    if env_sel.empty:
        st.error("선택한 범위의 환경 데이터가 없습니다.")
    else:
        env_sel2 = env_sel.sort_values("time")

        fig_t = px.line(
            env_sel2,
            x="time",
            y="temperature",
            color="school" if selected_school == SCHOOL_LABEL_ALL else None,
            title="온도 변화",
        )
        fig_t.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
        st.plotly_chart(fig_t, use_container_width=True)

        fig_h = px.line(
            env_sel2,
            x="time",
            y="humidity",
            color="school" if selected_school == SCHOOL_LABEL_ALL else None,
            title="습도 변화",
        )
        fig_h.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
        st.plotly_chart(fig_h, use_container_width=True)

        fig_e = px.line(
            env_sel2,
            x="time",
            y="ec",
            color="school" if selected_school == SCHOOL_LABEL_ALL else None,
            title="EC 변화 (목표 EC 기준선 포함)",
        )

        if selected_school == SCHOOL_LABEL_ALL:
            for s in SCHOOLS:
                fig_e.add_hline(
                    y=EC_TARGET_BY_SCHOOL.get(s),
                    line_width=1,
                    line_dash="dash",
                    opacity=0.5,
                    annotation_text=s + " 목표 EC",
                    annotation_position="top left",
                )
        else:
            fig_e.add_hline(
                y=EC_TARGET_BY_SCHOOL.get(selected_school),
                line_width=2,
                line_dash="dash",
                annotation_text="목표 EC",
                annotation_position="top left",
            )

        fig_e.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
        st.plotly_chart(fig_e, use_container_width=True)

    with st.expander("환경 데이터 원본 보기 / 다운로드"):
        if env_sel.empty:
            st.info("표시할 환경 데이터가 없습니다.")
        else:
            st.dataframe(env_sel.sort_values(["school", "time"]), use_container_width=True, hide_index=True)
            csv_bytes = to_download_csv_bytes(env_sel.sort_values(["school", "time"]))
            st.download_button(
                label="CSV 다운로드",
                data=csv_bytes,
                file_name="환경데이터_선택범위.csv",
                mime="text/csv",
            )


# =============================
# Tab 3
# =============================
with tab3:
    st.subheader("🥇 핵심 결과: EC별 평균 생중량")

    if growth_summary.empty:
        st.error("생육 결과 요약을 생성할 데이터가 없습니다.")
    else:
        gs = growth_summary.copy()
        gs["EC"] = gs["목표 EC"]

        max_row = gs.loc[gs["평균_생중량"].idxmax()] if len(gs) else None
        if max_row is not None:
            best_ec = float(max_row["EC"])
            best_school = str(max_row["school"])
            best_weight = float(max_row["평균_생중량"])

            c1, c2, c3 = st.columns([1, 1, 2])
            c1.metric("최대 평균 생중량", f"{best_weight:.3f} g")
            c2.metric("해당 EC", f"{best_ec:.1f}")
            c3.metric("해당 학교", best_school)

        fig_core = px.bar(
            gs.sort_values("EC"),
            x="EC",
            y="평균_생중량",
            text="평균_생중량",
            title="EC별 평균 생중량 비교 (최댓값이 최적 후보)",
        )
        fig_core.add_vline(x=2.0, line_width=2, line_dash="dash", annotation_text="최적(하늘고 EC 2.0)", opacity=0.7)
        fig_core.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
        fig_core.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), yaxis_title="평균 생중량(g)")
        st.plotly_chart(fig_core, use_container_width=True)

    st.divider()
    st.subheader("EC별 생육 비교 (2x2)")

    if growth_summary.empty:
        st.error("생육 비교 그래프를 생성할 데이터가 없습니다.")
    else:
        gs_view = growth_summary.copy()
        if selected_school != SCHOOL_LABEL_ALL:
            gs_view = gs_view[gs_view["school"] == selected_school]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 생중량(g) ⭐", "평균 잎 수", "평균 지상부 길이(mm)", "개체수 비교"),
            horizontal_spacing=0.12,
            vertical_spacing=0.18,
        )

        x_ec = gs_view["목표 EC"].astype(float)

        fig.add_trace(go.Bar(x=x_ec, y=gs_view["평균_생중량"], name="평균 생중량"), row=1, col=1)
        fig.add_trace(go.Bar(x=x_ec, y=gs_view["평균_잎수"], name="평균 잎 수"), row=1, col=2)
        fig.add_trace(go.Bar(x=x_ec, y=gs_view["평균_지상부"], name="평균 지상부 길이"), row=2, col=1)
        fig.add_trace(go.Bar(x=x_ec, y=gs_view["개체수"], name="개체수"), row=2, col=2)

        for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            fig.add_vline(x=2.0, line_width=2, line_dash="dash", opacity=0.5, row=r, col=c)

        fig.update_layout(
            height=720,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False,
            font=dict(family=PLOTLY_FONT_FAMILY),
        )
        fig.update_xaxes(title_text="EC")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("학교별 생중량 분포")

    if growth_sel.empty:
        st.error("선택한 범위의 생육 데이터가 없습니다.")
    else:
        gd = growth_sel.copy()
        gd["목표 EC"] = gd["school"].map(EC_TARGET_BY_SCHOOL)

        fig_dist = px.violin(
            gd,
            x="school" if selected_school == SCHOOL_LABEL_ALL else "목표 EC",
            y="생중량(g)",
            box=True,
            points="all",
            title="생중량 분포 (학교/EC 기준)",
            color="school" if selected_school == SCHOOL_LABEL_ALL else None,
        )
        fig_dist.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()
    st.subheader("상관관계 분석")

    if growth_sel.empty:
        st.error("상관관계 산점도를 생성할 데이터가 없습니다.")
    else:
        gd = growth_sel.copy()
        left, right = st.columns(2)

        with left:
            fig_sc1 = px.scatter(
                gd,
                x="잎 수(장)",
                y="생중량(g)",
                color="school" if selected_school == SCHOOL_LABEL_ALL else None,
                title="잎 수 vs 생중량",
                hover_data=["개체번호"],
            )
            fig_sc1.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
            st.plotly_chart(fig_sc1, use_container_width=True)

        with right:
            fig_sc2 = px.scatter(
                gd,
                x="지상부 길이(mm)",
                y="생중량(g)",
                color="school" if selected_school == SCHOOL_LABEL_ALL else None,
                title="지상부 길이 vs 생중량",
                hover_data=["개체번호"],
            )
            fig_sc2.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
            st.plotly_chart(fig_sc2, use_container_width=True)

    with st.expander("학교별 생육 데이터 원본 보기 / XLSX 다운로드"):
        if growth_sel.empty:
            st.info("표시할 생육 데이터가 없습니다.")
        else:
            st.dataframe(growth_sel.sort_values(["school", "개체번호"]), use_container_width=True, hide_index=True)

            xlsx_bytes = to_download_xlsx_bytes(
                growth_sel.sort_values(["school", "개체번호"]),
                sheet_name="생육데이터",
            )
            st.download_button(
                label="XLSX 다운로드",
                data=xlsx_bytes,
                file_name="생육데이터_선택범위.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
