import io
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================
# Page config & Korean font
# =============================
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

PLOTLY_FONT = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"

# =============================
# Constants
# =============================
SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]
ALL = "전체"

EC_TARGET = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적
    "아라고": 4.0,
    "동산고": 8.0,
}

ENV_FILES = [
    "송도고_환경데이터.csv",
    "하늘고_환경데이터.csv",
    "아라고_환경데이터.csv",
    "동산고_환경데이터.csv",
]
GROWTH_FILE = "4개교_생육결과데이터.xlsx"


# =============================
# Unicode-safe helpers
# =============================
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def same_name(a: str, b: str) -> bool:
    return nfc(a) == nfc(b)


def find_data_dir(start: Path) -> Path | None:
    """
    main.py 위치부터 상위로 올라가며 data/ 폴더 탐색
    """
    cur = start
    for _ in range(5):  # 최대 5단계 상위까지
        candidate = cur / "data"
        if candidate.exists() and candidate.is_dir():
            return candidate
        cur = cur.parent
    return None


# =============================
# Data loading
# =============================
@st.cache_data
def load_environment_data(data_dir: Path) -> pd.DataFrame:
    dfs = []
    for fname in ENV_FILES:
        for p in data_dir.iterdir():
            if p.is_file() and same_name(p.name, fname):
                df = pd.read_csv(p)
                df.columns = [c.strip() for c in df.columns]
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
                df["school"] = fname.split("_")[0]
                dfs.append(df)
                break

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


@st.cache_data
def load_growth_data(data_dir: Path) -> pd.DataFrame:
    for p in data_dir.iterdir():
        if p.is_file() and same_name(p.name, GROWTH_FILE):
            sheets = pd.read_excel(p, sheet_name=None, engine="openpyxl")
            rows = []
            for sheet, df in sheets.items():
                df.columns = [c.strip() for c in df.columns]
                df["school"] = sheet
                rows.append(df)
            return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


# =============================
# Locate data directory
# =============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = find_data_dir(BASE_DIR)

st.title("🌱 극지식물 최적 EC 농도 연구")

if DATA_DIR is None:
    st.error("❌ data/ 폴더를 찾을 수 없습니다. main.py와 같은 레벨에 data/가 있어야 합니다.")
    st.stop()

# =============================
# Load data
# =============================
with st.spinner("데이터 로딩 중..."):
    env_df = load_environment_data(DATA_DIR)
    growth_df = load_growth_data(DATA_DIR)

if env_df.empty:
    st.error("환경 데이터(CSV)를 찾거나 읽을 수 없습니다.")
    st.stop()

if growth_df.empty:
    st.error("생육 결과 데이터(XLSX)를 찾거나 읽을 수 없습니다.")
    st.stop()

# =============================
# Sidebar
# =============================
with st.sidebar:
    school = st.selectbox("학교 선택", [ALL] + SCHOOLS)

def filt(df):
    if school == ALL:
        return df
    return df[df["school"] == school]


env_sel = filt(env_df)
growth_sel = filt(growth_df)

# =============================
# Tabs
# =============================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# =============================
# Tab 1
# =============================
with tab1:
    st.subheader("학교별 EC 조건")
    rows = []
    for s in SCHOOLS:
        rows.append(
            {
                "학교": s,
                "EC 목표": EC_TARGET[s],
                "개체수": int((growth_df["school"] == s).sum()),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# =============================
# Tab 2
# =============================
with tab2:
    avg_env = (
        env_df.groupby("school")[["temperature", "humidity", "ph", "ec"]]
        .mean()
        .reset_index()
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 온도", "평균 습도", "평균 pH", "평균 EC"),
    )

    fig.add_bar(x=avg_env["school"], y=avg_env["temperature"], row=1, col=1)
    fig.add_bar(x=avg_env["school"], y=avg_env["humidity"], row=1, col=2)
    fig.add_bar(x=avg_env["school"], y=avg_env["ph"], row=2, col=1)
    fig.add_bar(x=avg_env["school"], y=avg_env["ec"], row=2, col=2)

    fig.update_layout(height=700, font=dict(family=PLOTLY_FONT))
    st.plotly_chart(fig, use_container_width=True)

# =============================
# Tab 3
# =============================
with tab3:
    growth_df["EC"] = growth_df["school"].map(EC_TARGET)

    summary = (
        growth_df.groupby("EC")["생중량(g)"]
        .mean()
        .reset_index()
        .sort_values("EC")
    )

    fig = px.bar(
        summary,
        x="EC",
        y="생중량(g)",
        title="EC별 평균 생중량",
        text="생중량(g)",
    )
    fig.add_vline(x=2.0, line_dash="dash", annotation_text="최적 EC (하늘고)")
    fig.update_layout(font=dict(family=PLOTLY_FONT))
    st.plotly_chart(fig, use_container_width=True)
