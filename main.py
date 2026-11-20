# main.py
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ──────────────── 데이터 로딩 (utf-8-sig → cp949) ────────────────
@st.cache_data
def load_csv(file):
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ──────────────── 열 이름 표준화 (KeyError 방지) ────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df

# ──────────────── 한글 폰트 적용 헬퍼 ────────────────
def set_kor(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    return fig

# ──────────────── 사이드바 ────────────────
st.sidebar.title("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")

env_df = load_csv(env_file)
growth_df = load_csv(growth_file)

if not env_df.empty:
    env_df = std_cols(env_df)
if not growth_df.empty:
    growth_df = std_cols(growth_df)

# 학교명: 업로드 파일명에서 추출
school_name = "우리 학교"
if env_file is not None:
    fname = os.path.basename(env_file.name)
    school_name = fname.split("_")[0] if "_" in fname else fname.replace(".csv", "")
if growth_file is not None and school_name == "우리 학교":
    fname = os.path.basename(growth_file.name)
    school_name = fname.split("_")[0] if "_" in fname else fname.replace(".csv", "")

st.sidebar.info(f"📌 업로드된 학교: {school_name}" if school_name != "우리 학교" else "📌 아직 파일이 업로드되지 않았습니다.")

# 생육 측정 항목 선택
growth_items = []
if not growth_df.empty:
    growth_items = [
        c for c in ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"]
        if c in growth_df.columns
    ]
selected_items = st.sidebar.multiselect(
    "🌱 생육 측정 항목 (그래프에 표시)",
    options=growth_items,
    default=[c for c in ["잎 수", "지상부 길이", "생중량"] if c in growth_items],
)

# ──────────────── 탭 ────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ── Tab 1: 실험 개요 ──────────────────────────────────────────
with tab1:
    try:
        st.header(f"🌱 {school_name} 극지식물 개체별 생육 차이 분석")

        st.subheader("📌 연구 배경 및 목적")
        st.markdown("""
- 극지식물은 춥고 건조한 환경에서도 살아남는 강한 식물입니다.
- 같은 환경에서 키워도 **개체마다 생중량(식물 전체 무게)** 이 다를 수 있습니다.
- 이 대시보드에서는 우리 학교 극지식물 **개체별 생육 차이**를 숫자(평균·표준편차)와 그래프로 살펴봅니다.
- **평균**: 모든 값을 더해 개수로 나눈 대표값
- **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 값 (클수록 들쭉날쭉)
- **CV(변동계수)**: 표준편차를 평균으로 나눈 값 → 서로 다른 단위의 비교에 유용
        """)

        # 데이터 요약
        st.subheader("📋 데이터 요약")
        summary_rows = []
        if not growth_df.empty:
            summary_rows.append(["개체 수", f"{len(growth_df)}개"])
            if "time" in growth_df.columns and growth_df["time"].notna().any():
                dmin = growth_df["time"].min()
                dmax = growth_df["time"].max()
                summary_rows.append(["측정 기간 (생육)", f"{dmin.strftime('%Y-%m-%d') if pd.notna(dmin) else '─'} ~ {dmax.strftime('%Y-%m-%d') if pd.notna(dmax) else '─'}"])
            available = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
            summary_rows.append(["생육 측정 항목", " / ".join(available) if available else "없음"])
        if not env_df.empty:
            summary_rows.append(["환경 데이터 행 수", f"{len(env_df):,}행"])
            if "time" in env_df.columns and env_df["time"].notna().any():
                e_min = env_df["time"].min()
                e_max = env_df["time"].max()
                summary_rows.append(["환경 측정 기간", f"{e_min.strftime('%Y-%m-%d') if pd.notna(e_min) else '─'} ~ {e_max.strftime('%Y-%m-%d') if pd.notna(e_max) else '─'}"])
            e_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            summary_rows.append(["환경 측정 항목", " / ".join(e_cols) if e_cols else "없음"])
        if summary_rows:
            st.table(pd.DataFrame(summary_rows, columns=["항목", "값"]))
        else:
            st.info("📁 CSV 파일을 업로드하면 데이터 요약이 표시됩니다.")

        st.divider()

        # 주요 지표 카드
        col1, col2, col3, col4 = st.columns(4)
        # 총 개체수
        if not growth_df.empty and "개체번호" in growth_df.columns:
            n = len(growth_df)
            col1.metric("👆 총 개체수", f"{n}개")
        else:
            col1.metric("👆 총 개체수", "─")

        # 평균 온도
        if not env_df.empty and "temperature" in env_df.columns:
            tmp = env_df["temperature"].dropna()
            if len(tmp) >= 2:
                col2.metric("🌡️ 평균 온도", f"{tmp.mean():.1f} ℃")
            else:
                col2.metric("🌡️ 평균 온도", "데이터 부족")
        else:
            col2.metric("🌡️ 평균 온도", "환경 데이터 없음")

        # 평균 습도
        if not env_df.empty and "humidity" in env_df.columns:
            hum = env_df["humidity"].dropna()
            if len(hum) >= 2:
                col3.metric("💧 평균 습도", f"{hum.mean():.1f} %")
            else:
                col3.metric("💧 평균 습도", "데이터 부족")
        else:
            col3.metric("💧 평균 습도", "환경 데이터 없음")

        # 평균 생중량
        if not growth_df.empty and "생중량" in growth_df.columns:
            bio = growth_df["생중량"].dropna()
            if len(bio) >= 2:
                col4.metric("⚖️ 평균 생중량", f"{bio.mean():.2f} g")
            else:
                col4.metric("⚖️ 평균 생중량", "데이터 부족")
        else:
            col4.metric("⚖️ 평균 생중량", "생중량 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── Tab 2: 환경 데이터 ──────────────────────────────────────────
with tab2:
    try:
        if env_df.empty:
            st.info("🌡️ 환경 CSV를 업로드하면 환경 데이터 분석이 표시됩니다.")
        else:
            st.header(f"🌡️ {school_name} 환경 데이터")

            # 환경 요약 표
            st.subheader("📊 환경 요약 (평균 · 최소 · 최대)")
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_cols:
                unit_map = {"temperature": "℃", "humidity": "%", "ph": "", "ec": "mS/cm"}
                rows = []
                for c in env_cols:
                    vals = env_df[c].dropna()
                    if len(vals) >= 1:
                        rows.append([
                            f"{c} ({unit_map.get(c, '')})".strip(),
                            f"{vals.mean():.2f}",
                            f"{vals.min():.2f}",
                            f"{vals.max():.2f}",
                        ])
                if rows:
                    st.table(pd.DataFrame(rows, columns=["항목", "평균", "최소", "최대"]))
                else:
                    st.info("환경 수치 데이터가 충분하지 않습니다.")
            else:
                st.info("📁 환경 데이터에 온도·습도·pH·EC 열이 없습니다.")

            # 일별 시계열
            st.subheader("📈 날짜별 환경 변화 (일별 평균)")
            if "time" in env_df.columns and env_df["time"].notna().any():
                env_df_daily = env_df.copy()
                env_df_daily["date"] = env_df_daily["time"].dt.date
                daily = env_df_daily.groupby("date").agg(
                    {c: "mean" for c in env_cols}
                ).reset_index()

                if "temperature" in daily.columns and daily["temperature"].notna().any():
                    fig_tmp = go.Figure()
                    fig_tmp.add_trace(go.Scatter(
                        x=daily["date"], y=daily["temperature"],
                        mode="lines", name="온도",
                        line=dict(color="#ef4444", width=2),
                        hovertemplate="날짜: %{x}<br>온도: %{y:.1f}℃<extra></extra>",
                    ))
                    set_kor(fig_tmp)
                    fig_tmp.update_layout(
                        title="🌡️ 일별 평균 온도 변화",
                        xaxis_title="날짜", yaxis_title="온도 (℃)",
                        hovermode="x unified",
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_tmp, width="stretch")

                if "humidity" in daily.columns and daily["humidity"].notna().any():
                    fig_hum = go.Figure()
                    fig_hum.add_trace(go.Scatter(
                        x=daily["date"], y=daily["humidity"],
                        mode="lines", name="습도",
                        line=dict(color="#3b82f6", width=2),
                        hovertemplate="날짜: %{x}<br>습도: %{y:.1f}%<extra></extra>",
                    ))
                    set_kor(fig_hum)
                    fig_hum.update_layout(
                        title="💧 일별 평균 습도 변화",
                        xaxis_title="날짜", yaxis_title="습도 (%)",
                        hovermode="x unified",
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_hum, width="stretch")

                if "ph" in daily.columns and daily["ph"].notna().any():
                    fig_ph = go.Figure()
                    fig_ph.add_trace(go.Scatter(
                        x=daily["date"], y=daily["ph"],
                        mode="lines", name="pH",
                        line=dict(color="#8b5cf6", width=2),
                        hovertemplate="날짜: %{x}<br>pH: %{y:.2f}<extra></extra>",
                    ))
                    set_kor(fig_ph)
                    fig_ph.update_layout(
                        title="🧪 일별 평균 pH 변화",
                        xaxis_title="날짜", yaxis_title="pH",
                        hovermode="x unified",
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_ph, width="stretch")

                if "ec" in daily.columns and daily["ec"].notna().any():
                    fig_ec = go.Figure()
                    fig_ec.add_trace(go.Scatter(
                        x=daily["date"], y=daily["ec"],
                        mode="lines", name="EC",
                        line=dict(color="#10b981", width=2),
                        hovertemplate="날짜: %{x}<br>EC: %{y:.2f} mS/cm<extra></extra>",
                    ))
                    set_kor(fig_ec)
                    fig_ec.update_layout(
                        title="⚡ 일별 평균 EC 변화",
                        xaxis_title="날짜", yaxis_title="EC (mS/cm)",
                        hovermode="x unified",
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_ec, width="stretch")

                if not any(c in daily.columns and daily[c].notna().any() for c in env_cols):
                    st.info("일별 평균 계산을 할 수 있는 수치 데이터가 충분하지 않습니다.")
            else:
                st.info("📁 환경 데이터에 시간(time) 열이 없어 일별 그래프를 그릴 수 없습니다.")

            # 원본 테이블 + 다운로드
            with st.expander("📄 환경 데이터 원본 보기 + CSV 다운로드"):
                if not env_df.empty:
                    st.dataframe(env_df, width="stretch", use_container_width=False)
                    csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ 환경 데이터 CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"{school_name}_환경데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("환경 CSV를 업로드하면 이 섹션이 활성화됩니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── Tab 3: 생육 결과 ──────────────────────────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info("📊 생육 결과 CSV를 업로드하면 생육 분석이 표시됩니다.")
        else:
            st.header(f"📊 {school_name} 극지식물 생육 결과")

            # ── 핵심 결과 카드 ──
            st.subheader("🥇 핵심 결과")
            kcol1, kcol2, kcol3 = st.columns(3)
            if "생중량" in growth_df.columns:
                bio = growth_df["생중량"].dropna()
                if len(bio) >= 2:
                    kcol1.metric("📊 평균 생중량", f"{bio.mean():.2f} g")
                    kcol2.metric("📈 최대 생중량", f"{bio.max():.2f} g")
                    kcol3.metric("📉 최소 생중량", f"{bio.min():.2f} g")
                else:
                    kcol1.metric("📊 평균 생중량", "데이터 부족")
                    kcol2.metric("📈 최대 생중량", "─")
                    kcol3.metric("📉 최소 생중량", "─")
            else:
                kcol1.metric("📊 평균 생중량", "생중량 열 없음")
                kcol2.metric("📈 최대 생중량", "─")
                kcol3.metric("📉 최소 생중량", "─")

            # 상위 25% / 하위 25% 비교
            st.divider()
            if "생중량" in growth_df.columns:
                bio = growth_df["생중량"].dropna()
                if len(bio) >= 4:
                    q75 = bio.quantile(0.75)
                    q25 = bio.quantile(0.25)
                    top_df = growth_df[growth_df["생중량"] >= q75].dropna(subset=["생중량"])
                    bot_df = growth_df[growth_df["생중량"] <= q25].dropna(subset=["생중량"])

                    st.subheader("🏆 생중량 상위 25% vs 하위 25% 개체 비교")
                    comp_cols = st.columns(2)
                    with comp_cols[0]:
                        st.markdown(f"#### 🔴 상위 25% (생중량 ≥ {q75:.2f} g)")
                        if not top_df.empty:
                            tavg = top_df["생중량"].mean()
                            tmin = top_df["생중량"].min()
                            tmax = top_df["생중량"].max()
                            tcount = len(top_df)
                            st.markdown(f"- 개체 수: **{tcount}개**")
                            st.markdown(f"- 평균 생중량: **{tavg:.2f} g**")
                            st.markdown(f"- 범위: **{tmin:.2f} ~ {tmax:.2f} g**")
                            if "잎 수" in top_df.columns:
                                st.markdown(f"- 평균 잎 수: **{top_df['잎 수'].mean():.1f}장**")
                            if "지상부 길이" in top_df.columns:
                                st.markdown(f"- 평균 지상부 길이: **{top_df['지상부 길이'].mean():.1f} mm**")
                    with comp_cols[1]:
                        st.markdown(f"#### 🟢 하위 25% (생중량 ≤ {q25:.2f} g)")
                        if not bot_df.empty:
                            bavg = bot_df["생중량"].mean()
                            bmin = bot_df["생중량"].min()
                            bmax = bot_df["생중량"].max()
                            bcount = len(bot_df)
                            st.markdown(f"- 개체 수: **{bcount}개**")
                            st.markdown(f"- 평균 생중량: **{bavg:.2f} g**")
                            st.markdown(f"- 범위: **{bmin:.2f} ~ {bmax:.2f} g**")
                            if "잎 수" in bot_df.columns:
                                st.markdown(f"- 평균 잎 수: **{bot_df['잎 수'].mean():.1f}장**")
                            if "지상부 길이" in bot_df.columns:
                                st.markdown(f"- 평균 지상부 길이: **{bot_df['지상부 길이'].mean():.1f} mm**")

            # ── 개체별 생육 비교 (막대 그래프) ──
            st.divider()
            st.subheader("📊 개체별 생육 비교 (막대 그래프)")

            if "생중량" in growth_df.columns and "개체번호" in growth_df.columns:
                bar_cols = [c for c in selected_items if c in growth_df.columns and c != "개체번호"]
                if bar_cols:
                    df_bar = growth_df[["개체번호"] + bar_cols].dropna(subset=bar_cols, how="all")
                    # 개체번호 원래 값 그대로 유지
                    df_bar = df_bar.sort_values("생중량" if "생중량" in bar_cols else bar_cols[0], ascending=False)

                    fig_bar = make_subplots(rows=1, cols=len(bar_cols), shared_yaxes=False,
                                            subplot_titles=bar_cols)
                    colors = ["#22c55e", "#3b82f6", "#f59e0b", "#8b5cf6", "#ef4444", "#10b981"]
                    for i, col in enumerate(bar_cols):
                        fig_bar.add_trace(
                            go.Bar(
                                x=df_bar["개체번호"],
                                y=df_bar[col],
                                name=col,
                                marker_color=colors[i % len(colors)],
                                hovertemplate=f"{col}: %{{y:.2f}}<extra>{df_bar['개체번호'][j] if 'j' in dir() else ''}</extra>",
                            ),
                            row=1, col=i + 1,
                        )
                    set_kor(fig_bar)
                    fig_bar.update_layout(
                        title="🌱 개체별 생육 지표 비교",
                        height=400,
                        margin=dict(l=40, r=20, t=50, b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        showlegend=True,
                    )
                    for i in range(len(bar_cols)):
                        fig_bar.update_yaxes(title_text=bar_cols[i], row=1, col=i + 1)
                    st.plotly_chart(fig_bar, width="stretch")
                else:
                    st.info("선택한 생육 측정 항목이 없습니다. 사이드바에서 선택해 주세요.")

            # ── 생중량 분포 (박스플롯) ──
            st.divider()
            if "생중량" in growth_df.columns:
                bio = growth_df["생중량"].dropna()
                if len(bio) >= 3:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=bio,
                        name="생중량",
                        fillcolor="#22c55e",
                        marker=dict(color="#22c55e"),
                        line=dict(color="#16a34a"),
                        hovertemplate="생중량: %{y:.2f} g<extra></extra>",
                        boxmean=True,
                    ))
                    set_kor(fig_box)
                    fig_box.update_layout(
                        title="📦 생중량 분포 (박스플롯)",
                        yaxis_title="생중량 (g)",
                        margin=dict(l=40, r=20, t=50, b=40),
                        annotations=[
                            dict(
                                x=0.5, y=1.08, xref="paper", yref="paper",
                                text=(
                                    f"평균: {bio.mean():.2f} g　"
                                    f"표준편차: {bio.std():.2f} g　"
                                    f"CV: {bio.std()/bio.mean()*100:.1f}%"
                                ),
                                showarrow=False,
                                font=dict(size=13),
                            )
                        ],
                    )
                    st.plotly_chart(fig_box, width="stretch")

                    st.markdown("""
- **박스플롯**: 데이터의 분포와 이상치를 한눈에 보는 그래프
- **상자(box)**: 가운데 50% 데이터가 모인 구간 (25%~75%)
- **median 선**: 중앙값 (데이터를 크기순으로 나열했을 때 정가운데 값)
- **CV(변동계수)**: `(표준편차 ÷ 평균) × 100` → 값이 평균에서 얼마나 상대적으로 흩어져 있는지 나타냄
                    """)
                else:
                    st.info("생중량 데이터가 충분하지 않아 박스플롯을 그릴 수 없습니다.")

            # ── 상관관계 분석 (산점도 + 추세선) ──
            st.divider()
            st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")

            if "생중량" in growth_df.columns:
                bio = growth_df["생중량"].dropna()
                if len(bio) >= 3:
                    scatter_pairs = []
                    if "잎 수" in growth_df.columns:
                        scatter_pairs.append(("잎 수", "잎 수"))
                    if "지상부 길이" in growth_df.columns:
                        scatter_pairs.append(("지상부 길이", "지상부 길이"))

                    if scatter_pairs:
                        fig_scatter = make_subplots(
                            rows=1, cols=len(scatter_pairs),
                            subplot_titles=[f"{x_label} vs 생중량" for x_label, _ in scatter_pairs],
                            shared_yaxes=False,
                        )
                        colors_pair = ["#3b82f6", "#f59e0b"]
                        for i, (x_label, x_col) in enumerate(scatter_pairs):
                            tmp_df = growth_df[[x_col, "생중량"]].dropna()
                            if len(tmp_df) >= 2:
                                x_vals = tmp_df[x_col].values
                                y_vals = tmp_df["생중량"].values
                                r = np.corrcoef(x_vals, y_vals)[0, 1]

                                fig_scatter.add_trace(
                                    go.Scatter(
                                        x=x_vals, y=y_vals,
                                        mode="markers",
                                        marker=dict(
                                            color=colors_pair[i % len(colors_pair)],
                                            size=8, opacity=0.7,
                                            line=dict(width=1, color="white"),
                                        ),
                                        name=f"{x_label} vs 생중량",
                                        hovertemplate=f"{x_label}: %{{x:.1f}}<br>생중량: %{{y:.2f}} g<extra></extra>",
                                    ),
                                    row=1, col=i + 1,
                                )

                                # 추세선: np.polyfit
                                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                                y_line = slope * x_line + intercept
                                fig_scatter.add_trace(
                                    go.Scatter(
                                        x=x_line, y=y_line,
                                        mode="lines",
                                        line=dict(color="#ef4444", width=2, dash="dash"),
                                        name=f"추세선 (r={r:.2f})",
                                        showlegend=(i == 0),
                                        hoverinfo="skip",
                                    ),
                                    row=1, col=i + 1,
                                )

                                fig_scatter.add_annotation(
                                    x=0.98, y=0.02, xref="paper", yref="paper",
                                    text=f"상관계수 r = {r:.2f}",
                                    showarrow=False,
                                    font=dict(size=12, color="#374151"),
                                    align="right",
                                    row=1, col=i + 1,
                                )

                        set_kor(fig_scatter)
                        fig_scatter.update_layout(
                            title="🔗 생육 지표 vs 생중량 상관관계",
                            height=420,
                            margin=dict(l=40, r=20, t=50, b=40),
                            showlegend=True,
                        )
                        fig_scatter.update_yaxes(title_text="생중량 (g)", row=1, col=1)
                        fig_scatter.update_xaxes(title_text=scatter_pairs[0][0], row=1, col=1)
                        if len(scatter_pairs) > 1:
                            fig_scatter.update_xaxes(title_text=scatter_pairs[1][0], row=1, col=2)
                        st.plotly_chart(fig_scatter, width="stretch")

                        st.markdown("""
- **상관계수 r**: 두 지표가 함께 움직이는 정도 (-1 ~ +1)
  - `+1`에 가까울수록 함께 증가 (양의 상관)
  - `0`에 가까울수록 관계 거의 없음
  - `-1`에 가까울수록 한쪽 증가 시 다른 쪽 감소 (음의 상관)
- **추세선**: np.polyfit으로 구한 직선 — 데이터가 대체로 어떤 방향으로 늘어나는지 보여줌
                        """)
                    else:
                        st.info("산점도를 그릴 수 있는 생육 지표가 충분하지 않습니다.")
                else:
                    st.info("생중량 데이터가 3개 미만이라 상관관계 분석을 할 수 없습니다.")

            # ── 원본 테이블 + 다운로드 ──
            with st.expander("📄 생육 데이터 원본 보기 + CSV 다운로드"):
                if not growth_df.empty:
                    st.dataframe(growth_df, width="stretch", use_container_width=False)
                    csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ 생육 결과 CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"{school_name}_생육결과.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("생육 CSV를 업로드하면 이 섹션이 활성화됩니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
