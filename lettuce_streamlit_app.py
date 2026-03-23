import re
import json
from datetime import date, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:  # pragma: no cover
    gspread = None
    Credentials = None


st.set_page_config(
    page_title="식물공장 상추 수확량 예측 시스템",
    page_icon="🌿",
    layout="wide",
)


# =========================
# 기본 상수
# =========================
SHEET_URL_DEFAULT = "https://docs.google.com/spreadsheets/d/1FMxN2iS0srEZD2bQ5dlQp2-JaZaXvd24W-glJpqReuo/edit?usp=sharing"
DB_WORKSHEET_NAME = "DB_배치데이터"
LOG_WORKSHEET_NAME = "예측결과_log"

DB_COLS = [
    "batch_id",
    "sow_date",
    "transplant_date",
    "plant_date",
    "harvest_date",
    "grow_days",
    "bed_type",
    "bed_id",
    "tray_or_gutter",
    "weight_per_plant_g",
    "loss_rate",
    "actual_yield",
    "actual_weight_kg",
    "note",
]

FIXED_BED_CONFIG = {
    1: 40,
    2: 40,
    3: 40,
    4: 40,
    5: 40,
    6: 40,
    7: 40,
    8: 32,
    9: 32,
    10: 32,
    11: 32,
    12: 32,
    13: 32,
    14: 32,
    15: 32,
    16: 32,
    17: 32,
    18: 32,
    19: 40,
    20: 40,
}
PLANTS_PER_TRAY = 16
PLANTS_PER_GUTTER = 13
PREDICTION_WINDOW = [3, 4]
WEEKDAYS_KR = ["월", "화", "수", "목", "금", "토", "일"]


# =========================
# 스타일
# =========================
st.markdown(
    """
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {border-right: 1px solid #eef1eb;}
.small-muted {font-size: 0.82rem; color: #7f8c7a;}
.section-title {font-size: 1.05rem; font-weight: 700; margin-bottom: .4rem;}
.kpi-box {
    background: #27500A;
    color: #EAF3DE;
    border-radius: 16px;
    padding: 16px 20px;
}
.status-ok {
    display: inline-block; background:#EAF3DE; color:#27500A; border-radius:999px;
    padding:2px 10px; font-size:12px; font-weight:700;
}
.status-warn {
    display: inline-block; background:#FAEEDA; color:#854F0B; border-radius:999px;
    padding:2px 10px; font-size:12px; font-weight:700;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# 유틸
# =========================
def extract_sheet_id(sheet_url: str) -> str:
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        raise ValueError("Google Sheets 링크에서 sheet id를 찾을 수 없습니다.")
    return match.group(1)


def extract_gid(sheet_url: str) -> Optional[str]:
    match = re.search(r"[?&#]gid=([0-9]+)", sheet_url)
    return match.group(1) if match else None


def empty_db() -> pd.DataFrame:
    return pd.DataFrame(columns=DB_COLS)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DB_COLS:
        if col not in out.columns:
            out[col] = None
    return out[DB_COLS]


def to_str_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace({np.nan: None})
    for col in out.columns:
        out[col] = out[col].apply(
            lambda x: "" if x is None else (x.strftime("%Y-%m-%d") if hasattr(x, "strftime") and not isinstance(x, str) else str(x))
        )
    return out


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["sow_date", "transplant_date", "plant_date", "harvest_date"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def normalize_db(df: pd.DataFrame, default_weight_g: float, default_loss_rate: float, mgs_total_gutters: Optional[int]) -> pd.DataFrame:
    d = ensure_columns(df)
    d = parse_dates(d)
    d["grow_days"] = pd.to_numeric(d["grow_days"], errors="coerce")
    d["tray_or_gutter"] = pd.to_numeric(d["tray_or_gutter"], errors="coerce")
    d["weight_per_plant_g"] = pd.to_numeric(d["weight_per_plant_g"], errors="coerce").fillna(default_weight_g)
    d["loss_rate"] = pd.to_numeric(d["loss_rate"], errors="coerce").fillna(default_loss_rate)
    d["actual_yield"] = pd.to_numeric(d["actual_yield"], errors="coerce")
    d["actual_weight_kg"] = pd.to_numeric(d["actual_weight_kg"], errors="coerce")
    d["bed_type"] = d["bed_type"].astype(str).str.lower().replace({"nan": ""})
    if mgs_total_gutters is not None:
        d.loc[d["bed_type"] == "mgs", "tray_or_gutter"] = mgs_total_gutters
    bad = d[(d["harvest_date"].notna()) & (d["plant_date"].notna()) & (d["harvest_date"] < d["plant_date"])]
    if not bad.empty:
        d = d.drop(index=bad.index).copy()
    return d


def calc_predictions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # 빈 DataFrame 에서 row-wise apply 결과를 바로 2개 컬럼에 대입하면
    # pandas 버전에 따라 "Columns must be same length as key" 에러가 날 수 있어
    # 벡터화 계산으로 고정한다.
    d["predicted_plants"] = np.nan
    d["predicted_kg"] = np.nan

    if d.empty:
        d["total_days"] = pd.Series(dtype="float64")
        d["status"] = pd.Series(dtype="object")
        return d

    bed_type = d["bed_type"].astype(str).str.lower().fillna("")
    plants_per_unit = np.where(bed_type.eq("fixed"), PLANTS_PER_TRAY, PLANTS_PER_GUTTER)

    valid_mask = d["tray_or_gutter"].notna() & d["loss_rate"].notna() & d["weight_per_plant_g"].notna()
    predicted_plants = np.round(
        d.loc[valid_mask, "tray_or_gutter"].astype(float)
        * plants_per_unit[valid_mask]
        * (1 - d.loc[valid_mask, "loss_rate"].astype(float))
    )
    d.loc[valid_mask, "predicted_plants"] = predicted_plants
    d.loc[valid_mask, "predicted_kg"] = np.round(
        predicted_plants * d.loc[valid_mask, "weight_per_plant_g"].astype(float) / 1000,
        2,
    )

    d["total_days"] = (d["harvest_date"] - d["sow_date"]).dt.days
    d["status"] = np.where(d["predicted_plants"].notna(), "규칙 기반", "N/A")
    return d


def fmt_date_short(v: Any) -> str:
    if pd.isna(v):
        return "—"
    try:
        return pd.Timestamp(v).strftime("%m-%d")
    except Exception:
        return "—"


def fmt_date_long(v: Any) -> str:
    if pd.isna(v):
        return "—"
    try:
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    except Exception:
        return "—"


def compute_actual_weight_g(row: pd.Series) -> Optional[float]:
    if pd.notna(row.get("actual_yield")) and pd.notna(row.get("actual_weight_kg")) and float(row["actual_yield"]) > 0:
        return round(float(row["actual_weight_kg"]) * 1000 / float(row["actual_yield"]), 1)
    return None


def day_sum(df: pd.DataFrame, target_date: date) -> Tuple[int, float, Optional[int], Optional[float], bool]:
    sub = df[df["harvest_date"].dt.date == target_date]
    pp = sub["predicted_plants"].sum(skipna=True)
    pk = sub["predicted_kg"].sum(skipna=True)
    ap = sub["actual_yield"].sum(skipna=True)
    ak = sub["actual_weight_kg"].sum(skipna=True)
    has_actual = sub["actual_weight_kg"].notna().any()
    return (
        int(pp) if pp else 0,
        round(float(pk), 1) if pk else 0.0,
        int(ap) if ap else None,
        round(float(ak), 1) if has_actual and ak == ak else None,
        has_actual,
    )


def make_public_csv_urls(sheet_id: str, worksheet_name: str, gid: Optional[str] = None) -> List[str]:
    urls: List[str] = []

    # 1) 시트 이름 기반 gviz CSV: 공유 링크(보기 권한)만 있어도 동작하는 경우가 많음
    urls.append(
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={worksheet_name}"
    )

    # 2) gid가 있으면 해당 탭 직접 export
    if gid:
        urls.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}")

    # 3) 첫 시트 fallback
    urls.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0")
    return urls


# =========================
# Google Sheets 연결
# =========================
def get_service_account_info() -> Optional[Dict[str, Any]]:
    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
    elif "gcp_service_account_json" in st.secrets:
        raw = st.secrets["gcp_service_account_json"]
        info = json.loads(raw)
    else:
        return None

    if "private_key" in info and "\\n" in info["private_key"]:
        info["private_key"] = info["private_key"].replace("\\n", "\n")
    return info


@st.cache_resource(show_spinner=False)
def get_gspread_client() -> Optional[Any]:
    info = get_service_account_info()
    if not info or gspread is None or Credentials is None:
        return None
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(credentials)


def open_spreadsheet(sheet_url: str):
    client = get_gspread_client()
    if client is None:
        return None
    sheet_id = extract_sheet_id(sheet_url)
    return client.open_by_key(sheet_id)


def get_or_create_worksheet(spreadsheet, title: str, rows: int = 1000, cols: int = 30):
    try:
        return spreadsheet.worksheet(title)
    except Exception:
        ws = spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)
        if title == DB_WORKSHEET_NAME:
            ws.update("A1", [DB_COLS])
        elif title == LOG_WORKSHEET_NAME:
            ws.update(
                "A1",
                [[
                    "snapshot_date",
                    "prediction_base_date",
                    "batch_id",
                    "bed_type",
                    "bed_id",
                    "sow_date",
                    "harvest_date",
                    "total_days",
                    "tray_or_gutter",
                    "weight_per_plant_g",
                    "loss_rate",
                    "predicted_plants",
                    "predicted_kg",
                    "actual_yield",
                    "actual_weight_kg",
                    "status",
                    "note",
                ]],
            )
        return ws


def read_db(sheet_url: str) -> Tuple[pd.DataFrame, bool, str]:
    spreadsheet = open_spreadsheet(sheet_url)
    if spreadsheet is not None:
        ws = get_or_create_worksheet(spreadsheet, DB_WORKSHEET_NAME)
        values = ws.get_all_records()
        df = pd.DataFrame(values)
        return ensure_columns(df), True, "service_account"

    # read-only fallback
    sheet_id = extract_sheet_id(sheet_url)
    gid = extract_gid(sheet_url)
    errors = []
    for public_csv_url in make_public_csv_urls(sheet_id, DB_WORKSHEET_NAME, gid=gid):
        try:
            df = pd.read_csv(public_csv_url)
            return ensure_columns(df), False, "public_csv"
        except Exception as exc:
            errors.append(str(exc))

    st.session_state["sheet_read_errors"] = errors
    return empty_db(), False, "unavailable"


def write_db(sheet_url: str, df: pd.DataFrame) -> Tuple[bool, str]:
    spreadsheet = open_spreadsheet(sheet_url)
    if spreadsheet is None:
        return False, "쓰기 가능한 Google Sheets 연결이 없습니다. st.secrets 설정이 필요합니다."
    ws = get_or_create_worksheet(spreadsheet, DB_WORKSHEET_NAME, rows=max(len(df) + 20, 1000), cols=len(DB_COLS) + 3)
    out = to_str_df(ensure_columns(df))
    rows = [out.columns.tolist()] + out.values.tolist()
    ws.clear()
    ws.update("A1", rows)
    return True, f"Google Sheets 저장 완료 ({len(out)}건)"


def append_prediction_log(sheet_url: str, prediction_base_date: date, target_df: pd.DataFrame) -> Tuple[bool, str]:
    spreadsheet = open_spreadsheet(sheet_url)
    if spreadsheet is None:
        return False, "로그 저장은 service account 연결에서만 가능합니다."
    ws = get_or_create_worksheet(spreadsheet, LOG_WORKSHEET_NAME, rows=2000, cols=20)
    save_cols = [
        "batch_id",
        "bed_type",
        "bed_id",
        "sow_date",
        "harvest_date",
        "total_days",
        "tray_or_gutter",
        "weight_per_plant_g",
        "loss_rate",
        "predicted_plants",
        "predicted_kg",
        "actual_yield",
        "actual_weight_kg",
        "status",
        "note",
    ]
    payload = target_df[[c for c in save_cols if c in target_df.columns]].copy()
    payload.insert(0, "prediction_base_date", str(prediction_base_date))
    payload.insert(0, "snapshot_date", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    payload = to_str_df(payload)
    ws.append_rows(payload.values.tolist(), value_input_option="USER_ENTERED")
    return True, f"예측결과_log 시트에 {len(payload)}건 추가했습니다."


# =========================
# HTML 렌더
# =========================
def render_day_card(label: str, color: str, dt: date, pp: int, pk: float, ap: Optional[int], ak: Optional[float], has_actual: bool) -> str:
    border = f"border:2px solid {color}" if color != "#888" else "border:0.5px solid #ccc"
    date_s = f"{dt.month}월 {dt.day}일 ({WEEKDAYS_KR[dt.weekday()]})"
    pred_p = f"{pp:,}주" if pp else ("수확 없음" if not has_actual else "—")
    pred_k = f"{pk:.1f} kg" if pk else "—"
    act_p = f"{ap:,}주" if ap is not None else "—"
    act_k = f"{ak:.1f} kg" if ak is not None else "—"
    diff_html = ""
    if pk and ak is not None:
        diff_v = round(ak - pk, 1)
        sign = "+" if diff_v >= 0 else ""
        diff_color = "#27500A" if diff_v >= 0 else "#A32D2D"
        diff_html = f"<div style='text-align:right;font-size:12px;font-weight:500;color:{diff_color}'>오차 {sign}{diff_v} kg</div>"
    actual_block = ""
    if has_actual:
        actual_block = f"""
        <div style="border-top:0.5px solid #e0e0e0;margin-top:10px;padding-top:10px">
            <div style="font-size:10px;font-weight:600;color:#888;letter-spacing:.6px;margin-bottom:6px">실적</div>
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">
                <span style="font-size:11px;color:#888">실제 주수</span>
                <span style="font-size:15px;font-weight:500">{act_p}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">
                <span style="font-size:11px;color:#888">실제 무게</span>
                <span style="font-size:15px;font-weight:500;color:#185FA5">{act_k}</span>
            </div>
            {diff_html}
        </div>
        """
    return f"""
    <div style="background:#fff;{border};border-radius:12px;padding:14px 16px">
      <div style="font-size:10px;font-weight:600;letter-spacing:.8px;color:{color};margin-bottom:6px">{label}</div>
      <div style="font-size:14px;font-weight:500;margin-bottom:12px">{date_s}</div>
      <div style="font-size:10px;font-weight:600;color:#888;letter-spacing:.6px;margin-bottom:6px">예측</div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">
        <span style="font-size:11px;color:#888">주수</span>
        <span style="font-size:{'17' if pp else '13'}px;font-weight:500;color:{'#1a1a2e' if pp else '#aaa'}">{pred_p}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:baseline">
        <span style="font-size:11px;color:#888">무게</span>
        <span style="font-size:17px;font-weight:500;color:{color if pk else '#aaa'}">{pred_k}</span>
      </div>
      {actual_block}
    </div>
    """


def render_dashboard_html(
    prediction_date: date,
    loss_rate_pct: int,
    default_weight_g: int,
    db_all: pd.DataFrame,
    target: pd.DataFrame,
) -> str:
    dash_dates = [prediction_date, prediction_date + timedelta(days=3), prediction_date + timedelta(days=4)]
    d0_pp, d0_pk, d0_ap, d0_ak, d0_ha = day_sum(db_all, dash_dates[0])
    d3_pp, d3_pk, d3_ap, d3_ak, d3_ha = day_sum(db_all, dash_dates[1])
    d4_pp, d4_pk, d4_ap, d4_ak, d4_ha = day_sum(db_all, dash_dates[2])

    total_pp = d3_pp + d4_pp
    total_pk = round(d3_pk + d4_pk, 1)
    valid = target[target["predicted_plants"].notna()].copy()
    fixed_pp = int(valid[valid["bed_type"] == "fixed"]["predicted_plants"].sum()) if not valid.empty else 0
    fixed_pk = round(float(valid[valid["bed_type"] == "fixed"]["predicted_kg"].sum()), 1) if not valid.empty else 0.0
    mgs_pp = int(valid[valid["bed_type"] == "mgs"]["predicted_plants"].sum()) if not valid.empty else 0
    mgs_pk = round(float(valid[valid["bed_type"] == "mgs"]["predicted_kg"].sum()), 1) if not valid.empty else 0.0
    has_na = target["predicted_plants"].isna().any() if not target.empty else False
    any_actual = target["actual_weight_kg"].notna().any() if not target.empty else False

    detail_rows = ""
    for _, row in target.sort_values(["harvest_date", "batch_id"]).iterrows():
        bt_lbl = "고정" if row["bed_type"] == "fixed" else "MGS"
        bt_bg = "#EAF3DE" if row["bed_type"] == "fixed" else "#E6F1FB"
        bt_fg = "#27500A" if row["bed_type"] == "fixed" else "#0C447C"
        hdate = fmt_date_short(row["harvest_date"])
        sdate = fmt_date_short(row["sow_date"])
        tdays = f"{int(row['total_days'])}일" if pd.notna(row["total_days"]) else "—"
        tg = int(row["tray_or_gutter"]) if pd.notna(row["tray_or_gutter"]) else "—"
        wpg = f"{int(row['weight_per_plant_g'])}g" if pd.notna(row["weight_per_plant_g"]) else "—"
        loss_pct = f"{float(row['loss_rate']) * 100:.0f}%"
        pp = f"{int(row['predicted_plants']):,}주" if pd.notna(row["predicted_plants"]) else "<span style='color:#aaa'>N/A</span>"
        pk = f"{float(row['predicted_kg']):.1f} kg" if pd.notna(row["predicted_kg"]) else "<span style='color:#aaa'>N/A</span>"
        ap = f"{int(row['actual_yield']):,}주" if pd.notna(row["actual_yield"]) else ""
        ak = f"{float(row['actual_weight_kg']):.1f} kg" if pd.notna(row["actual_weight_kg"]) else ""
        awpg = compute_actual_weight_g(row)
        diff_html = ""
        if pd.notna(row["actual_weight_kg"]) and pd.notna(row["predicted_kg"]):
            diff = round(float(row["actual_weight_kg"]) - float(row["predicted_kg"]), 1)
            sign = "+" if diff >= 0 else ""
            clr = "#27500A" if diff >= 0 else "#A32D2D"
            diff_html = f"<br><span style='font-size:10px;color:{clr}'>{sign}{diff} kg</span>"
        note_txt = str(row.get("note", "")) if pd.notna(row.get("note", "")) else str(row.get("status", ""))
        detail_rows += f"""
        <tr>
          <td style="font-family:monospace;font-size:11px">{row['batch_id']}</td>
          <td><span style="background:{bt_bg};color:{bt_fg};padding:2px 7px;border-radius:10px;font-size:11px;font-weight:600">{bt_lbl}</span></td>
          <td style="text-align:center">{row['bed_id']}</td>
          <td style="text-align:center">{sdate}</td>
          <td style="text-align:center">{hdate}</td>
          <td style="text-align:center">{tdays}</td>
          <td style="text-align:center">{tg}</td>
          <td style="text-align:center">{wpg}</td>
          <td style="text-align:center">{loss_pct}</td>
          <td style="text-align:right;font-weight:500">{pp}</td>
          <td style="text-align:right;font-weight:500;color:#27500A">{pk}{diff_html}</td>
          <td style="text-align:right;color:#185FA5">{ap}{('<br>' if ap and ak else '')}{ak}</td>
          <td style="text-align:right;color:#185FA5">{f'{awpg:.1f} g' if awpg is not None else '—'}</td>
          <td style="font-size:11px;color:#999">{note_txt}</td>
        </tr>
        """

    na_warn = ""
    if has_na:
        na_warn = "<p style='color:#854F0B;background:#FAEEDA;padding:8px 12px;border-radius:8px;font-size:12px;margin-top:10px'>⚠️ 거터 수 미입력 배치 포함 — MGS 총 거터 수를 입력하면 자동 반영됩니다.</p>"
    actual_note = ""
    if any_actual:
        actual_note = "<p style='font-size:12px;color:#0C447C;background:#E6F1FB;padding:8px 12px;border-radius:8px;margin-top:10px'>📥 실적 데이터 포함 — 카드 하단 및 무게 컬럼에 오차가 표시됩니다.</p>"

    return f"""
    <div style="font-family:-apple-system,sans-serif;max-width:960px">
      <div style="font-size:11px;color:#999;margin-bottom:16px">
        예측 기준일: {prediction_date} &nbsp;·&nbsp; 로스율 {loss_rate_pct}% &nbsp;·&nbsp; 주당 기본 무게 {default_weight_g}g
      </div>

      <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-bottom:14px">
        {render_day_card('D+0 · 오늘', '#888', dash_dates[0], d0_pp, d0_pk, d0_ap, d0_ak, d0_ha)}
        {render_day_card('D+3 · 수확 예정', '#3B6D11', dash_dates[1], d3_pp, d3_pk, d3_ap, d3_ak, d3_ha)}
        {render_day_card('D+4 · 수확 예정', '#185FA5', dash_dates[2], d4_pp, d4_pk, d4_ap, d4_ak, d4_ha)}
      </div>

      <div style="background:#27500A;border-radius:12px;padding:14px 20px;display:flex;justify-content:space-around;margin-bottom:14px">
        <div style="text-align:center">
          <div style="font-size:10px;color:#C0DD97;margin-bottom:4px">이번 주 예측 주수 (D+3~4)</div>
          <div style="font-size:20px;font-weight:500;color:#EAF3DE">{total_pp:,}주</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:10px;color:#C0DD97;margin-bottom:4px">이번 주 예측 무게 (D+3~4)</div>
          <div style="font-size:20px;font-weight:500;color:#EAF3DE">{total_pk:.1f} kg</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:10px;color:#C0DD97;margin-bottom:4px">수확률</div>
          <div style="font-size:20px;font-weight:500;color:#EAF3DE">{100-loss_rate_pct}%</div>
          <div style="font-size:10px;color:#97C459">로스율 {loss_rate_pct}%</div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px">
        <div style="border:0.5px solid #e0e0e0;border-radius:8px;padding:12px 16px">
          <div style="font-size:11px;font-weight:600;color:#555;margin-bottom:8px">고정 재배대</div>
          <div style="display:flex;gap:20px">
            <div style="text-align:center"><div style="font-size:16px;font-weight:500">{fixed_pp:,}주</div><div style="font-size:10px;color:#999">예측 주수</div></div>
            <div style="text-align:center"><div style="font-size:16px;font-weight:500;color:#27500A">{fixed_pk:.1f} kg</div><div style="font-size:10px;color:#999">예측 무게</div></div>
          </div>
        </div>
        <div style="border:0.5px solid #e0e0e0;border-radius:8px;padding:12px 16px">
          <div style="font-size:11px;font-weight:600;color:#555;margin-bottom:8px">MGS (NFT)</div>
          <div style="display:flex;gap:20px">
            <div style="text-align:center"><div style="font-size:16px;font-weight:500;{'color:#aaa' if not mgs_pp else ''}">{f'{mgs_pp:,}주' if mgs_pp else 'N/A'}</div><div style="font-size:10px;color:#999">예측 주수</div></div>
            <div style="text-align:center"><div style="font-size:16px;font-weight:500;{'color:#aaa' if not mgs_pk else 'color:#185FA5'}">{f'{mgs_pk:.1f} kg' if mgs_pk else 'N/A'}</div><div style="font-size:10px;color:#999">예측 무게</div></div>
          </div>
        </div>
      </div>

      <div style="font-size:13px;font-weight:500;color:#444;margin-bottom:8px">배치별 상세 (D+3~4)</div>
      <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap">
        <thead>
          <tr style="background:#f7f7f7">
            <th style="padding:7px 8px;text-align:left;border-bottom:1.5px solid #ddd;font-size:11px;color:#555;font-weight:600">배치 ID</th>
            <th style="padding:7px 8px;text-align:left;border-bottom:1.5px solid #ddd;font-size:11px;color:#555;font-weight:600">방식</th>
            <th style="padding:7px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555;font-weight:600">재배대</th>
            <th style="padding:7px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555;font-weight:600">파종일</th>
            <th style="padding:7px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555;font-weight:600">수확예정일</th>
            <th style="padding:7px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555;font-weight:600">총 재배일</th>
            <th style="padding:7px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555;font-weight:600">판/거터</th>
            <th style="padding:7px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555;font-weight:600">주당 무게</th>
            <th style="padding:7px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555;font-weight:600">로스율</th>
            <th style="padding:7px 8px;text-align:right;border-bottom:1.5px solid #ddd;font-size:11px;color:#555;font-weight:600">예측 주수</th>
            <th style="padding:7px 8px;text-align:right;border-bottom:1.5px solid #ddd;font-size:11px;color:#27500A;font-weight:600">예측 무게</th>
            <th style="padding:7px 8px;text-align:right;border-bottom:1.5px solid #ddd;font-size:11px;color:#185FA5;font-weight:600">실적</th>
            <th style="padding:7px 8px;text-align:right;border-bottom:1.5px solid #ddd;font-size:11px;color:#185FA5;font-weight:600">실제 주당무게</th>
            <th style="padding:7px 8px;text-align:left;border-bottom:1.5px solid #ddd;font-size:11px;color:#555;font-weight:600">비고</th>
          </tr>
        </thead>
        <tbody>{detail_rows}</tbody>
      </table>
      </div>
      {na_warn}{actual_note}
    </div>
    """


def render_monthly_html(df_m: pd.DataFrame, loss_rate_pct: int, default_weight_g: int) -> str:
    if df_m.empty:
        return "<p>표시할 데이터가 없습니다.</p>"

    temp = df_m.copy()
    temp["ym"] = temp["harvest_date"].dt.to_period("M")
    months = sorted(temp["ym"].unique())
    month_blocks = ""

    for ym in months:
        sub = temp[temp["ym"] == ym].sort_values(["harvest_date", "batch_id"])
        m_pp = int(sub["predicted_plants"].sum(skipna=True))
        m_pk = round(float(sub["predicted_kg"].sum(skipna=True)), 1)
        m_ap_raw = sub["actual_yield"].sum(skipna=True)
        m_ak_raw = sub["actual_weight_kg"].sum(skipna=True)
        has_ak = sub["actual_weight_kg"].notna().any()
        m_ap = int(m_ap_raw) if m_ap_raw else None
        m_ak = round(float(m_ak_raw), 1) if has_ak else None
        n_na = int(sub["predicted_plants"].isna().sum())

        actual_summary = ""
        if has_ak and m_ak is not None:
            diff = round(float(m_ak) - float(m_pk), 1)
            sign = "+" if diff >= 0 else ""
            diff_c = "#C0DD97" if diff >= 0 else "#F09595"
            actual_summary = f"""
            <div style="text-align:center">
              <div style="font-size:10px;color:#C0DD97;margin-bottom:4px">실제 무게</div>
              <div style="font-size:18px;font-weight:500;color:#EAF3DE">{m_ak:.1f} kg</div>
            </div>
            <div style="text-align:center">
              <div style="font-size:10px;color:#C0DD97;margin-bottom:4px">오차</div>
              <div style="font-size:18px;font-weight:500;color:{diff_c}">{sign}{diff} kg</div>
            </div>
            """

        batch_rows = ""
        for _, row in sub.iterrows():
            bt_lbl = "고정" if row["bed_type"] == "fixed" else "MGS"
            bt_bg = "#EAF3DE" if row["bed_type"] == "fixed" else "#E6F1FB"
            bt_fg = "#27500A" if row["bed_type"] == "fixed" else "#0C447C"
            hdate = fmt_date_short(row["harvest_date"])
            sdate = fmt_date_short(row["sow_date"])
            tdays = f"{int(row['total_days'])}일" if pd.notna(row["total_days"]) else "—"
            tg = int(row["tray_or_gutter"]) if pd.notna(row["tray_or_gutter"]) else "—"
            wpg = f"{int(row['weight_per_plant_g'])}g" if pd.notna(row["weight_per_plant_g"]) else "—"
            loss_p = f"{float(row['loss_rate']) * 100:.0f}%"
            pp_s = f"{int(row['predicted_plants']):,}주" if pd.notna(row["predicted_plants"]) else "<span style='color:#aaa'>N/A</span>"
            pk_s = f"{float(row['predicted_kg']):.1f} kg" if pd.notna(row["predicted_kg"]) else "<span style='color:#aaa'>N/A</span>"
            ap_s = f"{int(row['actual_yield']):,}주" if pd.notna(row["actual_yield"]) else "—"
            ak_s = f"{float(row['actual_weight_kg']):.1f} kg" if pd.notna(row["actual_weight_kg"]) else "—"
            awpg = compute_actual_weight_g(row)
            ak_cell = ak_s
            if pd.notna(row["actual_weight_kg"]) and pd.notna(row["predicted_kg"]):
                d_val = round(float(row["actual_weight_kg"]) - float(row["predicted_kg"]), 1)
                sign = "+" if d_val >= 0 else ""
                clr = "#27500A" if d_val >= 0 else "#A32D2D"
                ak_cell = f"{ak_s}<br><span style='font-size:10px;color:{clr}'>{sign}{d_val} kg</span>"
            note_txt = str(row.get("note", "")) if pd.notna(row.get("note", "")) else ""
            batch_rows += f"""
            <tr style="border-bottom:0.5px solid #eee">
              <td style="padding:7px 8px;font-family:monospace;font-size:11px">{row['batch_id']}</td>
              <td style="padding:7px 8px"><span style="background:{bt_bg};color:{bt_fg};padding:2px 7px;border-radius:10px;font-size:11px;font-weight:600">{bt_lbl}</span></td>
              <td style="padding:7px 8px;text-align:center">{row['bed_id']}</td>
              <td style="padding:7px 8px;text-align:center">{sdate}</td>
              <td style="padding:7px 8px;text-align:center">{hdate}</td>
              <td style="padding:7px 8px;text-align:center">{tdays}</td>
              <td style="padding:7px 8px;text-align:center">{tg}</td>
              <td style="padding:7px 8px;text-align:center">{wpg}</td>
              <td style="padding:7px 8px;text-align:center">{loss_p}</td>
              <td style="padding:7px 8px;text-align:right;font-weight:500">{pp_s}</td>
              <td style="padding:7px 8px;text-align:right;font-weight:500;color:#27500A">{pk_s}</td>
              <td style="padding:7px 8px;text-align:right;color:#185FA5">{ap_s}</td>
              <td style="padding:7px 8px;text-align:right">{ak_cell}</td>
              <td style="padding:7px 8px;text-align:right;color:#185FA5">{f'{awpg:.1f} g' if awpg is not None else '—'}</td>
              <td style="padding:7px 8px;font-size:11px;color:#999">{note_txt}</td>
            </tr>
            """

        na_note = f"<span style='font-size:11px;color:#854F0B'> · N/A {n_na}건 포함</span>" if n_na else ""
        month_blocks += f"""
        <div style="margin-bottom:24px">
          <div style="background:#27500A;border-radius:12px;padding:12px 20px;display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
            <div style="font-size:16px;font-weight:500;color:#EAF3DE">{ym.year}년 {ym.month}월{na_note}</div>
            <div style="display:flex;gap:24px">
              <div style="text-align:center"><div style="font-size:10px;color:#C0DD97;margin-bottom:2px">예측 주수</div><div style="font-size:18px;font-weight:500;color:#EAF3DE">{m_pp:,}주</div></div>
              <div style="text-align:center"><div style="font-size:10px;color:#C0DD97;margin-bottom:2px">예측 무게</div><div style="font-size:18px;font-weight:500;color:#EAF3DE">{m_pk:.1f} kg</div></div>
              {actual_summary}
              <div style="text-align:center"><div style="font-size:10px;color:#C0DD97;margin-bottom:2px">배치 수</div><div style="font-size:18px;font-weight:500;color:#EAF3DE">{len(sub)}건</div></div>
            </div>
          </div>
          <div style="overflow-x:auto">
          <table style="width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap">
            <thead>
              <tr style="background:#f7f7f7">
                <th style="padding:6px 8px;text-align:left;border-bottom:1.5px solid #ddd;font-size:11px;color:#555">배치 ID</th>
                <th style="padding:6px 8px;text-align:left;border-bottom:1.5px solid #ddd;font-size:11px;color:#555">방식</th>
                <th style="padding:6px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555">재배대</th>
                <th style="padding:6px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555">파종일</th>
                <th style="padding:6px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555">수확예정일</th>
                <th style="padding:6px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555">총 재배일</th>
                <th style="padding:6px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555">판/거터</th>
                <th style="padding:6px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555">주당 무게</th>
                <th style="padding:6px 8px;text-align:center;border-bottom:1.5px solid #ddd;font-size:11px;color:#555">로스율</th>
                <th style="padding:6px 8px;text-align:right;border-bottom:1.5px solid #ddd;font-size:11px;color:#555">예측 주수</th>
                <th style="padding:6px 8px;text-align:right;border-bottom:1.5px solid #ddd;font-size:11px;color:#27500A">예측 무게</th>
                <th style="padding:6px 8px;text-align:right;border-bottom:1.5px solid #ddd;font-size:11px;color:#185FA5">실제 주수</th>
                <th style="padding:6px 8px;text-align:right;border-bottom:1.5px solid #ddd;font-size:11px;color:#185FA5">실제 무게</th>
                <th style="padding:6px 8px;text-align:right;border-bottom:1.5px solid #ddd;font-size:11px;color:#185FA5">실제 주당무게</th>
                <th style="padding:6px 8px;text-align:left;border-bottom:1.5px solid #ddd;font-size:11px;color:#555">비고</th>
              </tr>
            </thead>
            <tbody>{batch_rows}</tbody>
          </table>
          </div>
        </div>
        """

    total_all_pp = int(temp["predicted_plants"].sum(skipna=True))
    total_all_pk = round(float(temp["predicted_kg"].sum(skipna=True)), 1)
    return f"""
    <div style="font-family:-apple-system,sans-serif;max-width:980px">
      <div style="font-size:18px;font-weight:500;margin-bottom:4px">달별 전체 예측 현황</div>
      <div style="font-size:11px;color:#999;margin-bottom:6px">DB 전체 {len(temp)}건 · 로스율 {loss_rate_pct}% · 주당 기본 무게 {default_weight_g}g</div>
      <div style="display:inline-flex;gap:20px;background:#f4f4f4;border-radius:8px;padding:10px 16px;margin-bottom:20px">
        <div><span style="font-size:11px;color:#888">전체 예측 주수 &nbsp;</span><strong>{total_all_pp:,}주</strong></div>
        <div><span style="font-size:11px;color:#888">전체 예측 무게 &nbsp;</span><strong style="color:#27500A">{total_all_pk:.1f} kg</strong></div>
        <div><span style="font-size:11px;color:#888">기간 &nbsp;</span><strong>{months[0]} ~ {months[-1]}</strong></div>
      </div>
      {month_blocks}
    </div>
    """


def build_notion_markdown(prediction_date: date, target: pd.DataFrame, total_pp: int, total_pk: float) -> str:
    lines = [
        f"## 수확량 예측 — {prediction_date} 기준",
        "",
        "| 수확 예정일 | 방식 | 재배대 | 파종일 | 총 재배일 | 예측 주수 | 예측 무게(kg) | 주당 무게(g) | 로스율 | 실적(kg) | 비고 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in target.sort_values(["harvest_date", "batch_id"]).iterrows():
        hd = fmt_date_long(row["harvest_date"])
        sd = fmt_date_long(row["sow_date"])
        td = f"{int(row['total_days'])}일" if pd.notna(row["total_days"]) else "—"
        btype = "고정 재배대" if row["bed_type"] == "fixed" else "MGS"
        pp = f"{int(row['predicted_plants']):,}주" if pd.notna(row["predicted_plants"]) else "N/A"
        pk = f"{float(row['predicted_kg']):.1f}" if pd.notna(row["predicted_kg"]) else "N/A"
        wpg = f"{int(row['weight_per_plant_g'])}g" if pd.notna(row["weight_per_plant_g"]) else "—"
        loss = f"{float(row['loss_rate']) * 100:.0f}%" if pd.notna(row["loss_rate"]) else "—"
        ak = f"{float(row['actual_weight_kg']):.1f}" if pd.notna(row["actual_weight_kg"]) else "—"
        note = str(row.get("note", "")) if pd.notna(row.get("note", "")) else ""
        lines.append(f"| {hd} | {btype} | {row['bed_id']} | {sd} | {td} | {pp} | {pk} | {wpg} | {loss} | {ak} | {note} |")
    lines.append(f"| **합계** | | | | | **{total_pp:,}주** | **{total_pk:.1f}** | | | | |")
    return "\n".join(lines)


def build_capacity_df(default_weight_g: int, loss_rate_pct: int) -> pd.DataFrame:
    yield_rate = 1 - (loss_rate_pct / 100)
    rows = []
    for bid, trays in FIXED_BED_CONFIG.items():
        max_p = trays * PLANTS_PER_TRAY
        pred_p = round(max_p * yield_rate)
        pred_k = round(pred_p * default_weight_g / 1000, 1)
        rows.append(
            {
                "재배대": f"{bid}번",
                "판 수": trays,
                "최대 식재": max_p,
                f"예측 주수({100-loss_rate_pct}%)": pred_p,
                f"예측 kg({default_weight_g}g/주)": pred_k,
            }
        )
    cap_df = pd.DataFrame(rows)
    tot_row = {
        "재배대": "합계",
        "판 수": int(cap_df["판 수"].sum()),
        "최대 식재": int(cap_df["최대 식재"].sum()),
        f"예측 주수({100-loss_rate_pct}%)": int(cap_df[f"예측 주수({100-loss_rate_pct}%)"].sum()),
        f"예측 kg({default_weight_g}g/주)": round(float(cap_df[f"예측 kg({default_weight_g}g/주)"].sum()), 1),
    }
    return pd.concat([cap_df, pd.DataFrame([tot_row])], ignore_index=True)


# =========================
# 사이드바 설정
# =========================
st.title("🌿 식물공장 상추 수확량 예측 시스템")
st.caption("Google Sheets 연동 · 주수 + kg · 3일 대시보드 · 달별 전체 뷰 · 실적 반영")

with st.sidebar:
    st.markdown("### ⚙️ 설정")
    sheet_url = st.text_input("Google Sheets 링크", value=SHEET_URL_DEFAULT)
    prediction_date = st.date_input("예측 기준일", value=date.today())
    loss_rate_pct = st.number_input("로스율 (%)", min_value=0, max_value=100, value=20, step=1)
    default_weight_g = st.number_input("주당 기본 무게 (g)", min_value=1, max_value=1000, value=100, step=1)
    mgs_total_gutters = st.number_input("MGS 총 거터 수 (선택)", min_value=0, max_value=100000, value=0, step=1)
    mgs_total_gutters = None if mgs_total_gutters == 0 else int(mgs_total_gutters)

    dash_dates = [prediction_date, prediction_date + timedelta(days=3), prediction_date + timedelta(days=4)]
    st.markdown("---")
    st.markdown("### 📅 예측 범위")
    st.write(f"- 오늘: {dash_dates[0]}")
    st.write(f"- D+3: {dash_dates[1]}")
    st.write(f"- D+4: {dash_dates[2]}")
    st.write(f"- 수확률: {100-loss_rate_pct}%")

    st.markdown("---")
    info = get_service_account_info()
    if info:
        st.markdown("<span class='status-ok'>쓰기 가능</span>", unsafe_allow_html=True)
        st.caption(f"service account: {info.get('client_email', '')}")
    else:
        st.markdown("<span class='status-warn'>읽기 전용 / 미연결</span>", unsafe_allow_html=True)
        st.caption("st.secrets 미설정 시 public CSV 읽기만 시도합니다.")


# =========================
# 데이터 로드
# =========================
try:
    raw_df, is_writable, source_mode = read_db(sheet_url)
except Exception as exc:
    st.error(f"Google Sheets 로드 실패: {exc}")
    st.stop()

loss_rate = loss_rate_pct / 100

db_df = normalize_db(
    raw_df,
    default_weight_g=default_weight_g,
    default_loss_rate=loss_rate,
    mgs_total_gutters=mgs_total_gutters,
)
pred_df = calc_predictions(db_df)
target_dates = [prediction_date + timedelta(days=3), prediction_date + timedelta(days=4)]
target_df = pred_df[pred_df["harvest_date"].dt.date.isin(target_dates)].copy()
monthly_df = pred_df[pred_df["harvest_date"].notna()].copy()

total_pp = int(target_df["predicted_plants"].sum(skipna=True)) if not target_df.empty else 0
total_pk = round(float(target_df["predicted_kg"].sum(skipna=True)), 1) if not target_df.empty else 0.0

mode_label = {
    "service_account": "Google Sheets API (읽기/쓰기)",
    "public_csv": "공개 CSV (읽기 전용)",
    "unavailable": "연결 실패",
}.get(source_mode, source_mode)

col_a, col_b, col_c = st.columns([1.2, 1, 1])
with col_a:
    st.info(f"데이터 소스: {mode_label}")
with col_b:
    st.metric("DB 배치 수", f"{len(pred_df):,}건")
with col_c:
    st.metric("이번 주 예측", f"{total_pk:.1f} kg")

if source_mode == "unavailable":
    st.warning("Google Sheets를 읽지 못했습니다. 링크 공개 상태, DB 시트명(DB_배치데이터), 또는 service account 설정을 확인하세요.")
    read_errors = st.session_state.get("sheet_read_errors", [])
    with st.expander("연결 실패 원인 보기"):
        st.write("아래 중 하나일 가능성이 큽니다.")
        st.write("1) Streamlit Secrets에 서비스 계정 정보가 없음")
        st.write("2) Google Sheet가 공개 CSV로 열리지 않음")
        st.write("3) DB 탭 이름이 'DB_배치데이터'와 다름")
        st.write("4) 서비스 계정 이메일이 이 스프레드시트에 편집자로 공유되지 않음")
        if read_errors:
            st.code("
".join(read_errors))


# =========================
# 탭 UI
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 3일 대시보드",
    "📅 달별 전체 예측",
    "📋 배치 DB 관리",
    "✍️ 실적 입력",
    "📋 노션 마크다운",
    "📚 용량표",
])


with tab1:
    if pred_df.empty:
        st.warning("DB가 비어 있습니다.")
    else:
        st.markdown(render_dashboard_html(prediction_date, loss_rate_pct, default_weight_g, pred_df, target_df), unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.download_button(
                "D+3~4 예측 CSV 다운로드",
                data=to_str_df(target_df).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"prediction_{prediction_date}.csv",
                mime="text/csv",
            )
        with c2:
            if st.button("현재 D+3~4 결과를 예측결과_log 시트에 저장", use_container_width=True, disabled=not is_writable):
                ok, msg = append_prediction_log(sheet_url, prediction_date, target_df)
                (st.success if ok else st.error)(msg)


with tab2:
    if monthly_df.empty:
        st.warning("월별로 보여줄 데이터가 없습니다.")
    else:
        st.markdown(render_monthly_html(monthly_df, loss_rate_pct, default_weight_g), unsafe_allow_html=True)


with tab3:
    st.markdown("### 현재 DB")
    view_df = pred_df.copy()
    view_df["actual_weight_per_plant_g"] = view_df.apply(compute_actual_weight_g, axis=1)
    show_cols = [
        "batch_id", "bed_type", "bed_id", "sow_date", "transplant_date", "plant_date", "harvest_date",
        "grow_days", "tray_or_gutter", "weight_per_plant_g", "loss_rate", "predicted_plants", "predicted_kg",
        "actual_yield", "actual_weight_kg", "actual_weight_per_plant_g", "note"
    ]
    st.dataframe(to_str_df(view_df[show_cols]), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 배치 추가 / 업데이트")
    with st.form("add_batch_form"):
        c1, c2, c3, c4 = st.columns(4)
        batch_id = c1.text_input("batch_id")
        bed_type = c2.selectbox("방식", ["fixed", "mgs"])
        bed_id = c3.text_input("재배대 / 구역", placeholder="예: 1 / MGS-A")
        tray_or_gutter = c4.number_input("판 수 / 거터 수", min_value=0.0, value=0.0, step=1.0)

        c5, c6, c7, c8 = st.columns(4)
        sow_date = c5.date_input("파종일", value=None)
        transplant_date = c6.date_input("이식일", value=None)
        plant_date = c7.date_input("정식일", value=None)
        harvest_date = c8.date_input("수확 예정일", value=None)

        c9, c10, c11 = st.columns(3)
        grow_days = c9.number_input("생육일수", min_value=0, value=0, step=1)
        weight_per_plant_g = c10.number_input("주당 예상 무게(g)", min_value=0, value=int(default_weight_g), step=1)
        loss_rate_input_pct = c11.number_input("배치별 로스율(%)", min_value=0, max_value=100, value=int(loss_rate_pct), step=1)
        note = st.text_input("비고")
        submitted = st.form_submit_button("배치 저장", use_container_width=True, disabled=not is_writable)

    if submitted:
        if not batch_id.strip():
            st.error("batch_id는 필수입니다.")
        else:
            new_row = {
                "batch_id": batch_id.strip(),
                "sow_date": sow_date.isoformat() if sow_date else "",
                "transplant_date": transplant_date.isoformat() if transplant_date else "",
                "plant_date": plant_date.isoformat() if plant_date else "",
                "harvest_date": harvest_date.isoformat() if harvest_date else "",
                "grow_days": grow_days or None,
                "bed_type": bed_type,
                "bed_id": bed_id,
                "tray_or_gutter": None if tray_or_gutter == 0 else int(tray_or_gutter),
                "weight_per_plant_g": None if weight_per_plant_g == default_weight_g else int(weight_per_plant_g),
                "loss_rate": round(loss_rate_input_pct / 100, 4),
                "actual_yield": None,
                "actual_weight_kg": None,
                "note": note,
            }
            save_df = ensure_columns(raw_df.copy())
            save_df = save_df[save_df["batch_id"].astype(str) != batch_id.strip()].copy()
            save_df = pd.concat([save_df, pd.DataFrame([new_row])], ignore_index=True)
            ok, msg = write_db(sheet_url, save_df)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown("### 배치 삭제")
    delete_candidates = pred_df["batch_id"].dropna().astype(str).tolist()
    delete_ids = st.multiselect("삭제할 batch_id", options=delete_candidates)
    if st.button("선택 배치 삭제", disabled=(not is_writable or not delete_ids)):
        save_df = ensure_columns(raw_df.copy())
        save_df = save_df[~save_df["batch_id"].astype(str).isin(delete_ids)].copy()
        ok, msg = write_db(sheet_url, save_df)
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)


with tab4:
    st.markdown("### 실적 입력")
    if pred_df.empty:
        st.warning("실적을 입력할 DB가 없습니다.")
    else:
        edit_df = pred_df[["batch_id", "harvest_date", "predicted_plants", "predicted_kg", "actual_yield", "actual_weight_kg", "note"]].copy()
        edit_df["harvest_date"] = edit_df["harvest_date"].dt.strftime("%Y-%m-%d")
        edit_df = edit_df.sort_values(["harvest_date", "batch_id"])
        edited = st.data_editor(
            edit_df,
            use_container_width=True,
            hide_index=True,
            disabled=["batch_id", "harvest_date", "predicted_plants", "predicted_kg"],
            column_config={
                "actual_yield": st.column_config.NumberColumn("실제 주수", min_value=0, step=1),
                "actual_weight_kg": st.column_config.NumberColumn("실제 무게(kg)", min_value=0.0, step=0.1, format="%.1f"),
            },
        )
        if st.button("실적 저장", use_container_width=True, disabled=not is_writable):
            save_df = ensure_columns(raw_df.copy())
            edited_map = edited.set_index("batch_id")[["actual_yield", "actual_weight_kg", "note"]].to_dict(orient="index")
            for idx in save_df.index:
                bid = str(save_df.at[idx, "batch_id"])
                if bid in edited_map:
                    save_df.at[idx, "actual_yield"] = edited_map[bid]["actual_yield"]
                    save_df.at[idx, "actual_weight_kg"] = edited_map[bid]["actual_weight_kg"]
                    save_df.at[idx, "note"] = edited_map[bid]["note"]
            ok, msg = write_db(sheet_url, save_df)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


with tab5:
    st.markdown("### 노션 붙여넣기용 마크다운")
    markdown_text = build_notion_markdown(prediction_date, target_df, total_pp, total_pk)
    st.code(markdown_text, language="markdown")
    st.download_button(
        "마크다운 파일 다운로드",
        data=markdown_text.encode("utf-8"),
        file_name=f"notion_prediction_{prediction_date}.md",
        mime="text/markdown",
    )


with tab6:
    st.markdown("### 고정 재배대 전체 용량표")
    cap_df = build_capacity_df(default_weight_g, loss_rate_pct)
    st.dataframe(cap_df, use_container_width=True, hide_index=True)
    st.caption(f"1판 = {PLANTS_PER_TRAY}주 · 기본 {default_weight_g}g/주 · 수확률 {100-loss_rate_pct}%")
