import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go

@st.cache_resource
def load_model():
    model  = joblib.load("models/model_kalori_mlp.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

HISTORY_FILE = "data/riwayat_kalori.csv"
os.makedirs("data", exist_ok=True)

KATEGORI_MAKANAN = {
    "🍚 Makanan Pokok": {
        "Nasi putih (1 piring)":          {"carbs": 52,  "protein": 4,   "fats": 0.4},
        "Nasi goreng (1 piring)":         {"carbs": 55,  "protein": 8,   "fats": 12},
        "Nasi uduk (1 piring)":           {"carbs": 50,  "protein": 5,   "fats": 8},
        "Nasi kuning (1 piring)":         {"carbs": 48,  "protein": 5,   "fats": 7},
        "Nasi padang (1 porsi)":          {"carbs": 60,  "protein": 20,  "fats": 18},
        "Mie goreng (1 porsi)":           {"carbs": 60,  "protein": 9,   "fats": 14},
        "Mie rebus (1 porsi)":            {"carbs": 55,  "protein": 8,   "fats": 8},
        "Mie ayam (1 porsi)":             {"carbs": 58,  "protein": 14,  "fats": 10},
        "Bihun goreng (1 porsi)":         {"carbs": 45,  "protein": 5,   "fats": 8},
        "Roti tawar (2 lembar)":          {"carbs": 26,  "protein": 4,   "fats": 2},
        "Roti bakar (2 lembar)":          {"carbs": 30,  "protein": 4,   "fats": 6},
        "Kentang rebus (1 porsi)":        {"carbs": 30,  "protein": 3,   "fats": 0.1},
        "Kentang goreng (1 porsi)":       {"carbs": 35,  "protein": 3,   "fats": 15},
        "Bubur ayam (1 porsi)":           {"carbs": 35,  "protein": 10,  "fats": 5},
        "Lontong (2 potong)":             {"carbs": 40,  "protein": 2,   "fats": 0.5},
        "Indomie rebus (1 bungkus)":      {"carbs": 52,  "protein": 8,   "fats": 10},
        "Indomie goreng (1 bungkus)":     {"carbs": 50,  "protein": 8,   "fats": 14},
    },
    "🥩 Lauk Hewani": {
        "Ayam goreng (1 potong)":         {"carbs": 0,   "protein": 25,  "fats": 14},
        "Ayam bakar (1 potong)":          {"carbs": 0,   "protein": 27,  "fats": 8},
        "Ayam geprek (1 porsi)":          {"carbs": 5,   "protein": 24,  "fats": 16},
        "Ayam penyet (1 porsi)":          {"carbs": 5,   "protein": 24,  "fats": 15},
        "Ikan goreng (1 ekor)":           {"carbs": 0,   "protein": 22,  "fats": 10},
        "Ikan bakar (1 ekor)":            {"carbs": 0,   "protein": 24,  "fats": 6},
        "Ikan tongkol (1 porsi)":         {"carbs": 0,   "protein": 20,  "fats": 5},
        "Ikan lele goreng (1 ekor)":      {"carbs": 0,   "protein": 18,  "fats": 9},
        "Udang goreng (1 porsi)":         {"carbs": 1,   "protein": 20,  "fats": 8},
        "Telur goreng (1 butir)":         {"carbs": 0,   "protein": 6,   "fats": 7},
        "Telur rebus (1 butir)":          {"carbs": 0,   "protein": 6,   "fats": 5},
        "Telur dadar (1 lembar)":         {"carbs": 1,   "protein": 7,   "fats": 8},
        "Daging sapi (1 porsi)":          {"carbs": 0,   "protein": 26,  "fats": 15},
        "Rendang (1 porsi)":              {"carbs": 4,   "protein": 28,  "fats": 20},
        "Sate ayam (5 tusuk)":            {"carbs": 5,   "protein": 20,  "fats": 10},
        "Bakso (1 mangkok)":              {"carbs": 30,  "protein": 15,  "fats": 8},
        "Soto ayam (1 mangkok)":          {"carbs": 20,  "protein": 18,  "fats": 8},
        "Rawon (1 mangkok)":              {"carbs": 15,  "protein": 22,  "fats": 12},
        "Opor ayam (1 porsi)":            {"carbs": 5,   "protein": 24,  "fats": 18},
    },
    "🌱 Lauk Nabati": {
        "Tahu goreng (2 potong)":         {"carbs": 2,   "protein": 8,   "fats": 6},
        "Tahu bacem (2 potong)":          {"carbs": 8,   "protein": 8,   "fats": 5},
        "Tempe goreng (2 potong)":        {"carbs": 10,  "protein": 10,  "fats": 6},
        "Tempe bacem (2 potong)":         {"carbs": 14,  "protein": 10,  "fats": 5},
        "Tempe mendoan (2 potong)":       {"carbs": 15,  "protein": 9,   "fats": 8},
        "Pecel (1 porsi)":                {"carbs": 18,  "protein": 7,   "fats": 10},
        "Gado-gado (1 porsi)":            {"carbs": 20,  "protein": 8,   "fats": 12},
    },
    "🥦 Sayur & Sup": {
        "Sayur bayam (1 porsi)":          {"carbs": 4,   "protein": 3,   "fats": 0.4},
        "Tumis kangkung (1 porsi)":       {"carbs": 5,   "protein": 2,   "fats": 3},
        "Tumis buncis (1 porsi)":         {"carbs": 8,   "protein": 2,   "fats": 3},
        "Capcay (1 porsi)":               {"carbs": 10,  "protein": 5,   "fats": 4},
        "Sayur asem (1 mangkok)":         {"carbs": 12,  "protein": 3,   "fats": 1},
        "Sayur lodeh (1 mangkok)":        {"carbs": 14,  "protein": 4,   "fats": 6},
        "Sup ayam (1 mangkok)":           {"carbs": 10,  "protein": 12,  "fats": 5},
        "Sup jagung (1 mangkok)":         {"carbs": 18,  "protein": 5,   "fats": 3},
        "Lalapan (1 porsi)":              {"carbs": 3,   "protein": 1,   "fats": 0.2},
    },
    "🍜 Jajanan & Street Food": {
        "Gorengan (2 buah)":              {"carbs": 20,  "protein": 3,   "fats": 10},
        "Martabak telur (1 porsi)":       {"carbs": 30,  "protein": 12,  "fats": 15},
        "Martabak manis (1 porsi)":       {"carbs": 55,  "protein": 6,   "fats": 18},
        "Pisang goreng (2 buah)":         {"carbs": 35,  "protein": 2,   "fats": 8},
        "Siomay (1 porsi)":               {"carbs": 20,  "protein": 10,  "fats": 6},
        "Batagor (1 porsi)":              {"carbs": 22,  "protein": 10,  "fats": 10},
        "Pempek (2 buah)":                {"carbs": 28,  "protein": 10,  "fats": 5},
        "Ketoprak (1 porsi)":             {"carbs": 45,  "protein": 10,  "fats": 8},
        "Kupat tahu (1 porsi)":           {"carbs": 48,  "protein": 10,  "fats": 9},
    },
    "🍌 Buah-buahan": {
        "Pisang (1 buah)":                {"carbs": 23,  "protein": 1,   "fats": 0.3},
        "Apel (1 buah)":                  {"carbs": 19,  "protein": 0.5, "fats": 0.2},
        "Jeruk (1 buah)":                 {"carbs": 15,  "protein": 1,   "fats": 0.2},
        "Mangga (1 buah)":                {"carbs": 25,  "protein": 1,   "fats": 0.4},
        "Semangka (1 potong)":            {"carbs": 12,  "protein": 1,   "fats": 0.2},
        "Pepaya (1 potong)":              {"carbs": 14,  "protein": 0.6, "fats": 0.1},
        "Alpukat (1/2 buah)":             {"carbs": 9,   "protein": 2,   "fats": 15},
        "Melon (1 potong)":               {"carbs": 10,  "protein": 0.5, "fats": 0.1},
        "Jambu biji (1 buah)":            {"carbs": 14,  "protein": 1,   "fats": 0.5},
    },
    "🥤 Minuman": {
        "Air putih (1 gelas)":            {"carbs": 0,   "protein": 0,   "fats": 0},
        "Susu sapi (1 gelas)":            {"carbs": 12,  "protein": 8,   "fats": 8},
        "Susu kedelai (1 gelas)":         {"carbs": 14,  "protein": 7,   "fats": 4},
        "Teh manis (1 gelas)":            {"carbs": 20,  "protein": 0,   "fats": 0},
        "Kopi susu (1 gelas)":            {"carbs": 15,  "protein": 2,   "fats": 4},
        "Jus jeruk (1 gelas)":            {"carbs": 26,  "protein": 1,   "fats": 0.5},
        "Jus alpukat (1 gelas)":          {"carbs": 24,  "protein": 3,   "fats": 14},
        "Es teh manis (1 gelas)":         {"carbs": 22,  "protein": 0,   "fats": 0},
    },
    "🍰 Camilan & Dessert": {
        "Biskuit (5 keping)":             {"carbs": 22,  "protein": 2,   "fats": 5},
        "Keripik singkong (1 bungkus)":   {"carbs": 30,  "protein": 2,   "fats": 10},
        "Keripik kentang (1 bungkus)":    {"carbs": 28,  "protein": 2,   "fats": 12},
        "Coklat (1 batang kecil)":        {"carbs": 25,  "protein": 2,   "fats": 8},
        "Es krim (1 scoop)":              {"carbs": 20,  "protein": 2,   "fats": 7},
        "Donat (1 buah)":                 {"carbs": 30,  "protein": 4,   "fats": 12},
        "Onde-onde (2 buah)":             {"carbs": 28,  "protein": 3,   "fats": 6},
        "Bubur kacang hijau (1 mangkok)": {"carbs": 35,  "protein": 6,   "fats": 3},
    },
}

SEMUA_MAKANAN = {}
for kat, items in KATEGORI_MAKANAN.items():
    for nama, nilai in items.items():
        SEMUA_MAKANAN[nama] = {**nilai}

def get_bmi_info(bmi):
    if bmi < 17:
        return "Sangat Kurus", "#b91c1c", "⚠️", ["Tingkatkan asupan kalori 500–700 kcal/hari", "Konsumsi protein tinggi: telur, ayam, ikan, tempe", "Makan 5–6 kali sehari dengan porsi kecil", "Konsultasikan dengan dokter atau ahli gizi"]
    elif bmi < 18.5:
        return "Kurus", "#c2410c", "⚡", ["Tambah kalori 300–500 kcal/hari", "Perbanyak lauk protein bergizi", "Konsumsi camilan sehat: kacang, alpukat, susu", "Makan 3x sehari ditambah 2 camilan"]
    elif bmi < 25:
        return "Normal", "#15803d", "✅", ["Pertahankan pola makan saat ini", "Jaga keseimbangan karbo, protein, dan lemak", "Minum air putih minimal 8 gelas/hari", "Olahraga rutin 30 menit/hari"]
    elif bmi < 30:
        return "Gemuk", "#c2410c", "⚠️", ["Kurangi kalori 300–500 kcal/hari", "Perbanyak sayur dan buah", "Kurangi gorengan dan minuman manis", "Olahraga rutin minimal 45 menit/hari"]
    else:
        return "Obesitas", "#b91c1c", "🚨", ["Konsultasikan dengan dokter segera", "Kurangi kalori 500–700 kcal/hari", "Hindari makanan tinggi lemak jenuh", "Mulai olahraga ringan: jalan kaki 30 menit/hari"]

def save_history(nama, gender, age, height, weight, exercise, kalori_prediksi, kalori_makanan, bmi, tc, tp, tf):
    data = {"Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M"), "Nama": nama, "Gender": gender, "Umur": age, "Tinggi": height, "Berat": weight, "Aktivitas": exercise, "BMI": round(bmi, 2), "Kalori_Prediksi": round(kalori_prediksi, 2), "Kalori_Makanan": round(kalori_makanan, 2), "Karbo_g": round(tc, 1), "Protein_g": round(tp, 1), "Lemak_g": round(tf, 1)}
    df_new = pd.DataFrame([data])
    if os.path.exists(HISTORY_FILE):
        df_all = pd.concat([pd.read_csv(HISTORY_FILE), df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(HISTORY_FILE, index=False)

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame()

if "mode" not in st.session_state:
    st.session_state.mode = "input"
if "hasil_prediksi" not in st.session_state:
    st.session_state.hasil_prediksi = None
if "reset_key" not in st.session_state:
    st.session_state.reset_key = 0

# =========================
# CSS — SAGE BERSIH & TEGAS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}

/* Background utama — sage lembut */
.stApp {
    background-color: #e8f0e8 !important;
}
.block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1080px !important;
}

/* ── TOP NAV ── */
.top-nav {
    background: #2d5a3d;
    border-radius: 16px;
    padding: 1.1rem 1.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.6rem;
}
.nav-left { display: flex; align-items: center; gap: 0.75rem; }
.nav-icon {
    width: 40px; height: 40px;
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
}
.nav-title { font-size: 1.15rem; font-weight: 800; color: #ffffff; letter-spacing: -0.3px; }
.nav-sub { font-size: 0.72rem; color: rgba(255,255,255,0.65); font-weight: 400; margin-top: 1px; }
.nav-date { font-size: 0.75rem; color: rgba(255,255,255,0.55); font-weight: 500; }

/* ── KARTU PUTIH ── */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    border: 1.5px solid #c8d8c8;
    margin-bottom: 1rem;
}

/* ── SECTION TITLE ── */
.sec-title {
    font-size: 0.7rem;
    font-weight: 700;
    color: #3d6b4a;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin: 1.6rem 0 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-title::after {
    content: '';
    flex: 1;
    height: 1.5px;
    background: #c8d8c8;
}

/* ── STAT PILLS ── */
.stat-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin: 0.8rem 0; }
.stat-pill {
    flex: 1; min-width: 105px;
    background: #f2f7f2;
    border: 1.5px solid #c8d8c8;
    border-radius: 12px;
    padding: 0.85rem 1rem;
}
.sp-label {
    font-size: 0.65rem; font-weight: 700;
    color: #5a7a5a; text-transform: uppercase;
    letter-spacing: 0.6px; margin-bottom: 0.25rem;
}
.sp-val {
    font-size: 1.3rem; font-weight: 800;
    color: #1a3322; line-height: 1;
}
.sp-unit { font-size: 0.68rem; color: #7a9a7a; font-weight: 500; }

/* ── BMI STRIP ── */
.bmi-strip {
    background: #f2f7f2;
    border: 1.5px solid #c8d8c8;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    display: flex; align-items: center; gap: 1rem;
    margin: 0.75rem 0;
}
.bmi-icon-big { font-size: 1.8rem; }
.bmi-tag { font-size: 0.65rem; font-weight: 700; color: #5a7a5a; text-transform: uppercase; letter-spacing: 0.6px; }
.bmi-num { font-size: 1.15rem; font-weight: 800; margin: 0.1rem 0; }
.bmi-scale { font-size: 0.72rem; color: #6a8a6a; }

/* ── RESULT HERO ── */
.result-hero {
    background: #2d5a3d;
    border-radius: 18px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin: 0.5rem 0 1.2rem;
}
.rh-label {
    font-size: 0.7rem; font-weight: 700;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase; letter-spacing: 1.4px;
}
.rh-value {
    font-size: 4.5rem; font-weight: 800;
    color: #ffffff; line-height: 1;
    letter-spacing: -3px; margin: 0.3rem 0 0.1rem;
}
.rh-unit { font-size: 1.3rem; color: rgba(255,255,255,0.5); font-weight: 400; }
.rh-badge {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px;
    padding: 0.3rem 1.1rem;
    color: rgba(255,255,255,0.9);
    font-size: 0.82rem; font-weight: 600;
    margin-top: 0.7rem;
}

/* ── FOOD TABLE ── */
.food-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.83rem;
}
.food-table th {
    background: #eef5ee;
    color: #2d5a3d;
    font-weight: 700;
    padding: 0.6rem 0.9rem;
    text-align: left;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    border-bottom: 2px solid #c8d8c8;
}
.food-table td {
    padding: 0.55rem 0.9rem;
    border-bottom: 1px solid #eef5ee;
    color: #1a3322;
    font-weight: 500;
}
.food-table tr:last-child td { border-bottom: none; }
.food-table tr:hover td { background: #f5fbf5; }
.kcal-td { font-weight: 700 !important; color: #2d5a3d !important; }

/* ── SARAN LIST ── */
.saran-wrap { display: flex; flex-direction: column; gap: 0.55rem; margin-top: 0.5rem; }
.saran-row { display: flex; align-items: flex-start; gap: 0.6rem; font-size: 0.85rem; color: #1a3322; font-weight: 500; }
.saran-dot {
    width: 7px; height: 7px;
    border-radius: 50%; background: #4a7c59;
    margin-top: 5px; flex-shrink: 0;
}

/* ── BUTTONS ── */
div[data-testid="stButton"] > button {
    background: #2d5a3d !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.4rem !important;
    letter-spacing: 0.2px !important;
    transition: background 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    background: #3d6e4e !important;
}

/* ── INPUTS label ── */
.stTextInput label,
.stNumberInput label,
.stSelectbox label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: #2d5a3d !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}

/* ── INPUT BOX ── */
div[data-testid="stNumberInput"] input,
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {
    background: #f8fbf8 !important;
    border: 1.5px solid #b8ceb8 !important;
    border-radius: 8px !important;
    color: #1a3322 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* ── EXPANDER — PERBAIKAN UTAMA ── */
details {
    background: #ffffff !important;
    border: 1.5px solid #c8d8c8 !important;
    border-radius: 12px !important;
    margin-bottom: 0.5rem !important;
    overflow: hidden !important;
}
details summary {
    background: #f2f7f2 !important;
    padding: 0.8rem 1rem !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: #1a3322 !important;
    cursor: pointer !important;
    list-style: none !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    border-radius: 10px !important;
}
details summary::-webkit-details-marker { display: none !important; }
details summary::before {
    content: '▶';
    font-size: 0.6rem;
    color: #4a7c59;
    transition: transform 0.2s;
    flex-shrink: 0;
}
details[open] summary::before { transform: rotate(90deg); }
details[open] summary {
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 1.5px solid #c8d8c8 !important;
}
details > div {
    padding: 1rem !important;
    background: #ffffff !important;
}

/* Checkbox label */
.stCheckbox label {
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: #1a3322 !important;
}
.stCheckbox span { color: #1a3322 !important; }

/* ── METRIC ── */
[data-testid="stMetric"] {
    background: #f2f7f2 !important;
    border-radius: 12px !important;
    padding: 0.85rem 1rem !important;
    border: 1.5px solid #c8d8c8 !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    color: #5a7a5a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: #1a3322 !important;
}

/* ── TABS ── */
[data-testid="stTabs"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 0.2rem 0.5rem 0;
    border: 1.5px solid #c8d8c8;
    margin-bottom: 1rem;
}
button[role="tab"] {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: #5a7a5a !important;
    padding: 0.6rem 1.2rem !important;
}
button[role="tab"][aria-selected="true"] {
    color: #2d5a3d !important;
    border-bottom: 2.5px solid #2d5a3d !important;
}

/* ── ALERT ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1.5px solid #c8d8c8 !important;
    border-radius: 10px !important;
}

/* Caption */
.stCaption { color: #5a7a5a !important; font-size: 0.78rem !important; font-weight: 500 !important; }

/* Divider */
hr { border-color: #c8d8c8 !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# TOP NAV
# =========================
today = datetime.now().strftime("%d %B %Y")
st.markdown(f"""
<div class="top-nav">
    <div class="nav-left">
        <div class="nav-icon">🌿</div>
        <div>
            <div class="nav-title">NutriPredict</div>
            <div class="nav-sub">Sistem Prediksi Kebutuhan Kalori Harian</div>
        </div>
    </div>
    <div class="nav-date">📅 {today}</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  🔍  Prediksi Kalori  ", "  📊  Riwayat & Tren  ", "  📖  Panduan  "])

# ================================================
# TAB 1
# ================================================
with tab1:

    # ── TAMPILAN HASIL ──
    if st.session_state.mode == "hasil" and st.session_state.hasil_prediksi:
        r  = st.session_state.hasil_prediksi
        hasil      = r["hasil"]
        bmi        = r["bmi"]
        bmi_status = r["bmi_status"]
        bmi_color  = r["bmi_color"]
        bmi_icon   = r["bmi_icon"]
        saran_list = r["saran_list"]
        tc         = r["tc"]
        tp         = r["tp"]
        tf         = r["tf"]
        tkm        = r["tkm"]
        snap       = r["snap"]

        if hasil < 1500:   kat = "Sangat Rendah — konsultasikan ahli gizi"
        elif hasil < 2000: kat = "Normal — cocok aktivitas ringan"
        elif hasil < 2500: kat = "Normal — cocok aktivitas sedang"
        elif hasil < 3000: kat = "Tinggi — sesuai aktivitas berat"
        else:              kat = "Sangat Tinggi — atlet / kerja fisik berat"

        st.markdown(f"""
        <div class="result-hero">
            <div class="rh-label">Kebutuhan Kalori Harian</div>
            <div class="rh-value">{hasil:.0f} <span class="rh-unit">kcal</span></div>
            <div class="rh-badge">{kat}</div>
        </div>""", unsafe_allow_html=True)

        selisih = hasil - tkm
        c1, c2, c3 = st.columns(3)
        c1.metric("Kalori dari Makanan", f"{tkm:.0f} kcal")
        c2.metric("Kebutuhan Kalori",    f"{hasil:.0f} kcal")
        c3.metric("Selisih", f"{abs(selisih):.0f} kcal",
                  delta=f"{'Kurang' if selisih>0 else 'Lebih'} {abs(selisih):.0f} kcal",
                  delta_color="inverse")

        if selisih > 200:
            st.info(f"💡 Kamu masih kurang **{selisih:.0f} kcal**. Tambahkan makanan bergizi seperti telur, tempe, atau buah.")
        elif selisih < -200:
            st.warning(f"⚠️ Kamu kelebihan **{abs(selisih):.0f} kcal**. Kurangi porsi atau tingkatkan aktivitas fisik.")
        else:
            st.success("🎉 Kalori harianmu sudah seimbang! Pertahankan pola makan ini.")

        st.markdown('<div class="sec-title">Status Berat Badan & Saran Gizi</div>', unsafe_allow_html=True)
        col_g, col_s = st.columns([1, 2])
        with col_g:
            fig_bmi = go.Figure(go.Indicator(
                mode="gauge+number", value=bmi,
                number={"suffix": " kg/m²", "font": {"size": 18, "color": "#1a3322", "family": "Inter"}},
                gauge={
                    "axis": {"range": [10, 40], "tickwidth": 1, "tickcolor": "#aaa", "tickfont": {"size": 9}},
                    "bar": {"color": bmi_color, "thickness": 0.25},
                    "bgcolor": "#f8fbf8",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [10, 18.5], "color": "#fef9c3"},
                        {"range": [18.5, 25], "color": "#dcfce7"},
                        {"range": [25, 30],   "color": "#ffedd5"},
                        {"range": [30, 40],   "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": bmi_color, "width": 3}, "thickness": 0.75, "value": bmi},
                },
                title={"text": f"<b>{bmi_icon} {bmi_status}</b>", "font": {"size": 13, "color": bmi_color, "family": "Inter"}},
            ))
            fig_bmi.update_layout(height=220, margin=dict(t=50, b=0, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
            st.plotly_chart(fig_bmi, use_container_width=True)
        with col_s:
            st.markdown(f"<div style='font-size:0.7rem;font-weight:700;color:#3d6b4a;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.6rem'>Saran untuk Status {bmi_status}</div>", unsafe_allow_html=True)
            saran_html = '<div class="saran-wrap">' + "".join([f'<div class="saran-row"><div class="saran-dot"></div><span>{s}</span></div>' for s in saran_list]) + '</div>'
            st.markdown(saran_html, unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Analisis Nutrisi</div>', unsafe_allow_html=True)
        ideal_c = (hasil * 0.55) / 4
        ideal_p = (hasil * 0.20) / 4
        ideal_f = (hasil * 0.25) / 9
        cp, cb  = st.columns(2)
        with cp:
            vals = [tc, tp, tf] if tc+tp+tf > 0 else [1,1,1]
            fig_pie = go.Figure(go.Pie(
                labels=["Karbohidrat","Protein","Lemak"], values=vals,
                hole=0.55, marker_colors=["#2d5a3d","#6b9e7a","#b8d4b8"],
                textinfo="percent", textfont=dict(size=11, family="Inter"),
                hovertemplate="%{label}: %{value:.1f}g<extra></extra>",
            ))
            fig_pie.add_annotation(text=f"<b>{tc+tp+tf:.0f}g</b><br><span style='font-size:10px'>total</span>",
                x=0.5, y=0.5, showarrow=False, font=dict(size=13, color="#1a3322", family="Inter"))
            fig_pie.update_layout(title=dict(text="<b>Komposisi Nutrisi</b>", font=dict(size=12, color="#2d5a3d", family="Inter")),
                height=270, margin=dict(t=40,b=10,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, font=dict(size=10, family="Inter")))
            st.plotly_chart(fig_pie, use_container_width=True)
        with cb:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name="Aktual", x=["Karbo","Protein","Lemak"], y=[tc,tp,tf],
                marker_color="#2d5a3d", text=[f"{tc:.0f}g",f"{tp:.0f}g",f"{tf:.0f}g"],
                textposition="outside", textfont=dict(size=10, family="Inter")))
            fig_bar.add_trace(go.Bar(name="Ideal", x=["Karbo","Protein","Lemak"], y=[ideal_c,ideal_p,ideal_f],
                marker_color="#b8d4b8", text=[f"{ideal_c:.0f}g",f"{ideal_p:.0f}g",f"{ideal_f:.0f}g"],
                textposition="outside", textfont=dict(size=10, family="Inter")))
            fig_bar.update_layout(title=dict(text="<b>Aktual vs Ideal</b>", font=dict(size=12, color="#2d5a3d", family="Inter")),
                barmode="group", height=270, margin=dict(t=40,b=10,l=0,r=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(242,247,242,0.7)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10, family="Inter")),
                yaxis=dict(showgrid=True, gridcolor="#ddeadd"), xaxis=dict(showgrid=False))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<div class="sec-title">Makanan yang Dikonsumsi</div>', unsafe_allow_html=True)
        rows = ""
        for nm, porsi in snap.items():
            d   = SEMUA_MAKANAN[nm]
            kal = (d["carbs"]*4 + d["protein"]*4 + d["fats"]*9) * porsi
            rows += f"<tr><td>{nm}</td><td style='text-align:center'>{porsi}×</td><td style='text-align:center'>{d['carbs']*porsi:.0f} g</td><td style='text-align:center'>{d['protein']*porsi:.0f} g</td><td style='text-align:center'>{d['fats']*porsi:.0f} g</td><td style='text-align:center' class='kcal-td'>{kal:.0f}</td></tr>"
        st.markdown(f"""
        <div style="border:1.5px solid #c8d8c8;border-radius:12px;overflow:hidden">
        <table class="food-table">
            <thead><tr>
                <th>Makanan</th><th style="text-align:center">Porsi</th>
                <th style="text-align:center">Karbo</th><th style="text-align:center">Protein</th>
                <th style="text-align:center">Lemak</th><th style="text-align:center">Kalori</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb2 = st.columns([3,1])
        with ca:
            if st.button("💾  Simpan ke Riwayat", use_container_width=True):
                save_history(r["nama"], r["gender"], r["age"], r["height"], r["weight"], r["exercise"], hasil, tkm, bmi, tc, tp, tf)
                st.success("✅ Hasil berhasil disimpan ke riwayat!")
        with cb2:
            if st.button("🔄  Prediksi Baru", use_container_width=True):
                st.session_state.mode = "input"
                st.session_state.hasil_prediksi = None
                st.session_state.reset_key += 1
                st.rerun()

    # ── TAMPILAN INPUT ──
    else:
        rk = st.session_state.reset_key

        st.markdown('<div class="sec-title">Data Diri</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            nama   = st.text_input("Nama (Opsional)", placeholder="Masukkan namamu...", key=f"nama_{rk}")
            gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"], key=f"gender_{rk}")
        with c2:
            age    = st.number_input("Umur (Tahun)", min_value=10, max_value=100, value=25, key=f"age_{rk}")
            height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=220, value=165, key=f"height_{rk}")
        with c3:
            weight    = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=60.0, key=f"weight_{rk}")
            meal_freq = st.number_input("Frekuensi Makan / Hari", min_value=1, max_value=10, value=3, key=f"meal_{rk}")
            exercise  = st.selectbox("Aktivitas Fisik", ["Tidak aktif (0)","Sangat ringan (1)","Ringan (2)","Sedang (3)","Tinggi (4)"], key=f"ex_{rk}")

        bmi = weight / ((height/100)**2)
        bmi_status, bmi_color, bmi_icon, saran_list = get_bmi_info(bmi)
        st.markdown(f"""
        <div class="bmi-strip">
            <div class="bmi-icon-big">{bmi_icon}</div>
            <div>
                <div class="bmi-tag">Indeks Massa Tubuh (BMI)</div>
                <div class="bmi-num" style="color:{bmi_color}">{bmi:.1f} kg/m²  —  {bmi_status}</div>
                <div class="bmi-scale">Kurus &lt;18.5 &nbsp;·&nbsp; Normal 18.5–24.9 &nbsp;·&nbsp; Gemuk 25–29.9 &nbsp;·&nbsp; Obesitas ≥30</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Pilih Makanan & Jumlah Porsi</div>', unsafe_allow_html=True)
        st.caption("Buka kategori → centang makanan → atur jumlah porsi yang dikonsumsi")

        porsi_makanan = {}
        for kategori, items in KATEGORI_MAKANAN.items():
            with st.expander(kategori):
                nama_list = list(items.keys())
                for i in range(0, len(nama_list), 2):
                    col_a, col_b = st.columns(2)
                    for col, j in zip([col_a, col_b], [i, i+1]):
                        if j < len(nama_list):
                            nm = nama_list[j]
                            with col:
                                if st.checkbox(nm, key=f"cb_{rk}_{nm}"):
                                    porsi = st.number_input(
                                        "Jumlah Porsi", min_value=1, max_value=10, value=1,
                                        key=f"p_{rk}_{nm}",
                                        help=f"Berapa kali kamu makan {nm} hari ini?"
                                    )
                                    porsi_makanan[nm] = porsi

        tc  = sum(SEMUA_MAKANAN[m]["carbs"]   * p for m,p in porsi_makanan.items())
        tp  = sum(SEMUA_MAKANAN[m]["protein"] * p for m,p in porsi_makanan.items())
        tf  = sum(SEMUA_MAKANAN[m]["fats"]    * p for m,p in porsi_makanan.items())
        tkm = tc*4 + tp*4 + tf*9

        st.markdown('<div class="sec-title">Ringkasan Nutrisi</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill">
                <div class="sp-label">🍚 Karbohidrat</div>
                <div class="sp-val">{tc:.0f}<span class="sp-unit"> g</span></div>
            </div>
            <div class="stat-pill">
                <div class="sp-label">🥩 Protein</div>
                <div class="sp-val">{tp:.0f}<span class="sp-unit"> g</span></div>
            </div>
            <div class="stat-pill">
                <div class="sp-label">🧈 Lemak</div>
                <div class="sp-val">{tf:.0f}<span class="sp-unit"> g</span></div>
            </div>
            <div class="stat-pill">
                <div class="sp-label">🔥 Kalori Makanan</div>
                <div class="sp-val">{tkm:.0f}<span class="sp-unit"> kcal</span></div>
            </div>
            <div class="stat-pill">
                <div class="sp-label">🍽️ Item Dipilih</div>
                <div class="sp-val">{len(porsi_makanan)}<span class="sp-unit"> item</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

        gender_enc   = 0 if gender == "Laki-laki" else 1
        exercise_enc = int(exercise.split("(")[1].replace(")", ""))
        bmr = (88.362 + 13.397*weight + 4.799*height - 5.677*age) if gender=="Laki-laki" else (447.593 + 9.247*weight + 3.098*height - 4.330*age)
        st.markdown(f"<div style='font-size:0.75rem;color:#5a7a5a;font-weight:600;margin:0.4rem 0 1rem'>⚡ BMR: <b style='color:#2d5a3d'>{bmr:.0f} kcal</b> — dihitung otomatis dari data diri</div>", unsafe_allow_html=True)

        if st.button("🌿  Prediksi Kebutuhan Kalori Saya", use_container_width=True):
            if not porsi_makanan:
                st.warning("⚠️ Pilih minimal 1 makanan terlebih dahulu.")
            else:
                inp   = np.array([[gender_enc, age, meal_freq, exercise_enc, height, weight, bmr, tc, tp, tf]])
                hasil = model.predict(scaler.transform(inp))[0]
                st.session_state.hasil_prediksi = {
                    "hasil": hasil, "bmi": bmi, "bmi_status": bmi_status,
                    "bmi_color": bmi_color, "bmi_icon": bmi_icon, "saran_list": saran_list,
                    "tc": tc, "tp": tp, "tf": tf, "tkm": tkm,
                    "nama": nama or "Pengguna", "gender": gender, "age": age,
                    "height": height, "weight": weight, "exercise": exercise,
                    "snap": dict(porsi_makanan),
                }
                st.session_state.mode = "hasil"
                st.rerun()

# ================================================
# TAB 2 — RIWAYAT
# ================================================
with tab2:
    df = load_history()
    st.markdown('<div class="sec-title">Ringkasan Riwayat</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("Belum ada riwayat. Lakukan prediksi dan klik 'Simpan ke Riwayat'.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Prediksi",   len(df))
        c2.metric("Rata-rata Kalori", f"{df['Kalori_Prediksi'].mean():.0f} kcal")
        c3.metric("Tertinggi",        f"{df['Kalori_Prediksi'].max():.0f} kcal")
        c4.metric("Terendah",         f"{df['Kalori_Prediksi'].min():.0f} kcal")

        st.markdown('<div class="sec-title">Tren Kalori Harian</div>', unsafe_allow_html=True)
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=df["Tanggal"], y=df["Kalori_Prediksi"],
            mode="lines+markers", name="Kebutuhan",
            line=dict(color="#2d5a3d", width=2.5), marker=dict(size=7, color="#2d5a3d"),
            fill="tozeroy", fillcolor="rgba(45,90,61,0.07)"))
        fig_t.add_trace(go.Scatter(x=df["Tanggal"], y=df["Kalori_Makanan"],
            mode="lines+markers", name="Konsumsi",
            line=dict(color="#8ab88a", width=2, dash="dot"), marker=dict(size=5, color="#8ab88a")))
        fig_t.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(242,247,242,0.6)", font=dict(family="Inter", color="#1a3322"),
            margin=dict(t=15,b=10,l=0,r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#ddeadd", title="kcal", tickfont=dict(size=10)))
        st.plotly_chart(fig_t, use_container_width=True)

        st.markdown('<div class="sec-title">Tren BMI</div>', unsafe_allow_html=True)
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=df["Tanggal"], y=df["BMI"],
            mode="lines+markers", line=dict(color="#3d6e4e", width=2.5), marker=dict(size=7, color="#3d6e4e"),
            fill="tozeroy", fillcolor="rgba(61,110,78,0.07)"))
        for y_val, clr, lbl in [(18.5,"#ca8a04","Batas Kurus"),(25,"#c2410c","Batas Normal"),(30,"#b91c1c","Batas Gemuk")]:
            fig_b.add_hline(y=y_val, line_dash="dash", line_color=clr, line_width=1.3,
                annotation_text=lbl, annotation_font_size=10, annotation_font_color=clr)
        fig_b.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(242,247,242,0.6)", font=dict(family="Inter", color="#1a3322"),
            margin=dict(t=15,b=10,l=0,r=0), showlegend=False,
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#ddeadd", title="BMI", tickfont=dict(size=10)))
        st.plotly_chart(fig_b, use_container_width=True)

        st.markdown('<div class="sec-title">Tabel Riwayat Lengkap</div>', unsafe_allow_html=True)
        st.dataframe(df.sort_values("Tanggal", ascending=False), use_container_width=True, hide_index=True)

        cx, cy = st.columns([3,1])
        with cx:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Download Riwayat CSV", csv, "riwayat_kalori.csv", "text/csv", use_container_width=True)
        with cy:
            if st.button("🗑️  Hapus Semua", use_container_width=True):
                os.remove(HISTORY_FILE); st.rerun()

# ================================================
# TAB 3 — PANDUAN
# ================================================
with tab3:
    st.markdown('<div class="sec-title">Cara Menggunakan Aplikasi</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Isi data diri** — gender, umur, tinggi, berat, dan frekuensi makan
    2. **Pilih aktivitas fisik** sesuai kebiasaan harianmu
    3. **Centang makanan** yang dikonsumsi hari ini dan atur jumlah porsinya
    4. **Klik Prediksi** — lihat kebutuhan kalori, status BMI, dan grafik nutrisi
    5. **Simpan hasilmu** untuk memantau tren dari hari ke hari
    6. **Prediksi Baru** — tekan tombol reset untuk mengulang dari awal, semua makanan ikut ter-reset
    """)
    st.markdown("---")
    st.markdown('<div class="sec-title">Level Aktivitas Fisik</div>', unsafe_allow_html=True)
    st.markdown("""
    | Level | Keterangan |
    |---|---|
    | 0 — Tidak aktif | Kerja kantoran, hampir tidak bergerak |
    | 1 — Sangat ringan | Sesekali jalan kaki, banyak duduk |
    | 2 — Ringan | Olahraga ringan 1–3× per minggu |
    | 3 — Sedang | Olahraga rutin 3–5× per minggu |
    | 4 — Tinggi | Olahraga berat setiap hari / kerja fisik berat |
    """)
    st.markdown("---")
    st.markdown('<div class="sec-title">Kategori BMI (WHO)</div>', unsafe_allow_html=True)
    st.markdown("""
    | BMI | Status |
    |---|---|
    | < 17.0 | Sangat Kurus |
    | 17.0 – 18.4 | Kurus |
    | 18.5 – 24.9 | ✅ Normal |
    | 25.0 – 29.9 | Gemuk |
    | ≥ 30.0 | Obesitas |
    """)