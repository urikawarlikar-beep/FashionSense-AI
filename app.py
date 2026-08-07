import pickle
from pathlib import Path
import os
import requests
import json

import pandas as pd
import scipy.sparse
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "styles.csv"
CACHE_DF = ROOT_DIR / "fashion_data.pkl"
CACHE_MATRIX = ROOT_DIR / "feature_matrix.npz"
CACHE_VECTORIZER = ROOT_DIR / "tfidf_vectorizer.pkl"

FEATURE_COLUMNS = [
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "usage",
]

# ---------- Ollama Setup ----------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ollama_available = False

def check_ollama():
    """Check if Ollama is running and the model is available"""
    global ollama_available, OLLAMA_MODEL
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            # Check if our model is available, or use the first available
            if any(OLLAMA_MODEL in name for name in model_names):
                ollama_available = True
                print(f"✅ Ollama connected with model: {OLLAMA_MODEL}")
            elif model_names:
                # Use the first available model
                OLLAMA_MODEL = model_names[0].split(":")[0]
                ollama_available = True
                print(f"✅ Ollama connected with model: {OLLAMA_MODEL}")
            else:
                print("⚠️ No models found in Ollama")
        else:
            print(f"⚠️ Ollama server returned status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("⚠️ Ollama not running. Please start Ollama with: ollama serve")
    except Exception as e:
        print(f"⚠️ Ollama initialisation error: {e}")

# Check Ollama availability on startup
check_ollama()


def build_feature_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna("").astype(str)
    df["features"] = df[FEATURE_COLUMNS].agg(" ".join, axis=1)
    return df


@st.cache_resource
def load_model() -> tuple[pd.DataFrame, scipy.sparse.spmatrix, TfidfVectorizer]:
    if CACHE_DF.exists() and CACHE_MATRIX.exists() and CACHE_VECTORIZER.exists():
        df = pickle.load(open(CACHE_DF, "rb"))
        feature_matrix = scipy.sparse.load_npz(CACHE_MATRIX)
        vectorizer = pickle.load(open(CACHE_VECTORIZER, "rb"))
    else:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH, on_bad_lines="skip")
        df = build_feature_text(df)
        vectorizer = TfidfVectorizer(stop_words="english")
        feature_matrix = vectorizer.fit_transform(df["features"])

        pickle.dump(df, open(CACHE_DF, "wb"))
        scipy.sparse.save_npz(CACHE_MATRIX, feature_matrix)
        pickle.dump(vectorizer, open(CACHE_VECTORIZER, "wb"))

    return df, feature_matrix, vectorizer


def recommend(product_name: str, df: pd.DataFrame, feature_matrix: scipy.sparse.spmatrix, top_n: int = 5) -> pd.DataFrame:
    matches = df.index[df["productDisplayName"] == product_name].tolist()
    if not matches:
        return pd.DataFrame()

    idx = matches[0]
    sim_scores = cosine_similarity(feature_matrix[idx], feature_matrix).flatten()
    indices = sim_scores.argsort()[::-1]
    top_indices = [i for i in indices if i != idx][:top_n]

    recommendations = df.loc[top_indices].copy()
    recommendations["similarity"] = sim_scores[top_indices]
    return recommendations.reset_index(drop=True)


def product_details(df: pd.DataFrame, product_name: str) -> dict:
    row = df[df["productDisplayName"] == product_name]
    if row.empty:
        return {}
    return {
        "Product": row.iloc[0]["productDisplayName"],
        "Gender": row.iloc[0]["gender"],
        "Category": row.iloc[0]["masterCategory"],
        "Subcategory": row.iloc[0]["subCategory"],
        "Article Type": row.iloc[0]["articleType"],
        "Base Colour": row.iloc[0]["baseColour"],
        "Season": row.iloc[0]["season"],
        "Usage": row.iloc[0]["usage"],
    }


# ---------- Ollama AI Functions ----------
def generate_ollama_response(prompt: str) -> str:
    """Generate a response using Ollama"""
    if not ollama_available:
        return "🤖 Ollama not available. Please make sure Ollama is running with: ollama serve"

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 800
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated")
        else:
            return f"⚠️ Ollama error: Status {response.status_code}"
    except requests.exceptions.Timeout:
        return "⚠️ Ollama request timed out. The model might be slow to respond."
    except Exception as e:
        return f"⚠️ Ollama error: {str(e)}"


def ai_stylist(user: dict, product: pd.Series) -> str:
    if not ollama_available:
        return (
            "🤖 Ollama not available. Please make sure Ollama is running.\n\n"
            "For now, here's a generic tip: This outfit is versatile and can be dressed up or down. "
            "Pair it with neutral sneakers and a crossbody bag for a casual look."
        )

    product_info = f"""
    Product Name: {product['productDisplayName']}
    Gender: {product['gender']}
    Category: {product['masterCategory']}
    Sub Category: {product['subCategory']}
    Article Type: {product['articleType']}
    Color: {product['baseColour']}
    Season: {product['season']}
    Usage: {product['usage']}
    """

    prompt = f"""You are an AI personal fashion stylist.

USER PROFILE:
Gender: {user['gender']}
Age: {user['age']}
Skin Tone: {user['skin_tone']}
Body Type: {user['body_type']}
Occasion: {user['occasion']}
Season: {user['season']}
Preferred Style: {user['style_preference']}
Budget: {user['budget']}

PRODUCT:
{product_info}

Provide a simple, friendly fashion recommendation covering:
1. Why this outfit suits the user
2. Matching colours
3. Footwear recommendation
4. Accessories
5. Two alternative outfit ideas
6. One useful fashion tip

Keep it short and practical."""

    return generate_ollama_response(prompt)


def analyze_outfit(description: str, user: dict) -> str:
    """Analyze an outfit description using Ollama"""
    if not ollama_available:
        return "🤖 Ollama not available. Please make sure Ollama is running."

    prompt = f"""You are an AI fashion stylist. A user describes their outfit as:

"{description}"

User profile:
Gender: {user['gender']}
Age: {user['age']}
Skin Tone: {user['skin_tone']}
Body Type: {user['body_type']}
Occasion: {user['occasion']}
Season: {user['season']}
Preferred Style: {user['style_preference']}
Budget: {user['budget']}

Based on this description, give friendly, practical fashion advice covering:
1. Overall style assessment
2. Do the colours and pieces match well?
3. Suggested footwear
4. Suggested accessories
5. Best occasion for this outfit
6. What could be improved
7. Two alternative outfit ideas

Keep the advice concise and positive."""

    return generate_ollama_response(prompt)


# ------------------ UI HELPERS ------------------
def render_page_style() -> None:
    page_style = """
    <style>
    /* Main App Background - Deep Burgundy */
    .stApp {
        background: linear-gradient(135deg, #4A0015 0%, #6B0018 30%, #4A0015 70%, #2D000B 100%) !important;
    }
    
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('https://images.unsplash.com/photo-1558769132-cb1aea458c5e?q=80&w=1974&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0.15;
        z-index: 0;
        pointer-events: none;
    }
    
    /* Main content area */
    .main > div {
        position: relative;
        z-index: 1;
    }
    
    /* Sidebar - Deep Burgundy with Dark Red borders */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #4A0015 0%, #6B0018 100%) !important;
        backdrop-filter: blur(20px);
        border-right: 3px solid #800020 !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
        backdrop-filter: blur(20px);
    }
    
    /* Sidebar text - Black */
    [data-testid="stSidebar"] .stMarkdown {
        color: #000000 !important;
    }
    
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #000000 !important;
    }
    
    /* Sidebar select boxes */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 2px solid #800020 !important;
        border-radius: 8px;
        color: #000000 !important;
    }
    
    [data-testid="stSidebar"] .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 2px solid #800020 !important;
        color: #000000 !important;
    }
    
    /* Sidebar select dropdown text */
    [data-testid="stSidebar"] .stSelectbox > div > div > div {
        color: #000000 !important;
    }
    
    /* Sidebar caption */
    [data-testid="stSidebar"] .stCaption {
        color: #000000 !important;
    }
    
    /* Radio buttons - Dark Red accent */
    .stRadio > div {
        background: rgba(74, 0, 21, 0.85) !important;
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 10px;
        border: 2px solid #800020;
    }
    
    .stRadio label {
        color: #000000 !important;
    }
    
    .stRadio [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    
    /* Soft Light Pink Cards - FashionSense AI, Product Overview */
    .hero-block, .glass-panel, .glass-card {
        background: linear-gradient(135deg, #FFF0F5 0%, #FFE4ED 30%, #FFF0F5 70%, #FFD6E5 100%) !important;
        backdrop-filter: blur(10px);
        border: 3px solid #800020;
        border-radius: 25px;
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        padding: 2.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-block::after, .glass-panel::after, .glass-card::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, transparent 0%, rgba(255, 255, 255, 0.3) 50%, transparent 100%);
        pointer-events: none;
    }
    
    /* TEXT COLORS FOR SOFT PINK CARDS - BLACK for main, DARK BURGUNDY for headings */
    .hero-title {
        color: #1A1A1A !important;
        font-size: clamp(2.8rem, 4vw, 4.2rem);
        letter-spacing: -0.02em;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.08);
    }
    
    .hero-subtitle {
        color: #1A1A1A !important;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Product overview cards - Soft Light Pink with Black text */
    .product-overview {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 24px;
        margin-top: 20px;
    }
    
    .overview-card {
        background: linear-gradient(135deg, #FFF0F5 0%, #FFE4ED 30%, #FFF0F5 70%, #FFD6E5 100%) !important;
        border-radius: 20px;
        padding: 1.5rem;
        border: 3px solid #800020;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }
    
    .overview-card h4 {
        color: #4A0015 !important;
        margin-bottom: 1rem;
        letter-spacing: 0.08em;
        font-weight: 800;
        font-size: 1.3rem;
        text-transform: uppercase;
        border-bottom: 2px solid #800020;
        padding-bottom: 0.5rem;
    }
    
    .overview-item {
        display: flex;
        justify-content: space-between;
        padding: 0.7rem 0;
        border-bottom: 1px solid rgba(128, 0, 32, 0.3);
    }
    
    .overview-label {
        color: #1A1A1A !important;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .overview-value {
        font-weight: 700;
        color: #1A1A1A !important;
        font-size: 1rem;
    }
    
    /* Section title - Black on dark background */
    .section-title {
        color: #000000 !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Table styling - Soft Pink with Dark Red borders */
    .table-container {
        padding: 1.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #FFF0F5 0%, #FFE4ED 30%, #FFF0F5 70%, #FFD6E5 100%) !important;
        border: 3px solid #800020;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(18px);
        margin-top: 24px;
    }
    
    table.custom-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    table.custom-table th {
        border-bottom: 3px solid #800020 !important;
        padding: 1rem 1.2rem;
        background: rgba(74, 0, 21, 0.05);
        color: #4A0015 !important;
        text-align: left;
        font-size: 0.95rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 800;
    }
    
    table.custom-table td {
        border-bottom: 1px solid rgba(128, 0, 32, 0.3) !important;
        padding: 0.9rem 1.2rem;
        background: transparent !important;
        color: #1A1A1A !important;
        font-size: 0.95rem;
        font-weight: 600;
    }
    
    table.custom-table tr:hover td {
        background: rgba(128, 0, 32, 0.08) !important;
    }
    
    /* Buttons - Dark Red */
    .stButton>button {
        background: linear-gradient(135deg, #800020, #6B0018) !important;
        color: #FFFFFF !important;
        border: 2px solid #A31A3C !important;
        border-radius: 30px;
        padding: 0.7rem 2.5rem;
        font-weight: 700;
        box-shadow: 0 6px 20px rgba(128, 0, 32, 0.4);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.03);
        background: linear-gradient(135deg, #A31A3C, #800020) !important;
        box-shadow: 0 8px 25px rgba(128, 0, 32, 0.6);
        border-color: #C41E3A !important;
    }
    
    /* Recommendation cards - Soft Light Pink with Dark Red borders */
    .recommendation-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 24px;
        margin-top: 20px;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #FFF0F5 0%, #FFE4ED 30%, #FFF0F5 70%, #FFD6E5 100%) !important;
        border-radius: 20px;
        padding: 1.5rem;
        border: 3px solid #800020;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(4px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(128, 0, 32, 0.3);
    }
    
    .recommendation-card h4 {
        color: #4A0015 !important;
        margin-bottom: 0.5rem;
        font-weight: 800;
        font-size: 1.1rem;
    }
    
    .recommendation-card p {
        color: #1A1A1A !important;
        font-weight: 600;
    }
    
    .recommendation-card .similarity-score {
        color: #4A0015 !important;
        font-weight: 800;
        font-size: 1.1rem;
    }
    
    /* General text colors - Black on dark background */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #000000 !important;
    }
    
    .stMarkdown p, .stMarkdown span {
        color: #000000 !important;
    }
    
    /* Select box styling - Soft Pink with Dark Red border */
    .stSelectbox > div > div {
        background: rgba(255, 240, 245, 0.95) !important;
        border: 2px solid #800020 !important;
        border-radius: 10px;
        color: #1A1A1A !important;
    }
    
    .stSelectbox > div > div > div {
        color: #1A1A1A !important;
    }
    
    /* Text input styling - Soft Pink with Dark Red border */
    .stTextInput > div > div > input {
        background: rgba(255, 240, 245, 0.95) !important;
        border: 2px solid #800020 !important;
        border-radius: 10px;
        color: #1A1A1A !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #666666 !important;
    }
    
    /* Text area styling - Soft Pink with Dark Red border */
    .stTextArea > div > div > textarea {
        background: rgba(255, 240, 245, 0.95) !important;
        border: 2px solid #800020 !important;
        border-radius: 10px;
        color: #1A1A1A !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #666666 !important;
    }
    
    /* Labels for inputs - Black */
    .stSelectbox label, .stTextInput label, .stTextArea label {
        color: #000000 !important;
    }
    
    /* Warning and info messages */
    .stAlert {
        background: rgba(255, 240, 245, 0.95) !important;
        border-left: 4px solid #800020 !important;
        color: #1A1A1A !important;
    }
    
    .stAlert .stMarkdown p {
        color: #1A1A1A !important;
    }
    
    /* Divider - Dark Red */
    hr {
        border-color: rgba(128, 0, 32, 0.4) !important;
        border-width: 2px !important;
    }
    
    /* Spinner - Dark Red */
    .stSpinner > div {
        border-color: #800020 !important;
    }
    
    /* File uploader - Soft Pink with Dark Red dashed border */
    .stFileUploader > div {
        background: rgba(255, 240, 245, 0.1) !important;
        border: 2px dashed #800020 !important;
        border-radius: 15px;
    }
    
    .stFileUploader > div > div {
        color: #000000 !important;
    }
    
    /* Camera input styling */
    .stCameraInput > div {
        border: 2px solid #800020 !important;
        border-radius: 15px;
    }
    
    /* Homepage text - Black on soft pink */
    .homepage-title {
        color: #1A1A1A !important;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .homepage-subtitle {
        color: #1A1A1A !important;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Image captions */
    .stImage figcaption {
        color: #000000 !important;
    }
    
    /* Column text in recommendations page */
    .recommendation-product-name {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .recommendation-product-details {
        color: #4A0015 !important;
        font-size: 0.9rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(74, 0, 21, 0.7) !important;
        border: 2px solid #800020 !important;
        border-radius: 10px 10px 0 0 !important;
        color: #000000 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 240, 245, 0.9) !important;
        color: #4A0015 !important;
        border-bottom: 3px solid #800020 !important;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .product-overview {
            grid-template-columns: 1fr;
        }
        
        .hero-title {
            font-size: 2rem;
        }
    }
    </style>
    """
    st.markdown(page_style, unsafe_allow_html=True)


def render_table(df: pd.DataFrame) -> str:
    table_html = df.to_html(classes="custom-table", index=False, escape=False)
    return f"<div class='table-container'>{table_html}</div>"


def render_recommendation_cards(recommendations: pd.DataFrame) -> None:
    cards_html = [
        "<div class='recommendation-card'>"
        f"<h4>{row['productDisplayName']}</h4>"
        f"<p>{row['articleType']} · {row['baseColour']} · {row['season']} · {row['usage']}</p>"
        f"<p class='similarity-score'>Similarity: {row['similarity']:.3f}</p>"
        "</div>"
        for _, row in recommendations.iterrows()
    ]
    st.markdown("".join(cards_html), unsafe_allow_html=True)


# ------------------ MAIN APP ------------------
def main() -> None:
    st.set_page_config(page_title="FashionSense AI", layout="wide", page_icon="👗")
    render_page_style()

    with st.sidebar:
        st.markdown("### 👤 Your Style Profile")
        gender = st.selectbox("Gender", ["Women", "Men"])
        age = st.number_input("Age", 15, 60, 20)
        skin_tone = st.selectbox("Skin Tone", ["Fair", "Medium", "Dark"])
        body_type = st.selectbox("Body Type", ["Hourglass", "Pear", "Rectangle", "Apple", "Athletic"])
        occasion = st.selectbox("Occasion", ["College", "Casual", "Party", "Formal", "Wedding", "Date"])
        style_preference = st.selectbox("Preferred Style", ["Casual", "Formal", "Streetwear", "Ethnic", "Party", "Sporty"])
        budget = st.selectbox("Budget", ["Low", "Medium", "High", "Luxury"])
        st.divider()
        
        # Show Ollama status
        if ollama_available:
            st.success(f"✅ Ollama ({OLLAMA_MODEL}) connected")
        else:
            st.warning("⚠️ Ollama not connected")
            st.caption("Run: `ollama serve` and `ollama pull llama3.2`")
        
        st.caption("✨ FashionSense AI v3.0 (Ollama)")

    user_profile = {
        "gender": gender,
        "age": age,
        "skin_tone": skin_tone,
        "body_type": body_type,
        "occasion": occasion,
        "season": "Summer",
        "style_preference": style_preference,
        "budget": budget
    }

    page = st.radio(
        "",
        ["🏠 Home", "👗 Recommendations", "🤖 AI Stylist", "📸 Outfit Analyzer"],
        horizontal=True,
        index=0
    )

    with st.spinner("Loading the recommendation engine..."):
        df, feature_matrix, _ = load_model()

    if page == "🏠 Home":
        st.markdown(
            "<div class='hero-block'>"
            "<h1 class='hero-title'>FashionSense AI</h1>"
            "<p class='hero-subtitle'>Discover premium product matches with a clean, visual-first recommendation experience inspired by high-end design systems.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        product_list = sorted(df["productDisplayName"].astype(str).unique())
        selected_product = st.selectbox(
            "Choose a product you like:", 
            product_list, 
            index=0
        )

        selected_details = product_details(df, selected_product)
        if selected_details:
            st.markdown(
                "<div class='product-overview'>"
                "<div class='overview-card'>"
                "<h4>Product overview</h4>"
                f"<div class='overview-item'><span class='overview-label'>Product</span><span class='overview-value'>{selected_details['Product']}</span></div>"
                f"<div class='overview-item'><span class='overview-label'>Category</span><span class='overview-value'>{selected_details['Category']}</span></div>"
                f"<div class='overview-item'><span class='overview-label'>Subcategory</span><span class='overview-value'>{selected_details['Subcategory']}</span></div>"
                f"<div class='overview-item'><span class='overview-label'>Gender</span><span class='overview-value'>{selected_details['Gender']}</span></div>"
                f"<div class='overview-item'><span class='overview-label'>Colour</span><span class='overview-value'>{selected_details['Base Colour']}</span></div>"
                "</div>"
                "<div class='overview-card'>"
                "<h4>Design details</h4>"
                f"<div class='overview-item'><span class='overview-label'>Season</span><span class='overview-value'>{selected_details['Season']}</span></div>"
                f"<div class='overview-item'><span class='overview-label'>Usage</span><span class='overview-value'>{selected_details['Usage']}</span></div>"
                f"<div class='overview-item'><span class='overview-label'>Matches</span><span class='overview-value'>Top 5 curated results</span></div>"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        if st.button("✨ Recommend"):
            with st.spinner("Finding the closest style matches..."):
                recommendations = recommend(selected_product, df, feature_matrix)

            if recommendations.empty:
                st.warning("No recommendations found for the selected item.")
            else:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<p class='section-title'>✨ Recommended matches</p>", unsafe_allow_html=True)

                table_html = render_table(
                    recommendations[
                        ["productDisplayName", "articleType", "baseColour", "season", "usage", "similarity"]
                    ].rename(columns={
                        "productDisplayName": "Product",
                        "articleType": "Article Type",
                        "baseColour": "Colour",
                    })
                )
                st.markdown(table_html, unsafe_allow_html=True)

                st.markdown("<div class='recommendation-grid'>", unsafe_allow_html=True)
                render_recommendation_cards(recommendations)
                st.markdown("</div>", unsafe_allow_html=True)

    elif page == "👗 Recommendations":
        st.header("✨ Recommended For You")
        usage_map = {
            "Casual": "Casual",
            "Formal": "Formal",
            "Party": "Party",
            "Ethnic": "Ethnic",
            "Sporty": "Sports",
            "Streetwear": "Casual"
        }
        usage_filter = usage_map.get(style_preference, None)
        filtered = df[(df["gender"] == gender) & (df["usage"] == usage_filter)] if usage_filter else df[df["gender"] == gender]

        if len(filtered) == 0:
            st.warning("No exact matches found. Try another style.")
        else:
            products = filtered.sample(min(8, len(filtered)))
            cols = st.columns(4)
            for i, (_, row) in enumerate(products.iterrows()):
                with cols[i % 4]:
                    img_path = ROOT_DIR / "images" / f"{row['id']}.jpg"
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/150?text=No+Image", width=150)
                    st.markdown(f"<p class='recommendation-product-name'>{row['productDisplayName']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='recommendation-product-details'>{row['baseColour']} • {row['articleType']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='recommendation-product-details'>Usage: {row['usage']}</p>", unsafe_allow_html=True)

    elif page == "🤖 AI Stylist":
        st.markdown("<h2>🤖 Chat with Your AI Stylist</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #000000;'>Ask anything about fashion, or get advice on a specific product.</p>", unsafe_allow_html=True)
        
        if not ollama_available:
            st.warning("⚠️ Ollama is not connected. Please make sure Ollama is running with: `ollama serve`")
            st.info("💡 You can still use the recommendation features without AI.")
        
        product_names = df["productDisplayName"].unique()
        selected_product_name = st.selectbox("Choose a product (optional)", ["None"] + list(product_names))
        question = st.text_input("What would you like to know?")

        if st.button("✨ Get Advice"):
            if question or selected_product_name != "None":
                with st.spinner("AI is thinking..."):
                    if selected_product_name != "None":
                        product_row = df[df["productDisplayName"] == selected_product_name].iloc[0]
                        advice = ai_stylist(user_profile, product_row)
                    else:
                        # Use a generic product if none selected
                        generic_product = pd.Series({
                            "productDisplayName": "a versatile piece",
                            "gender": gender,
                            "masterCategory": "",
                            "subCategory": "",
                            "articleType": "",
                            "baseColour": "",
                            "season": "",
                            "usage": ""
                        })
                        advice = ai_stylist(user_profile, generic_product)
                    st.markdown("---")
                    st.markdown("<h3>💬 AI Stylist Says:</h3>", unsafe_allow_html=True)
                    st.write(advice)
            else:
                st.warning("Please ask a question or select a product.")

    elif page == "📸 Outfit Analyzer":
        st.markdown("<h2>📸 Outfit Analyzer</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #000000;'>Upload a photo (optional) and describe your outfit in the text box below. The AI will give you style feedback.</p>", unsafe_allow_html=True)
        
        if not ollama_available:
            st.warning("⚠️ Ollama is not connected. Please make sure Ollama is running with: `ollama serve`")

        uploaded_file = st.file_uploader("Upload your outfit photo (optional)", type=["jpg", "jpeg", "png"])
        camera_photo = st.camera_input("📷 Take a photo of your outfit (optional)")

        image = None
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Outfit", use_container_width=True)
        elif camera_photo:
            image = Image.open(camera_photo)
            st.image(image, caption="Camera Snapshot", use_container_width=True)

        outfit_description = st.text_area(
            "Describe your outfit in detail (colours, pieces, style):",
            placeholder="e.g., I'm wearing a white button-down shirt, dark blue jeans, and brown leather boots."
        )

        if st.button("🔍 Analyze My Outfit"):
            if not outfit_description.strip():
                st.warning("Please describe your outfit first.")
            else:
                with st.spinner("Analysing your style..."):
                    result = analyze_outfit(outfit_description, user_profile)
                    st.markdown("---")
                    st.markdown("<h3>📋 AI Stylist Report</h3>", unsafe_allow_html=True)
                    st.write(result)


if __name__ == "__main__":
    main()