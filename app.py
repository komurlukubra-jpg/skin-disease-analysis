# app.py

import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib
from skimage.feature import hog

st.set_page_config(
    page_title="Cilt Foto Analizi — DEMO (Ödev)",
    page_icon="💗",
    layout="wide"
)

# PEMBE TEMA 

st.markdown("""
<style>

/* Arka plan */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #ffe4ec, #ffd6ea, #fff1f8);
}

/* Sidebar arka planı */
[data-testid="stSidebar"] {
    background: #fff0f7;
}

/* Başlık font & renkleri */
h1, h2, h3, h4 {
    font-family: "Montserrat", sans-serif;
}

/* Info / warning kutuları */
.stAlert {
    border-radius: 12px !important;
}

/* File uploader kartı */
[data-testid="stFileUploader"] {
    border: 2px dashed #d81b60 !important;
    padding: 20px;
    border-radius: 16px;
    background: #ffffffcc;
}

/* Slider */
.stSlider > div[data-baseweb="slider"] > div {
    background: #f8bbd0 !important;
}
.stSlider > div[data-baseweb="slider"] > div > div {
    background: #d81b60 !important;
}

/* DataFrame container */
[data-testid="stDataFrame"] {
    background: #fff5fa !important;
    border-radius: 12px !important;
    padding: 8px;
}

/* Bar chart arka planı */
[data-testid="stVerticalBlock"] .plot-container {
    background: #fff0f7 !important;
    border-radius: 12px !important;
}

/* Genel buton stili */
div.stButton > button {
    background-color: #d81b60 !important;
    color: white !important;
    border-radius: 999px;
    padding: 0.6rem 1.4rem;
    border: none;
    font-weight: 600;
    font-size: 15px;
}

/* Üst kart (header) */
.app-header-card {
    background: #ffffffcc;
    padding: 18px 22px;
    border-radius: 20px;
    border-left: 6px solid #d81b60;
}

/* Küçük etiket */
.badge {
    display: inline-block;
    background: #fce4ec;
    color: #ad1457;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .03em;
}

/* Küçük yazı */
.note-text {
    font-size: 13px;
    color: #666;
}

/* Mobilde biraz margin */
@media only screen and (max-width: 600px) {
    .app-header-card {
        margin-top: 12px;
    }
}

</style>
""", unsafe_allow_html=True)


IMG_SIZE = (256, 256)

@st.cache_resource
def load_assets():
    model = joblib.load("models/model.joblib")
    with open("models/classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)
    return model, classes

def extract_features(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)

  
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.float32) / 255.0
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )


    hsv = Image.fromarray(arr).convert("HSV")
    hsv = np.array(hsv)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hist, _ = np.histogramdd(
        [h.ravel(), s.ravel(), v.ravel()],
        bins=(8, 8, 8),
        range=((0, 255), (0, 255), (0, 255))
    )
    color_feat = (hist.flatten() / (hist.sum() + 1e-8)).astype(np.float32)

    return np.concatenate([hog_feat.astype(np.float32), color_feat], axis=0)


st.markdown(
    """
<div class="app-header-card">
    <span class="badge">DEMO / Ödev Projesi</span>
    <h1 style="margin-bottom:0.3rem;">💗 Cilt Foto Analizi — DEMO</h1>
    <p class="note-text">
        Bu uygulama, makine öğrenmesi dersi kapsamında hazırlanmış eğitim amaçlı bir demodur.
        <b>Tıbbi tanı için kullanılamaz.</b>
    </p>
</div>
""",
    unsafe_allow_html=True
)

st.write("")  

try:
    model, CLASSES = load_assets()
except Exception:
    st.error("Model bulunamadı. Lütfen önce terminalde `python train.py` komutuyla modeli eğitin.")
    st.stop()


with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    threshold = st.slider(
        "Emin değilim eşiği:",
        0.0, 0.9, 0.50, 0.05,
        help="Bu değerin altındaki olasılıklarda model 'emin değilim' diyecek."
    )
    st.markdown("---")
    st.markdown(
        """
        **Nasıl kullanılır?**
        1. Yakın plan bir cilt fotoğrafı seçin 📸  
        2. Uygulamaya yükleyin  
        3. Aşağıda tahmini ve sınıf olasılıklarını görün  
        """
    )
    st.markdown("---")
    st.markdown(
        "<span class='note-text'>Not: Sınıflar, eğitim verinizdeki klasör isimlerine göre belirlenmiştir.</span>",
        unsafe_allow_html=True
    )

st.markdown("### 📸 Bir cilt fotoğrafı yükleyin")

file = st.file_uploader(
    "JPG, JPEG, PNG, BMP veya WEBP formatında bir görsel seçin:",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if file:
    img = Image.open(file)

    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.markdown("#### Görsel Önizleme")
        st.image(img, caption="Yüklenen görsel", use_container_width=True)

    with right_col:
        st.markdown("#### 🔍 Model Tahmini")

        feat = extract_features(img)[None, ...]
        probs = model.predict_proba(feat)[0]
        top = int(np.argmax(probs))
        top_prob = float(probs[top])
        top_label = CLASSES[top]

      
        if top_prob < threshold:
            st.warning(
                f"En olası sınıf: **{top_label}** ({top_prob:.2f}). "
                f"Ama seçtiğiniz eşik ({threshold:.2f}) altında → **emin değilim**."
            )
        else:
            st.success(
                f"Tahmin: **{top_label}** — Olasılık: {top_prob:.2f} "
                f"(Eşik: {threshold:.2f})"
            )

      
        st.markdown(
            f"""
            <div style="
                background:#fff0f7;
                padding:15px;
                border-radius:16px;
                border-left:6px solid #d81b60;
                font-size:17px;
                margin-top:12px;">
            💗 <b>Özet</b><br>
            Model bu görüntü için en yüksek olasılığı <b>{top_label}</b> sınıfına vermiştir. <br>
            Hesaplanan olasılık: <b>{top_prob:.2f}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📊 Sınıf Olasılıkları")

    df = pd.DataFrame(
        {"class": CLASSES, "probability": probs}
    ).sort_values("probability", ascending=False)

    # Tablo
    st.dataframe(
        df.style.format({"probability": "{:.2f}"}),
        use_container_width=True,
        height=320
    )

    # Grafik
    st.bar_chart(df.set_index("class"))

else:
    st.info("Başlamak için sol taraftan bir cilt fotoğrafı yükleyin 💕")
