import streamlit as st
import fitz  # PyMuPDF
import easyocr
from deep_translator import GoogleTranslator
import numpy as np
from PIL import Image, ImageDraw, ImageFont, Image
import io, math, gc, re

# Max image size disable
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Multi-page Structural PDF Translator", layout="wide")

# ---------- UI CSS ----------
st.markdown("""
<style>
.main-header {
    font-size: 2.6rem !important;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1.5rem;
}
.stButton > button {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    color: white;
    border-radius: 999px;
    border: none;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: scale(1.04);
    box-shadow: 0 8px 16px rgba(0,0,0,0.18);
}
</style>
""", unsafe_allow_html=True)

# ---------- Helper functions ----------

@st.cache_resource
def load_ocr_reader():
    # CPU mode only (no GPU, no explicit torch version)
    return easyocr.Reader(['ch_sim', 'en'], gpu=False)

def get_dynamic_font(font_size):
    safe_size = max(8, min(int(font_size * 0.55), 45))
    try:
        return ImageFont.truetype("arial.ttf", safe_size)
    except:
        return ImageFont.load_default()

def skip_drawing_text(text: str) -> bool:
    text = text.strip()
    if len(text) < 2:
        return True

    # symbols typical for dimensions / arrows
    skip_chars = ['→','↑','↓','↗','↘','Φ','Ø','±','×','÷','°','^','|','/','\\']
    if any(c in text for c in skip_chars):
        return True

    # mostly numbers → dimension
    digits = sum(c.isdigit() for c in text)
    if len(text) > 0 and digits / len(text) > 0.7:
        return True

    # short codes like "A1", "B3"
    if re.fullmatch(r'[A-Z0-9]{1,3}', text):
        return True

    return False

# ---------- Layout ----------

st.markdown('<h1 class="main-header">🏗️ Structural PDF Translator Pro</h1>', unsafe_allow_html=True)
st.markdown("Chinese / English structural drawings ကို English ပြန်ဖတ်ပေးမယ့် tool ပါ။")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_ths = st.slider("OCR Confidence threshold", 0.4, 0.9, 0.6, 0.05)
    preview_first_page = st.checkbox("Preview first page", True)
    st.markdown("**Tip:** Confidence အနည်းငယ်ချိုမြော့ထားရင် စာပိုများထပ်ကွင်းမိပါလိမ့်မယ်။")

col_main, col_info = st.columns([2, 1])

with col_main:
    uploaded_file = st.file_uploader("📁 Structural Drawing PDF တင်ပါ", type=["pdf"])

with col_info:
    st.subheader("📌 Features")
    st.write("- Drawing lines မဖုံးဘဲ text overlay only")
    st.write("- Dimensions / symbols မပြန်ပေးဘူး")
    st.write("- Multi-page PDF support")
    st.write("- English overlay PDF download လုပ်လို့ရ")

# ---------- Main logic ----------

if uploaded_file is not None:
    # Load OCR once
    if 'reader' not in st.session_state:
        with st.spinner("🤖 AI OCR model ကို ပြင်ဆင်နေပါသည်..."):
            st.session_state.reader = load_ocr_reader()

    if st.button("🚀 စာမျက်နှာ အားလုံးကို ဘာသာပြန်မည်"):
        try:
            translator = GoogleTranslator(source='zh-CN', target='en')

            # Read PDF bytes
            pdf_bytes = uploaded_file.read()
            uploaded_file.seek(0)

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)

            output_pdf = fitz.open()
            status = st.empty()
            progress = st.progress(0.0)

            if preview_first_page:
                prev_col1, prev_col2 = st.columns(2)
            else:
                prev_col1 = prev_col2 = None

            for page_index in range(total_pages):
                status.info(f"📄 Page {page_index + 1}/{total_pages} ကို ဘာသာပြန်နေပါသည်...")

                page = doc[page_index]

                # resolution scale
                scale = 2.5 if page.rect.width > 2000 else 3.0
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("RGBA")
                draw = ImageDraw.Draw(img)
                img_np = np.array(img.convert("RGB"))

                # OCR
                try:
                    results = st.session_state.reader.readtext(
                        img_np,
                        detail=1,
                        paragraph=False,
                    )
                except Exception as e:
                    st.error(f"OCR error on page {page_index+1}: {e}")
                    results = []

                # Translate and overlay
                for bbox, text, prob in results:
                    if prob < confidence_ths:
                        continue
                    if skip_drawing_text(text):
                        continue

                    (tl, tr, br, bl) = bbox
                    h = math.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
                    if h <= 8:
                        continue

                    try:
                        src_txt = text.strip()
                        if not src_txt:
                            continue
                        translated = translator.translate(src_txt)
                    except Exception:
                        continue

                    if not translated or len(translated) > 60:
                        continue

                    font = get_dynamic_font(h)
                    # semi-transparent white strip behind text
                    draw.rectangle(
                        [tl[0], tl[1], br[0], tl[1] + h * 0.9],
                        fill=(255, 255, 255, 200)
                    )
                    draw.text((tl[0], tl[1] - 2), translated, fill="black", font=font)

                # Preview first page only
                if preview_first_page and page_index == 0:
                    with prev_col1:
                        st.image(img_np, caption="📐 Original (Page 1)", use_column_width=True)
                    with prev_col2:
                        st.image(np.array(img.convert("RGB")), caption="✅ Translated (Page 1)", use_column_width=True)

                # Save page to output PDF
                rgb_img = img.convert("RGB")
                buf = io.BytesIO()
                rgb_img.save(buf, format="PNG", optimize=True, quality=95)
                buf_value = buf.getvalue()

                new_page = output_pdf.new_page(width=page.rect.width, height=page.rect.height)
                new_page.insert_image(page.rect, stream=buf_value)

                # cleanup
                del img, img_np, pix, results, rgb_img, buf
                gc.collect()

                progress.progress((page_index + 1) / total_pages)

            status.success(f"✅ စာမျက်နှာ {total_pages} ခုလုံး ဘာသာပြန်ပြီးပါပြီ!")
            st.balloons()

            st.download_button(
                "⬇️ ဘာသာပြန်ပြီးသား PDF ကို Download လုပ်ရန်",
                data=output_pdf.tobytes(),
                file_name=f"translated_structural_{total_pages}pages.pdf",
                mime="application/pdf"
            )

            output_pdf.close()
            doc.close()

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("GitHub repo ထဲ app.py / requirements.txt ကို မိမိ ပြန်စစ်ကြည့်ပါ။")
else:
    st.info("📁 ပထမဆုံး Structural drawing PDF ဖိုင် တင်ပါ။")
