import streamlit as st
import fitz
import easyocr
from deep_translator import GoogleTranslator
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io, math, gc
import re

# Custom CSS လှလှလေး UI အတွက်
st.markdown("""
<style>
.main-header {
    font-size: 3rem !important;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}
.stButton > button {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    color: white;
    border-radius: 25px;
    border: none;
    padding: 0.8rem 2rem;
    font-weight: bold;
    font-size: 1.1rem;
    transition: all 0.3s;
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 15px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

Image.MAX_IMAGE_PIXELS = None 

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ch_sim', 'en'], gpu=False)

def get_dynamic_font(font_size):
    safe_size = max(8, min(int(font_size * 0.55), 45))
    try:
        return ImageFont.truetype("arial.ttf", safe_size)
    except:
        return ImageFont.load_default()

def skip_drawing_text(text):
    text = text.strip()
    if len(text) < 2: return True
    skip_chars = ['→','↑','↓','Φ','Ø','±','×','÷','°','^']
    if any(char in text for char in skip_chars): return True
    numeric_ratio = sum(c.isdigit() for c in text) / len(text)
    if numeric_ratio > 0.7: return True
    return False

# Header
st.markdown('<h1 class="main-header">🏗️ Structural PDF Translator Pro</h1>', unsafe_allow_html=True)
st.markdown("**Chinese/English → English** ဆွဲပုံဘာသာပြန်ကိရိယာ")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ ဆက်တင်များ")
    confidence = st.slider("OCR Confidence", 0.5, 0.9, 0.6)
    preview_pages = st.checkbox("Preview ပြမည်", value=True)

# Main content
col1, col2 = st.columns([2,1])
with col1:
    uploaded_file = st.file_uploader("📁 Drawing PDF တင်ပါ", type=["pdf"], 
                                   help="ဆွဲပုံ PDF ဖိုင်ကို ရွေးချယ်ပါ")

with col2:
    st.info("✅ **လုပ်ဆောင်ချက်များ**")
    st.success("• Drawing lines မပျက်ပါ")
    st.success("• Dimensions မပြောင်းပါ") 
    st.success("• Multi-page support")

if uploaded_file is not None:
    # OCR Load
    if 'reader' not in st.session_state:
        with st.spinner("🤖 AI Models လုပ်ငန်းစဉ်တက်နေပါသည်..."):
            st.session_state.reader = load_ocr_reader()
    
    # Process button
    if st.button("🚀 အားလုံးကို ဘာသာပြန်မည်", type="primary"):
        with st.container():
            # Progress container
            progress_container = st.container()
            
            translator = GoogleTranslator(source='zh-CN', target='en')
            file_content = uploaded_file.read()
            uploaded_file.seek(0)
            doc = fitz.open(stream=file_content, filetype="pdf")
            total_pages = len(doc)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 စာမျက်နှာ", total_pages)
            with col2:
                st.metric("🤖 OCR Ready", "✅")
            with col3:
                st.metric("🌐 Translator", "Google")
            
            output_pdf = fitz.open()
            
            for page_num in range(total_pages):
                with progress_container:
                    progress_bar = st.progress((page_num + 1) / total_pages)
                    st.info(f"📄 Page {page_num + 1}/{total_pages}")
                
                # Processing logic (same as before but cleaner)
                page = doc[page_num]
                scale_factor = 2.5 if page.rect.width > 2000 else 3.0
                pix = page.get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("RGBA")
                draw = ImageDraw.Draw(img)
                img_np = np.array(img.convert("RGB"))
                
                results = st.session_state.reader.readtext(img_np, detail=1)
                
                for bbox, text, prob in results:
                    if prob > confidence and not skip_drawing_text(text):
                        (tl, tr, br, bl) = bbox
                        h = math.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)
                        if h > 8:
                            try:
                                translated = translator.translate(text.strip())
                                if translated and len(translated) < 50:
                                    font = get_dynamic_font(h)
                                    draw.rectangle([tl[0], tl[1], br[0], tl[1]+h], 
                                                 fill=(255,255,255,180))
                                    draw.text((tl[0], tl[1]-2), translated, fill="black", font=font)
                            except: continue
                
                # Save page
                img_final = img.convert("RGB")
                img_byte_arr = io.BytesIO()
                img_final.save(img_byte_arr, format='PNG', quality=95)
                new_page = output_pdf.new_page(width=page.rect.width, height=page.rect.height)
                new_page.insert_image(page.rect, stream=img_byte_arr.getvalue())
                
                # Preview first page
                if preview_pages and page_num == 0:
                    col_a, col_b = st.columns(2)
                    with col_a: st.image(img_np, caption="📐 Original", use_column_width=True)
                    with col_b: st.image(np.array(img_final), caption="✅ Translated", use_column_width=True)
                
                # Cleanup
                del img, img_np, pix, results
                gc.collect()
            
            # Download
            st.balloons()
            st.success(f"🎉 {total_pages} pages completed!")
            st.download_button(
                label="⬇️ Download Translated PDF", 
                data=output_pdf.tobytes(),
                file_name=f"translated_drawing_{total_pages}pages.pdf",
                mime="application/pdf"
            )
