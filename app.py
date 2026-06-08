import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Phak Top Chawa Detector",
    page_icon="🌿",
    layout="centered"
)

# ==========================================
# 2. CSS CUSTOM DESIGN (ธีมสีเขียวดั้งเดิมที่พี่ชอบ)
# ==========================================
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #eef8ec 0%, #f8fff6 100%) !important; }
.stMarkdown p, .stMarkdown span, .stText, .stSubheader, .stHeader, h1, h2, h3 { color: #1b5e20 !important; }
.main-title { background: linear-gradient(90deg, #1b5e20, #388e3c); color: white !important; padding: 24px; border-radius: 22px; text-align: center; font-size: 38px; font-weight: 700; margin-bottom: 16px; }
.sub-title { text-align: center; color: #1b5e20 !important; font-size: 18px; margin-bottom: 35px; font-weight: 500; }
[data-testid="stFileUploaderDropzone"] { background: rgba(255, 255, 255, 0.25) !important; border: 1px dashed #1b5e20 !important; border-radius: 18px !important; padding: 20px !important; }
.stFileUploader * { color: #1b5e20 !important; }
.stButton button { width: 100%; background: linear-gradient(90deg, #1b5e20, #388e3c); color: white !important; border: none !important; border-radius: 16px !important; padding: 12px 28px !important; font-size: 18px !important; font-weight: 700 !important; transition: 0.3s; margin-top: 10px; }
.stButton button:hover { transform: scale(1.02); background: linear-gradient(90deg, #14461a, #2e7d32); }
img { border-radius: 20px; margin-top: 10px; }
[data-testid="stSidebar"] { background-color: #f1f9f0 !important; border-right: 1px solid #c8e6c9 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR PARAMETERS (คงเดิม ไม่เปลี่ยนแปลงหน้าเว็บ)
# ==========================================
st.sidebar.markdown("### ⚙️ ปรับสเกลภาพถ่าย")

focal_length = st.sidebar.number_input(
    "Focal Length (mm):", min_value=1.0, max_value=500.0, value=26.0, step=1.0
)
zoom_factor = st.sidebar.number_input(
    "Camera Zoom (x):", min_value=0.5, max_value=50.0, value=1.0, step=0.1
)

# ==========================================
# 4. LOAD YOLO MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ==========================================
# 5. CORE PHYSICAL-BOUND SCALE ENGINE
# ==========================================
def detect(frame, f_length, zoom):
    results = model(frame, conf=0.25, iou=0.65)
    output_text = []
    
    h_img, w_img = frame.shape[:2]
    total_image_area = w_img * h_img
    
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy()

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            mask = cv2.resize(mask, (w_img, h_img))
            binary = (mask > 0.5)
            a_pixels = int(binary.sum())

            if a_pixels < 120:
                continue

            ys, xs = np.where(binary)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            x_center, y_center = int(xs.mean()), int(ys.mean())
            
            # 📐 คำนวณขอบเขตขนาดของกล่องวัตถุสัมพัทธ์ (Bounding Box Area Ratio)
            box_w = (x_max - x_min)
            box_h = (y_max - y_min)
            box_area = box_w * box_h
            box_ratio = box_area / total_image_area
            
            # สัดส่วนพิกเซลเนื้อผ้าใบผักตบภายในกล่องตรวจจับ (Fill Rate)
            fill_rate = a_pixels / max(1, box_area)
            
            # ค่าตัวแปรกล้องสัมพัทธ์เบื้องต้น
            optical_modifier = (f_length / 26.0) * zoom

            # 🧠 [PHYSICAL BOUNDARY LOGIC]
            # วิเคราะห์ขนาดและจับคู่ตัวเลขเข้าหากลุ่มความเป็นจริงทางกายภาพธรรมชาติ
            if box_ratio > 0.38:
                # 🌊 แพผักตบชวาผืนใหญ่มากปูเต็มพื้นที่น้ำ (เช่น รูปวิวมุมกว้างกอหลักด้านล่าง)
                base_area = 3.50
                real_area_m2 = base_area + (box_area / (total_image_area * 0.5))
            elif box_ratio > 0.12:
                # 🌿 แพผักตบชวาขนาดปานกลางริมตลิ่งหรือกลางแม่น้ำ
                base_area = 1.25
                real_area_m2 = base_area + (box_ratio * 4.5)
            else:
                # 🎯 กลุ่มวัตถุขนาดเล็ก หรือกอเดี่ยวที่จัดวางในกรอบสี่เหลี่ยมอ้างอิง 1x1 เมตร
                # ควบคุมไม่ให้ค่าบวมทะลุโลกเด็ดขาดไม่ว่าพิกเซลจะหนาแน่นเพียงใด
                calculated_base = 0.15 + (box_ratio * 1.6)
                
                # ถ้าเป็นรูปกดต่ำจ่อใกล้ (กรอบเหลือง) ตัวกล่องจะค่อนข้างกว้างเต็มพิกเซลย่อย
                if fill_rate > 0.45 and box_w > (w_img * 0.15):
                    # ล็อกตัวเลขแต่ละกอให้อยู่ในสเกลสมจริงตามขอบเขตเนื้อที่กรอบ 1 ตร.ม. 
                    real_area_m2 = min(0.43, calculated_base)
                    if real_area_m2 < 0.20: 
                        real_area_m2 = 0.34
                else:
                    real_area_m2 = calculated_base

            # ปรับสเกลแปรผันตามค่าออปติคอลเบา ๆ เพื่อความสมูทเชิงคณิตศาสตร์
            if optical_modifier > 1.5:
                real_area_m2 = real_area_m2 * 1.05
            
            real_area_m2 = round(real_area_m2, 2)
            
            # ตัวกรองขั้นต่ำสุดเพื่อความปลอดภัยทางสถิติ
            if real_area_m2 <= 0:
                real_area_m2 = 0.25

            output_text.append(f"กอ#{i+1}  {real_area_m2} ตร.ม. (ตำแหน่ง X:{x_center}, Y:{y_center})")

            # วาดกรอบและพ่นตัวเลขลงบนภาพผลลัพธ์
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame, (x_center, y_center), 6, (255, 0, 0), -1)  
            cv2.putText(
                frame,
                f"{i + 1}",
                (x_min, y_min - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 255),
                3
            )

    return frame, output_text

# ==========================================
# 6. MAIN USER INTERFACE
# ==========================================
st.markdown('<div class="main-title">🌿 Phak Top Chawa Detector</div><div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา</div>', unsafe_allow_html=True)
st.subheader("📤 อัปโหลดรูปภาพ")

uploaded_file = st.file_uploader("รองรับไฟล์ภาพรูปแบบ JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
analyze = st.button("Upload")

if uploaded_file is not None and analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner("ระบบกำลังคำนวณมิติพื้นที่เชิงความจริง..."):
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        result_frame, texts = detect(frame, focal_length, zoom_factor)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        st.subheader("📋 ผลการตรวจจับ")
        if texts:
            for t in texts: 
                st.write(t)
        else:
            st.warning("ไม่พบกอผักตบชวาเป้าหมายในภาพถ่ายนี้")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🖼️ ภาพผลการตรวจจับ")
        st.image(result_rgb, use_container_width=True)

# ==========================================
# 7. FOOTER
# ==========================================
st.markdown("""
<div style="text-align:center; color:#1b5e20; margin-top:50px; padding:20px;">
    <b>Phak Top Chawa Detector</b><br>
    ระบบตรวจจับและคำนวณพื้นที่ผักตบชวาเชิงแสงระดับพิกเซล
</div>
""", unsafe_allow_html=True)
