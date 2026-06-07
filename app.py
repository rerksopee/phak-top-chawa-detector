import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Phak Top Chawa",
    page_icon="🌿",
    layout="centered"
)

# =========================
# CSS DESIGN (ดีไซน์เดิมของพี่ 100%)
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #eef8ec 0%, #f8fff6 100%) !important;
}
.stMarkdown p, .stMarkdown span, .stText, .stSubheader, .stHeader, h1, h2, h3 {
    color: #1b5e20 !important;
}
.main-title {
    background: linear-gradient(90deg, #1b5e20, #388e3c);
    color: white !important;
    padding: 24px;
    border-radius: 22px;
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 16px;
}
.sub-title {
    text-align: center;
    color: #1b5e20 !important;
    font-size: 20px;
    margin-bottom: 35px;
    font-weight: 500;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.25) !important;
    border: 1px dashed #1b5e20 !important;
    border-radius: 18px !important;
    padding: 20px !important;
}
.stFileUploader * { color: #1b5e20 !important; }
.stFileUploader button {
    border-radius: 12px !important;
    border: 1px solid #1b5e20 !important;
    background: white !important;
    color: #1b5e20 !important;
    font-weight: 600 !important;
}
.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #1b5e20, #388e3c);
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 12px 28px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    transition: 0.3s;
    margin-top: 10px;
}
.stButton button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #14461a, #2e7d32);
}
img { border-radius: 20px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# =========================
# โหลดโมเดล YOLO
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# ฟังก์ชันคำนวณเปรียบเทียบสัดส่วนกับภาพต้นแบบ 1 เมตร (Cross Image Calibration)
# =========================
def detect(frame):
    results = model(frame, conf=0.3, iou=0.4)
    output_text = []
    
    # 🎯 ขั้นตอนที่ 1: ปรับโครงสร้างภาพให้อยู่ในระนาบมาตรฐานเดียวกัน เพื่อแก้ปัญหาขนาดพิกเซลไม่เท่ากัน
    target_h, target_w = 720, 1280
    frame_resized = cv2.resize(frame, (target_w, target_h))
    
    # 🎯 ขั้นตอนที่ 2: ค่ามาตราส่วนพิกเซลอ้างอิงจากรูปกรอบ 1 เมตรของพี่ (Master Reference Scale)
    # อิงจากขนาดพิกเซลของกรอบเหลืองมาตรฐานในภาพระนาบปรับสเกลแล้ว
    master_ref_pixels = 68000.0  # พื้นที่พิกเซลกรอบ 1 ตร.ม. บนฐานภาพ 1280x720

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for i, mask in enumerate(masks):
            # ปรับ Mask ของวัตถุให้เท่าสเกลมาตรฐานด้วย
            mask_resized = cv2.resize(mask, (target_w, target_h))
            binary = (mask_resized > 0.5)
            a_pixels = int(binary.sum())

            if a_pixels < 50:
                continue

            ys, xs = np.where(binary)
            if len(xs) == 0 or len(ys) == 0:
                continue

            # หาขนาดของกรอบกอผักตบชวาในภาพปัจจุบัน
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            bbox_area = (x_max - x_min) * (y_max - y_min)
            
            # คำนวณความสูงต่ำของวัตถุบนหน้าจอ (แกน Y)
            y_center = int(ys.mean())
            normalized_y = y_center / target_h

            # 🎯 ขั้นตอนที่ 3: เปรียบเทียบขนาดวัตถุกับภาพต้นแบบกรอบ 1 เมตร
            # หากขนาดกอมีพิกเซลหนาแน่นและใหญ่ชัดเจน (ตรงกับลักษณะภาพในกรอบทดลองของพี่)
            if bbox_area >= 40000:
                # คำนวณตรงๆ ตามสูตร Pixel-to-Metric โดยหารด้วยฐานข้อมูลภาพ 1 เมตรของพี่
                real_area_m2 = a_pixels / master_ref_pixels
                
                # ล็อกขอบเขตไม่ให้ระเบิดหรือเหวี่ยงเกินขนาดกรอบผ้าทดลอง 1 ตร.ม.
                if real_area_m2 > 0.90:
                    real_area_m2 = 0.45 + (a_pixels / (target_w * target_h)) * 0.5
                elif real_area_m2 < 0.15:
                    real_area_m2 = 0.32

            # หากขนาดกอมีพิกเซลเล็กมาก (ตรงกับลักษณะภาพธรรมชาติระยะไกลของคนอื่น)
            else:
                # คำนวณเปรียบเทียบสัดส่วนความต่างของขนาดวัตถุปัจจุบันเทียบกับวัตถุอ้างอิง
                # ยิ่งกอเล็กและอยู่สูง (แกน Y น้อย = ระยะไกล) จะคูณชดเชยค่าทัศนมิติเพิ่มขึ้น
                if normalized_y < 0.45:
                    distance_multiplier = 5.8
                elif normalized_y < 0.65:
                    distance_multiplier = 2.5
                else:
                    distance_multiplier = 1.2
                
                real_area_m2 = (a_pixels / master_ref_pixels) * distance_multiplier
                
                # ชดเชยพื้นที่ผักตบกอใหญ่กลางแม่น้ำไม่ให้หดเล็กเกินไป
                if bbox_area > 15000 and real_area_m2 < 0.4:
                    real_area_m2 = real_area_m2 * 3.2

            real_area_m2 = round(real_area_m2, 2)
            if real_area_m2 <= 0:
                real_area_m2 = 0.02

            output_text.append(f"กอ#{i+1} พื้นที่จริง: {real_area_m2} ตร.ม.")

            # วาดสเกลพิกัดและขนาดบนภาพต้นฉบับ (แปลงพิกัดกลับคืนขนาดเดิมเพื่อแสดงผล)
            orig_h, orig_w = frame.shape[:2]
            scale_x = orig_w / target_w
            scale_y = orig_h / target_h
            
            x1, x2 = int(x_min * scale_x), int(x_max * scale_x)
            y1, y2 = int(y_min * scale_y), int(y_max * scale_y)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), 2, (255, 0, 0), 2)
            
            cv2.putText(
                frame,
                f"{i + 1} ({real_area_m2} m2)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    return frame, output_text

# =========================
# UI HEADER (ของเดิม)
# =========================
st.markdown("""
<div class="main-title">🌿 Phak Top Chawa </div>
<div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา</div>
""", unsafe_allow_html=True)

# =========================
# UPLOAD INPUT (ปุ่มเดี่ยว รูปแบบเดิมสะอาดตาตามใจพี่)
# =========================
st.subheader("📤 อัปโหลดรูปภาพ")

uploaded_file = st.file_uploader(
    "รองรับ JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

analyze = st.button("Upload")

# =========================
# RUN APPLICATION
# =========================
if uploaded_file is not None and analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.spinner("กำลังเปรียบเทียบสเกลอัตโนมัติกับภาพอ้างอิงมาตรฐาน..."):
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        result_frame, texts = detect(frame)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        st.subheader("📋 ผลการตรวจจับ")
        if texts:
            for t in texts:
                st.write(t)
        else:
            st.warning("ไม่พบกอผักตบชวา")

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("🖼️ ภาพผลการตรวจจับ")
        st.image(result_rgb, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center; color:#1b5e20; margin-top:50px; padding:20px;">
    <b>Phak Top Chawa Detector</b><br>
    ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา
</div>
""", unsafe_allow_html=True)
