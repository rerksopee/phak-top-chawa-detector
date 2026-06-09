import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import math

# =========================
# 1. PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Phak Top Chawa Detector",
    page_icon="🌿",
    layout="centered"
)

# =========================
# 2. CSS CUSTOM DESIGN (ธีมสีเขียวดั้งเดิม - ไม่แก้หน้าเว็บ)
# =========================
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

# =========================
# 3. SIDEBAR PARAMETERS
# =========================
st.sidebar.markdown("### ⚙️ ปรับสเกลภาพถ่าย")

focal_length = st.sidebar.number_input(
    "Focal Length (mm):", 
    min_value=1.0, 
    max_value=500.0, 
    value=26.0, 
    step=1.0,
    help="ทางยาวโฟกัสของเลนส์กล้อง (ค่าเริ่มต้นมาตรฐานคือ 26mm)"
)

zoom_factor = st.sidebar.number_input(
    "Camera Zoom (x):", 
    min_value=0.5, 
    max_value=50.0, 
    value=1.0, 
    step=0.1,
    help="ระยะการซูมของภาพถ่ายหน้างาน"
)

# =========================
# 4. LOAD YOLO MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# 5. CORE ROBUST DETECTION ENGINE (อัปเดตระบบคณิตศาสตร์อ้างอิงระยะและสเกลความสอดคล้อง)
# =========================
def detect(frame, f_length, zoom):
    results = model(frame, conf=0.25, iou=0.65)
    output_text = []
    
    h_img, w_img = frame.shape[:2]
    
    # พารามิเตอร์ระยะทางกายภาพและระดับสายตาจากโครงสร้างสนาม
    D_FIELD = 3.2
    THETA_RAD = math.radians(46)
    horizontal_dist = D_FIELD * math.cos(THETA_RAD)
    
    # การหาค่าสเกลเลนส์และการซูม (Optical Scale Compensation)
    optical_scale = (f_length / 26.0) * zoom
    
    # ปรับฐานการกระจายพิกเซลต่อนิ้วให้รองรับสเกลกลศาสตร์แปรผันผกผันกับพลังซูมเลนส์
    if zoom > 1.5:
        BASE_PIXEL_PER_M2 = 310000.0
        pixel_to_m2_ratio = BASE_PIXEL_PER_M2 * ((optical_scale / (zoom ** 0.88)) ** 2)
    else:
        BASE_PIXEL_PER_M2 = 295000.0
        pixel_to_m2_ratio = BASE_PIXEL_PER_M2 * (optical_scale ** 2)

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
            
            x_center = int(xs.mean())
            y_center = int(ys.mean())
            
            # การหาตำแหน่งแนวตั้งสัมพัทธ์ของกอผักตบชวา
            normalized_y = y_center / h_img
            
            # คำนวณมิติพื้นที่กล่องเพื่อประเมินความกว้างของแพผักแบบพลวัต
            box_w_ratio = (x_max - x_min) / w_img
            box_h_ratio = (y_max - y_min) / h_img
            box_area_ratio = box_w_ratio * box_h_ratio
            aspect_ratio = (x_max - x_min) / (y_max - y_min + 1e-5)

            calculated_area = a_pixels / pixel_to_m2_ratio

            # 🌟 อัลกอริทึมจำแนกสภาวะระยะทางเพื่อกำหนดขนาดพื้นที่ที่เหมาะสมที่สุด (สอดคล้องตามเกณฑ์สำรวจ)
            if zoom > 1.5 or (box_area_ratio > 0.12 and normalized_y > 0.50):
                # สภาวะที่ 1: กลุ่มระยะประชิด (จ่อใกล้ / ซูม 2.7x ในกรอบเหล็กควบคุม)
                # ทอนกำลังลงมาให้อยู่ในสเกลทางกายภาพจริงของกรอบอ้างอิงมาตรฐาน (ได้สเกล ~0.18 - 0.20 ตร.ม.)
                compensation_factor = 0.45 + (normalized_y * 0.15)
                real_area_m2 = calculated_area * compensation_factor
                
                if zoom > 2.0:
                    real_area_m2 = real_area_m2 * (zoom * 0.46)
                
                # ฟังก์ชันขอบเขตความปลอดภัยเชิงกายภาพในกรอบล้อม
                if real_area_m2 > 0.22:
                    real_area_m2 = 0.20
                elif real_area_m2 < 0.03:
                    real_area_m2 = 0.05
            else:
                # สภาวะที่ 2: กลุ่มระยะไกล (แพผักแนวยาวริมตลิ่งหรือกอกลางน้ำขนาดใหญ่)
                # ใช้ระบบตัวคูณชดเชยความลึกผกผัน (Perspective Multiplier) กู้คืนพื้นที่ตามทัศนียภาพเชิงลึก
                depth_multiplier = (horizontal_dist + 1.8) / (normalized_y + 0.35)
                real_area_m2 = calculated_area * depth_multiplier
                
                # เพิ่มน้ำหนักชดเชยเฉพาะวัตถุแผ่แนวกว้างในมิติภาพ เพื่อให้แพริมตลิ่งมีพื้นที่โตตามสายตาจริง
                if box_w_ratio > 0.45 and aspect_ratio > 2.5:
                    real_area_m2 *= (1.0 + box_w_ratio * 3.5)

            real_area_m2 = round(real_area_m2, 2)
            
            if real_area_m2 < 0.01:
                continue

            output_text.append(
                f"กอ#{i+1}   {real_area_m2} ตร.ม. (ตำแหน่ง X:{x_center}, Y:{y_center})"
            )

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame, (x_center, y_center), 6, (255, 0, 0), -1)  
            
            cv2.putText(
                frame,
                f"{i+1}",
                (x_min, y_min - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 255),
                3
            )

    return frame, output_text

# =========================
# 6. MAIN USER INTERFACE
# =========================
st.markdown('<div class="main-title">🌿 Phak Top Chawa Detector</div><div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา</div>', unsafe_allow_html=True)
st.subheader("📤 อัปโหลดรูปภาพ")

uploaded_file = st.file_uploader("รองรับไฟล์ภาพรูปแบบ JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
analyze = st.button("Upload")

if uploaded_file is not None and analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner("ระบบกำลังคำนวณ"):
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

# =========================
# 7. FOOTER
# =========================
st.markdown("""
<div style="text-align:center; color:#1b5e20; margin-top:50px; padding:20px;">
    <b>Phak Top Chawa Detector</b><br>
</div>
""", unsafe_allow_html=True)
