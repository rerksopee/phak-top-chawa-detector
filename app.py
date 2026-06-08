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
# 2. CSS CUSTOM DESIGN (ธีมสีเขียวดั้งเดิม)
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
    step=1.0
)

zoom_factor = st.sidebar.number_input(
    "Camera Zoom (x):", 
    min_value=0.5, 
    max_value=50.0, 
    value=1.0, 
    step=0.1
)

# =========================
# 4. LOAD YOLO MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# 5. MATHEMATICAL DETECTION ENGINE
# =========================
def detect(frame, f_length, zoom):
    # ปรับ iou เป็น 0.50 เพื่อให้ทำงานร่วมกับภาพสภาพแวดล้อมอื่น ๆ ได้อย่างเสถียร ไม่หลุดขอบเขต
    results = model(frame, conf=0.24, iou=0.50)
    output_text = []
    
    h_img, w_img = frame.shape[:2]

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy()

        valid_masks = []
        valid_boxes = []

        # สแกนกรองเอาเฉพาะวัตถุที่มีความสมบูรณ์เชิงพิกเซล
        for mask, box in zip(masks, boxes):
            resized_mask = cv2.resize(mask, (w_img, h_img))
            binary_mask = (resized_mask > 0.5)
            a_pixels = int(binary_mask.sum())
            
            if a_pixels >= 150:
                valid_masks.append((binary_mask, a_pixels))
                valid_boxes.append(box)

        if not valid_masks:
            return frame, output_text

        # 📐 [DYNAMIC SCALE CURVE] คำนวณหาตัวหารพิกเซลที่ยืดหยุ่นสัมพันธ์ตามระดับการซูมจริง
        # โดยอ้างอิงจากฐานพิกเซลกลางของเลนส์ เพื่อไม่ให้ภาพมุมกว้างหรือภาพแคบเกิดการเอ๋อ
        base_ratio = 420000.0 if zoom <= 1.2 else 210000.0
        optical_modifier = math.pow((f_length / 26.0) * zoom, 1.85)
        pixel_to_m2_ratio = base_ratio * optical_modifier

        for i, ((binary_mask, a_pixels), box) in enumerate(zip(valid_masks, valid_boxes)):
            ys, xs = np.where(binary_mask)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            x_center = int(xs.mean())
            y_center = int(ys.mean())
            
            # คำนวณหาค่าพื้นที่ดิบตามความหนาแน่นพิกเซลที่เลนส์กล้องบันทึกได้
            raw_area = a_pixels / pixel_to_m2_ratio
            
            # 🌊 [PERSPECTIVE SMOOTHING] ใช้ตัวชดเชยมิติภาพตามความสูงแกน Y แบบต่อเนื่อง
            # ป้องกันไม่ให้เกิดการตัดขอบตัวเลขแบบกระโดดข้ามขั้น
            y_ratio = y_center / h_img
            depth_factor = 0.55 + (y_ratio * 0.65)
            real_area_m2 = raw_area * depth_factor

            # ตรวจสอบจุดสัมผัสขอบจอภาพเพื่อคัดกรองสัญญาณรบกวนริมตลิ่ง
            is_touching_edge = (x_min <= 8 or x_max >= w_img - 8 or y_min <= 8 or y_max >= h_img - 8)
            if is_touching_edge and zoom <= 1.2:
                real_area_m2 = min(real_area_m2, 0.35)

            # กำหนดขอบเขตขั้นต่ำและขั้นสูงสุดทางกายภาพของวัตถุใบผักตบชวา
            real_area_m2 = max(0.05, min(real_area_m2, 2.50))
            real_area_m2 = round(real_area_m2, 2)
            
            # รายงานสถิติตัวเลขเข้าสู่ระบบ
            output_text.append(f"กอ#{i+1}  {real_area_m2} ตร.ม. (ตำแหน่ง X:{x_center}, Y:{y_center})")

            # วาดสัญลักษณ์ Bounding Box และจุดศูนย์กลางวัตถุ
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame, (x_center, y_center), 6, (255, 0, 0), -1)  
            
            # แสดงเฉพาะตัวเลขลำดับกอขนาดใหญ่พิเศษ ชัดเจน สะอาดตา ไม่บังเนื้อวัตถุ
            cv2.putText(
                frame,
                f"{i + 1}",
                (x_min, y_min - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
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
    with st.spinner("ระบบกำลังคำนวณพื้นที่ตามสเกลทัศนศาสตร์จริง..."):
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
    <b>Phak Top Chawa Detector (Dynamic Scale Edition)</b><br>
    ระบบตรวจจับคำนวณพื้นที่ผิวผักตบชวารองรับโครงสร้างภาพหลายมิติ
</div>
""", unsafe_allow_html=True)
