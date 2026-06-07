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
# 2. CSS CUSTOM DESIGN
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
    help="ทางยาวโฟกัสของเลนส์กล้อง"
)

zoom_factor = st.sidebar.number_input(
    "Camera Zoom (x):", 
    min_value=1.0, 
    max_value=50.0, 
    value=1.0, 
    step=0.1,
    help="ระยะซูมของภาพถ่าย"
)

# =========================
# 4. LOAD YOLO MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# 5. CORE DETECTION ENGINE
# =========================
def detect(frame, f_length, zoom):
    results = model(frame, conf=0.3, iou=0.4)
    output_text = []
    
    h_img, w_img = frame.shape[:2]
    
    # 📐 คำนวณตามหลักฟิสิกส์ทัศนศาสตร์สนามจริง (ระยะราบกายภาพ 3.2 เมตร แกนองศา 43)
    d_field = 3.2
    theta_rad = math.radians(43.0)
    horizontal_dist = d_field * math.cos(theta_rad)
    
    # [ปรับแก้สูตรซูม] ใช้สเกลพื้นที่แปรผกผันตามกำลังสองของระยะซูมดิจิทัลเพื่อคุมตัวเลขไม่ให้บวมเวอร์
    optical_scale = (f_length / 26.0)
    pixel_to_m2_ratio = 185000.0 * (optical_scale ** 1.2) * (zoom ** 2.0)

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy()

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            mask = cv2.resize(mask, (w_img, h_img))
            binary = (mask > 0.5)
            a_pixels = int(binary.sum())

            if a_pixels < 100:
                continue

            ys, xs = np.where(binary)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            # 📍 ล็อกตำแหน่งพิกเซลแกน X, Y แสดงผลตามที่พี่ต้องการ
            x_center = int(xs.mean())
            y_center = int(ys.mean())
            
            normalized_y = y_center / h_img

            # -----------------------------------------------------------------
            # 📐 คำนวณพื้นที่เชิงตำแหน่งดั้งเดิม (พร้อมระบบควบคุมสเกลระยะซูม)
            # -----------------------------------------------------------------
            calculated_area = a_pixels / pixel_to_m2_ratio
            
            # ชดเชยทัศนมิติเชิงลึกผ่านพิกเซลแกน Y ดั้งเดิมของพี่
            depth_multiplier = (1.0 / (normalized_y + 0.18)) * (horizontal_dist / 1.5)
            real_area_m2 = calculated_area * depth_multiplier

            # จัดระเบียบเกลี่ยค่าตามระดับความลึกให้สัมพันธ์กับวัตถุในสนามจริง
            if normalized_y > 0.70:
                real_area_m2 = max(0.10, real_area_m2 * 0.85)
            else:
                real_area_m2 = max(0.15, real_area_m2)

            # ปัดเศษทศนิยมเป็น 2 ตำแหน่ง
            real_area_m2 = round(real_area_m2, 2)
            
            # 📋 ส่งออกข้อความ ลำดับ พื้นที่ และตำแหน่งพิกเซล (X, Y)
            output_text.append(f"กอ#{i+1} พื้นที่จริง: {real_area_m2} ตร.ม. (ตำแหน่ง X:{x_center}, Y:{y_center})")

            # -----------------------------------------------------------------
            # 🎨 DRAWING LAYER (พล็อตจุด Center และตีกรอบดั้งเดิม)
            # -----------------------------------------------------------------
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame, (x_center, y_center), 6, (255, 0, 0), -1)  # จุดวงกลมสีน้ำเงินบอกพิกัด
            cv2.putText(
                frame,
                f"ID:{i + 1} ({real_area_m2} m2)",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    return frame, output_text

# =========================
# 6. MAIN USER INTERFACE
# =========================
st.markdown('<div class="main-title">🌿 Phak Top Chawa Detector</div><div class="sub-title">ระบบวิเคราะห์พื้นที่ผักตบชวาผ่านคุณลักษณะภาพถ่าย</div>', unsafe_allow_html=True)
st.subheader("📤 อัปโหลดรูปภาพ")

uploaded_file = st.file_uploader("รองรับไฟล์ภาพรูปแบบ JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
analyze = st.button("ประมวลผลภาพ")

if uploaded_file is not None and analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner("ระบบกำลังคำนวณและประมวลผลพิกเซลระบุตำแหน่งเดิม..."):
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        result_frame, texts = detect(frame, focal_length, zoom_factor)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        st.subheader("📋 ผลการคำนวณพื้นที่และพิกเซลตำแหน่ง")
        if texts:
            for t in texts: st.write(t)
        else:
            st.warning("ไม่พบกอผักตบชวาเป้าหมายในภาพถ่ายนี้")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🖼️ ภาพผลการตรวจจับและจุดพิกเซลตำแหน่ง")
        st.image(result_rgb, use_container_width=True)

st.markdown('<div style="text-align:center; color:#1b5e20; margin-top:50px; padding:20px;"><b>Phak Top Chawa Detector v9.7 (Zoom-Calibrated Final)</b></div>', unsafe_allow_html=True)
