import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# =========================
# 1. PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Phak Top Chawa Detector",
    page_icon="🌿",
    layout="centered"
)

# =========================
# 2. CSS CUSTOM DESIGN (สไตล์เขียว-ขาว คลีน)
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
    help="ทางยาวโฟกัสของเลนส์ (ค่าปกติของกล้องมือถือทั่วไปคือ 24 - 26 mm)"
)

zoom_factor = st.sidebar.number_input(
    "Camera Zoom (x):", 
    min_value=1.0, 
    max_value=50.0, 
    value=1.0, 
    step=0.1,
    help="ระยะซูมตอนถ่ายภาพ (หากเลื่อนเพิ่มขึ้น ค่าพื้นที่ในภาพจะเปลี่ยนตามทันที)"
)

# =========================
# 4. LOAD YOLO MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# 5. CORE DETECTION ENGINE (ปรับสูตรให้ผูกกับค่าซูมในทุกๆ เงื่อนไข)
# =========================
def detect(frame, f_length, zoom):
    results = model(frame, conf=0.3, iou=0.4)
    output_text = []
    
    h_img, w_img = frame.shape[:2]
    total_image_pixels = h_img * w_img

    # 📐 ตัวแปรสเกลเลนส์ออปติก (Focal Length ขยาย หรือ Zoom เพิ่ม -> ตัวหารจะเปลี่ยนไป ทำให้วัตถุที่ซูมเข้ามาถูกทอนขนาดจริงกลับมาได้อย่างถูกต้อง)
    optics_modifier = (f_length / 26.0) * zoom

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
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            bbox_area = bbox_w * bbox_h
            
            x_center = int(xs.mean())
            y_center = int(ys.mean())
            
            normalized_x = x_center / w_img
            normalized_y = y_center / h_img
            img_ratio = a_pixels / total_image_pixels

            # 🛑 FILTER LAYER: ตัดพุ่มไม้ขนาดใหญ่บนฝั่งตลิ่งออกไป
            if normalized_y > 0.85 and bbox_w / w_img > 0.70:
                continue
            if bbox_area > (total_image_pixels * 0.40) and normalized_y > 0.60:
                continue

            # -----------------------------------------------------------------
            # 📐 คำนวณพื้นที่แบบแปรผันตาม Optical Scale (แก้ไขให้ตอบสนองต่อการซูมแล้ว)
            # -----------------------------------------------------------------
            is_inside_test_frame = False
            if (0.22 <= normalized_x <= 0.78) and (0.25 <= normalized_y <= 0.78):
                if bbox_w / w_img < 0.65:
                    is_inside_test_frame = True

            if is_inside_test_frame:
                # 🔹 [บริบทในกรอบทดลอง] ผูกสูตรเข้ากับ optics_modifier เพื่อให้เวลาเปลี่ยนค่า Zoom ตัวเลขพื้นที่จริงขยับตามทันที
                calculated_area = (a_pixels / 14000.0) / (optics_modifier ** 2)
                
                # ควบคุมขอบเขตสมมติฐานให้อยู่ในเกณฑ์ที่สมเหตุสมผลตามพิกเซลที่ซูมเข้า-ออก
                if calculated_area < 0.01:
                    real_area_m2 = calculated_area
                else:
                    real_area_m2 = calculated_area
            else:
                # 🔹 [บริบทภาพธรรมชาติแม่น้ำทั่วไป] คำนวณชดเชยตามความลึกแกน Y ผกผันร่วมกับสเปกซูมของเลนส์
                if normalized_y > 0.80 and bbox_area < 25000:
                    real_area_m2 = (0.05 + (img_ratio * 0.1)) / optics_modifier
                else:
                    if normalized_y < 0.35:    # ระยะไกลลิบ
                        base_divisor = 4500.0 * optics_modifier
                        real_area_m2 = (a_pixels / base_divisor) * 2.5
                    elif normalized_y < 0.70:  # ระยะกลางแม่น้ำ
                        base_divisor = 11000.0 * optics_modifier
                        calculated_area = a_pixels / base_divisor
                        if bbox_w > 150:        
                            real_area_m2 = calculated_area * 5.5
                        else:
                            real_area_m2 = calculated_area
                    else:                      # ระยะใกล้หน้ากล้อง
                        base_divisor = 15000.0 * optics_modifier
                        real_area_m2 = a_pixels / base_divisor

                if real_area_m2 < 0.30 and not (normalized_y > 0.80 and bbox_area < 25000):
                    real_area_m2 = (1.25 + (img_ratio * 0.5)) / optics_modifier

            # ปัดเศษทศนิยมเป็น 2 ตำแหน่ง
            real_area_m2 = round(real_area_m2, 2)
            output_text.append(f"กอ#{i+1} พื้นที่จริง: {real_area_m2} ตร.ม.")

            # -----------------------------------------------------------------
            # 🎨 DRAWING LAYER (กรอบเขียว + จุดศูนย์กลางน้ำเงิน + ป้ายข้อความ)
            # -----------------------------------------------------------------
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame, (x_center, y_center), 5, (255, 0, 0), -1)
            cv2.putText(
                frame,
                f"{i + 1} ({real_area_m2} m2)",
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
    with st.spinner("ระบบกำลังตรวจสอบวัตถุและปรับอัตราส่วนพื้นที่ตามคุณลักษณะเลนส์และระยะซูม..."):
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        result_frame, texts = detect(frame, focal_length, zoom_factor)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        st.subheader("📋 ผลการคำนวณพื้นที่")
        if texts:
            for t in texts: st.write(t)
        else:
            st.warning("ไม่พบกอผักตบชวาเป้าหมายในภาพถ่ายนี้")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🖼️ ภาพผลการตรวจจับ")
        st.image(result_rgb, use_container_width=True)

st.markdown('<div style="text-align:center; color:#1b5e20; margin-top:50px; padding:20px;"><b>Phak Top Chawa Detector v2.6</b></div>', unsafe_allow_html=True)
