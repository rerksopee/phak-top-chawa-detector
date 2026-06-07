import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# =========================
# 1. PAGE CONFIGURATION
# =========================
st.set_page_config(page_title="Phak Top Chawa Detector", page_icon="🌿", layout="centered")

# CSS สไตล์เขียว-ขาว คลีน
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #eef8ec 0%, #f8fff6 100%) !important; }
.stMarkdown p, .stMarkdown span, .stText, .stSubheader, .stHeader, h1, h2, h3 { color: #1b5e20 !important; }
.main-title { background: linear-gradient(90deg, #1b5e20, #388e3c); color: white !important; padding: 24px; border-radius: 22px; text-align: center; font-size: 38px; font-weight: 700; margin-bottom: 16px; }
.sub-title { text-align: center; color: #1b5e20 !important; font-size: 18px; margin-bottom: 35px; font-weight: 500; }
[data-testid="stFileUploaderDropzone"] { background: rgba(255, 255, 255, 0.25) !important; border: 1px dashed #1b5e20 !important; border-radius: 18px !important; padding: 20px !important; }
.stButton button { width: 100%; background: linear-gradient(90deg, #1b5e20, #388e3c); color: white !important; border: none !important; border-radius: 16px !important; padding: 12px 28px !important; font-size: 18px !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. LOAD YOLO MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# 3. CORE ENGINE: อ้างอิงสเกลจากขนาดใบในกรอบควบคุม
# =========================
def detect(frame):
    results = model(frame, conf=0.3, iou=0.4)
    output_text = []
    
    h_img, w_img = frame.shape[:2]
    total_pixels = h_img * w_img

    # 📏 ค่าอ้างอิงมาตรฐาน (ใบผักตบในระยะกล้องปกติกอขนาด ~0.21 ตร.ม. จะใช้พื้นที่ประมาณ 12,500 พิกเซล)
    base_reference_pixels = 12500.0 
    base_m2 = 0.21

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
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            x_center = int(xs.mean())
            y_center = int(ys.mean())
            normalized_y = y_center / h_img
            bbox_w = x_max - x_min

            # 🛑 กรองพุ่มไม้ยักษ์บนตลิ่งออก เพื่อไม่ให้ค่าสถิติพัง
            if normalized_y > 0.85 and bbox_w / w_img > 0.65:
                continue

            # -----------------------------------------------------------------
            # 📐 คำนวณแบบอ้างอิงขนาดใบมาตรฐาน + ชดเชยระยะลึกสายตา (Y-Axis Perspective)
            # -----------------------------------------------------------------
            # อัตราส่วนพิกเซลวัตถุ เทียบกับ พิกเซลมาตรฐานของกอควบคุม
            pixel_ratio = a_pixels / base_reference_pixels
            
            # สมการชดเชยระยะลึก (ยิ่งอยู่ด้านบนของภาพหรือค่าน้อย ค่าตัวคูณจะยิ่งสูงขึ้นเพื่อแก้ปัญหาใบหดเล็ก)
            # ตัวเลขนี้ปรับมาให้สมดุลกับภาพมุมก้ม 43 องศาของพี่โดยเฉพาะ
            depth_compensation = 1.0 / (normalized_y + 0.12)

            # คำนวณพื้นที่จริง (ตารางเมตร)
            real_area_m2 = pixel_ratio * base_m2 * (depth_compensation * 0.65)

            # ล็อกเพดานล่างและบนให้สมเหตุสมผลตามพิกเซลธรรมชาติ
            if normalized_y > 0.75 and a_pixels < 10000:
                real_area_m2 = round(0.05 + (a_pixels / 50000.0), 2)
            else:
                real_area_m2 = max(0.10, round(real_area_m2, 2))

            output_text.append(f"กอ#{i+1} พื้นที่ประเมินจริง: {real_area_m2} ตร.ม.")

            # 🎨 วาดกรอบแสดงผล
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame, (x_center, y_center), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"{i + 1} ({real_area_m2} m2)", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, output_text

# =========================
# 4. USER INTERFACE
# =========================
st.markdown('<div class="main-title">🌿 Phak Top Chawa Detector</div><div class="sub-title">ระบบประเมินพื้นที่ผักตบชวาอ้างอิงสเกลใบมาตรฐาน</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("อัปโหลดรูปภาพแม่น้ำธรรมชาติหรือแปลงทดลอง", type=["jpg", "jpeg", "png"])
analyze = st.button("ประมวลผลและคำนวณพื้นที่")

if uploaded_file is not None and analyze:
    image = Image.open(uploaded_file).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    result_frame, texts = detect(frame)
    result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

    st.subheader("📋 ผลการประเมินขนาดกอผักตบ")
    for t in texts: st.write(t)
    st.image(result_rgb, use_container_width=True)
