import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# =========================
# ตั้งค่าหน้าเว็บ
# =========================
st.set_page_config(
    page_title="Phak Top Chawa Detector",
    page_icon="🌿",
    layout="centered"
)

# =========================
# CSS ตกแต่งเว็บ
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #eef8ec 0%, #f8fff6 100%);
}

.main-title {
    background: linear-gradient(90deg, #1b5e20, #388e3c);
    color: white;
    padding: 24px;
    border-radius: 20px;
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 10px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}

.sub-title {
    text-align: center;
    color: #2e7d32;
    font-size: 20px;
    margin-bottom: 30px;
    font-weight: 500;
}

.custom-box {
    background: white;
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 24px;
    border: 1px solid #dcefd8;
}

.stButton > button {
    background: linear-gradient(90deg, #2e7d32, #43a047);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 24px;
    border: none;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1b5e20, #2e7d32);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# โหลดโมเดล
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# ค่าการแปลง pixel -> ตารางเมตร
# =========================
PIXELS_PER_SQUARE_METER = 2500.0


# =========================
# ฟังก์ชันตรวจจับ
# =========================
def detect(frame):
    results = model(frame, conf=0.3, iou=0.4)

    output_text = []

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for i, mask in enumerate(masks):

            # resize mask
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # binary mask
            binary = (mask > 0.5)

            # pixel area
            area_pixels = int(binary.sum())

            # กัน noise
            if area_pixels < 50:
                continue

            # แปลงเป็น m²
            area_m2 = area_pixels / PIXELS_PER_SQUARE_METER
            area_m2 = round(area_m2, 2)

            # centroid
            ys, xs = np.where(binary)

            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

            # เก็บข้อความ
            output_text.append(
                f"กอ#{i+1} {area_m2} m² (x={cx}, y={cy})"
            )

            # contour
            contours, _ = cv2.findContours(
                (binary * 255).astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    1
                )

            # centroid จุดแดง
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 2)

            # label
            cv2.putText(
                frame,
                str(i + 1),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

    return frame, output_text


# =========================
# UI
# =========================
st.markdown("""
<div class="main-title">🌿 Phak Top Chawa Detector</div>
<div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-box">', unsafe_allow_html=True)

st.subheader("📤 อัปโหลดรูปภาพ")

uploaded_file = st.file_uploader(
    "รองรับ JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

analyze = st.button("🔍 วิเคราะห์ภาพ")

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# RUN
# =========================
if uploaded_file is not None and analyze:

    with st.spinner("กำลังวิเคราะห์ภาพ..."):

        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        result_frame, texts = detect(frame)

        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # =========================
        # ผลลัพธ์ข้อความ
        # =========================
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        st.subheader("📋 ผลการตรวจจับ")

        if texts:
            for t in texts:
                st.write(t)
        else:
            st.warning("ไม่พบกอผักตบชวา")

        st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # ภาพผลลัพธ์
        # =========================
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        st.subheader("🖼️ ภาพผลการตรวจจับ")
        st.image(result_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =========================
# Footer
# =========================
st.markdown("""
<div style="text-align:center; color:#2e7d32; margin-top:40px; padding:20px;">
    <b>Phak Top Chawa Detector</b><br>
    ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา
</div>
""", unsafe_allow_html=True)
