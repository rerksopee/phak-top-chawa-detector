import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# =========================
# ตั้งค่าหน้าเว็บ
# =========================
st.set_page_config(
    page_title="Phak Top Chawa",
    page_icon="🌿",
    layout="centered"
)

# =========================
# CSS (ล้างกล่องขาวปริศนาออกให้เกลี้ยง + คุมโทนเขียว)
# =========================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(180deg, #eef8ec 0%, #f8fff6 100%) !important;
}

/* TEXT FIX */
.stMarkdown p,
.stMarkdown span,
.stText,
.stSubheader,
.stHeader {
    color: #1b5e20 !important;
}

/* label ของ uploader */
label {
    color: #1b5e20 !important;
}

/* file uploader */
.stFileUploader * {
    color: #1b5e20 !important;
}

/* TITLE */
.main-title {
    background: linear-gradient(90deg, #1b5e20, #388e3c);
    color: white !important;
    padding: 24px;
    border-radius: 20px;
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 10px;
}

/* subtitle */
.sub-title {
    text-align: center;
    color: #1b5e20 !important;
    font-size: 20px;
    margin-bottom: 20px;
    font-weight: 500;
}

/* 🔥 ไม้ตายสุดท้าย: สั่งทำลายและซ่อนกล่องขาวๆ เปล่าๆ ทั้งหมดที่โผล่มากวนใจทิ้งไปให้หมด */
[data-testid="stVerticalBlock"] > div:empty,
.stElementContainer:empty,
div:has(> hr) {
    display: none !important;
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}

/* จัดสไตล์เส้นตรงบางเฉียบสีเขียวเข้มไม่ให้หลุดเฟรม */
.real-green-line {
    display: block !important;
    width: 100% !important;
    height: 1px !important;
    background-color: #1b5e20 !important;
    margin: 20px 0 !important;
    opacity: 1.0 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# ฟังก์ชันบังคับวาดเส้นตรงสีเขียวเข้ม (ยัดใส่ st.caption เอาชนะระบบดักกรอง)
# =========================
def draw_line():
    st.caption('<div class="real-green-line"></div>', unsafe_allow_html=True)
# =========================
# โหลดโมเดล
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# ค่าคาลิเบรต pixel -> m²
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

            # binary
            binary = (mask > 0.5)

            # pixel area
            area_pixels = int(binary.sum())

            # กัน noise
            if area_pixels < 50:
                continue

            # 🔥 แปลงเป็น m²
            area_m2 = round(area_pixels / PIXELS_PER_SQUARE_METER, 2)

            # centroid
            ys, xs = np.where(binary)

            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

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
                    2
                )

            # centroid
            cv2.circle(frame, (cx, cy), 2, (255, 0, 0), 2)

            # label
            cv2.putText(
                frame,
                str(i + 1),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    return frame, output_text

# =========================
# UI HEADER
# =========================
st.markdown("""
<div class="main-title">🌿 Phak Top Chawa </div>
<div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา</div>
""", unsafe_allow_html=True)

# =========================
# UPLOAD
# =========================
st.markdown('<div class="custom-box">', unsafe_allow_html=True)

st.subheader("📤 อัปโหลดรูปภาพ")

uploaded_file = st.file_uploader(
    "รองรับ JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

analyze = st.button("Upload")

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
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center; color:#1b5e20; margin-top:40px; padding:20px;">
    <b>Phak Top Chawa Detector</b><br>
    ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา
</div>
""", unsafe_allow_html=True)
