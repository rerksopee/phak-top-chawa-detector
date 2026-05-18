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
    background-color: #f4f8f2;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1b5e20;
    margin-top: 20px;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #2e7d32;
    margin-bottom: 30px;
}

/* กล่องหลัก */
.custom-box {
    background-color: #ffffff;
    border: 1px solid #a5d6a7;
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 25px;
}

/* ปุ่ม */
.stButton > button {
    background-color: #2e7d32;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    width: 100%;
}

.stButton > button:hover {
    background-color: #1b5e20;
    color: white;
}

/* ลบกรอบเส้นประของ File Uploader */
section[data-testid="stFileUploader"] {
    border: none !important;
    background-color: transparent !important;
    padding: 0 !important;
}

section[data-testid="stFileUploader"] > div {
    border: none !important;
    background-color: transparent !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #2e7d32;
    margin-top: 40px;
    padding: 20px;
    font-size: 14px;
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
# ปรับตามการสอบเทียบจริง
# =========================
PIXELS_PER_SQUARE_METER = 2500.0


# =========================
# ฟังก์ชันตรวจจับ
# =========================
def detect(frame):
    results = model(frame, conf=0.3, iou=0.4)

    output_text = []

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for i, mask in enumerate(masks):

            # ปรับขนาด mask
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # Binary Image
            binary = (mask > 0.5)

            # พื้นที่ (pixel)
            area_pixels = int(binary.sum())

            # แปลงเป็นตารางเมตร
            area_m2 = area_pixels / PIXELS_PER_SQUARE_METER

            # หา centroid
            ys, xs = np.where(binary)
            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

            # เก็บผลลัพธ์รายกอ
            output_text.append(
                f"กอที่ {i+1}: {area_m2:.2f} ตารางเมตร ({area_pixels:,} pixels)"
            )

            # หา contour
            contours, _ = cv2.findContours(
                binary.astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)

                # วาดกรอบสีเขียว
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # แสดงพื้นที่บนภาพ
                cv2.putText(
                    frame,
                    f"{area_m2:.2f} m²",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 128, 0),
                    2
                )

            # จุด centroid
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # หมายเลขกอ
            cv2.putText(
                frame,
                str(i + 1),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    return frame, output_text


# =========================
# ส่วนหัวเว็บ
# =========================
st.markdown("""
<div class="main-title">🌿 Phak Top Chawa Detector</div>
<div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา</div>
""", unsafe_allow_html=True)

# =========================
# อัปโหลดไฟล์
# =========================
st.markdown('<div class="custom-box">', unsafe_allow_html=True)

st.subheader("📤 อัปโหลดรูปภาพเพื่อตรวจจับกอผักตบชวา")

uploaded_file = st.file_uploader(
    "รองรับไฟล์ JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

analyze = st.button("🔍 วิเคราะห์ภาพ")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# วิเคราะห์ภาพ
# =========================
if uploaded_file is not None and analyze:

    with st.spinner("กำลังวิเคราะห์ภาพ..."):

        # อ่านภาพ
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        # RGB -> BGR
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ตรวจจับ
        result_frame, texts = detect(frame)

        # BGR -> RGB
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # =========================
        # ผลการตรวจจับ (แสดงรายกอเท่านั้น)
        # =========================
        if texts:
            st.markdown('<div class="custom-box">', unsafe_allow_html=True)

            st.subheader("📊 ผลการตรวจจับ")

            for t in texts:
                st.write("•", t)

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
<div class="footer">
    <b>Phak Top Chawa Detector</b><br>
    ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา
</div>
""", unsafe_allow_html=True)
