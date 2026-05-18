import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# =========================
# ตั้งค่าหน้าเว็บ
# =========================
# =========================
# CSS ตกแต่งเว็บ (สไตล์เรียบง่าย โทนสีเขียว)
# ให้นำส่วนนี้ไปแทน st.markdown("""<style>...</style>""", unsafe_allow_html=True)
# =========================
st.markdown("""
<style>
/* พื้นหลังเว็บ */
.stApp {
    background-color: #f4f8f2;
}

/* หัวข้อหลัก */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1b5e20;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* คำอธิบายใต้หัวข้อ */
.sub-title {
    text-align: center;
    font-size: 20px;
    color: #2e7d32;
    margin-bottom: 30px;
}

/* กล่องหลัก */
.custom-box {
    background-color: #ffffff;
    border: 2px solid #c8e6c9;
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 25px;
}

/* กล่องผลลัพธ์ */
.result-box {
    background-color: #f8fff8;
    border: 2px solid #dcedc8;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}

/* กล่องตัวเลข */
.metric-card {
    background-color: #e8f5e9;
    border: 1px solid #c8e6c9;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
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

/* ปุ่มเมื่อเอาเมาส์ชี้ */
.stButton > button:hover {
    background-color: #1b5e20;
    color: white;
}

/* File uploader */
section[data-testid="stFileUploader"] {
    border: 2px dashed #81c784;
    border-radius: 10px;
    padding: 15px;
    background-color: #f9fff9;
}

/* เส้นแบ่ง */
hr {
    border: none;
    border-top: 2px solid #c8e6c9;
    margin-top: 20px;
    margin-bottom: 20px;
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
# ปรับค่าตามการสอบเทียบจริง
# =========================
PIXELS_PER_SQUARE_METER = 2500.0


# =========================
# ฟังก์ชันตรวจจับ
# =========================
def detect(frame):
    results = model(frame, conf=0.3, iou=0.4)

    output_text = []
    total_area_pixels = 0

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for i, mask in enumerate(masks):

            # ปรับขนาด mask ให้ตรงกับภาพ
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # แปลงเป็น Binary Image
            binary = (mask > 0.5)

            # คำนวณพื้นที่จริง
            area = int(binary.sum())
            total_area_pixels += area

            # หา centroid
            ys, xs = np.where(binary)
            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

            output_text.append(
                f"กอ#{i+1}: {area:,} pixel (x={cx}, y={cy})"
            )

            # สร้าง Bounding Box จาก Mask
            contours, _ = cv2.findContours(
                binary.astype('uint8'),
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

            # วาด centroid สีแดง
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # แสดงหมายเลขกอ
            cv2.putText(
                frame,
                str(i + 1),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # แปลงเป็นตารางเมตร
    area_m2 = total_area_pixels / PIXELS_PER_SQUARE_METER

    return frame, output_text, total_area_pixels, area_m2


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
        result_frame, texts, total_pixels, area_m2 = detect(frame)

        # BGR -> RGB
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # ===== ผลการตรวจจับ =====
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        st.subheader("📊 ผลการตรวจจับ")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>พื้นที่รวมของผักตบชวา</h4>
                <h1>{area_m2:.2f}</h1>
                <p>ตารางเมตร</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>จำนวนพิกเซลทั้งหมด</h4>
                <h1>{total_pixels:,}</h1>
                <p>pixels</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ===== รายละเอียดแต่ละกอ =====
        if texts:
            st.markdown('<div class="custom-box">', unsafe_allow_html=True)
            st.subheader("📋 รายละเอียดแต่ละกอ")
            for t in texts:
                st.write("•", t)
            st.markdown('</div>', unsafe_allow_html=True)

        # ===== แสดงภาพผลลัพธ์ =====
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
