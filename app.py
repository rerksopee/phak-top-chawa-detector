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
# ปรับค่าตามการสอบเทียบจริง
# =========================
PIXELS_PER_SQUARE_METER = 2500.0


# =========================
# ฟังก์ชันตรวจจับ
# =========================
def detect(frame):
    # รันโมเดล
    results = model(frame, conf=0.3, iou=0.4)

    # เก็บผลลัพธ์ข้อความ (รูปแบบเดียวกับโค้ดเดิม)
    output_text = []

    # ถ้ามี segmentation mask
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        # วนลูปทีละกอ
        for i, mask in enumerate(masks):

            # 1) ปรับขนาด mask ให้เท่ากับภาพต้นฉบับ
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # 2) แปลงเป็น Binary Image
            binary = (mask > 0.5)

            # 3) คำนวณพื้นที่ (pixel)
            area_pixels = int(binary.sum())

            # 4) แปลงเป็นตารางเมตร
            area_m2 = area_pixels / PIXELS_PER_SQUARE_METER

            # 5) หา centroid
            ys, xs = np.where(binary)

            # ถ้าไม่มี pixel ให้ข้าม
            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

            # 6) เก็บข้อความผลลัพธ์
            # รูปแบบเหมือนโค้ดเดิม:
            # กอ#1 12.34 ตารางเมตร (x=123, y=456)
            output_text.append(
                f"กอ#{i+1} {area_m2:.2f} ตารางเมตร (x={cx}, y={cy})"
            )

            # 7) สร้าง Bounding Box จาก Mask
            contours, _ = cv2.findContours(
                binary.astype('uint8'),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # เลือก contour ที่มีพื้นที่มากที่สุด
                cnt = max(contours, key=cv2.contourArea)

                # หา Bounding Box
                x, y, w, h = cv2.boundingRect(cnt)

                # วาดกรอบสีเขียว
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    1
                )

            # 8) วาดจุด centroid 
            cv2.circle(
                frame,
                (cx, cy),
                1,
                (255, 0, 0),
                1
            )

            # 9) แสดงหมายเลขกอ 
            cv2.putText(
                frame,
                str(i + 1),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
               0.5,
                (0, 0, 255),
                1
            )

    # คืนค่าเหมือนโค้ดเดิม
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
        # รายละเอียดแต่ละกอ
        # แสดงเป็นคนละบรรทัดธรรมดา
        # ไม่มีกรอบแยกแต่ละกอ
        # =========================
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        st.subheader("📋 ผลการตรวจจับ")

        if texts:
            for t in texts:
                st.write(t)   # คนละบรรทัดธรรมดา
        else:
            st.warning("ไม่พบกอผักตบชวาในภาพ")

        st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # แสดงภาพผลลัพธ์
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
