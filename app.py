import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# ==========================================================
# ตั้งค่าหน้าเว็บ
# ==========================================================
st.set_page_config(
    page_title="Phak Top Chawa Detector",
    page_icon="🌿",
    layout="wide"
)

# ==========================================================
# CSS ตกแต่งเว็บ (ธีมสีเขียว เรียบง่าย)
# ==========================================================
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

/* คำอธิบาย */
.sub-title {
    text-align: center;
    font-size: 20px;
    color: #2e7d32;
    margin-bottom: 30px;
}

/* กล่องเนื้อหา */
.custom-box {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
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

/* File uploader เป็นเส้นขอบเขียว */
section[data-testid="stFileUploader"] {
    border: 2px dashed #81c784;
    border-radius: 10px;
    padding: 15px;
    background-color: #f9fff9;
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

# ==========================================================
# โหลดโมเดล
# ==========================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # ต้องมีไฟล์ best.pt อยู่ในโฟลเดอร์เดียวกับ app.py

model = load_model()

# ==========================================================
# ค่าการแปลง pixel -> ตารางเมตร
# ปรับค่านี้ตามการสอบเทียบจริงของคุณ
# ==========================================================
PIXELS_PER_SQUARE_METER = 2500.0


# ==========================================================
# ฟังก์ชันตรวจจับ
# ==========================================================
def detect(frame):
    """
    ตรวจจับผักตบชวาและคำนวณพื้นที่แต่ละกอ
    คืนค่า:
    - frame: ภาพที่วาดกรอบและหมายเลข
    - result_data: รายการข้อมูลแต่ละกอ
    """

    results = model(frame, conf=0.3, iou=0.4)
    result_data = []

    # ถ้ามี Segmentation Mask
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for i, mask in enumerate(masks):

            # 1) ปรับขนาด mask ให้ตรงกับภาพจริง
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # 2) แปลงเป็น Binary Image
            binary = (mask > 0.5)

            # 3) คำนวณพื้นที่ (pixels)
            area_pixels = int(binary.sum())

            # 4) แปลงเป็นตารางเมตร
            area_m2 = area_pixels / PIXELS_PER_SQUARE_METER

            # 5) หา centroid
            ys, xs = np.where(binary)
            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

            # เก็บข้อมูลแต่ละกอ
            result_data.append({
                "id": i + 1,
                "pixels": area_pixels,
                "area_m2": area_m2,
                "x": cx,
                "y": cy
            })

            # 6) สร้าง Bounding Box จาก Mask
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

            # 7) วาดจุด centroid
            cv2.circle(
                frame,
                (cx, cy),
                4,
                (0, 0, 255),
                -1
            )

            # 8) แสดงเฉพาะหมายเลขกอ (ไม่มีขนาดบนภาพ)
            cv2.putText(
                frame,
                str(i + 1),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    return frame, result_data


# ==========================================================
# ส่วนหัวเว็บ
# ==========================================================
st.markdown("""
<div class="main-title">🌿 Phak Top Chawa Detector</div>
<div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา</div>
""", unsafe_allow_html=True)

# ==========================================================
# อัปโหลดไฟล์
# ==========================================================
st.markdown('<div class="custom-box">', unsafe_allow_html=True)
st.subheader("📤 อัปโหลดรูปภาพเพื่อตรวจจับกอผักตบชวา")

uploaded_file = st.file_uploader(
    "รองรับไฟล์ JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

analyze = st.button("🔍 วิเคราะห์ภาพ")
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# วิเคราะห์ภาพ
# ==========================================================
if uploaded_file is not None and analyze:
    with st.spinner("กำลังวิเคราะห์ภาพ..."):

        # อ่านภาพ
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        # RGB -> BGR
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ตรวจจับ
        result_frame, result_data = detect(frame)

        # BGR -> RGB
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # ==================================================
        # รายละเอียดแต่ละกอ (ไม่มีพื้นที่รวม)
        # ==================================================
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        st.subheader("📊 ผลการตรวจจับ")

        if result_data:
            for item in result_data:
                st.markdown(
                    f"""
                    <div style="
                        background:#f8fff8;
                        padding:15px 20px;
                        margin-bottom:12px;
                        border-left:5px solid #2e7d32;
                        border-radius:8px;
                    ">
                        <b>กอที่ {item['id']}</b><br>
                        พื้นที่: <b>{item['area_m2']:.2f}</b> ตารางเมตร<br>
                        จำนวนพิกเซล: <b>{item['pixels']:,}</b> pixels<br>
                        ตำแหน่ง centroid: ({item['x']}, {item['y']})
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("ไม่พบกอผักตบชวาในภาพ")

        st.markdown('</div>', unsafe_allow_html=True)

        # ==================================================
        # แสดงภาพผลลัพธ์
        # (บนภาพมีเฉพาะกรอบ + หมายเลขกอ ไม่มีข้อความขนาด)
        # ==================================================
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        st.subheader("🖼️ ภาพผลการตรวจจับ")
        st.image(result_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# Footer
# ==========================================================
st.markdown("""
<div class="footer">
    <b>Phak Top Chawa Detector</b><br>
    ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา
</div>
""", unsafe_allow_html=True)
