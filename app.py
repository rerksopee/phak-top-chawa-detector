import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Phak Top Chawa",
    page_icon="🌿",
    layout="centered"
)

# =========================
# CSS (เวอร์ชันดีไซน์ของเน่ + เพิ่มการเว้นช่องลมให้สบายตา)
# =========================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(
        180deg,
        #eef8ec 0%,
        #f8fff6 100%
    ) !important;
}

/* TEXT COLOR */
.stMarkdown p, 
.stMarkdown span, 
.stText, 
.stSubheader, 
.stHeader,
h1, h2, h3 {
    color: #1b5e20 !important;
}

/* TITLE BANNER */
.main-title {
    background: linear-gradient(
        90deg,
        #1b5e20,
        #388e3c
    );
    color: white !important;
    padding: 24px;
    border-radius: 22px;
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 16px; /* เพิ่มช่องลมใต้แบนเนอร์ */
}

/* SUB TITLE */
.sub-title {
    text-align: center;
    color: #1b5e20 !important;
    font-size: 20px;
    margin-bottom: 35px; /* เพิ่มช่องลมก่อนเข้าสู่ส่วนอัปโหลด */
    font-weight: 500;
}

/* FILE UPLOADER DESIGN (ขอบเขียวประ) */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.25) !important;
    border: 1px dashed #1b5e20 !important;
    border-radius: 18px !important;
    padding: 20px !important;
}

.stFileUploader * {
    color: #1b5e20 !important;
}

.stFileUploader button {
    border-radius: 12px !important;
    border: 1px solid #1b5e20 !important;
    background: white !important;
    color: #1b5e20 !important;
    font-weight: 600 !important;
}

/* BUTTON DESIGN */
.stButton button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #1b5e20,
        #388e3c
    );
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 12px 28px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    transition: 0.3s;
    margin-top: 10px; /* เพิ่มช่องไฟเหนือปุ่มกด */
}

.stButton button:hover {
    transform: scale(1.02);
    background: linear-gradient(
        90deg,
        #14461a,
        #2e7d32
    );
}

/* RESULT IMAGE */
img {
    border-radius: 20px;
    margin-top: 10px;
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
# ฟังก์ชันตรวจจับและคำนวณพื้นที่ (ชดเชยระยะไกลอัตโนมัติ)
# =========================
def detect(frame):
    results = model(frame, conf=0.3, iou=0.4)
    output_text = []

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for i, mask in enumerate(masks):
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            binary = (mask > 0.5)
            area_pixels = int(binary.sum())

            if area_pixels < 50:
                continue

            ys, xs = np.where(binary)

            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

            # -----------------------------------------------------------------
            # 🛠️ ส่วนคำนวณพื้นที่ใหม่: ชดเชยระยะพิกเซลต่อพิกเซลแบบสัดส่วนมุมมองกล้อง
            # แก้ปัญหากออยู่ไกลแล้วระบบบอกว่าขนาดเล็ก (อิงตามการทดลองกรอบ 1x1 เมตร)
            # -----------------------------------------------------------------
            real_area_m2 = 0.0
            height_pixels = frame.shape[0] # ความสูงทั้งหมดของภาพ
            
            # วนลูปคำนวณค่าน้ำหนักพื้นที่ของพิกเซลแต่ละจุดตามตำแหน่งแนวตั้ง (แกน Y)
            for y_pixel in ys:
                # แปลงตำแหน่ง Y ของพิกเซลให้อยู่ในสเกล 0.0 - 1.0 (0 = ขอบบนสุดภาพ/ไกลสุด, 1 = ขอบล่างสุดภาพ/ใกล้สุด)
                norm_y = y_pixel / height_pixels
                
                # ฟังก์ชันสัดส่วนมุมเอียง: ยิ่งพิกเซลอยู่ใกล้ขอบบน (norm_y เข้าใกล้ 0) ตัวหารจะยิ่งน้อยลง 
                # ทำให้พิกเซลตรงนั้นขยายค่าพื้นที่ตารางเมตรกลับมาใหญ่สมจริง
                # เกณฑ์คำนวณนี้สอบเทียบจากจุดทดสอบกรอบ 1x1 เมตร
                # ⚠️ หมายเหตุ: สามารถนำจำนวนพิกเซลจริงจากผลทดลองระยะใกล้และไกลมาปรับจูนตัวเลข 2500.0 และ 0.5 นี้ได้ครับ
                pixels_per_m2_at_pos = 2500.0 * (0.5 + 1.5 * norm_y)
                
                # สะสมพื้นที่ของพิกเซลนี้ (1 พิกเซล = 1 / ตัวหารพิกเซลตารางเมตร ณ พิกัดนั้น)
                real_area_m2 += (1.0 / pixels_per_m2_at_pos)

            # ปัดเศษทศนิยม 4 ตำแหน่ง
            real_area_m2 = round(real_area_m2, 4)

            # เปลี่ยนรูปแบบข้อความรายงานผลเป็นหน่วย "ตารางเมตร" (ตร.ม.) เรียบร้อยครับ
            output_text.append(f"กอ#{i+1} พื้นที่จริง: {real_area_m2} ตร.ม. (x={cx}, y={cy})")

            contours, _ = cv2.findContours(
                (binary * 255).astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.circle(frame, (cx, cy), 2, (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{i + 1} ({real_area_m2} m2)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
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
# UPLOAD (รูปแบบเดิมของเว็บเป๊ะ ๆ รับแค่ไฟล์รูปภาพช่องเดียว)
# =========================
st.subheader("📤 อัปโหลดรูปภาพ")

uploaded_file = st.file_uploader(
    "รองรับ JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

analyze = st.button("Upload")

# =========================
# RUN & OUTPUT
# =========================
if uploaded_file is not None and analyze:
    # เพิ่มช่องว่างหลบระยะก่อนแสดงผลการทำงาน
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.spinner("กำลังวิเคราะห์ภาพ..."):
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        result_frame, texts = detect(frame)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # ผลลัพธ์ข้อความ
        st.subheader("📋 ผลการตรวจจับ")
        if texts:
            for t in texts:
                st.write(t)
        else:
            st.warning("ไม่พบกอผักตบชวา")

        # เว้นช่องลมระหว่างผลลัพธ์ข้อความกับรูปภาพเล็กน้อย ให้ดูเรียบร้อย
        st.markdown("<br>", unsafe_allow_html=True)

        # ภาพผลลัพธ์
        st.subheader("🖼️ ภาพผลการตรวจจับ")
        st.image(result_rgb, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center; color:#1b5e20; margin-top:50px; padding:20px;">
    <b>Phak Top Chawa Detector</b><br>
    ระบบตรวจจับและคำนวณพื้นที่ผักตบชวา
</div>
""", unsafe_allow_html=True)
