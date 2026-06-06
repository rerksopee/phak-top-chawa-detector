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
    margin-bottom: 16px; 
}

/* SUB TITLE */
.sub-title {
    text-align: center;
    color: #1b5e20 !important;
    font-size: 20px;
    margin-bottom: 35px; 
    font-weight: 500;
}

/* FILE UPLOADER DESIGN */
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
    margin-top: 10px; 
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
# ฟังก์ชันตรวจจับและคำนวณพื้นที่จริงตามระยะลึก (กล้อง Depth)
# =========================
def detect(frame, depth_frame):
    """
    frame: ภาพสี RGB/BGR
    depth_frame: อาร์เรย์ระยะลึกที่ได้จากกล้องวัดระยะในเวลาเดียวกัน (หน่วยมิลลิเมตร mm)
    """
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

            # หาพิกัดพิกเซลทั้งหมดที่เป็นกอผักตบชวานี้
            ys, xs = np.where(binary)

            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

            # -----------------------------------------------------------------
            # 🛠️ ส่วนปรับปรุงหลัก: คำนวณพื้นที่พิกเซลต่อพิกเซลอิงระยะลึกจากกล้องจริง
            # -----------------------------------------------------------------
            real_area_m2 = 0.0
            
            # วนลูปคำนวณหาพื้นที่จริงของทุกพิกเซลที่รวมกันเป็นกอนี้
            for y_pt, x_pt in zip(ys, xs):
                # ดึงค่าระยะลึก Z ของพิกเซลนั้น (แปลงหน่วยจาก มิลลิเมตร mm เป็น เมตร m)
                z_meters = depth_frame[y_pt, x_pt] / 1000.0
                
                if z_meters > 0:
                    # ⚠️ เปลี่ยนค่า 15000.0 เป็นจำนวนพิกเซลจริงที่คุณนับได้ในกรอบ 1x1 เมตรที่ระยะ 1 เมตร
                    pixels_constant_at_1m = 15000.0 
                    
                    # ชดเชยกำลังสองผกผัน: หาขนาดพื้นที่จริง (ตร.ม.) ของพิกเซลจุดนี้ ณ ระยะ Z
                    pixel_area_m2 = 1.0 / (pixels_constant_at_1m / (z_meters ** 2))
                    real_area_m2 += pixel_area_m2

            # ปัดเศษทศนิยม 4 ตำแหน่ง
            real_area_m2 = round(real_area_m2, 4)

            # ดึงระยะลึกเฉลี่ยมาเพื่อแสดงผลบนหน้าเว็บเพิ่มเติม
            valid_depths = depth_frame[ys, xs]
            valid_depths = valid_depths[valid_depths > 0]
            avg_z_m = round(np.mean(valid_depths) / 1000.0, 2) if len(valid_depths) > 0 else 0.0

            # เปลี่ยนรายงานผลเป็นหน่วย "ตารางเมตร (sq.m.)" เรียบร้อยครับ
            output_text.append(f"กอ#{i+1} พื้นที่จริง: {real_area_m2} ตร.ม. (ระยะห่าง: {avg_z_m} ม.)")

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
<div class="sub-title">ระบบตรวจจับและคำนวณพื้นที่ผักตบชวาตามระยะลึกกล้องจริง</div>
""", unsafe_allow_html=True)

# =========================
# UPLOAD
# =========================
st.subheader("📤 อัปโหลดรูปภาพและข้อมูลระยะลึก")

# ส่วนรับภาพสี RGB ปกติ
uploaded_file = st.file_uploader(
    "1. เลือกไฟล์ภาพสีผักตบชวา (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    key="rgb_image"
)

# ส่วนรับไฟล์ระยะลึก (เพื่อนำมาวิเคราะห์คู่กัน)
# แนะนำให้บันทึกค่าเป็นไฟล์ข้อมูลอาร์เรย์ .npy หรือภาพ Depth Map สเกลเทา
uploaded_depth = st.file_uploader(
    "2. เลือกไฟล์ข้อมูลระยะลึกจากกล้อง (ไฟล์ข้อมูลอาร์เรย์ .npy หรือภาพ Depth Map)",
    type=["npy", "png", "jpg"],
    key="depth_data"
)

analyze = st.button("วิเคราะห์และคำนวณพื้นที่")

# =========================
# RUN & OUTPUT
# =========================
if uploaded_file is not None and uploaded_depth is not None and analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.spinner("กำลังประมวลผลคำนวณพื้นที่แบบสมจริง..."):
        # อ่านภาพสี
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # อ่านข้อมูลระยะลึก (Depth Frame)
        if uploaded_depth.name.endswith('.npy'):
            depth_frame = np.load(uploaded_depth)
        else:
            # หากอัปโหลดเป็นภาพ Depth Map สีเทา ให้เปลี่ยนเป็นข้อมูลตัวเลขความลึก
            depth_img = Image.open(uploaded_depth).convert("L")
            depth_frame = np.array(depth_img).astype(np.float32) 
            # หมายเหตุ: ควรทำการ Map ค่าพิกเซลสีเทา (0-255) กลับมาเป็นหน่วยมิลลิเมตรตามสเปกกล้องของคุณ

        # ปรับขนาดข้อมูลภาพลึกให้ตรงกับขนาดภาพสี
        if depth_frame.shape[:2] != frame.shape[:2]:
            depth_frame = cv2.resize(depth_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # ประมวลผลลัพธ์
        result_frame, texts = detect(frame, depth_frame)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # ผลลัพธ์ข้อความ
        st.subheader("📋 ผลการคำนวณขนาดพื้นที่ตารางเมตร")
        if texts:
            for t in texts:
                st.write(t)
        else:
            st.warning("ไม่พบกอผักตบชวาในระบบ")

        st.markdown("<br>", unsafe_allow_html=True)

        # ภาพผลลัพธ์
        st.subheader("🖼️ ภาพผลการตรวจจับ (พร้อมค่าตารางเมตรชดเชยระยะ)")
        st.image(result_rgb, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center; color:#1b5e20; margin-top:50px; padding:20px;">
    <b>Phak Top Chawa Detector v2</b><br>
    ระบบตรวจจับและประมาณการพื้นที่สมจริงด้วยกล้องวัดระยะ 3 มิติ
</div>
""", unsafe_allow_html=True)
