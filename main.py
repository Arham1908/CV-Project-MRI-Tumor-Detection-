from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
from datetime import datetime
import tensorflow as tf
import keras
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# model = load_model('model/model.h5', compile=False)


from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Arham1908/Tumor_Detection",
    filename="model.h5"
)

model = load_model(model_path, compile=False)


# Class labels
class_labels = ['notumor', 'glioma', 'meningioma', 'pituitary']

# Folders
UPLOAD_FOLDER = './uploads'
REPORT_FOLDER = './reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER

def predict_tumor(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence_score = np.max(predictions)
    label = class_labels[predicted_class_index]
    return ("No Tumor" if label == 'notumor' else f"Tumor: {label}", confidence_score)

def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines

def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

def create_pdf_report(report_path, image_path, filename, result, confidence, patient_info):
    page_width, page_height = 1240, 1754
    margin = 90
    report = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(report)

    title_font = get_font(48)
    heading_font = get_font(30)
    body_font = get_font(26)
    small_font = get_font(22)

    blue = (26, 78, 140)
    dark = (31, 41, 55)
    muted = (95, 108, 128)
    light_blue = (232, 241, 252)
    border = (204, 216, 232)

    draw.rectangle((0, 0, page_width, 170), fill=blue)
    draw.text((margin, 55), "MRI Brain Tumor Detection Report", fill="white", font=title_font)
    draw.text((margin, 120), "AI-assisted prediction summary", fill=(220, 235, 255), font=small_font)

    y = 230
    draw.text((margin, y), "Patient Details", fill=dark, font=heading_font)
    y += 50

    patient_box_bottom = y + 245
    draw.rounded_rectangle((margin, y, page_width - margin, patient_box_bottom), radius=18, fill=(248, 250, 252), outline=border, width=2)

    patient_details = [
        ("Report ID", patient_info["report_id"]),
        ("Patient Name", patient_info["patient_name"]),
        ("Age", patient_info["patient_age"]),
        ("Gender", patient_info["patient_gender"]),
        ("Doctor", patient_info["doctor_name"]),
        ("Hospital/Clinic", patient_info["hospital_name"]),
    ]

    left_x = margin + 35
    right_x = margin + 560
    detail_y = y + 35

    for index, (label, value) in enumerate(patient_details):
        column_x = left_x if index % 2 == 0 else right_x
        row_y = detail_y + (index // 2) * 68
        draw.text((column_x, row_y), label, fill=muted, font=small_font)
        draw.text((column_x, row_y + 28), value or "Not provided", fill=dark, font=body_font)

    y = patient_box_bottom + 70
    draw.text((margin, y), "Patient Image", fill=dark, font=heading_font)
    y += 50

    image_box = (margin, y, margin + 430, y + 430)
    draw.rounded_rectangle(image_box, radius=18, outline=border, width=3, fill=(248, 250, 252))

    with Image.open(image_path) as uploaded_image:
        uploaded_image = uploaded_image.convert("RGB")
        uploaded_image.thumbnail((390, 390))
        image_x = image_box[0] + ((image_box[2] - image_box[0]) - uploaded_image.width) // 2
        image_y = image_box[1] + ((image_box[3] - image_box[1]) - uploaded_image.height) // 2
        report.paste(uploaded_image, (image_x, image_y))

    details_x = margin + 500
    details_y = y + 10
    draw.text((details_x, details_y), "Prediction Details", fill=dark, font=heading_font)
    details_y += 60

    details = [
        ("Filename", filename),
        ("Prediction", result),
        ("Confidence", f"{confidence * 100:.2f}%"),
        ("Generated", patient_info["generated_at"]),
    ]

    for label, value in details:
        draw.text((details_x, details_y), label, fill=muted, font=small_font)
        details_y += 30
        draw.text((details_x, details_y), value, fill=dark, font=body_font)
        details_y += 60

    y += 510
    draw.rounded_rectangle((margin, y, page_width - margin, y + 250), radius=18, fill=light_blue, outline=border, width=2)
    draw.text((margin + 35, y + 35), "Important Note", fill=blue, font=heading_font)

    note = (
        "This report was generated by an AI-powered deep learning model trained on MRI images. "
        "It is intended to support screening workflows and must not be used as a final medical diagnosis. "
        "Please consult a qualified medical professional for interpretation and treatment decisions."
    )
    note_y = y + 90
    for line in wrap_text(draw, note, body_font, page_width - (2 * margin) - 70):
        draw.text((margin + 35, note_y), line, fill=dark, font=body_font)
        note_y += 38

    footer_text = "Generated by MRI Brain Tumor Detection System"
    footer_bbox = draw.textbbox((0, 0), footer_text, font=small_font)
    draw.text(
        ((page_width - footer_bbox[2]) // 2, page_height - 95),
        footer_text,
        fill=muted,
        font=small_font,
    )

    report.save(report_path, "PDF", resolution=100.0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            patient_info = {
                "report_id": f"BT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "patient_name": request.form.get("patient_name", "").strip(),
                "patient_age": request.form.get("patient_age", "").strip(),
                "patient_gender": request.form.get("patient_gender", "").strip(),
                "doctor_name": request.form.get("doctor_name", "").strip(),
                "hospital_name": request.form.get("hospital_name", "").strip(),
                "generated_at": generated_at,
            }

            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result, confidence = predict_tumor(file_path)

            report_filename = filename.rsplit('.', 1)[0] + '_report.pdf'
            report_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)

            create_pdf_report(report_path, file_path, filename, result, confidence, patient_info)

            print(f"[INFO] File saved at: {file_path}")
            print(f"[INFO] Report generated at: {report_path}")

            return render_template('index.html',
                                   result=result,
                                   confidence=f"{confidence*100:.2f}",
                                   file_path=f'/uploads/{filename}',
                                   report_filename=report_filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"[INFO] Serving uploaded file: {path}")
    return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)

@app.route('/download-report/<filename>')
def download_report(filename):
    abs_report_folder = os.path.abspath(app.config['REPORT_FOLDER'])
    file_path = os.path.join(abs_report_folder, filename)
    print(f"[INFO] Attempting to send report file: {file_path}")

    if os.path.exists(file_path):
        return send_from_directory(abs_report_folder, filename, as_attachment=True)
    else:
        print("[ERROR] Report file not found!")
        return "Report file not found", 404

if __name__ == '__main__':
    app.run(debug=True)
