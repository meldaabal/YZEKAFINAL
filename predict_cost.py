import pandas as pd
import joblib
import gradio as gr

# Model ve scaler yükleniyor
model = joblib.load("model_output/random_forest_model.pkl")
scaler = joblib.load("model_output/scaler.pkl")

# Modelin eğitimde gördüğü sütun sırası
expected_columns = [
    'age', 'bmi', 'children', 'sex_male', 'smoker_yes',
    'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'
]

def tahmin_et(age, bmi, children, sex, smoker, region):
    try:
        # Kullanıcıdan gelen girdileri uygun hale getir
        input_data = {
            'age': int(age),
            'bmi': float(bmi),
            'children': int(children),
            'sex_male': 1 if sex == "male" else 0,
            'smoker_yes': 1 if smoker == "yes" else 0,
            'region_northeast': 1 if region == "northeast" else 0,
            'region_northwest': 1 if region == "northwest" else 0,
            'region_southeast': 1 if region == "southeast" else 0,
            'region_southwest': 1 if region == "southwest" else 0,
        }

        # DataFrame’e dönüştür ve sütun sırasını sabitle
        input_df = pd.DataFrame([input_data])[expected_columns]

        # Veriyi ölçekle ve tahmin yap
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        return f"💰 Tahmini Sigorta Maliyeti: **{int(prediction)} TL**"

    except Exception as e:
        return f"❌ HATA: {str(e)}"

# Gradio arayüzü
interface = gr.Interface(
    fn=tahmin_et,
    inputs=[
        gr.Number(label="Yaş"),
        gr.Number(label="BMI (Vücut Kitle İndeksi)"),
        gr.Number(label="Çocuk Sayısı"),
        gr.Radio(["male", "female"], label="Cinsiyet"),
        gr.Radio(["yes", "no"], label="Sigara Kullanıyor mu?"),
        gr.Dropdown(["northeast", "northwest", "southeast", "southwest"], label="Bölge"),
    ],
    outputs=gr.Markdown(),
    title="🩺 Sigorta Maliyeti Tahmini",
    description="Bilgilerinizi girin, tahmini sağlık sigorta ücretinizi öğrenin!"
)

# Arayüzü başlat
interface.launch()
