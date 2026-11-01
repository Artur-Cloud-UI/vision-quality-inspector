import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

# Konfiguracja strony
st.set_page_config(
    page_title="Vision Quality Inspector - Batch",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Vision Quality Inspector - Analiza wsadowa (Folder / Wiele plików)")
st.markdown("Wgraj wiele zdjęć naraz, aby przetestować cały folder i zobaczyć wynik dla każdego pliku w czytelnej tabeli.")

# Panel boczny z ustawieniami
st.sidebar.header("Ustawienia detekcji")
threshold = st.sidebar.slider(
    "Próg anomalii (Threshold)", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.47, 
    step=0.01,
    help="Wszystko powyżej tej wartości zostanie uznane za DEFECT."
)

st.sidebar.info(
    "💡 **Wskazówka:** Możesz zaznaczyć myszką cały folder zdjęć (Ctrl+A w folderze) "
    "i wgrać je wszystkie naraz do poniższego pola!"
)

# Wgrywanie wielu plików naraz
uploaded_files = st.file_uploader(
    "Wybierz zdjęcia do analizy (możesz zaznaczyć wiele plików)...", 
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 Wybrano plików do analizy: **{len(uploaded_files)}**")
    
    if st.button("Uruchom analizę wsadową", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Pętla przechodząca przez każdy wgrany plik
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Przetwarzanie ({i+1}/{len(uploaded_files)}): {uploaded_file.name}...")
            
            try:
                image = Image.open(uploaded_file)
                img_byte_arr = io.BytesIO()
                image_format = image.format if image.format else "PNG"
                image.save(img_byte_arr, format=image_format)
                img_byte_arr = img_byte_arr.getvalue()
                
                files = {"file": (uploaded_file.name, img_byte_arr, f"image/{image_format.lower()}_bytes")}
                
                # Strzał do naszego backendu FastAPI
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    score = data.get("anomaly_score")
                    
                    # Ocena na podstawie suwaka progu
                    is_defect = score > threshold
                    status = "DEFECT" if is_defect else "OK"
                    
                    results.append({
                        "Plik": uploaded_file.name,
                        "Wynik (Score)": round(score, 4),
                        "Status": status
                    })
                else:
                    results.append({
                        "Plik": uploaded_file.name,
                        "Wynik (Score)": 0.0,
                        "Status": "BŁĄD API"
                    })
            except Exception as e:
                results.append({
                    "Plik": uploaded_file.name,
                    "Wynik (Score)": 0.0,
                    "Status": f"Błąd: {str(e)}"
                })
            
            # Aktualizacja paska postępu
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.success("Analiza wsadowa zakończona pomyślnie!")
        
        # Wyświetlenie tabeli z wynikami
        if results:
            df = pd.DataFrame(results)
            
            st.subheader("📊 Wyniki dla wszystkich plików:")
            st.dataframe(df, use_container_width=True)
            
            # Podsumowanie liczbowe (metryki)
            ok_count = sum(1 for r in results if r["Status"] == "OK")
            defect_count = sum(1 for r in results if r["Status"] == "DEFECT")
            
            col1, col2 = st.columns(2)
            col1.metric("Liczba elementów poprawnych (OK)", ok_count)
            col2.metric("Liczba wykrytych wad (DEFECT)", defect_count)