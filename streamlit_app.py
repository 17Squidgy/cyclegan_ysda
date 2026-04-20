import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tr
import os
import io

from models.cyclegan import CycleGAN


# ============================================================
# Конфигурация
# ============================================================
CHECKPOINT_PATH = "checkpoints/cycle_gan.pt"
IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Нормализация как при обучении
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


# ============================================================
# Кэшированная загрузка модели (загружается один раз)
# ============================================================
@st.cache_resource
def load_model(checkpoint_path: str):
    """Загружает модель из чекпоинта и возвращает её в eval-режиме."""
    model = CycleGAN(input_channels=3)

    if not os.path.exists(checkpoint_path):
        st.error(f"Чекпоинт не найден: {checkpoint_path}")
        st.stop()

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Функции предобработки и постобработки
# ============================================================
def get_transform():
    """Трансформация входного изображения для инференса."""
    return tr.Compose([
        tr.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BICUBIC),
        tr.ToTensor(),
        tr.Normalize(mean=MEAN, std=STD),
    ])


def de_normalize(tensor: torch.Tensor) -> np.ndarray:
    """Обратная нормализация тензора → numpy [H, W, 3] в диапазоне [0, 255]."""
    tensor = tensor.detach().cpu().clone()
    mean_t = torch.tensor(MEAN).view(3, 1, 1)
    std_t = torch.tensor(STD).view(3, 1, 1)

    image = tensor * std_t + mean_t
    image = image.clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return image


def process_image(model, pil_image: Image.Image, direction: str) -> Image.Image:
    """
    Принимает PIL-изображение, прогоняет через генератор,
    возвращает результат как PIL-изображение.
    """
    transform = get_transform()

    # Конвертируем в RGB на случай RGBA/grayscale
    pil_image = pil_image.convert("RGB")

    # Предобработка
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    # Инференс
    with torch.no_grad():
        if direction == "AtoB":
            output_tensor = model.GenAB(input_tensor)
        else:
            output_tensor = model.GenBA(input_tensor)

    # Постобработка
    output_array = de_normalize(output_tensor.squeeze(0))
    output_image = Image.fromarray(output_array)

    return output_image


# ============================================================
# Интерфейс Streamlit
# ============================================================
def main():
    # --- Настройки страницы ---
    st.set_page_config(
        page_title="CycleGAN Demo — Summer ↔ Winter",
        page_icon="🔄",
        layout="wide",
    )

    # --- Заголовок ---
    st.title("🔄 CycleGAN: Summer ↔ Winter Yosemite")
    st.markdown("""
    Это приложение демонстрирует работу модели CycleGAN, обученной
    на датасете summer2winter_yosemite.

    - Summer → Winter: загрузите летнее фото, получите зимнюю версию
    - Winter → Summer: загрузите зимнее фото, получите летнюю версию

    ---
    """)

    # --- Загрузка модели ---
    with st.spinner("Загрузка модели..."):
        model = load_model(CHECKPOINT_PATH)
    st.success("Модель загружена!")

    # --- Боковая панель: выбор направления ---
    st.sidebar.header("⚙️ Настройки")
    direction = st.sidebar.radio(
        "Направление преобразования:",
        options=["Summer → Winter", "Winter → Summer"],
        index=0,
    )

    direction_code = "AtoB" if "Summer → Winter" in direction else "BtoA"

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    Описание доменов:
    - Domain A — Летние фотографии Йосемити
    - Domain B — Зимние фотографии Йосемити
    """)

    # --- Основная область: загрузка изображения ---
    st.header(f"📸 {direction}")

    col_upload, col_result = st.columns(2)

    with col_upload:
        st.subheader("Входное изображение")

        # Выбор источника изображения
        input_method = st.radio(
            "Способ загрузки:",
            options=["Загрузить файл", "Использовать пример"],
            horizontal=True,
        )

        uploaded_image = None

        if input_method == "Загрузить файл":
            uploaded_file = st.file_uploader(
                "Выберите изображение",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                help="Загрузите фотографию для преобразования",
            )
            if uploaded_file is not None:
                uploaded_image = Image.open(uploaded_file)

        else:
            # Примеры изображений
            examples_dir = "examples"
            if os.path.exists(examples_dir):
                example_files = sorted([
                    f for f in os.listdir(examples_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                if example_files:
                    selected_example = st.selectbox(
                        "Выберите пример:",
                        options=example_files,
                    )
                    example_path = os.path.join(examples_dir, selected_example)
                    uploaded_image = Image.open(example_path)
                else:
                    st.warning("Папка examples/ пуста.")
            else:
                st.warning("Папка examples/ не найдена.")

        # Отображение загруженного изображения
        if uploaded_image is not None:
            st.image(
                uploaded_image,
                caption=f"Оригинал ({uploaded_image.size[0]}×{uploaded_image.size[1]})",
                use_container_width=True,
            )

    with col_result:
        st.subheader("Результат")

        if uploaded_image is not None:
            # Кнопка запуска
            if st.button("🚀 Преобразовать", type="primary", use_container_width=True):
                with st.spinner("Обработка изображения..."):
                    result_image = process_image(model, uploaded_image, direction_code)

                st.image(
                    result_image,
                    caption=f"Результат ({direction})",
                    use_container_width=True,
                )

                # Кнопка скачивания результата
                buf = io.BytesIO()
                result_image.save(buf, format="PNG")
                buf.seek(0)

                st.download_button(
                    label="📥 Скачать результат",
                    data=buf,
                    file_name="cyclegan_result.png",
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            st.info("👈 Загрузите изображение слева для преобразования")

    # --- Нижняя секция: информация о модели ---
    st.markdown("---")

    with st.expander("ℹ️ Информация о модели"):
        st.markdown(f"""
        | Параметр | Значение |
        |---|---|
        | Архитектура генератора | ResNet (9 блоков) |
        | Архитектура дискриминатора | PatchGAN (3 слоя) |
        | Размер входа | {IMAGE_SIZE}×{IMAGE_SIZE} |
        | Число параметров генератора | ~11.4M |
        | Число параметров дискриминатора | ~2.8M |
        | Loss | MSE Adversarial + Cycle Consistency (λ=10) |
        | Датасет | summer2winter_yosemite |
        | Устройство | {DEVICE} |
        """)

    # --- Batch-режим ---
    st.markdown("---")
    st.header("📦 Пакетная обработка")

    uploaded_files = st.file_uploader(
        "Загрузите несколько изображений",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="batch_upload",
    )

if uploaded_files:
        if st.button("🚀 Обработать все", use_container_width=True):
            results = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Обработка {i + 1}/{len(uploaded_files)}: {file.name}")

                pil_img = Image.open(file)
                result_img = process_image(model, pil_img, direction_code)
                results.append((file.name, pil_img, result_img))

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("Готово!")

            # Отображение результатов в сетке
            for name, original, result in results:
                st.markdown(f"**{name}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original, caption="Оригинал", use_container_width=True)
                with col2:
                    st.image(result, caption="Результат", use_container_width=True)
                st.markdown("---")


if name == "__main__":
    main()