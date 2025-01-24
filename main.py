import logging
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import re
import fitz  # PyMuPDF
from fastapi.responses import StreamingResponse
from zipfile import ZipFile
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Настроим логирование
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI()

# Загрузка модели и токенизатора
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Порог схожести
SIMILARITY_THRESHOLD = 0.85


# Подключение к базе данных
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="alta1204",
        database="dataset"
    )


# Функция для извлечения названия курса студента
def extract_student_course_name(text: str, tables: list = None) -> list:
    """
    Извлекает все названия курсов из текста или таблиц, если они присутствуют.

    :param text: Текст, извлеченный из PDF.
    :param tables: Список таблиц (опционально), если они есть.
    :return: Список названий курсов.
    """
    course_names = []

    # Попытка найти все названия курсов в тексте
    matches = re.findall(r"(?i)course\s*title[:\-]?\s*(.+)", text)
    for match in matches:
        course_name = match.strip()
        if course_name:
            logger.debug(f"Название курса найдено в тексте: {course_name}")
            course_names.append(course_name)

    # Если не нашли в тексте, ищем в таблицах
    if tables:
        for table in tables:
            for row in table:
                if row and len(row) > 0:
                    # Предполагаем, что название курса в первой ячейке строки таблицы
                    course_name = row[0].strip()
                    if course_name:
                        logger.debug(f"Название курса найдено в таблице: {course_name}")
                        course_names.append(course_name)

    # Если не нашли названия, возвращаем список с "Unknown"
    if not course_names:
        logger.debug("Названия курсов не найдены.")
        course_names.append("Unknown")

    return course_names

def extract_credits_for_course(text: str, course_name: str) -> list:
    """Извлекает кредиты, относящиеся к конкретному названию курса."""
    pattern = rf"(?i){re.escape(course_name)}\s*(?:.*?\b(\d+)\b)?"
    matches = re.findall(pattern, text)
    credits = []
    for match in matches:
        if match:
          try:
            credits.append(int(match))
          except ValueError:
            pass
    return credits


# Функция для вычисления сходства
def compute_similarity(text1: str, text2: str) -> float:
    inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

    similarity = cosine_similarity(embeddings1.numpy(), embeddings2.numpy())[0][0]
    return similarity


# Функция для извлечения текста из PDF
def extract_text_from_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        raise


def extract_credits_from_text(text: str) -> list:
    """
    Извлекает все кредиты из текста, используя регулярные выражения.

    :param text: Текст, извлеченный из PDF.
    :return: Список кредитов.
    """
    # Используем регулярное выражение для поиска всех чисел (кредитов)
    matches = re.findall(r"\b(\d+)\b", text)
    credits = [int(match) for match in matches]

    if credits:
        logger.debug(f"Кредиты, найденные в тексте: {credits}")
    return credits


def extract_credits_from_table(tables: list) -> list:
    """
    Извлекает кредиты из таблиц, предполагая, что кредиты находятся в 3-м столбце.

    :param tables: Список таблиц (списки строк).
    :return: Список кредитов.
    """
    credits = []
    if tables:
        for table in tables:
            for row in table:
                if row and len(row) > 2:  # Предполагаем, что кредиты находятся в третьем столбце
                    try:
                        credit = int(row[2].strip())  # Извлекаем кредиты из третьей ячейки
                        credits.append(credit)
                        logger.debug(f"Кредиты, найденные в таблице: {credit}")
                    except ValueError:
                        continue  # Если не удалось преобразовать в число, игнорируем
    return credits


def extract_credits(text: str, course_name: str) -> list:
    """
    Извлекает кредиты, относящиеся к конкретному названию курса.
    """
    # Улучшенное регулярное выражение: ищем кредиты рядом с названием курса
    pattern = rf"(?i){re.escape(course_name)}\s*(?:.*?\b(\d+)\b)?" # Ищем цифры (кредиты) после названия курса
    matches = re.findall(pattern, text)
    credits = []
    for match in matches:
        if match:
          try:
            credits.append(int(match))
          except ValueError:
            pass # Игнорируем нечисловые значения
    return credits



# Генерация PDF с результатами анализа
def generate_pdf(results: list, language: str = "en") -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Заголовки на разных языках
    headers = {
        "en": ["Student Course Name", "UniCam Course Name", "Similarity (%)", "Student Credits", "UniCam Credits",
               "Credits Difference", "Subject Area"],
        "it": ["Nome del Corso Studente", "Nome del Corso UniCam", "Somiglianza (%)", "Crediti Studente",
               "Crediti UniCam", "Differenza Crediti", "Area Tematica"]
    }

    elements = []

    # Заголовок PDF
    title_text = "File Analysis Results" if language == "en" else "Risultati dell'Analisi dei File"
    title = Paragraph(f"<b>{title_text}</b>", styles['Title'])
    elements.append(title)

    for result in results:
        # Добавление имени файла
        file_name_label = "File" if language == "en" else "File"
        file_name = Paragraph(f"<b>{file_name_label}:</b> {result['file_name']}", styles['Heading2'])
        elements.append(file_name)

        # Подготовка данных таблицы
        data = [headers[language]]

        for course_result in result['analysis']:
            data.append([
                Paragraph(course_result['student_course_name'], styles['Normal']),
                Paragraph(course_result['course_name'], styles['Normal']),
                f"{course_result['similarity_percentage']}%",
                course_result['student_credits'],
                course_result['original_credits'],
                course_result['credit_difference'],
                Paragraph(course_result['subject_area'], styles['Normal'])
            ])

        # Создание таблицы
        table = Table(data, colWidths=[120, 150, 80, 80, 80, 80, 80])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Серый фон для заголовка
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Белый текст для заголовка
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Выравнивание текста по центру
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Жирный шрифт для заголовка
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Отступы снизу для заголовка
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),  # Светлый фон для остальных строк
            ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Четкие границы
            ('FONTSIZE', (0, 0), (-1, -1), 10),  # Размер шрифта
            ('LEFTPADDING', (0, 0), (-1, -1), 5),  # Отступы слева
            ('RIGHTPADDING', (0, 0), (-1, -1), 5)  # Отступы справа
        ]))

        elements.append(table)
        elements.append(PageBreak())  # Добавление переноса страницы между файлами

    # Генерация PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


# Генерация ZIP-архива с PDF и оригинальными файлами
def generate_combined_zip(pdf_file: BytesIO, files: List[UploadFile]) -> BytesIO:
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.writestr("analysis_results.pdf", pdf_file.read())

        for file in files:
            zip_file.writestr(file.filename, file.file.read())

    zip_buffer.seek(0)
    return zip_buffer


@app.post("/analyze_files")
async def analyze_files_with_max_similarity(
        files: List[UploadFile] = File(...),
        language: str = "en"  # По умолчанию английский
):
    try:
        results = []
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)

        db = connect_to_db()
        cursor = db.cursor()
        cursor.execute("SELECT `Course name`, `Goals`, `CFU`, `Subject Area` FROM Courses")
        courses = cursor.fetchall()

        logger.debug(f"Fetched courses from DB: {courses}")

        for file in files:
            file_path = os.path.join(temp_dir, f"temp_{file.filename}")
            file_content = await file.read()
            with open(file_path, "wb") as f:
                f.write(file_content)

            if file.filename.endswith(".pdf"):
                try:
                    # Извлекаем текст из PDF
                    pdf_text = extract_text_from_pdf(file_path)
                    logger.debug(f"Extracted text from {file.filename}: {pdf_text[:500]}...")  # Логируем первые 500 символов текста

                    # Извлекаем все названия курсов
                    student_course_names = extract_student_course_name(pdf_text)
                    logger.debug(f"Extracted student course names: {student_course_names}")

                    # Используем список для всех результатов курсов с их схожестью
                    course_results = []  # Список для всех курсов с их схожестью

                    # Обрабатываем все курсы студента
                    for student_course_name in student_course_names:
                        best_similarity = 0
                        best_course_name = ""
                        best_course_credits = 0
                        best_subject_area = ""
                        best_credit_difference = 0

                        # Проходим по всем курсам из базы данных и находим наибольшую схожесть
                        for course_name, course_goal, course_credits, subject_area in courses:
                            combined_course_text = f"{course_name}. {course_goal}"

                            # Вычисляем схожесть
                            similarity = compute_similarity(pdf_text, combined_course_text)
                            similarity_percentage = round(similarity * 100, 2)

                            # Извлекаем кредиты для данного текста
                            student_credits = extract_credits(pdf_text, student_course_name)
                            if student_credits:
                                student_credit = student_credits[0]  # Для каждого курса берем первый кредит

                            credit_difference = abs(student_credit - int(course_credits))  # Разница в кредитах

                            # Если схожесть выше, обновляем лучший курс
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_course_name = course_name
                                best_course_credits = int(course_credits)
                                best_subject_area = subject_area
                                best_credit_difference = credit_difference

                        # Добавляем лучший курс для текущего курса студента в результаты
                        if best_similarity >= SIMILARITY_THRESHOLD:
                            course_results.append({
                                "student_course_name": student_course_name,
                                "course_name": best_course_name,
                                "similarity_percentage": round(best_similarity * 100, 2),
                                "original_credits": best_course_credits,
                                "student_credits": student_credit,
                                "credit_difference": best_credit_difference,
                                "subject_area": best_subject_area
                            })

                    # Добавляем все результаты для текущего файла
                    if course_results:
                        results.append({
                            "file_name": file.filename,
                            "analysis": course_results
                        })
                    else:
                        logger.debug(f"No matching results for {file.filename}")

                except Exception as ex:
                    logger.error(f"Error analyzing file {file.filename}: {str(ex)}")

            # Удаляем временный файл
            os.remove(file_path)

        # Если есть результаты для файлов, генерируем PDF и ZIP
        if results:
            pdf_file = generate_pdf(results)
            combined_zip = generate_combined_zip(pdf_file, files)

            return StreamingResponse(combined_zip, media_type="application/zip", headers={
                "Content-Disposition": "attachment; filename=analysis_results.zip"
            })
        else:
            logger.warning("No results to generate PDF.")

    except Exception as e:
        logger.error(f"Error analyzing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing files: {str(e)}")
