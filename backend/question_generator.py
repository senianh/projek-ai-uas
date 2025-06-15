import random

def generate_mcq(kalimat, label=None, topik=None):
    # Buat pertanyaan MCQ berdasarkan kalimat
    question_templates = [
        f"Apa pernyataan yang paling tepat terkait: '{kalimat[:50]}...'",
        f"Apa arti dari kalimat berikut: '{kalimat[:50]}...'",
        f"Apa makna dari pernyataan berikut: '{kalimat[:50]}...'",
        f"Manakah yang paling sesuai dengan kalimat: '{kalimat[:50]}...'"
    ]
    question = random.choice(question_templates)

    # Jawaban benar
    correct = kalimat

    # Distraktor dummy
    distractors = [
        "Pilihan yang tampaknya benar tapi tidak tepat.",
        "Pernyataan lain yang tidak relevan.",
        "Kalimat dengan informasi yang menyesatkan."
    ]
    options = [correct] + random.sample(distractors, k=3)
    random.shuffle(options)

    return {
        "question": question,
        "options": options,
        "answer": correct,
        "label": label,
        "topic": topik
    }

def generate_essay(kalimat, label=None, topik=None):
    return {
        "question": f"Jelaskan lebih lanjut tentang: \"{kalimat}\"",
        "label": label,
        "topic": topik
    }
