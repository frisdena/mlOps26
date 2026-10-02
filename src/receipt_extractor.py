import json
import os
import re
from pathlib import Path

import lmstudio as lms


IMAGE = Path("data/raw/nota-sample.png")
MODEL = os.environ["LM_STUDIO_MODEL"]
OUTPUT = Path("reports/receipt.json")


# Menyiapkan gambar
image = lms.prepare_image(str(IMAGE))

# Memuat model LM Studio
model = lms.llm(MODEL)

# Membuat prompt
chat = lms.Chat()

chat.add_user_message(
    "Baca nota pada gambar ini. "
    "Ekstrak informasi berikut: merchant, tanggal, item, "
    "subtotal, pajak, dan total. "
    "Keluarkan HANYA JSON valid tanpa markdown atau teks tambahan. "
    "Jika pajak tidak terlihat, isi 0. "
    "Jangan mengarang data.",
    images=[image],
)

# Meminta model memproses gambar
prediction = model.respond(chat)

# Melihat respons asli model
print("\n=== RESPONS LM STUDIO ===")
print(prediction.content)
print("=========================\n")

content = prediction.content.strip()


# Menghapus markdown code fence jika ada
content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
content = re.sub(r"^```\s*", "", content)
content = re.sub(r"\s*```$", "", content)

content = content.strip()


# Mengubah respons menjadi JSON
try:
    result = json.loads(content)

except json.JSONDecodeError:
    # Mencoba mengambil bagian JSON dari respons
    match = re.search(r"\{.*\}", content, re.DOTALL)

    if not match:
        raise ValueError(
            "Respons LM Studio bukan JSON yang valid:\n"
            + prediction.content
        )

    result = json.loads(match.group(0))


# Membuat folder reports jika belum ada
OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

# Menyimpan hasil
OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

# Menampilkan hasil
print("=== HASIL EKSTRAKSI ===")
print(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )
)

print(f"\nHasil disimpan di: {OUTPUT}")
