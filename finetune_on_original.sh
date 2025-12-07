#!/bin/bash
# =============================================================================
# Fine-tuning na oryginalnych danych (orig.txt + anonymized.txt)
# =============================================================================

set -e

echo "=============================================="
echo "    Fine-tuning HerBERT na oryginalnych danych"
echo "=============================================="

OUTPUT_DIR="output"
MODEL_DIR="model_output"

# Krok 1: Konwersja oryginalnych danych do formatu CoNLL
echo ""
echo "[1/4] Konwersja oryginalnych danych..."
python convert_original_data.py \
    --orig orig.txt \
    --anon anonymized.txt \
    --output-dir "$OUTPUT_DIR"

# Krok 2: Fine-tuning na oryginalnych danych (kontynuacja z poprzedniego modelu)
echo ""
echo "[2/4] Fine-tuning modelu..."

# Sprawdź czy istnieje poprzedni model
if [ -d "$MODEL_DIR/best_model" ]; then
    echo "  Kontynuacja treningu z: $MODEL_DIR/best_model"
    BASE_MODEL="$MODEL_DIR/best_model"
else
    echo "  Brak poprzedniego modelu, trenuję od zera z herbert-base-cased"
    BASE_MODEL="allegro/herbert-base-cased"
fi

python train_herbert_ner.py train \
    --train "$OUTPUT_DIR/orig_train.conll" \
    --dev "$OUTPUT_DIR/orig_dev.conll" \
    --output-dir "${MODEL_DIR}_finetuned" \
    --model "$BASE_MODEL" \
    --epochs 5 \
    --batch-size 4 \
    --lr 2e-5 \
    --max-length 128

# Krok 3: Anonimizacja pliku testowego
echo ""
echo "[3/4] Anonimizacja pliku orig.txt z nowym modelem..."
python train_herbert_ner.py predict \
    --model-path "${MODEL_DIR}_finetuned/best_model" \
    --input orig.txt \
    --output "$OUTPUT_DIR/predicted_finetuned.txt"

# Krok 4: Ewaluacja
echo ""
echo "[4/4] Ewaluacja wyników..."
python evaluate_anonymization.py \
    --orig "$OUTPUT_DIR/predicted_finetuned.txt" \
    --gold anonymized.txt \
    --output-dir "$OUTPUT_DIR" \
    --save-csv

# Zmień nazwę pliku z wynikami
mv "$OUTPUT_DIR/evaluation_results.csv" "$OUTPUT_DIR/evaluation_finetuned.csv" 2>/dev/null || true

echo ""
echo "=============================================="
echo "    Fine-tuning zakończony!"
echo "=============================================="
echo ""
echo "Wyniki:"
echo "  - Model: ${MODEL_DIR}_finetuned/best_model/"
echo "  - Zanonimizowany plik: $OUTPUT_DIR/predicted_finetuned.txt"
echo "  - Raport ewaluacji: $OUTPUT_DIR/evaluation_finetuned.csv"

