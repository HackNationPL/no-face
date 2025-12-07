## Uruchamianie (katalog `no-jak-to-nie-to-nie-wiem copy/nask_train`)

### Wymagania
- Python 3.9+
- `pip install -r requirements.txt`

### Trening na `orig.txt` i `anonymized.txt`
Skrypt tworzy dane CoNLL i trenuje model. Wynik ląduje w `model_output_finetuned/best_model`:
```bash
bash finetune_on_original.sh
```

### Predykcja (anonimizacja) – tylko model
```bash
python train_herbert_ner.py predict \
  --model-path model_output/best_model \
  --input orig.txt \
  --output output/predicted_model_only.txt \
  --batch-size 16
```

### Ewaluacja
```bash
python evaluate_anonymization.py \
  --orig output/predicted_model_only.txt \
  --gold anonymized.txt \
  --save-csv --save-errors
```

### Uwagi
- Anonimizacja przebiega wyłącznie przez model (bez słowników/regexów).
- Pliki pomocnicze zostały usunięte, aby zostawić tylko główny pipeline.

