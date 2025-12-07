## Slajd 1 – Cel i kontekst
- Cel: bezpieczna anonimizacja danych PL (RODO, etyka) z zachowaniem sensu i struktury.
- Użycie: przygotowanie danych treningowych dla PLLuM / modeli PL.
- Zakres: 25 klas wrażliwych (name, surname, pesel, phone, address, city, document-number, company, itp.).
- Offline: brak zewnętrznych API; cały pipeline lokalny.

## Slajd 2 – Dane i przygotowanie
- Wejście: `orig.txt` (surowy tekst), `anonymized.txt` (gold).
- Konwersja do CoNLL: `convert_original_data.py` → `orig_train/dev/test.conll`.
- Tokenizacja: HuggingFace tokenizer (HerBERT).
- Split: train/dev/test, batching, max_length=256 (trening), 512 (inference).

## Slajd 3 – Model
- Baza: `allegro/herbert-base-cased`.
- Fine-tuning NER: `train_herbert_ner.py train` (seqeval F1 jako metric).
- Etykiety BIO dla 25 klas (B-/I- + O).
- Checkpoint i best model zapisywany lokalnie (`model_output_finetuned/best_model`).

## Slajd 4 – Inferencja / Anonimizacja
- Komenda: `train_herbert_ner.py predict --model-path ... --input orig.txt --output output/predicted_model_only.txt`.
- Tylko model (bez słowników/regex); urządzenie: CUDA/MPS/CPU.
- Offset mapping → łączenie subtokenów → scalanie encji → wstawianie tagów `[name]`, `[pesel]`, itd.
- Obsługa batchy, tworzenie katalogu wyjściowego automatycznie.

## Slajd 5 – Ewaluacja
- Skrypt: `evaluate_anonymization.py --orig output/predicted_model_only.txt --gold anonymized.txt --save-csv --save-errors`.
- Metryki: Precision, Recall, F1 per klasa; micro/macro F1; liczba linii perfekcyjnych.
- Zgodność z 25 klasami z zadania.

## Slajd 6 – Heurystyki i decyzje
- Rezygnacja z regexów/słowników w finalnym pipeline (czyste modelowe podejście).
- Scalanie nakładających się encji po pozycji (offsety).
- Brak zewnętrznych API → pełna zgodność offline.

## Slajd 7 – Wydajność i uruchomienie
- Batch inference, 512 max_length; CPU/MPS/CUDA autodetect.
- Szybkie tworzenie katalogu `output/` i zapis wyników.
- Instrukcje w `README.md` (trening `finetune_on_original.sh`, predykcja, ewaluacja).

## Slajd 8 – Co dalej (opcjonalne)
- Możliwa rozbudowa o generator syntetycznych danych (morfologia PL).
- Distillation/optimizacja dla throughput na TB danych.

