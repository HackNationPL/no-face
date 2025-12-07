#!/usr/bin/env python3
"""
Skrypt do fine-tuningu modelu HerBERT na zadanie NER (Named Entity Recognition)
dla anonimizacji danych wrażliwych w tekstach polskich.

Użycie:
    python train_herbert_ner.py --train output/train_split.conll --dev output/dev.conll --output-dir model_output

Wymagania:
    pip install transformers torch datasets seqeval accelerate tqdm
"""

import os
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# KONFIGURACJA ETYKIET
# ============================================================================

# Wszystkie kategorie tagów z zadania
ENTITY_TYPES = [
    'name', 'surname', 'age', 'date-of-birth', 'date', 'sex',
    'religion', 'political-view', 'ethnicity', 'sexual-orientation',
    'health', 'relative', 'city', 'address', 'email', 'phone',
    'pesel', 'document-number', 'company', 'school-name', 'job-title',
    'bank-account', 'credit-card-number', 'username', 'secret'
]

def create_label_list() -> List[str]:
    """Tworzy listę etykiet BIO dla wszystkich kategorii."""
    labels = ['O']  # Outside
    for entity in ENTITY_TYPES:
        labels.append(f'B-{entity}')
        labels.append(f'I-{entity}')
    return labels

LABEL_LIST = create_label_list()
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


# ============================================================================
# WCZYTYWANIE DANYCH CONLL
# ============================================================================

def read_conll_file(filepath: str) -> List[Tuple[List[str], List[str]]]:
    """
    Wczytuje plik w formacie CoNLL.
    Zwraca listę (tokeny, etykiety) dla każdego przykładu.
    """
    examples = []
    tokens = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                # Koniec przykładu
                if tokens:
                    examples.append((tokens, labels))
                    tokens = []
                    labels = []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    token, label = parts[0], parts[1]
                    tokens.append(token)
                    # Normalizuj etykietę
                    if label not in LABEL2ID:
                        label = 'O'
                    labels.append(label)
    
    # Ostatni przykład
    if tokens:
        examples.append((tokens, labels))
    
    return examples


# ============================================================================
# DATASET DLA PYTORCH
# ============================================================================

class NERDataset(Dataset):
    """Dataset dla zadania NER z tokenizacją HerBERT."""
    
    def __init__(
        self, 
        examples: List[Tuple[List[str], List[str]]], 
        tokenizer,
        max_length: int = 512
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens, labels = self.examples[idx]
        
        # Tokenizacja z zachowaniem informacji o słowach
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=False,
        )
        
        # Mapowanie etykiet na subtokeny
        word_ids = encoding.word_ids()
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Pierwszy subtoken słowa - zachowaj etykietę
                label_ids.append(LABEL2ID.get(labels[word_idx], 0))
            else:
                # Kolejny subtoken tego samego słowa
                # Użyj I- zamiast B- lub -100 (ignoruj przy obliczaniu loss)
                original_label = labels[word_idx]
                if original_label.startswith('B-'):
                    # Zamień B- na I- dla kontynuacji
                    new_label = 'I-' + original_label[2:]
                    label_ids.append(LABEL2ID.get(new_label, 0))
                else:
                    label_ids.append(LABEL2ID.get(original_label, 0))
            previous_word_idx = word_idx
        
        encoding['labels'] = label_ids
        
        return {key: torch.tensor(val) for key, val in encoding.items()}


# ============================================================================
# METRYKI
# ============================================================================

def compute_metrics(eval_pred):
    """Oblicza metryki NER używając seqeval."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Konwertuj ID na etykiety, pomijając -100
    true_labels = []
    true_predictions = []
    
    for prediction, label in zip(predictions, labels):
        true_label = []
        true_pred = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_label.append(ID2LABEL[l])
                true_pred.append(ID2LABEL[p])
        true_labels.append(true_label)
        true_predictions.append(true_pred)
    
    return {
        'precision': precision_score(true_labels, true_predictions),
        'recall': recall_score(true_labels, true_predictions),
        'f1': f1_score(true_labels, true_predictions),
    }


# ============================================================================
# TRENING
# ============================================================================

def train_model(
    train_file: str,
    dev_file: str,
    output_dir: str,
    model_name: str = 'allegro/herbert-base-cased',
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    max_length: int = 256,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    fp16: bool = True,
    early_stopping_patience: int = 3,
):
    """Trenuje model HerBERT na zadaniu NER."""
    
    logger.info(f"Ładowanie modelu: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    # Wczytaj dane
    logger.info(f"Wczytywanie danych treningowych: {train_file}")
    train_examples = read_conll_file(train_file)
    logger.info(f"  Załadowano {len(train_examples)} przykładów treningowych")
    
    logger.info(f"Wczytywanie danych walidacyjnych: {dev_file}")
    dev_examples = read_conll_file(dev_file)
    logger.info(f"  Załadowano {len(dev_examples)} przykładów walidacyjnych")
    
    # Tworzenie datasetów
    train_dataset = NERDataset(train_examples, tokenizer, max_length)
    dev_dataset = NERDataset(dev_examples, tokenizer, max_length)
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
    )
    
    # Argumenty treningu
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',  # zmienione z evaluation_strategy
        save_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Symuluje większy batch
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=False,  # Wyłącz dla MPS (Mac)
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        report_to='none',  # Wyłącz wandb/tensorboard
        dataloader_pin_memory=False,  # Dla MPS
    )
    
    # Callbacks
    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # Trening
    logger.info("Rozpoczynam trening...")
    trainer.train()
    
    # Zapisz najlepszy model
    logger.info(f"Zapisywanie modelu do: {output_dir}/best_model")
    trainer.save_model(f'{output_dir}/best_model')
    tokenizer.save_pretrained(f'{output_dir}/best_model')
    
    # Zapisz mapowanie etykiet
    with open(f'{output_dir}/best_model/label_map.json', 'w') as f:
        json.dump({'id2label': ID2LABEL, 'label2id': LABEL2ID}, f, indent=2)
    
    # Ewaluacja końcowa
    logger.info("Ewaluacja końcowa...")
    results = trainer.evaluate()
    logger.info(f"Wyniki: {results}")
    
    # Szczegółowy raport
    logger.info("\nSzczegółowy raport klasyfikacji:")
    predictions, labels, _ = trainer.predict(dev_dataset)
    predictions = np.argmax(predictions, axis=2)
    
    true_labels = []
    true_predictions = []
    for prediction, label in zip(predictions, labels):
        true_label = []
        true_pred = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_label.append(ID2LABEL[l])
                true_pred.append(ID2LABEL[p])
        true_labels.append(true_label)
        true_predictions.append(true_pred)
    
    report = classification_report(true_labels, true_predictions, digits=4)
    logger.info(f"\n{report}")
    
    # Zapisz raport
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write(report)
    
    return trainer, results


# ============================================================================
# INFERENCJA
# ============================================================================

class NERAnonymizer:
    """Klasa do anonimizacji tekstu przy użyciu wytrenowanego modelu (bez fallbacków regex/słownikowych)."""
    
    def __init__(self, model_path: str):
        logger.info(f"Ładowanie modelu z: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        label_map_path = Path(model_path) / 'label_map.json'
        if label_map_path.exists():
            with open(label_map_path) as f:
                label_map = json.load(f)
                self.id2label = {int(k): v for k, v in label_map['id2label'].items()}
        else:
            self.id2label = ID2LABEL
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"Używam urządzenia: {self.device}")
        self.model.to(self.device)
    
    def predict_batch(self, texts: List[str]) -> List[List[Tuple[int, int, str]]]:
        """
        Przewiduje encje dla batcha tekstów.
        
        Returns:
            Lista list (start, end, label) dla każdego tekstu
        """
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
            return_offsets_mapping=True,
        )
        
        offset_mappings = inputs.pop('offset_mapping').tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2).cpu().tolist()
        
        all_results = []
        for text, preds, offsets in zip(texts, predictions, offset_mappings):
            results = []
            for pred, (start, end) in zip(preds, offsets):
                if start == end:  # Special token lub padding
                    continue
                label = self.id2label.get(pred, 'O')
                if label != 'O':
                    results.append((start, end, label))
            all_results.append(results)
        
        return all_results
    
    def anonymize(self, text: str) -> str:
        """Anonimizuje tekst wyłącznie na podstawie predykcji modelu."""
        if not text.strip():
            return text
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        
        offset_mapping = inputs.pop('offset_mapping')[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
        
        entities = []  # (start, end, entity_type)
        current_entity = None
        current_start = None
        current_end = None
        
        for pred, (start, end) in zip(predictions, offset_mapping):
            if start == end:  # Special token
                continue
            
            label = self.id2label.get(pred, 'O')
            
            if label.startswith('B-'):
                # Zakończ poprzednią encję
                if current_entity:
                    entities.append((current_start, current_end, current_entity))
                # Rozpocznij nową
                current_entity = label[2:]
                current_start = start
                current_end = end
            elif label.startswith('I-'):
                entity_type = label[2:]
                if current_entity == entity_type:
                    # Kontynuacja
                    current_end = end
                else:
                    # Niezgodność - zakończ poprzednią, rozpocznij nową
                    if current_entity:
                        entities.append((current_start, current_end, current_entity))
                    current_entity = entity_type
                    current_start = start
                    current_end = end
            else:  # O
                if current_entity:
                    entities.append((current_start, current_end, current_entity))
                    current_entity = None
        
        if current_entity:
            entities.append((current_start, current_end, current_entity))
        
        if not entities:
            return text
        
        entities.sort(key=lambda x: x[0])
        
        entities = self._merge_overlapping(entities)
        
        result = []
        last_end = 0
        
        for start, end, entity_type in entities:
            if start > last_end:
                result.append(text[last_end:start])
            result.append(f'[{entity_type}]')
            last_end = end
        
        if last_end < len(text):
            result.append(text[last_end:])
        
        return ''.join(result)
    
    def _merge_overlapping(self, entities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """Scala nakładające się encje."""
        if not entities:
            return entities
        
        merged = [entities[0]]
        for start, end, entity_type in entities[1:]:
            last_start, last_end, last_type = merged[-1]
            
            if start < last_end:
                # Nakładają się - weź dłuższą lub pierwszą
                if end > last_end:
                    merged[-1] = (last_start, end, last_type)
            else:
                merged.append((start, end, entity_type))
        
        return merged
    
    def anonymize_file(self, input_path: str, output_path: str, batch_size: int = 32):
        """Anonimizuje cały plik z batch processing."""
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total = len(lines)
        results = []
        
        # Przetwarzaj w batchach
        for i in tqdm(range(0, total, batch_size), desc='Anonimizacja'):
            batch = [line.strip() for line in lines[i:i+batch_size]]
            
            for text in batch:
                anonymized = self.anonymize(text)
                results.append(anonymized)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))
        
        logger.info(f"Zapisano {len(results)} linii do: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Trening HerBERT NER do anonimizacji danych wrażliwych'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Dostępne komendy')
    
    # Komenda: train
    train_parser = subparsers.add_parser('train', help='Trenuj model')
    train_parser.add_argument('--train', type=str, required=True,
                              help='Plik treningowy (CoNLL)')
    train_parser.add_argument('--dev', type=str, required=True,
                              help='Plik walidacyjny (CoNLL)')
    train_parser.add_argument('--output-dir', type=str, default='model_output',
                              help='Katalog wyjściowy dla modelu')
    train_parser.add_argument('--model', type=str, default='allegro/herbert-base-cased',
                              help='Nazwa modelu bazowego')
    train_parser.add_argument('--epochs', type=int, default=5,
                              help='Liczba epok')
    train_parser.add_argument('--batch-size', type=int, default=16,
                              help='Rozmiar batcha')
    train_parser.add_argument('--lr', type=float, default=5e-5,
                              help='Learning rate')
    train_parser.add_argument('--max-length', type=int, default=256,
                              help='Maksymalna długość sekwencji')
    train_parser.add_argument('--no-fp16', action='store_true',
                              help='Wyłącz mixed precision')
    
    # Komenda: predict
    predict_parser = subparsers.add_parser('predict', help='Anonimizuj tekst')
    predict_parser.add_argument('--model-path', type=str, required=True,
                                help='Ścieżka do wytrenowanego modelu')
    predict_parser.add_argument('--input', type=str,
                                help='Plik wejściowy do anonimizacji')
    predict_parser.add_argument('--output', type=str,
                                help='Plik wyjściowy')
    predict_parser.add_argument('--text', type=str,
                                help='Tekst do anonimizacji (zamiast pliku)')
    predict_parser.add_argument('--batch-size', type=int, default=32,
                                help='Rozmiar batcha dla przetwarzania plików')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(
            train_file=args.train,
            dev_file=args.dev,
            output_dir=args.output_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
            fp16=not args.no_fp16,
        )
    
    elif args.command == 'predict':
        anonymizer = NERAnonymizer(args.model_path)
        
        if args.text:
            result = anonymizer.anonymize(args.text)
            print(f"Wynik: {result}")
        elif args.input and args.output:
            anonymizer.anonymize_file(args.input, args.output, batch_size=args.batch_size)
            print(f"Zapisano do: {args.output}")
        else:
            print("Podaj --text lub --input/--output")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

