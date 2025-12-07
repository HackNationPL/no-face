#!/usr/bin/env python3
"""
Konwertuje oryginalne dane (orig.txt + anonymized.txt) do formatu CoNLL dla treningu NER.

Metoda: Wyrównanie tekstu oryginalnego z zanonimizowanym przez porównanie fragmentów.
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import difflib
import json


@dataclass
class Entity:
    """Reprezentuje encję w tekście."""
    tag: str
    text: str
    start: int
    end: int


def extract_tags_from_anonymized(anon_text: str) -> List[Tuple[str, int, int]]:
    """
    Wyciąga pozycje tagów z zanonimizowanego tekstu.
    Zwraca: [(tag_name, start, end), ...]
    """
    pattern = r'\[([a-z-]+)\]'
    tags = []
    for match in re.finditer(pattern, anon_text):
        tags.append((match.group(1), match.start(), match.end()))
    return tags


def align_texts(orig_text: str, anon_text: str) -> List[Entity]:
    """
    Wyrównuje tekst oryginalny z zanonimizowanym i wyciąga encje.
    
    Metoda:
    1. Znajdź wszystkie tagi [xxx] w anon_text
    2. Dla każdego tagu, znajdź odpowiadający fragment w orig_text
       przez porównanie kontekstu przed i po tagu
    """
    entities = []
    tags = extract_tags_from_anonymized(anon_text)
    
    if not tags:
        return entities
    
    # Pozycja w oryginalnym tekście
    orig_pos = 0
    anon_pos = 0
    
    for i, (tag_name, tag_start, tag_end) in enumerate(tags):
        # Tekst przed tagiem w zanonimizowanym
        prefix = anon_text[anon_pos:tag_start]
        
        # Znajdź ten prefix w oryginale
        if prefix:
            # Szukaj dokładnego dopasowania
            prefix_idx = orig_text.find(prefix, orig_pos)
            if prefix_idx != -1:
                orig_pos = prefix_idx + len(prefix)
            else:
                # Spróbuj znaleźć podobny fragment
                # Weź ostatnie 20 znaków prefixu
                short_prefix = prefix[-20:] if len(prefix) > 20 else prefix
                prefix_idx = orig_text.find(short_prefix, orig_pos)
                if prefix_idx != -1:
                    orig_pos = prefix_idx + len(short_prefix)
        
        # Tekst po tagu w zanonimizowanym (suffix)
        next_tag_start = tags[i + 1][1] if i + 1 < len(tags) else len(anon_text)
        suffix = anon_text[tag_end:next_tag_start]
        
        # Znajdź początek suffixu w oryginale
        entity_start = orig_pos
        
        if suffix:
            # Szukaj początku suffixu
            # Weź pierwsze 30 znaków suffixu (lub mniej jeśli krótszy)
            search_suffix = suffix[:30].lstrip()
            if search_suffix:
                suffix_idx = orig_text.find(search_suffix, orig_pos)
                if suffix_idx != -1:
                    entity_end = suffix_idx
                    # Wyciągnij tekst encji
                    entity_text = orig_text[entity_start:entity_end].strip()
                    
                    if entity_text:
                        entities.append(Entity(
                            tag=tag_name,
                            text=entity_text,
                            start=entity_start,
                            end=entity_end
                        ))
                    
                    orig_pos = suffix_idx
                else:
                    # Nie znaleziono - spróbuj krótszego fragmentu
                    for length in [20, 10, 5]:
                        search_suffix = suffix[:length].lstrip()
                        if search_suffix:
                            suffix_idx = orig_text.find(search_suffix, orig_pos)
                            if suffix_idx != -1:
                                entity_end = suffix_idx
                                entity_text = orig_text[entity_start:entity_end].strip()
                                if entity_text:
                                    entities.append(Entity(
                                        tag=tag_name,
                                        text=entity_text,
                                        start=entity_start,
                                        end=entity_end
                                    ))
                                orig_pos = suffix_idx
                                break
        
        anon_pos = tag_end
    
    return entities


def simple_tokenize(text: str) -> List[Tuple[str, int, int]]:
    """
    Prosta tokenizacja - zwraca [(token, start, end), ...]
    """
    tokens = []
    pattern = r'\S+'
    for match in re.finditer(pattern, text):
        token = match.group()
        # Oddziel interpunkcję na końcu
        punct_match = re.match(r'^(.+?)([.,;:!?…\)\]]+)$', token)
        if punct_match and len(punct_match.group(1)) > 0:
            tokens.append((punct_match.group(1), match.start(), match.start() + len(punct_match.group(1))))
            tokens.append((punct_match.group(2), match.start() + len(punct_match.group(1)), match.end()))
        else:
            # Oddziel interpunkcję na początku
            punct_match = re.match(r'^([(\[„"]+)(.+)$', token)
            if punct_match:
                tokens.append((punct_match.group(1), match.start(), match.start() + len(punct_match.group(1))))
                tokens.append((punct_match.group(2), match.start() + len(punct_match.group(1)), match.end()))
            else:
                tokens.append((token, match.start(), match.end()))
    return tokens


def create_bio_labels(text: str, entities: List[Entity]) -> List[Tuple[str, str]]:
    """
    Tworzy etykiety BIO dla tokenów tekstu.
    Zwraca: [(token, label), ...]
    """
    tokens = simple_tokenize(text)
    labels = []
    
    for token_text, tok_start, tok_end in tokens:
        label = 'O'
        
        for entity in entities:
            # Sprawdź czy token jest wewnątrz encji
            if tok_start >= entity.start and tok_end <= entity.end:
                # Token jest w encji
                # Sprawdź czy to początek encji
                if tok_start == entity.start:
                    label = f'B-{entity.tag}'
                else:
                    # Sprawdź czy poprzedni token też był w tej encji
                    label = f'I-{entity.tag}'
                break
            elif tok_start < entity.end and tok_end > entity.start:
                # Częściowe pokrycie
                if tok_start <= entity.start:
                    label = f'B-{entity.tag}'
                else:
                    label = f'I-{entity.tag}'
                break
        
        labels.append((token_text, label))
    
    return labels


def convert_line(orig_line: str, anon_line: str) -> List[Tuple[str, str]]:
    """
    Konwertuje jedną linię do formatu BIO.
    """
    entities = align_texts(orig_line, anon_line)
    bio_labels = create_bio_labels(orig_line, entities)
    return bio_labels


def convert_files(orig_path: str, anon_path: str, output_path: str, 
                  train_ratio: float = 0.8, dev_ratio: float = 0.1):
    """
    Konwertuje pliki orig.txt i anonymized.txt do formatu CoNLL.
    """
    print(f"Wczytywanie: {orig_path}")
    with open(orig_path, 'r', encoding='utf-8') as f:
        orig_lines = f.readlines()
    
    print(f"Wczytywanie: {anon_path}")
    with open(anon_path, 'r', encoding='utf-8') as f:
        anon_lines = f.readlines()
    
    if len(orig_lines) != len(anon_lines):
        print(f"⚠ Różna liczba linii: orig={len(orig_lines)}, anon={len(anon_lines)}")
        min_lines = min(len(orig_lines), len(anon_lines))
    else:
        min_lines = len(orig_lines)
    
    print(f"Konwersja {min_lines} linii...")
    
    all_examples = []
    skipped = 0
    
    for i in range(min_lines):
        orig_line = orig_lines[i].strip()
        anon_line = anon_lines[i].strip()
        
        if not orig_line or not anon_line:
            skipped += 1
            continue
        
        try:
            bio_labels = convert_line(orig_line, anon_line)
            if bio_labels:
                all_examples.append(bio_labels)
        except Exception as e:
            print(f"  Błąd w linii {i+1}: {e}")
            skipped += 1
    
    print(f"Przekonwertowano: {len(all_examples)} przykładów (pominięto: {skipped})")
    
    # Podział na train/dev/test
    import random
    random.shuffle(all_examples)
    
    train_size = int(len(all_examples) * train_ratio)
    dev_size = int(len(all_examples) * dev_ratio)
    
    train_examples = all_examples[:train_size]
    dev_examples = all_examples[train_size:train_size + dev_size]
    test_examples = all_examples[train_size + dev_size:]
    
    # Zapisz pliki
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_conll(examples, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in examples:
                for token, label in example:
                    f.write(f"{token}\t{label}\n")
                f.write("\n")
    
    save_conll(train_examples, output_dir / 'orig_train.conll')
    save_conll(dev_examples, output_dir / 'orig_dev.conll')
    save_conll(test_examples, output_dir / 'orig_test.conll')
    
    # Zapisz wszystko razem
    save_conll(all_examples, output_dir / 'orig_all.conll')
    
    print(f"\nZapisano:")
    print(f"  - {output_dir}/orig_train.conll ({len(train_examples)} przykładów)")
    print(f"  - {output_dir}/orig_dev.conll ({len(dev_examples)} przykładów)")
    print(f"  - {output_dir}/orig_test.conll ({len(test_examples)} przykładów)")
    print(f"  - {output_dir}/orig_all.conll ({len(all_examples)} przykładów)")
    
    # Statystyki etykiet
    label_counts = {}
    for example in all_examples:
        for _, label in example:
            if label != 'O':
                tag = label[2:] if label.startswith(('B-', 'I-')) else label
                label_counts[tag] = label_counts.get(tag, 0) + 1
    
    print("\nStatystyki etykiet:")
    for tag, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")
    
    return len(all_examples)


def main():
    parser = argparse.ArgumentParser(
        description='Konwertuje orig.txt + anonymized.txt do formatu CoNLL'
    )
    parser.add_argument('--orig', type=str, default='orig.txt',
                        help='Plik z oryginalnymi danymi')
    parser.add_argument('--anon', type=str, default='anonymized.txt',
                        help='Plik z zanonimizowanymi danymi')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Katalog wyjściowy')
    args = parser.parse_args()
    
    convert_files(args.orig, args.anon, args.output_dir)


if __name__ == '__main__':
    main()

