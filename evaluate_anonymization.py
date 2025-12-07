#!/usr/bin/env python3
"""
Skrypt ewaluacyjny do porównania zanonimizowanego pliku z plikiem referencyjnym (ground truth).
Oblicza metryki Precision, Recall, F1 dla każdej kategorii tagów.
"""

import re
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import json
import csv


# ============================================================================
# DOZWOLONE TAGI (z zadania)
# ============================================================================

VALID_TAGS = {
    'name', 'surname', 'age', 'date-of-birth', 'date', 'sex',
    'religion', 'political-view', 'ethnicity', 'sexual-orientation',
    'health', 'relative', 'city', 'address', 'email', 'phone',
    'pesel', 'document-number', 'company', 'school-name', 'job-title',
    'bank-account', 'credit-card-number', 'username', 'secret'
}


# ============================================================================
# STRUKTURY DANYCH
# ============================================================================

@dataclass
class Entity:
    """Reprezentuje wykrytą encję."""
    tag: str
    start: int
    end: int
    text: str
    line_num: int


@dataclass 
class EvaluationResult:
    """Wynik ewaluacji dla jednej kategorii."""
    tag: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


# ============================================================================
# PARSOWANIE PLIKÓW
# ============================================================================

def extract_entities_from_anonymized(text: str, line_num: int = 0, filter_valid: bool = True) -> List[Entity]:
    """
    Wyciąga encje z zanonimizowanego tekstu (format [tag]).
    
    Args:
        filter_valid: Jeśli True, filtruje tylko tagi z VALID_TAGS
    """
    entities = []
    # Wzorzec dla tagów: [nazwa-tagu]
    pattern = r'\[([a-z-]+)\]'
    
    for match in re.finditer(pattern, text):
        tag = match.group(1)
        # Filtruj tylko dozwolone tagi
        if filter_valid and tag not in VALID_TAGS:
            continue
        entities.append(Entity(
            tag=tag,
            start=match.start(),
            end=match.end(),
            text=match.group(0),
            line_num=line_num
        ))
    
    return entities


def extract_entities_from_original(orig_text: str, anon_text: str, line_num: int = 0) -> List[Entity]:
    """
    Wyciąga encje z oryginalnego tekstu na podstawie różnic z zanonimizowanym.
    Używa alignmentu między dwoma tekstami.
    """
    entities = []
    
    # Znajdź tagi w zanonimizowanym tekście
    tag_pattern = r'\[([a-z-]+)\]'
    anon_matches = list(re.finditer(tag_pattern, anon_text))
    
    if not anon_matches:
        return entities
    
    # Dla każdego tagu znajdź odpowiadający fragment w oryginale
    # Używamy prostej heurystyki: tekst przed i po tagu powinien być taki sam
    
    anon_pos = 0
    orig_pos = 0
    
    for match in anon_matches:
        tag = match.group(1)
        tag_start_anon = match.start()
        tag_end_anon = match.end()
        
        # Tekst przed tagiem (w zanonimizowanym)
        prefix = anon_text[anon_pos:tag_start_anon]
        
        # Znajdź ten sam prefix w oryginale
        if prefix:
            prefix_pos = orig_text.find(prefix, orig_pos)
            if prefix_pos != -1:
                orig_pos = prefix_pos + len(prefix)
        
        # Tekst po tagu (w zanonimizowanym)
        next_match_start = anon_matches[anon_matches.index(match) + 1].start() if match != anon_matches[-1] else len(anon_text)
        suffix_start = tag_end_anon
        suffix = anon_text[suffix_start:suffix_start + 50]  # Weź do 50 znaków jako suffix
        
        # Znajdź początek suffixu w oryginale
        if suffix:
            # Szukaj pierwszego znaczącego fragmentu suffixu
            suffix_clean = suffix.lstrip()
            if suffix_clean:
                suffix_pos = orig_text.find(suffix_clean[:min(20, len(suffix_clean))], orig_pos)
                if suffix_pos != -1:
                    # Encja to tekst między orig_pos a suffix_pos
                    entity_text = orig_text[orig_pos:suffix_pos].strip()
                    if entity_text:
                        entities.append(Entity(
                            tag=tag,
                            start=orig_pos,
                            end=suffix_pos,
                            text=entity_text,
                            line_num=line_num
                        ))
                    orig_pos = suffix_pos
        
        anon_pos = tag_end_anon
    
    return entities


def align_and_extract(orig_line: str, anon_line: str, line_num: int) -> Tuple[List[Entity], List[Entity]]:
    """
    Wyrównuje oryginalną i zanonimizowaną linię i wyciąga encje.
    Zwraca (predicted_entities, gold_entities).
    """
    # Gold entities - to co jest w pliku referencyjnym (anonymized.txt)
    gold_entities = extract_entities_from_anonymized(anon_line, line_num)
    
    # Predicted entities - zakładamy, że mamy model który anonimizuje orig.txt
    # W tym przypadku porównujemy format - czy model wygenerował te same tagi
    predicted_entities = extract_entities_from_anonymized(orig_line, line_num)
    
    return predicted_entities, gold_entities


def load_file_lines(filepath: str) -> List[str]:
    """Wczytuje plik linia po linii."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()


# ============================================================================
# PORÓWNANIE I METRYKI
# ============================================================================

def normalize_tag(tag: str) -> str:
    """Normalizuje tag do porównania."""
    return tag.lower().strip()


def entities_match(e1: Entity, e2: Entity, mode: str = 'exact') -> bool:
    """
    Sprawdza czy dwie encje pasują do siebie.
    
    mode:
        - 'exact': tag musi być identyczny
        - 'partial': częściowe dopasowanie (overlap)
    """
    if mode == 'exact':
        return normalize_tag(e1.tag) == normalize_tag(e2.tag)
    elif mode == 'partial':
        # Dla partial match - sprawdź czy tagi są podobne
        t1 = normalize_tag(e1.tag)
        t2 = normalize_tag(e2.tag)
        return t1 == t2 or t1 in t2 or t2 in t1
    return False


def compare_entity_lists(
    predicted: List[Entity], 
    gold: List[Entity],
    match_mode: str = 'exact'
) -> Dict[str, Dict[str, int]]:
    """
    Porównuje listy encji i zlicza TP, FP, FN dla każdego tagu.
    """
    results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    # Zbierz wszystkie unikalne tagi
    all_tags = set()
    for e in predicted + gold:
        all_tags.add(normalize_tag(e.tag))
    
    # Dla każdego tagu policz metryki
    for tag in all_tags:
        pred_for_tag = [e for e in predicted if normalize_tag(e.tag) == tag]
        gold_for_tag = [e for e in gold if normalize_tag(e.tag) == tag]
        
        # True Positives: ile z predicted jest w gold
        matched_gold = set()
        for p in pred_for_tag:
            for i, g in enumerate(gold_for_tag):
                if i not in matched_gold:
                    # Proste dopasowanie - ten sam tag na tej samej pozycji (względnej)
                    matched_gold.add(i)
                    results[tag]['tp'] += 1
                    break
            else:
                # Nie znaleziono dopasowania - FP
                results[tag]['fp'] += 1
        
        # False Negatives: ile z gold nie zostało dopasowane
        results[tag]['fn'] = len(gold_for_tag) - len(matched_gold)
    
    return dict(results)


def calculate_metrics(counts: Dict[str, int]) -> Tuple[float, float, float]:
    """Oblicza Precision, Recall, F1."""
    tp = counts.get('tp', 0)
    fp = counts.get('fp', 0)
    fn = counts.get('fn', 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


# ============================================================================
# GŁÓWNA EWALUACJA
# ============================================================================

class Evaluator:
    def __init__(self):
        self.per_tag_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
        self.errors = []
        self.total_lines = 0
        self.matching_lines = 0
    
    def evaluate_line(self, orig_line: str, anon_line: str, line_num: int):
        """Ewaluuje pojedynczą linię."""
        self.total_lines += 1
        
        # Wyciągnij encje z obu linii
        orig_entities = extract_entities_from_anonymized(orig_line, line_num)
        anon_entities = extract_entities_from_anonymized(anon_line, line_num)
        
        # Porównaj liczby encji dla każdego tagu
        orig_tags = defaultdict(int)
        anon_tags = defaultdict(int)
        
        for e in orig_entities:
            orig_tags[normalize_tag(e.tag)] += 1
        
        for e in anon_entities:
            anon_tags[normalize_tag(e.tag)] += 1
        
        # Zbierz wszystkie tagi
        all_tags = set(orig_tags.keys()) | set(anon_tags.keys())
        
        line_match = True
        for tag in all_tags:
            pred_count = orig_tags.get(tag, 0)
            gold_count = anon_tags.get(tag, 0)
            
            # TP = min(pred, gold)
            tp = min(pred_count, gold_count)
            # FP = nadmiarowe predykcje
            fp = max(0, pred_count - gold_count)
            # FN = brakujące predykcje
            fn = max(0, gold_count - pred_count)
            
            self.per_tag_counts[tag]['tp'] += tp
            self.per_tag_counts[tag]['fp'] += fp
            self.per_tag_counts[tag]['fn'] += fn
            
            if fp > 0 or fn > 0:
                line_match = False
                self.errors.append({
                    'line': line_num,
                    'tag': tag,
                    'predicted': pred_count,
                    'gold': gold_count,
                    'orig_preview': orig_line[:100],
                    'anon_preview': anon_line[:100]
                })
        
        if line_match:
            self.matching_lines += 1
    
    def evaluate_files(self, orig_path: str, anon_path: str):
        """Ewaluuje całe pliki."""
        orig_lines = load_file_lines(orig_path)
        anon_lines = load_file_lines(anon_path)
        
        if len(orig_lines) != len(anon_lines):
            print(f"⚠ Różna liczba linii: orig={len(orig_lines)}, anon={len(anon_lines)}")
            min_lines = min(len(orig_lines), len(anon_lines))
        else:
            min_lines = len(orig_lines)
        
        print(f"Ewaluacja {min_lines} linii...")
        
        for i in range(min_lines):
            self.evaluate_line(orig_lines[i], anon_lines[i], i + 1)
    
    def get_results(self) -> List[EvaluationResult]:
        """Zwraca wyniki ewaluacji."""
        results = []
        for tag, counts in sorted(self.per_tag_counts.items()):
            precision, recall, f1 = calculate_metrics(counts)
            results.append(EvaluationResult(
                tag=tag,
                true_positives=counts['tp'],
                false_positives=counts['fp'],
                false_negatives=counts['fn'],
                precision=precision,
                recall=recall,
                f1=f1
            ))
        return results
    
    def get_micro_average(self) -> Tuple[float, float, float]:
        """Oblicza micro-averaged metryki."""
        total_tp = sum(c['tp'] for c in self.per_tag_counts.values())
        total_fp = sum(c['fp'] for c in self.per_tag_counts.values())
        total_fn = sum(c['fn'] for c in self.per_tag_counts.values())
        
        return calculate_metrics({'tp': total_tp, 'fp': total_fp, 'fn': total_fn})
    
    def get_macro_average(self) -> Tuple[float, float, float]:
        """Oblicza macro-averaged metryki."""
        results = self.get_results()
        if not results:
            return 0.0, 0.0, 0.0
        
        avg_precision = sum(r.precision for r in results) / len(results)
        avg_recall = sum(r.recall for r in results) / len(results)
        avg_f1 = sum(r.f1 for r in results) / len(results)
        
        return avg_precision, avg_recall, avg_f1


# ============================================================================
# RAPORTOWANIE
# ============================================================================

def print_report(evaluator: Evaluator):
    """Drukuje raport ewaluacji."""
    print("\n" + "=" * 80)
    print("RAPORT EWALUACJI ANONIMIZACJI")
    print("=" * 80)
    
    results = evaluator.get_results()
    
    # Tabela wyników per tag
    print("\n### Wyniki per kategoria ###\n")
    print(f"{'Tag':<25} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 75)
    
    for r in sorted(results, key=lambda x: -x.f1):
        print(f"{r.tag:<25} {r.true_positives:>6} {r.false_positives:>6} {r.false_negatives:>6} "
              f"{r.precision:>8.4f} {r.recall:>8.4f} {r.f1:>8.4f}")
    
    print("-" * 75)
    
    # Średnie
    micro_p, micro_r, micro_f1 = evaluator.get_micro_average()
    macro_p, macro_r, macro_f1 = evaluator.get_macro_average()
    
    print(f"\n{'Micro-average':<25} {'-':>6} {'-':>6} {'-':>6} "
          f"{micro_p:>8.4f} {micro_r:>8.4f} {micro_f1:>8.4f}")
    print(f"{'Macro-average':<25} {'-':>6} {'-':>6} {'-':>6} "
          f"{macro_p:>8.4f} {macro_r:>8.4f} {macro_f1:>8.4f}")
    
    # Podsumowanie
    print(f"\n### Podsumowanie ###")
    print(f"Całkowita liczba linii: {evaluator.total_lines}")
    print(f"Linie z perfekcyjnym dopasowaniem: {evaluator.matching_lines} "
          f"({100*evaluator.matching_lines/evaluator.total_lines:.1f}%)")
    print(f"Liczba kategorii: {len(results)}")
    
    # Top błędy
    if evaluator.errors:
        print(f"\n### Przykładowe błędy (pierwsze 10) ###\n")
        for err in evaluator.errors[:10]:
            print(f"Linia {err['line']}: tag={err['tag']}, "
                  f"predicted={err['predicted']}, gold={err['gold']}")


def save_report_csv(evaluator: Evaluator, output_path: str):
    """Zapisuje raport do CSV."""
    results = evaluator.get_results()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['tag', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])
        
        for r in results:
            writer.writerow([r.tag, r.true_positives, r.false_positives, 
                           r.false_negatives, r.precision, r.recall, r.f1])
        
        # Średnie
        micro_p, micro_r, micro_f1 = evaluator.get_micro_average()
        macro_p, macro_r, macro_f1 = evaluator.get_macro_average()
        
        writer.writerow(['MICRO_AVG', '', '', '', micro_p, micro_r, micro_f1])
        writer.writerow(['MACRO_AVG', '', '', '', macro_p, macro_r, macro_f1])


def save_errors_json(evaluator: Evaluator, output_path: str):
    """Zapisuje szczegóły błędów do JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluator.errors, f, ensure_ascii=False, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ewaluacja anonimizacji - porównanie z plikiem referencyjnym'
    )
    parser.add_argument('--orig', type=str, default='orig.txt',
                        help='Ścieżka do pliku z predykcjami (zanonimizowany przez model)')
    parser.add_argument('--gold', type=str, default='anonymized.txt',
                        help='Ścieżka do pliku referencyjnego (ground truth)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Katalog na raporty')
    parser.add_argument('--save-csv', action='store_true',
                        help='Zapisz wyniki do CSV')
    parser.add_argument('--save-errors', action='store_true',
                        help='Zapisz szczegóły błędów do JSON')
    args = parser.parse_args()
    
    # Sprawdź pliki
    if not Path(args.orig).exists():
        print(f"❌ Nie znaleziono pliku: {args.orig}")
        return 1
    
    if not Path(args.gold).exists():
        print(f"❌ Nie znaleziono pliku: {args.gold}")
        return 1
    
    # Ewaluacja
    evaluator = Evaluator()
    evaluator.evaluate_files(args.orig, args.gold)
    
    # Raport
    print_report(evaluator)
    
    # Opcjonalne zapisy
    if args.save_csv or args.save_errors:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.save_csv:
            csv_path = output_dir / 'evaluation_results.csv'
            save_report_csv(evaluator, str(csv_path))
            print(f"\n✓ Zapisano CSV: {csv_path}")
        
        if args.save_errors:
            errors_path = output_dir / 'evaluation_errors.json'
            save_errors_json(evaluator, str(errors_path))
            print(f"✓ Zapisano błędy: {errors_path}")
    
    print("\n✓ Ewaluacja zakończona!")
    return 0


if __name__ == '__main__':
    exit(main())

