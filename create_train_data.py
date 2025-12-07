#!/usr/bin/env python3
"""
Skrypt do tworzenia danych treningowych w formacie JSONL.
Losowo wybiera 15% linijek z plików orig.txt i anonymized.txt.
"""

import json
import random
from pathlib import Path

# Konfiguracja
SAMPLE_RATIO = 0.15  # 15% danych
RANDOM_SEED = 42     # Dla powtarzalności wyników

def load_lines(filepath: Path) -> list[str]:
    """Wczytuje linie z pliku tekstowego."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def create_train_data(orig_path: Path, anon_path: Path, output_path: Path, sample_ratio: float = 0.15, seed: int = 42):
    """
    Tworzy dane treningowe w formacie JSONL.
    
    Args:
        orig_path: Ścieżka do pliku z oryginalnymi danymi
        anon_path: Ścieżka do pliku z zanonimizowanymi danymi
        output_path: Ścieżka do pliku wyjściowego JSONL
        sample_ratio: Procent danych do wybrania (0.15 = 15%)
        seed: Ziarno dla generatora losowego
    """
    # Wczytaj dane
    orig_lines = load_lines(orig_path)
    anon_lines = load_lines(anon_path)
    
    # Sprawdź zgodność liczby linii
    if len(orig_lines) != len(anon_lines):
        raise ValueError(f"Liczba linii się nie zgadza: orig={len(orig_lines)}, anon={len(anon_lines)}")
    
    total_lines = len(orig_lines)
    sample_size = int(total_lines * sample_ratio)
    
    print(f"📊 Statystyki:")
    print(f"   - Całkowita liczba linii: {total_lines}")
    print(f"   - Procent do wybrania: {sample_ratio * 100:.1f}%")
    print(f"   - Liczba wybranych linii: {sample_size}")
    
    # Losowe wybieranie indeksów
    random.seed(seed)
    selected_indices = random.sample(range(total_lines), sample_size)
    selected_indices.sort()  # Sortuj dla porządku
    
    # Tworzenie danych treningowych
    train_data = []
    for idx in selected_indices:
        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "Jesteś asystentem do anonimizacji danych osobowych. Twoim zadaniem jest zastąpienie wszystkich danych osobowych odpowiednimi tagami: [name] dla imion, [surname] dla nazwisk, [phone] dla numerów telefonów, [address] dla adresów, [city] dla nazw miast, [date] dla dat, [company] dla nazw firm."
                },
                {
                    "role": "user",
                    "content": f"Zanonimizuj następujący tekst:\n\n{orig_lines[idx]}"
                },
                {
                    "role": "assistant",
                    "content": anon_lines[idx]
                }
            ]
        }
        train_data.append(entry)
    
    # Zapisz do JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Zapisano {len(train_data)} przykładów do: {output_path}")
    
    return train_data

def main():
    # Ścieżki plików
    base_dir = Path(__file__).parent
    orig_path = base_dir / "orig.txt"
    anon_path = base_dir / "anonymized.txt"
    output_path = base_dir / "train_data.jsonl"
    
    # Sprawdź czy pliki istnieją
    if not orig_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {orig_path}")
    if not anon_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {anon_path}")
    
    # Utwórz dane treningowe
    create_train_data(
        orig_path=orig_path,
        anon_path=anon_path,
        output_path=output_path,
        sample_ratio=SAMPLE_RATIO,
        seed=RANDOM_SEED
    )
    
    # Pokaż przykładowe wpisy
    print("\n📝 Przykładowe wpisy:")
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            entry = json.loads(line)
            user_msg = entry["messages"][1]["content"][:100] + "..."
            assistant_msg = entry["messages"][2]["content"][:100] + "..."
            print(f"\n--- Przykład {i+1} ---")
            print(f"USER: {user_msg}")
            print(f"ASSISTANT: {assistant_msg}")

if __name__ == "__main__":
    main()

