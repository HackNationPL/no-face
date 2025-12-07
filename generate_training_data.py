#!/usr/bin/env python3
"""
Generator syntetycznych danych treningowych dla HerBERT NER do anonimizacji tekstów polskich.
Generuje dane w formacie BIO/CoNLL z pełnym pokryciem wszystkich wymaganych kategorii.
"""

import random
import re
import json
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import argparse

# ============================================================================
# SŁOWNIKI DANYCH SYNTETYCZNYCH
# ============================================================================

MALE_NAMES = [
    "Adam", "Adrian", "Aleksander", "Andrzej", "Antoni", "Arkadiusz", "Artur",
    "Bartłomiej", "Bartosz", "Błażej", "Bogdan", "Bogusław", "Borys", "Bruno",
    "Cezary", "Cyprian", "Czesław", "Damian", "Daniel", "Dariusz", "Dawid",
    "Dominik", "Edward", "Emil", "Ernest", "Eugeniusz", "Fabian", "Filip",
    "Franciszek", "Gabriel", "Grzegorz", "Henryk", "Hubert", "Igor", "Ireneusz",
    "Jacek", "Jakub", "Jan", "Janusz", "Jarosław", "Jerzy", "Józef", "Julian",
    "Juliusz", "Kacper", "Kajetan", "Kamil", "Karol", "Kazimierz", "Konrad",
    "Kornel", "Krystian", "Krzysztof", "Leon", "Leszek", "Łukasz", "Maciej",
    "Maksymilian", "Marcin", "Marek", "Mariusz", "Mateusz", "Maurycy", "Michał",
    "Mieczysław", "Mikołaj", "Miłosz", "Natan", "Nikodem", "Norbert", "Olaf",
    "Oskar", "Patryk", "Paweł", "Piotr", "Przemysław", "Radosław", "Rafał",
    "Robert", "Roman", "Ryszard", "Sebastian", "Sławomir", "Stanisław", "Stefan",
    "Sylwester", "Szymon", "Tadeusz", "Tomasz", "Waldemar", "Wiktor", "Witold",
    "Władysław", "Wojciech", "Zbigniew", "Zdzisław", "Zenon"
]

FEMALE_NAMES = [
    "Agata", "Agnieszka", "Aleksandra", "Alicja", "Amelia", "Anastazja", "Anna",
    "Antonina", "Barbara", "Beata", "Bożena", "Celina", "Dagmara", "Danuta",
    "Daria", "Dominika", "Dorota", "Edyta", "Eliza", "Elżbieta", "Emilia",
    "Ewa", "Ewelina", "Gabriela", "Grażyna", "Halina", "Hanna", "Helena",
    "Ilona", "Irena", "Iza", "Izabela", "Jadwiga", "Janina", "Joanna", "Jolanta",
    "Julia", "Julianna", "Justyna", "Kaja", "Kalina", "Kamila", "Karolina",
    "Katarzyna", "Kinga", "Klaudia", "Kornelia", "Krystyna", "Laura", "Lena",
    "Lidia", "Liliana", "Liwia", "Lucyna", "Magdalena", "Maja", "Małgorzata",
    "Marta", "Martyna", "Maria", "Marianna", "Marlena", "Melania", "Milena",
    "Monika", "Natalia", "Natasza", "Nina", "Oliwia", "Patrycja", "Paulina",
    "Renata", "Róża", "Sandra", "Sara", "Sonia", "Stefania", "Sylwia", "Teresa",
    "Urszula", "Wanda", "Weronika", "Wiktoria", "Zofia", "Zuzanna"
]

SURNAMES = [
    "Adamczyk", "Andrzejewski", "Baranowski", "Bąk", "Bielecki", "Błaszczyk",
    "Borkowski", "Brzeziński", "Chmielewski", "Cieślak", "Czarnecki", "Dąbrowski",
    "Dudek", "Duda", "Gajewski", "Głowacki", "Górski", "Grabowski", "Jabłoński",
    "Jakubowski", "Jankowski", "Jasiński", "Kaczmarek", "Kamiński", "Kaźmierczak",
    "Kołodziej", "Kowalczyk", "Kowalski", "Kozłowski", "Król", "Krawczyk",
    "Kubiak", "Kwiatkowski", "Laskowski", "Lewandowski", "Majewski", "Makowski",
    "Malinowski", "Michalak", "Michalski", "Mróz", "Nowak", "Nowakowski",
    "Olszewski", "Ostrowski", "Pawlak", "Pawłowski", "Pietrzak", "Piotrowski",
    "Przybylski", "Rutkowski", "Sadowski", "Sawicki", "Sikorski", "Sobczak",
    "Sokołowski", "Stępień", "Szewczyk", "Szulc", "Szymański", "Szymczak",
    "Tomaszewski", "Urbański", "Walczak", "Wasilewski", "Wieczorek", "Wiśniewski",
    "Witkowski", "Włodarczyk", "Wojciechowski", "Woźniak", "Wójcik", "Wysocki",
    "Zając", "Zakrzewski", "Zalewski", "Zawadzki", "Zieliński", "Ziółkowski"
]

CITIES = [
    "Warszawa", "Kraków", "Łódź", "Wrocław", "Poznań", "Gdańsk", "Szczecin",
    "Bydgoszcz", "Lublin", "Białystok", "Katowice", "Gdynia", "Częstochowa",
    "Radom", "Sosnowiec", "Toruń", "Kielce", "Rzeszów", "Gliwice", "Zabrze",
    "Olsztyn", "Bielsko-Biała", "Bytom", "Zielona Góra", "Rybnik", "Ruda Śląska",
    "Opole", "Tychy", "Gorzów Wielkopolski", "Elbląg", "Płock", "Dąbrowa Górnicza",
    "Wałbrzych", "Włocławek", "Tarnów", "Chorzów", "Koszalin", "Kalisz",
    "Legnica", "Grudziądz", "Jaworzno", "Słupsk", "Jastrzębie-Zdrój", "Nowy Sącz",
    "Jelenia Góra", "Siedlce", "Mysłowice", "Konin", "Piła", "Piotrków Trybunalski",
    "Inowrocław", "Lubin", "Ostrów Wielkopolski", "Suwałki", "Stargard", "Gniezno",
    "Ostrowiec Świętokrzyski", "Siemianowice Śląskie", "Głogów", "Pabianice",
    "Leszno", "Żory", "Pruszków", "Zamość", "Łomża", "Ełk", "Tomaszów Mazowiecki",
    "Chełm", "Mielec", "Kędzierzyn-Koźle", "Przemyśl", "Stalowa Wola", "Tczew",
    "Biała Podlaska", "Będzin", "Zgierz", "Piekary Śląskie", "Racibórz", "Świętochłowice"
]

STREET_PREFIXES = ["ul.", "ulica", "al.", "aleja", "pl.", "plac", "os.", "osiedle"]

STREET_NAMES = [
    "Główna", "Polna", "Leśna", "Słoneczna", "Krótka", "Szkolna", "Ogrodowa",
    "Lipowa", "Brzozowa", "Łąkowa", "Kwiatowa", "Zielona", "Kościelna", "Parkowa",
    "Sportowa", "Kolejowa", "Dworcowa", "Przemysłowa", "Mickiewicza", "Słowackiego",
    "Sienkiewicza", "Kościuszki", "Piłsudskiego", "Konopnickiej", "Prusa",
    "Orzeszkowej", "Reymonta", "Chopina", "Matejki", "Kopernika", "Curie-Skłodowskiej",
    "Jana Pawła II", "Wolności", "Niepodległości", "Armii Krajowej", "Wojska Polskiego",
    "Powstańców", "Solidarności", "3 Maja", "11 Listopada", "Rynek", "Stary Rynek",
    "Nowy Świat", "Marszałkowska", "Krakowska", "Poznańska", "Warszawska", "Gdańska",
    "Wrocławska", "Łódzka", "Lubelska", "Katowicka", "Bydgoska", "Toruńska"
]

COMPANIES = [
    "Orlen", "PKN Orlen", "PGNiG", "KGHM Polska Miedź", "PKO Bank Polski",
    "PZU", "Grupa Lotos", "Tauron", "Energa", "PGE", "Enea", "Cyfrowy Polsat",
    "Orange Polska", "Play", "T-Mobile Polska", "mBank", "ING Bank Śląski",
    "Bank Millennium", "Santander Bank Polska", "BNP Paribas", "Alior Bank",
    "Żabka", "Biedronka", "Lidl Polska", "Kaufland", "Auchan", "Carrefour",
    "MediaMarkt", "RTV Euro AGD", "Empik", "CCC", "LPP", "Reserved", "Cropp",
    "House", "Mohito", "Sinsay", "Allegro", "OLX", "Pracuj.pl", "Interia",
    "Onet", "WP", "Gazeta Wyborcza", "TVP", "TVN", "Polsat", "Radio ZET",
    "CD Projekt", "Techland", "People Can Fly", "11 bit studios", "Comarch",
    "Asseco Poland", "Atos", "Capgemini", "Accenture", "Deloitte", "EY", "KPMG", "PwC"
]

SCHOOLS = [
    "Uniwersytet Warszawski", "Uniwersytet Jagielloński", "Politechnika Warszawska",
    "Politechnika Wrocławska", "Politechnika Gdańska", "Politechnika Poznańska",
    "Politechnika Śląska", "Politechnika Łódzka", "Politechnika Krakowska",
    "Akademia Górniczo-Hutnicza", "Szkoła Główna Handlowa", "Uniwersytet Wrocławski",
    "Uniwersytet Poznański", "Uniwersytet Gdański", "Uniwersytet Łódzki",
    "Uniwersytet Śląski", "Uniwersytet Mikołaja Kopernika", "Uniwersytet Marii Curie-Skłodowskiej",
    "Uniwersytet Ekonomiczny w Krakowie", "Uniwersytet Ekonomiczny w Poznaniu",
    "Uniwersytet Ekonomiczny we Wrocławiu", "Akademia Leona Koźmińskiego",
    "SWPS", "Collegium Civitas", "Akademia Finansów i Biznesu Vistula",
    "I Liceum Ogólnokształcące", "II Liceum Ogólnokształcące", "III LO im. Adama Mickiewicza",
    "Technikum Mechaniczne", "Technikum Informatyczne", "Zespół Szkół Technicznych",
    "Szkoła Podstawowa nr 1", "Szkoła Podstawowa nr 5", "Gimnazjum nr 3"
]

JOB_TITLES = [
    "programista", "analityk", "kierownik", "dyrektor", "specjalista", "konsultant",
    "inżynier", "technik", "asystent", "koordynator", "menedżer", "prezes",
    "wiceprezes", "sekretarka", "recepcjonistka", "księgowa", "księgowy",
    "prawnik", "radca prawny", "adwokat", "lekarz", "pielęgniarka", "pielęgniarz",
    "nauczyciel", "nauczycielka", "profesor", "wykładowca", "badacz", "naukowiec",
    "projektant", "architekt", "grafik", "copywriter", "redaktor", "dziennikarz",
    "fotograf", "operator", "mechanik", "elektryk", "hydraulik", "stolarz",
    "murarz", "malarz", "spawacz", "kierowca", "kurier", "magazynier",
    "sprzedawca", "kasjer", "kelner", "kucharz", "szef kuchni", "barman",
    "fryzjer", "kosmetyczka", "masażysta", "trener", "coach", "psycholog",
    "terapeuta", "fizjoterapeuta", "farmaceuta", "weterynarz", "agronom"
]

HEALTH_CONDITIONS = [
    "depresja", "lęki", "nerwica", "bezsenność", "migrena", "cukrzyca",
    "nadciśnienie", "astma", "alergia", "choroba serca", "arytmia",
    "niedoczynność tarczycy", "nadczynność tarczycy", "anemia", "osteoporoza",
    "artretyzm", "reumatyzm", "ból kręgosłupa", "przepuklina", "żylaki",
    "refluks", "wrzody żołądka", "kamica nerkowa", "prostata", "endometrioza",
    "PCOS", "zapalenie stawów", "padaczka", "stwardnienie rozsiane", "Parkinson",
    "Alzheimer", "demencja", "schizofrenia", "choroba dwubiegunowa", "ADHD",
    "autyzm", "zespół Aspergera", "bulimia", "anoreksja", "otyłość",
    "COVID-19", "grypa", "zapalenie płuc", "bronchit", "zapalenie zatok"
]

RELIGIONS = [
    "katolik", "katoliczka", "protestant", "protestantka", "prawosławny", "prawosławna",
    "muzułmanin", "muzułmanka", "żyd", "żydówka", "buddysta", "buddystka",
    "hinduista", "hinduistka", "ateista", "ateistka", "agnostyk", "agnostyczka",
    "świadek Jehowy", "ewangelik", "ewangeliczka", "zielonoświątkowiec",
    "adwentysta", "adwentystka", "baptystka", "baptist", "metodystka", "metodysta"
]

ETHNICITIES = [
    "Polak", "Polka", "Niemiec", "Niemka", "Ukrainiec", "Ukrainka", "Rosjanin",
    "Rosjanka", "Białorusin", "Białorusinka", "Litwin", "Litwinka", "Czech",
    "Czeszka", "Słowak", "Słowaczka", "Węgier", "Węgierka", "Rumun", "Rumunka",
    "Bułgar", "Bułgarka", "Wietnamczyk", "Wietnamka", "Chińczyk", "Chinka",
    "Rom", "Romka", "Żyd", "Żydówka", "Ślązak", "Ślązaczka", "Kaszub", "Kaszubka"
]

POLITICAL_VIEWS = [
    "konserwatysta", "konserwatystka", "liberał", "liberałka", "socjalista",
    "socjalistka", "socjaldemokrata", "socjaldemokratka", "lewicowiec", "lewicowczyni",
    "prawicowiec", "prawicówka", "centrystka", "centrysta", "narodowiec",
    "nacjonalista", "nacjonalistka", "anarchista", "anarchistka", "libertarianin",
    "libertarianka", "ekolog", "ekolożka", "zieloni", "sympatyk PiS", "sympatyk PO",
    "sympatyk Lewicy", "sympatyk Konfederacji", "sympatyk PSL", "sympatyk Polski 2050"
]

SEXUAL_ORIENTATIONS = [
    "heteroseksualny", "heteroseksualna", "homoseksualny", "homoseksualna",
    "biseksualny", "biseksualna", "panseksualny", "panseksualna", "aseksualny",
    "aseksualna", "gay", "lesbijka", "queer", "nieheteronormatywny", "nieheteronormatywna"
]

EMAIL_DOMAINS = [
    "gmail.com", "wp.pl", "onet.pl", "interia.pl", "o2.pl", "tlen.pl",
    "yahoo.com", "outlook.com", "hotmail.com", "example.com", "example.org",
    "example.net", "firma.pl", "praca.pl", "mail.com"
]

# ============================================================================
# GENERATORY DANYCH
# ============================================================================

def generate_pesel(birth_date: Optional[datetime] = None, sex: Optional[str] = None) -> str:
    """Generuje prawidłowy numer PESEL."""
    if birth_date is None:
        year = random.randint(1950, 2005)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        birth_date = datetime(year, month, day)
    
    year = birth_date.year
    month = birth_date.month
    day = birth_date.day
    
    # Kodowanie miesiąca w zależności od wieku
    if year >= 2000:
        month += 20
    elif year >= 1900:
        pass  # bez zmian
    
    yy = year % 100
    
    # Numer seryjny (3 cyfry) + płeć (1 cyfra)
    serial = random.randint(0, 999)
    if sex == "kobieta" or sex == "K":
        sex_digit = random.choice([0, 2, 4, 6, 8])
    else:
        sex_digit = random.choice([1, 3, 5, 7, 9])
    
    pesel_base = f"{yy:02d}{month:02d}{day:02d}{serial:03d}{sex_digit}"
    
    # Obliczanie cyfry kontrolnej
    weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    checksum = sum(int(pesel_base[i]) * weights[i] for i in range(10))
    control = (10 - (checksum % 10)) % 10
    
    return pesel_base + str(control)


def generate_phone() -> str:
    """Generuje polski numer telefonu."""
    formats = [
        "{}{}{} {}{}{} {}{}{}",
        "+48 {}{}{} {}{}{} {}{}{}",
        "{}{}{}-{}{}{}-{}{}{}",
        "+48{}{}{}{}{}{}{}{}{}"
    ]
    digits = [str(random.randint(0, 9)) for _ in range(9)]
    # Pierwszy digit to 5, 6, 7 lub 8 (polskie komórkowe)
    digits[0] = str(random.choice([5, 6, 7, 8]))
    return random.choice(formats).format(*digits)


def generate_document_number() -> str:
    """Generuje numer dokumentu tożsamości."""
    formats = [
        "{}{}{}{}{}{}",  # stary format
        "{}{}{}{}-{}{}{}{}-{}{}{}{}",  # nowy format
        "{}{}{} {}{}{}{}{}{}",  # inny format
    ]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    fmt = random.choice(formats)
    result = ""
    for char in fmt:
        if char == "{":
            continue
        elif char == "}":
            if random.random() < 0.3:
                result += random.choice(letters)
            else:
                result += str(random.randint(0, 9))
        else:
            result += char
    return result


def generate_bank_account() -> str:
    """Generuje numer konta bankowego w formacie IBAN PL."""
    # PL + 2 cyfry kontrolne + 24 cyfry
    digits = [str(random.randint(0, 9)) for _ in range(24)]
    control = str(random.randint(10, 99))
    
    formats = [
        "PL{} {} {} {} {} {} {}".format(control, ''.join(digits[0:4]), ''.join(digits[4:8]), 
                                        ''.join(digits[8:12]), ''.join(digits[12:16]),
                                        ''.join(digits[16:20]), ''.join(digits[20:24])),
        "{} {} {} {} {} {} {}".format(control, ''.join(digits[0:4]), ''.join(digits[4:8]), 
                                      ''.join(digits[8:12]), ''.join(digits[12:16]),
                                      ''.join(digits[16:20]), ''.join(digits[20:24])),
        "{}{}".format(control, ''.join(digits)),
    ]
    return random.choice(formats)


def generate_credit_card() -> str:
    """Generuje numer karty kredytowej."""
    # Format: 4 grupy po 4 cyfry
    groups = [''.join([str(random.randint(0, 9)) for _ in range(4)]) for _ in range(4)]
    formats = [
        "{} {} {} {}".format(*groups),
        "{}-{}-{}-{}".format(*groups),
        "{}{}{}{}".format(*groups),
    ]
    return random.choice(formats)


def generate_email(name: str = None, surname: str = None) -> str:
    """Generuje adres email."""
    if name is None:
        name = random.choice(MALE_NAMES + FEMALE_NAMES).lower()
    if surname is None:
        surname = random.choice(SURNAMES).lower()
    
    # Normalizacja polskich znaków
    trans = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
    name = name.translate(trans)
    surname = surname.translate(trans)
    
    patterns = [
        f"{name}.{surname}",
        f"{surname}.{name}",
        f"{name}{surname}",
        f"{name[0]}{surname}",
        f"{name}{random.randint(1, 99)}",
        f"{surname}{random.randint(1, 99)}",
        f"{name}_{surname}",
    ]
    
    return f"{random.choice(patterns)}@{random.choice(EMAIL_DOMAINS)}"


def generate_address() -> Tuple[str, str]:
    """Generuje adres (ulica + numer + kod + miasto)."""
    street_prefix = random.choice(STREET_PREFIXES)
    street_name = random.choice(STREET_NAMES)
    number = str(random.randint(1, 200))
    if random.random() < 0.4:
        number += f"/{random.randint(1, 100)}"
    
    postal_code = f"{random.randint(10, 99)}-{random.randint(100, 999)}"
    city = random.choice(CITIES)
    
    formats = [
        f"{street_prefix} {street_name} {number} {postal_code} {city}",
        f"{street_prefix} {street_name} {number}, {postal_code} {city}",
        f"{street_name} {number}, {postal_code} {city}",
    ]
    
    return random.choice(formats), city


def generate_date(start_year: int = 1990, end_year: int = 2024) -> str:
    """Generuje datę w różnych formatach."""
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    
    formats = [
        f"{day:02d}.{month:02d}.{year}",
        f"{day:02d}-{month:02d}-{year}",
        f"{day:02d}/{month:02d}/{year}",
        f"{day} {'stycznia lutego marca kwietnia maja czerwca lipca sierpnia września października listopada grudnia'.split()[month-1]} {year}",
        f"{day}.{month:02d}.{year} r.",
    ]
    return random.choice(formats)


def generate_date_of_birth(min_age: int = 18, max_age: int = 80) -> Tuple[str, int, datetime]:
    """Generuje datę urodzenia i wiek."""
    today = datetime.now()
    age = random.randint(min_age, max_age)
    birth_year = today.year - age
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)
    birth_date = datetime(birth_year, birth_month, birth_day)
    
    formats = [
        f"{birth_day:02d}.{birth_month:02d}.{birth_year}",
        f"{birth_day:02d}-{birth_month:02d}-{birth_year}",
        f"{birth_day} {'stycznia lutego marca kwietnia maja czerwca lipca sierpnia września października listopada grudnia'.split()[birth_month-1]} {birth_year}",
    ]
    return random.choice(formats), age, birth_date


def generate_username() -> str:
    """Generuje nazwę użytkownika."""
    patterns = [
        f"{random.choice(MALE_NAMES + FEMALE_NAMES).lower()}{random.randint(1, 999)}",
        f"{random.choice(SURNAMES).lower()}_{random.randint(1, 99)}",
        f"user_{random.randint(1000, 9999)}",
        f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10)))}",
        f"@{random.choice(MALE_NAMES + FEMALE_NAMES).lower()}{random.randint(1, 99)}",
    ]
    return random.choice(patterns)


def generate_secret() -> str:
    """Generuje hasło lub klucz API."""
    patterns = [
        f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%', k=random.randint(8, 16)))}",
        f"sk-{''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=32))}",
        f"api_key_{''.join(random.choices('0123456789abcdef', k=16))}",
        f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=3))}{''.join(random.choices('0123456789', k=4))}",
    ]
    return random.choice(patterns)


# ============================================================================
# SZABLONY KONTEKSTOWE
# ============================================================================

# Szablony dla różnych kategorii - każdy szablon zawiera placeholdery {tag}
TEMPLATES = {
    # Dane osobowe podstawowe
    "name_surname_intro": [
        "Cześć, jestem {name} {surname}.",
        "Nazywam się {name} {surname}.",
        "Hej, tu {name} {surname}.",
        "Mam na imię {name}, nazwisko {surname}.",
        "Witam, {name} {surname} z tej strony.",
        "Jestem {name}, {surname} z zawodu.",
        "{name} {surname} – to ja.",
    ],
    
    "age_context": [
        "Mam {age} lat.",
        "Mam {age} lata.",
        "Jestem {age}-latkiem.",
        "Jestem {age}-latką.",
        "W wieku {age} lat.",
        "{age} lat i już mam dość.",
        "Skończyłem/am {age} lat.",
        "Jako {age}-latek/latka...",
    ],
    
    "sex_context": [
        "Jestem {sex}.",
        "Płeć: {sex}.",
        "({sex})",
        "jako {sex}",
        "{sex}, {age} lat",
    ],
    
    "date_of_birth_context": [
        "Urodziłem się {date-of-birth}.",
        "Urodziłam się {date-of-birth}.",
        "Data urodzenia: {date-of-birth}.",
        "Urodzony/a {date-of-birth}.",
        "Rocznik {date-of-birth}.",
    ],
    
    # Kontakt
    "phone_context": [
        "Mój telefon to {phone}.",
        "Zadzwoń pod {phone}.",
        "Kontakt: {phone}.",
        "Tel. {phone}.",
        "Numer telefonu: {phone}.",
        "Dzwoń na {phone}.",
        "Jak coś, to {phone}.",
    ],
    
    "email_context": [
        "Mój mail to {email}.",
        "Pisz na {email}.",
        "Email: {email}.",
        "Kontakt mailowy: {email}.",
        "Adres e-mail: {email}.",
    ],
    
    # Adresy vs miasta (kluczowe rozróżnienie!)
    "address_context": [
        "Mieszkam na {address}.",
        "Mieszkam przy {address}.",
        "Mój adres to {address}.",
        "Adres zamieszkania: {address}.",
        "Pod adresem {address}.",
        "Zameldowany/a na {address}.",
    ],
    
    "city_context_general": [
        "Jestem z {city}.",
        "Pochodzę z {city}.",
        "W {city} jest fajnie.",
        "U nas w {city}...",
        "W całym {city} nie ma...",
        "Byłem wczoraj w {city}.",
        "Jadę do {city} na weekend.",
        "W {city} wszyscy się znają.",
    ],
    
    # Identyfikatory
    "pesel_context": [
        "Mój PESEL to {pesel}.",
        "PESEL: {pesel}.",
        "Numer PESEL {pesel}.",
        "Podaję PESEL: {pesel}.",
    ],
    
    "document_context": [
        "Numer dowodu: {document-number}.",
        "Dowód osobisty: {document-number}.",
        "Nr dowodu {document-number}.",
        "Legitymacja nr {document-number}.",
    ],
    
    # Dane zawodowe
    "company_context": [
        "Pracuję w {company}.",
        "Jestem zatrudniony/a w {company}.",
        "Firma {company}.",
        "W {company} od lat.",
    ],
    
    "school_context": [
        "Studiuję na {school-name}.",
        "Skończyłem/am {school-name}.",
        "Uczę się w {school-name}.",
        "Absolwent/ka {school-name}.",
    ],
    
    "job_title_context": [
        "Jestem {job-title}.",
        "Pracuję jako {job-title}.",
        "Stanowisko: {job-title}.",
        "Z zawodu {job-title}.",
    ],
    
    # Finanse
    "bank_account_context": [
        "Numer konta: {bank-account}.",
        "Przelej na {bank-account}.",
        "Rachunek bankowy: {bank-account}.",
        "Konto: {bank-account}.",
    ],
    
    "credit_card_context": [
        "Numer karty: {credit-card-number}.",
        "Karta kredytowa: {credit-card-number}.",
        "Nr karty {credit-card-number}.",
    ],
    
    # Dane wrażliwe
    "health_context": [
        "Mam {health}.",
        "Cierpię na {health}.",
        "Zdiagnozowano u mnie {health}.",
        "Leczy się na {health}.",
        "Problemy z {health}.",
    ],
    
    "religion_context": [
        "Jestem {religion}.",
        "Wyznanie: {religion}.",
        "Z wyznania {religion}.",
        "Jako {religion}...",
    ],
    
    "ethnicity_context": [
        "Jestem {ethnicity}.",
        "Z pochodzenia {ethnicity}.",
        "Narodowość: {ethnicity}.",
    ],
    
    "political_view_context": [
        "Jestem {political-view}.",
        "Poglądy: {political-view}.",
        "Jako {political-view}...",
        "Głosuję jako {political-view}.",
    ],
    
    "sexual_orientation_context": [
        "Jestem {sexual-orientation}.",
        "Orientacja: {sexual-orientation}.",
    ],
    
    # Relacje rodzinne
    "relative_context": [
        "Mój brat {name}.",
        "Moja siostra {name}.",
        "Syn {name} {surname}.",
        "Córka {name}.",
        "Mąż {name} {surname}.",
        "Żona {name}.",
        "Ojciec {name}.",
        "Matka {name}.",
    ],
    
    # Loginy i sekrety
    "username_context": [
        "Login: {username}.",
        "Mój nick to {username}.",
        "Username: {username}.",
        "Konto: {username}.",
    ],
    
    "secret_context": [
        "Hasło: {secret}.",
        "Password: {secret}.",
        "Klucz API: {secret}.",
        "Token: {secret}.",
    ],
    
    # Daty wydarzeń
    "date_event_context": [
        "Przyjęto {date}.",
        "Data wizyty: {date}.",
        "Dnia {date}...",
        "W dniu {date}.",
        "Od {date} do...",
    ],
}

# Pełne szablony rozmów/tekstów
FULL_TEMPLATES = [
    # Nieformalne wiadomości
    "Cześć, jestem {name} {surname}, mam {age} lat ({sex}). Mieszkam w {city} na {address}. Kontakt: {phone}, mail {email}. PESEL {pesel}, dowód {document-number}.",
    
    "Hej, piszę bo potrzebuję pomocy. Nazywam się {name}, {age} lat, {sex}. Mieszkam przy {address}. Mój telefon to {phone}, a mail {email}.",
    
    "Muszę się komuś wygadać. Jestem {name} {surname}, {age}-letni/a {sex} z {city}. Pracuję jako {job-title} w {company}. Kontakt: {phone}.",
    
    # Formularze/zgłoszenia
    "Dane osobowe: {name} {surname}, ur. {date-of-birth}, PESEL {pesel}, dowód {document-number}. Adres: {address}. Tel: {phone}, email: {email}.",
    
    "Zgłaszam incydent. Poszkodowany: {name} {surname}, {age} lat, zam. {address}. Data zdarzenia: {date}. Kontakt: {phone}.",
    
    # Wiadomości o zdrowiu
    "Jestem {name}, {age} lat. Mam problem z {health}. Mieszkam w {city}, tel. {phone}. Mój lekarz mówi, że...",
    
    # Dane wrażliwe - rozbudowane
    "Jestem {name} {surname}, {age} lat, {sex}. Z wyznania {religion}, pochodzę jako {ethnicity}. Poglądy: {political-view}.",
    "Jako {religion} i {ethnicity} mam inne zdanie. Jestem {name}, {age} lat, {sex}. Poglądy polityczne: {political-view}.",
    "{name} {surname} ({age} lat, {sex}) jest {religion}. Jego pochodzenie to {ethnicity}, a poglądy - {political-view}.",
    "Wyznanie: {religion}. Narodowość: {ethnicity}. Poglady: {political-view}. Dane: {name} {surname}, {age} lat.",
    
    # Orientacja seksualna - dodatkowe
    "Jestem {name} {surname}, {age} lat, {sex}. Orientacja: {sexual-orientation}. Kontakt: {phone}.",
    "Jako osoba {sexual-orientation}, {name} {surname} ({age} lat) potrzebuje wsparcia. Email: {email}.",
    "{name} ({age} lat, {sex}) jest {sexual-orientation}. Mieszka w {city}.",
    "Orientacja seksualna: {sexual-orientation}. Imię: {name} {surname}. Wiek: {age}.",
    
    # Dane finansowe - rozbudowane
    "Proszę o przelew na konto {bank-account}. Dane: {name} {surname}, {address}. Numer karty: {credit-card-number}.",
    "Rachunek bankowy: {bank-account}. Karta kredytowa: {credit-card-number}. Właściciel: {name} {surname}.",
    "Dane do przelewu: {bank-account}. Alternatywnie karta: {credit-card-number}. {name} {surname}, {address}.",
    "Nr konta: {bank-account}. Nr karty: {credit-card-number}. Kontakt: {phone}, {email}.",
    
    # Praca/edukacja - rozbudowane
    "{name} {surname} pracuje w {company} jako {job-title}. Ukończył/a {school-name}. Kontakt: {email}.",
    "CV: {name} {surname}, {age} lat. Firma: {company}. Stanowisko: {job-title}. Edukacja: {school-name}.",
    "Absolwent {school-name}, obecnie {job-title} w {company}. Dane: {name} {surname}, tel. {phone}.",
    "Szkoła: {school-name}. Pracodawca: {company}. Funkcja: {job-title}. {name} {surname}.",
    
    # Rodzina - rozbudowane
    "Mój {relative} ({name} {surname}) mieszka w {city}. Jego/jej telefon to {phone}.",
    "Mój {relative} to {name} {surname}, {age} lat. Kontakt: {email}, tel. {phone}.",
    "{relative} {name} zachorował na {health}. Dane: {name} {surname}, {address}.",
    "Zgłaszam za {relative}: {name} {surname}. PESEL {pesel}, dowód {document-number}.",
    
    # Loginy - rozbudowane
    "Moje dane logowania: username {username}, hasło {secret}. Email do odzyskiwania: {email}.",
    "Login: {username}. Password: {secret}. Konto: {email}. Właściciel: {name} {surname}.",
    "User: {username}. Hasło: {secret}. Tel. kontaktowy: {phone}.",
    "Dane dostępowe: {username} / {secret}. Backup email: {email}.",
    
    # Mieszane konteksty
    "Spotkałem wczoraj {name} {surname} w {city}. Pracuje teraz w {company}. Dał mi swój numer: {phone}.",
    
    "W {city} zawsze jest jakiś problem. Ja, {name} ({age} lat, {sex}), mieszkam przy {address} i mam dość tej sytuacji.",
    
    "Jestem {name} {surname}, PESEL {pesel}. Urodziłem/am się {date-of-birth}. Aktualny adres: {address}. Tel: {phone}.",
    
    # Dokumenty - dodatkowe
    "Dowód osobisty: {document-number}. PESEL: {pesel}. Dane: {name} {surname}, ur. {date-of-birth}.",
    "Nr dokumentu: {document-number}. Seria i numer PESEL: {pesel}. Imię i nazwisko: {name} {surname}.",
    "Legitymacja {document-number}. PESEL {pesel}. Właściciel: {name} {surname}, {address}.",
    
    # Daty - dodatkowe
    "Data zdarzenia: {date}. Poszkodowany: {name} {surname}. Data urodzenia: {date-of-birth}. Tel: {phone}.",
    "Przyjęto dnia {date}. Pacjent: {name} {surname}, ur. {date-of-birth}. Diagnoza: {health}.",
    "Wydarzenie z {date}: {name} {surname} ({age} lat) zgłosił {health}. Adres: {address}.",
    
    # Zdrowie - rozbudowane
    "Pacjent {name} {surname}, {age} lat, {sex}. Diagnoza: {health}. PESEL: {pesel}. Tel: {phone}.",
    "Leczę się na {health}. Jestem {name}, {age} lat. Kontakt: {email}, {phone}.",
    "Zdiagnozowano {health} u {name} {surname}. Data: {date}. Adres: {address}.",
    "Stan zdrowia: {health}. Pacjent: {name} {surname}, ur. {date-of-birth}. Dowód: {document-number}.",
]


# ============================================================================
# TOKENIZACJA I FORMAT BIO
# ============================================================================

@dataclass
class Token:
    text: str
    label: str  # B-xxx, I-xxx, lub O


def simple_tokenize(text: str) -> List[str]:
    """Prosta tokenizacja na słowa i interpunkcję."""
    # Rozdziel na słowa, zachowując interpunkcję jako osobne tokeny
    tokens = re.findall(r'\S+', text)
    result = []
    for token in tokens:
        # Oddziel interpunkcję na końcu
        match = re.match(r'^(.+?)([.,;:!?…]+)$', token)
        if match:
            result.append(match.group(1))
            result.append(match.group(2))
        else:
            result.append(token)
    return result


def create_bio_sequence(text: str, entities: Dict[str, List[Tuple[int, int, str]]]) -> List[Token]:
    """
    Tworzy sekwencję tokenów z etykietami BIO.
    
    entities: dict mapping tag_name -> list of (start, end, original_text)
    """
    tokens = simple_tokenize(text)
    labels = ['O'] * len(tokens)
    
    # Znajdź pozycje tokenów w tekście
    token_positions = []
    pos = 0
    for token in tokens:
        idx = text.find(token, pos)
        token_positions.append((idx, idx + len(token)))
        pos = idx + len(token)
    
    # Przypisz etykiety
    for tag_name, spans in entities.items():
        for start, end, _ in spans:
            for i, (tok_start, tok_end) in enumerate(token_positions):
                # Token jest w zakresie encji
                if tok_start >= start and tok_end <= end:
                    if tok_start == start or labels[i] == 'O':
                        labels[i] = f'B-{tag_name}'
                    else:
                        labels[i] = f'I-{tag_name}'
                elif tok_start < end and tok_end > start:
                    # Częściowe pokrycie
                    if labels[i] == 'O':
                        labels[i] = f'B-{tag_name}'
    
    return [Token(text=t, label=l) for t, l in zip(tokens, labels)]


# ============================================================================
# GENERATOR PRZYKŁADÓW
# ============================================================================

class TrainingDataGenerator:
    def __init__(self, target_per_category: int = 1000):
        self.target_per_category = target_per_category
        self.category_counts = {
            'name': 0, 'surname': 0, 'age': 0, 'date-of-birth': 0, 'date': 0,
            'sex': 0, 'religion': 0, 'political-view': 0, 'ethnicity': 0,
            'sexual-orientation': 0, 'health': 0, 'relative': 0,
            'city': 0, 'address': 0, 'email': 0, 'phone': 0,
            'pesel': 0, 'document-number': 0,
            'company': 0, 'school-name': 0, 'job-title': 0,
            'bank-account': 0, 'credit-card-number': 0,
            'username': 0, 'secret': 0,
        }
        self.examples = []
    
    def get_underrepresented_categories(self) -> List[str]:
        """Zwraca kategorie, które potrzebują więcej przykładów."""
        return [cat for cat, count in self.category_counts.items() 
                if count < self.target_per_category]
    
    def generate_person_data(self) -> Dict[str, str]:
        """Generuje spójne dane osoby."""
        sex = random.choice(["mężczyzna", "kobieta"])
        if sex == "mężczyzna":
            name = random.choice(MALE_NAMES)
        else:
            name = random.choice(FEMALE_NAMES)
        surname = random.choice(SURNAMES)
        
        dob_str, age, birth_date = generate_date_of_birth()
        address, city = generate_address()
        
        return {
            'name': name,
            'surname': surname,
            'age': str(age),
            'sex': sex,
            'date-of-birth': dob_str,
            'date': generate_date(),
            'phone': generate_phone(),
            'email': generate_email(name, surname),
            'address': address,
            'city': city,
            'pesel': generate_pesel(birth_date, sex),
            'document-number': generate_document_number(),
            'company': random.choice(COMPANIES),
            'school-name': random.choice(SCHOOLS),
            'job-title': random.choice(JOB_TITLES),
            'bank-account': generate_bank_account(),
            'credit-card-number': generate_credit_card(),
            'health': random.choice(HEALTH_CONDITIONS),
            'religion': random.choice(RELIGIONS),
            'ethnicity': random.choice(ETHNICITIES),
            'political-view': random.choice(POLITICAL_VIEWS),
            'sexual-orientation': random.choice(SEXUAL_ORIENTATIONS),
            'username': generate_username(),
            'secret': generate_secret(),
            'relative': random.choice(['brat', 'siostra', 'syn', 'córka', 'mąż', 'żona', 'ojciec', 'matka']),
        }
    
    def fill_template(self, template: str, data: Dict[str, str]) -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
        """
        Wypełnia szablon danymi i zwraca tekst oraz pozycje encji.
        """
        entities = {}
        result = template
        
        # Znajdź wszystkie placeholdery
        pattern = r'\{([a-z-]+)\}'
        
        # Wypełniaj od końca, żeby nie zaburzyć pozycji
        matches = list(re.finditer(pattern, template))
        
        for match in reversed(matches):
            tag = match.group(1)
            if tag in data:
                value = data[tag]
                start = match.start()
                # Zamień placeholder na wartość
                result = result[:start] + value + result[match.end():]
        
        # Teraz znajdź pozycje encji w finalnym tekście
        for tag, value in data.items():
            if '{' + tag + '}' in template:
                # Znajdź wszystkie wystąpienia tej wartości
                pos = 0
                while True:
                    idx = result.find(value, pos)
                    if idx == -1:
                        break
                    if tag not in entities:
                        entities[tag] = []
                    entities[tag].append((idx, idx + len(value), value))
                    pos = idx + len(value)
        
        return result, entities
    
    def generate_single_example(self, focus_categories: List[str] = None) -> Tuple[str, List[Token]]:
        """Generuje pojedynczy przykład treningowy."""
        data = self.generate_person_data()
        
        # Wybierz szablon - preferuj te z brakującymi kategoriami
        if focus_categories and random.random() < 0.7:
            # Szukaj szablonu z brakującymi kategoriami
            relevant_templates = []
            for template in FULL_TEMPLATES:
                for cat in focus_categories:
                    if '{' + cat + '}' in template:
                        relevant_templates.append(template)
                        break
            if relevant_templates:
                template = random.choice(relevant_templates)
            else:
                template = random.choice(FULL_TEMPLATES)
        else:
            template = random.choice(FULL_TEMPLATES)
        
        text, entities = self.fill_template(template, data)
        bio_tokens = create_bio_sequence(text, entities)
        
        # Aktualizuj liczniki
        for tag in entities:
            if tag in self.category_counts:
                self.category_counts[tag] += len(entities[tag])
        
        return text, bio_tokens
    
    def generate_focused_example(self, category: str) -> Tuple[str, List[Token]]:
        """Generuje przykład skupiony na konkretnej kategorii."""
        data = self.generate_person_data()
        
        # Wybierz odpowiedni szablon dla kategorii
        template_key = f"{category.replace('-', '_')}_context"
        if template_key in TEMPLATES:
            base_template = random.choice(TEMPLATES[template_key])
        else:
            base_template = random.choice(FULL_TEMPLATES)
        
        # Dodaj kontekst
        context_templates = [
            "Cześć, " + base_template,
            "Hej, " + base_template,
            "Witam, " + base_template,
            base_template + " Jakby co, to kontakt: {phone}.",
            base_template + " Pozdrawiam, {name}.",
        ]
        
        template = random.choice(context_templates)
        text, entities = self.fill_template(template, data)
        bio_tokens = create_bio_sequence(text, entities)
        
        # Aktualizuj liczniki
        for tag in entities:
            if tag in self.category_counts:
                self.category_counts[tag] += len(entities[tag])
        
        return text, bio_tokens
    
    def generate_dataset(self, total_examples: int = 5000) -> List[Tuple[str, List[Token]]]:
        """Generuje pełny zbiór danych."""
        examples = []
        
        # Faza 1: Generuj ogólne przykłady
        general_count = total_examples // 2
        for _ in range(general_count):
            underrep = self.get_underrepresented_categories()
            text, tokens = self.generate_single_example(underrep)
            examples.append((text, tokens))
        
        # Faza 2: Doładuj słabo reprezentowane kategorie
        remaining = total_examples - general_count
        underrep = self.get_underrepresented_categories()
        
        while remaining > 0 and underrep:
            for category in underrep:
                if remaining <= 0:
                    break
                text, tokens = self.generate_focused_example(category)
                examples.append((text, tokens))
                remaining -= 1
            underrep = self.get_underrepresented_categories()
        
        # Faza 3: Wypełnij pozostałe
        while remaining > 0:
            text, tokens = self.generate_single_example()
            examples.append((text, tokens))
            remaining -= 1
        
        return examples


# ============================================================================
# EKSPORT DO FORMATU CONLL
# ============================================================================

def export_to_conll(examples: List[Tuple[str, List[Token]]], output_path: str):
    """Eksportuje przykłady do formatu CoNLL."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for text, tokens in examples:
            for token in tokens:
                f.write(f"{token.text}\t{token.label}\n")
            f.write("\n")  # Pusta linia między przykładami


def export_to_json(examples: List[Tuple[str, List[Token]]], output_path: str):
    """Eksportuje przykłady do formatu JSON (dla porównania)."""
    data = []
    for text, tokens in examples:
        data.append({
            'text': text,
            'tokens': [{'text': t.text, 'label': t.label} for t in tokens]
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generator danych treningowych dla HerBERT NER')
    parser.add_argument('--output-dir', type=str, default='output', 
                        help='Katalog wyjściowy')
    parser.add_argument('--total-examples', type=int, default=10000,
                        help='Liczba przykładów do wygenerowania')
    parser.add_argument('--target-per-category', type=int, default=1000,
                        help='Docelowa liczba wystąpień na kategorię')
    parser.add_argument('--format', type=str, choices=['conll', 'json', 'both'], 
                        default='both', help='Format wyjściowy')
    args = parser.parse_args()
    
    # Utwórz katalog wyjściowy
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generowanie {args.total_examples} przykładów treningowych...")
    print(f"Docelowa liczba na kategorię: {args.target_per_category}")
    
    generator = TrainingDataGenerator(target_per_category=args.target_per_category)
    examples = generator.generate_dataset(total_examples=args.total_examples)
    
    print(f"\nWygenerowano {len(examples)} przykładów.")
    print("\nStatystyki kategorii:")
    for cat, count in sorted(generator.category_counts.items(), key=lambda x: -x[1]):
        status = "✓" if count >= args.target_per_category else "⚠"
        print(f"  {status} {cat}: {count}")
    
    # Eksport
    if args.format in ['conll', 'both']:
        conll_path = output_dir / 'train.conll'
        export_to_conll(examples, str(conll_path))
        print(f"\nZapisano format CoNLL: {conll_path}")
    
    if args.format in ['json', 'both']:
        json_path = output_dir / 'train.json'
        export_to_json(examples, str(json_path))
        print(f"Zapisano format JSON: {json_path}")
    
    # Podział na train/dev/test
    random.shuffle(examples)
    train_size = int(len(examples) * 0.8)
    dev_size = int(len(examples) * 0.1)
    
    train_examples = examples[:train_size]
    dev_examples = examples[train_size:train_size + dev_size]
    test_examples = examples[train_size + dev_size:]
    
    if args.format in ['conll', 'both']:
        export_to_conll(train_examples, str(output_dir / 'train_split.conll'))
        export_to_conll(dev_examples, str(output_dir / 'dev.conll'))
        export_to_conll(test_examples, str(output_dir / 'test.conll'))
        print(f"\nPodziały: train={len(train_examples)}, dev={len(dev_examples)}, test={len(test_examples)}")
    
    print("\n✓ Generowanie zakończone!")


if __name__ == '__main__':
    main()

