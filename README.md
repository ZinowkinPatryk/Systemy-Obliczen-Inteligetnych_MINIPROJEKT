# Miniprojekt – Systemy obliczeń inteligentnych
## Wariant A: Klasyfikacja obrazów satelitarnych ze zbioru EuroSAT

---

### Opis problemu
Celem projektu jest przygotowanie modelu rozpoznającego rodzaj pokrycia lub wykorzystania terenu na podstawie obrazu satelitarnego. Model otrzymuje kolorowy obraz i przypisuje go do jednej z dziesięciu klas.

Tego typu klasyfikacja może być wykorzystywana m.in. w:
- analizie zagospodarowania przestrzennego,
- monitorowaniu środowiska,
- analizie zmian użytkowania terenu,
- automatycznym przetwarzaniu zobrazowań satelitarnych.

---

### Zbiór danych (EuroSAT)
W projekcie wykorzystywana jest wersja RGB zbioru EuroSAT udostępniona na platformie Hugging Face w repozytorium [`timm/eurosat-rgb`](https://huggingface.co/datasets/timm/eurosat-rgb). Zbiór zawiera **27 000** obrazów RGB o rozdzielczości **64 × 64 piksele**, należących do **10 klas**:
- `AnnualCrop`
- `Forest`
- `HerbaceousVegetation`
- `Highway`
- `Industrial`
- `Pasture`
- `PermanentCrop`
- `Residential`
- `River`
- `SeaLake`

---

### Zakres i etapy realizacji projektu

#### 1. Analiza i przygotowanie danych
1. Pozyskanie zbioru danych z platformy Hugging Face.
2. Przeanalizowanie struktury zbioru danych.
3. Wyświetlenie przykładowych obrazów wraz z etykietami.
4. Sprawdzenie liczebności poszczególnych klas.
5. Podział danych na zbiór treningowy, walidacyjny i testowy.
6. Określenie, jaka część dostępnych danych zostanie wykorzystana w eksperymentach.

#### 2. Przygotowanie modelu
1. Przygotowanie danych w formacie oczekiwanym przez wybrany model (transformacje, normalizacja).
2. Dostosowanie głowicy klasyfikacyjnej do liczby klas (`10`).
3. Określenie parametrów modelu podlegających uczeniu.
4. Dobór podstawowych hiperparametrów uczenia.
5. Przeprowadzenie treningu modelu i monitorowanie wyników na zbiorze walidacyjnym.

#### 3. Transfer learning i fine-tuning
- **Faza I:** dostrajanie wyłącznie samej głowicy klasyfikacyjnej modelu (zamrożony ekstraktor cech).
- **Faza II:** stopniowe odmrażanie kolejnych warstw sieci w celach badawczych oraz analiza wpływu głębokości dostrajania na jakość predykcji.

#### 4. Ocena rozwiązania i eksperymenty
- Analiza przebiegu uczenia i walidacji (krzywe *loss* oraz *accuracy*).
- Ewaluacja z wykorzystaniem *accuracy* oraz dodatkowych metryk (np. *precision*, *recall*, *F1-score*).
- Wyznaczenie i analiza macierzy pomyłek (*confusion matrix*).
- Porównanie wybranych wariantów modelu.
- Prezentacja przykładów poprawnych oraz błędnych predykcji.

---
