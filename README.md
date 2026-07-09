# Math Problem Checker

ML-система для пошаговой классификации решений математических задач. Модель получает условие, предыдущие шаги и текущий шаг, а затем предсказывает:

- `correct`;
- `incorrect`;
- `incomplete`;
- `suspicious`;
- тип ошибки;
- уверенность.

LLM используется опционально: не как единственный решатель, а как explanation generator поверх уже найденных сигналов. Основной пайплайн:

```text
Parser -> Step Splitter -> Feature Extractor -> SymPy Checker -> ML Classifier -> Report Generator
                                                                      |
                                                                      v
                                                        Optional LLM Explanation
```

## ML-задача

Формулировка: классификация шагов решения математической задачи на корректные, ошибочные, неполные и подозрительные с последующим объяснением типа ошибки.

Вход классификатора:

```text
[PROBLEM] условие задачи
[PREVIOUS] предыдущие шаги
[STEP] текущий шаг
```

Выход:

```json
{
  "label": "incorrect",
  "error_type": "calculation_error",
  "confidence": 0.91
}
```

## Датасет

В проекте есть воспроизводимый синтетический датасет для baseline-экспериментов:

```bash
PYTHONPATH=src python -m math_solution_analyzer.dataset --n 800 --output data/processed/step_classification.csv
```

Схема строки:

```json
{
  "problem": "...",
  "previous_steps": "...",
  "current_step": "...",
  "next_step": "...",
  "step_index": 2,
  "label": "incorrect",
  "error_type": "calculation_error",
  "explanation": "..."
}
```

Покрытые типы ошибок:

- `calculation_error`;
- `sign_error`;
- `wrong_formula`;
- `missing_condition`;
- `invalid_transition`;
- `wrong_substitution`;
- `wrong_final_answer`;
- `probability_without_replacement`.

## Baseline

Реализован честный baseline без нейросетей:

- TF-IDF по тексту `[PROBLEM] [PREVIOUS] [STEP]`;
- числовые признаки шага;
- SymPy-признак арифметической ошибки;
- Logistic Regression.

Обучение:

```bash
PYTHONPATH=src python -m math_solution_analyzer.models.train \
  --dataset data/processed/step_classification.csv \
  --model models/tfidf_logreg.joblib \
  --metrics reports/metrics.json
```

Оценка и confusion matrix:

```bash
PYTHONPATH=src python -m math_solution_analyzer.evaluation \
  --model models/tfidf_logreg.joblib \
  --dataset data/processed/step_classification.csv \
  --metrics reports/evaluation.json \
  --confusion-matrix reports/confusion_matrix.png
```

## Метрики

Метрики сохраняются в `reports/metrics.json` и `reports/evaluation.json`.

| Модель | Accuracy | Macro-F1 | Error step F1 |
| --- | ---: | ---: | ---: |
| Rule-based baseline | planned | planned | planned |
| TF-IDF + LogReg | 1.000 | 1.000 | 1.000 |
| CatBoost | planned | planned | planned |
| RuBERT-tiny | planned | planned | planned |
| Hybrid SymPy + RuBERT | planned | planned | planned |

Текущие значения получены на синтетическом шаблонном split из `data/processed/step_classification.csv`; они нужны как воспроизводимый baseline, а не как финальная оценка качества на реальных ученических решениях.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Сгенерировать датасет, обучить baseline и запустить тесты:

```bash
PYTHONPATH=src python -m math_solution_analyzer.dataset --n 800
PYTHONPATH=src python -m math_solution_analyzer.models.train
PYTHONPATH=src python -m math_solution_analyzer.evaluation
PYTHONPATH=src pytest
```

Запуск Streamlit:

```bash
PYTHONPATH=src streamlit run app.py
```

Для опционального LLM-объяснения укажите:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4.1-mini"
```

## Структура проекта

```text
.
├── app.py
├── data
│   ├── raw
│   └── processed
├── examples
│   ├── inputs
│   └── outputs
├── models
├── notebooks
│   ├── eda.ipynb
│   └── train_baseline.ipynb
├── reports
│   ├── metrics.json
│   └── confusion_matrix.png
├── src/math_solution_analyzer
│   ├── dataset.py
│   ├── evaluation.py
│   ├── features.py
│   ├── checker.py
│   ├── parser.py
│   ├── pipeline.py
│   ├── report_generator.py
│   ├── schema.py
│   ├── step_splitter.py
│   └── models
│       ├── train.py
│       └── predict.py
└── tests
```

## Streamlit-приложение

Приложение принимает условие и решение, разбивает решение на шаги и показывает:

- что верно;
- где возможная ошибка;
- какой шаг пропущен;
- как исправить;
- предсказание ML baseline, если модель обучена и лежит в `models/tfidf_logreg.joblib`.

## Формулировка для резюме

Разработала ML-систему для пошаговой классификации математических решений: собрала и разметила синтетический датасет ошибочных и корректных шагов, реализовала feature extraction с SymPy-признаками, обучила baseline-модель TF-IDF + Logistic Regression, сравнила качество по accuracy/macro-F1/step-level F1 и интегрировала классификатор в Streamlit-приложение.

## Ограничения

Текущий датасет синтетический и нужен для воспроизводимого pet-project baseline. Для production-качества его нужно расширить реальными школьными и вузовскими решениями, добавить ручную валидацию разметки и обучить transformer-based classifier.
