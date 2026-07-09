# Math Problem Checker

ML-система для пошаговой классификации решений математических задач. Модель получает условие, предыдущие шаги и текущий шаг, а затем предсказывает:

- `correct`;
- `incorrect`;
- `incomplete`;
- `suspicious`;
- тип ошибки;
- уверенность.

LLM используется опционально: не как единственный решатель и не как классификатор, а как explanation generator поверх уже найденных rule/ML-сигналов. Основной пайплайн:

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

В проекте есть воспроизводимый синтетический датасет для baseline-экспериментов. Он содержит несколько доменов: арифметика, линейные и квадратные уравнения, системы, производные, интегралы, пределы, матрицы, комбинаторика и вероятность.

```bash
PYTHONPATH=src python -m math_solution_analyzer.dataset --n 2000 --output data/processed/step_classification.csv
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
- числовые, структурные и доменные признаки шага;
- SymPy-признаки арифметики, эквивалентности выражений, линейных уравнений, производных и вероятности без возвращения;
- Logistic Regression для `label`;
- отдельная Logistic Regression для `error_type`.

Оценка использует `GroupShuffleSplit` по `problem_id`, поэтому шаги одной задачи не могут одновременно попасть в train и test. Это убирает главный источник завышения качества в случайном split.

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
| Rule-based baseline | 0.6265 | 0.5936 | 0.6545 |
| TF-IDF + numeric + SymPy LogReg | 0.9486 | 0.9638 | 0.9583 |
| CatBoost | planned | planned | planned |
| RuBERT-tiny | planned | planned | planned |
| Hybrid SymPy + RuBERT | planned | planned | planned |

Для отдельной задачи классификации `error_type` текущий baseline даёт `accuracy=0.9506`, `macro-F1=0.9633` на group split. Подробности и ограничения описаны в `reports/error_analysis.md`.

Ранний случайный split давал 1.000/1.000, но это было плохим сигналом: модель видела почти одинаковые шаблоны одной задачи в train и test. Поэтому текущая оценка считается только через group split по `problem_id`.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Сгенерировать датасет, обучить baseline и запустить тесты:

```bash
PYTHONPATH=src python -m math_solution_analyzer.dataset --n 2000
PYTHONPATH=src python -m math_solution_analyzer.models.train
PYTHONPATH=src python -m math_solution_analyzer.evaluation
PYTHONPATH=src pytest
```

Запуск Streamlit:

```bash
PYTHONPATH=src streamlit run app.py
```

CLI-проверка решения:

```bash
PYTHONPATH=src python -m math_solution_analyzer.cli check \
  --problem examples/cli/problem.txt \
  --solution examples/cli/solution.txt \
  --model models/tfidf_logreg.joblib \
  --output reports/demo_cli_output.json \
  --no-llm
```

Для опционального LLM-объяснения укажите:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4.1-mini"
```

После этого можно убрать `--no-llm`: классификация всё равно выполняется rule/ML слоями, а OpenAI API используется только для короткого человеческого объяснения найденных сигналов.
Если OpenAI недоступен, отчёт всё равно строится, а в `metadata.llm_warning` появляется предупреждение. Для rule-only режима используйте `--no-ml`.

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
│   ├── evaluation.json
│   ├── error_analysis.md
│   └── confusion_matrix.png
├── src/math_solution_analyzer
│   ├── dataset.py
│   ├── evaluation.py
│   ├── features.py
│   ├── cli.py
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
