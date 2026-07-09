# Система анализа решений математических задач

MVP-инструмент проверяет текстовое или LaTeX-решение математической задачи по шагам. Он разбивает решение на логические фрагменты, проверяет каждый шаг отдельно, собирает структурированный JSON через Pydantic и формирует итоговый отчёт:

- что верно;
- где возможная ошибка;
- какой шаг пропущен;
- как исправить.

Проект специально не отправляет всё решение в LLM одним запросом. Пайплайн устроен так:

1. `parser` нормализует условие и решение.
2. `step_splitter` выделяет отдельные логические шаги.
3. `checker` проверяет каждый шаг отдельно через LangChain/OpenAI или локальные эвристики.
4. `schema` задаёт структурированный формат ответа через Pydantic.
5. `report_generator` агрегирует результаты в итоговый отчёт.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Для LLM-проверки укажите `OPENAI_API_KEY` и, при необходимости, `OPENAI_MODEL`:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4.1-mini"
```

Запуск Streamlit:

```bash
PYTHONPATH=src streamlit run app.py
```

Локальный тест без API-ключа:

```bash
PYTHONPATH=src pytest
```

## Пример использования из Python

```python
from math_solution_analyzer import analyze_solution

report = analyze_solution(
    problem="Вычислите 2 + 2.",
    solution="1. Складываем числа.\n2. 2 + 2 = 5.\n3. Ответ: 5.",
    use_llm=False,
)

print(report.model_dump_json(indent=2))
```

## Структура проекта

```text
.
├── app.py
├── examples
│   ├── inputs
│   └── outputs
├── src/math_solution_analyzer
│   ├── checker.py
│   ├── parser.py
│   ├── pipeline.py
│   ├── report_generator.py
│   ├── schema.py
│   └── step_splitter.py
└── tests
```

## Демонстрационные задачи

В папке `examples/inputs` лежат 5 примеров:

- линейная алгебра: решение системы уравнений;
- вероятность: условная вероятность;
- математический анализ: производная произведения;
- пределы: типичный пропуск обоснования;
- арифметика: очевидная вычислительная ошибка.

Готовые JSON-ответы находятся в `examples/outputs`.

## Ограничения MVP

Локальная эвристика нужна для демонстрации и тестов без API-ключа. Она ловит очевидные вычислительные ошибки и пропущенные объяснения, но не заменяет полноценную LLM-проверку. При наличии `OPENAI_API_KEY` используется LangChain `ChatOpenAI` со структурированным выводом `StepAnalysis`.
