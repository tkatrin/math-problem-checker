from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ERROR_TYPES = [
    "none",
    "calculation_error",
    "sign_error",
    "wrong_formula",
    "missing_condition",
    "invalid_transition",
    "wrong_substitution",
    "wrong_final_answer",
    "probability_without_replacement",
]

PHRASES = {
    "get": ["Получаем", "Отсюда следует", "Значит", "Тогда имеем", "Можно записать"],
    "therefore": ["Следовательно", "Итак", "Отсюда", "Значит", "В итоге"],
    "substitute": ["Подставим найденное значение", "Подставляя это значение", "Используем найденное значение"],
    "obvious": ["Очевидно", "Сразу видно", "Без дополнительных вычислений", "Ясно, что"],
}


def build_synthetic_dataset(n: int = 800, seed: int = 42) -> list[dict[str, str | int]]:
    random.seed(seed)
    rows: list[dict[str, str | int]] = []
    templates = [
        _linear_system_case,
        _probability_case,
        _calculus_case,
        _limit_case,
        _arithmetic_case,
        _linear_equation_case,
        _quadratic_case,
        _derivative_chain_case,
        _integral_case,
        _matrix_case,
        _combinatorics_case,
        _conditional_probability_case,
    ]
    while len(rows) < n:
        rows.extend(random.choice(templates)())
    return rows[:n]


def save_dataset(rows: list[dict[str, str | int]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem_id",
        "domain",
        "problem",
        "previous_steps",
        "current_step",
        "next_step",
        "step_index",
        "label",
        "error_type",
        "explanation",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    *,
    problem_id: str,
    domain: str,
    problem: str,
    previous_steps: list[str],
    current_step: str,
    next_step: str,
    step_index: int,
    label: str,
    error_type: str,
    explanation: str,
) -> dict[str, str | int]:
    return {
        "problem_id": problem_id,
        "domain": domain,
        "problem": problem,
        "previous_steps": " ||| ".join(previous_steps),
        "current_step": current_step,
        "next_step": next_step,
        "step_index": step_index,
        "label": label,
        "error_type": error_type,
        "explanation": explanation,
    }


def _arithmetic_case() -> list[dict[str, str | int]]:
    a = random.randint(2, 30)
    b = random.randint(2, 30)
    correct = a + b
    wrong = correct + random.choice([-2, -1, 1, 2, 3])
    problem = _noise(f"Вычислите {a} + {b}.")
    pid = f"arith-{a}-{b}-{random.randint(1000, 9999)}"
    intro = f"{_phrase('get')} сумму чисел."
    return [
        _row(problem_id=pid, domain="arithmetic", problem=problem, previous_steps=[], current_step=intro, next_step=f"{a} + {b} = {correct}.", step_index=1, label="correct", error_type="none", explanation="План вычисления корректен."),
        _row(problem_id=pid, domain="arithmetic", problem=problem, previous_steps=[intro], current_step=f"{_phrase('obvious')} получаем ответ.", next_step=f"Ответ: {correct}.", step_index=2, label="suspicious", error_type="invalid_transition", explanation="Шаг слишком общий и не содержит вычисления."),
        _row(problem_id=pid, domain="arithmetic", problem=problem, previous_steps=[intro], current_step=f"{a} + {b} = {wrong}.", next_step=f"Ответ: {wrong}.", step_index=2, label="incorrect", error_type="calculation_error", explanation=f"Правильная сумма равна {correct}."),
        _row(problem_id=pid, domain="arithmetic", problem=problem, previous_steps=[f"{a} + {b} = {correct}."], current_step=f"Ответ: {wrong}.", next_step="", step_index=3, label="incorrect", error_type="wrong_final_answer", explanation="Финальный ответ не совпадает с вычислением."),
    ]


def _linear_system_case() -> list[dict[str, str | int]]:
    x = random.randint(1, 8)
    y = random.randint(1, 8)
    s = x + y
    d = x - y
    wrong_y = y + random.choice([-2, -1, 1, 2])
    problem = _noise(f"Решите систему: x + y = {s}, x - y = {d}.")
    pid = f"lin-{x}-{y}-{random.randint(1000, 9999)}"
    step1 = f"{_phrase('get')} после сложения уравнений: 2x = {2 * x}."
    step2 = f"{_phrase('get')} x = {x}."
    return [
        _row(problem_id=pid, domain="linear_algebra", problem=problem, previous_steps=[], current_step=step1, next_step=step2, step_index=1, label="correct", error_type="none", explanation="Сложение уравнений выполнено корректно."),
        _row(problem_id=pid, domain="linear_algebra", problem=problem, previous_steps=[step1], current_step=step2, next_step=f"Тогда y = {s} - {x} = {y}.", step_index=2, label="correct", error_type="none", explanation="Значение x найдено верно."),
        _row(problem_id=pid, domain="linear_algebra", problem=problem, previous_steps=[step1, step2], current_step="Дальше сразу находим y.", next_step=f"Ответ: x = {x}, y = {y}.", step_index=3, label="suspicious", error_type="missing_condition", explanation="Нет явной подстановки в одно из уравнений."),
        _row(problem_id=pid, domain="linear_algebra", problem=problem, previous_steps=[step1, step2], current_step=f"Тогда y = {s} - {x} = {wrong_y}.", next_step=f"Ответ: x = {x}, y = {wrong_y}.", step_index=3, label="incorrect", error_type="wrong_substitution", explanation="Подстановка в первое уравнение дала неверное значение y."),
    ]


def _probability_case() -> list[dict[str, str | int]]:
    red = random.randint(3, 8)
    blue = random.randint(2, 7)
    total = red + blue
    problem = _noise(f"В урне {red} красных и {blue} синих шаров. Два шара вытаскивают без возвращения. Найдите вероятность двух красных шаров.")
    pid = f"prob-{red}-{blue}-{random.randint(1000, 9999)}"
    first = f"Вероятность первого красного шара равна {red}/{total}."
    return [
        _row(problem_id=pid, domain="probability", problem=problem, previous_steps=[], current_step=first, next_step=f"Вероятность второго красного шара равна {red - 1}/{total - 1}.", step_index=1, label="correct", error_type="none", explanation="Вероятность первого выбора записана верно."),
        _row(problem_id=pid, domain="probability", problem=problem, previous_steps=[first], current_step=f"Вероятность второго красного шара равна {red}/{total}.", next_step="Перемножаем вероятности.", step_index=2, label="incorrect", error_type="probability_without_replacement", explanation="После первого выбора состав урны меняется."),
        _row(problem_id=pid, domain="probability", problem=problem, previous_steps=[first], current_step="Вероятность второго события аналогична первому.", next_step="Перемножаем вероятности.", step_index=2, label="suspicious", error_type="probability_without_replacement", explanation="Фраза скрывает условную вероятность при выборе без возвращения."),
        _row(problem_id=pid, domain="probability", problem=problem, previous_steps=[first], current_step="Следовательно, вероятности нужно перемножить.", next_step="", step_index=3, label="incomplete", error_type="missing_condition", explanation="Не показана условная вероятность второго события."),
    ]


def _calculus_case() -> list[dict[str, str | int]]:
    problem = _noise("Найдите производную функции f(x)=x^2 sin x.")
    pid = f"calc-{random.randint(1000, 9999)}"
    return [
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=[], current_step="Используем правило произведения.", next_step="(x^2)'=2x, (sin x)'=cos x.", step_index=1, label="correct", error_type="none", explanation="Правило выбрано верно."),
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=["Используем правило произведения."], current_step="Производная sin x равна -cos x.", next_step="", step_index=2, label="incorrect", error_type="wrong_formula", explanation="Производная sin x равна cos x."),
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=["Используем правило произведения."], current_step="После дифференцирования получаем итоговую формулу.", next_step="", step_index=2, label="suspicious", error_type="invalid_transition", explanation="Не показаны производные множителей и подстановка в правило."),
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=["Используем правило произведения."], current_step="Следовательно, f'(x)=2x sin x.", next_step="", step_index=3, label="incomplete", error_type="invalid_transition", explanation="Пропущен второй член x^2 cos x."),
    ]


def _limit_case() -> list[dict[str, str | int]]:
    problem = _noise("Найдите предел lim_{x->0} sin(x)/x.")
    pid = f"lim-{random.randint(1000, 9999)}"
    return [
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=[], current_step="При x -> 0 числитель и знаменатель стремятся к 0.", next_step="Используем замечательный предел.", step_index=1, label="correct", error_type="none", explanation="Неопределённость замечена верно."),
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=["При x -> 0 числитель и знаменатель стремятся к 0."], current_step="Дальше применяем известный факт.", next_step="Предел равен 1.", step_index=2, label="suspicious", error_type="invalid_transition", explanation="Не назван конкретный замечательный предел."),
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=["При x -> 0 числитель и знаменатель стремятся к 0."], current_step="Следовательно, предел равен 1.", next_step="", step_index=2, label="incomplete", error_type="invalid_transition", explanation="Нужно сослаться на замечательный предел или доказать переход."),
    ]


def _linear_equation_case() -> list[dict[str, str | int]]:
    a = random.randint(2, 9)
    x = random.randint(-5, 8)
    b = random.randint(-8, 8)
    c = a * x + b
    wrong = x + random.choice([-2, -1, 1, 2])
    problem = _noise(f"Решите линейное уравнение {a}x + {b} = {c}.")
    pid = f"lineq-{a}-{b}-{c}-{random.randint(1000,9999)}"
    step1 = f"{_phrase('get')} {a}x = {c - b}."
    step2 = f"x = {(c - b)}/{a} = {x}."
    return [
        _row(problem_id=pid, domain="algebra", problem=problem, previous_steps=[], current_step=step1, next_step=step2, step_index=1, label="correct", error_type="none", explanation="Свободный член перенесён корректно."),
        _row(problem_id=pid, domain="algebra", problem=problem, previous_steps=[step1], current_step=f"x = {(c - b)}/{a} = {wrong}.", next_step=f"Ответ: {wrong}.", step_index=2, label="incorrect", error_type="calculation_error", explanation="Ошибка при делении коэффициента."),
        _row(problem_id=pid, domain="algebra", problem=problem, previous_steps=[step1, step2], current_step=f"Ответ: {x}.", next_step="", step_index=3, label="correct", error_type="none", explanation="Ответ совпадает с найденным значением."),
    ]


def _quadratic_case() -> list[dict[str, str | int]]:
    r1 = random.randint(1, 6)
    r2 = random.randint(1, 6)
    b = -(r1 + r2)
    c = r1 * r2
    wrong_d = b * b - 4 * c + random.choice([-4, -1, 1, 4])
    problem = _noise(f"Решите квадратное уравнение x^2 {b:+d}x {c:+d} = 0.")
    pid = f"quad-{r1}-{r2}-{random.randint(1000,9999)}"
    d = b * b - 4 * c
    return [
        _row(problem_id=pid, domain="algebra", problem=problem, previous_steps=[], current_step=f"Дискриминант D = ({b})^2 - 4*1*{c} = {d}.", next_step="", step_index=1, label="correct", error_type="none", explanation="Дискриминант вычислен верно."),
        _row(problem_id=pid, domain="algebra", problem=problem, previous_steps=[], current_step=f"Дискриминант D = ({b})^2 - 4*1*{c} = {wrong_d}.", next_step="", step_index=1, label="incorrect", error_type="calculation_error", explanation="Дискриминант посчитан неверно."),
        _row(problem_id=pid, domain="algebra", problem=problem, previous_steps=[f"D = {d}."], current_step=f"{_phrase('therefore')}, x = {r1} или x = {r2}.", next_step="", step_index=2, label="correct", error_type="none", explanation="Корни соответствуют уравнению."),
    ]


def _derivative_chain_case() -> list[dict[str, str | int]]:
    k = random.randint(2, 6)
    problem = _noise(f"Найдите производную функции f(x)=sin({k}x).")
    pid = f"chain-{k}-{random.randint(1000,9999)}"
    return [
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=[], current_step="Используем правило цепочки.", next_step=f"f'(x)={k}cos({k}x).", step_index=1, label="correct", error_type="none", explanation="Правильно выбрано правило цепочки."),
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=["Используем правило цепочки."], current_step=f"f'(x)=cos({k}x).", next_step="", step_index=2, label="incomplete", error_type="wrong_formula", explanation="Пропущен множитель внутренней производной."),
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=["Используем правило цепочки."], current_step=f"f'(x)=-{k}cos({k}x).", next_step="", step_index=2, label="incorrect", error_type="sign_error", explanation="У производной sin знак положительный."),
    ]


def _integral_case() -> list[dict[str, str | int]]:
    n = random.randint(1, 5)
    problem = _noise(f"Найдите неопределённый интеграл ∫ x^{n} dx.")
    pid = f"int-{n}-{random.randint(1000,9999)}"
    correct_power = n + 1
    return [
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=[], current_step="Используем степенную формулу интегрирования.", next_step="", step_index=1, label="correct", error_type="none", explanation="Выбрана подходящая формула."),
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=["Используем степенную формулу интегрирования."], current_step=f"∫ x^{n} dx = x^{correct_power}/{correct_power} + C.", next_step="", step_index=2, label="correct", error_type="none", explanation="Интеграл найден верно."),
        _row(problem_id=pid, domain="calculus", problem=problem, previous_steps=["Используем степенную формулу интегрирования."], current_step=f"∫ x^{n} dx = {n}x^{n-1} + C.", next_step="", step_index=2, label="incorrect", error_type="wrong_formula", explanation="Использована формула производной вместо интеграла."),
    ]


def _matrix_case() -> list[dict[str, str | int]]:
    a, b, c, d = [random.randint(-3, 5) for _ in range(4)]
    det = a * d - b * c
    wrong = a * d + b * c
    problem = _noise(f"Найдите determinant матрицы [[{a}, {b}], [{c}, {d}]].")
    pid = f"mat-{a}-{b}-{c}-{d}-{random.randint(1000,9999)}"
    return [
        _row(problem_id=pid, domain="linear_algebra", problem=problem, previous_steps=[], current_step=f"det A = {a}*{d} - {b}*{c}.", next_step=f"det A = {det}.", step_index=1, label="correct", error_type="none", explanation="Формула определителя 2x2 записана верно."),
        _row(problem_id=pid, domain="linear_algebra", problem=problem, previous_steps=[], current_step=f"det A = {a}*{d} + {b}*{c} = {wrong}.", next_step="", step_index=1, label="incorrect", error_type="sign_error", explanation="В формуле определителя нужен минус."),
        _row(problem_id=pid, domain="linear_algebra", problem=problem, previous_steps=[f"det A = {a}*{d} - {b}*{c}."], current_step=f"Ответ: det A = {det}.", next_step="", step_index=2, label="correct", error_type="none", explanation="Ответ согласован с вычислением."),
    ]


def _combinatorics_case() -> list[dict[str, str | int]]:
    n = random.randint(5, 10)
    k = random.randint(2, min(4, n - 1))
    problem = _noise(f"Сколькими способами можно выбрать {k} объектов из {n}?")
    pid = f"comb-{n}-{k}-{random.randint(1000,9999)}"
    return [
        _row(problem_id=pid, domain="combinatorics", problem=problem, previous_steps=[], current_step=f"Используем сочетания C({n},{k}).", next_step="", step_index=1, label="correct", error_type="none", explanation="Порядок выбора не важен, поэтому нужны сочетания."),
        _row(problem_id=pid, domain="combinatorics", problem=problem, previous_steps=[], current_step=f"Используем размещения A({n},{k}).", next_step="", step_index=1, label="incorrect", error_type="wrong_formula", explanation="Выбранная формула учитывает порядок, которого нет в условии."),
        _row(problem_id=pid, domain="combinatorics", problem=problem, previous_steps=[f"Используем C({n},{k})."], current_step=f"{_phrase('obvious')} получаем число способов.", next_step="", step_index=2, label="suspicious", error_type="invalid_transition", explanation="Не показано вычисление сочетаний."),
    ]


def _conditional_probability_case() -> list[dict[str, str | int]]:
    total = random.randint(80, 200)
    a = random.randint(20, 60)
    b_given_a = random.choice([0.2, 0.25, 0.5])
    both = int(a * b_given_a)
    problem = _noise(f"Из {total} студентов {a} изучают Python. Среди изучающих Python {both} знают NumPy. Найдите P(NumPy | Python).")
    pid = f"condprob-{total}-{a}-{both}-{random.randint(1000,9999)}"
    return [
        _row(problem_id=pid, domain="probability", problem=problem, previous_steps=[], current_step=f"P(NumPy | Python) = {both}/{a}.", next_step="", step_index=1, label="correct", error_type="none", explanation="Условная вероятность берётся относительно группы Python."),
        _row(problem_id=pid, domain="probability", problem=problem, previous_steps=[], current_step=f"P(NumPy | Python) = {both}/{total}.", next_step="", step_index=1, label="incorrect", error_type="missing_condition", explanation="Знаменатель должен быть числом изучающих Python, а не общим числом студентов."),
        _row(problem_id=pid, domain="probability", problem=problem, previous_steps=[], current_step="Делим подходящее количество на общее.", next_step="", step_index=1, label="suspicious", error_type="missing_condition", explanation="Неясно, какое множество используется как условие."),
    ]


def _phrase(kind: str) -> str:
    return random.choice(PHRASES[kind])


def _noise(text: str) -> str:
    if random.random() < 0.2:
        text = text.replace("x", "\\(x\\)")
    if random.random() < 0.2:
        text = text.replace(" = ", "=")
    if random.random() < 0.15:
        text = " ".join(text.split())
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/processed/step_classification.csv"))
    args = parser.parse_args()
    rows = build_synthetic_dataset(n=args.n, seed=args.seed)
    save_dataset(rows, args.output)
    print(f"saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
