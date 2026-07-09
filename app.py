from __future__ import annotations

import json

import streamlit as st

from math_solution_analyzer import analyze_solution


st.set_page_config(page_title="Анализатор решений", page_icon="∑", layout="wide")

st.title("Анализатор решений математических задач")

with st.sidebar:
    use_llm = st.toggle("Проверять через OpenAI API", value=True)
    show_json = st.toggle("Показать JSON", value=True)

problem = st.text_area(
    "Условие задачи",
    height=160,
    placeholder="Например: Найдите производную функции f(x)=x^2 sin x.",
)

solution = st.text_area(
    "Решение",
    height=260,
    placeholder="1. Используем правило произведения...\n2. ...",
)

if st.button("Проверить решение", type="primary"):
    try:
        report = analyze_solution(problem, solution, use_llm=use_llm)
    except Exception as exc:
        st.error(f"Не удалось выполнить анализ: {exc}")
    else:
        st.subheader("Итог")
        st.write(report.summary)

        tab_correct, tab_errors, tab_missing, tab_fix, tab_steps = st.tabs(
            ["Что верно", "Где возможная ошибка", "Какой шаг пропущен", "Как исправить", "Шаги"]
        )

        with tab_correct:
            for item in report.what_is_correct:
                st.markdown(f"- {item}")

        with tab_errors:
            for item in report.where_possible_error:
                st.markdown(f"- {item}")

        with tab_missing:
            for item in report.missing_steps:
                st.markdown(f"- {item}")

        with tab_fix:
            for item in report.how_to_fix:
                st.markdown(f"- {item}")

        with tab_steps:
            for step in report.steps:
                with st.expander(f"Шаг {step.step.index}: {step.status.value}"):
                    st.write(step.step.text)
                    if step.possible_errors:
                        st.markdown("**Замечания**")
                        for issue in step.possible_errors:
                            st.markdown(f"- `{issue.severity.value}` {issue.title}: {issue.explanation}")
                    if step.missing_steps:
                        st.markdown("**Пропущено**")
                        for missing in step.missing_steps:
                            st.markdown(f"- {missing}")
                    if step.how_to_fix:
                        st.markdown("**Исправление**")
                        for fix in step.how_to_fix:
                            st.markdown(f"- {fix}")

        if show_json:
            st.subheader("Структурированный JSON")
            st.code(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), language="json")
