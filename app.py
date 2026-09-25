"""Интерактивная лаборатория машинного обучения для 5–8 классов."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

from lab_core import (CLASSES, FEATURES, MODELS, TEST_SIZE, class_scores,
                      data_fingerprint, generate_data, logistic_boundary,
                      logistic_explanation, neighbor_details, train, tree_steps,
                      validate)
from missions import DEFAULT, MISSIONS, mission_progress

COLORS = {"Яблоко": "#11887d", "Груша": "#e39332"}
TEST_APPLE_PERCENT = 20  # На тесте 24 яблока и 96 груш: общее качество может обманывать.
DATA_KEYS = ("count", "apple", "spread", "measurement", "mistakes", "outliers")
WIDGET_KEYS = {key: f"setting_{key}" for key in DEFAULT}


def current_params():
    return {key: st.session_state[WIDGET_KEYS[key]] for key in DEFAULT}


def update_data(reshuffle=False):
    if reshuffle:
        st.session_state.seed += 1
    p = current_params()
    st.session_state.data = generate_data(*(p[key] for key in DATA_KEYS),
                                          st.session_state.seed)
    st.session_state.config = {key: p[key] for key in DATA_KEYS}
    st.session_state.results = None
    st.session_state.editor_revision += 1


def start_mission(mission_id):
    _, _, preset = MISSIONS[mission_id]
    for key, value in preset.items():
        st.session_state[WIDGET_KEYS[key]] = value
    st.session_state.active_mission = mission_id
    update_data()


def leave_mission():
    st.session_state.active_mission = None


def use_baseline(run_number):
    run = next(r for r in st.session_state.history if r["number"] == run_number)
    st.session_state.baseline_id = run["number"]
    st.session_state.seed = run["seed"]
    for key, value in run["params"].items():
        st.session_state[WIDGET_KEYS[key]] = value
    st.session_state.data = run["data"].copy()
    st.session_state.config = {key: run["params"][key] for key in DATA_KEYS}
    st.session_state.results = None
    st.session_state.editor_revision += 1


def record_experiment(data):
    p = {**st.session_state.config,
         "k": st.session_state.setting_k,
         "depth": st.session_state.setting_depth}
    results = train(data, st.session_state.test, p["k"], p["depth"])
    st.session_state.results = results
    st.session_state.trained_data = data.copy().reset_index(drop=True)
    st.session_state.trained_params = p.copy()
    st.session_state.history.append({
        "number": len(st.session_state.history) + 1,
        "data": data.copy().reset_index(drop=True), "fingerprint": data_fingerprint(data),
        "seed": st.session_state.seed, "mission": st.session_state.active_mission,
        "params": p, "count": len(data),
        "apples": int((data["Верный класс"] == "Яблоко").sum()),
        "flipped": int((data["Метка"] != data["Верный класс"]).sum()),
        "unusual": int(data["Выброс"].sum()),
        "train_pct": {name: round(100 * r["train_correct"] / len(data), 1)
                      for name, r in results.items()},
        "test_pct": {name: round(100 * r["test_correct"] / TEST_SIZE, 1)
                     for name, r in results.items()},
    })


def draw_chart(data, results, name, test=None, point=None):
    model = results[name]["model"]
    xgrid, ygrid = np.meshgrid(np.linspace(0, 10, 141), np.linspace(0, 10, 141))
    grid = pd.DataFrame({"Размер": xgrid.ravel(), "Округлость": ygrid.ravel()})
    regions = (model.predict(grid) == "Груша").astype(int).reshape(xgrid.shape)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.pcolormesh(xgrid, ygrid, regions, shading="auto",
                  cmap=ListedColormap(["#ddf4ef", "#fff0dc"]), alpha=0.9)
    for cls in CLASSES:
        rows = data[data["Метка"] == cls]
        ax.scatter(rows["Размер"], rows["Округлость"], s=64,
                   c=COLORS[cls], edgecolors="white", linewidths=0.8,
                   label=f"{cls}: обучение", zorder=3)
    wrong = results[name]["train_predictions"] != data["Метка"].to_numpy()
    if wrong.any():
        ax.scatter(data.loc[wrong, "Размер"], data.loc[wrong, "Округлость"],
                   s=150, facecolors="none", edgecolors="#d73046", linewidths=2,
                   label="Ошибка на обучении", zorder=4)
    flipped = data["Метка"] != data["Верный класс"]
    if flipped.any():
        ax.scatter(data.loc[flipped, "Размер"], data.loc[flipped, "Округлость"],
                   marker="x", s=95, c="#5d3f9d", linewidths=2,
                   label="Неверная метка", zorder=5)
    if data["Выброс"].any():
        unusual = data["Выброс"].astype(bool)
        ax.scatter(data.loc[unusual, "Размер"], data.loc[unusual, "Округлость"],
                   marker="D", s=190, facecolors="none", edgecolors="#166ac2",
                   linewidths=2, label="Выброс", zorder=4)
    if test is not None:
        errors = results[name]["test_predictions"] != test["Верный класс"].to_numpy()
        shown = test[errors].head(20)
        if not shown.empty:
            ax.scatter(shown["Размер"], shown["Округлость"], marker="s", s=80,
                       facecolors="none", edgecolors="#242f47", linewidths=1.7,
                       label="Ошибка на новых данных (до 20)", zorder=4)
    if name == MODELS[2]:
        weights, offset, _ = logistic_boundary(model)
        if abs(weights[1]) > 1e-10:
            xs = np.linspace(0, 10, 100)
            ys = -(offset + weights[0] * xs) / weights[1]
            ax.plot(xs, ys, color="#364b9a", linewidth=2.6,
                    label="Граница модели", zorder=5)
        elif abs(weights[0]) > 1e-10:
            ax.axvline(-offset / weights[0], color="#364b9a", linewidth=2.6,
                       label="Граница модели", zorder=5)
    if point is not None:
        if name == MODELS[0]:
            _, indices = neighbor_details(model, data, point)
            neighbors = data.iloc[indices]
            ax.scatter(neighbors["Размер"], neighbors["Округлость"],
                       s=240, facecolors="none", edgecolors="#284ab0", linewidths=2.2,
                       label="Ближайшие соседи", zorder=5)
        ax.scatter(point["Размер"], point["Округлость"], marker="*", s=420,
                   c="#27324a", edgecolors="white", linewidths=1.2,
                   label="Новый объект", zorder=6)
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="Размер (0–10)",
           ylabel="Округлость (0–10)")
    ax.grid(alpha=0.15)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    return fig


def show_chart(data, results, name, test=None, point=None):
    fig = draw_chart(data, results, name, test, point)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def show_score(name, result, data):
    score = result[name]
    a, b = st.columns(2)
    a.metric("Знакомые карточки", f"{score['train_correct']} из {len(data)}",
             help="По учебным меткам, включая ошибочные подписи.")
    b.metric("Новые карточки", f"{score['test_correct']} из {TEST_SIZE}",
             help="По настоящим классам фиксированного тестового набора.")
    apples = score["by_class"]["Яблоко"]
    pears = score["by_class"]["Груша"]
    st.write(f"🍎 Яблок распознано: **{apples[0]} из {apples[1]}**  ·  "
             f"🍐 Груш распознано: **{pears[0]} из {pears[1]}**")
    matrix = score["matrix"]
    school_matrix = pd.DataFrame({
        "Модель сказала 🍎": [f"✅ {matrix[0, 0]}", f"❌ {matrix[1, 0]}"],
        "Модель сказала 🍐": [f"❌ {matrix[0, 1]}", f"✅ {matrix[1, 1]}"],
    }, index=["На самом деле 🍎", "На самом деле 🍐"])
    st.table(school_matrix)
    if matrix[0, 1] > matrix[1, 0]:
        st.caption("Чаще яблоки ошибочно называют грушами.")
    elif matrix[1, 0] > matrix[0, 1]:
        st.caption("Чаще груши ошибочно называют яблоками.")
    else:
        st.caption("Оба вида путаницы встречаются одинаково часто.")
    with st.expander("Термины для 7–8 класса"):
        st.write("**Accuracy:** доля правильных ответов из всех 120. "
                 "**Recall класса:** сколько объектов этого класса распознано верно. "
                 "**Confusion matrix:** таблица настоящих классов и ответов модели. "
                 "**Test set:** новые карточки, не использованные при обучении.")


def show_explanation(name, results, data, point):
    model = results[name]["model"]
    predicted = model.predict(point)[0]
    st.markdown(f"**{name} говорит: {predicted}. Почему?**")
    if name == MODELS[0]:
        neighbors, _ = neighbor_details(model, data, point)
        labels = neighbors["Метка"].value_counts()
        st.write(f"Модель ищет {len(neighbors)} ближайших карточек "
                 "(расстояние учитывает масштаб обоих признаков). "
                 f"Среди них яблок: {labels.get('Яблоко', 0)}, груш: {labels.get('Груша', 0)}. "
                 "Ответ даёт большинство учебных меток.")
        st.dataframe(neighbors[["Карточка №", *FEATURES, "Метка", "Расстояние"]],
                     hide_index=True, width="stretch")
    elif name == MODELS[1]:
        st.write("Дерево задаёт вопросы по очереди:")
        for i, step in enumerate(tree_steps(model, point), 1):
            st.write(f"{i}. {step}")
        st.caption(f"После этих вопросов дерево выбрало: {predicted}.")
    else:
        st.write(logistic_explanation(model, point))
        st.caption("Синяя линия на графике — граница. По разные стороны от неё модель даёт разные ответы.")


def history_table(history):
    rows = []
    for run in history:
        p = run["params"]
        rows.append({"Опыт": run["number"], "Карточек": run["count"],
                     "Яблок, %": round(100 * run["apples"] / run["count"]),
                     "Ошибки меток, %": p["mistakes"], "Неверных меток": run["flipped"],
                     "Выбросы, %": p["outliers"], "Выбросов": run["unusual"],
                     "Разброс, %": p["spread"], "Измерения, %": p["measurement"],
                     "k": p["k"], "Глубина": p["depth"],
                     **{f"{name} на новых, %": run["test_pct"][name] for name in MODELS}})
    return pd.DataFrame(rows)


st.set_page_config(page_title="Лаборатория данных", page_icon="🍎", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1350px;}
  h1, h2, h3 {color: #21334a;}
  [data-testid="stMetric"] {background: #f4f8fa; padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

if "seed" not in st.session_state:
    st.session_state.seed = 42
    for key, value in DEFAULT.items():
        st.session_state[WIDGET_KEYS[key]] = value
    st.session_state.data = generate_data(*(DEFAULT[key] for key in DATA_KEYS), 42)
    st.session_state.test = generate_data(TEST_SIZE, TEST_APPLE_PERCENT, 25, 0, 0, 0, 2026)
    st.session_state.config = {key: DEFAULT[key] for key in DATA_KEYS}
    st.session_state.results = train(st.session_state.data, st.session_state.test)
    st.session_state.trained_data = st.session_state.data.copy()
    st.session_state.trained_params = DEFAULT.copy()
    st.session_state.history = []
    st.session_state.editor_revision = 0
    st.session_state.active_mission = None
    st.session_state.baseline_id = None
# Обновление открытой вкладки со старой версией приложения.
if "trained_params" not in st.session_state:
    st.session_state.test = generate_data(TEST_SIZE, TEST_APPLE_PERCENT, 25, 0, 0, 0, 2026)
    st.session_state.history = []
    st.session_state.active_mission = None
    st.session_state.baseline_id = None
    for key, value in DEFAULT.items():
        st.session_state[WIDGET_KEYS[key]] = value
    st.session_state.config = {key: DEFAULT[key] for key in DATA_KEYS}
    if "Выброс" not in st.session_state.data:
        st.session_state.data["Выброс"] = False
    st.session_state.results = None
    st.session_state.trained_params = DEFAULT.copy()
    st.session_state.editor_revision += 1

st.title("🍎 Лаборатория данных")
st.caption("Создай карточки → обучи модели → проверь на новых → сравни опыты")

with st.sidebar:
    st.header("1. Данные")
    st.slider("Количество карточек", 6, 120, step=2, key="setting_count")
    st.slider("Доля яблок, %", 10, 90, step=5, key="setting_apple")
    st.slider("Разброс признаков, %", 0, 100, step=5, key="setting_spread",
              help="Насколько сами плоды одного класса различаются.")
    st.slider("Ошибка измерений, %", 0, 100, step=5, key="setting_measurement",
              help="Сила случайной ошибки при измерении размера и округлости.")
    st.slider("Перепутанные метки, %", 0, 40, step=5, key="setting_mistakes")
    st.slider("Выбросы, %", 0, 40, step=5, key="setting_outliers")
    if st.button("Создать выборку", width="stretch"):
        update_data()
        st.rerun()
    if st.button("🎲 Другой случайный набор", width="stretch"):
        update_data(reshuffle=True)
        st.rerun()
    st.caption("Меняй одну настройку за раз: обычная кнопка сохраняет случайную основу.")
    st.divider()
    st.header("2. Модели")
    st.select_slider("Соседей у kNN", options=[1, 3, 5, 7, 9], key="setting_k")
    st.slider("Глубина дерева", 1, 8, key="setting_depth")
    chosen = st.selectbox("Показать на графике", MODELS)
    show_test_errors = st.checkbox("Показать ошибки на новых данных")
    if st.session_state.active_mission:
        st.info(f"🎯 Миссия: {MISSIONS[st.session_state.active_mission][0]}")
        st.button("Вернуться к свободной лаборатории", on_click=leave_mission,
                  width="stretch")

lab_tab, missions_tab, history_tab = st.tabs(["🧪 Лаборатория", "🎯 Миссии", "📈 История опытов"])

with lab_tab:
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.subheader("Карточки для обучения")
        st.caption("Модель видит размер, округлость и метку. Верный класс и выброс нужны тебе для проверки.")
        with st.expander("Редактировать карточки вручную"):
            edited = st.data_editor(
                st.session_state.data, key=f"editor_{st.session_state.editor_revision}",
                hide_index=True, width="stretch", height=320,
                disabled=["Верный класс", "Выброс"],
                column_config={
                    "Размер": st.column_config.NumberColumn("Размер", min_value=0.0, max_value=10.0, step=0.1),
                    "Округлость": st.column_config.NumberColumn("Округлость", min_value=0.0, max_value=10.0, step=0.1),
                    "Метка": st.column_config.SelectboxColumn("Метка", options=CLASSES, required=True),
                    "Верный класс": st.column_config.TextColumn("Верный класс", help="Скрыт от модели"),
                    "Выброс": st.column_config.CheckboxColumn("Выброс"),
                },
            )
            st.session_state.data = edited.copy().reset_index(drop=True)
            add_tab, delete_tab = st.tabs(["Добавить", "Удалить"])
            with add_tab:
                with st.form("add_card", clear_on_submit=True):
                    c1, c2 = st.columns(2)
                    size = c1.number_input("Размер", 0.0, 10.0, 6.0, 0.1)
                    roundness = c2.number_input("Округлость", 0.0, 10.0, 7.0, 0.1)
                    true_class = c1.selectbox("Верный класс", CLASSES)
                    label = c2.selectbox("Метка для обучения", CLASSES)
                    add = st.form_submit_button("Добавить")
                if add:
                    row = pd.DataFrame([{"Размер": size, "Округлость": roundness,
                                         "Метка": label, "Верный класс": true_class,
                                         "Выброс": False}])
                    st.session_state.data = pd.concat([edited, row], ignore_index=True)
                    st.session_state.results = None
                    st.session_state.editor_revision += 1
                    st.rerun()
            with delete_tab:
                selected = st.multiselect("Номера строк (сверху от 1)",
                                          options=list(range(1, len(edited) + 1)))
                if st.button("Удалить выбранные") and selected:
                    st.session_state.data = edited.drop(index=[n - 1 for n in selected]).reset_index(drop=True)
                    st.session_state.results = None
                    st.session_state.editor_revision += 1
                    st.rerun()
        st.caption(f"Сейчас карточек: {len(edited)}; неверных меток: "
                   f"{(edited['Метка'] != edited['Верный класс']).sum()}; "
                   f"выбросов: {edited['Выброс'].sum()}.")
        if st.button("🚀 Обучить три модели", type="primary", width="stretch"):
            error = validate(edited)
            if error:
                st.error(error)
            else:
                record_experiment(edited)
                st.rerun()
    with right:
        st.subheader("Как модель делит пространство")
        if st.session_state.results is None:
            st.info("Создай данные и нажми «Обучить три модели».")
        else:
            data = st.session_state.trained_data
            if (not edited.equals(data) or
                    any(st.session_state[WIDGET_KEYS[key]] != st.session_state.trained_params[key]
                        for key in ("k", "depth"))):
                st.warning("Данные или настройки модели изменились. Результаты показаны для последнего обучения.")
            results = st.session_state.results
            show_chart(data, results, chosen,
                       st.session_state.test if show_test_errors else None)
            st.caption("Фон — ответы модели; красный круг — ошибка на обучении; "
                       "фиолетовый крест — неверная метка; синий ромб — выброс.")
            show_score(chosen, results, data)
            st.caption("Новые карточки одинаковы во всех опытах: 24 яблока и 96 груш. "
                       "Если модель всегда говорит «груша», она получит 80%, но не узнает ни одного яблока.")
    if st.session_state.results is not None:
        st.divider()
        st.subheader("Сравни три модели")
        comparison = pd.DataFrame([{
            "Модель": name,
            "Знакомые": f"{r['train_correct']}/{len(data)}",
            "Новые": f"{r['test_correct']}/{TEST_SIZE}",
            "Яблоки": f"{r['by_class']['Яблоко'][0]}/{r['by_class']['Яблоко'][1]}",
            "Груши": f"{r['by_class']['Груша'][0]}/{r['by_class']['Груша'][1]}",
        } for name, r in st.session_state.results.items()])
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.subheader("Испытай модель на своём объекте")
        p1, p2 = st.columns(2)
        new_size = p1.slider("Размер нового объекта", 0.0, 10.0, 6.0, 0.1)
        new_roundness = p2.slider("Округлость нового объекта", 0.0, 10.0, 7.0, 0.1)
        point = pd.DataFrame({"Размер": [new_size], "Округлость": [new_roundness]})
        st.markdown("  \n".join(f"**{name}:** {r['model'].predict(point)[0]}"
                                  for name, r in st.session_state.results.items()))
        st.caption("Для придуманного объекта настоящий класс неизвестен: это только прогноз.")
        show_explanation(chosen, st.session_state.results, data, point)
        with st.expander("Показать объект и объяснение на графике"):
            show_chart(data, st.session_state.results, chosen, point=point)
    with st.expander("Инструкция к уроку"):
        st.markdown("""
1. Создай 12 карточек и обучи модели. Посмотри на результат на новых карточках.
2. Создай 60 карточек с теми же остальными настройками, обучи и сравни результат.
3. По очереди меняй неверные метки, ошибку измерений и выбросы. После каждого изменения создавай данные и обучай модели.
4. Сравни яблоки и груши по отдельности. Может ли общее число правильных ответов обмануть?
5. Выбери новый объект и посмотри, как соседи, дерево и прямая граница помогают моделям принять решение.
6. Открой историю, сравни все три модели и сделай вывод. Во вкладке «Миссии» тебя ждут задания.

**Подумай:** почему модель иногда хорошо отвечает на знакомых карточках, но ошибается на новых?
""")

with missions_tab:
    st.subheader("Миссии исследователя")
    st.caption("Начни задание, меняй настройки слева и обучай модели во вкладке «Лаборатория». "
               "Прогресс сохраняется до закрытия приложения.")
    completed = sum(mission_progress(mid, st.session_state.history)[0] for mid in MISSIONS)
    st.progress(completed / len(MISSIONS), text=f"Выполнено: {completed} из {len(MISSIONS)}")
    for i, (mid, (title, description, _)) in enumerate(MISSIONS.items(), 1):
        done, next_step = mission_progress(mid, st.session_state.history)
        with st.container(border=True):
            st.markdown(f"### {'✅' if done else '🎯'} {i}. {title}")
            st.write(description)
            if done:
                st.success("Задание выполнено! Сравни результаты и объясни их своими словами.")
            else:
                st.caption(next_step)
            st.button("Начать / повторить", key=f"mission_{mid}",
                      on_click=start_mission, args=(mid,))
    st.caption("В миссии «Почини датасет» сначала обучи предложенный плохой набор, "
               "потом меняй настройки и добейся ≥85% хотя бы у одной модели. "
               "Результат может зависеть от случайного набора.")

with history_tab:
    st.subheader("История экспериментов")
    if not st.session_state.history:
        st.info("Обучи модели, чтобы здесь появился первый опыт.")
    else:
        history = st.session_state.history
        table = history_table(history)
        st.dataframe(table, hide_index=True, width="stretch")
        chart = pd.DataFrame({"Опыт": [r["number"] for r in history],
                              **{name: [r["test_pct"][name] for r in history]
                                 for name in MODELS}}).set_index("Опыт")
        st.write("**Качество на новых карточках по опытам, %**")
        st.line_chart(chart, y=MODELS, width="stretch")
        runs_by_number = {r["number"]: r for r in history}
        chosen_number = st.selectbox(
            "Выбери опыт для сравнения",
            options=list(runs_by_number),
            format_func=lambda n: (
                f"Опыт №{n}: {runs_by_number[n]['count']} карточек"
            ),
            key="history_choice_number",
        )
        st.button(
            "📌 Использовать этот опыт как базовый",
            on_click=use_baseline,
            args=(chosen_number,),
        )
        if st.session_state.baseline_id is not None:
            baseline = next((r for r in history if r["number"] == st.session_state.baseline_id), None)
            if baseline:
                latest = history[-1]
                st.caption(f"Базовый опыт №{baseline['number']}. Его данные и настройки загружены "
                           "для следующего сравнения; измени один параметр.")
                if latest["number"] != baseline["number"]:
                    st.write("**Изменение результата последнего опыта относительно базового:**")
                    st.dataframe(pd.DataFrame([{
                        "Модель": name,
                        "Базовый, %": baseline["test_pct"][name],
                        "Последний, %": latest["test_pct"][name],
                        "Разница, п. п.": round(latest["test_pct"][name] - baseline["test_pct"][name], 1),
                    } for name in MODELS]), hide_index=True, width="stretch")
        st.caption("Все модели проверяются на одних и тех же 120 новых карточках. "
                   "Один опыт ещё не доказывает, что изменение всегда улучшает качество.")
