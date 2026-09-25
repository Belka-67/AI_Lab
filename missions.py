"""Шесть заданий и условия, проверяемые по сохранённым опытам."""

DEFAULT = dict(count=40, apple=50, spread=25, measurement=0,
               mistakes=0, outliers=0, k=3, depth=5)
MISSIONS = {
    "few": ("Мало данных", "Сравни 12 и 60 карточек, меняя только их количество.",
            dict(DEFAULT, count=12)),
    "labels": ("Перепутанные метки", "Сравни 0% и 30% неверных подписей на одной основе.",
               dict(DEFAULT, count=60)),
    "balance": ("Редкое яблоко", "Сделай 10% яблок и проверь ошибки отдельно по яблокам.",
                dict(DEFAULT, count=100, apple=10)),
    "outliers": ("Странные плоды", "Сравни 0% и 20% выбросов на одной основе.",
                 dict(DEFAULT, count=60)),
    "tree": ("Слишком умное дерево", "Обучи одно и то же дерево с глубиной 1 и 8. Сравни знакомые и новые карточки.",
             dict(DEFAULT, count=60, mistakes=15, outliers=10, depth=1)),
    "repair": ("Почини датасет", "Сначала обучи модель на плохих данных, затем меняй настройки и добейся 85% на новых карточках.",
               dict(DEFAULT, count=20, apple=20, mistakes=15, outliers=20)),
}


def comparable(a, b, changed):
    """Одна случайная основа и те же настройки, кроме исследуемой."""
    return (a["seed"] == b["seed"] and
            all(a["params"][key] == b["params"][key]
                for key in DEFAULT if key != changed))


def mission_progress(mission_id, history):
    runs = [r for r in history if r.get("mission") == mission_id]
    if mission_id == "balance":
        done = any(r["params"]["apple"] <= 10 and r["count"] >= 20 for r in runs)
        return done, "Проведи опыт с 10% яблок и посмотри, сколько яблок распознано."
    if mission_id == "repair":
        baseline = [r for r in runs if r["params"]["count"] == 20 and
                    r["params"]["apple"] == 20 and r["params"]["mistakes"] == 15 and
                    r["params"]["outliers"] == 20]
        done = any(any(v >= 85 for v in r["test_pct"].values()) and
                   r["params"] != bad["params"] and r["seed"] == bad["seed"]
                   for bad in baseline for r in runs)
        return done, "Обучи плохой набор, затем измени данные и достигни 85% на новых карточках."
    ranges = {
        "few": ("count", lambda v: v <= 12, lambda v: v >= 60),
        "labels": ("mistakes", lambda v: v == 0, lambda v: v >= 30),
        "outliers": ("outliers", lambda v: v == 0, lambda v: v >= 20),
        "tree": ("depth", lambda v: v == 1, lambda v: v >= 8),
    }
    changed, first, second = ranges[mission_id]
    done = any(first(a["params"][changed]) and second(b["params"][changed]) and
               comparable(a, b, changed) and
               (changed != "depth" or a["fingerprint"] == b["fingerprint"])
               for a in runs for b in runs)
    return done, "Нужны два обучения при тех же остальных настройках и случайной основе."
