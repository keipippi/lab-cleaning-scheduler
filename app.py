import streamlit as st
import pandas as pd
from collections import deque, defaultdict
import random

st.set_page_config(page_title="Lab Cleaning Scheduler", layout="wide")

st.title("🧹 Weekly Cleaning Scheduler")

st.markdown("""
- Vacuum / Mop / Garbage / Student Room は人数が多め（Vacuum, Mop=3、Garbage, Student Room=2）
- Water alcohol, Chip Tube, Autoclave Waste, Autoclave Drain, Drying Racks, Consumable Goods は各1人
- Liquid Waste は指定した週だけ+1人
- Kawakami / Kawano は Student Room から除外
- Unavailable に入れた人は完全に除外
- Monday-unavailable は Chip Tube / Autoclave Waste / Student Room / Consumable Goods を優先
- 人数が足りないときは Vacuum→Mop→Drying Racks の順で自動で枠を減らす（Vacuum/Mop/Garbage/Student Room は最低2人）
""")

def normalize_multiline(text: str) -> str:
    # ',', '/', 改行 で分割して、1行1人に整形
    import re
    parts = re.split(r'[,/\n]+', text)
    names = [p.strip() for p in parts if p.strip()]
    return "\n".join(names)

with st.sidebar:
    st.header("⚙️ Settings")

    default_members = [
        "Sumi","Sarah","Zhang","Felix","Nishimiya","Nakano","Yu","Komiyama",
        "Onishi","Ardra","Takahashi","Urakawa","Yong Sen","Nia","Kawakami","Kawano"
    ]
    default_group_A = ["Zhang","Yu","Ardra","Yong Sen","Nia"]
    default_group_B = ["Sumi","Sarah","Felix","Nishimiya","Nakano","Komiyama","Onishi","Takahashi","Urakawa"]

    members_text_raw = st.text_area("Members (one per line)", value="\n".join(default_members), height=260)
    group_A_text_raw = st.text_area("Student Room Group A", value="\n".join(default_group_A), height=120)
    group_B_text_raw = st.text_area("Student Room Group B", value="\n".join(default_group_B), height=140)

    st.markdown("---")
    st.subheader("🚫 Unavailable members (全週参加不可)")
    unavail_raw = st.text_area("参加できないメンバー（1行1人）", value="", height=100)

    st.subheader("🗓 Monday-unavailable members")
    monday_raw = st.text_area("月曜に来れないメンバー（1行1人）", value="", height=100)

    # 正規化（UIの見た目はそのままだけど、裏側ではちゃんと1行1人にして扱う）
    members_text = normalize_multiline(members_text_raw)
    group_A_text = normalize_multiline(group_A_text_raw)
    group_B_text = normalize_multiline(group_B_text_raw)
    unavail_text = normalize_multiline(unavail_raw)
    monday_text = normalize_multiline(monday_raw)

    members = [m for m in members_text.splitlines() if m.strip()]
    group_A = {m for m in group_A_text.splitlines() if m.strip()}
    group_B = {m for m in group_B_text.splitlines() if m.strip()}
    unavailable = {m for m in unavail_text.splitlines() if m.strip()}
    monday_unavail = {m for m in monday_text.splitlines() if m.strip()}

    excluded_SR = {"Kawakami","Kawano"}

    weeks = st.number_input("Number of weeks", min_value=4, max_value=52, value=8)
    random_seed = st.number_input("Random seed", min_value=0, max_value=10000, value=42)
    liquid_weeks_input = st.text_input("Liquid Waste weeks (例: 1,3,5)")
    liquid_weeks = set()
    if liquid_weeks_input.strip():
        for t in liquid_weeks_input.split(","):
            t = t.strip()
            if not t:
                continue
            try:
                w = int(t)
                if 1 <= w <= weeks:
                    liquid_weeks.add(w)
            except ValueError:
                pass

    generate = st.button("🔁 Generate / Regenerate")

# --- validation ---
if not group_A.isdisjoint(group_B):
    st.error("Group A と Group B は重複しないようにしてください。")
    st.stop()

group_A_eff = (group_A - unavailable) - excluded_SR
group_B_eff = (group_B - unavailable) - excluded_SR

if len(group_A_eff) < 1 or len(group_B_eff) < 1:
    st.error("Student Room のA/Bグループの有効メンバーが足りません。")
    st.stop()

BASE_COUNTS = {
    "Vacuum":3,
    "Mop":3,
    "Garbage":2,
    "Student Room":2,
    "Chip Tube":1,
    "Autoclave Waste":1,
    "Autoclave Drain":1,
    "Drying Racks":1,
    "Water alcohol":1,
    "Consumable Goods":1,
}
ADJUST_MIN = {
    "Vacuum":2,
    "Mop":2,
    "Garbage":2,
    "Student Room":2,
    "Chip Tube":1,
    "Autoclave Waste":1,
    "Autoclave Drain":1,
    "Drying Racks":0,
    "Water alcohol":1,
    "Consumable Goods":1,
}

priority_tasks_for_monday_unavail = {"Chip Tube","Autoclave Waste","Student Room","Consumable Goods"}

def adapt_counts(available_count: int, has_liquid: bool):
    counts = dict(BASE_COUNTS)
    warnings = []
    total_slots = sum(counts.values()) + (1 if has_liquid else 0)
    deficit = total_slots - available_count
    if deficit <= 0:
        return counts, warnings

    def reduce(name: str):
        nonlocal deficit
        if counts[name] > ADJUST_MIN[name] and deficit > 0:
            counts[name] -= 1
            deficit -= 1
            return True
        return False

    while deficit > 0:
        changed = False
        if reduce("Vacuum"):
            changed = True
        if deficit > 0 and reduce("Mop"):
            changed = True

        if deficit <= 0:
            break

        if counts["Drying Racks"] > 0:
            counts["Drying Racks"] = 0
            deficit -= 1
            warnings.append("人数不足のため Drying Racks を 0 にしました。")
            changed = True

        if not changed:
            warnings.append("人数不足のため、いくつかの枠は未割当になります。")
            break

    return counts, warnings

def pick_candidate(dq, used_this_week, last_task, task_name, *, required_group=None, preferred=None, blacklist=None):
    blacklist = blacklist or set()

    def try_pick(order, relax_same=False):
        tmp = deque(order)
        tried = len(tmp)
        while tried > 0:
            cand = tmp[0]
            tmp.rotate(-1)
            tried -= 1
            if cand in used_this_week:
                continue
            if cand in blacklist:
                continue
            if required_group and cand not in required_group:
                continue
            if (not relax_same) and last_task.get(cand) == task_name:
                continue
            return cand, tmp
        return None, dq

    if preferred:
        pref_list = [m for m in dq if m in preferred]
        rest_list = [m for m in dq if m not in preferred]
        cand, newdq = try_pick(pref_list + rest_list, relax_same=False)
        if cand:
            return cand, newdq

    cand, newdq = try_pick(list(dq), relax_same=False)
    if cand:
        return cand, newdq

    if preferred:
        pref_list = [m for m in dq if m in preferred]
        rest_list = [m for m in dq if m not in preferred]
        cand, newdq = try_pick(pref_list + rest_list, relax_same=True)
        if cand:
            return cand, newdq

    cand, newdq = try_pick(list(dq), relax_same=True)
    return cand, newdq

def assign_one_week(start_deque, last_task, has_liquid, counts, group_A_eff, group_B_eff, available_set):
    used = set()
    dq = deque(start_deque)
    assigned = defaultdict(list)
    unfilled = []

    slots = []
    if counts["Student Room"] >= 2:
        slots.append("Student Room (A)")
        slots.append("Student Room (B)")
    elif counts["Student Room"] == 1:
        slots.append("Student Room (A)")

    for name in ["Chip Tube","Autoclave Waste","Autoclave Drain","Vacuum","Mop","Garbage","Drying Racks","Water alcohol","Consumable Goods"]:
        for _ in range(counts[name]):
            slots.append(name)

    if has_liquid:
        slots = ["Liquid Waste"] + slots

    base_blacklist = set(unavailable)

    for slot in slots:
        if slot == "Student Room (A)":
            pref = monday_unavail & group_A_eff
            cand, dq = pick_candidate(
                dq, used, last_task, "Student Room",
                required_group=group_A_eff,
                preferred=pref if pref else None,
                blacklist=base_blacklist | excluded_SR
            )
            if not cand:
                unfilled.append(slot)
                continue
            assigned["Student Room"].append(cand)
            used.add(cand)

        elif slot == "Student Room (B)":
            pref = monday_unavail & group_B_eff
            cand, dq = pick_candidate(
                dq, used, last_task, "Student Room",
                required_group=group_B_eff,
                preferred=pref if pref else None,
                blacklist=base_blacklist | excluded_SR
            )
            if not cand:
                unfilled.append(slot)
                continue
            assigned["Student Room"].append(cand)
            used.add(cand)

        else:
            pref = monday_unavail if slot in priority_tasks_for_monday_unavail else None
            cand, dq = pick_candidate(
                dq, used, last_task, slot,
                preferred=pref,
                blacklist=base_blacklist
            )
            if not cand:
                unfilled.append(slot)
                continue
            assigned[slot].append(cand)
            used.add(cand)

    new_start = deque(start_deque)
    new_start.rotate(-5 if len(new_start) >= 6 else -1)
    return assigned, new_start, unfilled

def build_schedule():
    available_members = [m for m in members if m not in unavailable]
    if not available_members:
        st.error("参加可能なメンバーが0人です。")
        st.stop()

    random.seed(random_seed)
    dq = deque(sorted(available_members))
    random.shuffle(dq)
    last_task = {m: None for m in available_members}

    all_weeks = []
    info = []

    for w in range(1, weeks + 1):
        has_liquid = (w in liquid_weeks)
        counts, warns = adapt_counts(len(available_members), has_liquid)
        week_assign, dq, unfilled = assign_one_week(
            dq, last_task, has_liquid, counts,
            group_A_eff, group_B_eff, set(available_members)
        )
        for t, ppl in week_assign.items():
            for p in ppl:
                last_task[p] = t
        all_weeks.append(week_assign)
        info.append((counts, warns, unfilled))

    return available_members, all_weeks, info

if generate or "run_once" not in st.session_state:
    st.session_state["run_once"] = True
    available_members, all_weeks, weekly_info = build_schedule()

    task_columns = ["Chip Tube","Autoclave Waste","Autoclave Drain","Vacuum","Mop","Garbage",
                    "Student Room","Drying Racks","Water alcohol","Consumable Goods"]
    if liquid_weeks:
        task_columns = ["Liquid Waste"] + task_columns

    def join_names(lst):
        return ", ".join(lst) if isinstance(lst, list) else (lst or "-")

    rows = []
    for i, week_assign in enumerate(all_weeks, start=1):
        row = {"Week": f"Week {i}"}
        for col in task_columns:
            if col == "Student Room":
                row[col] = join_names(week_assign.get("Student Room", []))
            else:
                row[col] = join_names(week_assign.get(col, []))
        rows.append(row)

    df = pd.DataFrame(rows)
    st.subheader("📅 Schedule")
    st.dataframe(df, use_container_width=True)

    counts = {m: 0 for m in available_members}
    for _, r in df.iterrows():
        for col in task_columns:
            names = r[col]
            if names and names != "-":
                for n in [x.strip() for x in names.split(",")]:
                    if n in counts:
                        counts[n] += 1
    df_cnt = pd.DataFrame(sorted(counts.items()), columns=["Member","Total Assignments"])
    st.subheader("📈 Assignment Counts")
    st.dataframe(df_cnt, use_container_width=True)

    with st.expander("ℹ️ Weekly adjustments & warnings"):
        for i, (cnt, warns, unfilled) in enumerate(weekly_info, start=1):
            st.markdown(
                f"**Week {i}** — Vacuum {cnt['Vacuum']}, Mop {cnt['Mop']}, "
                f"Garbage {cnt['Garbage']}, SR {cnt['Student Room']}, DR {cnt['Drying Racks']}"
            )
            for wmsg in warns:
                st.write("•", wmsg)
            if unfilled:
                st.write("未割当スロット:", ", ".join(unfilled))

    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="cleaning_schedule.csv",
        mime="text/csv"
    )
