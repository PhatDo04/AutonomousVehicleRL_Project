# -*- coding: utf-8 -*-
"""Tính lại số 3-seed từ thesis_night_summary + đối chiếu Bảng 14 DO_AN. Chạy 1 lần."""
import re, statistics as st
from pathlib import Path
LOGS = Path("outputs/logs")
NIGHTS = {0:"20260611_225051", 1:"20260612_135116", 2:"20260612_232535"}
TXT = {s:(LOGS/f"thesis_night_summary_{t}.md").read_text(encoding="utf-8") for s,t in NIGHTS.items()}

def block(txt, label):
    m = re.search(r"##### %s\n(.*?)(?=#####|\Z|\n##)" % re.escape(label), txt, re.S)
    if not m: return None
    b = m.group(1); d = {}
    for k, pat in [("s", r"success=([\d.]+)%"), ("c", r"collision=([\d.]+)%"),
                   ("y", r"NH\S*NG-merger\(<35m\)=(\d+)%")]:
        mm = re.search(pat, b)
        if mm: d[k] = float(mm.group(1))
    return d

def agg(pre, suf, key, eseeds=(0,1,2)):
    per = []
    for s in NIGHTS:
        vs = [block(TXT[s], f"{pre} seed{es}{suf}") for es in eseeds]
        vs = [d[key] for d in vs if d and key in d]
        if vs: per.append(st.mean(vs))
    return (f"{st.mean(per):.0f}±{st.pstdev(per):.0f}" if len(per) > 1 else "?"), [round(x) for x in per]

print("TÍNH LẠI TỪ 3 ĐÊM OVERNIGHT → so Bảng 14 DO_AN")
print("="*64)
rows = [
 ("PPO S2-best success",  "PPO-best", " khiên", "s", "86±6"),
 ("PPO S2-best collision", "PPO-best", " khiên", "c", "14±6"),
 ("PPO S2-best no-shield", "PPO-best", " NO-SHIELD", "s", "14±6"),
 ("PPO S3-yield success",  "PPO-S3-yieldbest", " khiên", "s", "86±7"),
 ("PPO S3-yield nhường",   "PPO-S3-yieldbest", " khiên", "y", "89±6"),
 ("PPO S4-prop success",   "S4-best", " khiên", "s", "83±9"),
 ("PPO S4-prop no-shield", "S4-best", " NO-SHIELD", "s", "28±17"),
 ("A2C S2-best success",   "A2C-best", " khiên (stoch)", "s", "73±7"),
 ("A2C S2-best collision", "A2C-best", " khiên (stoch)", "c", "27±7"),
 ("A2C S2-best nhường",    "A2C-best", " khiên (stoch)", "y", "37±4"),
 ("A2C S2-best no-shield", "A2C-best", " NO-SHIELD (stoch)", "s", "52±7"),
]
for name, pre, suf, key, doan in rows:
    val, per = agg(pre, suf, key)
    ok = "OK" if val == doan else "  <<< LECH!"
    print(f"  {name:24s}: tính={val:7s} | Bảng14={doan:7s} {ok}  {per}")

# Greedy (luật, eval seed 0/1/2, đêm 0)
gk = [block(TXT[0], f"GREEDY seed{i} khiên") for i in (0,1,2)]
gkv = [d["s"] for d in gk if d]
gv = f"{st.mean(gkv):.0f}±{st.pstdev(gkv):.0f}"
print(f"  {'Greedy success':24s}: tính={gv:7s} | Bảng14=53±12 {'OK' if gv=='53±12' else '<<<'}  {[round(x) for x in gkv]}")
print(f"  {'Greedy collision':24s}: tính={100-st.mean(gkv):.0f}±{st.pstdev(gkv):.0f} | Bảng14=47±12")

# Gate S1 (1 giá trị/đêm)
print("\nGATE S1 (mỗi đêm 1 giá trị):")
for name, lbl, doan in [("PPO argmax","PPO-S1 argmax","100±0"),("PPO stoch","PPO-S1 stochastic","97±5"),
                        ("A2C argmax","A2C-S1(sharpened) argmax","0±0"),("A2C stoch","A2C-S1(sharpened) stochastic","77±5")]:
    vs=[block(TXT[s],lbl)["s"] for s in NIGHTS if block(TXT[s],lbl)]
    v=f"{st.mean(vs):.0f}±{st.pstdev(vs):.0f}"
    print(f"  {name:12s}: tính={v:7s} | Bảng15={doan:6s} {'OK' if v==doan else '<<< LECH'}  {[round(x) for x in vs]}")
