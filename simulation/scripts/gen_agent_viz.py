"""gen_agent_viz.py — 인터랙티브 agent-tracker 시각화 생성 (2026-06-26, G-370).

agent_history(실 SEIR-V-D kernel)로 한 agent의 Seoul 통근·감염상태 변화 + 인구집단 SEIR
데이터를 만들고, **자족형 HTML**(브라우저서 바로 열림, 시간 슬라이더)을 출력한다. 채팅 inline
위젯을 repo 코드로 영구화 — 재현 가능(같은 seed = 같은 결과).

입력: 없음(simulate_with_history 내부에서 합성인구 생성). 출력: figures/agent_tracker_seoul.html.
Usage: .venv/bin/python -m simulation.scripts.gen_agent_viz
Returns: HTML 경로(print). Side effects: HTML 작성. Performance: ~10s(kernel 120-day chaining).
"""
from __future__ import annotations

import json
import os

import numpy as np

# Seoul 25구 지리좌표 (정규화 0-1, x=서→동, y=남→북) — 전체명 키
_GEO = {
    "강남구": [.61, .33], "강동구": [.80, .42], "강북구": [.50, .75], "강서구": [.19, .48],
    "관악구": [.43, .27], "광진구": [.64, .50], "구로구": [.28, .34], "금천구": [.33, .26],
    "노원구": [.59, .80], "도봉구": [.50, .85], "동대문구": [.56, .60], "동작구": [.44, .36],
    "마포구": [.38, .51], "서대문구": [.41, .60], "서초구": [.53, .30], "성동구": [.56, .52],
    "성북구": [.51, .68], "송파구": [.72, .34], "양천구": [.27, .42], "영등포구": [.38, .43],
    "용산구": [.45, .43], "은평구": [.39, .70], "종로구": [.46, .62], "중구": [.48, .52],
    "중랑구": [.64, .62],
}


def build_data(*, n_agents: int = 3000, t_days: int = 120, seed: int = 42) -> dict:
    """실 kernel로 통근·감염 agent 1명 + 인구집단 SEIR 데이터 구성.

    Args:
        n_agents: 합성인구 수. t_days: 시뮬 일수. seed: 결정성 시드.

    Returns:
        dict {agent, attrs, home, work, traj, transitions, agg(T×4), gu_names, gu_pos, peakI, peakDay}.
    Side effects: agent_history.simulate_with_history 가 DB(read-only)서 합성인구 생성.
    """
    from simulation.abm.agent_history import (
        extract_agent_trajectory, simulate_with_history,
    )
    from simulation.abm.synthetic_population import GU_NAMES

    r = simulate_with_history(N=n_agents, T_days=t_days, seed=seed, beta=0.42,
                              sigma=1 / 2.2, gamma=1 / 5.0, delta=0.003,
                              import_rate=0.0008, beta_amp=0.25, year=2024)
    hs = r["history_state"]
    attrs = r["attrs"]
    home, work = attrs["home_gu"], attrs["work_gu"]
    infected = np.where((hs == 2).any(axis=0))[0]
    commute_inf = [a for a in infected if home[a] != work[a]]
    if not commute_inf:
        commute_inf = list(infected) or [0]
    a = int(commute_inf[len(commute_inf) // 3])
    tr = extract_agent_trajectory(r, a)
    agg = r["aggregate"]
    names = [n[:-1] if n != "중구" else "중구" for n in GU_NAMES]
    pos = [_GEO[n] for n in GU_NAMES]
    return {
        "agent": a, "attrs": tr["attrs"], "home": int(home[a]), "work": int(work[a]),
        "traj": [int(x) for x in hs[:, a]], "transitions": tr["transitions"],
        "agg": [[int(agg[t, s]) for s in range(4)] for t in range(hs.shape[0])],
        "T": int(hs.shape[0]), "gu_names": names, "gu_pos": pos,
        "peakI": int(agg[:, 2].max()), "peakDay": int(agg[:, 2].argmax()),
    }


_HTML = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>ABM agent tracker — Seoul</title>
<style>
 body{{font-family:'Apple SD Gothic Neo','Malgun Gothic',sans-serif;margin:1.2rem;color:#222;background:#fff}}
 .card{{background:#f4f3ee;border-radius:8px;padding:.8rem 1rem;display:inline-block;margin:0 8px 8px 0;vertical-align:top}}
 .lg{{display:inline-flex;align-items:center;gap:4px;font-size:13px;color:#555;margin-right:12px}}
 .dot{{width:11px;height:11px;border-radius:50%;display:inline-block}}
 svg{{width:100%;max-width:760px;background:#fff;border:1px solid #ddd;border-radius:8px}}
 input[type=range]{{width:100%}} button{{padding:4px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}}
 .gul{{font-size:9px;fill:#888}} h2{{font-size:17px;font-weight:500}}
</style></head><body>
<h2>서울 ABM agent-tracker — agent #{agent} 통근·감염 + 인구집단 SEIR</h2>
<div>
 <div class="card"><div style="font-size:12px;color:#666">선택 agent #{agent} (실 SEIR-V-D kernel · agent_history)</div>
  <div style="margin-top:6px;font-size:14px">{age}대 · {sex} · 기저질환 {sev} · {home_nm} ⇄ {work_nm}</div></div>
 <div class="card"><div style="font-size:12px"><span style="color:#666">day <b id="dayv">0</b> / {tmax}</span>
  &nbsp;<span id="statev" style="font-weight:500">상태 S · {home_nm}</span></div>
  <input type="range" min="0" max="{tmax}" value="0" step="1" id="day"/>
  <button id="play">▶ 재생</button></div>
</div>
<div style="margin:6px 0">
 <span class="lg"><span class="dot" style="background:#378ADD"></span>S 감수성</span>
 <span class="lg"><span class="dot" style="background:#EF9F27"></span>E 노출</span>
 <span class="lg"><span class="dot" style="background:#E24B4A"></span>I 감염</span>
 <span class="lg"><span class="dot" style="background:#639922"></span>R 회복</span></div>
<svg id="map" viewBox="0 0 680 300"></svg>
<div style="font-size:13px;color:#666;margin:8px 0 2px">인구집단 SEIR (N={N}, 25구 커널) — 정점 I={peakI} @ day {peakDay}. agent 감염창 표시</div>
<svg id="pop" viewBox="0 0 680 130"></svg>
<script>
const D={data};
const SCOL=['#378ADD','#EF9F27','#E24B4A','#639922'],SNM=['S','E','I','R'],NS='http://www.w3.org/2000/svg';
const AGG=D.agg,TRAJ=D.traj,GU=D.gu_names,P=D.gu_pos,HOME=D.home,WORK=D.work,TMAX=D.T-1;
const sx=p=>p[0]*620+30,sy=p=>(1-p[1])*250+25,map=document.getElementById('map');
function ce(t,a){{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}}
map.appendChild(ce('line',{{x1:sx(P[HOME]),y1:sy(P[HOME]),x2:sx(P[WORK]),y2:sy(P[WORK]),stroke:'#999','stroke-width':1.2,'stroke-dasharray':'4 3'}}));
P.forEach((p,i)=>{{const hl=(i===HOME||i===WORK);map.appendChild(ce('circle',{{cx:sx(p),cy:sy(p),r:hl?9:5.5,fill:hl?'#cfe5fb':'#eee',stroke:'#bbb','stroke-width':hl?1:.5}}));
 const t=ce('text',{{x:sx(p),y:sy(p)-9,class:'gul','text-anchor':'middle'}});t.textContent=GU[i];map.appendChild(t);}});
const aMark=ce('circle',{{r:8,stroke:'#fff','stroke-width':1.5}});map.appendChild(aMark);
const pop=document.getElementById('pop'),mx=Math.max(...AGG.map(r=>r[0]));
function px(d){{return 30+d/TMAX*620;}}function py(v){{return 110-v/mx*95;}}
[[0,'#378ADD'],[3,'#639922'],[2,'#E24B4A'],[1,'#EF9F27']].forEach(([s,c])=>{{let dd='M';AGG.forEach((r,d)=>{{dd+=px(d).toFixed(1)+' '+py(r[s]).toFixed(1)+' ';if(d===0)dd+='L';}});pop.appendChild(ce('path',{{d:dd,fill:'none',stroke:c,'stroke-width':1.5}}));}});
const tE=(D.transitions.find(t=>t[2]==='E')||[0])[0],tR=(D.transitions.find(t=>t[2]==='R')||[TMAX])[0];
pop.appendChild(ce('rect',{{x:px(tE),y:15,width:Math.max(2,px(tR)-px(tE)),height:95,fill:'#E24B4A',opacity:.12}}));
const popMark=ce('line',{{y1:15,y2:110,stroke:'#999','stroke-width':1}});pop.appendChild(popMark);
const dEl=document.getElementById('day'),dv=document.getElementById('dayv'),sv=document.getElementById('statev');
function upd(d){{const loc=(d%2===0)?WORK:HOME,st=TRAJ[d],p=P[loc];aMark.setAttribute('cx',sx(p));aMark.setAttribute('cy',sy(p));aMark.setAttribute('fill',SCOL[st]);
 dv.textContent=d;sv.textContent='상태 '+SNM[st]+' · '+GU[loc]+'구';sv.style.color=SCOL[st];popMark.setAttribute('x1',px(d));popMark.setAttribute('x2',px(d));}}
dEl.addEventListener('input',e=>upd(+e.target.value));upd(0);
let tm=null;const pb=document.getElementById('play');
pb.addEventListener('click',()=>{{if(tm){{clearInterval(tm);tm=null;pb.textContent='▶ 재생';return;}}pb.textContent='⏸ 정지';
 tm=setInterval(()=>{{let d=(+dEl.value+1);if(d>TMAX)d=0;dEl.value=d;upd(d);}},120);}});
</script></body></html>"""


def main() -> int:
    d = build_data()
    at = d["attrs"]
    html = _HTML.format(
        agent=d["agent"], age=at.get("age_band", 0) * 10, sex=at.get("sex_label", "?"),
        sev=at.get("severity_label", "?"), home_nm=at.get("home_gu_name", ""),
        work_nm=at.get("work_gu_name", ""), tmax=d["T"] - 1, N=sum(d["agg"][0]),
        peakI=d["peakI"], peakDay=d["peakDay"], data=json.dumps(d, ensure_ascii=False),
    )
    out = "simulation/results/figures/agent_tracker_seoul.html"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  agent #{d['agent']} ({at.get('home_gu_name')}⇄{at.get('work_gu_name')}, "
          f"전이 {d['transitions']}) · 인구 peak I={d['peakI']}@{d['peakDay']}")
    print(f"  → {out} ({len(html) // 1024} KB, 브라우저서 열기)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
