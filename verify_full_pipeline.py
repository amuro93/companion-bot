# -*- coding: utf-8 -*-
"""完整管線驗證：真實對話每一句都走 split → 規則 → 決策模型 → key 選擇。

驗收重點：
  1. 「約定與計畫」那 12 句有沒有散到不同主題（改動前全部是純分類，必然互撞）
  2. 目標事實（小明英文課時間、家族旅遊）有沒有進得去
  3. 印出每一步的 how，看純分類還有幾句
"""
import sqlite3
from collections import Counter

import memory_gate as mg
import memory_keyer as k

DECISION_URL = "http://127.0.0.1:18081/v1/systemone"
MODEL_URL = "http://127.0.0.1:18080/v1/chat/completions"

with open("startlux_8092.key", encoding="utf-8") as f:
    DECISION_KEY = f.read().strip()


def model_reply(messages):
    import json
    import urllib.request
    body = {
        "model": "qwen3.6-35b",
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 512,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        MODEL_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


def main():
    prefilter = k.DecisionPrefilter(DECISION_URL, DECISION_KEY)
    chooser = k.ModelKeyChooser(model_reply)

    con = sqlite3.connect("file:synthetic.sqlite3?mode=ro", uri=True)
    rows = con.execute(
        "SELECT rowid, content FROM messages WHERE role='user' ORDER BY rowid"
    ).fetchall()

    keys = []
    hows = Counter()
    print("%-4s %-20s %-6s %s" % ("#", "key", "how", "句子"))
    print("-" * 96)
    for rid, content in rows:
        for sentence in mg.split_sentences(content):
            if not mg.prefilter(sentence):
                continue
            remember, category, pre_how = prefilter.judge(sentence)
            if remember is False:
                hows["prefilter_no"] += 1
                continue
            if pre_how != "ok":
                hows["prefilter_" + pre_how] += 1
                continue
            key, key_how = chooser.choose(sentence, (), category=category)
            hows[key_how] += 1
            keys.append(key)
            print("%-4s %-20s %-6s %s" % (rid, key, key_how, sentence[:44]))

    print()
    print("=== 統計 ===")
    print("寫入 %d 筆｜distinct key %d 個" % (len(keys), len(set(keys))))
    print("how 分佈:", dict(hows))
    print()
    dupes = [(key, n) for key, n in Counter(keys).most_common() if n > 1]
    if dupes:
        print("⚠️ 仍然互撞的 key：")
        for key, n in dupes:
            print("   %-24s %d 筆" % (key, n))
    else:
        print("✅ 沒有任何 key 被兩件事共用")


if __name__ == "__main__":
    main()
