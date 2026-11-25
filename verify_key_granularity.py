# -*- coding: utf-8 -*-
"""改前 vs 改後的關鍵驗證：拿真實 bot 產生的那幾句去跑新管線。

驗收標準：在改動前，seq7/8/9/13/14 全部落在純「約定與計畫」（一格裝到底）；
改動後應該散到不同主題，不再互撞。
"""
import json
import urllib.request

import memory_keyer as k

MODEL_URL = "http://127.0.0.1:18080/v1/chat/completions"
DECISION_URL = "http://127.0.0.1:18081/v1/systemone"

with open("startlux_8092.key", encoding="utf-8") as f:
    DECISION_KEY = f.read().strip()


def model_reply(messages):
    body = {
        "model": "qwen3.6-35b",
        "messages": messages,
        # ICE 25G 是 thinking 模型：不關思考的話 reasoning 會把 max_tokens 吃光，
        # content 回空字串（finish_reason=length）。bot 本身也是這樣設的。
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
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


# 真實 bot 對話裡實際產生的句子（訊息 #47 到 #77）
SENTENCES = [
    "明天星期六，妳應該不用加班吧?",
    "小明早上十點半英文課，還有下午兩點的圍棋課，妳要接送還是我接送?",
    "我都退伍了沒在上班還請什麼假，妳下午要睡覺就去睡，小明的圍棋課我接送就好了",
    "那中午妳接小明英文課回來想吃什麼午餐?",
    "我不喜歡吃魚耶，而且等妳接小明回來都12點多了，再煮午餐就太晚了，還是我去買回來吃就好?",
    "我買涼麵加壽司好了，巷口那間涼麵六日都有擺攤，大寶愛吃青醬的，小明愛吃麻醬的，妳呢?",
    "垃圾車晚上7點45分才來啦",
    "媽媽走了啦，只剩爸爸，他沒有出過國，突然想要去日本逛逛",
]


def main():
    prefilter = k.DecisionPrefilter(DECISION_URL, DECISION_KEY)
    chooser = k.ModelKeyChooser(model_reply)
    keys = []
    print("%-52s %-6s %-22s %s" % ("句子", "記?", "key", "how"))
    print("-" * 100)
    for sentence in SENTENCES:
        remember, category, how = prefilter.judge(sentence)
        if remember is False:
            print("%-52s %-6s %-22s %s" % (sentence[:50], "不記", "-", how))
            continue
        if how != "ok":
            print("%-52s %-6s %-22s %s" % (sentence[:50], "?", "-", how))
            continue
        key, key_how = chooser.choose(sentence, (), category=category)
        keys.append(key)
        print("%-52s %-6s %-22s %s" % (sentence[:50], "記", key, key_how))

    from collections import Counter
    print()
    print("=== 結果 ===")
    print("寫入 %d 筆，distinct key %d 個" % (len(keys), len(set(keys))))
    for key, count in Counter(keys).most_common():
        print("  %-24s %d 筆%s" % (key, count, "  ⚠️ 互撞" if count > 1 else ""))


if __name__ == "__main__":
    main()
