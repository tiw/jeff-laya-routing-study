#!/usr/bin/env python3
"""Formal E2: topic-technicality gradient, 4 levels x 20 topics x 3 action frames = 240 items."""
import json, os, time, urllib.request

from laya_mlx import Router

CRITERIA = {
    "researcher": "gathers information, searches, summarizes, or investigates questions",
    "coder": "writes, edits, debugs code or scripts, or implements features",
    "writer": "drafts, polishes, translates, or formats prose and documents",
    "reviewer": "critiques, audits, or checks existing work and gives feedback",
}
QUESTION = {
    "type": "choice",
    "instructions": "You are routing an incoming user request in `request` to one of four sub-agents. Pick exactly one.",
    "criteria": CRITERIA,
}

TOPICS = {
    0: ["我的吉他弦总是很快生锈", "我的面包一再发酵失败", "我的月季长满了蚜虫", "我的白衬衫洗后发黄",
        "我跑步时膝盖疼", "我冲的咖啡总是偏酸", "我的米饭经常夹生", "我的皮鞋开胶了",
        "我最近脱发很严重", "我晚上总是失眠", "我的猫突然不吃猫粮", "我的多肉植物徒长了",
        "我的铁锅生锈了", "我的蜂蜜结晶了", "我冬天长冻疮", "我牙疼了一周",
        "我的近视度数加深很快", "我的衣柜发霉了", "我的自行车链条异响", "我的牛奶放两天就结块"],
    1: ["我的 Excel 求和结果不对", "我的打印机连不上 Wi-Fi", "我的 Word 页码乱了", "我的手机电池掉电特别快",
        "我的微信消息总是延迟", "我的 PPT 字体在别的电脑上丢失", "我的路由器信号很差", "我的 U 盘读不出来",
        "我的网银转账一直失败", "我的邮箱收不到验证码", "我的照片备份总是失败", "我的视频会议很卡",
        "我的输入法候选词错乱", "我的日历同步丢失事件", "我的 PDF 文件打不开", "我的二维码扫不出来",
        "我的蓝牙耳机频繁断连", "我的手机充电时发烫", "我的会员被自动扣费", "我的快递单号查不到物流"],
    2: ["我的网站首页打开很慢", "我的 Docker 容器启动后立刻退出", "我的 API 一直返回 500", "我的前端页面样式全部错乱",
        "我的 Git 合并冲突解决不了", "我的 Nginx 配置总是报错", "我的域名解析不生效", "我的数据库连接池总是耗尽",
        "我的消息队列严重积压", "我的缓存经常被击穿", "我的 CI 构建老是失败", "我的服务器日志把磁盘写满了",
        "我的 HTTPS 证书过期了", "我的 CDN 缓存一直不更新", "我的端口被占用", "我的定时任务不触发",
        "我的 WebSocket 频繁断开", "我的爬虫被封 IP", "我的依赖版本冲突装不上", "我的服务内存一直在涨"],
    3: ["我的模型训练 loss 周期性 spike", "我的 PostgreSQL 出现慢查询", "我的 CUDA 训练报 OOM", "我的 Transformer 注意力熵塌缩",
        "我的向量索引召回率很低", "我的 GPU 利用率上不去", "我的分布式训练 NCCL 超时", "我的 LoRA 微调严重过拟合",
        "我的量化模型精度损失很大", "我的 RAG 检索总是产生幻觉", "我的 K8s Pod 反复重启", "我的 Kafka 消费者严重滞后",
        "我的 ES 集群发生脑裂", "我的模型推理延迟很高", "我的微调出现灾难性遗忘", "我的深层网络梯度消失",
        "我的 tokenizer 并行预处理很慢", "我的 checkpoint 文件损坏", "我的显存碎片化严重", "我的词表太大导致 softmax 很慢"],
}

FRAMES = {
    "explain": ("用户请求：解释一下为什么{x}。", "researcher"),
    "fix": ("用户请求：帮我排查并解决{x}的问题。", "coder"),
    "write": ("用户请求：给初学者写一篇关于{x}的一页纸科普介绍。", "writer"),
}

r = Router(models={"typed-decisions": os.path.expanduser("~/.cache/laya/typed-decisions")},
           default="typed-decisions")

def laya_ask(state_text):
    out = r.predict({"request": state_text}, {"route": QUESTION}, model="typed-decisions")
    a = out["answers"]["route"]
    return a["choice"], a.get("probabilities")

def jeff_ask(state_text):
    body = json.dumps({"model": "jeff-latest", "state": state_text,
                       "questions": {"route": QUESTION}}).encode()
    req = urllib.request.Request("http://localhost:8766/v1/systemone",
                                 data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.load(resp)
    a = out["answers"]["route"]
    return a["choice"], a["probabilities"]

results = []
n = 0
for frame, (tmpl, gold) in FRAMES.items():
    for level, topics in TOPICS.items():
        for topic in topics:
            state = tmpl.format(x=topic)
            jc, jp = jeff_ask(state)
            lc, lp = laya_ask(state)
            results.append({"frame": frame, "level": level, "topic": topic, "gold": gold,
                            "jeff_choice": jc, "jeff_probs": jp,
                            "laya_choice": lc, "laya_probs": lp})
            n += 1
            if n % 40 == 0:
                print(f"{n}/240 done", flush=True)

json.dump(results, open("tests_local/exp2_formal_results.json", "w"), ensure_ascii=False, indent=1)
print("saved tests_local/exp2_formal_results.json")
