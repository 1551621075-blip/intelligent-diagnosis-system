"""Lightweight Chinese complaint classifier trained from original short phrases.

No keyword-to-domain lookup or serialized Python objects. The process-local
TF-IDF/logistic model includes an explicit non-diagnostic class; confidence and
similarity gates deliberately abstain on vague and unrelated text.
"""

from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .schemas import Domain
from .synthetic import CLASS_DOMAIN, COMPLAINTS

ROUTER_VERSION = "original-char-tfidf-logistic-v1"
ROUTING_THRESHOLD = 0.70
MIN_CLASS_MARGIN = 0.28
MIN_TRAINING_SIMILARITY = 0.24

_PHRASES: dict[str, tuple[str, ...]] = {
    Domain.SAFETY.value: (
        "安全回路报警", "安全联锁失效", "保护回路发生故障", "安全传感器异常",
        "安全报警灯亮了", "联锁检测信号中断", "保护开关没有反应", "联锁线路断开了",
        "安全回路疑似短路", "安全电路电阻过大", "保护传感器供电太低", "安全线路阻值很小",
        "安全回路状态一直异常", "安全检测报警不能复位", "联锁信号时有时无",
        "安全回路的传感器输出异常", "保护装置联锁动作失败", "安全回路检测到断路",
        "安全回路检测到短接", "联锁报警后保护功能失灵", "安全联锁电阻读数跳变",
        "请排查安全传感器为什么报警", "安全保护回路故障需要诊断",
        "设备运行中安全回路指示变红", "安全联锁现在是否正常", "检查保护回路工作状态",
        "设备的安全回路报警了请帮我诊断", "联锁开关故障需要检查", "安全系统报回路异常",
        "保护电路故障灯一直亮", "机器显示安全传感器供电故障", "安全回路阻值异常",
        "安全回路灯报警，设备无法进入正常运行", "联锁保护传感器可能坏了",
        "确认当前安全回路没有异常", "检查安全保护开关有没有失效",
    ),
    Domain.LOW_VOLTAGE.value: (
        "电瓶亏电无法启动", "蓄电池电压不足", "低压供电报警", "充电系统故障",
        "电池端子接触不良", "启动电压掉得很低", "发电机不给电池充电", "电瓶接线柱发烫",
        "低压电源不稳定", "蓄电池没电启动不了", "电池电压偏低启动无力", "充电电压上不来",
        "发动机工作时电池还在掉电", "充电指示灯亮红灯", "电池端子压降异常",
        "电源接头发热压降太大", "蓄电池负极端子松动", "低压电路供电不足",
        "电池电量耗尽了", "电瓶电压一直下降", "发动机启动困难怀疑电池亏电",
        "运行中低压电源故障", "发电机运行但是充电电流为负", "检测蓄电池电压是否正常",
        "请诊断电池与充电系统", "低压报警灯突然点亮", "蓄电池接线处异常发热",
        "电瓶供电断断续续", "请检查充电回路是否正常", "电池充不进电需要检查",
        "启动时蓄电池压降太大", "低压系统的电池端子坏了", "设备提示蓄电池欠压",
        "电池充电状态一直异常", "确认当前低压供电没有故障", "检查蓄电池端子连接情况",
    ),
    "other": (
        "你好", "谢谢", "请帮助我", "设备有问题", "机器坏了", "请检查一下", "怎么回事",
        "不知道哪里出了问题", "我要了解你们的产品", "请给我推荐设备", "咨询报价和价格",
        "我想购买一个电池", "蓄电池销售优惠活动", "安全传感器推广合作", "联锁设备采购报价",
        "充电系统产品介绍", "低压电源最新促销", "安全回路培训课程多少钱", "电瓶批发价格",
        "安全回路是什么原理", "请介绍一下蓄电池的工作原理", "我想咨询售后政策",
        "电池质保多久", "哪里可以买到联锁开关", "安全保护产品广告投放",
        "充电设备招商加盟", "给我一份安全回路的宣传资料", "电池营销文案怎么写",
        "安全传感器购买链接", "低压电源供应商联系方式", "设备安全操作培训报名",
        "咨询联锁系统的型号参数", "想咨询安全回路产品的价格", "推广安全回路诊断服务",
        "电池故障诊断课程优惠促销", "电瓶报警器产品报价咨询", "售卖低压故障检测设备",
        "液压泵压力不足", "发动机冷却水温过高", "油温报警了", "机械轴承声音很大",
        "网络断开连接不上", "轮胎气压低", "空调不制冷", "燃油不足", "制动系统异响",
        "发动机漏油", "今天的天气怎么样", "帮我订酒店", "写一首诗", "午饭吃什么",
        "安全", "电池", "故障", "正常", "报警", "我想咨询一下", "产品咨询不是设备诊断",
        "请问低压是什么意思", "电池供电技术文章", "安全联锁标准下载", "安全培训推广",
        "电瓶维修服务推广广告", "安全回路报警器促销购买", "咨询电池故障报警器售价",
        "充电异常检测仪器推广", "请报价安全传感器报警设备", "购买电池端子故障检测器",
    ),
}


@lru_cache(maxsize=1)
def _trained_router():
    texts, labels = [], []
    for label, phrases in _PHRASES.items():
        texts.extend(phrases)
        labels.extend([label] * len(phrases))
    for class_id, phrases in COMPLAINTS.items():
        texts.extend(phrases)
        labels.extend([CLASS_DOMAIN[class_id].value] * len(phrases))
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(C=12.0, class_weight="balanced", max_iter=600, random_state=17)
    classifier.fit(matrix, labels)
    return vectorizer, classifier, matrix, np.asarray(labels)


def classify_complaint(text: str) -> tuple[Domain | None, float]:
    """Return a domain or abstention with an uncalibrated text-classifier score."""
    normalized = " ".join(text.strip().split())
    if len(normalized) < 4 or len(normalized) > 2000:
        return None, 0.0
    vectorizer, classifier, training_matrix, training_labels = _trained_router()
    vector = vectorizer.transform([normalized])
    if vector.nnz == 0:
        return None, 0.0
    probabilities = classifier.predict_proba(vector)[0]
    order = np.argsort(probabilities)[::-1]
    label = str(classifier.classes_[order[0]])
    confidence = float(probabilities[order[0]])
    if label == "other":
        return None, confidence
    similarity = float((training_matrix[training_labels == label] @ vector.T).max())
    if (
        confidence < ROUTING_THRESHOLD
        or confidence - float(probabilities[order[1]]) < MIN_CLASS_MARGIN
        or similarity < MIN_TRAINING_SIMILARITY
    ):
        return None, confidence
    return Domain(label), confidence
