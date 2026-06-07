"""Query planner: natural language -> structured retrieval constraints.

Two modes:
  - rule-based (default): deterministic keyword/synonym matching. Used by the eval harness.
  - llm: Claude API decomposition. Used by the user-facing demo.

Both map free text onto AIDE's fixed event vocabulary.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict

from packages.core.config import settings


# ---- The fixed AIDE vocabulary (from your Step 10.8 output) ----
DRIVER_STATES = [
    "peace", "normal_driving", "looking_around", "anxiety", "making_phone",
    "weariness", "body_movement", "happiness", "anger", "talking", "smoking", "dozing_off",
]
ROAD_STATES = [
    "smooth_traffic", "forward_moving", "parking", "waiting",
    "traffic_jam", "turning", "backward_moving", "changing_lane",
]

# ---- Synonym maps for the rule-based planner ----
DRIVER_SYNONYMS = {
    "looking_around": ["looking around", "distracted", "glancing", "looking away",
                        "not paying attention", "eyes off", "inattentive"],
    "making_phone":   ["phone", "calling", "on the phone", "texting", "mobile", "cellphone"],
    "anxiety":        ["anxious", "anxiety", "nervous", "stressed", "worried", "tense"],
    "weariness":      ["weary", "weariness", "fatigued", "tired", "exhausted"],
    "dozing_off":     ["dozing", "asleep", "sleeping", "nodding off", "falling asleep", "drowsy"],
    "smoking":        ["smoking", "cigarette", "smoke", "vaping"],
    "happiness":      ["happy", "smiling", "cheerful", "joy", "laughing"],
    "anger":          ["angry", "anger", "mad", "furious", "rage"],
    "talking":        ["talking", "chatting", "speaking", "conversation", "conversing"],
    "body_movement":  ["fidgeting", "body movement", "restless", "shifting", "moving around"],
    "peace":          ["calm", "peaceful", "relaxed", "at ease", "composed"],
    "normal_driving": ["normal driving", "attentive", "focused", "driving normally", "eyes on road"],
}
ROAD_SYNONYMS = {
    "changing_lane":  ["lane change", "changing lane", "changing lanes", "switching lane",
                       "merge", "merging", "lane switch"],
    "turning":        ["turning", "turn", "cornering", "making a turn"],
    "traffic_jam":    ["traffic jam", "congestion", "congested", "heavy traffic",
                       "stuck in traffic", "gridlock", "bumper to bumper"],
    "smooth_traffic": ["smooth traffic", "light traffic", "flowing traffic", "clear road",
                       "open road", "free flowing"],
    "waiting":        ["waiting", "stopped", "at a light", "red light", "idle", "stationary"],
    "parking":        ["parking", "parked", "pulling in", "parking lot", "parking space"],
    "forward_moving": ["moving forward", "driving forward", "going straight", "forward", "cruising"],
    "backward_moving":["reversing", "backing up", "backward", "reverse", "going backwards"],
}

# ---- Object class synonyms (COCO classes YOLO detects) ----
OBJECT_SYNONYMS = {
    "person":        ["pedestrian", "person", "people", "walker", "pedestrians"],
    "truck":         ["truck", "lorry", "trucks"],
    "car":           ["car", "cars", "sedan", "vehicle"],
    "motorcycle":    ["motorcycle", "motorbike", "motorcyclist"],
    "bus":           ["bus", "buses"],
    "bicycle":       ["bicycle", "cyclist", "bike", "bicycles"],
    "traffic light": ["traffic light", "stoplight", "stop light", "signal", "traffic signal"],
    "stop sign":     ["stop sign"],
}


@dataclass
class QueryPlan:
    query: str
    driver_states: list[str] = field(default_factory=list)
    road_states: list[str] = field(default_factory=list)
    object_classes: list[str] = field(default_factory=list)
    semantic_query: str = ""
    explanation: str = ""
    planner: str = "rule"  # "rule" or "llm"

    def to_dict(self) -> dict:
        return asdict(self)


def _match(text: str, synonym_map: dict[str, list[str]]) -> list[str]:
    text = text.lower()
    found = []
    for canonical, phrases in synonym_map.items():
        if any(p in text for p in phrases):
            found.append(canonical)
    return found


def plan_rule_based(query: str) -> QueryPlan:
    driver = _match(query, DRIVER_SYNONYMS)
    road = _match(query, ROAD_SYNONYMS)
    objects = _match(query, OBJECT_SYNONYMS)
    return QueryPlan(
        query=query,
        driver_states=driver,
        road_states=road,
        object_classes=objects,
        semantic_query=query,  # rule-based keeps the full query for the semantic layer
        explanation=f"rule-based: driver={driver}, road={road}, objects={objects}",
        planner="rule",
    )


PLANNER_SYSTEM = f"""You decompose natural-language video-search queries into structured constraints
for a driving-video retrieval system with two synchronized cameras: an in-cabin camera (driver) and
a forward camera (road).

You MUST map the query onto these EXACT vocabularies. Do not invent values outside them.

DRIVER_STATES (in-cabin camera): {DRIVER_STATES}
ROAD_STATES (forward camera): {ROAD_STATES}
OBJECT_CLASSES (detected on road): person, car, truck, bus, motorcycle, bicycle, traffic light, stop sign

Output STRICT JSON only, no prose, no markdown fences:
{{
  "driver_states": [subset of DRIVER_STATES, or []],
  "road_states": [subset of ROAD_STATES, or []],
  "object_classes": [subset of OBJECT_CLASSES, or []],
  "semantic_query": "a short phrase for appearance-based search, or the original query",
  "explanation": "one sentence on how you mapped it"
}}

Rules:
- Pick at most ONE driver emotion and ONE driver behavior (the dataset has one of each per clip).
- If a concept doesn't map to any vocabulary value, leave that list empty and rely on semantic_query.
- Be conservative: only include a value if the query clearly implies it."""


def plan_llm(query: str) -> QueryPlan:
    from anthropic import Anthropic
    client = Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",   # fast + cheap; good for bounded extraction
        max_tokens=400,
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )
    raw = msg.content[0].text.strip()
    # strip accidental fences just in case
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    data = json.loads(raw)
    # keep only valid vocabulary values (defensive)
    driver = [d for d in data.get("driver_states", []) if d in DRIVER_STATES]
    road = [r for r in data.get("road_states", []) if r in ROAD_STATES]
    return QueryPlan(
        query=query,
        driver_states=driver,
        road_states=road,
        object_classes=data.get("object_classes", []),
        semantic_query=data.get("semantic_query") or query,
        explanation=data.get("explanation", ""),
        planner="llm",
    )


def make_plan(query: str, use_llm: bool = False) -> QueryPlan:
    """Unified entry point. Defaults to deterministic rule-based."""
    if use_llm:
        try:
            return plan_llm(query)
        except Exception as e:
            # graceful fallback so a missing key / network blip never breaks search
            fallback = plan_rule_based(query)
            fallback.explanation += f"  (LLM failed: {e}; used rule-based)"
            return fallback
    return plan_rule_based(query)
