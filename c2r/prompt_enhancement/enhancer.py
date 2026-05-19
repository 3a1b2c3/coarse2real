import gc
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

from ..data.video import VideoData


DESCRIPTION_PROMPT_VERSION = 14
DESCRIPTION_POSTPROCESS_PROMPT_VERSION = 3
FUSION_PROMPT_VERSION = 3

SCENE_SUBJECT_SYSTEM_PROMPT = (
    "You analyze temporally ordered frames from a coarse control video for controllable video generation. "
    "This is not a scene-captioning task. Extract only spatial layout, background element placement and size, the exact count of distinct subjects, and their coarse motion. "
    "Return valid JSON only with exactly these keys: `scene_layout`, `background_elements`, `subject_count`, `subjects`. "
    "`scene_layout` must be one or two short sentences describing the static frame layout after mentally removing all subjects. "
    "Describe only locations, relative sizes, depth, and spatial relationships of background buildings, structures, ground regions, walls, corridors, paths, or large scene elements. "
    "Use phrases such as building in the foreground, building to the right, large building next to a smaller structure, central open area, left-side boundary, right-side structures, distant background structures, or strong forward depth. "
    "Do not mention color, texture, material, lighting, weather, time of day, render style, visual quality, clothing, people, subjects, motion, camera behavior, fine facade details, or shape words in `scene_layout`. "
    "Avoid words such as dark, black, darker, light, lighter, bright, colorful, blocky, rectangular, cubic, curved, angular, round, flat, square, circular, cylindrical, polygonal, geometric, or low-poly for scene and background descriptions. "
    "`background_elements` must be a JSON array of the main static background elements. Each item must have exactly these string keys: `element`, `position`, `relative_size`, `relation`. "
    "For `element`, use generic names such as building, wall, path, ground area, structure, arch, stairs, or background mass. "
    "For `position`, use frame-relative and depth-relative placement such as on the right foreground, in the left background, across the central midground, or at the distant center. "
    "For `relative_size`, describe approximate size only, such as large and tall, small, wide, narrow, or medium-sized. Do not describe shape. "
    "For `relation`, describe adjacency or relative placement such as next to a smaller structure, behind the open foreground, beside the central path, or flanking the left side. "
    "Do not invent semantic details; if a structure type is uncertain, call it a building, structure, wall, path, or background mass. "
    "`subject_count` must be the exact number of distinct visible persons or character-like subjects represented anywhere across the whole video sequence, not only in the first or last frame. "
    "Track subjects across frames so the same subject is not counted twice. Count a subject once even if it moves, disappears briefly, or changes size. "
    "If multiple subjects move together, you may describe them as one group, but their group `count` must still contribute to the exact total. "
    "`subjects` must be a JSON array of motion groups whose `count` values add up exactly to `subject_count`. Each item must have exactly these keys: `count`, `position`, `relative_size`, `motion`. "
    "Use `count` as an integer. Use `position` for rough frame location and depth, such as in the foreground center, on the right side of the frame, or in the distant background. "
    "Use `relative_size` for approximate subject size in the frame, such as large foreground subject, medium midground subject, or small distant subject. "
    "Use `motion` as one concise phrase describing the net action or direction, such as is walking forward, is walking toward the camera, are stationary while gesturing with arms, or is waving arms in the background. "
    "Mention talking, waving, hand motion, head motion, or stationary behavior when visible. Do not mention identity, emotion, exact body shape, clothing, color, style, or speculative intent. "
    "Do not include camera motion in this response. Return JSON only."
)

SCENE_SUBJECT_USER_PROMPT = (
    "These images are temporally ordered frames from the same coarse control video. "
    "Analyze all frames together. First describe the static layout and background element positions and relative sizes after mentally removing every subject. "
    "Then count the exact number of distinct visible persons or character-like subjects across the whole sequence, avoiding double-counting the same subject over time. "
    "Finally describe each subject or subject group with count, rough position, relative size, and one concise motion phrase. "
    "Return JSON only with `scene_layout`, `background_elements`, `subject_count`, and `subjects`."
)

CAMERA_SYSTEM_PROMPT = (
    "You analyze temporally ordered frames from a coarse control video and extract only the camera or viewpoint motion. "
    "Return valid JSON only with exactly this key: `camera_motion`. "
    "Choose exactly one label from `forward`, `backward`, `leftward`, `rightward`, `pan_left`, `pan_right`, `static`, `unclear`. "
    "You must explicitly compare the first sampled frame and the last sampled frame, using the middle frames only to confirm the direction. "
    "If the fixed background changes size, position, alignment, or parallax between the first and last sampled frames, the camera is not static. "
    "Use `forward` whenever the viewpoint advances into the scene, background structures grow larger, the path expands toward the viewer, or there is consistent depth progression. Prefer `forward` over `static` whenever there is any clear forward progression. "
    "Use `backward` when the viewpoint retreats and background structures shrink or more surrounding layout becomes visible. "
    "Use `pan_left` when the view rotates or sweeps left across the fixed background. Use `pan_right` when the view rotates or sweeps right across the fixed background. "
    "Use `leftward` or `rightward` when the camera translates sideways rather than rotating. "
    "Use `static` only as the last resort when the fixed background remains aligned and does not shift between the first and last sampled frames except for subject movement. "
    "If there is any noticeable background shift, parallax, zoom, or forward depth change, do not use `static`. "
    "Do not describe subjects, background layout, appearance, style, lighting, or anything else. Return JSON only."
)

CAMERA_USER_PROMPT = (
    "These images are temporally ordered frames from the same coarse control video. "
    "Compare the first sampled frame against the last sampled frame first. If the fixed background shifted, changed size, or changed alignment, discard `static` and choose the best movement label. "
    "Use middle frames only to confirm whether the movement is forward, backward, pan_left, pan_right, leftward, or rightward. Return JSON only with the label."
)

DESCRIPTION_POSTPROCESS_SYSTEM_PROMPT = (
    "You rewrite structured control-video descriptions for controllable video generation. "
    "You will receive JSON produced by a vision-language model with keys `scene_layout`, `background_elements`, `subject_count`, `subjects`, and `camera_motion`. "
    "Return valid JSON only with exactly these keys: `scene_layout`, `background_elements`, `subject_count`, `subjects`, `camera_motion`. "
    "Preserve the exact subject count unless the input is internally inconsistent and the subjects array clearly adds to a different total. "
    "Clean the wording so it is useful as explicit control guidance, but do not invent buildings, scene elements, subjects, motion, or camera behavior. "
    "For `scene_layout`, describe only static layout, background element locations, approximate sizes, depth, and spatial relationships. "
    "Remove colors, textures, materials, lighting, weather, time of day, render style, visual quality, clothing, people, subject words, camera behavior, and building-shape descriptions from `scene_layout`. "
    "Remove words such as dark, black, darker, light, lighter, bright, colorful, blocky, rectangular, cubic, curved, angular, round, flat, square, circular, cylindrical, polygonal, geometric, or low-poly from scene/background fields. "
    "For `background_elements`, preserve the main static elements and keep only generic element names, frame/depth position, relative size, and relation to nearby structures. "
    "For `subjects`, preserve the group counts so they add up to `subject_count`, and keep one concise motion phrase per subject or subject group. "
    "Use motion wording that can follow phrases like `1 subject in the foreground is walking forward` or `2 subjects on the right side are stationary while gesturing with arms`. "
    "For `camera_motion`, return one natural camera-motion sentence beginning with `The camera` when possible. Return JSON only."
)

DESCRIPTION_POSTPROCESS_USER_PROMPT = (
    "Clean this control-video description JSON without changing the underlying layout or motion:\n"
    "{description_json}\n\n"
    "Return JSON only with `scene_layout`, `background_elements`, `subject_count`, `subjects`, and `camera_motion`."
)

FUSION_SYSTEM_PROMPT = (
    "You are a prompt fusion assistant for controllable video generation. "
    "Combine a user prompt with structured control-video guidance into one final prompt. "
    "Preserve the user prompt intent first. "
    "Use the control-video guidance only for layout, background element placement and size, exact subject count, subject motion, and camera behavior. "
    "Never overwrite explicit user instructions about clothing, weather, style, architecture, location, or time of day. "
    "If the user prompt already describes a control detail, merge the guidance without repeating it awkwardly. "
    "Do not invent unsupported details. "
    "The caller will append a mandatory natural-language control guidance sentence, so do not create a separate checklist, markdown section, labels, or JSON-like structure. "
    "Return only the fused prompt as one concise paragraph."
)


@dataclass
class PromptEnhancementArtifacts:
    prompt_overrides: dict[tuple[int, int], str]
    control_video_descriptions: dict[int, dict[str, Any]]
    descriptions_path: Path
    prompts_path: Path


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def _release_model(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sample_frame_indices(total_frames: int, target_samples: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Control video has no frames to describe.")
    sample_count = min(total_frames, max(1, target_samples))
    if sample_count == 1:
        return [0]
    indices = []
    for slot in range(sample_count):
        position = round(slot * (total_frames - 1) / (sample_count - 1))
        if not indices or indices[-1] != position:
            indices.append(position)
    return indices


def _strip_wrappers(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _decode_json_string(value: str) -> str:
    return json.loads(f'"{value}"')


_JSON_REPAIR_KEYS = (
    "background_layout",
    "background_elements",
    "camera_motion",
    "scene_layout",
    "subject_count",
    "subject_motion",
    "subjects",
)


def _extract_jsonish_fields(text: str) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}

    for key in ("background_layout", "camera_motion", "scene_layout"):
        match = re.search(rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
        if match:
            payload[key] = _decode_json_string(match.group(1)).strip()

    match = re.search(r'"subject_count"\s*:\s*(-?\d+)', text, flags=re.DOTALL)
    if match:
        payload["subject_count"] = int(match.group(1))

    for key in ("background_elements", "subject_motion", "subjects"):
        match = re.search(rf'"{key}"\s*:\s*(\[[\s\S]*?\])', text, flags=re.DOTALL)
        if not match:
            continue
        parsed = None
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            if key == "subject_motion":
                parsed = [
                    _decode_json_string(item).strip()
                    for item in re.findall(r'"((?:\\.|[^"\\])*)"', match.group(1), flags=re.DOTALL)
                ]
        if parsed is not None:
            payload[key] = parsed

    if payload:
        return payload
    return None


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_wrappers(text)
    direct = _try_parse_json_object(cleaned)
    if direct is not None:
        return direct

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        parsed = _try_parse_json_object(candidate)
        if parsed is not None:
            return parsed

        repaired = candidate
        repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
        repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        repaired = re.sub(
            rf'(["}}\]])(\s*)("(?:(?:{"|".join(_JSON_REPAIR_KEYS)}))"\s*:)',
            r"\1,\2\3",
            repaired,
        )
        repaired = re.sub(r'("\s*)(\n\s*)(")', r'\1,\2\3', repaired)
        parsed = _try_parse_json_object(repaired)
        if parsed is not None:
            return parsed

        extracted = _extract_jsonish_fields(candidate)
        if extracted is not None:
            return extracted
    raise ValueError(f"Failed to parse JSON payload from model output: {text[:500]}")


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


_CAMERA_LABEL_SENTENCES = {
    "forward": "The camera slowly pushes forward through the scene.",
    "push_forward": "The camera slowly pushes forward through the scene.",
    "push_in": "The camera slowly pushes forward through the scene.",
    "dolly_in": "The camera slowly pushes forward through the scene.",
    "backward": "The camera pulls backward away from the scene.",
    "pull_backward": "The camera pulls backward away from the scene.",
    "pull_back": "The camera pulls backward away from the scene.",
    "dolly_out": "The camera pulls backward away from the scene.",
    "leftward": "The camera tracks leftward across the scene.",
    "left": "The camera tracks leftward across the scene.",
    "track_left": "The camera tracks leftward across the scene.",
    "rightward": "The camera tracks rightward across the scene.",
    "right": "The camera tracks rightward across the scene.",
    "track_right": "The camera tracks rightward across the scene.",
    "pan_left": "The camera gently pans across the scene from right to left.",
    "pan_right_to_left": "The camera gently pans across the scene from right to left.",
    "pan_right": "The camera gently pans across the scene from left to right.",
    "pan_left_to_right": "The camera gently pans across the scene from left to right.",
    "static": "The camera remains mostly static with slight handheld motion.",
    "unclear": "The camera motion is unclear.",
}


def _clean_text(value: Any, fallback: str = "unclear") -> str:
    if value is None:
        return fallback
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or fallback


_SCENE_BACKGROUND_BANNED_WORDS = (
    "dark",
    "darker",
    "darkest",
    "black",
    "white",
    "gray",
    "grey",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "pink",
    "purple",
    "brown",
    "colorful",
    "colored",
    "bright",
    "brighter",
    "brightest",
    "light",
    "lighter",
    "lightest",
    "glowing",
    "texture",
    "textured",
    "material",
    "materials",
    "appearance",
    "shape",
    "shaped",
    "shapes",
    "blocky",
    "boxy",
    "rectangular",
    "cubic",
    "cube",
    "curved",
    "angular",
    "round",
    "rounded",
    "flat",
    "square",
    "circular",
    "oval",
    "cylindrical",
    "polygonal",
    "geometric",
    "low-poly",
    "low poly",
)


_SCENE_BACKGROUND_BANNED_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in _SCENE_BACKGROUND_BANNED_WORDS) + r")\b",
    flags=re.IGNORECASE,
)


def _clean_scene_background_text(value: Any, fallback: str = "unclear") -> str:
    text = _clean_text(value, fallback=fallback)
    if text.lower() == "unclear":
        return text
    text = _SCENE_BACKGROUND_BANNED_PATTERN.sub("", text)
    text = re.sub(
        r",?\s*(?:while|and)\s+the\s+(?:left|right|center|middle)(?:\s+side)?\s+(?:has|have|is|are)\s+(?:a|an|the)?\s*(?=[,.;]|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:with|has|have|is|are)\s+(?:a|an|the)?\s*(?=[,.;]|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:more|very)\s+(?=[,.;])", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(and|or)\s+(?=[,.;])", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(large|small|medium-sized|medium|tall|wide|narrow)\s+and\s+(?=(?:building|structure|wall|path|ground|area|mass|element|background)\b)",
        r"\1 ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,;:])\s*([,;:.])", r"\2", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:")
    text = re.sub(r"\b(?:has|have|is|are|with|and|or|a|an|the)\s*[.]$", ".", text, flags=re.IGNORECASE)
    text = text.strip()
    if not text:
        return fallback
    return text


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_count(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0, int(value))
    text = _clean_text(value, fallback="").lower()
    if not text:
        return default
    if text in _NUMBER_WORDS:
        return _NUMBER_WORDS[text]
    match = re.search(r"\d+", text)
    if match:
        return max(0, int(match.group(0)))
    return default


def _normalize_background_elements(value: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            normalized.append(
                {
                    "element": _clean_scene_background_text(
                        item.get("element") or item.get("name") or item.get("type"),
                        fallback="background structure",
                    ),
                    "position": _clean_scene_background_text(
                        item.get("position") or item.get("location") or item.get("placement")
                    ),
                    "relative_size": _clean_scene_background_text(
                        item.get("relative_size") or item.get("size") or item.get("shape")
                    ),
                    "relation": _clean_scene_background_text(
                        item.get("relation") or item.get("relative_position") or item.get("next_to")
                    ),
                }
            )
        else:
            text = _clean_scene_background_text(item, fallback="")
            if text:
                normalized.append(
                    {
                        "element": "background structure",
                        "position": text,
                        "relative_size": "unclear",
                        "relation": "unclear",
                    }
                )
    return normalized[:8]


def _normalize_camera_sentence(value: Any) -> str:
    camera_motion = _clean_text(value)
    compact = camera_motion.lower().strip(".").replace(" ", "_")
    if compact in _CAMERA_LABEL_SENTENCES:
        return _CAMERA_LABEL_SENTENCES[compact]
    if camera_motion.lower() == "unclear":
        return _CAMERA_LABEL_SENTENCES["unclear"]
    if camera_motion.lower().startswith("the camera"):
        return camera_motion
    if camera_motion.lower().startswith("camera "):
        return f"The {camera_motion}"
    return f"The camera {camera_motion[:1].lower()}{camera_motion[1:]}"


def _normalize_subject_groups(payload: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    raw_subjects = (
        payload.get("subjects")
        if payload.get("subjects") is not None
        else payload.get("subject_groups")
    )
    if raw_subjects is None:
        raw_subjects = payload.get("subject_motion")

    groups: list[dict[str, Any]] = []
    for item in _as_list(raw_subjects):
        if isinstance(item, dict):
            count = _coerce_count(
                item.get("count") or item.get("subject_count") or item.get("subjects"),
                default=1,
            )
            groups.append(
                {
                    "count": max(1, count),
                    "position": _clean_text(
                        item.get("position") or item.get("location") or item.get("frame_position")
                    ),
                    "relative_size": _clean_text(
                        item.get("relative_size") or item.get("size") or item.get("scale")
                    ),
                    "motion": _clean_text(
                        item.get("motion")
                        or item.get("subject_motion")
                        or item.get("description")
                        or item.get("action")
                    ),
                }
            )
        else:
            text = _clean_text(item, fallback="")
            if not text:
                continue
            groups.append(
                {
                    "count": max(1, _coerce_count(text, default=1)),
                    "position": "unclear",
                    "relative_size": "unclear",
                    "motion": text,
                }
            )

    explicit_count = _coerce_count(payload.get("subject_count"), default=0)
    group_count = sum(int(group["count"]) for group in groups)
    if not groups:
        if explicit_count > 0:
            return explicit_count, [
                {
                    "count": explicit_count,
                    "position": "unclear",
                    "relative_size": "unclear",
                    "motion": "has unclear motion",
                }
            ]
        return 0, []

    if explicit_count > group_count:
        groups.append(
            {
                "count": explicit_count - group_count,
                "position": "unclear",
                "relative_size": "unclear",
                "motion": "has unclear motion",
            }
        )
        group_count = explicit_count
    elif explicit_count == 0 or explicit_count < group_count:
        explicit_count = group_count

    return explicit_count, groups


def _normalize_description(payload: dict[str, Any]) -> dict[str, Any]:
    subject_count, subjects = _normalize_subject_groups(payload)
    return {
        "scene_layout": _clean_scene_background_text(payload.get("scene_layout") or payload.get("background_layout")),
        "background_elements": _normalize_background_elements(payload.get("background_elements")),
        "subject_count": subject_count,
        "subjects": subjects,
        "camera_motion": _normalize_camera_sentence(payload.get("camera_motion")),
    }


def _normalize_layout_subjects(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_description(payload)
    return {
        "scene_layout": normalized["scene_layout"],
        "background_elements": normalized["background_elements"],
        "subject_count": normalized["subject_count"],
        "subjects": normalized["subjects"],
    }


def _normalize_camera_payload(payload: dict[str, Any]) -> str:
    return _normalize_camera_sentence(payload.get("camera_motion"))


def _clean_fused_prompt(text: str) -> str:
    cleaned = _strip_wrappers(text)
    cleaned = cleaned.strip().strip('"').strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _has_content(text: str) -> bool:
    return bool(text and text.lower() != "unclear")


def _position_phrase(position: str) -> str:
    position = _clean_text(position)
    if not _has_content(position):
        return ""
    lower = position.lower()
    if lower.startswith(("in ", "on ", "at ", "near ", "beside ", "behind ", "across ", "along ", "toward ")):
        return f" {position}"
    return f" in {position}"


def _format_background_element(element: dict[str, str]) -> str:
    name = _clean_scene_background_text(element.get("element"), fallback="background structure")
    relative_size = _clean_scene_background_text(element.get("relative_size"))
    position = _clean_scene_background_text(element.get("position"))
    relation = _clean_scene_background_text(element.get("relation"))

    if _has_content(relative_size):
        description = f"{relative_size} {name}"
    else:
        description = name
    description += _position_phrase(position)
    if _has_content(relation):
        description = f"{description} {relation}"
    return _clean_scene_background_text(description).strip()


def _format_background_elements(elements: list[dict[str, str]]) -> str:
    formatted = [_format_background_element(element) for element in elements]
    formatted = [item for item in formatted if item]
    if not formatted:
        return "the main background element locations and sizes are unclear"
    return "; ".join(formatted)


def _motion_phrase(count: int, motion: str) -> str:
    motion = _clean_text(motion, fallback="has unclear motion")
    lower = motion.lower()
    if lower.startswith(("is ", "are ", "was ", "were ", "has ", "have ")):
        return motion
    if lower.startswith(
        (
            "walking",
            "moving",
            "standing",
            "staying",
            "remaining",
            "talking",
            "gesturing",
            "waving",
            "turning",
            "facing",
            "crossing",
            "approaching",
        )
    ):
        return f"{'is' if count == 1 else 'are'} {motion}"
    return motion


def _format_subject_motion(group: dict[str, Any]) -> str:
    count = max(1, _coerce_count(group.get("count"), default=1))
    noun = "subject" if count == 1 else "subjects"
    position = _position_phrase(_clean_text(group.get("position")))
    motion = _motion_phrase(count, _clean_text(group.get("motion"), fallback="has unclear motion"))
    sentence = f"{count} {noun}{position} {motion}".strip()
    return sentence.rstrip(".") + "."


def _format_subject_count(count: int) -> str:
    if count <= 0:
        return "There are no clearly visible persons in this video."
    if count == 1:
        return "There is 1 person in this video."
    return f"There are {count} persons in this video."


def _format_control_guidance(description: dict[str, Any]) -> str:
    normalized = _normalize_description(description)
    layout = _clean_scene_background_text(normalized["scene_layout"], fallback="The scene layout is unclear.")
    background = _format_background_elements(normalized["background_elements"])
    subject_count = _format_subject_count(int(normalized["subject_count"]))
    subject_motion = " ".join(_format_subject_motion(group) for group in normalized["subjects"])
    if not subject_motion:
        subject_motion = "No distinct subject motion is visible."
    camera_motion = _normalize_camera_sentence(normalized["camera_motion"]).rstrip(".") + "."
    if layout.lower() == "unclear":
        layout_sentence = "The scene layout is unclear."
    else:
        layout_sentence = f"The scene layout has {layout[:1].lower()}{layout[1:].rstrip('.')}."
    return (
        f"{layout_sentence} "
        f"The background includes {background.rstrip('.')}. "
        f"{subject_count} "
        f"{subject_motion} "
        f"{camera_motion}"
    )


def _ensure_control_guidance_suffix(prompt: str, control_guidance: str) -> str:
    prompt = _clean_fused_prompt(prompt)
    control_guidance = _clean_fused_prompt(control_guidance)
    if not control_guidance:
        return prompt
    if not prompt:
        return control_guidance
    if prompt.endswith(control_guidance):
        return prompt
    if control_guidance in prompt:
        prompt = prompt.replace(control_guidance, "").strip()
    return f"{prompt.rstrip()} {control_guidance}"


def _load_json_file(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _greedy_generation_config(model) -> Any:
    config = deepcopy(model.generation_config)
    config.do_sample = False
    for attr in (
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "typical_p",
        "epsilon_cutoff",
        "eta_cutoff",
    ):
        if hasattr(config, attr):
            setattr(config, attr, None)
    return config


class QwenControlVideoDescriber:
    def __init__(
        self,
        model_id: str,
        device: str,
        torch_dtype: torch.dtype,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 768 * 28 * 28,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.processor = None
        self.model = None
        self.generation_config = None

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            use_fast=False,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=self.torch_dtype,
            attn_implementation="sdpa",
        ).eval()
        self.model.to(self.device)
        self.generation_config = _greedy_generation_config(self.model)

    @torch.no_grad()
    def _generate_json(self, messages: list[dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
        self._ensure_loaded()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.device)
        output_ids = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        try:
            return _extract_json_object(response)
        except ValueError as exc:
            print(
                "Warning: failed to parse VLM JSON response; falling back to empty payload. "
                f"Model output excerpt: {response[:300]!r}. Error: {exc}"
            )
            return {}

    @torch.no_grad()
    def describe_layout_and_subjects(self, frame_paths: list[Path]) -> dict[str, Any]:
        content: list[dict[str, str]] = []
        for index, frame_path in enumerate(frame_paths, start=1):
            content.append({"type": "text", "text": f"Frame {index}:"})
            content.append({"type": "image", "path": str(frame_path)})
        content.append({"type": "text", "text": SCENE_SUBJECT_USER_PROMPT})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SCENE_SUBJECT_SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        return _normalize_layout_subjects(self._generate_json(messages, max_new_tokens=768))

    @torch.no_grad()
    def describe_camera_motion(self, frame_paths: list[Path]) -> str:
        content: list[dict[str, str]] = []
        for index, frame_path in enumerate(frame_paths, start=1):
            content.append({"type": "text", "text": f"Frame {index}:"})
            content.append({"type": "image", "path": str(frame_path)})
        content.append({"type": "text", "text": CAMERA_USER_PROMPT})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": CAMERA_SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        return _normalize_camera_payload(self._generate_json(messages, max_new_tokens=32))

    def close(self) -> None:
        processor = self.processor
        model = self.model
        self.processor = None
        self.model = None
        self.generation_config = None
        _release_model(model, processor)


class QwenPromptEnhancementLLM:
    def __init__(self, model_id: str, device: str, torch_dtype: torch.dtype):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.tokenizer = None
        self.model = None
        self.generation_config = None

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self.torch_dtype,
            attn_implementation="sdpa",
        ).eval()
        self.model.to(self.device)
        self.generation_config = _greedy_generation_config(self.model)

    @torch.no_grad()
    def _generate_from_messages(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        self._ensure_loaded()
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **chat_template_kwargs,
            )
        except TypeError:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                **chat_template_kwargs,
            )
        inputs = self.tokenizer([formatted], return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = [
            output_ids_row[len(input_ids) :]
            for input_ids, output_ids_row in zip(inputs.input_ids, output_ids)
        ]
        return self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

    @torch.no_grad()
    def post_process_description(self, description: dict[str, Any]) -> dict[str, Any]:
        original = _normalize_description(description)
        messages = [
            {"role": "system", "content": DESCRIPTION_POSTPROCESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": DESCRIPTION_POSTPROCESS_USER_PROMPT.format(
                    description_json=json.dumps(description, ensure_ascii=False, indent=2)
                ),
            },
        ]
        response = self._generate_from_messages(messages, max_new_tokens=512)
        try:
            cleaned = _normalize_description(_extract_json_object(response))
        except ValueError as exc:
            print(
                "Warning: failed to parse description post-processing response; "
                f"keeping original VLM description. Model output excerpt: {response[:300]!r}. Error: {exc}"
            )
            return original

        if cleaned["scene_layout"].lower() == "unclear" and original["scene_layout"].lower() != "unclear":
            cleaned["scene_layout"] = original["scene_layout"]
        if not cleaned["background_elements"] and original["background_elements"]:
            cleaned["background_elements"] = original["background_elements"]
        if cleaned["subject_count"] == 0 and original["subject_count"] > 0:
            cleaned["subject_count"] = original["subject_count"]
            cleaned["subjects"] = original["subjects"]
        if cleaned["camera_motion"].lower() == "the camera motion is unclear." and original["camera_motion"].lower() != "the camera motion is unclear.":
            cleaned["camera_motion"] = original["camera_motion"]
        return cleaned

    @torch.no_grad()
    def fuse(self, prompt: str, description: dict[str, Any]) -> str:
        normalized_description = _normalize_description(description)
        control_guidance = _format_control_guidance(normalized_description)
        messages = [
            {"role": "system", "content": FUSION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "User prompt:\n"
                    f"{prompt}\n\n"
                    "Control-video description JSON:\n"
                    f"{json.dumps(normalized_description, ensure_ascii=False, indent=2)}\n\n"
                    "Mandatory natural-language ending that must appear at the end of the final enhanced prompt:\n"
                    f"{control_guidance}"
                ),
            },
        ]
        response = self._generate_from_messages(messages, max_new_tokens=384)
        fused_prompt = _clean_fused_prompt(response)
        return _ensure_control_guidance_suffix(fused_prompt or prompt, control_guidance)

    def close(self) -> None:
        tokenizer = self.tokenizer
        model = self.model
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        _release_model(model, tokenizer)


def _sample_frame_paths(
    video_path: Path,
    sample_count: int,
    frame_cache_dir: Path,
    height: int,
    width: int,
) -> list[Path]:
    video_stats = video_path.stat()
    video_key = _stable_hash(
        {
            "path": str(video_path.resolve()),
            "mtime_ns": video_stats.st_mtime_ns,
            "size": video_stats.st_size,
            "sample_count": sample_count,
            "height": height,
            "width": width,
        }
    )
    sample_dir = frame_cache_dir / video_key
    sample_dir.mkdir(parents=True, exist_ok=True)

    video = VideoData(str(video_path), height=height, width=width)
    indices = _sample_frame_indices(len(video), sample_count)
    frame_paths: list[Path] = []
    for slot, frame_index in enumerate(indices):
        frame_path = sample_dir / f"frame_{slot:02d}_{frame_index:03d}.jpg"
        if not frame_path.exists():
            video[frame_index].save(frame_path, format="JPEG", quality=90)
        frame_paths.append(frame_path)
    return frame_paths


def prepare_prompt_enhancement(
    prompts: list[str],
    control_video_paths: list[Path],
    *,
    output_dir: str | Path,
    cache_dir: str | Path,
    height: int,
    width: int,
    sample_frames: int,
    camera_sample_frames: int,
    vlm_model_id: str,
    llm_model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    rank: int = 0,
) -> PromptEnhancementArtifacts:
    if rank != 0:
        raise ValueError("Prompt enhancement preprocessing must be executed on rank 0 only.")

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_root / "prompt_enhancement"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    frame_cache_dir = cache_root / "sampled_frames"
    frame_cache_dir.mkdir(parents=True, exist_ok=True)

    raw_description_cache_path = cache_root / "video_descriptions_vlm_cache.json"
    description_cache_path = cache_root / "video_descriptions_cache.json"
    fusion_cache_path = cache_root / "enhanced_prompts_cache.json"
    descriptions_path = artifacts_dir / "video_descriptions.json"
    prompts_path = artifacts_dir / "enhanced_prompts.jsonl"
    raw_description_cache = _load_json_file(raw_description_cache_path, {})
    description_cache = _load_json_file(description_cache_path, {})
    fusion_cache = _load_json_file(fusion_cache_path, {})

    raw_video_descriptions: dict[int, dict[str, Any]] = {}
    video_descriptions: dict[int, dict[str, Any]] = {}
    description_records: list[dict[str, Any]] = []
    raw_description_cache_hits = 0
    cleaned_description_cache_hits = 0

    describer = QwenControlVideoDescriber(
        model_id=vlm_model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    description_iterator = None
    try:
        description_iterator = tqdm(
            list(enumerate(control_video_paths)),
            desc="Prompt enhancement: describing control videos",
            unit="video",
        )
        for control_video_id, control_video_path in description_iterator:
            video_path = Path(control_video_path).resolve()
            frame_paths = _sample_frame_paths(
                video_path=video_path,
                sample_count=sample_frames,
                frame_cache_dir=frame_cache_dir,
                height=height,
                width=width,
            )
            camera_frame_paths = _sample_frame_paths(
                video_path=video_path,
                sample_count=camera_sample_frames,
                frame_cache_dir=frame_cache_dir,
                height=height,
                width=width,
            )
            raw_cache_key = _stable_hash(
                {
                    "version": DESCRIPTION_PROMPT_VERSION,
                    "video_path": str(video_path),
                    "video_mtime_ns": video_path.stat().st_mtime_ns,
                    "video_size": video_path.stat().st_size,
                    "sample_frames": sample_frames,
                    "camera_sample_frames": camera_sample_frames,
                    "height": height,
                    "width": width,
                    "frame_paths": [str(path) for path in frame_paths],
                    "camera_frame_paths": [str(path) for path in camera_frame_paths],
                    "vlm_model_id": vlm_model_id,
                }
            )
            raw_description = raw_description_cache.get(raw_cache_key)
            if raw_description is None:
                layout_subjects = describer.describe_layout_and_subjects(frame_paths)
                camera_motion = describer.describe_camera_motion(camera_frame_paths)
                raw_description = {
                    "scene_layout": layout_subjects["scene_layout"],
                    "background_elements": layout_subjects["background_elements"],
                    "subject_count": layout_subjects["subject_count"],
                    "subjects": layout_subjects["subjects"],
                    "camera_motion": camera_motion,
                }
                raw_description_cache[raw_cache_key] = raw_description
                cache_state = "computed"
            else:
                raw_description_cache_hits += 1
                cache_state = "cached"
            raw_video_descriptions[control_video_id] = _normalize_description(raw_description)
            description_iterator.set_postfix_str(cache_state)
    finally:
        if description_iterator is not None:
            description_iterator.close()
        describer.close()

    raw_description_cache_path.write_text(
        json.dumps(raw_description_cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    llm = QwenPromptEnhancementLLM(
        model_id=llm_model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    try:
        cleanup_iterator = None
        try:
            cleanup_iterator = tqdm(
                list(enumerate(control_video_paths)),
                desc="Prompt enhancement: cleaning video descriptions",
                unit="video",
            )
            for control_video_id, control_video_path in cleanup_iterator:
                video_path = Path(control_video_path).resolve()
                raw_description = raw_video_descriptions[control_video_id]
                frame_paths = _sample_frame_paths(
                    video_path=video_path,
                    sample_count=sample_frames,
                    frame_cache_dir=frame_cache_dir,
                    height=height,
                    width=width,
                )
                camera_frame_paths = _sample_frame_paths(
                    video_path=video_path,
                    sample_count=camera_sample_frames,
                    frame_cache_dir=frame_cache_dir,
                    height=height,
                    width=width,
                )
                clean_cache_key = _stable_hash(
                    {
                        "version": DESCRIPTION_POSTPROCESS_PROMPT_VERSION,
                        "raw_description_hash": _stable_hash(raw_description),
                        "llm_model_id": llm_model_id,
                    }
                )
                cached_description = description_cache.get(clean_cache_key)
                if cached_description is None:
                    cached_description = llm.post_process_description(raw_description)
                    description_cache[clean_cache_key] = cached_description
                    cache_state = "computed"
                else:
                    cleaned_description_cache_hits += 1
                    cache_state = "cached"
                normalized_description = _normalize_description(cached_description)
                video_descriptions[control_video_id] = normalized_description
                description_records.append(
                    {
                        "control_video_id": control_video_id,
                        "control_video_path": str(video_path),
                        "sampled_frames": [str(path) for path in frame_paths],
                        "camera_sampled_frames": [str(path) for path in camera_frame_paths],
                        "raw_description": raw_description,
                        "description": normalized_description,
                    }
                )
                cleanup_iterator.set_postfix_str(cache_state)
        finally:
            if cleanup_iterator is not None:
                cleanup_iterator.close()

        description_cache_path.write_text(
            json.dumps(description_cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        descriptions_path.write_text(
            json.dumps(description_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Prompt enhancement: saved video descriptions to {descriptions_path}")

        prompt_overrides: dict[tuple[int, int], str] = {}
        enhanced_prompt_records: list[dict[str, Any]] = []
        fusion_cache_hits = 0

        fusion_iterator = None
        try:
            fusion_total = len(video_descriptions) * len(prompts)
            fusion_iterator = tqdm(
                total=fusion_total,
                desc="Prompt enhancement: fusing prompts",
                unit="prompt",
            )
            for control_video_id, description in video_descriptions.items():
                description_hash = _stable_hash(description)
                for prompt_id, prompt in enumerate(prompts):
                    cache_key = _stable_hash(
                        {
                            "version": FUSION_PROMPT_VERSION,
                            "prompt": prompt,
                            "description_hash": description_hash,
                            "llm_model_id": llm_model_id,
                        }
                    )
                    fused_prompt = fusion_cache.get(cache_key)
                    if fused_prompt is None:
                        fused_prompt = llm.fuse(prompt, description)
                        fusion_cache[cache_key] = fused_prompt
                        cache_state = "computed"
                    else:
                        fusion_cache_hits += 1
                        cache_state = "cached"
                    prompt_overrides[(prompt_id, control_video_id)] = fused_prompt
                    enhanced_prompt_records.append(
                        {
                            "prompt_id": prompt_id,
                            "control_video_id": control_video_id,
                            "raw_prompt": prompt,
                            "enhanced_prompt": fused_prompt,
                        }
                    )
                    fusion_iterator.update(1)
                    fusion_iterator.set_postfix_str(cache_state)
        finally:
            if fusion_iterator is not None:
                fusion_iterator.close()

        print(
            "Prompt enhancement summary: "
            f"raw video descriptions {len(control_video_paths) - raw_description_cache_hits} computed / {raw_description_cache_hits} cached, "
            f"cleaned descriptions {len(control_video_paths) - cleaned_description_cache_hits} computed / {cleaned_description_cache_hits} cached, "
            f"fused prompts {len(enhanced_prompt_records) - fusion_cache_hits} computed / {fusion_cache_hits} cached."
        )

        fusion_cache_path.write_text(
            json.dumps(fusion_cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with prompts_path.open("w", encoding="utf-8") as handle:
            for record in enhanced_prompt_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        (artifacts_dir / "settings.json").write_text(
            json.dumps(
                {
                    "prompt_enhancement_mode": "enhanced",
                    "prompt_enhancement_num_frames": sample_frames,
                    "prompt_enhancement_camera_num_frames": camera_sample_frames,
                    "prompt_enhancement_vlm_model_id": vlm_model_id,
                    "prompt_enhancement_llm_model_id": llm_model_id,
                    "prompt_enhancement_description_postprocess": True,
                    "prompt_enhancement_description_schema": "scene_background_subject_count_camera_v3",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return PromptEnhancementArtifacts(
            prompt_overrides=prompt_overrides,
            control_video_descriptions=video_descriptions,
            descriptions_path=descriptions_path,
            prompts_path=prompts_path,
        )
    finally:
        llm.close()


def load_prompt_enhancement_artifacts(output_dir: str | Path) -> PromptEnhancementArtifacts:
    artifacts_dir = Path(output_dir).resolve() / "prompt_enhancement"
    descriptions_path = artifacts_dir / "video_descriptions.json"
    prompts_path = artifacts_dir / "enhanced_prompts.jsonl"
    if not descriptions_path.exists() or not prompts_path.exists():
        raise FileNotFoundError(
            f"Prompt enhancement artifacts were not found under {artifacts_dir}. "
            "Expected `video_descriptions.json` and `enhanced_prompts.jsonl`."
        )

    description_records = _load_json_file(descriptions_path, [])
    control_video_descriptions: dict[int, dict[str, Any]] = {}
    for record in description_records:
        control_video_descriptions[int(record["control_video_id"])] = _normalize_description(record["description"])

    prompt_overrides: dict[tuple[int, int], str] = {}
    for line in prompts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        prompt_overrides[(int(record["prompt_id"]), int(record["control_video_id"]))] = record["enhanced_prompt"]

    return PromptEnhancementArtifacts(
        prompt_overrides=prompt_overrides,
        control_video_descriptions=control_video_descriptions,
        descriptions_path=descriptions_path,
        prompts_path=prompts_path,
    )
