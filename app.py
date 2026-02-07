import json
import re
from pathlib import Path
from datetime import datetime
import time
from typing import List, Optional, Tuple, Dict, Any
import subprocess
import tempfile
from collections import Counter

from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from executor import RemoteConfig, RemoteMacExecutor, LocalMacExecutor
from actions import Action, ACTION_TYPES, PLANNER_ACTION_TYPES, applescript_for_action
from shortcuts import SHORTCUTS
from apikey_pool import ApiKeyPool
from shortcuts import resolve_shortcut

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
SCREEN_DIR = BASE_DIR / "static" / "screenshots"
SCREEN_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_DIR = BASE_DIR / "static" / "annotations"
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = BASE_DIR / "static" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LAST_SCREEN_PATH: Optional[Path] = None
LAST_SCREEN_SIZE: Optional[Tuple[int, int]] = None
LAST_DISAMBIGUATION: Optional[Dict[str, Any]] = None
LAST_OCCLUSION: Optional[Dict[str, Any]] = None
LAST_TAB_URLS: Optional[List[str]] = None
LAST_NEW_TAB_URL: Optional[str] = None
API_KEY_POOL: Optional[ApiKeyPool] = None
API_KEY_POOL_KEYS: Optional[List[str]] = None

def load_config() -> RemoteConfig:
    cfg_path = BASE_DIR / "config.json"
    if not cfg_path.exists():
        raise RuntimeError("Missing config.json. Copy config.example.json to config.json and edit.")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    return RemoteConfig(
        host=raw["host"],
        user=raw["user"],
        remote_tmp_screen_path=raw.get("remote_tmp_screen_path", "/tmp/agent_screen.png"),
        ssh_options=raw.get("ssh_options", []),
        rate_limit_seconds=float(raw.get("rate_limit_seconds", 5)),
    )

cfg = load_config()
executor = RemoteMacExecutor(cfg)
local_executor = LocalMacExecutor(rate_limit_seconds=cfg.rate_limit_seconds)


def select_executor(mode: Optional[str]):
    # Hackathon mode: remote(SSH) execution is disabled.
    # Keep mode field backward-compatible by mapping "remote" to local executor.
    if mode is None or mode == "" or mode == "local" or mode == "remote":
        return local_executor
    raise ValueError(f"Invalid executor mode: {mode}")


def set_api_keys(api_keys: Optional[List[str]]):
    if not api_keys:
        return
    cleaned = ApiKeyPool._normalize_keys(api_keys)
    global API_KEY_POOL, API_KEY_POOL_KEYS
    if API_KEY_POOL is None:
        API_KEY_POOL = ApiKeyPool(keys=cleaned)
        API_KEY_POOL_KEYS = cleaned
        return
    if API_KEY_POOL_KEYS != cleaned:
        API_KEY_POOL.update_keys(cleaned)
        API_KEY_POOL_KEYS = cleaned


def get_gemini_client(api_keys: Optional[List[str]] = None):
    if api_keys is not None:
        set_api_keys(api_keys)
    if API_KEY_POOL is None:
        raise RuntimeError("API keys are missing. Please enter at least one API key in the UI.")
    return API_KEY_POOL.get_client()


def sanitize_log_id(log_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", log_id or "").strip("_")
    if not safe:
        safe = "run"
    return safe[:64]


def new_log_path(log_id: Optional[str] = None) -> Path:
    if log_id:
        safe_id = sanitize_log_id(log_id)
        return LOG_DIR / f"{safe_id}.txt"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return LOG_DIR / f"run_{ts}.txt"


def ensure_log_header(path: Path, mode: Optional[str]):
    if path.exists():
        return
    header = [
        f"Run Actions Sequence Log",
        f"Started: {datetime.now().isoformat()}",
        f"Mode: {mode or 'remote'}",
        "-" * 40,
    ]
    path.write_text("\n".join(header) + "\n", encoding="utf-8")


def append_log_line(path: Path, line: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def show_completion_dialog(exec_impl, ok: bool, log_name: str):
    status = "OK" if ok else "FAILED"
    msg = f"Run Actions Sequence finished.\\nStatus: {status}\\nLog: {log_name}"
    script = f'''
tell application "System Events"
  display dialog "{msg}" buttons {{"OK"}} default button 1
end tell
'''
    try:
        exec_impl.run_osascript(script, timeout=10, rate_limit_seconds=0)
    except Exception:
        pass

app = FastAPI(title="Mac Action Orchestrator")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


def new_screen_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return SCREEN_DIR / f"{ts}.png"


def read_png_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with path.open("rb") as f:
            header = f.read(24)
        if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height
    except Exception:
        return None


def locate_click_with_gemini(
    image_path: Path,
    target: str,
    model: str,
    screen_size: Optional[Tuple[int, int]] = None,
    api_keys: Optional[List[str]] = None,
) -> Tuple[int, int]:
    from google.genai import types
    client = get_gemini_client(api_keys=api_keys)
    image_size = read_png_size(image_path)
    size_hint = f"Screenshot size: {image_size[0]}x{image_size[1]} pixels." if image_size else ""
    prompt = (
        "You are given a screenshot of a macOS desktop.\n"
        f"{size_hint}\n"
        "Find the pixel coordinates (x,y) of the UI element described below. "
        "Return the center point of the target element. "
        "Coordinates are in pixels with (0,0) at the top-left of the image.\n"
        "You must call the function set_click_coordinates with integer x and y.\n"
        f"Target: {target}"
    )
    function_decl = {
        "name": "set_click_coordinates",
        "description": "Return the target click coordinates in pixels.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Pixel x coordinate."},
                "y": {"type": "integer", "description": "Pixel y coordinate."},
            },
            "required": ["x", "y"],
        },
    }
    tools = types.Tool(function_declarations=[function_decl])
    config = types.GenerateContentConfig(tools=[tools])
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png"),
    ]
    content = types.Content(role="user", parts=parts)
    response = client.models.generate_content(
        model=model,
        contents=[content],
        config=config,
    )
    function_call = None
    parts = []
    try:
        parts = response.candidates[0].content.parts or []
    except Exception:
        parts = []
    for part in parts:
        if getattr(part, "function_call", None):
            function_call = part.function_call
            break
    if not function_call or not isinstance(function_call.args, dict):
        raise ValueError("Gemini did not return function call coordinates.")
    x = function_call.args.get("x")
    y = function_call.args.get("y")
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError("Gemini returned invalid x/y.")
    original_coords = (x, y)
    if image_size:
        iw, ih = image_size
        if not (0 <= x < iw and 0 <= y < ih):
            raise ValueError(f"Gemini returned out-of-bounds coords: {(x, y)} for {iw}x{ih}.")
        if screen_size:
            sw, sh = screen_size
            if (sw, sh) != (iw, ih):
                x = int(round(x * (sw / iw)))
                y = int(round(y * (sh / ih)))
    print(f"[locate_click_with_gemini] image_size={image_size} screen_size={screen_size} "
          f"coords_in={original_coords} coords_out={(x, y)}")
    if screen_size:
        sw, sh = screen_size
        if not (0 <= x < sw and 0 <= y < sh):
            raise ValueError(f"Scaled coords out-of-bounds: {(x, y)} for {sw}x{sh}.")
    return x, y


def extract_response_text(response) -> Optional[str]:
    try:
        return response.text
    except Exception:
        pass
    texts: List[str] = []
    try:
        for cand in getattr(response, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", []) or []:
                txt = getattr(part, "text", None)
                if txt:
                    texts.append(txt)
    except Exception:
        return None
    return "\n".join(texts) if texts else None


def resolve_local_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return p


def vision_ocr_tokens(path: Path) -> List[Dict[str, Any]]:
    if subprocess.run(["/usr/bin/which", "swift"], capture_output=True, text=True).returncode != 0:
        raise RuntimeError("Swift is not available. Install Xcode Command Line Tools: xcode-select --install")
    swift_code = r'''
import Foundation
import Vision
import AppKit

func toCGImage(_ nsImage: NSImage) -> CGImage? {
    var rect = CGRect(origin: .zero, size: nsImage.size)
    return nsImage.cgImage(forProposedRect: &rect, context: nil, hints: nil)
}

let args = CommandLine.arguments
if args.count < 2 {
    print("{\"ok\":false,\"error\":\"missing path\"}")
    exit(1)
}
let imagePath = args[1]
guard let nsImage = NSImage(contentsOfFile: imagePath) else {
    print("{\"ok\":false,\"error\":\"cannot load image\"}")
    exit(1)
}
guard let cgImage = toCGImage(nsImage) else {
    print("{\"ok\":false,\"error\":\"cannot create CGImage\"}")
    exit(1)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    print("{\"ok\":false,\"error\":\"vision request failed\"}")
    exit(1)
}

var tokens: [[String: Any]] = []
let results = request.results as? [VNRecognizedTextObservation] ?? []
let width = CGFloat(cgImage.width)
let height = CGFloat(cgImage.height)
for obs in results {
    guard let top = obs.topCandidates(1).first else { continue }
    let text = top.string
    let box = obs.boundingBox
    let left = Int(round(box.minX * width))
    let right = Int(round(box.maxX * width))
    let topY = Int(round((1.0 - box.maxY) * height))
    let bottomY = Int(round((1.0 - box.minY) * height))
    let w = max(0, right - left)
    let h = max(0, bottomY - topY)
    tokens.append([
        "text": text,
        "left": left,
        "top": topY,
        "width": w,
        "height": h
    ])
}

let output: [String: Any] = ["ok": true, "tokens": tokens]
if let json = try? JSONSerialization.data(withJSONObject: output, options: []) {
    print(String(data: json, encoding: .utf8) ?? "")
} else {
    print("{\"ok\":false,\"error\":\"json encode failed\"}")
    exit(1)
}
'''
    with tempfile.NamedTemporaryFile("w", suffix=".swift", delete=False) as f:
        f.write(swift_code)
        swift_path = f.name
    proc = subprocess.run(
        ["swift", swift_path, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Vision OCR failed: {proc.stderr.strip() or proc.stdout.strip()}")
    try:
        payload = json.loads(proc.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"Vision OCR returned invalid JSON: {e}")
    if not payload.get("ok"):
        raise RuntimeError(f"Vision OCR error: {payload.get('error')}")
    tokens = payload.get("tokens") or []
    return tokens


def ocr_find_text_center(path: Path, query: str) -> Tuple[int, int]:
    tokens = vision_ocr_tokens(path)
    if not tokens:
        raise ValueError("OCR returned no text tokens.")
    q = query.strip().lower()
    if not q:
        raise ValueError("Query text is empty.")
    # Try exact token match first
    for t in tokens:
        if str(t.get("text", "")).strip().lower() == q:
            cx = int(t.get("left", 0)) + int(t.get("width", 0)) // 2
            cy = int(t.get("top", 0)) + int(t.get("height", 0)) // 2
            return cx, cy
    # Try multi-token phrase match
    texts = [str(t.get("text", "")).strip().lower() for t in tokens]
    n = len(texts)
    for i in range(n):
        if not texts[i]:
            continue
        phrase = texts[i]
        for j in range(i + 1, min(n, i + 8)):
            phrase = phrase + " " + texts[j]
            if phrase == q:
                left = min(int(tokens[k].get("left", 0)) for k in range(i, j + 1))
                top = min(int(tokens[k].get("top", 0)) for k in range(i, j + 1))
                right = max(int(tokens[k].get("left", 0)) + int(tokens[k].get("width", 0)) for k in range(i, j + 1))
                bottom = max(int(tokens[k].get("top", 0)) + int(tokens[k].get("height", 0)) for k in range(i, j + 1))
                return (left + right) // 2, (top + bottom) // 2
    # Try substring match within a token
    for t in tokens:
        if q in str(t.get("text", "")).strip().lower():
            cx = int(t.get("left", 0)) + int(t.get("width", 0)) // 2
            cy = int(t.get("top", 0)) + int(t.get("height", 0)) // 2
            return cx, cy
    raise ValueError("Text not found by OCR.")


def ocr_find_text_candidates(path: Path, query: str) -> List[Dict[str, int]]:
    tokens = vision_ocr_tokens(path)
    if not tokens:
        raise ValueError("OCR returned no text tokens.")
    q = query.strip().lower()
    if not q:
        raise ValueError("Query text is empty.")

    def add_box(boxes: List[Dict[str, int]], left: int, top: int, width: int, height: int):
        if width <= 0 or height <= 0:
            return
        boxes.append({"left": int(left), "top": int(top), "width": int(width), "height": int(height)})

    # Exact token matches
    exact = []
    for t in tokens:
        if str(t.get("text", "")).strip().lower() == q:
            add_box(exact, t.get("left", 0), t.get("top", 0), t.get("width", 0), t.get("height", 0))
    if exact:
        return sorted(exact, key=lambda b: (b["top"], b["left"]))

    # Multi-token phrase match
    texts = [str(t.get("text", "")).strip().lower() for t in tokens]
    n = len(texts)
    phrase_boxes: List[Dict[str, int]] = []
    for i in range(n):
        if not texts[i]:
            continue
        phrase = texts[i]
        for j in range(i + 1, min(n, i + 8)):
            phrase = phrase + " " + texts[j]
            if phrase == q:
                left = min(int(tokens[k].get("left", 0)) for k in range(i, j + 1))
                top = min(int(tokens[k].get("top", 0)) for k in range(i, j + 1))
                right = max(int(tokens[k].get("left", 0)) + int(tokens[k].get("width", 0)) for k in range(i, j + 1))
                bottom = max(int(tokens[k].get("top", 0)) + int(tokens[k].get("height", 0)) for k in range(i, j + 1))
                add_box(phrase_boxes, left, top, right - left, bottom - top)
    if phrase_boxes:
        return sorted(phrase_boxes, key=lambda b: (b["top"], b["left"]))

    # Substring match
    partial = []
    for t in tokens:
        if q in str(t.get("text", "")).strip().lower():
            add_box(partial, t.get("left", 0), t.get("top", 0), t.get("width", 0), t.get("height", 0))
    if partial:
        return sorted(partial, key=lambda b: (b["top"], b["left"]))

    return []


def ocr_text_array(path: Path) -> List[str]:
    tokens = vision_ocr_tokens(path)
    texts: List[str] = []
    for t in tokens:
        txt = str(t.get("text", "")).strip()
        if txt:
            texts.append(txt)
    return texts


def get_chrome_tab_urls(executor_obj=None) -> Optional[List[str]]:
    exec_impl = executor_obj or executor
    script = r'''
tell application "Google Chrome"
  if not (exists front window) then return ""
  set urls to URL of tabs of front window
  set AppleScript's text item delimiters to "\n"
  return urls as text
end tell
'''
    rc, out, err = exec_impl.run_osascript(script, timeout=10, rate_limit_seconds=0)
    if rc != 0:
        return None
    if not out:
        return []
    return [line for line in out.split("\n") if line.strip()]


def update_latest_tab_url(
    previous_urls: Optional[List[str]] = None,
    executor_obj=None,
) -> Optional[str]:
    global LAST_TAB_URLS, LAST_NEW_TAB_URL
    urls = get_chrome_tab_urls(executor_obj=executor_obj)
    if urls is None:
        return LAST_NEW_TAB_URL
    base_urls = previous_urls if previous_urls is not None else LAST_TAB_URLS
    if base_urls is None:
        LAST_TAB_URLS = urls
        return LAST_NEW_TAB_URL
    old_counts = Counter(base_urls)
    new_urls: List[str] = []
    for u in urls:
        if old_counts.get(u, 0) > 0:
            old_counts[u] -= 1
        else:
            new_urls.append(u)
    if new_urls:
        LAST_NEW_TAB_URL = new_urls[-1]
    LAST_TAB_URLS = urls
    return LAST_NEW_TAB_URL


def build_plan_again_prompt(prompt: str, ocr_tokens: List[str]) -> str:
    payload = json.dumps(ocr_tokens, ensure_ascii=False)
    return (
        f"{prompt}\n\n"
        "OCR_TOKENS (click_at target must be exactly one of these items):\n"
        f"{payload}\n"
    )


def extract_prompt_keywords(prompt: str) -> List[str]:
    text = (prompt or "").lower()
    words = re.findall(r"[a-z0-9]+", text)
    stopwords = {
        "the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "with", "then",
        "at", "by", "from", "is", "are", "be", "as", "it", "this", "that", "these",
        "those", "please", "click", "open", "select", "choose", "pick", "press",
        "type", "enter", "search", "go", "navigate", "tab", "page", "result", "results",
        "lowest", "bottom", "bottommost", "top", "first", "second", "third",
    }
    keywords: List[str] = []
    seen = set()
    for w in words:
        if len(w) < 3 or w in stopwords:
            continue
        if w not in seen:
            seen.add(w)
            keywords.append(w)
    return keywords


def filter_ocr_tokens_by_prompt(ocr_tokens: List[str], prompt: str) -> List[str]:
    keywords = extract_prompt_keywords(prompt)
    def unique(tokens: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out
    if not keywords:
        return unique(ocr_tokens)
    filtered = []
    seen = set()
    for token in ocr_tokens:
        t = token.lower()
        if any(k in t for k in keywords):
            if token not in seen:
                seen.add(token)
                filtered.append(token)
    return filtered or unique(ocr_tokens)


def annotate_targets(image_path: Path, boxes: List[Dict[str, int]]) -> Path:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as e:
        raise RuntimeError(f"Missing dependency: {e}. Run: pip install pillow")
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for idx, box in enumerate(boxes, start=1):
        left = int(box["left"])
        top = int(box["top"])
        right = left + int(box["width"])
        bottom = top + int(box["height"])
        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0, 255), width=4)
        label = str(idx)
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = font.getsize(label)
        pad = 6
        label_box = [left, max(0, top - text_h - pad * 2 - 2), left + text_w + pad * 2, top + 2]
        draw.rectangle(label_box, fill=(255, 0, 0, 230))
        draw.text((left + pad, label_box[1] + pad), label, fill=(255, 255, 255, 255), font=font)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = ANNOTATION_DIR / f"targets_{ts}.png"
    img.convert("RGB").save(out_path, format="PNG")
    return out_path


def choose_target_index_with_gemini(
    image_path: Path,
    target: str,
    count: int,
    model: str = "gemini-3-flash-preview",
    api_keys: Optional[List[str]] = None,
) -> int:
    from google.genai import types
    client = get_gemini_client(api_keys=api_keys)
    prompt = (
        "You are given a screenshot with numbered boxes marking candidate targets.\n"
        "Choose the correct number for the target described below.\n"
        "Return ONLY a function call: choose_number.\n"
        "The argument 'index' must be an integer within the visible numbered boxes.\n"
        f"Target description: {target}\n"
        f"Valid range: 1 to {count}\n"
    )
    function_decl = {
        "name": "choose_number",
        "description": "Choose the correct target number from the annotated image.",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Chosen target number."},
            },
            "required": ["index"],
        },
    }
    tools = types.Tool(function_declarations=[function_decl])
    config = types.GenerateContentConfig(tools=[tools])
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png"),
    ]
    content = types.Content(role="user", parts=parts)
    response = client.models.generate_content(
        model=model,
        contents=[content],
        config=config,
    )
    function_call = None
    try:
        parts = response.candidates[0].content.parts or []
    except Exception:
        parts = []
    for part in parts:
        if getattr(part, "function_call", None):
            function_call = part.function_call
            break
    if not function_call or not isinstance(function_call.args, dict):
        raise ValueError("Gemini did not return a valid choose_number function call.")
    index = function_call.args.get("index")
    if not isinstance(index, int):
        raise ValueError("Gemini returned invalid index.")
    if index < 1 or index > count:
        raise ValueError(f"Gemini returned out-of-range index: {index}.")
    return index


def expand_actions(actions: List[Action]) -> List[Action]:
    expanded: List[Action] = []
    for action in actions:
        if action.type == "shortcut":
            if not action.name:
                raise ValueError("shortcut requires name")
            steps = resolve_shortcut(action.name)
            expanded.extend(steps)
        elif action.type == "open_url":
            if not action.url:
                raise ValueError("open_url requires url")
            expanded.extend([
                Action(type="open_chrome", url=action.url),
                Action(type="wait", seconds=1.0),
            ])
        else:
            expanded.append(action)
    return expanded


@app.post("/api/capture")
def capture():
    try:
        local_path = new_screen_path()
        executor.capture_screen_to_local(local_path)
        global LAST_SCREEN_PATH
        LAST_SCREEN_PATH = local_path
        url = f"/static/screenshots/{local_path.name}"
        return JSONResponse({"ok": True, "screenshot_url": url})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


class ActionsRequest(BaseModel):
    actions: List[Action]
    cooldown_seconds: Optional[float] = None
    max_plan_again: Optional[int] = None
    pause_on_plan_again: Optional[bool] = None
    context_prompt: Optional[str] = None
    mode: Optional[str] = None
    log_id: Optional[str] = None
    sequence_run: Optional[bool] = None
    api_keys: Optional[List[str]] = None


class ActionsSequenceRequest(BaseModel):
    batches: List[List[Action]]
    cooldown_seconds: Optional[float] = None
    max_plan_again: Optional[int] = None
    pause_on_plan_again: Optional[bool] = None
    context_prompt: Optional[str] = None
    mode: Optional[str] = None
    log_id: Optional[str] = None
    api_keys: Optional[List[str]] = None


class FindXYRequest(BaseModel):
    path: str
    text: str


class AnnotateTargetsRequest(BaseModel):
    path: str
    text: str


class ChooseTargetRequest(BaseModel):
    prompt: Optional[str] = None
    api_keys: Optional[List[str]] = None


class ExecuteClickRequest(BaseModel):
    index: int


class ResolveOcclusionRequest(BaseModel):
    prompt: Optional[str] = None
    api_keys: Optional[List[str]] = None


class ExecuteOcclusionRequest(BaseModel):
    mode: str


@app.post("/api/screen_size")
def screen_size():
    try:
        # Always use local screen size in hackathon mode.
        size, stdout, stderr, rc = local_executor.get_screen_size_debug()
        if not size:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Failed to get screen size.",
                    "returncode": rc,
                    "stdout": stdout,
                    "stderr": stderr,
                },
                status_code=500,
            )
        global LAST_SCREEN_SIZE
        LAST_SCREEN_SIZE = size
        return JSONResponse({"ok": True, "width": size[0], "height": size[1], "stdout": stdout})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/find_xy")
def find_xy(req: FindXYRequest):
    try:
        path = resolve_local_path(req.path)
        if not path.exists():
            return JSONResponse({"ok": False, "error": f"File not found: {path}"}, status_code=400)
        x, y = ocr_find_text_center(path, req.text)
        return JSONResponse({"ok": True, "x": x, "y": y, "path": str(path)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/annotate_targets")
def annotate_targets_api(req: AnnotateTargetsRequest):
    try:
        path = resolve_local_path(req.path)
        if not path.exists():
            return JSONResponse({"ok": False, "error": f"File not found: {path}"}, status_code=400)
        candidates = ocr_find_text_candidates(path, req.text)
        if not candidates:
            return JSONResponse({"ok": False, "error": "Text not found by OCR."}, status_code=404)
        annotated_path = annotate_targets(path, candidates)
        url = f"/static/annotations/{annotated_path.name}"
        return JSONResponse({"ok": True, "count": len(candidates), "annotated_url": url})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/choose_target")
def choose_target_api(req: ChooseTargetRequest):
    """
    Ask Gemini to choose a numbered target from the latest annotated screenshot.
    """
    global LAST_DISAMBIGUATION
    if not LAST_DISAMBIGUATION:
        return JSONResponse({"ok": False, "error": "No pending disambiguation."}, status_code=400)
    annotated_path = Path(LAST_DISAMBIGUATION.get("annotated_path", ""))
    if not annotated_path.exists():
        return JSONResponse({"ok": False, "error": "Annotated image not found."}, status_code=400)
    candidates = LAST_DISAMBIGUATION.get("candidates") or []
    if not candidates:
        return JSONResponse({"ok": False, "error": "No candidates available."}, status_code=400)
    target = LAST_DISAMBIGUATION.get("target") or ""
    context_prompt = req.prompt or LAST_DISAMBIGUATION.get("context_prompt") or ""
    if context_prompt and target and target.lower() not in context_prompt.lower():
        description = f"{context_prompt}\nTarget: {target}"
    else:
        description = context_prompt or target
    try:
        index = choose_target_index_with_gemini(
            annotated_path,
            description,
            len(candidates),
            api_keys=req.api_keys,
        )
        url = f"/static/annotations/{annotated_path.name}"
        return JSONResponse({
            "ok": True,
            "index": index,
            "count": len(candidates),
            "annotated_url": url,
            "prompt_used": description,
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/execute_click")
def execute_click(req: ExecuteClickRequest):
    """
    Execute click by chosen target index from the latest disambiguation.
    """
    global LAST_DISAMBIGUATION
    if not LAST_DISAMBIGUATION:
        return JSONResponse({"ok": False, "error": "No pending disambiguation."}, status_code=400)
    candidates = LAST_DISAMBIGUATION.get("candidates") or []
    if not candidates:
        return JSONResponse({"ok": False, "error": "No candidates available."}, status_code=400)
    index = req.index
    if index < 1 or index > len(candidates):
        return JSONResponse({"ok": False, "error": f"Index out of range: {index}."}, status_code=400)
    box = candidates[index - 1]
    x = int(box["left"]) + int(box["width"]) // 2
    y = int(box["top"]) + int(box["height"]) // 2
    image_size = LAST_DISAMBIGUATION.get("image_size")
    screen_size = LAST_DISAMBIGUATION.get("screen_size")
    if image_size and screen_size and tuple(screen_size) != tuple(image_size):
        iw, ih = image_size
        sw, sh = screen_size
        x = int(round(x * (sw / iw)))
        y = int(round(y * (sh / ih)))
    try:
        prev_tab_urls = get_chrome_tab_urls()
        action = Action(type="click_at", x=x, y=y)
        script = applescript_for_action(action)
        rc, out, err = executor.run_osascript(script, rate_limit_seconds=0)
        local_path = new_screen_path()
        try:
            executor.capture_screen_to_local(local_path, rate_limit_seconds=0)
            global LAST_SCREEN_PATH
            LAST_SCREEN_PATH = local_path
            url = f"/static/screenshots/{local_path.name}"
        except Exception as shot_err:
            url = None
            err = (err + f"\n[screenshot_error] {shot_err}").strip()
        return JSONResponse({
            "ok": (rc == 0),
            "returncode": rc,
            "stdout": out,
            "stderr": err,
            "action": action.model_dump(),
            "screenshot_url": url,
            "latest_tab_url": update_latest_tab_url(previous_urls=prev_tab_urls),
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/resolve_occlusion")
def resolve_occlusion(req: ResolveOcclusionRequest):
    """
    Ask Gemini to choose how to resolve occlusion (close_tab or close_tab_then_open_url).
    """
    global LAST_OCCLUSION
    if not LAST_OCCLUSION:
        return JSONResponse({"ok": False, "error": "No pending occlusion recovery."}, status_code=400)
    screen_path = Path(LAST_OCCLUSION.get("screen_path", ""))
    if not screen_path.exists():
        return JSONResponse({"ok": False, "error": "Occlusion screenshot not found."}, status_code=400)
    target = LAST_OCCLUSION.get("target") or ""
    occlusion_url = LAST_OCCLUSION.get("occlusion_url")
    modes = ["close_tab"]
    if occlusion_url:
        modes.append("close_tab_then_open_url")
    prompt = req.prompt or (
        "Target not found; likely occluded.\n"
        f"Click target: {target}\n"
        "Choose how to resolve occlusion. Do NOT click the target yet."
    )
    function_decls = [
        {
            "name": "resolve_occlusion",
            "description": "Resolve occluding tab before retrying click.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": modes},
                },
                "required": ["mode"],
            },
        }
    ]
    try:
        response = call_gemini_planner(
            prompt,
            model="gemini-3-flash-preview",
            image_paths=[screen_path],
            function_decls=function_decls,
            api_keys=req.api_keys,
        )
        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            return JSONResponse({"ok": False, "error": "No function call returned."}, status_code=500)
        call = tool_calls[0]
        if call.name != "resolve_occlusion" or not isinstance(call.args, dict):
            return JSONResponse({"ok": False, "error": "Invalid function call."}, status_code=500)
        mode = call.args.get("mode")
        if mode not in modes:
            return JSONResponse({"ok": False, "error": "Invalid mode returned."}, status_code=500)
        LAST_OCCLUSION["mode"] = mode
        return JSONResponse({"ok": True, "mode": mode, "modes": modes})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/execute_occlusion")
def execute_occlusion(req: ExecuteOcclusionRequest):
    """
    Execute the chosen occlusion resolution actions.
    """
    global LAST_OCCLUSION
    if not LAST_OCCLUSION:
        return JSONResponse({"ok": False, "error": "No pending occlusion recovery."}, status_code=400)
    occlusion_url = LAST_OCCLUSION.get("occlusion_url")
    mode = req.mode
    if mode not in {"close_tab", "close_tab_then_open_url"}:
        return JSONResponse({"ok": False, "error": "Invalid mode."}, status_code=400)
    actions = [Action(type="shortcut", name="close_tab")]
    if mode == "close_tab_then_open_url":
        if not occlusion_url:
            return JSONResponse({"ok": False, "error": "No URL available for open_url."}, status_code=400)
        actions.append(Action(type="open_url", url=occlusion_url))
    expanded = expand_actions(actions)
    results = []
    try:
        for act in expanded:
            if act.type == "wait":
                time.sleep(act.seconds or 0)
                results.append({
                    "ok": True,
                    "returncode": 0,
                    "stdout": "ok",
                    "stderr": "",
                    "action": act.model_dump(),
                })
                continue
            script = applescript_for_action(act)
            rc, out, err = executor.run_osascript(script, rate_limit_seconds=0)
            results.append({
                "ok": (rc == 0),
                "returncode": rc,
                "stdout": out,
                "stderr": err,
                "action": act.model_dump(),
            })
            if rc != 0:
                return JSONResponse({"ok": False, "results": results}, status_code=500)
        target = LAST_OCCLUSION.get("target")
        if not target:
            return JSONResponse({"ok": True, "results": results})
        prev_tab_urls = get_chrome_tab_urls()
        retry_action = Action(type="click_at", target=target)
        resolved, pause_resp, _used, reason_for_action = resolve_click_at_target(
            action=retry_action,
            cooldown_seconds=0,
            context_prompt=None,
            results=results,
            prev_tab_urls=prev_tab_urls,
            click_recovery_used=False,
        )
        if pause_resp is not None:
            return pause_resp
        retry_action = resolved
        script = applescript_for_action(retry_action)
        rc, out, err = executor.run_osascript(script, rate_limit_seconds=0)
        latest_tab_url = update_latest_tab_url(previous_urls=prev_tab_urls)
        retry_entry = {
            "ok": (rc == 0),
            "returncode": rc,
            "stdout": out,
            "stderr": err,
            "action": retry_action.model_dump(),
            "latest_tab_url": latest_tab_url,
        }
        if reason_for_action:
            retry_entry["reason"] = reason_for_action
        results.append(retry_entry)
        if rc != 0:
            return JSONResponse({"ok": False, "results": results, "latest_tab_url": latest_tab_url}, status_code=500)
        local_path = new_screen_path()
        try:
            executor.capture_screen_to_local(local_path, rate_limit_seconds=0)
            global LAST_SCREEN_PATH
            LAST_SCREEN_PATH = local_path
            url = f"/static/screenshots/{local_path.name}"
        except Exception:
            url = None
        return JSONResponse({
            "ok": True,
            "results": results,
            "screenshot_url": url,
            "latest_tab_url": latest_tab_url,
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "results": results}, status_code=500)


@app.post("/api/run_action")
def run_action(action: Action, cooldown_seconds: Optional[float] = None):
    """
    Execute one whitelisted action. Also captures a screenshot after execution for verification.
    """
    try:
        prev_tab_urls = get_chrome_tab_urls()
        script = applescript_for_action(action)
        rc, out, err = executor.run_osascript(script, rate_limit_seconds=cooldown_seconds)

        # After action, capture screen for verification (can be made optional)
        local_path = new_screen_path()
        try:
            executor.capture_screen_to_local(local_path, rate_limit_seconds=cooldown_seconds)
            global LAST_SCREEN_PATH
            LAST_SCREEN_PATH = local_path
            url = f"/static/screenshots/{local_path.name}"
        except Exception as shot_err:
            url = None
            err = (err + f"\n[screenshot_error] {shot_err}").strip()

        latest_tab_url = update_latest_tab_url(previous_urls=prev_tab_urls)
        return JSONResponse({
            "ok": (rc == 0),
            "returncode": rc,
            "stdout": out,
            "stderr": err,
            "action": action.model_dump(),
            "screenshot_url": url,
            "latest_tab_url": latest_tab_url,
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "action": action.model_dump()}, status_code=500)


@app.post("/api/run_actions")
def run_actions(req: ActionsRequest):
    """
    Execute multiple actions in order. Takes a screenshot after the sequence finishes.
    """
    results = []
    log_path: Optional[Path] = None
    exec_impl = None
    try:
        set_api_keys(req.api_keys)
        exec_impl = select_executor(req.mode)
        log_path = new_log_path(req.log_id)
        ensure_log_header(log_path, req.mode)
        log_url = f"/static/logs/{log_path.name}"
        expanded_actions = expand_actions(req.actions)
        global LAST_SCREEN_PATH
        plan_again_count = 0
        latest_tab_url: Optional[str] = None
        click_recovery_used = False
        idx = 0
        while idx < len(expanded_actions):
            prev_tab_urls = get_chrome_tab_urls(executor_obj=exec_impl)
            action = expanded_actions[idx]
            reason_for_action: Optional[str] = None
            if action.type == "click_at" and (action.x is None or action.y is None):
                if not action.target:
                    raise ValueError("click_at requires x/y or target.")
                resolved, pause_resp, click_recovery_used, reason_for_action = resolve_click_at_target(
                    action=action,
                    cooldown_seconds=req.cooldown_seconds,
                    context_prompt=req.context_prompt,
                    results=results,
                    prev_tab_urls=prev_tab_urls,
                    click_recovery_used=click_recovery_used,
                    executor_obj=exec_impl,
                )
                if pause_resp is not None:
                    if log_path:
                        append_log_line(log_path, "[PAUSED] click_at requires user input")
                        try:
                            payload = json.loads(pause_resp.body.decode("utf-8"))
                            payload["log_url"] = log_url
                            return JSONResponse(payload)
                        except Exception:
                            return pause_resp
                    return pause_resp
                action = resolved
                expanded_actions[idx] = action
            if action.type == "plan_again":
                if req.pause_on_plan_again:
                    if not action.prompt:
                        raise ValueError("plan_again requires prompt.")
                    local_path = new_screen_path()
                    exec_impl.capture_screen_to_local(local_path, rate_limit_seconds=req.cooldown_seconds)
                    LAST_SCREEN_PATH = local_path
                    url = f"/static/screenshots/{local_path.name}"
                    if log_path:
                        append_log_line(log_path, "[PAUSED] plan_again")
                        return JSONResponse({
                        "ok": True,
                        "paused": True,
                        "pause_reason": "plan_again",
                        "screenshot_url": url,
                        "action": action.model_dump(),
                        "results": results,
                        "latest_tab_url": update_latest_tab_url(previous_urls=prev_tab_urls, executor_obj=exec_impl),
                        "log_url": log_url,
                        **({"sequence_run": True} if req.sequence_run else {}),
                    })
                if req.max_plan_again is not None:
                    if req.max_plan_again < 0:
                        raise ValueError("max_plan_again must be >= 0")
                    if plan_again_count >= req.max_plan_again:
                        raise ValueError(f"plan_again limit reached ({req.max_plan_again}).")
                if not action.prompt:
                    raise ValueError("plan_again requires prompt.")
                plan_again_count += 1
                local_path = new_screen_path()
                exec_impl.capture_screen_to_local(local_path, rate_limit_seconds=req.cooldown_seconds)
                LAST_SCREEN_PATH = local_path
                ocr_tokens = ocr_text_array(local_path)
                allowed_targets = filter_ocr_tokens_by_prompt(ocr_tokens, action.prompt)
                model = "gemini-3-flash-preview"
                function_decls = build_action_function_declarations(
                    allowed_targets=allowed_targets,
                    include_plan_again=True,
                )
                response = call_gemini_planner(
                    action.prompt,
                    model=model,
                    image_paths=[local_path],
                    function_decls=function_decls,
                    api_keys=req.api_keys,
                )
                tool_calls = extract_tool_calls(response)
                batches, err, _raw_args = tool_calls_to_batches(
                    tool_calls,
                    prompt=action.prompt,
                    allowed_targets=allowed_targets,
                )
                if err:
                    raise ValueError(f"plan_again failed: {err}")
                new_actions = [Action.model_validate(batch[0]) for batch in batches]
                new_actions = expand_actions(new_actions)
                expanded_actions = expanded_actions[:idx] + new_actions + expanded_actions[idx + 1:]
                results.append({
                    "ok": True,
                    "returncode": 0,
                    "stdout": "plan_again",
                    "stderr": "",
                    "action": action.model_dump(),
                    "planned_actions": [a.model_dump() for a in new_actions],
                    "ocr_tokens": ocr_tokens,
                    "allowed_targets": allowed_targets,
                })
                if log_path:
                    append_log_line(log_path, json.dumps(results[-1], ensure_ascii=False))
                continue
            if action.type == "wait":
                if action.seconds is None:
                    raise ValueError("wait requires seconds")
                time.sleep(action.seconds)
                results.append({
                    "ok": True,
                    "returncode": 0,
                    "stdout": "ok",
                    "stderr": "",
                    "action": action.model_dump(),
                    "latest_tab_url": update_latest_tab_url(previous_urls=prev_tab_urls, executor_obj=exec_impl),
                    **({"reason": reason_for_action} if reason_for_action else {}),
                })
                if log_path:
                    append_log_line(log_path, json.dumps(results[-1], ensure_ascii=False))
                latest_tab_url = results[-1]["latest_tab_url"]
                idx += 1
                continue

            script = applescript_for_action(action)
            rc, out, err = exec_impl.run_osascript(script, rate_limit_seconds=0)
            latest_tab_url = update_latest_tab_url(previous_urls=prev_tab_urls, executor_obj=exec_impl)
            entry = {
                "ok": (rc == 0),
                "returncode": rc,
                "stdout": out,
                "stderr": err,
                "action": action.model_dump(),
                "latest_tab_url": latest_tab_url,
            }
            if reason_for_action:
                entry["reason"] = reason_for_action
            results.append(entry)
            if log_path:
                append_log_line(log_path, json.dumps(entry, ensure_ascii=False))
            if rc != 0:
                if log_path:
                    append_log_line(log_path, "[END] stopped_at=" + str(idx))
                show_completion_dialog(exec_impl, ok=False, log_name=log_path.name if log_path else "n/a")
                return JSONResponse({
                    "ok": False,
                    "stopped_at": idx,
                    "results": results,
                    "latest_tab_url": latest_tab_url,
                    "log_url": log_url,
                    **({"sequence_run": True} if req.sequence_run else {}),
                }, status_code=500)
            idx += 1

        local_path = new_screen_path()
        try:
            exec_impl.capture_screen_to_local(local_path, rate_limit_seconds=req.cooldown_seconds)
            LAST_SCREEN_PATH = local_path
            url = f"/static/screenshots/{local_path.name}"
        except Exception as shot_err:
            url = None
            results.append({
                "ok": False,
                "returncode": None,
                "stdout": "",
                "stderr": f"[screenshot_error] {shot_err}",
                "action": None,
            })
            if log_path:
                append_log_line(log_path, json.dumps(results[-1], ensure_ascii=False))

        if log_path:
            append_log_line(log_path, "[END] ok=true")
        show_completion_dialog(exec_impl, ok=True, log_name=log_path.name if log_path else "n/a")
        return JSONResponse({
            "ok": True,
            "results": results,
            "screenshot_url": url,
            "latest_tab_url": latest_tab_url,
            "log_url": log_url,
            **({"sequence_run": True} if req.sequence_run else {}),
        })
    except Exception as e:
        if log_path:
            append_log_line(log_path, f"[ERROR] {e}")
        log_url = f"/static/logs/{log_path.name}" if log_path else None
        try:
            if log_path:
                show_completion_dialog(exec_impl, ok=False, log_name=log_path.name)
        except Exception:
            pass
        return JSONResponse({
            "ok": False,
            "error": str(e),
            "results": results,
            **({"log_url": log_url} if log_url else {}),
            **({"sequence_run": True} if req.sequence_run else {}),
        }, status_code=500)


@app.post("/api/run_actions_sequence")
def run_actions_sequence(req: ActionsSequenceRequest):
    combined: List[Action] = []
    for i, batch in enumerate(req.batches):
        if not isinstance(batch, list):
            return JSONResponse({"ok": False, "error": f"Batch {i + 1} must be a list."}, status_code=400)
        combined.extend(batch)
        if req.cooldown_seconds is not None and req.cooldown_seconds > 0 and i < len(req.batches) - 1:
            combined.append(Action(type="wait", seconds=req.cooldown_seconds))
    run_req = ActionsRequest(
        actions=combined,
        cooldown_seconds=req.cooldown_seconds,
        max_plan_again=req.max_plan_again,
        pause_on_plan_again=req.pause_on_plan_again,
        context_prompt=req.context_prompt,
        mode=req.mode,
        log_id=req.log_id,
        sequence_run=True,
        api_keys=req.api_keys,
    )
    return run_actions(run_req)


class PlanRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    api_keys: Optional[List[str]] = None


class PlanAgainRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    api_keys: Optional[List[str]] = None


def build_action_function_declarations(
    allowed_targets: Optional[List[str]] = None,
    include_plan_again: bool = True,
) -> List[dict]:
    shortcut_names = sorted(SHORTCUTS.keys())
    click_at_params: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "target": {"type": "string"},
        },
        "required": ["target"],
    }
    if allowed_targets is not None:
        click_at_params["properties"]["target"]["enum"] = allowed_targets
    decls = [
        {
            "name": "shortcut",
            "description": "Run a named shortcut action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": shortcut_names},
                },
                "required": ["name"],
            },
        },
        {
            "name": "open_url",
            "description": "Open a URL in Chrome. Requires a full URL string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
        },
        {
            "name": "type_text",
            "description": "Type text into the focused input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "press_enter",
            "description": "Press the Enter key.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "scroll_page",
            "description": "Scroll down one page (space bar).",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "click_at",
            "description": "Click a UI element by target description. Do not return coordinates.",
            "parameters": click_at_params,
        },
    ]
    if include_plan_again:
        decls.append(
            {
                "name": "plan_again",
                "description": "Defer planning until a screenshot is available; put remaining instruction in prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                    },
                    "required": ["prompt"],
                },
            }
        )
    return decls

def planner_system_prompt() -> str:
    return (
        "You are an action planner. You must ONLY use function calls, no free text.\n"
        "Rules:\n"
        "- open_url must include url.\n"
        "- shortcut must include name and the name must be from the enum.\n"
        "- click_at must include target and must NOT include x/y.\n"
        "- press_enter has no parameters.\n"
        "- scroll_page has no parameters.\n"
        "- type_text must include text.\n"
        "- If you need a screenshot to continue (e.g., click a specific UI element after a page loads), "
        "call plan_again with the remaining instruction in prompt.\n"
        "- plan_again must appear at most once and MUST be the final step (no actions after it).\n"
        "Return steps in the correct order by calling tools.\n"
    )

def prompt_needs_visual_target(prompt: str) -> bool:
    text = (prompt or "").lower()
    patterns = [
        r"\b(click|tap|press|select|choose|pick|open|double\s+click|right\s+click)\b"
        r"(?:\s+\w+){0,3}\s+\b(button|menu|link|tab|result|item|option|checkbox|radio|dropdown|icon|image|profile|setting|settings|card)\b",
        r"\b(first|top|second|third|last)\s+(result|link|item|option)\b",
        r"\b(search results|results page|result list)\b",
        r"\b(click|select|choose|pick)\b\s+(the\s+)?(first|top|second|third|last)\b",
        r"\bopen\b\s+(the\s+)?(menu|settings|profile|account|notifications)\b",
        r"\bchoose\b\s+from\s+(dropdown|menu)\b",
        r"\b(select|choose|pick)\b\s+(an?|the)\s+(option|item)\b",
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    fallback_keywords = [
        "click",
        "tap",
        "select",
        "choose",
        "result",
        "button",
        "menu",
        "link",
        "icon",
        "checkbox",
        "dropdown",
        "option",
    ]
    return any(k in text for k in fallback_keywords)


def resolve_click_at_target(
    action: Action,
    cooldown_seconds: Optional[float],
    context_prompt: Optional[str],
    results: List[dict],
    prev_tab_urls: Optional[List[str]],
    click_recovery_used: bool,
    executor_obj=None,
):
    exec_impl = executor_obj or executor
    if not action.target:
        raise ValueError("click_at requires target.")

    # Always take a fresh screenshot before OCR-based click_at.
    def capture_for_ocr() -> Path:
        global LAST_SCREEN_PATH
        local_path = new_screen_path()
        exec_impl.capture_screen_to_local(local_path, rate_limit_seconds=cooldown_seconds)
        LAST_SCREEN_PATH = local_path
        return local_path

    capture_for_ocr()
    last_ocr_path = str(LAST_SCREEN_PATH)
    used_switch_retry = False
    reason_for_action: Optional[str] = None
    screen_size = exec_impl.get_screen_size() or LAST_SCREEN_SIZE
    if not screen_size:
        raise ValueError("Screen size is unavailable for click_at target.")
    image_size = read_png_size(LAST_SCREEN_PATH)
    if not image_size:
        raise ValueError("Failed to read screenshot size for click_at target.")
    candidates = ocr_find_text_candidates(LAST_SCREEN_PATH, action.target)
    if not candidates:
        if LAST_NEW_TAB_URL:
            reason_for_action = "target not found, switching back to latest tab and retrying"
            switch_action = Action(type="open_chrome", url=LAST_NEW_TAB_URL)
            script = applescript_for_action(switch_action)
            rc, out, err = exec_impl.run_osascript(script, rate_limit_seconds=0)
            if rc != 0:
                raise ValueError(f"Failed to open latest tab URL: {err or out}")
            time.sleep(3.0)
            capture_for_ocr()
            last_ocr_path = str(LAST_SCREEN_PATH)
            used_switch_retry = True
            image_size = read_png_size(LAST_SCREEN_PATH)
            if not image_size:
                raise ValueError("Failed to read screenshot size for click_at target.")
            candidates = ocr_find_text_candidates(LAST_SCREEN_PATH, action.target)
        if not candidates:
            if click_recovery_used:
                raise ValueError("Text not found by OCR after recovery.")
            click_recovery_used = True
            current_screen_path = LAST_SCREEN_PATH
            current_ocr_tokens = ocr_text_array(current_screen_path)
            occlusion_url = LAST_NEW_TAB_URL
            global LAST_OCCLUSION
            LAST_OCCLUSION = {
                "screen_path": str(current_screen_path),
                "target": action.target,
                "occlusion_url": occlusion_url,
            }
            return None, JSONResponse({
                "ok": True,
                "paused": True,
                "pause_reason": "occlusion_recovery",
                "screenshot_url": f"/static/screenshots/{current_screen_path.name}",
                "target": action.target,
                "ocr_tokens": current_ocr_tokens,
                "latest_tab_url": update_latest_tab_url(previous_urls=prev_tab_urls, executor_obj=exec_impl),
                "ocr_debug": {
                    "last_screen_path_used_for_ocr": last_ocr_path,
                    "used_switch_back_retry": used_switch_retry,
                },
                "results": results,
            }), click_recovery_used, reason_for_action
    if len(candidates) == 1:
        box = candidates[0]
    else:
        annotated_path = annotate_targets(LAST_SCREEN_PATH, candidates)
        global LAST_DISAMBIGUATION
        LAST_DISAMBIGUATION = {
            "candidates": candidates,
            "annotated_path": str(annotated_path),
            "target": action.target,
            "context_prompt": context_prompt,
            "image_size": image_size,
            "screen_size": screen_size,
        }
        url = f"/static/annotations/{annotated_path.name}"
        return None, JSONResponse({
            "ok": True,
            "paused": True,
            "pause_reason": "multi_target",
            "annotated_url": url,
            "target": action.target,
            "count": len(candidates),
            "prompt_used": context_prompt or action.target,
            "ocr_debug": {
                "last_screen_path_used_for_ocr": last_ocr_path,
                "used_switch_back_retry": used_switch_retry,
            },
            "results": results,
            "latest_tab_url": update_latest_tab_url(previous_urls=prev_tab_urls, executor_obj=exec_impl),
            **({"reason": reason_for_action} if reason_for_action else {}),
        }), click_recovery_used, reason_for_action
    x = int(box["left"]) + int(box["width"]) // 2
    y = int(box["top"]) + int(box["height"]) // 2
    iw, ih = image_size
    sw, sh = screen_size
    if (sw, sh) != (iw, ih):
        x = int(round(x * (sw / iw)))
        y = int(round(y * (sh / ih)))
    resolved = action.model_copy(update={"x": x, "y": y})
    return resolved, None, click_recovery_used, reason_for_action

def extract_tool_calls(response) -> List[Any]:
    try:
        parts = response.candidates[0].content.parts
    except Exception:
        parts = []
    tool_calls = []
    for part in parts:
        if getattr(part, "function_call", None):
            tool_calls.append(part.function_call)
    return tool_calls

def tool_calls_to_batches(
    tool_calls,
    prompt: Optional[str] = None,
    allowed_targets: Optional[List[str]] = None,
) -> Tuple[Optional[List[List[dict]]], Optional[str], Optional[dict]]:
    if not tool_calls:
        return None, "No function call returned.", None
    batches = []
    for call in tool_calls:
        args = call.args if isinstance(call.args, dict) else {}
        name = call.name
        if name == "shortcut":
            action = {"type": "shortcut", "name": args.get("name")}
            if isinstance(args.get("url"), str):
                action["url"] = args.get("url")
        elif name == "open_url":
            action = {"type": "open_url", "url": args.get("url")}
        elif name == "type_text":
            action = {"type": "type_text", "text": args.get("text")}
        elif name == "press_enter":
            action = {"type": "press_enter"}
        elif name == "scroll_page":
            action = {"type": "scroll_page"}
        elif name == "click_at":
            action = {"type": "click_at", "target": args.get("target")}
        elif name == "plan_again":
            action = {"type": "plan_again", "prompt": args.get("prompt")}
        else:
            return None, f"Unknown tool call '{name}'.", None
        batches.append([action])
    raw_args = {"batches": batches}
    err = validate_planner_args(raw_args, prompt=prompt, allowed_targets=allowed_targets)
    if err:
        return None, err, raw_args
    return batches, None, raw_args

def call_gemini_planner(
    prompt: str,
    model: str,
    image_paths: Optional[List[Path]] = None,
    function_decls: Optional[List[dict]] = None,
    api_keys: Optional[List[str]] = None,
):
    from google.genai import types
    if function_decls is None:
        allow_plan_again = prompt_needs_visual_target(prompt)
        function_decls = build_action_function_declarations(include_plan_again=allow_plan_again)
    tools = types.Tool(function_declarations=function_decls)
    config = types.GenerateContentConfig(tools=[tools])
    client = get_gemini_client(api_keys=api_keys)
    if image_paths:
        parts = [types.Part.from_text(text=planner_system_prompt() + prompt)]
        for path in image_paths:
            parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"))
        content = types.Content(role="user", parts=parts)
        return client.models.generate_content(
            model=model,
            contents=[content],
            config=config,
        )
    return client.models.generate_content(
        model=model,
        contents=planner_system_prompt() + prompt,
        config=config,
    )

def validate_planner_args(
    args: dict,
    prompt: Optional[str] = None,
    allowed_targets: Optional[List[str]] = None,
) -> Optional[str]:
    if not isinstance(args, dict):
        return "Invalid args: expected object."
    batches = args.get("batches")
    if not isinstance(batches, list) or not batches:
        return "Invalid args: batches must be a non-empty array."
    shortcut_names = set(SHORTCUTS.keys())
    plan_again_index = None
    for i, batch in enumerate(batches, start=1):
        if not isinstance(batch, list):
            return f"Invalid batch {i}: must be an array."
        if len(batch) != 1:
            return f"Invalid batch {i}: must contain exactly one action."
        action = batch[0]
        if not isinstance(action, dict):
            return f"Invalid batch {i}: action must be an object."
        action_type = action.get("type")
        if action_type not in PLANNER_ACTION_TYPES:
            return f"Invalid batch {i}: unsupported action type '{action_type}'."
        if action_type == "shortcut":
            name = action.get("name")
            if name not in shortcut_names:
                return f"Invalid batch {i}: unknown shortcut name '{name}'."
            extra_keys = set(action.keys()) - {"type", "name"}
            if extra_keys:
                return f"Invalid batch {i}: shortcut has invalid fields {sorted(extra_keys)}."
        if action_type == "open_url":
            if not isinstance(action.get("url"), str):
                return f"Invalid batch {i}: open_url requires url."
            extra_keys = set(action.keys()) - {"type", "url"}
            if extra_keys:
                return f"Invalid batch {i}: open_url has invalid fields {sorted(extra_keys)}."
        if action_type == "click_at":
            has_target = isinstance(action.get("target"), str) and action.get("target")
            if not has_target:
                return f"Invalid batch {i}: click_at requires target text."
            if allowed_targets is not None and action.get("target") not in allowed_targets:
                return f"Invalid batch {i}: click_at target must be one of OCR tokens."
            extra_keys = set(action.keys()) - {"type", "target"}
            if extra_keys:
                return f"Invalid batch {i}: click_at has invalid fields {sorted(extra_keys)}."
        if action_type == "type_text":
            if not isinstance(action.get("text"), str):
                return f"Invalid batch {i}: type_text requires text."
            extra_keys = set(action.keys()) - {"type", "text"}
            if extra_keys:
                return f"Invalid batch {i}: type_text has invalid fields {sorted(extra_keys)}."
        if action_type == "press_enter":
            extra_keys = set(action.keys()) - {"type"}
            if extra_keys:
                return f"Invalid batch {i}: press_enter has invalid fields {sorted(extra_keys)}."
        if action_type == "scroll_page":
            extra_keys = set(action.keys()) - {"type"}
            if extra_keys:
                return f"Invalid batch {i}: scroll_page has invalid fields {sorted(extra_keys)}."
        if action_type == "plan_again":
            if not isinstance(action.get("prompt"), str):
                return f"Invalid batch {i}: plan_again requires prompt."
            extra_keys = set(action.keys()) - {"type", "prompt"}
            if extra_keys:
                return f"Invalid batch {i}: plan_again has invalid fields {sorted(extra_keys)}."
            if plan_again_index is not None:
                return "Invalid args: plan_again must appear at most once."
            plan_again_index = i
    if plan_again_index is not None and plan_again_index != len(batches):
        return "Invalid args: plan_again must be the final step with no actions after it."
    if plan_again_index == 1:
        return "Invalid args: prompt is too vague."
    if plan_again_index is not None and prompt is not None:
        if not prompt_needs_visual_target(prompt):
            return "Invalid args: prompt does not require plan_again."
    return None


@app.post("/api/plan_actions")
def plan_actions(req: PlanRequest):
    """
    Use Gemini function calling to produce a JSON actions array from a natural language prompt.
    """
    model = req.model or "gemini-3-flash-preview"
    try:
        response = call_gemini_planner(req.prompt, model=model, api_keys=req.api_keys)
    except RuntimeError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    except ModuleNotFoundError as e:
        return JSONResponse(
            {"ok": False, "error": f"Missing dependency: {e}. Run: pip install google-genai"},
            status_code=500,
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    tool_calls = extract_tool_calls(response)
    batches, err, raw_args = tool_calls_to_batches(tool_calls, prompt=req.prompt)
    if err:
        if err.startswith("Unknown tool call"):
            return JSONResponse(
                {"ok": False, "error": err, "raw_text": extract_response_text(response)},
                status_code=500,
            )
        if err == "No function call returned.":
            return JSONResponse(
                {"ok": False, "error": err, "raw_text": extract_response_text(response)},
                status_code=500,
            )
        return JSONResponse(
            {
                "ok": False,
                "error": err,
                "retryable": True,
                "function": "multi_tool_calls",
                "raw_args": raw_args,
                "raw_text": extract_response_text(response),
            },
            status_code=400,
        )

    return JSONResponse(
        {
            "ok": True,
            "model": model,
            "function": "multi_tool_calls",
            "args": {"batches": batches},
        }
    )


@app.post("/api/plan_again")
def plan_again(req: PlanAgainRequest):
    """
    Plan additional actions based on the latest screenshot and a remaining prompt.
    """
    if not LAST_SCREEN_PATH or not LAST_SCREEN_PATH.exists():
        return JSONResponse({"ok": False, "error": "No screenshot available for plan_again."}, status_code=400)
    model = req.model or "gemini-3-flash-preview"
    try:
        ocr_tokens = ocr_text_array(LAST_SCREEN_PATH)
        allowed_targets = filter_ocr_tokens_by_prompt(ocr_tokens, req.prompt)
        function_decls = build_action_function_declarations(
            allowed_targets=allowed_targets,
            include_plan_again=True,
        )
        response = call_gemini_planner(
            req.prompt,
            model=model,
            image_paths=[LAST_SCREEN_PATH],
            function_decls=function_decls,
            api_keys=req.api_keys,
        )
        tool_calls = extract_tool_calls(response)
        batches, err, raw_args = tool_calls_to_batches(
            tool_calls,
            prompt=req.prompt,
            allowed_targets=allowed_targets,
        )
        if err:
            return JSONResponse(
                {
                    "ok": False,
                    "error": err,
                    "raw_args": raw_args,
                    "raw_text": extract_response_text(response),
                    "ocr_tokens": ocr_tokens,
                    "allowed_targets": allowed_targets,
                },
                status_code=400,
            )
        return JSONResponse(
            {
                "ok": True,
                "model": model,
                "function": "multi_tool_calls",
                "args": {"batches": batches},
                "ocr_tokens": ocr_tokens,
                "allowed_targets": allowed_targets,
            }
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
