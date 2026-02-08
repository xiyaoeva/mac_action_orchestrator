from pydantic import BaseModel, Field
from typing import Literal, Optional


# Keep model output constrained to JSON, then map it to trusted script templates.
ACTION_TYPES = [
    "finder_about_this_mac",
    "open_chrome",
    "open_url",
    "type_text",
    "press_enter",
    "scroll_page",
    "press_key",
    "release_key",
    "wait",
    "click_at",
    "shortcut",
    "plan_again",
]

PLANNER_ACTION_TYPES = [
    "open_url",
    "type_text",
    "press_enter",
    "scroll_page",
    "wait",
    "click_at",
    "shortcut",
    "plan_again",
]
ActionType = Literal[
    "finder_about_this_mac",
    "open_chrome",
    "open_url",
    "type_text",
    "press_enter",
    "scroll_page",
    "press_key",
    "release_key",
    "wait",
    "click_at",
    "shortcut",
    "plan_again",
]


class Action(BaseModel):
    type: ActionType
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    url: Optional[str] = None
    key_code: Optional[int] = None
    seconds: Optional[float] = None
    name: Optional[str] = None
    target: Optional[str] = None
    prompt: Optional[str] = None


def applescript_for_action(action: Action) -> str:
    # Central mapping from validated action objects to AppleScript/JXA payloads.
    t = action.type

    # These are planner/meta actions and must be resolved before execution.
    if t == "shortcut":
        raise ValueError("shortcut must be expanded before execution")
    if t == "plan_again":
        raise ValueError("plan_again must be expanded before execution")

    if t == "finder_about_this_mac":
        # Note: menu item text is locale-specific; keep current label as-is.
        return r'''
tell application "Finder" to activate
tell application "System Events"
  tell process "Finder"
    set theItem to menu item "About This Mac" of menu 1 of menu bar item 1 of menu bar 1
    click theItem
  end tell
end tell
return "clicked"
'''

    if t == "open_chrome":
        if not action.url:
            raise ValueError("open_chrome requires url")
        url = action.url.replace('"', '\\"')
        return f'''
tell application "Google Chrome"
  activate
  open location "{url}"
end tell
return "ok"
'''

    if t == "type_text":
        if action.text is None:
            raise ValueError("type_text requires text")
        txt = action.text.replace('"', '\\"')
        return f'''
tell application "System Events"
  keystroke "{txt}"
end tell
return "ok"
'''

    if t == "press_enter":
        return r'''
tell application "System Events"
  key code 36
end tell
return "ok"
'''

    if t == "scroll_page":
        return r'''
tell application "System Events"
  key code 49
end tell
return "ok"
'''

    if t == "press_key":
        if action.key_code is None:
            raise ValueError("press_key requires key_code")
        return f'''
tell application "System Events"
  key down {action.key_code}
end tell
return "ok"
'''

    if t == "release_key":
        if action.key_code is None:
            raise ValueError("release_key requires key_code")
        return f'''
tell application "System Events"
  key up {action.key_code}
end tell
return "ok"
'''

    if t == "wait":
        if action.seconds is None:
            raise ValueError("wait requires seconds")
        return f'''
delay {action.seconds}
return "ok"
'''

    if t == "click_at":
        if action.x is None or action.y is None:
            raise ValueError("click_at requires x,y")
        # JXA block performs a native mouse move + click through Quartz APIs.
        # It also probes Accessibility permission and emits diagnostics.
        js = (
            'ObjC.import("ApplicationServices"); '
            "var trusted = false; "
            "try { "
            "  if ($.AXIsProcessTrustedWithOptions) { "
            "    var opts = $.NSDictionary.dictionaryWithObjectForKey(true, $.kAXTrustedCheckOptionPrompt); "
            "    trusted = $.AXIsProcessTrustedWithOptions(opts); "
            "  } else { "
            "    trusted = $.AXIsProcessTrusted(); "
            "  } "
            "} catch (e) { "
            "  console.log(\"ACCESSIBILITY_CHECK_ERROR:\" + e); "
            "} "
            "if (!trusted) { console.log(\"ACCESSIBILITY_NOT_TRUSTED\"); } "
            f"var point = {{x: {action.x}, y: {action.y}}}; "
            "try { "
            "  var move = $.CGEventCreateMouseEvent(null, $.kCGEventMouseMoved, point, $.kCGMouseButtonLeft); "
            "  $.CGEventPost($.kCGHIDEventTap, move); "
            "  var down = $.CGEventCreateMouseEvent(null, $.kCGEventLeftMouseDown, point, $.kCGMouseButtonLeft); "
            "  $.CGEventPost($.kCGHIDEventTap, down); "
            "  var up = $.CGEventCreateMouseEvent(null, $.kCGEventLeftMouseUp, point, $.kCGMouseButtonLeft); "
            "  $.CGEventPost($.kCGHIDEventTap, up); "
            "  console.log(\"CLICK_OK\"); "
            "} catch (e) { "
            "  console.log(\"CLICK_ERROR:\" + e); "
            "}"
        )
        cmd = f"/usr/bin/osascript -l JavaScript -e '{js}'".replace('"', '\\"')
        return f'''
set output to do shell script "{cmd}"
return output
'''

    raise ValueError(f"Unsupported action type: {t}")
