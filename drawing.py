import cv2


def draw_panel(img, x1, y1, x2, y2, color, border_color=None, alpha=0.75):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if border_color:
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)


def draw_simple_label(img, x, y, text, color):
    # Smaller object label so the camera view does not become crowded.
    y_top = max(0, y - 24)
    x2 = min(img.shape[1] - 1, x + 260)
    cv2.rectangle(img, (x, y_top), (x2, y), (20, 20, 20), -1)
    cv2.rectangle(img, (x, y_top), (x2, y), color, 1)
    cv2.putText(img, text, (x + 6, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


def draw_dashboard(frame, mode_name, humans, obstacles, path_found, center_depth, safety_distance, safety_state, fps, pose_ok, pose_source):
    """Compact HUD.

    The old one-line dashboard became too long and overlapped the camera preview.
    This version uses a small two-line box in the top-left corner.
    """
    h, w = frame.shape[:2]
    panel_w = min(w - 12, 360)
    panel_h = 42
    x1, y1 = 8, 8
    x2, y2 = x1 + panel_w, y1 + panel_h

    if safety_state == "CRITICAL":
        border = (0, 0, 255)
    elif safety_state == "WARNING":
        border = (0, 210, 255)
    else:
        border = (90, 120, 160)

    draw_panel(frame, x1, y1, x2, y2, (15, 18, 24), border, 0.62)
    center_text = "--" if center_depth is None else f"{center_depth:.2f}m"
    front_text = "--" if safety_distance is None else f"{safety_distance:.2f}m"
    pose_text = pose_source if pose_ok else "OFF"

    line1 = f"{mode_name} | Front {front_text} | Center {center_text} | {safety_state}"
    line2 = f"FPS {fps:.1f} | Pose {pose_text} | Path {'OK' if path_found else '--'}"

    cv2.putText(frame, line1, (x1 + 8, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1)
    cv2.putText(frame, line2, (x1 + 8, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 220, 235), 1)


def draw_safety_badge(frame, safety_state, safety_distance):
    if safety_distance is None or safety_state not in ("CRITICAL", "WARNING"):
        return
    h, w = frame.shape[:2]
    text = f"STOP {safety_distance:.2f}m" if safety_state == "CRITICAL" else f"NEAR {safety_distance:.2f}m"
    color = (0, 0, 255) if safety_state == "CRITICAL" else (0, 210, 255)
    box_w = 150
    box_h = 32
    x1 = max(8, (w - box_w) // 2)
    y1 = 8
    draw_panel(frame, x1, y1, x1 + box_w, y1 + box_h, (20, 20, 20), color, 0.70)
    cv2.putText(frame, text, (x1 + 12, y1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
