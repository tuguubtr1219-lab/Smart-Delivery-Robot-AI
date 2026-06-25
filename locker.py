"""Locker + delivery QR helpers.

The locker controller is safe for a classroom/demo robot:
- By default it runs in SIMULATED mode and only changes software state.
- If you later connect ESP32 over USB serial, set LOCKER_SERIAL_PORT in config.py
  and install pyserial. It will send simple text commands: LOCK and UNLOCK.
"""
import os
import random
import string
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

from config import SAVE_ROOT

try:
    import qrcode
except Exception:  # pragma: no cover - fallback only when dependency missing
    qrcode = None


LOCKER_SERIAL_PORT = os.environ.get("LOCKER_SERIAL_PORT", "").strip()
LOCKER_BAUD = int(os.environ.get("LOCKER_BAUD", "115200"))


class LockerController:
    def __init__(self):
        self.locked = False
        self.mode = "SIMULATED"
        self._serial = None
        if LOCKER_SERIAL_PORT:
            try:
                import serial
                self._serial = serial.Serial(LOCKER_SERIAL_PORT, LOCKER_BAUD, timeout=0.2)
                self.mode = f"SERIAL:{LOCKER_SERIAL_PORT}"
            except Exception as e:
                self.mode = f"SIMULATED (serial error: {e})"
                self._serial = None

    def _send(self, command):
        if self._serial is not None:
            self._serial.write((command + "\n").encode("utf-8"))

    def lock(self):
        self.locked = True
        self._send("LOCK")

    def unlock(self):
        self.locked = False
        self._send("UNLOCK")

    def close(self):
        try:
            if self._serial is not None:
                self._serial.close()
        except Exception:
            pass


def make_delivery_code(length=8):
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def make_delivery_payload(code):
    return f"SMART_DELIVERY_UNLOCK:{code}"


def create_qr_image(payload, box_size=10, border=4):
    """Return a PIL RGB image containing a real scannable QR payload.

    Primary generator: python-qrcode.
    Fallback generator: OpenCV QRCodeEncoder, so the QR still appears even if
    the user forgot to run `pip install qrcode`.
    """
    if qrcode is not None:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=box_size,
            border=border,
        )
        qr.add_data(payload)
        qr.make(fit=True)
        return qr.make_image(fill_color="black", back_color="white").convert("RGB")

    # Robust fallback: use OpenCV's built-in QR encoder.
    # This creates a real QR, not just text.
    try:
        encoder = cv2.QRCodeEncoder_create()
        qr_arr = encoder.encode(payload)
        if qr_arr is not None and qr_arr.size > 0:
            qr_arr = cv2.resize(qr_arr, (420, 420), interpolation=cv2.INTER_NEAREST)
            if len(qr_arr.shape) == 2:
                rgb = cv2.cvtColor(qr_arr, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(qr_arr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
    except Exception:
        pass

    # Last resort: show clear instructions and the code.
    # This screen is not scannable, but should almost never be used.
    img = Image.new("RGB", (420, 420), "white")
    arr = np.array(img)
    cv2.putText(arr, "QR generator missing", (28, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(arr, "Run: pip install qrcode[pil]", (22, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(arr, payload[-8:], (102, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    return Image.fromarray(arr)


def save_qr_png(payload, code):
    folder = os.path.join(SAVE_ROOT, "delivery_qr")
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"delivery_qr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{code}.png")
    img = create_qr_image(payload, box_size=10, border=4)
    img.save(filename)
    return filename


class QRCodeWindow(tk.Toplevel):
    """Simple window that displays the current delivery QR."""
    def __init__(self, parent, pil_image, code, payload, saved_path):
        super().__init__(parent)
        self.title("Delivery QR Unlock Code")
        self.configure(bg="#111827")
        self.resizable(False, False)
        self.photo = ImageTk.PhotoImage(pil_image.resize((360, 360)))

        tk.Label(
            self,
            text="Delivery QR - show this to the robot camera",
            bg="#111827", fg="#F9FAFB", font=("Segoe UI", 13, "bold")
        ).pack(padx=18, pady=(16, 8))

        tk.Label(self, image=self.photo, bg="#111827").pack(padx=18, pady=8)

        tk.Label(
            self,
            text=f"Code: {code}",
            bg="#111827", fg="#E5E7EB", font=("Consolas", 16, "bold")
        ).pack(padx=18, pady=(4, 4))

        tk.Label(
            self,
            text=f"Saved: {saved_path}",
            bg="#111827", fg="#94A3B8", wraplength=440, justify="center", font=("Segoe UI", 9)
        ).pack(padx=18, pady=(0, 12))

        tk.Button(
            self,
            text="Close",
            command=self.destroy,
            bg="#334155", fg="#FFFFFF", relief="flat", padx=22, pady=8,
            font=("Segoe UI", 10, "bold")
        ).pack(pady=(0, 16))


def decode_qr_from_frame(frame_bgr, max_width=360):
    """Decode one or more QR codes from a BGR frame. Returns list[str]."""
    if frame_bgr is None:
        return []
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.QRCodeDetector()
    results = []

    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(gray)
        if ok and decoded_info:
            results.extend([text for text in decoded_info if text])
    except Exception:
        pass

    if not results:
        try:
            text, points, _ = detector.detectAndDecode(gray)
            if text:
                results.append(text)
        except Exception:
            pass
    return results
