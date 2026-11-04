"""Giao diện Tkinter dùng trực tiếp logic từ pipeline.py."""

import time
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from processing.gaussian import gaussian_blur
from processing.morphology import opening
from detection.motion import detect_motion
from detection.motion import calculate_motion_ratio
from detection.contours import find_connected_components
from detection.contours import merge_near_boxes
from detection.person import is_person, split_by_gap


def to_grayscale(frame):
    blue = frame[:, :, 0]
    green = frame[:, :, 1]
    red = frame[:, :, 2]

    grayscale = (
        0.114 * blue +
        0.587 * green +
        0.299 * red
    )

    return grayscale.astype(np.uint8)


# ==============================
# CẤU HÌNH 
# ==============================

MOTION_THRESHOLD = 40
MOTION_RATIO = 5
STATIC_WAIT_TIME = 5
MIN_AREA = 2000
LIGHT_RATIO = 70
LIGHT_WAIT_TIME = 30


class FindHumanUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FIND HUMAN")
        self.root.geometry("1200x760")
        self.root.configure(bg="#17191D")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.close())

        self.camera = None
        self.running = False
        self.after_id = None

        # Các biến trạng thái của pipeline.py
        self.is_static = False
        self.start_time = None
        self.start_bg_time = None
        self.background_ready = False
        self.background_blurred = None
        self.light_start_time = None
        self.previous_frame = None

        self.build_ui()

    def build_ui(self):
        tk.Label(
            self.root,
            text="FIND HUMAN",
            font=("Segoe UI", 24, "bold"),
            fg="white",
            bg="#17191D",
        ).pack(pady=(16, 10))

        content = tk.Frame(self.root, bg="#17191D")
        content.pack(fill="both", expand=True, padx=22, pady=8)
        content.grid_columnconfigure(0, weight=2)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)

        camera_panel, self.camera_label = self.make_panel(
            content, "CAMERA", "Nhấn START để bật camera"
        )
        camera_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8))

        foreground_panel, self.foreground_label = self.make_panel(
            content, "FOREGROUND", "Ảnh chuyển động"
        )
        foreground_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 4))

        background_panel, self.background_label = self.make_panel(
            content, "BACKGROUND", "Chưa tạo background"
        )
        background_panel.grid(row=1, column=1, sticky="nsew", pady=(4, 0))

        controls = tk.Frame(self.root, bg="#22262D", padx=14, pady=12)
        controls.pack(fill="x", padx=22, pady=(0, 16))

        self.start_button = tk.Button(
            controls,
            text="START",
            command=self.start,
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg="#16A34A",
            activebackground="#15803D",
            relief="flat",
            cursor="hand2",
            width=14,
            pady=8,
        )
        self.start_button.pack(side="left")

        self.stop_button = tk.Button(
            controls,
            text="STOP",
            command=self.stop,
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg="#DC2626",
            activebackground="#B91C1C",
            relief="flat",
            cursor="hand2",
            width=14,
            pady=8,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=(10, 0))

        self.status = tk.Label(
            controls,
            text="Đã dừng",
            font=("Segoe UI", 10, "bold"),
            fg="#9CA3AF",
            bg="#22262D",
        )
        self.status.pack(side="left", padx=20)

    def make_panel(self, parent, title, placeholder):
        panel = tk.Frame(
            parent,
            bg="#22262D",
            highlightthickness=1,
            highlightbackground="#343A44",
        )
        tk.Label(
            panel,
            text=title,
            font=("Segoe UI", 9, "bold"),
            fg="white",
            bg="#22262D",
        ).pack(anchor="w", padx=10, pady=(8, 3))
        label = tk.Label(
            panel,
            text=placeholder,
            font=("Segoe UI", 14, "bold"),
            fg="#9CA3AF",
            bg="#111317",
        )
        label.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        return panel, label

    def start(self):
        if self.running:
            return

        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            messagebox.showerror("Camera", "Không thể mở camera")
            return

        self.running = True
        self.is_static = False
        self.start_time = None
        self.start_bg_time = None
        self.background_ready = False
        self.background_blurred = None
        self.light_start_time = None
        self.previous_frame = None

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status.config(text="Đang chạy", fg="#4ADE80")
        self.update_camera()

    def stop(self):
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status.config(text="Đã dừng", fg="#9CA3AF")

    def update_camera(self):
        if not self.running or self.camera is None:
            return

        current_time = time.time()
        ret, frame = self.camera.read()

        if not ret:
            self.stop()
            messagebox.showerror("Camera", "Khong doc duoc Camera")
            return

        # ==============================
        # VÒNG LẶP PIPELINE GỐC
        # ==============================
        grayscale_frame = to_grayscale(frame)
        blurred = gaussian_blur(grayscale_frame)
        foreground_opened = np.zeros_like(blurred)

        if self.previous_frame is None:
            self.previous_frame = blurred
            self.show_image(frame, self.camera_label, True, (800, 600))
            self.after_id = self.root.after(15, self.update_camera)
            return

        motion_mask = detect_motion(
            self.previous_frame,
            blurred,
            MOTION_THRESHOLD,
        )
        self.previous_frame = blurred

        opened = opening(motion_mask)
        motion_ratio = calculate_motion_ratio(opened)
        print("Motion ratio:", motion_ratio)

        if motion_ratio > MOTION_RATIO:
            self.is_static = False
            self.start_time = None
        else:
            if self.start_time is None:
                self.start_time = current_time
            elif current_time - self.start_time >= STATIC_WAIT_TIME:
                self.is_static = True

        if not self.background_ready:
            if self.is_static:
                if self.start_bg_time is None:
                    self.start_bg_time = time.time()
                elif time.time() - self.start_bg_time >= STATIC_WAIT_TIME:
                    background_float = blurred.astype(np.float32)
                    self.background_blurred = blurred.copy()
                    self.background_ready = True
                    self.start_bg_time = None
                    print("Đã tạo background")

        if self.background_ready:
            foreground_mask = detect_motion(
                self.background_blurred,
                blurred,
                MOTION_THRESHOLD,
            )
            foreground_opened = opening(foreground_mask)
            bg_ratio = calculate_motion_ratio(foreground_opened)
            print("Bacground ratio : ", bg_ratio)

            if bg_ratio > LIGHT_RATIO:
                print("hơn "+str(LIGHT_RATIO)+f"% do as môi trường")
                diff = blurred.astype(np.int16) - self.background_blurred.astype(np.int16)
                shift = int(np.median(diff))
                adjusted_bg = np.clip(
                    self.background_blurred.astype(np.int16) + shift,
                    0,
                    255,
                ).astype(np.uint8)
                if self.light_start_time is None:
                    self.light_start_time = current_time
                elif (
                    current_time - self.light_start_time >= LIGHT_WAIT_TIME
                    and self.is_static is True
                ):
                    self.background_blurred = blurred
                    self.light_start_time = None

                foreground_mask = detect_motion(
                    adjusted_bg,
                    blurred,
                    MOTION_THRESHOLD,
                )
                foreground_opened = opening(foreground_mask)
            else:
                self.light_start_time = None

            if self.is_static is False:
                self.light_start_time = None

            components = find_connected_components(foreground_opened)
            boxes = []
            for x, y, w, h in components:
                if w * h >= MIN_AREA:
                    boxes.append((x, y, w, h))

            boxes = merge_near_boxes(boxes)
            boxes = [
                p
                for b in boxes
                for p in split_by_gap(foreground_opened, b)
            ]
            people = [
                b
                for b in boxes
                if is_person(foreground_opened, b, frame.shape)
            ]
            for x, y, w, h in people:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

        # Chỉ thay phần cv2.imshow bằng phần hiển thị giao diện.
        self.show_image(frame, self.camera_label, True, (800, 600))
        self.show_image(foreground_opened, self.foreground_label, False, (420, 270))
        if self.background_blurred is not None:
            self.show_image(self.background_blurred, self.background_label, False, (420, 270))

        self.after_id = self.root.after(15, self.update_camera)

    def show_image(self, frame, label, color, size):
        if color:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            image = Image.fromarray(frame)
        image.thumbnail(size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        label.config(image=photo, text="")
        label.image = photo

    def close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FindHumanUI(root)
    root.mainloop()
