import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd


def load_color_dataset(csv_path: str) -> pd.DataFrame:
    """Load the color name dataset (color_name, hex, R, G, B)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Color dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"color_name", "hex", "R", "G", "B"}
    if not required.issubset(df.columns):
        raise ValueError(f"colors.csv must contain columns: {required}")
    return df


def closest_color_name(r: int, g: int, b: int, df: pd.DataFrame):
    """Return (name, hex, exact_match_bool) for the nearest color by
    Euclidean distance in RGB space."""
    diffs = (
        (df["R"] - r) ** 2 + (df["G"] - g) ** 2 + (df["B"] - b) ** 2
    )
    idx = diffs.idxmin()
    row = df.loc[idx]
    exact = diffs.loc[idx] == 0
    return row["color_name"], row["hex"], bool(exact)


class ColorPickerApp:
    def __init__(self, image_path: str, csv_path: str, max_display_width: int = 900):
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")

        # Resize for comfortable on-screen display while keeping a copy
        # of the original resolution for accurate pixel sampling.
        h, w = self.original.shape[:2]
        if w > max_display_width:
            scale = max_display_width / w
            self.display = cv2.resize(self.original, (int(w * scale), int(h * scale)))
        else:
            self.display = self.original.copy()
            scale = 1.0
        self.scale = scale

        self.df = load_color_dataset(csv_path)
        self.clicked_point = None
        self.last_label = ""
        self.shot_count = 0

        self.picker_window = "Color Picker (double-click a pixel, q to quit)"
        self.hsv_window = "HSV Threshold"

        cv2.namedWindow(self.picker_window)
        cv2.setMouseCallback(self.picker_window, self._on_mouse)

        cv2.namedWindow(self.hsv_window)
        # 6 trackbars: H/S/V min and max
        cv2.createTrackbar("H min", self.hsv_window, 0, 179, lambda x: None)
        cv2.createTrackbar("H max", self.hsv_window, 179, 179, lambda x: None)
        cv2.createTrackbar("S min", self.hsv_window, 0, 255, lambda x: None)
        cv2.createTrackbar("S max", self.hsv_window, 255, 255, lambda x: None)
        cv2.createTrackbar("V min", self.hsv_window, 0, 255, lambda x: None)
        cv2.createTrackbar("V max", self.hsv_window, 255, 255, lambda x: None)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicked_point = (x, y)
            # Map back to original-resolution coordinates for sampling.
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            orig_x = min(max(orig_x, 0), self.original.shape[1] - 1)
            orig_y = min(max(orig_y, 0), self.original.shape[0] - 1)

            b, g, r = self.original[orig_y, orig_x]
            name, hexcode, exact = closest_color_name(int(r), int(g), int(b), self.df)
            match = "exact" if exact else "closest"
            self.last_label = f"{name} ({match})  RGB({r},{g},{b})  {hexcode}"
            print(f"[Picked] pixel=({orig_x},{orig_y})  {self.last_label}")

    def _draw_picker_frame(self):
        frame = self.display.copy()
        if self.clicked_point:
            x, y = self.clicked_point
            cv2.circle(frame, (x, y), 6, (0, 0, 0), 2)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)

            label = self.last_label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            box_x, box_y = x + 12, max(y - 12, th + 10)
            cv2.rectangle(
                frame,
                (box_x - 5, box_y - th - 8),
                (box_x + tw + 5, box_y + 6),
                (245, 245, 245),
                -1,
            )
            cv2.rectangle(
                frame,
                (box_x - 5, box_y - th - 8),
                (box_x + tw + 5, box_y + 6),
                (30, 30, 30),
                1,
            )
            cv2.putText(
                frame, label, (box_x, box_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA,
            )
        return frame

    def _draw_hsv_frame(self):
        hsv = cv2.cvtColor(self.display, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("H min", self.hsv_window)
        h_max = cv2.getTrackbarPos("H max", self.hsv_window)
        s_min = cv2.getTrackbarPos("S min", self.hsv_window)
        s_max = cv2.getTrackbarPos("S max", self.hsv_window)
        v_min = cv2.getTrackbarPos("V min", self.hsv_window)
        v_max = cv2.getTrackbarPos("V max", self.hsv_window)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(self.display, self.display, mask=mask)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack([mask_bgr, result])
        return stacked

    def run(self):
        print("Double-click the image to identify a color.")
        print("Drag the HSV trackbars to isolate a color range.")
        print("Press 's' to save a screenshot, 'q' or ESC to quit.")
        while True:
            cv2.imshow(self.picker_window, self._draw_picker_frame())
            cv2.imshow(self.hsv_window, self._draw_hsv_frame())

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord("s"):
                self.shot_count += 1
                fname = f"screenshot_{self.shot_count}.png"
                cv2.imwrite(fname, self._draw_picker_frame())
                print(f"Saved {fname}")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive color detection tool")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "colors.csv"),
        help="Path to colors.csv dataset (default: colors.csv next to this script)",
    )
    args = parser.parse_args()

    try:
        app = ColorPickerApp(args.image, args.csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    app.run()


if __name__ == "__main__":
    main()
