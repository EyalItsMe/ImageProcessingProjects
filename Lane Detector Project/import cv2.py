import cv2
import numpy as np
from pathlib import Path
# Optional: only needed if you actually use plt.imshow/plt.show.
# Importing matplotlib can fail on some machines due to font-cache permissions.
# import matplotlib.pyplot as plt


# --- CONFIGURATION ---
VIDEO_PROFILE = "day"  # choose: "day" or "night"

_SCRIPT_DIR = Path(__file__).resolve().parent  # .../Project1/Lane Detector Project
_PROJECT_DIR = _SCRIPT_DIR.parent             # .../Project1
_VIDEO_INPUTS = {
    "day": _SCRIPT_DIR / "Dashcam highway.mp4",
    "night": _PROJECT_DIR / "night_drive_480.mp4",
}
VIDEO_INPUT = str(_VIDEO_INPUTS[VIDEO_PROFILE])
VIDEO_OUTPUT = 'lane_detection_output.mp4'
HISTORY_LENGTH = 20 

# --- ROI / GEOMETRY (resolution-independent) ---
# We keep *two* sets of parameters:
# - "day": based on the ORIGINAL hard-coded trapezoid you had (classic demo values).
# - "night": tighter/shifted ROI you tuned for `night_drive.mp4`.
#
# ORIGINAL (day) trapezoid in pixels (at 640x360 reference):
#   bottom_left=(100, 360), top_left=(280, 250), top_right=(450, 250), bottom_right=(600, 360)
LANE_PROFILES = {
    "day": {
        # ROI trapezoid (ratios)
        "ROI_BOTTOM_LEFT_X_RATIO": 100 / 640,
        "ROI_TOP_LEFT_X_RATIO": 350 / 640,
        "ROI_TOP_RIGHT_X_RATIO": 450 / 640,
        "ROI_BOTTOM_RIGHT_X_RATIO": 600 / 640,
        "ROI_TOP_Y_RATIO": 250 / 360,
        "ROI_BOTTOM_Y_RATIO": 360 / 360,
        # x_bottom windows for filtering "good" Hough lines (ratios)
        "LEFT_X_BOTTOM_MIN_RATIO": 130 / 640,
        "LEFT_X_BOTTOM_MAX_RATIO": 250 / 640,
        "RIGHT_X_BOTTOM_MIN_RATIO": 400 / 640,
        "RIGHT_X_BOTTOM_MAX_RATIO": 600 / 640,
        # Hough theta angle gates (degrees) for lane-like lines
        "LEFT_ANGLE_MIN_DEG": 35,
        "LEFT_ANGLE_MAX_DEG": 55,
        "RIGHT_ANGLE_MIN_DEG": 120,
        "RIGHT_ANGLE_MAX_DEG": 140,
        # Brightness adjustment (OFF by default for day)
        "APPLY_BRIGHTNESS": False,
        "BRIGHTNESS_BETA": 0,  # 0..60 (adds to V channel)
        "APPLY_CONTRAST": False,
        "CONTRAST_ALPHA": 1.0,  # 1.0 = no change
    },
    "night": {
        # Your tuned ROI (ratios) for night video
        "ROI_BOTTOM_LEFT_X_RATIO": 60 / 640,
        # Move this LEFT to widen the ROI on the left side near the horizon
        "ROI_TOP_LEFT_X_RATIO": 240 / 640,
        "ROI_TOP_RIGHT_X_RATIO": 470 / 640,
        "ROI_BOTTOM_RIGHT_X_RATIO": 600 / 640,
        "ROI_TOP_Y_RATIO": 260 / 360,
        "ROI_BOTTOM_Y_RATIO": 360 / 360,
        # Keep these unless you want to tune them separately for night
        "LEFT_X_BOTTOM_MIN_RATIO": 100 / 640,
        "LEFT_X_BOTTOM_MAX_RATIO": 180 / 640,
        "RIGHT_X_BOTTOM_MIN_RATIO": 450 / 640,
        "RIGHT_X_BOTTOM_MAX_RATIO": 600 / 640,
        # Night footage tends to be noisier; allow a wider band (especially for the right lane)
        "LEFT_ANGLE_MIN_DEG": 30,
        "LEFT_ANGLE_MAX_DEG": 65,
        "RIGHT_ANGLE_MIN_DEG": 120,
        "RIGHT_ANGLE_MAX_DEG": 140,
        # Brightness adjustment (night / low-light)
        "APPLY_BRIGHTNESS": True,
        "BRIGHTNESS_BETA": 15,  # start here; try 15..45
        # Contrast adjustment (night / low-light)
        "APPLY_CONTRAST": True,
        "CONTRAST_ALPHA": 1.20,  # start here; try 1.10..1.50
    },
}

if VIDEO_PROFILE not in LANE_PROFILES:
    raise ValueError(f"VIDEO_PROFILE must be one of {list(LANE_PROFILES.keys())}, got {VIDEO_PROFILE!r}")

# Export profile values as the globals your functions already use
_p = LANE_PROFILES[VIDEO_PROFILE]
ROI_BOTTOM_LEFT_X_RATIO = _p["ROI_BOTTOM_LEFT_X_RATIO"]
ROI_TOP_LEFT_X_RATIO = _p["ROI_TOP_LEFT_X_RATIO"]
ROI_TOP_RIGHT_X_RATIO = _p["ROI_TOP_RIGHT_X_RATIO"]
ROI_BOTTOM_RIGHT_X_RATIO = _p["ROI_BOTTOM_RIGHT_X_RATIO"]
ROI_TOP_Y_RATIO = _p["ROI_TOP_Y_RATIO"]
ROI_BOTTOM_Y_RATIO = _p["ROI_BOTTOM_Y_RATIO"]
LEFT_X_BOTTOM_MIN_RATIO = _p["LEFT_X_BOTTOM_MIN_RATIO"]
LEFT_X_BOTTOM_MAX_RATIO = _p["LEFT_X_BOTTOM_MAX_RATIO"]
RIGHT_X_BOTTOM_MIN_RATIO = _p["RIGHT_X_BOTTOM_MIN_RATIO"]
RIGHT_X_BOTTOM_MAX_RATIO = _p["RIGHT_X_BOTTOM_MAX_RATIO"]
LEFT_ANGLE_MIN_DEG = _p["LEFT_ANGLE_MIN_DEG"]
LEFT_ANGLE_MAX_DEG = _p["LEFT_ANGLE_MAX_DEG"]
RIGHT_ANGLE_MIN_DEG = _p["RIGHT_ANGLE_MIN_DEG"]
RIGHT_ANGLE_MAX_DEG = _p["RIGHT_ANGLE_MAX_DEG"]
APPLY_BRIGHTNESS = _p.get("APPLY_BRIGHTNESS", False)
BRIGHTNESS_BETA = int(_p.get("BRIGHTNESS_BETA", 0))
APPLY_CONTRAST = _p.get("APPLY_CONTRAST", False)
CONTRAST_ALPHA = float(_p.get("CONTRAST_ALPHA", 1.0))
del _p

# --- HELPER FUNCTIONS ---

def adjust_brightness_v_channel(image_bgr, beta):
    """
    Brighten/darken by shifting the V channel in HSV.
    beta > 0 brightens; beta < 0 darkens.
    """
    if beta == 0:
        return image_bgr
    # Keep in a safe range
    beta = int(max(-255, min(255, beta)))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Use numpy for robust scalar shift + clipping (avoids cv2.add dtype overload quirks)
    v = np.clip(v.astype(np.int16) + beta, 0, 255).astype(np.uint8)
    hsv2 = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

def adjust_contrast_v_channel(image_bgr, alpha):
    """
    Adjust contrast by scaling the HSV V channel around mid-gray (128).
    alpha > 1 increases contrast; 0 < alpha < 1 decreases.
    """
    if alpha is None or abs(float(alpha) - 1.0) < 1e-6:
        return image_bgr
    alpha = float(max(0.1, min(3.0, alpha)))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip((v.astype(np.float32) - 128.0) * alpha + 128.0, 0, 255).astype(np.uint8)
    hsv2 = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

def region_of_interest(image, show_debug=False, return_mask=False):
    """
    Applies a trapezoid mask using specific coordinates.
    """
    height, width = image.shape[:2]
    y_top = int(ROI_TOP_Y_RATIO * height)
    y_bottom = int(ROI_BOTTOM_Y_RATIO * height)
    polygons = np.array([
        [
            (int(ROI_BOTTOM_LEFT_X_RATIO * width), y_bottom),
            (int(ROI_TOP_LEFT_X_RATIO * width), y_top),
            (int(ROI_TOP_RIGHT_X_RATIO * width), y_top),
            (int(ROI_BOTTOM_RIGHT_X_RATIO * width), y_bottom),
        ]
    ])
    
    # Build both a 1-channel mask (for bitwise ops) and a 3-channel mask (for BGR images)
    mask_gray = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask_gray, polygons, 255)

    if len(image.shape) == 2:
        # Single-channel input (e.g., binary mask)
        masked_image = cv2.bitwise_and(image, image, mask=mask_gray)
    else:
        # Color input (BGR)
        mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
        masked_image = cv2.bitwise_and(image, mask_bgr)
    
    if show_debug:
        cv2.imshow("Debug 0: ROI Mask", masked_image)

    if return_mask:
        return masked_image, mask_gray
    return masked_image

def filter_white_pixels(image, show_debug=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Optimized for white lanes in shadow/sun
    lower_white = np.array([0, 0, 160]) 
    upper_white = np.array([180, 75, 255]) 
    mask = cv2.inRange(hsv, lower_white, upper_white)
    if show_debug:
        cv2.imshow("Debug 1: HSV White Filter", mask)
    return mask

def canny_edge_detector(image, show_debug=False):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    canny = cv2.Canny(blur, 130, 200)
    if show_debug:
        cv2.imshow("Debug 2: Canny Edges", canny)
        # plt.imshow(canny)

        # # Optional: Add labels to make it clear
        # plt.xlabel("Width (Pixels)")
        # plt.ylabel("Height (Pixels)")
        # plt.title("Image with Pixel Axes")
        # # Ensure axes are visible (they are on by default, but this forces it)
        # plt.axis('on')
        # plt.show()


    return canny

def detect_hough_lines(canny_image, rho_res, theta_res, threshold, color_image_for_debug, show_debug=False):
    all_lines = cv2.HoughLines(canny_image, rho_res, theta_res, threshold)
    if show_debug:
        hough_debug = color_image_for_debug.copy()
        if all_lines is not None:
            for r_t in all_lines:
                rho, theta = r_t[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(hough_debug, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow("Debug 3: Raw Hough Lines", hough_debug)
    return all_lines

def get_good_lane_lines(lines, height, width):
    good_left_lines = []
    good_right_lines = []
    if lines is None: return good_left_lines, good_right_lines
    
    left_min = int(LEFT_X_BOTTOM_MIN_RATIO * width)
    left_max = int(LEFT_X_BOTTOM_MAX_RATIO * width)
    right_min = int(RIGHT_X_BOTTOM_MIN_RATIO * width)
    right_max = int(RIGHT_X_BOTTOM_MAX_RATIO * width)

    for r_t in lines:
        rho = r_t[0, 0]
        theta = r_t[0, 1]
        angle_deg = np.degrees(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        if abs(a) < 0.001: continue 
        x_bottom = int((rho - height * b) / a)

        if LEFT_ANGLE_MIN_DEG < angle_deg < LEFT_ANGLE_MAX_DEG:
            if left_min < x_bottom < left_max:
                good_left_lines.append((rho, theta))
        elif RIGHT_ANGLE_MIN_DEG < angle_deg < RIGHT_ANGLE_MAX_DEG:
            if right_min < x_bottom < right_max:
                good_right_lines.append((rho, theta))
    return good_left_lines, good_right_lines

def get_average_lane(lines):
    if len(lines) == 0: return None, None
    rhos = [l[0] for l in lines]
    thetas = [l[1] for l in lines]
    return np.mean(rhos), np.mean(thetas)

def get_weighted_average(history):
    valid_lines = [line for line in history if line[0] is not None]
    N = len(valid_lines)
    if N == 0: return None, None
    
    weights = np.arange(1, N + 1)
    rhos = np.array([h[0] for h in valid_lines])
    thetas = np.array([h[1] for h in valid_lines])
    
    return np.sum(weights * rhos) / np.sum(weights), np.sum(weights * thetas) / np.sum(weights)

def draw_lane_polygon(image, left_line, right_line):
    if left_line[0] is None or right_line[0] is None: return

    rho_l, theta_l = left_line
    rho_r, theta_r = right_line
    height, width = image.shape[:2]
    
    def get_x(rho, theta, y):
        a = np.cos(theta)
        b = np.sin(theta)
        if abs(a) < 0.001: return 0
        return int((rho - y * b) / a)

    # Match ROI trapezoid vertical bounds, but resolution-independent
    y_top = int(ROI_TOP_Y_RATIO * height)
    y_bottom = int(ROI_BOTTOM_Y_RATIO * height)

    left_bottom  = (get_x(rho_l, theta_l, y_bottom), y_bottom)
    left_top     = (get_x(rho_l, theta_l, y_top), y_top)
    right_top    = (get_x(rho_r, theta_r, y_top), y_top)
    right_bottom = (get_x(rho_r, theta_r, y_bottom), y_bottom)
    
    pts = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    overlay = image.copy()
    
    cv2.fillPoly(overlay, pts, (255, 0, 0))
    cv2.addWeighted(overlay, 0.4, image, 1 - 0.4, 0, image)
    
    cv2.line(image, left_bottom, left_top, (255, 0, 0), 3)
    cv2.line(image, right_bottom, right_top, (255, 0, 0), 3)

def get_x_coordinate(rho, theta, y):
    a = np.cos(theta)
    b = np.sin(theta)
    if abs(a) < 0.001: return 0
    return int((rho - y * b) / a)

# --- MAIN EXECUTION ---

def process_video():
    left_line_history = []
    right_line_history = []
    
    # Lane Change Detection Variables
    prev_lane_center = None
    lane_center_trend = 0
    lane_change_status = ""
    last_valid_smooth_left = None
    last_valid_smooth_right = None
    # Lane change gating (reduce false positives):
    # Only display/confirm lane change after N consecutive "lane lost" frames
    # with a consistent direction signal.
    MIN_LANE_CHANGE_FRAMES = 15 if VIDEO_PROFILE == "night" else 7
    lane_change_candidate = None  # "left" | "right" | None
    lane_change_candidate_frames = 0
    
    cap = cv2.VideoCapture(VIDEO_INPUT)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (frame_width, frame_height))
    
    print(f"Profile: {VIDEO_PROFILE} | Playing video... Press 'q' to stop.")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break 
            
            h,w = frame.shape[:2]

            # --- 0. BRIGHTNESS (night only / low-light) ---
            if APPLY_BRIGHTNESS and VIDEO_PROFILE == "night":
                frame = adjust_brightness_v_channel(frame, BRIGHTNESS_BETA)
            # --- 0b. CONTRAST (night only / low-light) ---
            if APPLY_CONTRAST and VIDEO_PROFILE == "night":
                frame = adjust_contrast_v_channel(frame, CONTRAST_ALPHA)

            # --- 1. APPLY ROI FIRST ---
            # This blacks out everything outside the trapezoid immediately
            roi_image, roi_mask = region_of_interest(frame, show_debug=True, return_mask=True)
            
            # --- 2. Filter White (on the ROI image) ---
            white_lanes = filter_white_pixels(roi_image, show_debug=True)
            
            # --- 3. Canny Edges ---
            canny_image = canny_edge_detector(white_lanes, show_debug=True)
            
            # --- 4. Hough Transform ---
            all_lines = detect_hough_lines(canny_image, 1, np.pi / 180, 30, frame, show_debug=True)
            
            left_cand, right_cand = get_good_lane_lines(all_lines, frame_height, frame_width)
            curr_left = get_average_lane(left_cand)
            curr_right = get_average_lane(right_cand)
            
            left_line_history.append(curr_left)
            if len(left_line_history) > HISTORY_LENGTH: left_line_history.pop(0)
            
            right_line_history.append(curr_right)
            if len(right_line_history) > HISTORY_LENGTH: right_line_history.pop(0)

            smooth_left = get_weighted_average(left_line_history)
            smooth_right = get_weighted_average(right_line_history)
            
            # 4. Lane Change Logic & Drawing
            if smooth_left[0] is not None and smooth_right[0] is not None:
                # We have lanes, so we can calculate position and trend
                draw_lane_polygon(frame, smooth_left, smooth_right)
                
                # Update last known valid lanes
                last_valid_smooth_left = smooth_left
                last_valid_smooth_right = smooth_right

                # Calculate center x at bottom of crop
                x_left = get_x_coordinate(smooth_left[0], smooth_left[1], h)
                x_right = get_x_coordinate(smooth_right[0], smooth_right[1], h)
                current_lane_center = (x_left + x_right) / 2
                
                if prev_lane_center is not None:
                    movement = current_lane_center - prev_lane_center
                    # Exponential moving average for trend to smooth out jitter
                    lane_center_trend = 0.9 * lane_center_trend + 0.1 * movement
                
                prev_lane_center = current_lane_center
                lane_change_status = "" # Reset status when lanes are found
                lane_change_candidate = None
                lane_change_candidate_frames = 0
                
            else:
                # Lanes are lost (History not available)
                # Determine direction based on last known trend
                # If trend > 0 (Lanes moved Right) -> Car moved Left
                # If trend < 0 (Lanes moved Left) -> Car moved Right
                if abs(lane_center_trend) > 0.4: # Threshold to ignore minor drift
                    candidate = "left" if lane_center_trend > 0 else "right"

                    if lane_change_candidate == candidate:
                        lane_change_candidate_frames += 1
                    else:
                        lane_change_candidate = candidate
                        lane_change_candidate_frames = 1

                    # Only confirm/display after MIN_LANE_CHANGE_FRAMES consecutive frames.
                    if lane_change_candidate_frames >= MIN_LANE_CHANGE_FRAMES:
                        new_status = (
                            "Changing lanes to the Left"
                            if lane_change_candidate == "left"
                            else "Changing lanes to the Right"
                        )
                        if lane_change_status != new_status:
                            print(
                                f"[Lane Change Detected] {new_status} "
                                f"(trend={lane_center_trend:.3f}, frames={lane_change_candidate_frames})"
                            )
                            lane_change_status = new_status
                    else:
                        # Not enough evidence yet -> keep original output (no lane-change text)
                        lane_change_status = ""
                        if last_valid_smooth_left is not None and last_valid_smooth_right is not None:
                            draw_lane_polygon(frame, last_valid_smooth_left, last_valid_smooth_right)
                else:
                    # No strong trend -> reset candidate and keep original polygon if available
                    lane_change_candidate = None
                    lane_change_candidate_frames = 0
                    lane_change_status = ""
                    if last_valid_smooth_left is not None and last_valid_smooth_right is not None:
                        draw_lane_polygon(frame, last_valid_smooth_left, last_valid_smooth_right)
            
            # Display status on the main frame
            if lane_change_status:
                text_size = cv2.getTextSize(lane_change_status, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
                text_x = (frame_width - text_size[0]) // 2
                text_y = 100
                
                # Draw semi-transparent box
                overlay = frame.copy()
                cv2.rectangle(overlay, (text_x - 10, text_y - 40), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                cv2.putText(frame, lane_change_status, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 
                            1.2, (0, 255, 255), 2, cv2.LINE_AA)
                
            cv2.imshow("Debug 4: Final Output", frame)
            out.write(frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Done! Video saved.")

if __name__ == "__main__":
    process_video()