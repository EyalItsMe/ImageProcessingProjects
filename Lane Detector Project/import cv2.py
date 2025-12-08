import cv2
import numpy as np

# --- CONFIGURATION ---
VIDEO_INPUT = 'Dashcam highway.mp4'
VIDEO_OUTPUT = 'lane_detection_output.mp4'
CROP_Y = 250
HISTORY_LENGTH = 50 

# --- HELPER FUNCTIONS ---

def filter_white_pixels(image, white_threshold=200, show_debug=False):
    # hello 
    # hey
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary_image = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    if show_debug:
        cv2.imshow("Debug 1: White Filter", binary_image)
    return binary_image

def canny_edge_detector(image, show_debug=False):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    canny = cv2.Canny(blur, 130, 200)
    if show_debug:
        cv2.imshow("Debug 2: Canny Edges", canny)
    return canny

def detect_hough_lines(canny_image, rho_res, theta_res, threshold, color_image_for_debug, show_debug=False):
    all_lines = cv2.HoughLines(canny_image, rho_res, theta_res, threshold)
    if show_debug:
        hough_debug = color_image_for_debug.copy()
        if all_lines is not None:
            for r_t in all_lines:
                # Helper to draw raw lines for debug only
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
    if lines is None: 
        return good_left_lines, good_right_lines

    for r_t in lines:
        rho = r_t[0, 0]
        theta = r_t[0, 1]
        angle_deg = np.degrees(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        if abs(a) < 0.001: 
            continue 
        x_bottom = int((rho - height * b) / a)

        if 30 < angle_deg < 65:
            if 150 < x_bottom < (width / 2): 
                good_left_lines.append((rho, theta))
        elif 110 < angle_deg < 145:
            if 400 < x_bottom < 570:
                good_right_lines.append((rho, theta))
    return good_left_lines, good_right_lines

def get_average_lane(lines):
    if len(lines) == 0: return None, None
    rhos = [l[0] for l in lines]
    thetas = [l[1] for l in lines]
    return np.mean(rhos), np.mean(thetas)

def get_weighted_average(history):
    N = len(history)
    if N == 0: return None, None
    weights = np.arange(1, N + 1)
    rhos = np.array([h[0] for h in history])
    thetas = np.array([h[1] for h in history])
    return np.sum(weights * rhos) / np.sum(weights), np.sum(weights * thetas) / np.sum(weights)

def draw_lane_polygon(image, left_line, right_line):
    if left_line[0] is None or right_line[0] is None:
        return

    rho_l, theta_l = left_line
    rho_r, theta_r = right_line
    
    height, width = image.shape[:2]
    
    def get_x(rho, theta, y):
        a = np.cos(theta)
        b = np.sin(theta)
        return int((rho - y * b) / a)

    left_bottom  = (get_x(rho_l, theta_l, height), height)
    left_top     = (get_x(rho_l, theta_l, 0), 0)
    
    right_top    = (get_x(rho_r, theta_r, 0), 0)
    right_bottom = (get_x(rho_r, theta_r, height), height)
    
    pts = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    overlay = image.copy()
    
    cv2.fillPoly(overlay, pts, (255, 0, 0))
    cv2.addWeighted(overlay, 0.4, image, 1 - 0.4, 0, image)
    
    # Draw thicker borders for visibility
    cv2.line(image, left_bottom, left_top, (255, 0, 0), 3)
    cv2.line(image, right_bottom, right_top, (255, 0, 0), 3)

# --- MAIN EXECUTION ---

def process_video():
    left_line_history = []
    right_line_history = []
    
    cap = cv2.VideoCapture(VIDEO_INPUT)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (frame_width, frame_height))
    
    print(f"Playing video... Press 'q' to stop.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
        
        cropped_frame = frame[CROP_Y:, :]
        crop_h, crop_w = cropped_frame.shape[:2]
        
        # 1. Detect & Filter
        white_lanes = filter_white_pixels(cropped_frame, white_threshold=200, show_debug=True)
        canny_image = canny_edge_detector(white_lanes, show_debug=True)
        all_lines = detect_hough_lines(canny_image, 1, np.pi / 180, 30, cropped_frame, show_debug=True)
        
        left_cand, right_cand = get_good_lane_lines(all_lines, crop_h, crop_w)
        
        # Raw detection for this specific frame
        curr_left = get_average_lane(left_cand)
        curr_right = get_average_lane(right_cand)
        
        # 2. UPDATE HISTORY (Always update if detection is valid)
        if curr_left[0] is not None:
            left_line_history.append(curr_left)
            if len(left_line_history) > HISTORY_LENGTH:
                left_line_history.pop(0)
            
        if curr_right[0] is not None:
            right_line_history.append(curr_right)
            if len(right_line_history) > HISTORY_LENGTH:
                right_line_history.pop(0)

        # 3. CALCULATE SMOOTHED LINE
        smooth_left = get_weighted_average(left_line_history)
        smooth_right = get_weighted_average(right_line_history)
        
        # 4. DRAW
        if smooth_left[0] is not None and smooth_right[0] is not None:
            draw_lane_polygon(cropped_frame, smooth_left, smooth_right)
            
        frame[CROP_Y:, :] = cropped_frame
        cv2.imshow("Debug 4: Final Output", frame)
        out.write(frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    process_video()