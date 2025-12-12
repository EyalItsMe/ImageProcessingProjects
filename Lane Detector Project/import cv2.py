import cv2
import numpy as np
import matplotlib.pyplot as plt 


# --- CONFIGURATION ---
VIDEO_INPUT = 'Dashcam highway.mp4'
VIDEO_OUTPUT = 'lane_detection_output.mp4'
HISTORY_LENGTH = 20 

# --- HELPER FUNCTIONS ---

def region_of_interest(image, show_debug=False):
    """
    Applies a trapezoid mask using specific coordinates.
    """
    height, width = image.shape[:2]
    polygons = np.array([
        [(100, height), (350, 250), (450, 250), (600, height)]
    ])
    
    mask = np.zeros_like(image)
    
    cv2.fillPoly(mask, polygons, (255, 255, 255)) # 3 channels for BGR image
    masked_image = cv2.bitwise_and(image, mask)
    
    if show_debug:
        cv2.imshow("Debug 0: ROI Mask", masked_image)
        
    return masked_image

def filter_white_pixels(image, show_debug=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Optimized for white lanes in shadow/sun
    lower_white = np.array([0, 0, 165]) 
    upper_white = np.array([180, 80, 255]) 
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

    for r_t in lines:
        rho = r_t[0, 0]
        theta = r_t[0, 1]
        angle_deg = np.degrees(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        if abs(a) < 0.001: continue 
        x_bottom = int((rho - height * b) / a)

        if 35 < angle_deg < 55:
            if 130 < x_bottom < 250: 
                good_left_lines.append((rho, theta))
        elif 120 < angle_deg < 140:
            if 450 < x_bottom < 570:
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

    # Use the Top Y of your trapezoid (250)
    y_top = 250
    # Use the Bottom Y of your trapezoid (360)
    y_bottom = 360

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
    
    cap = cv2.VideoCapture(VIDEO_INPUT)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (frame_width, frame_height))
    
    print(f"Playing video... Press 'q' to stop.")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break 
            
            h,w = frame.shape[:2]
            # --- 1. APPLY ROI FIRST ---
            # This blacks out everything outside the trapezoid immediately
            roi_image = region_of_interest(frame, show_debug=True)
            
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
                
            else:
                # Lanes are lost (History not available)
                # Determine direction based on last known trend
                # If trend > 0 (Lanes moved Right) -> Car moved Left
                # If trend < 0 (Lanes moved Left) -> Car moved Right
                if abs(lane_center_trend) > 0.1: # Threshold to ignore minor drift
                    if lane_center_trend > 0:
                        new_status = "Changing lanes to the Left"
                    else:
                        new_status = "Changing lanes to the Right"

                    if lane_change_status != new_status:
                        print(f"[Lane Change Detected] {new_status}")
                        lane_change_status = new_status
                else:
                    # If lost lines but NO lane change detected, keep drawing old polygon
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