import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_seg = mp.solutions.selfie_segmentation
pose = mp_pose.Pose(enable_segmentation=True, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
seg = mp_seg.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(0)
current_style = 0
styles = ["HEATMAP", "DOTS", "NEON", "SKETCH", "FLUID"]

cv2.namedWindow("Human Visualizer", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # Run MediaPipe
    pose_results = pose.process(rgb)
    hands_results = hands.process(rgb)
    seg_results = seg.process(rgb)

    mask = seg_results.segmentation_mask  # float32, 0-1
    mask_bin = (mask > 0.5).astype(np.uint8)

    # ---- STYLE RENDERING ----
    canvas = np.zeros_like(frame)

    if styles[current_style] == "HEATMAP":
        blurred = cv2.GaussianBlur(mask, (51, 51), 0)
        heat = cv2.applyColorMap((blurred * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        canvas = cv2.bitwise_and(heat, heat, mask=mask_bin)

    elif styles[current_style] == "DOTS":
        ys, xs = np.where(mask_bin > 0)
        if len(ys) > 0:
            indices = np.random.choice(len(ys), min(3000, len(ys)), replace=False)
            for i in indices:
                color = (np.random.randint(0,100), np.random.randint(150,255), np.random.randint(200,255))
                cv2.circle(canvas, (xs[i], ys[i]), 2, color, -1)

    elif styles[current_style] == "NEON":
        # Draw pose connections
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            connections = mp_pose.POSE_CONNECTIONS
            for conn in connections:
                p1 = lm[conn[0]]
                p2 = lm[conn[1]]
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw hand and finger connections
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Draw hand connections
                connections = mp_hands.HAND_CONNECTIONS
                for conn in connections:
                    p1 = hand_landmarks.landmark[conn[0]]
                    p2 = hand_landmarks.landmark[conn[1]]
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta for hands
                
                # Draw finger landmarks as dots
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1)  # Yellow dots for joints
        
        # Apply glow effect to both pose and hands
        for r, a in [(15, 0.3), (7, 0.5), (3, 1.0)]:
            blur = cv2.GaussianBlur(canvas, (r*2+1, r*2+1), 0)
            canvas = cv2.addWeighted(canvas, 1.0, blur, a, 0)

    elif styles[current_style] == "SKETCH":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_masked = cv2.bitwise_and(edges, edges, mask=mask_bin)
        canvas = cv2.cvtColor(edges_masked, cv2.COLOR_GRAY2BGR)
        canvas = cv2.bitwise_not(canvas)
        canvas[mask_bin == 0] = [200, 200, 200]

    elif styles[current_style] == "FLUID":
        t = cv2.getTickCount() / cv2.getTickFrequency()
        rows = np.arange(h).astype(np.float32)
        shift = (np.sin(rows * 0.05 + t * 2) * 10).astype(np.float32)
        map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
        map_x += shift[:, None]
        map_y = np.tile(rows, (w, 1)).T
        colored = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_OCEAN)
        warped = cv2.remap(colored, map_x, map_y, cv2.INTER_LINEAR)
        canvas = cv2.bitwise_and(warped, warped, mask=mask_bin)

    # ---- UI OVERLAY ----
    # Cute style name with rounded rectangle background
    cv2.rectangle(canvas, (5, 5), (280, 45), (255, 182, 193), -1)  # Light pink background
    cv2.rectangle(canvas, (5, 5), (280, 45), (255, 105, 180), 2)     # Pink border
    cv2.putText(canvas, f"* {styles[current_style]} *", (15, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Cute controls with pastel background
    cv2.rectangle(canvas, (0, h-45), (w, h), (176, 224, 230), -1)  # Powder blue background
    cv2.putText(canvas, "(^.~) 1-5 = select style   >_> SPACE = next style   :) Q = quit", (15, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (75, 0, 130), 1)  # Indigo text

    # No human warning with cute styling
    if mask_bin.sum() < 1000:
        cv2.putText(canvas, ":( No human detected", (w//2 - 130, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 182, 193), 3)
        cv2.putText(canvas, ":( No human detected", (w//2 - 130, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 105, 180), 2)

    # ---- LIVE FEED CORNER ----
    # Create small live feed in top-right corner with cute frame
    corner_size = 160
    small_frame = cv2.resize(frame, (corner_size, corner_size))
    
    # Add cute pastel border
    border_color = (255, 182, 193) if mask_bin.sum() >= 1000 else (255, 218, 185)
    small_frame = cv2.copyMakeBorder(small_frame, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=border_color)
    
    # Position in top-right corner
    start_x = w - corner_size - 10
    start_y = 10
    canvas[start_y:start_y+corner_size+8, start_x:start_x+corner_size+8] = small_frame
    
    # Add cute "LIVE" text with heart
    cv2.putText(canvas, "(^._.^)/ LIVE", (start_x + 10, start_y + corner_size + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 105, 180), 1)

    cv2.imshow("Human Visualizer", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        current_style = (current_style + 1) % len(styles)
    elif key >= ord('1') and key <= ord('5'):
        style_index = key - ord('1')
        if style_index < len(styles):
            current_style = style_index

cap.release()
cv2.destroyAllWindows()
