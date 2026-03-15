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
styles = ["HEATMAP", "DOTS", "CONSTELLATION"]

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
        # Find edges for better outline
        edges = cv2.Canny(mask_bin * 255, 50, 150)
        edge_ys, edge_xs = np.where(edges > 0)
        
        # Also add some interior points for density
        interior_ys, interior_xs = np.where(mask_bin > 0)
        
        all_points = []
        
        # Add edge points (higher priority for outline)
        if len(edge_ys) > 0:
            edge_indices = np.random.choice(len(edge_ys), min(400, len(edge_ys)), replace=False)
            all_points.extend(zip(edge_xs[edge_indices], edge_ys[edge_indices]))
        
        # Add some interior points for fill
        if len(interior_ys) > 0:
            interior_indices = np.random.choice(len(interior_ys), min(400, len(interior_ys)), replace=False)
            all_points.extend(zip(interior_xs[interior_indices], interior_ys[interior_indices]))
        
        # Draw all points
        colors = [
            (0, 150, 255),    # Blue
            (255, 100, 200),  # Pink  
            (100, 255, 200),  # Cyan
            (255, 200, 100),  # Orange
            (200, 100, 255)   # Purple
        ]
        
        for i, (x, y) in enumerate(all_points):
            color_idx = i % len(colors)
            cv2.circle(canvas, (x, y), 2, colors[color_idx], -1)

    elif styles[current_style] == "CONSTELLATION":
        # Create constellation effect with connected dots
        # Add small background stars first
        for _ in range(100):  # Add 100 random small stars in background
            star_x = np.random.randint(0, w)
            star_y = np.random.randint(0, h)
            # Only place stars outside the human silhouette
            if mask_bin[star_y, star_x] == 0:
                cv2.circle(canvas, (star_x, star_y), 1, (100, 100, 150), -1)  # Small dim stars
        
        # Sample points from the mask for constellation
        ys, xs = np.where(mask_bin > 0)
        if len(ys) > 0:
            # Select fewer points for constellation effect
            num_points = min(40, len(ys))  # Reduced from 60
            indices = np.random.choice(len(ys), num_points, replace=False)
            points = list(zip(xs[indices], ys[indices]))
            
            # Draw fewer connections between nearby points
            for i, (x1, y1) in enumerate(points):
                # Only connect to nearest 2-3 neighbors instead of all nearby points
                distances = []
                for j, (x2, y2) in enumerate(points[i+1:], i+1):
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if distance < 60:  # Reduced connection distance from 80 to 60
                        distances.append((distance, j, (x2, y2)))
                
                # Sort by distance and connect to closest 2 neighbors
                distances.sort()
                for dist, j, (x2, y2) in distances[:2]:  # Max 2 connections per point
                    cv2.line(canvas, (x1, y1), (x2, y2), (50, 50, 100), 1)
            
            # Draw bright dots at constellation points
            for i, (x, y) in enumerate(points):
                # Bright star-like dots
                cv2.circle(canvas, (x, y), 3, (255, 255, 200), -1)  # Bright yellow-white
                cv2.circle(canvas, (x, y), 1, (255, 255, 255), -1)   # White center


    # ---- UI OVERLAY ----
    # Cute style name with rounded rectangle background
    cv2.rectangle(canvas, (5, 5), (280, 45), (255, 182, 193), -1)  # Light pink background
    cv2.rectangle(canvas, (5, 5), (280, 45), (255, 105, 180), 2)     # Pink border
    cv2.putText(canvas, f"* {styles[current_style]} *", (15, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Cute controls with pastel background
    cv2.rectangle(canvas, (0, h-45), (w, h), (176, 224, 230), -1)  # Powder blue background
    cv2.putText(canvas, "(^.^) 1-3 = select style   >_> SPACE = next style   :) Q = quit", (15, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (75, 0, 130), 1)  # Indigo text

    # No human warning with cute styling
    if mask_bin.sum() < 1000:
        cv2.putText(canvas, ":( No human detected", (w//2 - 130, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 182, 193), 3)
        cv2.putText(canvas, ":( No human detected", (w//2 - 130, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 105, 180), 2)

    
    cv2.imshow("Human Visualizer", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        current_style = (current_style + 1) % len(styles)
    elif key >= ord('1') and key <= ord('3'):
        style_index = key - ord('1')
        if style_index < len(styles):
            current_style = style_index

cap.release()
cv2.destroyAllWindows()
