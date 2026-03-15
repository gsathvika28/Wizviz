import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import os
import time

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_seg = mp.solutions.selfie_segmentation
pose = mp_pose.Pose(enable_segmentation=True, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
seg = mp_seg.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(0)
current_style = 0
styles = ["HEATMAP", "CONSTELLATION"]

# Initialize star positions for smooth movement
star_positions = []
num_stars = 100
for _ in range(num_stars):
    star_positions.append({
        'x': np.random.randint(0, 640),  # Will be updated to actual width
        'y': np.random.randint(0, 480),  # Will be updated to actual height
        'speed': np.random.uniform(0.1, 0.3)  # Slow horizontal speed
    })

# Initialize cloud positions for smooth movement
cloud_positions = []
for _ in range(5):
    cloud_positions.append({
        'x': np.random.randint(0, 640),  # Will be updated to actual width
        'y': np.random.randint(50, 160),  # Upper third of screen
        'size': np.random.randint(40, 80),
        'speed': np.random.uniform(0.2, 0.5),
        'puffs': np.random.randint(3, 6)
    })

cv2.namedWindow("Wizviz", cv2.WINDOW_FREERATIO)
cv2.setWindowProperty("Wizviz", cv2.WND_PROP_FULLSCREEN, 1)

# Function definitions first
def load_and_play_music():
    """Load and play background music"""
    music_file = "music/music.mp3"
    if os.path.exists(music_file):
        try:
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.play(-1)  # Loop indefinitely
            print(f"Playing: {music_file}")
        except Exception as e:
            print(f"Error loading music: {e}")
    else:
        print(f"Music file not found: {music_file}")
        print("Please place your background music at: music/music.mp3")

def update_music_volume(human_detected):
    """Update music volume based on human detection"""
    global current_volume, target_volume
    
    # Set target volume based on detection
    if human_detected:
        target_volume = 0.7  # Higher volume when human present
    else:
        target_volume = 0.1  # Lower volume when no human
    
    # Smooth volume transition
    if abs(current_volume - target_volume) > 0.01:
        current_volume += (target_volume - current_volume) * volume_smoothing
        pygame.mixer.music.set_volume(max(0.0, min(1.0, current_volume)))

# Audio system initialization
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
pygame.mixer.music.set_volume(0.1)  # Start with low volume

# Audio variables
current_volume = 0.1
target_volume = 0.1
volume_smoothing = 0.05

# Initialize background music
load_and_play_music()

# Mode timing variables
mode_start_time = time.time()
mode_duration = 7  # 7 seconds per mode

# Gesture cooldown variables
last_gesture_time = 0
gesture_cooldown = 1.0  # 1 second cooldown between gesture detections

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

    # ---- GESTURE DETECTION FOR MODE SWITCHING ----
    gesture_detected = False
    target_mode = None
    
    # Check if enough time has passed since last gesture
    current_time = time.time()
    if current_time - last_gesture_time < gesture_cooldown:
        gesture_detected = False  # Skip gesture detection during cooldown
    else:
        # Store previous hand positions for velocity calculation
        if 'prev_thumb_tip' not in globals():
            prev_thumb_tip = None
            prev_index_tip = None
            prev_middle_tip = None
        
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Check for specific finger gestures
                if hand_landmarks:
                    # Get key finger landmarks
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                
                    # Get corresponding MCP joints (base of fingers)
                    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                
                    # Check if fingers are closed (tips lower than MCP = closed hand)
                    thumb_closed = thumb_tip.y > thumb_mcp.y
                    index_closed = index_tip.y > index_mcp.y
                    middle_closed = middle_tip.y > middle_mcp.y
                    ring_closed = ring_tip.y > ring_mcp.y
                    pinky_closed = pinky_tip.y > pinky_mcp.y
                
                    # Check for thumb and index touch gesture (constellation mode)
                    thumb_index_touch = False
                    if thumb_tip is not None and index_tip is not None:
                        # Calculate distance between thumb and index
                        touch_distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
                        
                        # Touch detected if fingers are close
                        if touch_distance < 0.05:
                            thumb_index_touch = True
                
                    # Check for thumb and middle touch gesture (heatmap mode)
                    thumb_middle_touch = False
                    if thumb_tip is not None and middle_tip is not None:
                        # Calculate distance between thumb and middle
                        touch_distance = np.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)
                        
                        # Touch detected if fingers are close
                        if touch_distance < 0.05:
                            thumb_middle_touch = True
                
                    # Thumb + Index touch = HEATMAP mode
                    if thumb_index_touch:
                        target_mode = "HEATMAP"
                        gesture_detected = True
                        last_gesture_time = current_time  # Update cooldown timer
                        print(f"Thumb-index touch detected! Switched to {styles[current_style]} mode")
                        break
                
                    # Thumb + Middle touch = CONSTELLATION mode
                    if thumb_middle_touch:
                        target_mode = "CONSTELLATION"
                        gesture_detected = True
                        last_gesture_time = current_time  # Update cooldown timer
                        print(f"Thumb-middle touch detected! Switched to {styles[current_style]} mode")
                        break
    
    # Switch mode on gesture detection
    if gesture_detected and target_mode == "CONSTELLATION":
        current_style = 1
        mode_start_time = time.time()
        print(f"Touch gesture detected! Switched to {styles[current_style]} mode")
    elif gesture_detected and target_mode == "HEATMAP":
        current_style = 0
        mode_start_time = time.time()
        print(f"Touch gesture detected! Switched to {styles[current_style]} mode")

    # ---- STYLE RENDERING ----
    canvas = np.zeros_like(frame)

    if styles[current_style] == "HEATMAP":
        # Create green grid background like thermal cameras
        canvas[:] = (0, 50, 0)  # Dark green background
        
        # Draw grid lines
        grid_spacing = 20
        grid_color = (0, 80, 0)  # Slightly lighter green for grid
        for x in range(0, w, grid_spacing):
            cv2.line(canvas, (x, 0), (x, h), grid_color, 1)
        for y in range(0, h, grid_spacing):
            cv2.line(canvas, (0, y), (w, y), grid_color, 1)
        
        # Create heat effect
        blurred = cv2.GaussianBlur(mask, (51, 51), 0)
        heat = cv2.applyColorMap((blurred * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        
        # Apply heat only to human silhouette
        heat_masked = cv2.bitwise_and(heat, heat, mask=mask_bin)
        
        # Add heat ripple effect - create wavy distortion
        ripple_amplitude = 2
        ripple_frequency = 0.1
        time_factor = 0  # Can be animated if needed
        
        for y in range(0, h, 3):  # Sample every 3 pixels for performance
            for x in range(0, w, 3):
                if mask_bin[y, x] > 0:
                    # Calculate ripple offset
                    offset_x = int(ripple_amplitude * np.sin(ripple_frequency * x + time_factor))
                    offset_y = int(ripple_amplitude * np.cos(ripple_frequency * y + time_factor))
                    
                    # Apply offset within bounds
                    src_x = max(0, min(w-1, x + offset_x))
                    src_y = max(0, min(h-1, y + offset_y))
                    
                    # Copy heat pixel with ripple
                    if heat_masked[src_y, src_x].any() > 0:
                        cv2.circle(canvas, (x, y), 2, heat_masked[src_y, src_x].tolist(), -1)
        
        # Blend the heat effect with the background
        canvas = cv2.addWeighted(canvas, 0.3, heat_masked, 0.7, 0)
        
        # Add heat noise - small dots here and there in small density
        num_noise_dots = 50  # Small density of noise dots
        for _ in range(num_noise_dots):
            noise_x = np.random.randint(0, w)
            noise_y = np.random.randint(0, h)
            
            # Random heat colors for noise (shades of blue)
            noise_colors = [
                (0, 0, 50),      # Dark blue
                (0, 50, 100),    # Medium blue
                (0, 100, 150),   # Light blue
                (0, 150, 200),   # Bright blue
                (100, 150, 255)  # Very bright blue
            ]
            noise_color = noise_colors[np.random.randint(0, len(noise_colors))]
            noise_size = np.random.randint(1, 3)  # Small dots
            
            cv2.circle(canvas, (noise_x, noise_y), noise_size, noise_color, -1)

    elif styles[current_style] == "CONSTELLATION":
        # Create constellation effect with connected dots
        # Update and draw moving stars
        for star in star_positions:
            # Update star position (move left)
            star['x'] -= star['speed']
            
            # Wrap around when star goes off screen
            if star['x'] < 0:
                star['x'] = w
                star['y'] = np.random.randint(0, h)
            
            # Only draw stars outside the human silhouette
            star_y_int = int(star['y'])
            star_x_int = int(star['x'])
            if 0 <= star_y_int < h and 0 <= star_x_int < w:
                if mask_bin[star_y_int, star_x_int] == 0:
                    cv2.circle(canvas, (star_x_int, star_y_int), 1, (100, 100, 150), -1)
        
        # Add a small crescent moon in top-right corner
        moon_x, moon_y = w - 80, 60
        # Main moon circle
        cv2.circle(canvas, (moon_x, moon_y), 15, (200, 200, 220), -1)
        # Crescent shadow (creates crescent shape)
        cv2.circle(canvas, (moon_x + 5, moon_y - 2), 14, (0, 0, 0), -1)
        
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
    # No mode name displayed
    
    # Update music volume based on human detection
    human_detected = mask_bin.sum() >= 1000
    update_music_volume(human_detected)
    
    # No human warning with cute styling
    if not human_detected:
        cv2.putText(canvas, ":( No human detected", (w//2 - 130, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 182, 193), 3)
        cv2.putText(canvas, ":( No human detected", (w//2 - 130, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 105, 180), 2)

    
    cv2.imshow("Wizviz", canvas)

    key = cv2.waitKey(1) & 0xFF
    
    # Check if it's time to switch modes
    current_time = time.time()
    if current_time - mode_start_time >= mode_duration:
        current_style = (current_style + 1) % len(styles)
        mode_start_time = current_time
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()  # Clean up audio system
