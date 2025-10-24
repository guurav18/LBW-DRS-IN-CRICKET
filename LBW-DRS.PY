import cv2
import numpy as np
import pyttsx3

def speak_decision(decision):
    """
    Speaks the DRS decision using text-to-speech.
    This function will be used to say "OUT" or "NOT OUT".
    """
    engine = pyttsx3.init()
    engine.say(decision)
    engine.runAndWait()

def get_decision_from_tracking(trajectory, pitch_boundaries):
    """
    Analyzes the ball's trajectory to make an LBW decision based on
    a more robust set of cricket rules and an improved trajectory model.
    
    Args:
        trajectory (list): A list of (x, y) coordinates of the ball.
        pitch_boundaries (dict): A dictionary with calibrated pitch dimensions.
    """
    if not trajectory or len(trajectory) < 5: # Need more points for polyfit
        print("⚠️ Trajectory is too short to make an accurate decision.")
        return "NOT OUT"

    # Using the first point for pitching location and last point for impact
    pitch_x, pitch_y = trajectory[0]
    impact_x, impact_y = trajectory[-1]
    
    # Simple check for an "edge"
    # NOTE: This is a placeholder. A real system would need audio analysis
    # (snickometer) or bat-and-ball collision detection.
    edge = "No" 

    print("\n--- DRS ANALYSIS ---")
    print(f"Pitching at: ({pitch_x}, {pitch_y})")
    print(f"Impact at: ({impact_x}, {impact_y})")
    
    # 1. Check for an edge first (as per ICC rules)
    if edge == "Yes":
        print("Reason: Edge detected, not out.")
        return "NOT OUT"

    # 2. Determine pitching location based on calibrated boundaries
    # Pitching outside leg stump is not out.
    pitch_location = ""
    if pitch_x < pitch_boundaries["off_stump"]:
        pitch_location = "outside_off"
    elif pitch_x > pitch_boundaries["leg_stump"]:
        pitch_location = "outside_leg"
    else:
        pitch_location = "in_line"

    if pitch_location == "outside_leg":
        print("Reason: Ball pitched outside leg stump, not out.")
        return "NOT OUT"

    # 3. Determine impact location based on calibrated boundaries
    impact = ""
    if impact_x < pitch_boundaries["off_stump"] or impact_x > pitch_boundaries["leg_stump"]:
        impact = "outside"
    else:
        impact = "in_line"
    
    # If impact is outside off, the decision is not out unless the batsman
    # did not attempt a shot (which we are not simulating here).
    # We will assume a shot was played if the impact is outside off.
    if impact == "outside" and pitch_location != "outside_leg":
        print("Reason: Impact was outside the line of off stump, not out.")
        return "NOT OUT"
        
    # 4. Determine if the ball would have hit the stumps (height)
    # This is the key improvement using polynomial regression for better accuracy.
    x_values = np.array([p[0] for p in trajectory])
    y_values = np.array([p[1] for p in trajectory])

    # Fit a 2nd-degree polynomial (a parabola) to the ball's trajectory
    # The coefficients will model the curve of the ball's flight
    try:
        coefficients = np.polyfit(x_values, y_values, 2)
        poly_function = np.poly1d(coefficients)
    except np.linalg.LinAlgError:
        print("⚠️ Could not fit a polynomial curve. Returning NOT OUT.")
        return "NOT OUT"

    # Predict the y-coordinate at the stumps (a fixed x-value)
    # The x-value of the stumps is the middle of the pitch boundaries.
    stumps_x = (pitch_boundaries["off_stump"] + pitch_boundaries["leg_stump"]) / 2
    y_at_stumps = poly_function(stumps_x)
    
    # The height of the stumps can also be calibrated.
    # We'll use a placeholder value here.
    stumps_height_threshold = 300 
    
    height = "too_high"
    # A smaller y-value means it's lower and hitting the stumps
    if y_at_stumps > stumps_height_threshold:
        height = "hitting"

    print(f"Pitch: {pitch_location}, Impact: {impact}, Height: {height}")
    
    # 5. Final Decision Logic
    # Now that we have all three components, apply the final logic.
    if pitch_location in ["in_line", "outside_off"] and \
       impact == "in_line" and \
       height == "hitting":
        print("Reason: Pitched in line, impacted in line, and hitting the stumps. OUT!")
        return "OUT"
    else:
        print("Reason: Does not meet all conditions for OUT.")
        return "NOT OUT"

def track_ball(video_path):
    """
    Tracks the ball in a video and passes the trajectory for decision analysis.
    This version includes a more robust initial setup for ball detection.
    """
    cap = cv2.VideoCapture(video_path)
    ball_positions = []
    
    # --- Calibrate the pitch boundaries (simplified) ---
    # In a real system, you would get these values from a more complex
    # homography transformation, but for this simulation, we will
    # define them here. Adjust these values based on your video.
    PITCH_BOUNDARIES = {
        "off_stump": 250, # x-coordinate of the off stump
        "leg_stump": 400, # x-coordinate of the leg stump
        "stumps_y": 440   # y-coordinate of the stumps
    }
    
    # --- Improved ball color tracking (dynamic thresholds) ---
    # Instead of fixed values, let's allow for some dynamic range.
    # For a more advanced solution, you'd sample the ball's color
    # in the first few frames.
    lower_ball_color = np.array([5, 150, 150])
    upper_ball_color = np.array([15, 255, 255])
    
    # A variable to indicate if the ball was detected at all.
    ball_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Draw the calibrated pitch boundaries for visual feedback
        cv2.line(frame, (PITCH_BOUNDARIES["off_stump"], 0), (PITCH_BOUNDARIES["off_stump"], 480), (255, 0, 0), 2)
        cv2.line(frame, (PITCH_BOUNDARIES["leg_stump"], 0), (PITCH_BOUNDARIES["leg_stump"], 480), (255, 0, 0), 2)
        cv2.putText(frame, "Off Stump", (PITCH_BOUNDARIES["off_stump"] - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "Leg Stump", (PITCH_BOUNDARIES["leg_stump"] + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the ball color and perform noise reduction
        mask = cv2.inRange(hsv, lower_ball_color, upper_ball_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Filter out small detections that are likely noise
            if radius > 5:
                # Use moments to find the true center
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    ball_positions.append(center)
                    ball_detected = True
                    
                    # Draw a circle around the ball
                    cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                    cv2.putText(frame, f"Pos: ({center[0]}, {center[1]})", (center[0] + 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("DRS Simulation (Press 'q' to quit)", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if ball_detected:
        print("✅ Ball detected. Processing trajectory...")
        decision = get_decision_from_tracking(ball_positions, PITCH_BOUNDARIES)
        print("🏏 Final Decision:", decision)
        speak_decision(decision) # New line to speak the decision
    else:
        print("⚠️ Ball not detected properly! Please check video and color thresholds.")


# === Run the simulation ===
video_path = "xyz.mp4" # Make sure this file exists
track_ball(video_path)
