import cv2
import dlib
import numpy as np

# Fixed hyperparameters
EAR_THRESHOLD = 0.2  # Blink detection threshold
PITCH_THRESHOLD = 15  # Head tilt up/down threshold
YAW_THRESHOLD = 15    # Head turn left/right threshold

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(face, frame):
    """Extract facial landmarks."""
    shape = predictor(frame, face)
    return np.array([(p.x, p.y) for p in shape.parts()])

def get_gaze_direction(eye_landmarks):
    """Estimate gaze direction based on eye position."""
    eye_center = np.mean(eye_landmarks, axis=0)
    left_corner, right_corner = eye_landmarks[0], eye_landmarks[3]
    horizontal_ratio = (eye_center[0] - left_corner[0]) / (right_corner[0] - left_corner[0])
    
    if horizontal_ratio < 0.4:
        return "Looking Left"
    elif horizontal_ratio > 0.6:
        return "Looking Right"
    else:
        return "Looking Straight"

def estimate_head_pose(landmarks, frame_width, frame_height):
    """Estimate head pose using facial landmarks."""
    # Nose tip and center points
    nose_tip = landmarks[30]
    nose_center = landmarks[33]
    
    # Calculate head pose based on nose position relative to frame center
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    
    dx = nose_tip[0] - frame_center_x
    dy = nose_tip[1] - frame_center_y
    
    # Determine head pose based on nose position
    if abs(dx) > YAW_THRESHOLD:
        pose = "Head Turned Left" if dx < 0 else "Head Turned Right"
    elif abs(dy) > PITCH_THRESHOLD:
        pose = "Looking Up" if dy < 0 else "Looking Down"
    else:
        pose = "Neutral"
    
    return pose, dx, dy

# Tracking variables
total_blinks = 0

print("Focus Monitoring - Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = get_landmarks(face, gray)
        
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        gaze_direction = get_gaze_direction(left_eye)
        pose, dx, dy = estimate_head_pose(landmarks, frame.shape[1], frame.shape[0])
        
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        
        info_text = f"Pose: {pose} | Gaze: {gaze_direction}"
        angle_text = f"Yaw: {dx:.2f} | Pitch: {dy:.2f}"
        
        cv2.putText(frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, angle_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow('Focus Monitoring', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Monitoring Stopped.")

print(f"Total Blinks: {total_blinks}")