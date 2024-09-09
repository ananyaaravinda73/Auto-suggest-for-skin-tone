import cv2
import numpy as np

# Define ranges for skin color in HSV color space
def get_skin_mask(hsv_frame):
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)  # Lower bound for skin color
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)  # Upper bound for skin color
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
    return mask

# Classify skin tone based on average hue
def classify_skin_tone(hue_value):
    if hue_value < 10:
        return "Very Fair", ["Bright colors", "Dark colors", "Pastels"]
    elif hue_value < 20:
        return "Fair", ["Light colors", "Medium tones", "Pastels"]
    elif hue_value < 30:
        return "Brown", ["Earth tones", "Warm colors", "Bold colors"]
    else:
        return "Dark", ["Jewel tones", "Bold colors", "White"]

# Start video capture from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the skin mask
    skin_mask = get_skin_mask(hsv_frame)

    # Extract the skin region
    skin_region = cv2.bitwise_and(hsv_frame, hsv_frame, mask=skin_mask)

    # Calculate the average hue value in the skin region
    hue_values = skin_region[:, :, 0][skin_mask > 0]  # Extract hue values where mask is non-zero

    if len(hue_values) > 0:
        avg_hue = np.mean(hue_values)
        skin_tone, clothing_suggestions = classify_skin_tone(avg_hue)
        cv2.putText(frame, f'Skin Tone: {skin_tone}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Suggested Colors: {", ".join(clothing_suggestions)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No skin detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with skin tone and suggestions
    cv2.imshow("Skin Tone Detector", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

