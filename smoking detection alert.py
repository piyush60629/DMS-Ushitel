import cv2
import mediapipe as mp

# Initialize MediaPipe face and hands modules
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the face detection and hand tracking models
face_detection = mp_face_detection.FaceDetection()
hands = mp_hands.Hands()

# Initialize a flag to track smoking detection
smoking_detected = False

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert the frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_results = face_detection.process(frame_rgb)

    if face_results.detections:
        for detection in face_results.detections:
            ih, iw, _ = frame.shape
            bbox = (
                int(detection.location_data.relative_bounding_box.xmin * iw),
                int(detection.location_data.relative_bounding_box.ymin * ih),
                int(detection.location_data.relative_bounding_box.width * iw),
                int(detection.location_data.relative_bounding_box.height * ih)
            )

            # Draw the face bounding box
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)

            # Extract the coordinates of the lip (you may need to adjust the landmark index)
            lip_x = int(bbox[0] + bbox[2] / 2)
            lip_y = int(bbox[1] + bbox[3])

    # Detect and track hand landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)

    if hand_results.multi_hand_landmarks:
        for landmarks in hand_results.multi_hand_landmarks:
            # Extract hand landmarks
            hand_landmarks = landmarks.landmark

            # Get the coordinates of the index finger and middle finger
            index_x, index_y = int(hand_landmarks[8].x * frame.shape[1]), int(hand_landmarks[8].y * frame.shape[0])
            middle_x, middle_y = int(hand_landmarks[12].x * frame.shape[1]), int(hand_landmarks[12].y * frame.shape[0])

            # Calculate the distance between the index finger and middle finger
            distance = ((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2) ** 0.5

            # Define a threshold for finger proximity to the lips (adjust as needed)
            proximity_threshold = 50  # Adjust this value based on your preference

            # Check if the index and middle fingers are in proximity to the lips
            if distance < proximity_threshold and abs(index_x - lip_x) < proximity_threshold:
                # Set the smoking detection flag
                smoking_detected = True
            else:
                smoking_detected = False

    # Check the smoking detection flag and display an alert
    if smoking_detected:
        cv2.putText(frame, "Smoking detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()