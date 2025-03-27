import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

# Webcam parameters
wcam, hcam = 640, 480
frameR = 100  # Frame reduction
smoothening = 5  # Cursor smoothening factor

# Initialize previous and current location variables
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

# Initialize hand detector
detector = htm.handDetector(maxHands=1)

# Get screen size
wScr, hScr = autopy.screen.size()
print("Screen Size:", wScr, hScr)

pTime = 0  # Initialize previous time for FPS calculation

while True:
    # 1. Capture frame
    success, img = cap.read()
    if not success or img is None:
        print("❌ Failed to capture frame")
        continue

    # 2. Detect hands
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 3. Check if hand landmarks exist
    if lmList:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        fingers = detector.fingersUp()

        # 4. Move mouse if only the index finger is up
        cv2.rectangle(img, (frameR, frameR), (wcam - frameR, hcam - frameR), (255, 0, 255), 2)
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wcam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hcam - frameR), (0, hScr))

            # Smooth the cursor movement
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move mouse
            autopy.mouse.move(wScr - clocX, clocY)

            # Draw cursor circle
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            plocX, plocY = clocX, clocY  # Update previous location

        # 5. Click mouse if both index and middle fingers are up
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 6. Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 7. Show image
    cv2.imshow("Image", img)

    # 8. Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Clean exit")

