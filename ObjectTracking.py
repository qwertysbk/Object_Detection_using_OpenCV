import cv2

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Create the KCF tracker object
tracker = cv2.TrackerKCF_create()

# Define the initial bounding box
bbox = None
init_once = False

while True:
    # Read a frame from the video stream
    x, frame = cap.read()
    
    if not x:
        break
    
    # Initialize the tracker on the first frame
    if not init_once:
        bbox = cv2.selectROI("Tracking", frame, False)
        init_once = True
        
        # Initialize the tracker with the initial bounding box
        tracker.init(frame, bbox)
    else:
        # Update the tracker on subsequent frames
        ok, bbox = tracker.update(frame)
        
        if ok:
            # Draw the bounding box around the tracked object
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Object Lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)
    
    # Exit if ESC is pressed
    if cv2.waitKey(1) == 27:
        break
        
# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
