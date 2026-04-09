python3 -c "
import frame_capture
import cv2
frame_capture.open_camera()
import time; time.sleep(2)
f = frame_capture.grab_frame()
cv2.imwrite('/tmp/test_frame.jpg', f)
frame_capture.close_camera()
print('saved to /tmp/test_frame.jpg')
"
