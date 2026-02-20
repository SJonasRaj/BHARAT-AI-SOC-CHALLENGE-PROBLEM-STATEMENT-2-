import cv2
import time
import os

def capture_profiles(user_name):
    cap = cv2.VideoCapture(0)
    folder = "pro"
    if not os.path.exists(folder): os.makedirs(folder)
    
    poses = ["FRONT", "LEFT_SIDE", "RIGHT_SIDE", "LOOK_UP", "LOOK_DOWN"]
    
    print(f"Starting capture for {user_name}...")
    for pose in poses:
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            cv2.putText(frame, f"GET READY for {pose}: {i}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capture Profiles", frame)
            cv2.waitKey(1000)
        
        ret, frame = cap.read()
        filename = f"{folder}/{user_name}_{pose}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter your name: ")
    capture_profiles(name)
