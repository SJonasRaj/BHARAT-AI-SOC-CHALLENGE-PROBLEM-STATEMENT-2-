import cv2
import mediapipe as mp
import time
import threading
import subprocess
import math
import face_recognition
import numpy as np
import os
from pynput.keyboard import Key, Controller

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720
AI_INPUT_SIZE = 512  
AI_SIZE = (AI_INPUT_SIZE, AI_INPUT_SIZE)

SKIP_FRAMES = 2
MODEL_COMPLEXITY = 1
DETECTION_CONFIDENCE = 0.8
TRACKING_CONFIDENCE = 0.8

# HOLD THRESHOLDS (Seconds)
SKIP_HOLD = 2.0
PLAY_HOLD = 1.0
MUTE_HOLD = 1.0
SEEK_HOLD = 1.2
LOCK_HOLD = 1.5 

class CameraStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()
        self.new_frame_event = threading.Event()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if ret:
                self.frame = frame
                self.new_frame_event.set()

    def read(self):
        self.new_frame_event.wait(timeout=1.0)
        self.new_frame_event.clear()
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class GestureController:
    def __init__(self):
        self.keyboard = Controller()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
            model_complexity=MODEL_COMPLEXITY
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.last_action_time = 0
        self.cooldown = 0.5
        self.current_gest = "None"
        self.gesture_start_time = 0
        self.pending_gesture = None
        self.hold_progress = 0.0
        self.is_maximized = False
        self.is_locked = False 
        
        # --- MULTI-USER FACE RECOGNITION INIT ---
        self.face_authorized = False
        self.auth_grace_frames = 0
        self.authorized_encodings = []
        
        user_folder = "pro"
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
            print(f"Created folder '{user_folder}'.")

        print("--- LOADING AUTHORIZED USERS ---")
        for filename in os.listdir(user_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(user_folder, filename)
                try:
                    # Load and encode
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        # Append the first face found in the image
                        self.authorized_encodings.append(encodings[0])
                        print(f"SUCCESS: Loaded {filename}")
                    else:
                        print(f"WARNING: No face found in {filename}")
                except Exception as e:
                    print(f"ERROR: Could not process {filename} - {e}")
        
        print(f"--- TOTAL USERS READY: {len(self.authorized_encodings)} ---")

    def get_finger_state(self, hand_landmarks):
        lms = hand_landmarks.landmark
        palm_size = math.sqrt((lms[0].x - lms[9].x) ** 2 + (lms[0].y - lms[9].y) ** 2)
        fingers = []

        thumb_ext = abs(lms[4].x - lms[5].x)
        fingers.append(1 if thumb_ext > (palm_size * 0.45) else 0)

        for tip_id in [8, 12, 16, 20]:
            fingers.append(1 if lms[tip_id].y < lms[tip_id - 2].y else 0)

        return fingers

    def send_volume_cmd(self, xdotool_key, pynput_key):
        try:
            if pynput_key:
                self.keyboard.press(pynput_key)
                self.keyboard.release(pynput_key)
            cmd = f"xdotool search --onlyvisible --class vlc windowactivate key --clearmodifiers {xdotool_key}"
            subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
        except:
            pass

    def send_vlc_cmd(self, method):
        cmd = f"dbus-send --type=method_call --dest=org.mpris.MediaPlayer2.vlc /org/mpris/MediaPlayer2 org.mpris.MediaPlayer2.Player.{method}"
        subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)

    def send_seek_cmd(self, seconds):
        try:
            micro = int(seconds * 1_000_000)
            cmd = f"dbus-send --type=method_call --dest=org.mpris.MediaPlayer2.vlc /org/mpris/MediaPlayer2 org.mpris.MediaPlayer2.Player.Seek int64:{micro}"
            subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
        except:
            pass

    def execute_command(self, fingers, label):
        if not self.face_authorized:
            self.current_gest = "UNKNOWN USER"
            return

        curr_time = time.time()
        detected_gest = "None"

        if fingers == [1, 1, 1, 1, 1]: detected_gest = "Play/Pause"
        elif fingers == [0, 1, 0, 0, 0]: detected_gest = "Vol Up"
        elif fingers == [0, 1, 1, 0, 0]: detected_gest = "Vol Down"
        elif fingers == [0, 0, 0, 0, 0]: detected_gest = "Mute"
        elif fingers == [1, 0, 0, 0, 0]: detected_gest = "Next" if label == "Right" else "Prev"
        elif fingers == [1, 1, 0, 0, 0]: detected_gest = "Seek Forward" if label == "Right" else "Seek Backward"
        elif sum(fingers) == 3: detected_gest = "Max/Min Toggle"
        elif fingers == [0, 0, 0, 0, 1]: detected_gest = "Lock Toggle" 

        if detected_gest == "Lock Toggle":
            if self.pending_gesture == "Lock Toggle":
                elapsed = curr_time - self.gesture_start_time
                self.hold_progress = min(elapsed / LOCK_HOLD, 1.0)
                self.current_gest = "Unlocking..." if self.is_locked else "Locking..."
                
                if elapsed >= LOCK_HOLD and curr_time - self.last_action_time > self.cooldown:
                    self.is_locked = not self.is_locked
                    self.last_action_time = curr_time
                    self.pending_gesture = None
                    self.hold_progress = 0
            else:
                self.pending_gesture = "Lock Toggle"
                self.gesture_start_time = curr_time
            return

        if self.is_locked:
            self.current_gest = "SYSTEM LOCKED"
            self.pending_gesture = None
            self.hold_progress = 0
            return

        if detected_gest == "Max/Min Toggle":
            if curr_time - self.last_action_time > self.cooldown:
                if not self.is_maximized:
                    subprocess.run("xdotool search --onlyvisible --class vlc windowactivate key f", shell=True, stderr=subprocess.DEVNULL)
                    self.is_maximized = True
                else:
                    subprocess.run("xdotool search --class vlc windowminimize", shell=True, stderr=subprocess.DEVNULL)
                    self.is_maximized = False
                self.last_action_time = curr_time
                self.current_gest = detected_gest
            return

        if detected_gest in ["Vol Up", "Vol Down"]:
            self.pending_gesture = None
            self.hold_progress = 0
            if curr_time - self.last_action_time > self.cooldown:
                if detected_gest == "Vol Up": self.send_volume_cmd("Up", Key.media_volume_up)
                else: self.send_volume_cmd("Down", Key.media_volume_down)
                self.current_gest = detected_gest
                self.last_action_time = curr_time - 0.2
            return

        timed_actions = ["Play/Pause", "Next", "Prev", "Seek Forward", "Seek Backward", "Mute"]
        if detected_gest in timed_actions:
            target_hold = SEEK_HOLD if "Seek" in detected_gest else (
                PLAY_HOLD if detected_gest == "Play/Pause" else (
                    MUTE_HOLD if detected_gest == "Mute" else SKIP_HOLD))

            if self.pending_gesture == detected_gest:
                elapsed = curr_time - self.gesture_start_time
                self.hold_progress = min(elapsed / target_hold, 1.0)
                self.current_gest = f"Holding {detected_gest}"

                if elapsed >= target_hold and curr_time - self.last_action_time > self.cooldown:
                    if detected_gest == "Seek Forward": self.send_seek_cmd(10)
                    elif detected_gest == "Seek Backward": self.send_seek_cmd(-10)
                    elif detected_gest == "Play/Pause": self.send_vlc_cmd("PlayPause")
                    elif detected_gest == "Mute": self.send_volume_cmd("m", Key.media_volume_mute)
                    elif detected_gest == "Next": self.send_vlc_cmd("Next")
                    else: self.send_vlc_cmd("Previous")
                    self.last_action_time = curr_time
                    self.pending_gesture = None
            else:
                self.pending_gesture = detected_gest
                self.gesture_start_time = curr_time
                self.hold_progress = 0
        else:
            self.pending_gesture = None
            self.hold_progress = 0
            if curr_time - self.last_action_time > 1.2:
                self.current_gest = "None"

def main():
    try: subprocess.run(["sudo", "jetson_clocks"], stderr=subprocess.DEVNULL)
    except: pass

    cap = CameraStream().start()
    controller = GestureController()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    prev_frame_time = time.time()
    display_fps, fps_smoothing = 0, 0.9
    frame_count = 0
    results = None
    
    latency_ms, latency_smoothing = 0.0, 0.9
    accuracy_smooth, accuracy_smoothing = 0.0, 0.9

    while True:
        frame_raw = cap.read()
        if frame_raw is None: continue

        current_time = time.time()
        time_diff = current_time - prev_frame_time
        prev_frame_time = current_time
        if time_diff > 0:
            actual_fps = 1.0 / time_diff
            display_fps = (fps_smoothing * display_fps) + ((1.0 - fps_smoothing) * actual_fps)

        frame_count += 1
        frame = frame_raw[:, ::-1, :].copy()

        # --- FACE RECOGNITION LOGIC ---
        if frame_count % 15 == 0 and controller.authorized_encodings:
            rgb_small = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)[:, :, ::-1]
            rgb_small = np.ascontiguousarray(rgb_small)
            
            face_locs = face_recognition.face_locations(rgb_small, number_of_times_to_upsample=1, model="hog")
            
            if len(face_locs) > 0:
                try:
                    face_encodings = face_recognition.face_encodings(rgb_small, face_locs)
                    any_match = False
                    for enc in face_encodings:
                        # Compare against ALL loaded encodings in one go
                        matches = face_recognition.compare_faces(controller.authorized_encodings, enc, tolerance=0.7)
                        if True in matches:
                            any_match = True
                            break
                    
                    if any_match:
                        controller.face_authorized = True
                        controller.auth_grace_frames = 5 
                    else:
                        controller.auth_grace_frames -= 1
                except:
                    controller.auth_grace_frames -= 1
            else:
                controller.auth_grace_frames -= 1

            if controller.auth_grace_frames <= 0:
                controller.face_authorized = False
                controller.auth_grace_frames = 0

        # --- GESTURE INFERENCE ---
        if frame_count % SKIP_FRAMES == 0:
            infer_start = time.time()
            small = cv2.resize(frame, AI_SIZE, interpolation=cv2.INTER_NEAREST)
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            rgb = cv2.cvtColor(limg, cv2.COLOR_Lab2RGB)
            results = controller.hands.process(rgb)
            
            infer_end = time.time()
            raw_latency = (infer_end - infer_start) * 1000.0
            latency_ms = (latency_smoothing * latency_ms) + ((1.0 - latency_smoothing) * raw_latency)

        if results and results.multi_hand_landmarks:
            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                controller.mp_draw.draw_landmarks(frame, hand_lms, controller.mp_hands.HAND_CONNECTIONS)
                hand_conf = results.multi_handedness[i].classification[0].score
                accuracy_smooth = (accuracy_smoothing * accuracy_smooth + (1.0 - accuracy_smoothing) * hand_conf)
                
                label = results.multi_handedness[i].classification[0].label
                state = controller.get_finger_state(hand_lms)
                controller.execute_command(state, label)
        else:
            accuracy_smooth = 0.0

        if frame_count % 30 == 0:
            print(f"ACCURACY: {accuracy_smooth * 100:.2f}% | AUTH: {controller.face_authorized} | LATENCY: {latency_ms:.2f} ms")

        # --- COMPACT UI ---
        if not controller.face_authorized:
            ui_color = (0, 0, 180) 
        elif controller.is_locked:
            ui_color = (180, 0, 0) 
        else:
            ui_color = (20, 20, 20) 

        cv2.rectangle(frame, (10, 10), (250, 75), ui_color, -1)
        cv2.rectangle(frame, (10, 10), (250, 75), (200, 200, 200), 1)

        cv2.putText(frame, f"FPS: {int(display_fps)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        status_text = "LOCKED" if controller.is_locked and controller.current_gest == "None" else controller.current_gest
        if not controller.face_authorized: status_text = "ACCESS DENIED"
        
        cv2.putText(frame, f"ST: {status_text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if controller.hold_progress > 0:
            bar_width = int(220 * controller.hold_progress)
            cv2.rectangle(frame, (15, 82), (235, 88), (50, 50, 50), -1)
            cv2.rectangle(frame, (15, 82), (15 + bar_width, 88), (0, 255, 0), -1)

        cv2.imshow("Jetson Orin Nano HCI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
