import cv2
import mediapipe as mp
import time
import random
import math
import numpy as np

class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def countFingers(self, lmList):
        if len(lmList) == 0:
            return 0
        
        fingers = []
        
        # Thumb (Check x-coordinates relative to wrist to determine if open)
        # Assuming Right Hand facing camera
        if lmList[4][1] < lmList[3][1]: # Thumb
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        total = fingers.count(1)
        return total

class GameEngine:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.level = 1
        self.score = 0
        self.state = "PLAYING" # PLAYING, GAME_OVER
        
        self.target_count = 0
        self.correct_box_id = 0
        self.box_contents = [] # List of dicts
        self.choice_positions = [] # (x, y)
        self.choice_angles = []    # Current angle for moving boxes
        
        # Adaptive difficulty params
        self.max_number = 3
        self.num_choices = 3
        self.mixed_shapes = False
        self.moving_boxes = False
        
        # Timer & Delay Settings
        self.level_duration = 5.0  # 5 seconds per level
        self.input_delay = 1.0     # Wait 1 second before detecting
        self.level_start_time = time.time()
        
        # Shape types
        self.available_shapes = ['circle', 'square', 'triangle', 'diamond']
        
        self.generate_level()
        
        self.required_hold_duration = 1.0 # seconds to lock answer
        self.current_match_time = 0
        self.last_detected_number = -1

    def generate_level(self):
        # Reset Timer
        self.level_start_time = time.time()
        
        # --- ADAPTIVE LOGIC ---
        # Level 1-2: Easy (Static, Uniform Shapes, 1-3 items)
        if self.level <= 2:
            self.max_number = 3
            self.num_choices = 3
            self.mixed_shapes = False
            self.moving_boxes = False
            
        # Level 3-5: Medium (Static, Mixed Shapes, 1-5 items)
        elif self.level <= 5:
            self.max_number = 5
            self.num_choices = 3
            self.mixed_shapes = True
            self.moving_boxes = False
            
        # Level 6+: Hard (Moving, Mixed Shapes, 1-6 items, 4 choices)
        else:
            self.max_number = 6
            self.num_choices = 4
            self.mixed_shapes = True
            self.moving_boxes = True

        # 1. Generate Central Target
        self.target_count = random.randint(1, self.max_number)
        
        # Central target shapes
        if self.mixed_shapes:
            self.target_shapes_list = [random.choice(self.available_shapes) for _ in range(self.target_count)]
        else:
            s = random.choice(self.available_shapes)
            self.target_shapes_list = [s] * self.target_count
        
        # 2. Pick Winner Box ID
        self.correct_box_id = random.randint(1, self.num_choices)
        
        # 3. Generate Box Contents
        self.box_contents = []
        
        for i in range(1, self.num_choices + 1):
            box_id = i
            
            if box_id == self.correct_box_id:
                # Correct Box
                shapes = self.target_shapes_list.copy()
                count = self.target_count
            else:
                # Distractor Box
                wrong_count = random.randint(1, self.max_number)
                while wrong_count == self.target_count:
                    wrong_count = random.randint(1, self.max_number)
                
                count = wrong_count
                if self.mixed_shapes:
                    shapes = [random.choice(self.available_shapes) for _ in range(count)]
                else:
                    s = random.choice(self.available_shapes)
                    shapes = [s] * count
            
            self.box_contents.append({
                'id': box_id,
                'count': count,
                'shapes_list': shapes
            })
        
        # Initialize positions
        self.choice_angles = []
        for i in range(self.num_choices):
            angle = (2 * math.pi / self.num_choices) * i - (math.pi / 2)
            self.choice_angles.append(angle)
            
        self.calculate_positions()
        self.current_match_time = 0
        self.last_detected_number = -1

    def calculate_positions(self):
        self.choice_positions = []
        cx, cy = self.width // 2, self.height // 2
        radius = 200 
        
        for angle in self.choice_angles:
            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle))
            self.choice_positions.append((x, y))

    def draw_specific_shape(self, img, shape_type, x, y, size, color):
        if shape_type == 'circle':
            cv2.circle(img, (x, y), size, color, cv2.FILLED)
        elif shape_type == 'square':
            cv2.rectangle(img, (x-size, y-size), (x+size, y+size), color, cv2.FILLED)
        elif shape_type == 'triangle':
            pts = np.array([
                [x, y - size],        
                [x - int(size*0.86), y + int(size*0.5)], 
                [x + int(size*0.86), y + int(size*0.5)]  
            ], np.int32)
            cv2.fillPoly(img, [pts], color)
        elif shape_type == 'diamond':
            pts = np.array([
                [x, y - size],        
                [x + size, y],        
                [x, y + size],        
                [x - size, y]         
            ], np.int32)
            cv2.fillPoly(img, [pts], color)

    def draw_content_group(self, img, x, y, shapes_list, r=30, bg_color=(255, 255, 255), label=None):
        count = len(shapes_list)
        
        # Draw background bubble
        cv2.circle(img, (x, y), r + 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (x, y), r + 5, bg_color, cv2.FILLED)
        
        item_color = (255, 255, 255)
        item_size = 10 
        if count >= 5: item_size = 8
        
        offsets = []
        if count == 1: offsets = [(0, 0)]
        elif count == 2: offsets = [(-15, 0), (15, 0)]
        elif count == 3: offsets = [(0, -18), (-15, 12), (15, 12)]
        elif count == 4: offsets = [(-15, -15), (15, -15), (-15, 15), (15, 15)]
        elif count == 5: offsets = [(-15, -15), (15, -15), (0, 0), (-15, 15), (15, 15)]
        elif count == 6: offsets = [(-12, -20), (12, -20), (-20, 0), (20, 0), (-12, 20), (12, 20)]
        else: 
             offsets = [(random.randint(-20, 20), random.randint(-20, 20)) for _ in range(count)]
            
        for i, (dx, dy) in enumerate(offsets):
            if i < len(shapes_list):
                shape_type = shapes_list[i]
                self.draw_specific_shape(img, shape_type, x + dx, y + dy, item_size, item_color)

        if label is not None:
            tag_x, tag_y = int(x + r), int(y - r)
            cv2.circle(img, (tag_x, tag_y), 22, (50, 50, 50), cv2.FILLED)
            cv2.putText(img, str(label), (tag_x - 12, tag_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def update(self, img, fingers_detected):
        h, w, c = img.shape
        cx, cy = w // 2, h // 2

        # --- GAME OVER SCREEN ---
        if self.state == "GAME_OVER":
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            cv2.putText(img, "GAME OVER", (cx - 150, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            cv2.putText(img, f"Final Score: {self.score}", (cx - 120, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Press 'R' to Restart", (cx - 130, cy + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            return img
        
        # --- TIME MANAGEMENT ---
        elapsed = time.time() - self.level_start_time
        
        # 1. Grace Period Logic
        is_grace_period = elapsed < self.input_delay
        
        if is_grace_period:
            # Show Wait Message
            cv2.putText(img, "READY...", (cx - 70, cy - 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            # Disable detection during grace period
            fingers_detected = 0 
            
        else:
            # 2. Gameplay Timer Logic
            play_time = elapsed - self.input_delay
            time_left = self.level_duration - play_time
            
            # Draw Timer Bar
            timer_width = w - 100
            if time_left < 0: time_left = 0
            
            timer_fill = int((time_left / self.level_duration) * timer_width)
            timer_color = (0, 255, 0)
            if time_left < 2: timer_color = (0, 0, 255) # Red warning
            
            cv2.rectangle(img, (50, 20), (50 + timer_width, 40), (100, 100, 100), cv2.FILLED)
            cv2.rectangle(img, (50, 20), (50 + timer_fill, 40), timer_color, cv2.FILLED)
            cv2.putText(img, f"{time_left:.1f}s", (w // 2 - 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Check Timeout
            if time_left <= 0:
                self.score = max(0, self.score - 10) # Penalty
                self.state = "GAME_OVER"
                return img

        # --- UPDATE MOVEMENT ---
        if self.moving_boxes and not is_grace_period:
            speed = 0.01 + (self.level * 0.001)
            for i in range(len(self.choice_angles)):
                self.choice_angles[i] += speed
            self.calculate_positions()

        # 3. Draw Central Target
        pulse = int(math.sin(time.time() * 5) * 5)
        cv2.circle(img, (cx, cy), 90 + pulse, (255, 255, 255, 100), 2) 
        self.draw_content_group(img, cx, cy, self.target_shapes_list, r=70, bg_color=(0, 200, 200))
        cv2.putText(img, "MATCH COUNT", (cx - 80, cy - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 4. Draw Choices
        for idx, content in enumerate(self.box_contents):
            if idx < len(self.choice_positions):
                px, py = self.choice_positions[idx]
                box_id = content['id']
                
                color = (100, 100, 100) 
                scale = 0
                
                if box_id == fingers_detected and not is_grace_period:
                    if box_id == self.correct_box_id:
                         color = (0, 255, 0)
                    else:
                         color = (0, 0, 255)
                    scale = 10
                
                self.draw_content_group(img, px, py, content['shapes_list'], r=50 + scale, bg_color=color, label=box_id)

        # 5. Check Logic
        if fingers_detected > 0 and not is_grace_period:
            if fingers_detected == self.last_detected_number:
                if fingers_detected == self.correct_box_id:
                    # Correct Logic
                    self.current_match_time += (1.0 / 30.0) 
                    
                    bar_width = 200
                    bar_height = 20
                    bar_x = cx - bar_width // 2
                    bar_y = cy + 140
                    fill = min(1.0, self.current_match_time / self.required_hold_duration)
                    
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                    cv2.rectangle(img, (bar_x, bar_y), (int(bar_x + bar_width * fill), bar_y + bar_height), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, f"SELECTING {fingers_detected}...", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if self.current_match_time >= self.required_hold_duration:
                        points = 10 + (self.level * 5)
                        self.score += points
                        self.level += 1
                        self.level_up_anim(img)
                        self.generate_level()
                else:
                    # Wrong Logic
                    # Check if number is within valid range to consider it an attempt
                    if fingers_detected <= self.num_choices:
                         self.current_match_time += (1.0 / 30.0)
                         
                         # Red Progress bar for wrong choice
                         bar_width = 200
                         bar_height = 20
                         bar_x = cx - bar_width // 2
                         bar_y = cy + 140
                         fill = min(1.0, self.current_match_time / self.required_hold_duration)
                         
                         cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                         cv2.rectangle(img, (bar_x, bar_y), (int(bar_x + bar_width * fill), bar_y + bar_height), (0, 0, 255), cv2.FILLED)
                         cv2.putText(img, "LOCKING WRONG ANSWER...", (bar_x - 50, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                         if self.current_match_time >= self.required_hold_duration:
                             # LOCK IN WRONG ANSWER -> GAME OVER
                             self.score = max(0, self.score - 10)
                             self.state = "GAME_OVER"
            else:
                self.current_match_time = 0
                self.last_detected_number = fingers_detected
        else:
             self.current_match_time = 0
             self.last_detected_number = -1

        # 6. HUD
        cv2.putText(img, f"Level: {self.level}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Score: {self.score}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img

    def level_up_anim(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 255, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.putText(img, "CORRECT!", (self.width//2 - 120, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.imshow("Gesture Match", img)
        cv2.waitKey(1000)

def main():
    cap = cv2.VideoCapture(1)
    w, h = 1280, 720
    cap.set(3, w)
    cap.set(4, h)
    
    detector = HandDetector(detectionCon=0.75)
    
    _, img = cap.read()
    if img is None:
        print("Camera index 1 failed. Switching to index 0.")
        cap = cv2.VideoCapture(0)
        cap.set(3, w)
        cap.set(4, h)
        _, img = cap.read()
        if img is None:
            print("No camera found.")
            return
        
    game = GameEngine(img.shape[1], img.shape[0])
    
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1)
        
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        
        fingers = 0
        if len(lmList) != 0:
            fingers = detector.countFingers(lmList)
            cv2.putText(img, f"Hand: {fingers}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        img = game.update(img, fingers)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (w - 150, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Gesture Match", img)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('r'):
            game = GameEngine(img.shape[1], img.shape[0])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()