# from flask import Flask, Response, jsonify, request, send_file
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime, timedelta
# import os
# import atexit
# from flask_cors import CORS
# from insightface.app import FaceAnalysis
# import threading
# import queue
# import time
# import sqlite3
# import torch
# import bcrypt
# import jwt

# # --- IMPORT LOGIC MODULE ---
# try:
#     import shirt_tuck  # Logic from Code 2
# except ImportError:
#     print("WARNING: shirt_tuck.py not found. Please ensure it exists.")

# app = Flask(__name__)
# CORS(app)
# app.config["SECRET_KEY"] = "mysecret_production_key"

# # ---------------- CONFIGURATION & HARDWARE ----------------
# print("--- SYSTEM STARTUP ---")
# print("YOLO GPU available:", torch.cuda.is_available())
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Running on: {DEVICE}")

# # Prevent OpenCV from spawning too many threads (conflicts with PyTorch)
# cv2.setNumThreads(1)

# # Locks for Thread Safety
# count_lock = threading.Lock()
# results_lock = threading.Lock()
# visual_lock = threading.Lock()  
# inference_lock = threading.Lock() # Prevents GPU contention

# # Queues
# processing_queue = queue.Queue(maxsize=10)
# log_queue = queue.Queue(maxsize=1000)
# save_queue = queue.Queue(maxsize=1000)

# # State Management
# processing_ids = set()
# saved_id = set()
# processed_results = {}
# visual_data = {}

# # ---------------- MODELS ----------------
# print("Loading Models... (This might take 30s)")

# # 1. Pose Model
# model = YOLO(r"C:\\Users\\acer\\Desktop\\project\\Backend\\All_model\\yolov8n-pose.pt")

# # 2. ID Card Model
# id_model_path = r"C:\\Users\\acer\\Desktop\\project\\Backend\\All_model\\best.pt"
# model2 = YOLO(id_model_path)

# # 3. Face Analysis
# providers = ["CUDAExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
# face_app = FaceAnalysis(name="buffalo_l", providers=providers)
# face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1)

# # Load Face Embeddings
# db_embeddings = []
# db_names = []

# def load_face_database(db_path=r"C:\\Users\\acer\\Desktop\\project\\Backend\\face_embeddings"):
#     global db_embeddings, db_names
#     if not os.path.exists(db_path):
#         print(f"Warning: Face DB path {db_path} not found.")
#         return
    
#     loaded_names = []
#     for person in os.listdir(db_path):
#         person_folder = os.path.join(db_path, person)
#         if not os.path.isdir(person_folder): continue

#         for file in os.listdir(person_folder):
#             if file.endswith(".npy"):
#                 emb = np.load(os.path.join(person_folder, file))
#                 db_embeddings.append(emb)
#                 db_names.append(person)
#                 loaded_names.append(person)
#     print(f"Loaded {len(loaded_names)} identities.")

# load_face_database()
# print("Models Loaded.")

# # ---------------- DIRECTORIES ----------------
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = os.path.join(ROOT_DIR, "output")

# CAPTURED_DIR = os.path.join(BASE_DIR, "captured_image")
# DETECTION_DIR = os.path.join(BASE_DIR, "detections", "id_card_detection")
# DRESSING_DIR = os.path.join(BASE_DIR, "dressing")
# TUCKED_DIR = os.path.join(DRESSING_DIR, "tucked")
# UNTUCKED_DIR = os.path.join(DRESSING_DIR, "untucked")
# RECOGNITION_DIR = os.path.join(BASE_DIR, "recognition")
# RECOGNIZED_DIR = os.path.join(RECOGNITION_DIR, "recognized_faces")
# UNRECOGNIZED_DIR = os.path.join(RECOGNITION_DIR, "unrecognized_faces")

# for d in [CAPTURED_DIR, DETECTION_DIR, RECOGNITION_DIR, RECOGNIZED_DIR, UNRECOGNIZED_DIR, TUCKED_DIR, UNTUCKED_DIR]:
#     os.makedirs(d, exist_ok=True)

# # ---------------- GLOBAL COUNTS (UPDATED STRUCTURE) ----------------
# ui_count = {
#     "is_online": 1,
#     "live_flow": 0,          
#     "total_scanned": 0,
#     "disciplined_total": 0,
#     "undisciplined_total": 0,
#     "id_card_yes_total": 0,
#     "id_card_no_total": 0,
#     "white_dress_total": 0,
#     "improper_dress_total": 0,
#     "tucked_total": 0,
#     "untucked_total": 0,
#     "recognized_total": 0,
#     "unrecognized": 0,
#     "staff_total": 0,
#     "discipline_rate_total": 0.0,

#     # --- AN (AI & ML) Department ---
#     "recognized_an": 0, "disciplined_an": 0, "undisciplined_an": 0, "discipline_rate_an": 0.0,
#     "id_card_yes_an": 0, "id_card_no_an": 0, "white_dress_an": 0, "improper_dress_an": 0,
#     "tucked_an": 0, "untucked_an": 0,

#     # --- AN FY ---
#     "recognized_an_fy": 0, "disciplined_an_fy": 0, "undisciplined_an_fy": 0, "discipline_rate_an_fy": 0.0,
#     "id_card_yes_an_fy": 0, "id_card_no_an_fy": 0, "white_dress_an_fy": 0, "improper_dress_an_fy": 0,
#     "tucked_an_fy": 0, "untucked_an_fy": 0,

#     # --- AN SY ---
#     "recognized_an_sy": 0, "disciplined_an_sy": 0, "undisciplined_an_sy": 0, "discipline_rate_an_sy": 0.0,
#     "id_card_yes_an_sy": 0, "id_card_no_an_sy": 0, "white_dress_an_sy": 0, "improper_dress_an_sy": 0,
#     "tucked_an_sy": 0, "untucked_an_sy": 0,

#     # --- AN TY ---
#     "recognized_an_ty": 0, "disciplined_an_ty": 0, "undisciplined_an_ty": 0, "discipline_rate_an_ty": 0.0,
#     "id_card_yes_an_ty": 0, "id_card_no_an_ty": 0, "white_dress_an_ty": 0, "improper_dress_an_ty": 0,
#     "tucked_an_ty": 0, "untucked_an_ty": 0,

#     # --- CO (Computer) Department ---
#     "recognized_co": 0, "disciplined_co": 0, "undisciplined_co": 0, "discipline_rate_co": 0.0,
#     "id_card_yes_co": 0, "id_card_no_co": 0, "white_dress_co": 0, "improper_dress_co": 0,
#     "tucked_co": 0, "untucked_co": 0,
     
#     # Internal usage for Rate Calculation (Hidden from main UI logic usually, but kept for math)
#     "_scanned_an": 0, "_scanned_an_fy": 0, "_scanned_an_sy": 0, "_scanned_an_ty": 0, "_scanned_co": 0
# }

# # ---------------- CAMERA THREAD ----------------
# class CameraStream:
#     def __init__(self, src=0):
#         self.stream = cv2.VideoCapture(src)
#         if not self.stream.isOpened():
#             print("Warning: IP Camera not found, switching to webcam...")
#             self.stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#         self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce lag
#         self.ret, self.frame = self.stream.read()
#         self.stopped = False
#         self.lock = threading.Lock()

#     def start(self):
#         t = threading.Thread(target=self.update, args=(), daemon=True)
#         t.start()
#         return self

#     def update(self):
#         while True:
#             if self.stopped: return
#             ret, frame = self.stream.read()
#             with self.lock:
#                 self.ret = ret
#                 self.frame = frame
#             time.sleep(0.001)

#     def read(self):
#         with self.lock:
#             return self.ret, self.frame.copy() if self.frame is not None else None

#     def release(self):
#         self.stopped = True
#         self.stream.release()

# # Start Camera
# url = "http://192.168.238.175:8080/video"
# camera_stream = CameraStream(url).start()

# @atexit.register
# def cleanup():
#     camera_stream.release()

# # ---------------- HELPER FUNCTIONS ----------------
# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# def distance(p1, p2):
#     return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

# def blur_score(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray, cv2.CV_64F).var()

# def is_full_body(x1, y2, img_w, img_h, margin=5):
#     valid_left = x1 >= -50
#     valid_bottom = y2 <= img_h + 50
#     return valid_left and valid_bottom

# def keypoint_visible(keypoints, conf, idx, min_conf=0.60):
#     return conf[idx] > min_conf and keypoints[idx][0] > 0 and keypoints[idx][1] > 0

# def ankle_knee_visible(keypoints, conf, idx, min_conf=0.65):
#     return conf[idx] > min_conf and keypoints[idx][0] > 0 and keypoints[idx][1] > 0

# # ---------------- BACKGROUND WORKERS (DB & FILE) ----------------

# def db_logger_worker():
#     db_path = "college.db"
#     conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    
#     try:
#         conn.execute("PRAGMA journal_mode=WAL;")
#         conn.commit()
#     except Exception as e:
#         print(f"Warning: Could not set WAL mode: {e}")

#     cursor = conn.cursor()
#     print("--- DB Worker Running ---")
    
#     while True:
#         data = log_queue.get()
#         if data is None: break

#         max_wait_time = 30
#         start_time = time.time()
#         saved = False

#         while (time.time() - start_time) < max_wait_time:
#             try:
#                 cursor.execute("""
#                     INSERT INTO entries (
#                         date, time, enrollment, name, department, year, mobile,
#                         recognized, discipline, id_card, shirt_tucked, image_path
#                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                 """, (
#                     data["date"], data["time"], data["enrollment"], data["name"],
#                     data["department"], data["year"], data["mobile"],
#                     int(data["recognized"]), int(data["discipline"]),
#                     int(data["id_card"]), int(data["shirt_tucked"]),
#                     data["image_path"]
#                 ))
#                 conn.commit()
#                 saved = True
#                 break
            
#             except sqlite3.OperationalError as e:
#                 if "locked" in str(e).lower():
#                     time.sleep(0.5)
#                 else:
#                     print(f"❌ DB Error: {e}")
#                     break
#             except Exception as e:
#                 print(f"❌ DB Critical Error: {e}")
#                 break
        
#         if not saved:
#             print(f"❌ Failed to save record for {data['name']}")
        
#         log_queue.task_done()

# def file_writer_worker():
#     while True:
#         try:
#             path, img = save_queue.get()
#             cv2.imwrite(path, img)
#         except Exception as e:
#             print(f"File Save Error: {e}")
#         finally:
#             save_queue.task_done()

# def get_student_by_name(recognized_name):
#     if not recognized_name or recognized_name.lower() in ["unknown", "no face", "no database"]:
#         return None
#     try:
#         with sqlite3.connect("college.db", timeout=30.0) as conn:
#             c = conn.cursor()
#             c.execute("SELECT enrollment, name, department, year, mobile FROM students WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))", (recognized_name,))
#             row = c.fetchone()
#             if row:
#                 return {"enrollment": row[0], "name": row[1], "department": row[2], "year": row[3], "mobile": row[4]}
#     except Exception as e:
#         print(f"DB Lookup Error: {e}")
#     return None

# # ---------------- CORE PROCESSING ----------------

# def recognize_face_enhanced(full_frame, keypoints, db_embeddings, db_names, threshold=0.5):
#     # (Same as before)
#     nose_x, nose_y = keypoints[0]
#     if nose_x <= 0 or nose_y <= 0: return "No Face", False, None
#     h, w = full_frame.shape[:2]
#     l_sh, r_sh = keypoints[5], keypoints[6]
#     sh_width = np.linalg.norm(np.array(l_sh) - np.array(r_sh))
#     if sh_width < 20: crop_size = 150
#     else: crop_size = int(sh_width * 2.0)
#     x1 = max(0, int(nose_x - crop_size))
#     x2 = min(w, int(nose_x + crop_size))
#     y1 = max(0, int(nose_y - crop_size))
#     y2 = min(h, int(nose_y + crop_size * 1.5))
#     face_crop = full_frame[y1:y2, x1:x2]
#     if face_crop.size == 0: return "No Face", False, None

#     with inference_lock:
#         faces = face_app.get(face_crop)
#     if len(faces) == 0: return "No Face", False, face_crop

#     best_face = None
#     min_dist = float('inf')
#     rel_nose_x = nose_x - x1
#     rel_nose_y = nose_y - y1

#     for face in faces:
#         face_cx = (face.bbox[0] + face.bbox[2]) / 2
#         face_cy = (face.bbox[1] + face.bbox[3]) / 2
#         dist = ((face_cx - rel_nose_x)**2 + (face_cy - rel_nose_y)**2)**0.5
#         if dist < min_dist:
#             min_dist = dist
#             best_face = face

#     if min_dist > (crop_size * 0.45): return "Unknown", False, face_crop
#     if len(db_embeddings) == 0: return "No Database", False, face_crop

#     sims = [cosine_similarity(best_face.embedding, db) for db in db_embeddings]
#     best_i = np.argmax(sims)

#     if sims[best_i] >= threshold:
#         return db_names[best_i], True, face_crop
#     return "Unknown", False, face_crop

# def async_save_image(img, folder, filename):
#     abs_folder = os.path.join(ROOT_DIR, folder)
#     os.makedirs(abs_folder, exist_ok=True)
#     abs_path = os.path.join(abs_folder, filename)
#     rel_path = os.path.join(folder, filename)
#     try:
#         save_queue.put_nowait((abs_path, img))
#     except queue.Full:
#         pass
#     return rel_path

# def processing_worker():
#     while True:
#         try:
#             data = processing_queue.get()
#             track_id = data['track_id']
#             full_frame = data['full_frame']
#             best_person_crop = data['best_person_crop']
#             keypoints = data['keypoints']
#             xyxy = data['xyxy']
            
#             timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#             async_save_image(best_person_crop, CAPTURED_DIR, f"id_{track_id}_{timestamp}.png")

#             # --- SHIRT LOGIC ---
#             shirt_status = "UNKNOWN"
#             color_status = "UNKNOWN"
#             try:
#                 if 'shirt_tuck' in globals():
#                     shirt_status, color_status = shirt_tuck.shirt_tucked2(best_person_crop)
#                     save_dir = TUCKED_DIR if shirt_status == "TUCKED" else UNTUCKED_DIR
#                     async_save_image(best_person_crop, save_dir, f"{track_id}_{timestamp}.jpg")
#             except Exception: pass

#             # --- ID CARD LOGIC ---
#             id_detected = False
#             try:
#                 upper_y = max(0, int(keypoints[5][1] - 20))
#                 lower_y = int(keypoints[11][1])
#                 crop_y1 = int(xyxy[1])
#                 local_y1 = max(0, upper_y - crop_y1)
#                 local_y2 = min(best_person_crop.shape[0], lower_y - crop_y1)
#                 torso_crop = best_person_crop[local_y1:local_y2, :] if local_y2 > local_y1 else best_person_crop[0:int(best_person_crop.shape[0]*0.6), :]

#                 if torso_crop.size > 0:
#                     with inference_lock:
#                         results_id = model2(best_person_crop, verbose=False, conf=0.1)
                    
#                 for r in results_id:
#                     for b in r.boxes:
#                         if int(b.cls[0]) == 0:
#                             id_detected = True
#                             async_save_image(r.plot(), DETECTION_DIR, f"id_{track_id}_{timestamp}.jpg")
#                             break
#             except Exception: pass

#             # --- FACE RECOGNITION ---
#             person_name, is_recognized, face_crop = recognize_face_enhanced(full_frame, keypoints, db_embeddings, db_names)
            
#             display_name = person_name if person_name else "Unknown"
#             student_info = None
#             if is_recognized:
#                 student_info = get_student_by_name(person_name)
#                 if student_info:
#                     display_name = f"{student_info['name']} ({student_info['enrollment']})"
            
#             # Evidence Saving
#             if is_recognized:
#                 folder = os.path.join(RECOGNIZED_DIR, person_name)
#                 os.makedirs(folder, exist_ok=True)
#                 fname = f"{person_name}_{timestamp}.jpg"
#                 color = (0, 255, 0)
#             else:
#                 folder = UNRECOGNIZED_DIR
#                 fname = f"unknown_{timestamp}.jpg"
#                 display_name = "Unknown"
#                 color = (0, 0, 255)
            
#             labeled_img = best_person_crop.copy()
#             cv2.putText(labeled_img, display_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
#             image_path = async_save_image(labeled_img, folder, fname)

#             # --- UPDATE COUNTS (NEW LOGIC) ---
#             is_white = (color_status == "WHITE OK")
#             is_tucked = (shirt_status == "TUCKED")
#             is_disciplined = is_white and is_tucked and id_detected

#             with count_lock:
#                 # 1. Global Totals
#                 ui_count["total_scanned"] += 1
                
#                 if is_disciplined: ui_count["disciplined_total"] += 1
#                 else: ui_count["undisciplined_total"] += 1
                
#                 if id_detected: ui_count["id_card_yes_total"] += 1
#                 else: ui_count["id_card_no_total"] += 1
                
#                 if is_white: ui_count["white_dress_total"] += 1
#                 else: ui_count["improper_dress_total"] += 1
                
#                 if is_tucked: ui_count["tucked_total"] += 1
#                 else: ui_count["untucked_total"] += 1
                
#                 # 2. Recognition & Department Logic
#                 dept_str = ""
#                 year_str = ""
                
#                 if is_recognized and student_info:
#                     ui_count["recognized_total"] += 1
                    
#                     # Normalize Strings
#                     dept_str = str(student_info.get("department", "")).upper()
#                     year_str = str(student_info.get("year", "")).upper()
                    
#                     # A. Check AN Department
#                     if "AN" in dept_str or "AI" in dept_str or "DS" in dept_str:
#                         ui_count["_scanned_an"] += 1
#                         ui_count["recognized_an"] += 1
#                         if is_disciplined: ui_count["disciplined_an"] += 1
#                         else: ui_count["undisciplined_an"] += 1
                        
#                         if id_detected: ui_count["id_card_yes_an"] += 1
#                         else: ui_count["id_card_no_an"] += 1
                        
#                         if is_white: ui_count["white_dress_an"] += 1
#                         else: ui_count["improper_dress_an"] += 1
                        
#                         if is_tucked: ui_count["tucked_an"] += 1
#                         else: ui_count["untucked_an"] += 1

#                         # --- Year Specific Logic for AN ---
                        
#                         # FY
#                         if "FY" in year_str or "1" in year_str:
#                             ui_count["_scanned_an_fy"] += 1
#                             ui_count["recognized_an_fy"] += 1
#                             if is_disciplined: ui_count["disciplined_an_fy"] += 1
#                             else: ui_count["undisciplined_an_fy"] += 1
                            
#                             if id_detected: ui_count["id_card_yes_an_fy"] += 1
#                             else: ui_count["id_card_no_an_fy"] += 1
                            
#                             if is_white: ui_count["white_dress_an_fy"] += 1
#                             else: ui_count["improper_dress_an_fy"] += 1
                            
#                             if is_tucked: ui_count["tucked_an_fy"] += 1
#                             else: ui_count["untucked_an_fy"] += 1
                            
#                         # SY
#                         elif "SY" in year_str or "2" in year_str:
#                             ui_count["_scanned_an_sy"] += 1
#                             ui_count["recognized_an_sy"] += 1
#                             if is_disciplined: ui_count["disciplined_an_sy"] += 1
#                             else: ui_count["undisciplined_an_sy"] += 1
                            
#                             if id_detected: ui_count["id_card_yes_an_sy"] += 1
#                             else: ui_count["id_card_no_an_sy"] += 1
                            
#                             if is_white: ui_count["white_dress_an_sy"] += 1
#                             else: ui_count["improper_dress_an_sy"] += 1
                            
#                             if is_tucked: ui_count["tucked_an_sy"] += 1
#                             else: ui_count["untucked_an_sy"] += 1
                            
#                         # TY
#                         elif "TY" in year_str or "3" in year_str:
#                             ui_count["_scanned_an_ty"] += 1
#                             ui_count["recognized_an_ty"] += 1
#                             if is_disciplined: ui_count["disciplined_an_ty"] += 1
#                             else: ui_count["undisciplined_an_ty"] += 1
                            
#                             if id_detected: ui_count["id_card_yes_an_ty"] += 1
#                             else: ui_count["id_card_no_an_ty"] += 1
                            
#                             if is_white: ui_count["white_dress_an_ty"] += 1
#                             else: ui_count["improper_dress_an_ty"] += 1
                            
#                             if is_tucked: ui_count["tucked_an_ty"] += 1
#                             else: ui_count["untucked_an_ty"] += 1

#                     # B. Check CO Department
#                     elif "CO" in dept_str or "COMP" in dept_str or "COMPUTER" in dept_str:
#                         ui_count["_scanned_co"] += 1
#                         ui_count["recognized_co"] += 1
#                         if is_disciplined: ui_count["disciplined_co"] += 1
#                         else: ui_count["undisciplined_co"] += 1
                        
#                         if id_detected: ui_count["id_card_yes_co"] += 1
#                         else: ui_count["id_card_no_co"] += 1
                        
#                         if is_white: ui_count["white_dress_co"] += 1
#                         else: ui_count["improper_dress_co"] += 1
                        
#                         if is_tucked: ui_count["tucked_co"] += 1
#                         else: ui_count["untucked_co"] += 1

#                 else:
#                     ui_count["unrecognized"] += 1

#                 # 3. Recalculate Rates (Percentage)
#                 def calc_rate(disc, scanned):
#                     return round((disc / scanned) * 100, 1) if scanned > 0 else 0.0

#                 ui_count["discipline_rate_total"] = calc_rate(ui_count["disciplined_total"], ui_count["total_scanned"])
                
#                 # Dept Rates
#                 ui_count["discipline_rate_an"] = calc_rate(ui_count["disciplined_an"], ui_count["_scanned_an"])
#                 ui_count["discipline_rate_co"] = calc_rate(ui_count["disciplined_co"], ui_count["_scanned_co"])
                
#                 # Year Rates
#                 ui_count["discipline_rate_an_fy"] = calc_rate(ui_count["disciplined_an_fy"], ui_count["_scanned_an_fy"])
#                 ui_count["discipline_rate_an_sy"] = calc_rate(ui_count["disciplined_an_sy"], ui_count["_scanned_an_sy"])
#                 ui_count["discipline_rate_an_ty"] = calc_rate(ui_count["disciplined_an_ty"], ui_count["_scanned_an_ty"])

#             # --- DB ENTRY ---
#             now = datetime.now()
#             entry_data = {
#                 "date": now.strftime("%Y-%m-%d"),
#                 "time": now.strftime("%H:%M:%S"),
#                 "enrollment": student_info["enrollment"] if student_info else "NA",
#                 "name": student_info["name"] if student_info else person_name,
#                 "department": student_info["department"] if student_info else "Unknown",
#                 "year": student_info["year"] if student_info else "Unknown",
#                 "mobile": student_info["mobile"] if student_info else "NA",
#                 "recognized": is_recognized,
#                 "discipline": is_disciplined,
#                 "id_card": id_detected,
#                 "shirt_tucked": is_tucked,
#                 "image_path": image_path
#             }
#             log_queue.put(entry_data)

#             # --- SHARED RESULT UPDATES ---
#             final_color = (0, 255, 0) if is_disciplined else (0, 0, 255)
#             shirt_str = f"{shirt_status}/{'White' if is_white else 'Color'}"

#             with results_lock:
#                 processed_results[track_id] = {
#                     "name": display_name,
#                     "shirt": shirt_str,
#                     "id": "Yes" if id_detected else "No",
#                     "color": final_color,
#                     "timestamp": time.time()
#                 }
            
#             processing_ids.discard(track_id)
#             processing_queue.task_done()

#         except Exception as e:
#             print(f"Worker Error: {e}")
#             if 'track_id' in locals(): processing_ids.discard(track_id)

# # START THREADS
# for _ in range(3):
#     t = threading.Thread(target=processing_worker, daemon=True)
#     t.start()
# t = threading.Thread(target=db_logger_worker, daemon=True)
# t.start()
# t = threading.Thread(target=file_writer_worker, daemon=True)
# t.start()

# # ---------------- CAMERA LOGIC LOOP ----------------
# track_enter_time = {}
# frame_buffer = {}

# def camera_processing_loop():
#     print("--- Camera Logic Started ---")
#     while True:
#         ret, frame = camera_stream.read()
#         if not ret or frame is None:
#             time.sleep(0.01)
#             continue

#         frame_h, frame_w = frame.shape[:2]

#         # TRACKING UPDATE: Used bytetrack for better stability
#         results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, iou=0.5, verbose=False, classes=[0])

#         current_boxes = []
        
#         if results[0].boxes is not None and results[0].boxes.id is not None:
#             boxes = results[0].boxes
#             keypoints_data = results[0].keypoints.xy
#             conf_data = results[0].keypoints.conf

#             current_frame_ids = set()

#             for i, box in enumerate(boxes):
#                 track_id = int(box.id[0])
#                 current_frame_ids.add(track_id)
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
                
#                 current_boxes.append({
#                     "id": track_id,
#                     "coords": (x1, y1, x2, y2)
#                 })

#                 if track_id in saved_id or track_id in processing_ids:
#                     continue

#                 # Stabilization Logic
#                 now = time.time()
#                 if track_id not in track_enter_time:
#                     track_enter_time[track_id] = now
                
#                 if (now - track_enter_time[track_id]) < 1.0: continue

#                 if not is_full_body(x1, y2, frame_w, frame_h): continue

#                 kpts = keypoints_data[i]
#                 confs = conf_data[i]

#                 if not (keypoint_visible(kpts, confs, 0) and keypoint_visible(kpts, confs, 1)): continue
#                 if not ((ankle_knee_visible(kpts, confs, 13) or ankle_knee_visible(kpts, confs, 14))): continue

#                 person_crop = frame[y1:y2, x1:x2]
#                 if person_crop.size == 0: continue

#                 score = blur_score(person_crop)
#                 if score < 40: continue

#                 frame_buffer.setdefault(track_id, [])
#                 frame_buffer[track_id].append({
#                     "score": score,
#                     "crop": person_crop,
#                     "frame": frame.copy(),
#                     "box": [x1, y1, x2, y2],
#                     "kpts": kpts.cpu().numpy()
#                 })

#                 frame_buffer[track_id] = sorted(frame_buffer[track_id], key=lambda x: x["score"], reverse=True)[:3]

#                 if len(frame_buffer[track_id]) >= 3:
#                     best = frame_buffer[track_id][0]
#                     packet = {
#                         "track_id": track_id,
#                         "full_frame": best["frame"],
#                         "best_person_crop": best["crop"],
#                         "keypoints": best["kpts"],
#                         "xyxy": best["box"]
#                     }
#                     try:
#                         processing_queue.put_nowait(
#                             packet)
#                         processing_ids.add(track_id)
#                         saved_id.add(track_id)
#                         del frame_buffer[track_id]
#                     except queue.Full:
#                         pass
        
#             with count_lock:
#                 ui_count["live_flow"] = len(current_frame_ids) # UPDATED KEY

#         with visual_lock:
#             visual_data.clear()
#             for b in current_boxes:
#                 visual_data[b["id"]] = b["coords"]

#         time.sleep(0.01)

# threading.Thread(target=camera_processing_loop, daemon=True).start()

# # ---------------- API ROUTES ----------------

# @app.route('/count')
# def counts():
#     with count_lock:
#         return jsonify(ui_count)

# @app.route("/entries")
# def get_entries():
#     try:
#         with sqlite3.connect("college.db", timeout=30.0) as conn:
#             conn.row_factory = sqlite3.Row
#             cursor = conn.cursor()
#             cursor.execute("SELECT * FROM entries ORDER BY date DESC, time DESC LIMIT 200")
#             rows = cursor.fetchall()
#             return jsonify([dict(row) for row in rows])
#     except Exception as e:
#         print(f"API Error: {e}")
#         return jsonify([])

# @app.route('/video_feed')
# def video_feed():
#     def generate():
#         while True:
#             ret, frame = camera_stream.read()
#             if not ret:
#                 time.sleep(0.02)
#                 continue
            
#             # --- CLEANUP LOGIC ---
#             # Remove labels for people who have left the frame (expire after 2.0s)
#             current_time = time.time()
#             with results_lock:
#                 expired_ids = [tid for tid, data in processed_results.items() if (current_time - data['timestamp']) > 2.0]
#                 for tid in expired_ids:
#                     del processed_results[tid]
#                     # Optional: Allow re-processing if they return?
#                     # if tid in saved_id: saved_id.remove(tid)

#             # --- DRAWING ---
#             boxes_to_draw = {}
#             with visual_lock:
#                 boxes_to_draw = visual_data.copy()
            
#             labels_to_draw = {}
#             with results_lock:
#                 labels_to_draw = processed_results.copy()
            
#             for tid, coords in boxes_to_draw.items():
#                 x1, y1, x2, y2 = coords
                
#                 if tid in labels_to_draw:
#                     data = labels_to_draw[tid]
#                     label = data['name']
#                     sub_label = f"{data['shirt']} | ID:{data['id']}"
#                     color = data['color']
                    
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#                     cv2.putText(frame, sub_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
#                 elif tid in processing_ids:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
#                     cv2.putText(frame, "Processing...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
#                 else:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

#             _, buffer = cv2.imencode('.jpg', frame)
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route("/evidence")
# def serve_evidence():
#     rel_path = request.args.get("path", "")
    
#     # 1. Construct the full absolute path
#     # ROOT_DIR is already defined globally in your code
#     abs_path = os.path.join(ROOT_DIR, rel_path)
#     abs_path = os.path.normpath(abs_path)

#     # 2. Check if the resolved absolute path starts with BASE_DIR
#     if not abs_path.startswith(BASE_DIR):
#         print(f"SECURITY BLOCK: {abs_path} is not inside {BASE_DIR}")
#         return "Access denied", 403

#     if not os.path.isfile(abs_path):
#         return "File not found", 404

#     return send_file(abs_path, mimetype="image/jpeg")

# @app.route("/login", methods=["POST"])
# def login():
#     data = request.json
#     if not data: return jsonify({"error": "No data"}), 400
#     username = data.get("username")
#     password = data.get("password")
#     try:
#         with sqlite3.connect("users.db", timeout=10.0) as conn:
#             cur = conn.cursor()
#             cur.execute("SELECT password_hash, role FROM users WHERE username=?", (username,))
#             row = cur.fetchone()
            
#         if not row: return jsonify({"error": "Invalid username"}), 401
#         pw_hash, role = row
#         if isinstance(pw_hash, str): pw_hash = pw_hash.encode('utf-8')
#         if bcrypt.checkpw(password.encode('utf-8'), pw_hash):
#             token = jwt.encode({
#                 "username": username, "role": role,
#                 "exp": datetime.utcnow() + timedelta(hours=12)
#             }, app.config["SECRET_KEY"], algorithm="HS256")
#             return jsonify({"token": token, "role": role})
#         return jsonify({"error": "Invalid password"}), 401
#     except Exception as e:
#         return jsonify({"error": "Server error"}), 500

# if __name__ == '__main__':
#     # Threaded is required for Flask to handle Video + API concurrently
#     app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)


from flask import Flask, Response, jsonify, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import os
import atexit
from flask_cors import CORS
from insightface.app import FaceAnalysis
import threading
import queue
import time
import sqlite3
import torch
import bcrypt
import jwt

# --- IMPORT LOGIC MODULE ---
try:
    import shirt_tuck  # Logic from Code 2
except ImportError:
    print("WARNING: shirt_tuck.py not found. Please ensure it exists.")

app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = "mysecret_production_key"

# ---------------- CONFIGURATION & HARDWARE ----------------
print("--- SYSTEM STARTUP ---")
print("YOLO GPU available:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# Prevent OpenCV from spawning too many threads
cv2.setNumThreads(1)

# Locks for Thread Safety
count_lock = threading.Lock()
results_lock = threading.Lock()
visual_lock = threading.Lock()  
inference_lock = threading.Lock() 
unknown_lock = threading.Lock() # NEW: Protects the Unknown buffers

# Queues
processing_queue = queue.Queue(maxsize=10)
log_queue = queue.Queue(maxsize=1000)
save_queue = queue.Queue(maxsize=1000)

# State Management
processing_ids = set()
saved_id = set()
processed_results = {}
visual_data = {}
retry_counts = {}  

# NEW: State Management for tracking Unknowns until they leave
pending_unknowns = set()
unknown_last_seen_data = {}

# ---------------- MODELS ----------------
print("Loading Models... (This might take 30s)")

model = YOLO(r"C:\\Users\\acer\\Desktop\\project\\Backend\\All_model\\yolov8n-pose.pt")
id_model_path = r"C:\\Users\\acer\\Desktop\\project\\Backend\\All_model\\best.pt"
model2 = YOLO(id_model_path)

providers = ["CUDAExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
face_app = FaceAnalysis(name="buffalo_l", providers=providers)
face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1)

db_embeddings = []
db_names = []

def load_face_database(db_path=r"C:\\Users\\acer\\Desktop\\project\\Backend\\face_embeddings"):
    global db_embeddings, db_names
    if not os.path.exists(db_path):
        print(f"Warning: Face DB path {db_path} not found.")
        return
    
    loaded_names = []
    for person in os.listdir(db_path):
        person_folder = os.path.join(db_path, person)
        if not os.path.isdir(person_folder): continue

        for file in os.listdir(person_folder):
            if file.endswith(".npy"):
                emb = np.load(os.path.join(person_folder, file))
                db_embeddings.append(emb)
                db_names.append(person)
                loaded_names.append(person)
    print(f"Loaded {len(loaded_names)} identities.")

load_face_database()
print("Models Loaded.")

# ---------------- DIRECTORIES ----------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(ROOT_DIR, "output")

CAPTURED_DIR = os.path.join(BASE_DIR, "captured_image")
DETECTION_DIR = os.path.join(BASE_DIR, "detections", "id_card_detection")
DRESSING_DIR = os.path.join(BASE_DIR, "dressing")
TUCKED_DIR = os.path.join(DRESSING_DIR, "tucked")
UNTUCKED_DIR = os.path.join(DRESSING_DIR, "untucked")
RECOGNITION_DIR = os.path.join(BASE_DIR, "recognition")
RECOGNIZED_DIR = os.path.join(RECOGNITION_DIR, "recognized_faces")
UNRECOGNIZED_DIR = os.path.join(RECOGNITION_DIR, "unrecognized_faces")

for d in [CAPTURED_DIR, DETECTION_DIR, RECOGNITION_DIR, RECOGNIZED_DIR, UNRECOGNIZED_DIR, TUCKED_DIR, UNTUCKED_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------- GLOBAL COUNTS ----------------
ui_count = {
    "is_online": 1, "live_flow": 0, "total_scanned": 0,
    "disciplined_total": 0, "undisciplined_total": 0,
    "id_card_yes_total": 0, "id_card_no_total": 0,
    "white_dress_total": 0, "improper_dress_total": 0,
    "tucked_total": 0, "untucked_total": 0,
    "recognized_total": 0, "unrecognized": 0, "staff_total": 0,
    "discipline_rate_total": 0.0,
    
    "recognized_an": 0, "disciplined_an": 0, "undisciplined_an": 0, "discipline_rate_an": 0.0,
    "id_card_yes_an": 0, "id_card_no_an": 0, "white_dress_an": 0, "improper_dress_an": 0,
    "tucked_an": 0, "untucked_an": 0,
    "recognized_an_fy": 0, "disciplined_an_fy": 0, "undisciplined_an_fy": 0, "discipline_rate_an_fy": 0.0,
    "id_card_yes_an_fy": 0, "id_card_no_an_fy": 0, "white_dress_an_fy": 0, "improper_dress_an_fy": 0,
    "tucked_an_fy": 0, "untucked_an_fy": 0,
    "recognized_an_sy": 0, "disciplined_an_sy": 0, "undisciplined_an_sy": 0, "discipline_rate_an_sy": 0.0,
    "id_card_yes_an_sy": 0, "id_card_no_an_sy": 0, "white_dress_an_sy": 0, "improper_dress_an_sy": 0,
    "tucked_an_sy": 0, "untucked_an_sy": 0,
    "recognized_an_ty": 0, "disciplined_an_ty": 0, "undisciplined_an_ty": 0, "discipline_rate_an_ty": 0.0,
    "id_card_yes_an_ty": 0, "id_card_no_an_ty": 0, "white_dress_an_ty": 0, "improper_dress_an_ty": 0,
    "tucked_an_ty": 0, "untucked_an_ty": 0,
    "recognized_co": 0, "disciplined_co": 0, "undisciplined_co": 0, "discipline_rate_co": 0.0,
    "id_card_yes_co": 0, "id_card_no_co": 0, "white_dress_co": 0, "improper_dress_co": 0,
    "tucked_co": 0, "untucked_co": 0,
    
    "_scanned_an": 0, "_scanned_an_fy": 0, "_scanned_an_sy": 0, "_scanned_an_ty": 0, "_scanned_co": 0
}

# ---------------- CAMERA THREAD ----------------
class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print("Warning: IP Camera not found, switching to webcam...")
            self.stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            ret, frame = self.stream.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.stream.release()

url = "http://192.0.0.4:8080/video"
camera_stream = CameraStream(url).start()

@atexit.register
def cleanup():
    camera_stream.release()

# ---------------- HELPER FUNCTIONS ----------------
def cosine_similarity(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def distance(p1, p2): return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5
def blur_score(image): gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY); return cv2.Laplacian(gray, cv2.CV_64F).var()
def is_full_body(x1, y2, img_w, img_h, margin=5): return (x1 >= -50) and (y2 <= img_h + 50)
def keypoint_visible(keypoints, conf, idx, min_conf=0.60): return conf[idx] > min_conf and keypoints[idx][0] > 0 and keypoints[idx][1] > 0
def ankle_knee_visible(keypoints, conf, idx, min_conf=0.65): return conf[idx] > min_conf and keypoints[idx][0] > 0 and keypoints[idx][1] > 0

# ---------------- BACKGROUND WORKERS (DB & FILE) ----------------
def db_logger_worker():
    db_path = "college.db"
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    try: conn.execute("PRAGMA journal_mode=WAL;"); conn.commit()
    except Exception as e: print(f"Warning: Could not set WAL mode: {e}")
    cursor = conn.cursor()
    print("--- DB Worker Running ---")
    
    while True:
        data = log_queue.get()
        if data is None: break
        start_time = time.time()
        saved = False

        while (time.time() - start_time) < 30:
            try:
                cursor.execute("""
                    INSERT INTO entries (date, time, enrollment, name, department, year, mobile, recognized, discipline, id_card, shirt_tucked, image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (data["date"], data["time"], data["enrollment"], data["name"], data["department"], data["year"], data["mobile"],
                      int(data["recognized"]), int(data["discipline"]), int(data["id_card"]), int(data["shirt_tucked"]), data["image_path"]))
                conn.commit()
                saved = True
                break
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower(): time.sleep(0.5)
                else: break
            except Exception as e: break
        if not saved: print(f"❌ Failed to save record for {data['name']}")
        log_queue.task_done()

def file_writer_worker():
    while True:
        try:
            path, img = save_queue.get()
            cv2.imwrite(path, img)
        except Exception as e: print(f"File Save Error: {e}")
        finally: save_queue.task_done()

def get_student_by_name(recognized_name):
    if not recognized_name or recognized_name.lower() in ["unknown", "no face", "no database"]: return None
    try:
        with sqlite3.connect("college.db", timeout=30.0) as conn:
            c = conn.cursor()
            c.execute("SELECT enrollment, name, department, year, mobile FROM students WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))", (recognized_name,))
            row = c.fetchone()
            if row: return {"enrollment": row[0], "name": row[1], "department": row[2], "year": row[3], "mobile": row[4]}
    except Exception as e: print(f"DB Lookup Error: {e}")
    return None

# ---------------- CORE PROCESSING ----------------
def recognize_face_enhanced(full_frame, keypoints, db_embeddings, db_names, threshold=0.45):
    nose_x, nose_y = keypoints[0]
    if nose_x <= 0 or nose_y <= 0: return "Waiting", False, None
    
    h, w = full_frame.shape[:2]
    l_sh, r_sh = keypoints[5], keypoints[6]
    sh_width = np.linalg.norm(np.array(l_sh) - np.array(r_sh))
    crop_size = 150 if sh_width < 20 else int(sh_width * 2.0)
    
    x1, x2 = max(0, int(nose_x - crop_size)), min(w, int(nose_x + crop_size))
    y1, y2 = max(0, int(nose_y - crop_size)), min(h, int(nose_y + crop_size * 1.5))
    face_crop = full_frame[y1:y2, x1:x2]
    
    if face_crop.size == 0: return "Waiting", False, None

    with inference_lock: faces = face_app.get(face_crop)
    if len(faces) == 0: return "Waiting", False, face_crop

    best_face = None
    min_dist = float('inf')
    rel_nose_x, rel_nose_y = nose_x - x1, nose_y - y1

    for face in faces:
        face_cx = (face.bbox[0] + face.bbox[2]) / 2
        face_cy = (face.bbox[1] + face.bbox[3]) / 2
        dist = ((face_cx - rel_nose_x)**2 + (face_cy - rel_nose_y)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            best_face = face

    if best_face is not None and hasattr(best_face, 'det_score') and best_face.det_score < 0.45:
        return "Waiting", False, face_crop

    if min_dist > (crop_size * 0.45): return "Unknown", False, face_crop
    if len(db_embeddings) == 0: return "No Database", False, face_crop

    sims = [cosine_similarity(best_face.embedding, db) for db in db_embeddings]
    best_i = np.argmax(sims)
    if sims[best_i] >= threshold: return db_names[best_i], True, face_crop
    return "Unknown", False, face_crop

def async_save_image(img, folder, filename):
    abs_folder = os.path.join(ROOT_DIR, folder)
    os.makedirs(abs_folder, exist_ok=True)
    abs_path = os.path.join(abs_folder, filename)
    rel_path = os.path.join(folder, filename)
    try: save_queue.put_nowait((abs_path, img))
    except queue.Full: pass
    return rel_path

def processing_worker():
    while True:
        try:
            data = processing_queue.get()
            track_id = data['track_id']
            full_frame = data['full_frame']
            best_person_crop = data['best_person_crop']
            keypoints = data['keypoints']
            xyxy = data['xyxy']
            
            # Check if this frame is triggered by the user leaving the camera view
            is_final_leaving = data.get("is_final_leaving", False)
            
            if is_final_leaving:
                person_name = "Unknown"
                is_recognized = False
                face_crop = None
            else:
                person_name, is_recognized, face_crop = recognize_face_enhanced(full_frame, keypoints, db_embeddings, db_names)
                
                if person_name == "Waiting":
                    with results_lock:
                        processed_results[track_id] = {
                            "name": "Scanning Face...", "shirt": "Waiting...", "id": "Waiting...",
                            "color": (0, 255, 255), "timestamp": time.time()
                        }
                    processing_ids.discard(track_id)
                    if track_id in saved_id: saved_id.remove(track_id)
                    processing_queue.task_done()
                    continue
                
                if person_name == "Unknown":
                    retries = retry_counts.get(track_id, 0)
                    if retries < 3: 
                        retry_counts[track_id] = retries + 1
                        with results_lock:
                            processed_results[track_id] = {
                                "name": f"Retrying ({retries+1}/3)...", "shirt": "Checking...",
                                "id": "Checking...", "color": (0, 165, 255), "timestamp": time.time()
                            }
                        processing_ids.discard(track_id)
                        if track_id in saved_id: saved_id.remove(track_id)
                        processing_queue.task_done()
                        continue
                    
                    # --- ADD TO PENDING UNKNOWNS & DELAY SAVING ---
                    with unknown_lock:
                        pending_unknowns.add(track_id)
                        unknown_last_seen_data[track_id] = {
                            "last_seen": time.time(), "crop": best_person_crop.copy(),
                            "full_frame": full_frame.copy(), "kpts": keypoints, "box": xyxy
                        }
                    
                    with results_lock:
                        processed_results[track_id] = {
                            "name": "Unknown", "shirt": "Scanning...",
                            "id": "No Face", "color": (0, 0, 255), "timestamp": time.time()
                        }
                    processing_ids.discard(track_id)
                    processing_queue.task_done()
                    continue # Wait for them to leave the frame before saving

            # --- PROCEED WITH SAVING FOR RECOGNIZED OR 'LEAVING' UNKNOWNS ---
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            async_save_image(best_person_crop, CAPTURED_DIR, f"id_{track_id}_{timestamp}.png")

            # --- SHIRT LOGIC ---
            shirt_status = "UNKNOWN"
            color_status = "UNKNOWN"
            try:
                if 'shirt_tuck' in globals():
                    shirt_status, color_status = shirt_tuck.shirt_tucked2(best_person_crop)
                    save_dir = TUCKED_DIR if shirt_status == "TUCKED" else UNTUCKED_DIR
                    async_save_image(best_person_crop, save_dir, f"{track_id}_{timestamp}.jpg")
            except Exception: pass

            # --- ID CARD LOGIC ---
            id_detected = False
            try:
                upper_y = max(0, int(keypoints[5][1] - 20))
                lower_y = int(keypoints[11][1])
                crop_y1 = int(xyxy[1])
                local_y1 = max(0, upper_y - crop_y1)
                local_y2 = min(best_person_crop.shape[0], lower_y - crop_y1)
                torso_crop = best_person_crop[local_y1:local_y2, :] if local_y2 > local_y1 else best_person_crop[0:int(best_person_crop.shape[0]*0.6), :]

                if torso_crop.size > 0:
                    with inference_lock:
                        results_id = model2(best_person_crop, verbose=False, conf=0.1)
                    for r in results_id:
                        for b in r.boxes:
                            if int(b.cls[0]) == 0:
                                id_detected = True
                                async_save_image(r.plot(), DETECTION_DIR, f"id_{track_id}_{timestamp}.jpg")
                                break
            except Exception: pass
            
            display_name = person_name if person_name else "Unknown"
            student_info = None
            if is_recognized:
                student_info = get_student_by_name(person_name)
                if student_info:
                    display_name = f"{student_info['name']} ({student_info['enrollment']})"
                folder = os.path.join(RECOGNIZED_DIR, person_name)
                os.makedirs(folder, exist_ok=True)
                fname = f"{person_name}_{timestamp}.jpg"
                color = (0, 255, 0)
            else:
                folder = UNRECOGNIZED_DIR
                fname = f"unknown_{timestamp}.jpg"
                display_name = "Unknown"
                color = (0, 0, 255)
            
            labeled_img = best_person_crop.copy()
            cv2.putText(labeled_img, display_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            image_path = async_save_image(labeled_img, folder, fname)

            # --- UPDATE COUNTS ---
            is_white = (color_status == "WHITE OK")
            is_tucked = (shirt_status == "TUCKED")
            is_disciplined = is_white and is_tucked and id_detected

            with count_lock:
                ui_count["total_scanned"] += 1
                if is_disciplined: ui_count["disciplined_total"] += 1
                else: ui_count["undisciplined_total"] += 1
                if id_detected: ui_count["id_card_yes_total"] += 1
                else: ui_count["id_card_no_total"] += 1
                if is_white: ui_count["white_dress_total"] += 1
                else: ui_count["improper_dress_total"] += 1
                if is_tucked: ui_count["tucked_total"] += 1
                else: ui_count["untucked_total"] += 1
                
                dept_str, year_str = "", ""
                if is_recognized and student_info:
                    ui_count["recognized_total"] += 1
                    dept_str = str(student_info.get("department", "")).upper()
                    year_str = str(student_info.get("year", "")).upper()
                    
                    if "AN" in dept_str or "AI" in dept_str or "DS" in dept_str:
                        ui_count["_scanned_an"] += 1
                        ui_count["recognized_an"] += 1
                        if is_disciplined: ui_count["disciplined_an"] += 1
                        else: ui_count["undisciplined_an"] += 1
                        
                        if id_detected: ui_count["id_card_yes_an"] += 1
                        else: ui_count["id_card_no_an"] += 1
                        if is_white: ui_count["white_dress_an"] += 1
                        else: ui_count["improper_dress_an"] += 1
                        if is_tucked: ui_count["tucked_an"] += 1
                        else: ui_count["untucked_an"] += 1

                        if "FY" in year_str or "1" in year_str:
                            ui_count["_scanned_an_fy"] += 1; ui_count["recognized_an_fy"] += 1
                            if is_disciplined: ui_count["disciplined_an_fy"] += 1
                            else: ui_count["undisciplined_an_fy"] += 1
                            if id_detected: ui_count["id_card_yes_an_fy"] += 1
                            else: ui_count["id_card_no_an_fy"] += 1
                            if is_white: ui_count["white_dress_an_fy"] += 1
                            else: ui_count["improper_dress_an_fy"] += 1
                            if is_tucked: ui_count["tucked_an_fy"] += 1
                            else: ui_count["untucked_an_fy"] += 1
                        elif "SY" in year_str or "2" in year_str:
                            ui_count["_scanned_an_sy"] += 1; ui_count["recognized_an_sy"] += 1
                            if is_disciplined: ui_count["disciplined_an_sy"] += 1
                            else: ui_count["undisciplined_an_sy"] += 1
                            if id_detected: ui_count["id_card_yes_an_sy"] += 1
                            else: ui_count["id_card_no_an_sy"] += 1
                            if is_white: ui_count["white_dress_an_sy"] += 1
                            else: ui_count["improper_dress_an_sy"] += 1
                            if is_tucked: ui_count["tucked_an_sy"] += 1
                            else: ui_count["untucked_an_sy"] += 1
                        elif "TY" in year_str or "3" in year_str:
                            ui_count["_scanned_an_ty"] += 1; ui_count["recognized_an_ty"] += 1
                            if is_disciplined: ui_count["disciplined_an_ty"] += 1
                            else: ui_count["undisciplined_an_ty"] += 1
                            if id_detected: ui_count["id_card_yes_an_ty"] += 1
                            else: ui_count["id_card_no_an_ty"] += 1
                            if is_white: ui_count["white_dress_an_ty"] += 1
                            else: ui_count["improper_dress_an_ty"] += 1
                            if is_tucked: ui_count["tucked_an_ty"] += 1
                            else: ui_count["untucked_an_ty"] += 1
                    elif "CO" in dept_str or "COMP" in dept_str or "COMPUTER" in dept_str:
                        ui_count["_scanned_co"] += 1
                        ui_count["recognized_co"] += 1
                        if is_disciplined: ui_count["disciplined_co"] += 1
                        else: ui_count["undisciplined_co"] += 1
                        if id_detected: ui_count["id_card_yes_co"] += 1
                        else: ui_count["id_card_no_co"] += 1
                        if is_white: ui_count["white_dress_co"] += 1
                        else: ui_count["improper_dress_co"] += 1
                        if is_tucked: ui_count["tucked_co"] += 1
                        else: ui_count["untucked_co"] += 1
                else:
                    ui_count["unrecognized"] += 1

                def calc_rate(disc, scanned): return round((disc / scanned) * 100, 1) if scanned > 0 else 0.0
                ui_count["discipline_rate_total"] = calc_rate(ui_count["disciplined_total"], ui_count["total_scanned"])
                ui_count["discipline_rate_an"] = calc_rate(ui_count["disciplined_an"], ui_count["_scanned_an"])
                ui_count["discipline_rate_co"] = calc_rate(ui_count["disciplined_co"], ui_count["_scanned_co"])
                ui_count["discipline_rate_an_fy"] = calc_rate(ui_count["disciplined_an_fy"], ui_count["_scanned_an_fy"])
                ui_count["discipline_rate_an_sy"] = calc_rate(ui_count["disciplined_an_sy"], ui_count["_scanned_an_sy"])
                ui_count["discipline_rate_an_ty"] = calc_rate(ui_count["disciplined_an_ty"], ui_count["_scanned_an_ty"])

            # --- DB ENTRY ---
            now = datetime.now()
            entry_data = {
                "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"),
                "enrollment": student_info["enrollment"] if student_info else "NA",
                "name": student_info["name"] if student_info else person_name,
                "department": student_info["department"] if student_info else "Unknown",
                "year": student_info["year"] if student_info else "Unknown",
                "mobile": student_info["mobile"] if student_info else "NA",
                "recognized": is_recognized, "discipline": is_disciplined,
                "id_card": id_detected, "shirt_tucked": is_tucked, "image_path": image_path
            }
            log_queue.put(entry_data)

            final_color = (0, 255, 0) if is_disciplined else (0, 0, 255)
            shirt_str = f"{shirt_status}/{'White' if is_white else 'Color'}"

            with results_lock:
                processed_results[track_id] = {
                    "name": display_name, "shirt": shirt_str, "id": "Yes" if id_detected else "No",
                    "color": final_color, "timestamp": time.time()
                }
            
            processing_ids.discard(track_id)
            processing_queue.task_done()

        except Exception as e:
            print(f"Worker Error: {e}")
            if 'track_id' in locals(): processing_ids.discard(track_id)

# START THREADS
for _ in range(3): threading.Thread(target=processing_worker, daemon=True).start()
threading.Thread(target=db_logger_worker, daemon=True).start()
threading.Thread(target=file_writer_worker, daemon=True).start()

# ---------------- CAMERA LOGIC LOOP ----------------
track_enter_time = {}
frame_buffer = {}

def camera_processing_loop():
    print("--- Camera Logic Started ---")
    while True:
        ret, frame = camera_stream.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_h, frame_w = frame.shape[:2]
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, iou=0.5, verbose=False, classes=[0])

        current_boxes = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            keypoints_data = results[0].keypoints.xy
            conf_data = results[0].keypoints.conf

            current_frame_ids = set()

            for i, box in enumerate(boxes):
                track_id = int(box.id[0])
                current_frame_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_boxes.append({"id": track_id, "coords": (x1, y1, x2, y2)})

                if track_id in processing_ids:
                    continue

                # --- OVERRIDE FOR PENDING UNKNOWNS: CAPTURE THEIR LATEST FRAME ---
                is_pending_unk = False
                with unknown_lock:
                    is_pending_unk = track_id in pending_unknowns

                if is_pending_unk:
                    if not is_full_body(x1, y2, frame_w, frame_h): continue
                    
                    kpts = keypoints_data[i]
                    confs = conf_data[i]
                    if not (keypoint_visible(kpts, confs, 0) and keypoint_visible(kpts, confs, 1)): continue
                    
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0: continue
                    
                    # Update buffer with the best, most recent crop
                    with unknown_lock:
                        if track_id in unknown_last_seen_data:
                            unknown_last_seen_data[track_id].update({
                                "last_seen": time.time(),
                                "crop": person_crop.copy(),
                                "full_frame": frame.copy(),
                                "kpts": kpts.cpu().numpy(),
                                "box": [x1, y1, x2, y2]
                            })
                            
                    # Prevent UI from timing out their label while they are still tracked
                    with results_lock:
                        if track_id in processed_results:
                            processed_results[track_id]["timestamp"] = time.time()
                    continue

                if track_id in saved_id:
                    continue

                # Normal stabilization logic for new IDs
                now = time.time()
                if track_id not in track_enter_time:
                    track_enter_time[track_id] = now
                if (now - track_enter_time[track_id]) < 1.0: continue
                if not is_full_body(x1, y2, frame_w, frame_h): continue

                kpts = keypoints_data[i]
                confs = conf_data[i]

                if not (keypoint_visible(kpts, confs, 0) and keypoint_visible(kpts, confs, 1)): continue
                if not ((ankle_knee_visible(kpts, confs, 13) or ankle_knee_visible(kpts, confs, 14))): continue

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0: continue

                score = blur_score(person_crop)
                if score < 40: continue

                frame_buffer.setdefault(track_id, [])
                frame_buffer[track_id].append({
                    "score": score, "crop": person_crop, "frame": frame.copy(),
                    "box": [x1, y1, x2, y2], "kpts": kpts.cpu().numpy()
                })

                frame_buffer[track_id] = sorted(frame_buffer[track_id], key=lambda x: x["score"], reverse=True)[:3]

                if len(frame_buffer[track_id]) >= 3:
                    best = frame_buffer[track_id][0]
                    packet = {
                        "track_id": track_id, "full_frame": best["frame"],
                        "best_person_crop": best["crop"], "keypoints": best["kpts"], "xyxy": best["box"]
                    }
                    try:
                        processing_queue.put_nowait(packet)
                        processing_ids.add(track_id)
                        saved_id.add(track_id)
                        del frame_buffer[track_id]
                    except queue.Full: pass
            
            with count_lock: ui_count["live_flow"] = len(current_frame_ids) 

        # --- PROCESS UNKNOWNS WHO JUST LEFT THE FRAME ---
        current_time = time.time()
        expired_unknowns = []
        with unknown_lock:
            for tid, u_data in unknown_last_seen_data.items():
                if current_time - u_data["last_seen"] > 2.0: # Track lost for > 2 seconds
                    expired_unknowns.append(tid)
            
            for tid in expired_unknowns:
                u_data = unknown_last_seen_data.pop(tid)
                pending_unknowns.discard(tid)
                
                packet = {
                    "track_id": tid, "full_frame": u_data["full_frame"],
                    "best_person_crop": u_data["crop"], "keypoints": u_data["kpts"],
                    "xyxy": u_data["box"], "is_final_leaving": True  # TRIGGERS FINAL SAVE
                }
                try:
                    processing_queue.put_nowait(packet)
                    processing_ids.add(tid)
                except queue.Full: pass

        with visual_lock:
            visual_data.clear()
            for b in current_boxes: visual_data[b["id"]] = b["coords"]

        time.sleep(0.01)

threading.Thread(target=camera_processing_loop, daemon=True).start()

# ---------------- API ROUTES ----------------
@app.route('/count')
def counts():
    with count_lock: return jsonify(ui_count)

@app.route("/entries")
def get_entries():
    try:
        with sqlite3.connect("college.db", timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM entries ORDER BY date DESC, time DESC LIMIT 200")
            rows = cursor.fetchall()
            return jsonify([dict(row) for row in rows])
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify([])
    
# -------------------------------------------------------------------------------

@app.route("/all_students")
def get_all_students():
    try:
        with sqlite3.connect("college.db", timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Fetch all registered students from your database
            cursor.execute("SELECT enrollment, name, department, year, mobile FROM students")
            rows = cursor.fetchall()
            
            # Map numeric years (1, 2, 3) to text ('First', 'Second', 'Third') so React filters work perfectly
            year_map = {1: 'First', 2: 'Second', 3: 'Third'}
            
            students_list = []
            for row in rows:
                s_dict = dict(row)
                try:
                    y_num = int(s_dict['year'])
                    s_dict['year'] = year_map.get(y_num, 'Unknown')
                except (ValueError, TypeError):
                    s_dict['year'] = 'Unknown'
                    
                students_list.append(s_dict)
                
            return jsonify(students_list)
    except Exception as e:
        print(f"API Error fetching all students: {e}")
        return jsonify([])
# -----------------------------------------

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            ret, frame = camera_stream.read()
            if not ret:
                time.sleep(0.02)
                continue
            
            current_time = time.time()
            with results_lock:
                expired_ids = [tid for tid, data in processed_results.items() if (current_time - data['timestamp']) > 2.0]
                for tid in expired_ids:
                    del processed_results[tid]
                    if tid in retry_counts: del retry_counts[tid]

            boxes_to_draw = {}
            with visual_lock: boxes_to_draw = visual_data.copy()
            
            labels_to_draw = {}
            with results_lock: labels_to_draw = processed_results.copy()
            
            for tid, coords in boxes_to_draw.items():
                x1, y1, x2, y2 = coords
                if tid in labels_to_draw:
                    data = labels_to_draw[tid]
                    label = data['name']
                    sub_label = f"{data['shirt']} | ID:{data['id']}"
                    color = data['color']
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, sub_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                elif tid in processing_ids:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "Processing...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --------------------------------------------------------------

@app.route("/evidence")
def serve_evidence():
    rel_path = request.args.get("path", "")
    abs_path = os.path.normpath(os.path.join(ROOT_DIR, rel_path))
    if not abs_path.startswith(BASE_DIR): return "Access denied", 403
    if not os.path.isfile(abs_path): return "File not found", 404
    return send_file(abs_path, mimetype="image/jpeg")

# ----------------------------------------------------------------

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if not data: return jsonify({"error": "No data"}), 400
    username, password = data.get("username"), data.get("password")
    try:
        with sqlite3.connect("users.db", timeout=10.0) as conn:
            cur = conn.cursor()
            cur.execute("SELECT password_hash, role FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            
        if not row: return jsonify({"error": "Invalid username"}), 401
        pw_hash, role = row
        if isinstance(pw_hash, str): pw_hash = pw_hash.encode('utf-8')
        if bcrypt.checkpw(password.encode('utf-8'), pw_hash):
            token = jwt.encode({
                "username": username, "role": role,
                "exp": datetime.utcnow() + timedelta(hours=12)
            }, app.config["SECRET_KEY"], algorithm="HS256")
            return jsonify({"token": token, "role": role})
        return jsonify({"error": "Invalid password"}), 401
    except Exception as e: return jsonify({"error": "Server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)