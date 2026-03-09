import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from datetime import datetime
import os
from LAFS import PartfVit

# Attendance System with Part-fViT
class AttendanceSystem:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.face_detector = MTCNN(keep_all=True, device=self.device)
        
        # Initialize Part-fViT (Landmark-aware Vision Transformer)
        self.recognition_model = PartfVit.ViT_face_landmark_patch8(
            image_size=112,
            patch_size=8,
            dim=768,
            depth=12,
            heads=11,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            num_patches=196  # 14x14 landmark grid
        )
        
        self.load_checkpoint(model_path)
        self.recognition_model.to(self.device)
        self.recognition_model.eval()
        
        self.known_embeddings = {}
        self.attendance_log = set()

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        print(f"Loading Part-fViT checkpoint from {path}...")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats (teacher/student/model)
        if 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Clean keys - remove common prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "").replace("backbone.", "").replace("encoder.", "")
            new_state_dict[new_k] = v
            
        # Load compatible keys only (handles different model variants)
        model_dict = self.recognition_model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.recognition_model.load_state_dict(model_dict)
        
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from checkpoint.")
        
        # Report which key components were loaded
        stn_loaded = sum(1 for k in pretrained_dict if 'stn' in k)
        output_loaded = sum(1 for k in pretrained_dict if 'output_layer' in k)
        transformer_loaded = sum(1 for k in pretrained_dict if 'transformer' in k)
        print(f"  - Landmark backbone (stn): {stn_loaded} layers")
        print(f"  - Output layer: {output_loaded} layers")
        print(f"  - Transformer: {transformer_loaded} layers")

    def preprocess(self, img_bgr):
        """Preprocess face image for Part-fViT: BGR to RGB, resize to 112x112, normalize"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (112, 112))
        
        # Normalize (0.5 mean/std as per LAFS)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        img_tensor = img_tensor / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_tensor.unsqueeze(0).to(self.device)

    def get_embedding(self, face_img):
        """Get normalized face embedding from Part-fViT"""
        tensor = self.preprocess(face_img)
        with torch.no_grad():
            embedding = self.recognition_model(tensor)
            return torch.nn.functional.normalize(embedding).cpu().numpy().flatten()

    def register_student(self, name, image_path):
        """Register a student with their face image"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image for {name}")
            return

        # Detect face for registration
        boxes, _ = self.face_detector.detect(img)
        if boxes is not None:
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                print(f"Error: Empty face crop for {name}")
                return
            
            embedding = self.get_embedding(face)
            self.known_embeddings[name] = embedding
            print(f"✓ Registered: {name} (Part-fViT embedding: {embedding.shape[0]}-dim)")
        else:
            print(f"✗ No face detected in {image_path}")

    def mark_attendance(self, name):
        """Mark attendance for a recognized person"""
        if name not in self.attendance_log:
            time_now = datetime.now().strftime('%H:%M:%S')
            print(f"[ATTENDANCE] {name} marked at {time_now}")
            self.attendance_log.add(name)

    def run_live_tracking(self, threshold=0.6):
        """Run live face tracking and attendance marking (lower threshold for Part-fViT)"""
        cap = cv2.VideoCapture(0)
        print("Starting Part-fViT Live Tracking... Press 'q' to quit.")
        print(f"Recognition threshold: {threshold}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            boxes, _ = self.face_detector.detect(frame)
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    
                    # Ensure coordinates are within frame
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    # Get live embedding from Part-fViT
                    live_emb = self.get_embedding(face_crop)
                    
                    # Compare with database using cosine similarity
                    best_score = -1
                    best_name = "Unknown"
                    
                    for name, ref_emb in self.known_embeddings.items():
                        score = np.dot(live_emb, ref_emb)  # Cosine similarity (normalized vectors)
                        if score > best_score:
                            best_score = score
                            best_name = name
                    
                    # Visualization
                    color = (0, 0, 255)  # Red for Unknown
                    label = f"Unknown ({best_score:.2f})"
                    
                    if best_score > threshold:
                        color = (0, 255, 0)  # Green for Known
                        label = f"{best_name} ({best_score:.2f})"
                        self.mark_attendance(best_name)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add model info overlay
            cv2.putText(frame, "Model: Part-fViT (LAFS)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Attendance System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 1. Initialize System with Part-fViT
    system = AttendanceSystem(model_path='lafs_webface_finetune_withaugmentation.pth')

    # 2. Register Students
    system.register_student("Test1", "test1.jpg")
    system.register_student("Test2", "test2.jpg")
    system.register_student("Test3", "test3.jpg")
    
    # 3. Start Live Tracking
    system.run_live_tracking(threshold=0.6)