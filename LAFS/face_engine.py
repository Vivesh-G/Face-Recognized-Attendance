import os
import pickle
import numpy as np
import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
from .PartfVit import ViT_face_landmark_patch8


class FaceEngine:
    def __init__(self, model_path: str = None, threshold: float = 0.5, device: str = None):
        """
        Initialize Part-fViT Face Engine.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        # MTCNN: detection only
        self.detector = MTCNN(
            keep_all=True,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            device=self.device,
        )

        # Part-fViT: landmark-aware face embedding
        self.model = ViT_face_landmark_patch8(
            image_size=112,
            patch_size=8,
            dim=768,
            depth=12,
            heads=11,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            num_patches=196
        ).to(self.device).eval()

        # Load checkpoint if provided
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint
            
                model_dict = self.model.state_dict()
                compatible_keys = {}
                for k, v in state_dict.items():
                    clean_k = k.replace("module.", "").replace("backbone.", "").replace("encoder.", "")
                    if clean_k in model_dict and v.shape == model_dict[clean_k].shape:
                        compatible_keys[clean_k] = v
                
                if compatible_keys:
                    self.model.load_state_dict(compatible_keys, strict=False)
                    print(f"[INFO] Loaded {len(compatible_keys)}/{len(model_dict)} compatible weights from checkpoint")
                else:
                    print(f"[WARN] Checkpoint incompatible with current architecture (old MobileNetV3 vs new EdgeFace). Using pretrained EdgeFace.")
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint ({e}). Using pretrained EdgeFace.")

        # Preprocessing: 
        # resize to 112x112, 
        # normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.db: dict = {}



    def detect(self, frame_rgb: np.ndarray):
        """
        Return (boxes_list, probs_list) filtered by confidence >= 0.9.
        Implemented from MTCNN Code
        """
        img_pil = Image.fromarray(frame_rgb)
        boxes, probs = self.detector.detect(img_pil)
        if boxes is None:
            return [], []
        valid_boxes, valid_probs = [], []
        h, w = frame_rgb.shape[:2]
        for box, prob in zip(boxes, probs):
            if prob < 0.9:
                continue
            x1, y1, x2, y2 = (
                max(0, int(box[0])), max(0, int(box[1])),
                min(w, int(box[2])), min(h, int(box[3])),
            )
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            valid_boxes.append([x1, y1, x2, y2])
            valid_probs.append(float(prob))
        return valid_boxes, valid_probs

    def _crop_and_transform(self, frame_rgb: np.ndarray, box: list) -> torch.Tensor:
        x1, y1, x2, y2 = box
        crop = Image.fromarray(frame_rgb[y1:y2, x1:x2])
        return self.transform(crop)

    @torch.no_grad()
    def get_embeddings(self, frame_rgb: np.ndarray, boxes: list) -> np.ndarray:
        """Batch-extract L2-normalised embeddings for given boxes using Part-fViT."""
        if not boxes:
            return np.array([])
        tensors = [self._crop_and_transform(frame_rgb, b) for b in boxes]
        batch = torch.stack(tensors).to(self.device)
        embs = self.model(batch)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs.cpu().numpy()

    def identify(self, embedding: np.ndarray) -> tuple[str, float]:
        """Match embedding against database. Returns (name, similarity)."""
        if not self.db:
            return "UNK", 0.0
        best_name, best_sim = "UNK", -1.0
        for name, data in self.db.items():
            sim = float(np.dot(embedding, data["emb"]))
            if sim > best_sim:
                best_sim, best_name = sim, name
        if best_sim < self.threshold:
            return "UNK", best_sim
        return best_name, best_sim

    def process_frame(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Detect + recognize faces in a BGR frame.
        Returns: [{name, sim, box}, ...]
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = self.detect(frame_rgb)
        if not boxes:
            return []

        embs = self.get_embeddings(frame_rgb, boxes)
        results = []
        for box, emb in zip(boxes, embs):
            name, sim = self.identify(emb)
            results.append({
                "name": name,
                "sim": sim,
                "box": box,
            })
        return results

    # Registration
    def enroll(self, name: str, image_path: str, meta: dict = None) -> bool:
        """
        Enroll a person from an image. Returns True if successful.
        
        Args:
            name: Person's name
            image_path: Path to face image
            meta: Optional metadata {class, dept, etc}
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERROR] Failed to read {image_path}")
                return False

            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, probs = self.detect(frame_rgb)
            if not boxes:
                print(f"[WARN] No face detected in {image_path}")
                return False

            embs = self.get_embeddings(frame_rgb, boxes)
            # Use the first face (highest confidence)
            self.db[name] = {
                "emb": embs[0],
                "meta": meta or {},
            }
            print(f"[INFO] Enrolled {name}")
            return True
        except Exception as e:
            print(f"[ERROR] Enrollment failed for {name}: {e}")
            return False

    def load_db(self, embeddings_dir: str) -> None:
        """Load embeddings from .npy files in directory."""
        os.makedirs(embeddings_dir, exist_ok=True)
        for fname in os.listdir(embeddings_dir):
            if fname.endswith('.npy'):
                name = fname[:-4]
                emb_path = os.path.join(embeddings_dir, fname)
                meta_path = os.path.join(embeddings_dir, f"{name}_meta.pkl")
                
                emb = np.load(emb_path)
                meta = {}
                if os.path.exists(meta_path):
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                
                self.db[name] = {"emb": emb, "meta": meta}
        print(f"[INFO] Loaded {len(self.db)} identities from {embeddings_dir}")

    def save_embedding(self, name: str, embeddings_dir: str) -> None:
        """Save an embedding to disk."""
        if name not in self.db:
            return
        os.makedirs(embeddings_dir, exist_ok=True)
        np.save(os.path.join(embeddings_dir, f"{name}.npy"), self.db[name]["emb"])
        with open(os.path.join(embeddings_dir, f"{name}_meta.pkl"), 'wb') as f:
            pickle.dump(self.db[name]["meta"], f)

    def save_all(self, embeddings_dir: str) -> None:
        """Save all embeddings to disk."""
        for name in self.db.keys():
            self.save_embedding(name, embeddings_dir)
