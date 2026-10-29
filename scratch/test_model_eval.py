import torch
import cv2
import numpy as np
import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, "backend")

from models.face_liveness import PassiveAntiSpoofNet, MultiDomainFeatureExtractor, load_passive_liveness_model

def main():
    model = load_passive_liveness_model("cpu")
    ckpt = torch.load("checkpoints/liveness_best.pth", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    
    cap = cv2.VideoCapture("datasets/genuine/genuine_user_01.mp4")
    ret, frame = cap.read()
    cap.release()
    
    from video.face_detector import global_face_detector
    proc = global_face_detector.process_frame(frame)
    crop = proc["face_crop"]
    
    resized = cv2.resize(crop, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = (tensor - mean) / std
    batch = normalized.unsqueeze(0)
    
    aux_vec = MultiDomainFeatureExtractor.extract_auxiliary_vector(crop)
    aux_t = torch.from_numpy(aux_vec).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        logits_eval = model(batch, aux_t)
        probs_eval = torch.softmax(logits_eval, dim=1)
        
    model.train()
    with torch.no_grad():
        logits_train = model(batch, aux_t)
        probs_train = torch.softmax(logits_train, dim=1)
        
    print("Eval mode probs:", probs_eval)
    print("Train mode probs:", probs_train)

if __name__ == "__main__":
    main()
