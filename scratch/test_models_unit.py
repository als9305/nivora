"""
Unit test verification for DeepQShield PyTorch model components (Exact Paper Architecture).
"""

import sys
import torch

sys.path.insert(0, "d:/projects/DeepGuard")

from models.resnext_backbone import ResNeXtBackbone
from models.lwe_layer import LWELayer
from models.feature_fusion import FeatureFusion
from models.attention import FeatureAttention
from models.classifier import ClassifierHead
from models.deepqshield import DeepQShieldModel
from models.temporal_aggregator import TemporalAggregator


def test_deepqshield_pipeline():
    print("Testing DeepQShield model pipeline (Exact Paper Specification)...")
    
    # Dummy input image tensor (Batch size=2, 3 channels, 224x224)
    x = torch.randn(2, 3, 224, 224)
    
    # 1. Test Backbone
    backbone = ResNeXtBackbone(pretrained=False)
    b_feats = backbone(x)
    assert b_feats.shape == (2, 2048), f"Backbone output shape mismatch: {b_feats.shape}"
    print("[PASS] ResNeXtBackbone shape: (2, 2048)")
    
    # 2. Test TWO-Stage LWE Layer (2048 -> 256 -> 256)
    lwe = LWELayer(in_features=2048, lattice_dim=256, noise_std=0.1, seed=42)
    l_feats = lwe(b_feats)
    assert l_feats.shape == (2, 256), f"LWE output shape mismatch: {l_feats.shape}"
    print("[PASS] LWELayer (2-Stage 256-dim) shape: (2, 256)")
    
    # 3. Test Feature Fusion (Concat 2048 + 256 = 2304 -> Projected 2048)
    fusion = FeatureFusion(backbone_dim=2048, lwe_dim=256, fused_dim=2048)
    f_feats = fusion(b_feats, l_feats)
    assert f_feats.shape == (2, 2048), f"Fusion output shape mismatch: {f_feats.shape}"
    print("[PASS] FeatureFusion shape: (2, 2048)")
    
    # 4. Test Attention
    attention = FeatureAttention(feature_dim=2048)
    a_feats, attn_w = attention(f_feats)
    assert a_feats.shape == (2, 2048), f"Attention output shape mismatch: {a_feats.shape}"
    assert attn_w.shape == (2, 2048), f"Attention weights shape mismatch: {attn_w.shape}"
    print("[PASS] FeatureAttention shape: (2, 2048)")
    
    # 5. Test Classifier Head
    classifier = ClassifierHead(in_features=2048, hidden_dim=512, dropout_rate=0.3)
    logits = classifier(a_feats)
    assert logits.shape == (2, 2), f"Classifier logits shape mismatch: {logits.shape}"
    print("[PASS] ClassifierHead shape: (2, 2)")
    
    # 6. Test Full DeepQShield Model
    model = DeepQShieldModel(pretrained_backbone=False, lattice_dim=256, noise_std=0.1, seed=42)
    full_logits = model(x)
    assert full_logits.shape == (2, 2), f"Full model logits shape mismatch: {full_logits.shape}"
    print("[PASS] DeepQShieldModel forward pass shape: (2, 2)")
    
    # Test single prediction
    pred = model.predict_single(x[0])
    assert "fake_probability" in pred and "predicted_class" in pred
    print(f"[PASS] DeepQShieldModel predict_single output: {pred}")
    
    # 7. Test Temporal Aggregator
    aggregator = TemporalAggregator(strategy="suspicious_weighted", suspicious_threshold=0.50)
    scores = [0.1, 0.2, 0.85, 0.90, 0.15]
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]
    agg_res = aggregator.aggregate(scores, timestamps)
    assert agg_res["label"] == "DEEPFAKE"
    assert len(agg_res["suspicious_intervals"]) == 1
    print(f"[PASS] TemporalAggregator output: {agg_res}")
    
    print("\nALL DEEPQSHIELD MODEL UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_deepqshield_pipeline()
