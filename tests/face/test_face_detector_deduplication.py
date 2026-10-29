from video.face_detector import FaceDetector
import numpy as np


def test_overlapping_detector_boxes_are_one_face():
    boxes = [(100, 100, 120, 120), (105, 104, 118, 121)]

    indices = FaceDetector._deduplicated_indices(boxes)

    assert len(indices) == 1


def test_distinct_detector_boxes_remain_multiple_faces():
    boxes = [(40, 80, 100, 100), (320, 80, 100, 100)]

    indices = FaceDetector._deduplicated_indices(boxes)

    assert len(indices) == 2


def test_credible_box_filters_low_confidence_tiny_and_edge_noise():
    detector = FaceDetector(min_face_size=80)

    assert not detector._is_credible_box(100, 100, 100, 100, 0.69, 640, 480)
    assert not detector._is_credible_box(100, 100, 70, 70, 0.95, 640, 480)
    assert not detector._is_credible_box(-10, 100, 20, 100, 0.95, 640, 480)
    assert detector._is_credible_box(180, 120, 100, 100, 0.95, 640, 480)


def test_second_face_requires_consecutive_frames_before_rejection(monkeypatch):
    detector = FaceDetector(multi_face_persistence_frames=3)
    primary = [(220, 100, 160, 160)]
    two_faces = primary + [(40, 100, 120, 120)]
    detections = iter([two_faces, primary, two_faces, two_faces, two_faces])
    monkeypatch.setattr(detector, "detect", lambda _: next(detections))

    found, count = detector.has_persistent_second_face([np.zeros((480, 640, 3), dtype=np.uint8)] * 5)

    assert found is True
    assert count == 2
