import numpy as np

from video.face_tracker import FaceTracker


class StubDetector:
    def __init__(self, detections, persistent=False):
        self._detections = iter(detections)
        self._persistent = persistent

    def has_persistent_second_face(self, _frames):
        return self._persistent, 2 if self._persistent else 0

    def detect(self, _frame):
        return list(next(self._detections))

    def crop(self, frame, _box):
        return frame


def test_transient_second_face_box_does_not_reject_deepfake_sequence():
    primary = [(220, 100, 160, 160)]
    transient_multi = primary + [(40, 100, 100, 100)]
    tracker = FaceTracker(StubDetector([transient_multi, primary, primary], persistent=False))
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3

    result = tracker.track_sequence(frames)

    assert result["status"] == "OK"
    assert len(result["face_crops"]) == 3


def test_persistent_second_face_remains_a_hard_rejection():
    two_faces = [(220, 100, 160, 160), (40, 100, 100, 100)]
    tracker = FaceTracker(StubDetector([two_faces, two_faces, two_faces], persistent=True))
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3

    result = tracker.track_sequence(frames)

    assert result["status"] == "MULTIPLE_FACES_DETECTED"
    assert result["persistent_face_count"] == 2
