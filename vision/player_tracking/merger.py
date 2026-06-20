"""Smart Merging logic for player tracking using court geometry."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core_math.homography.projector import CourtProjector

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """State of a tracked entity."""
    track_id: int
    last_court_pos: tuple[float, float]
    frames_since_update: int = 0
    assigned_slot: int | None = None  # 0, 1 for Near, 2, 3 for Far


class PlayerMerger:
    """Merges fragmented ByteTrack IDs into 4 persistent player slots.

    Slots:
        0, 1: Near Team (y < 10m)
        2, 3: Far Team (y >= 10m)
    """

    def __init__(self, projector: CourtProjector, max_lost_frames: int = 60) -> None:
        """Initialize the merger.

        Args:
            projector: Calibrated CourtProjector instance.
            max_lost_frames: How many frames to remember a lost track.
        """
        self.projector = projector
        self.max_lost_frames = max_lost_frames

        # State: track_id -> TrackState
        self.tracks: dict[int, TrackState] = {}

        # Mapping from ByteTrack ID to Persistent Slot (0-3)
        self.id_to_slot: dict[int, int] = {}

        # Which track_id currently occupies which slot
        self.slot_occupancy: dict[int, int | None] = {0: None, 1: None, 2: None, 3: None}

    def _get_side(self, court_y: float) -> str:
        """Return 'near' or 'far' based on y coordinate."""
        return "near" if court_y < (self.projector.COURT_LENGTH_M / 2.0) else "far"

    def _get_available_slots(self, side: str) -> list[int]:
        """Get available slots (0,1 for near; 2,3 for far)."""
        slots = [0, 1] if side == "near" else [2, 3]
        return [s for s in slots if self.slot_occupancy[s] is None]

    def _find_best_lost_track(
        self, court_pos: tuple[float, float], side: str, max_dist_m: float = 3.0
    ) -> int | None:
        """Find the best recently lost track on the same side."""
        best_id = None
        min_dist = float("inf")

        for t_id, state in self.tracks.items():
            if state.frames_since_update > 0 and state.assigned_slot is not None:
                # Check if it was on the same side
                lost_side = self._get_side(state.last_court_pos[1])
                if lost_side == side:
                    dist = math.dist(state.last_court_pos, court_pos)
                    if dist < min_dist and dist < max_dist_m:
                        min_dist = dist
                        best_id = t_id

        return best_id

    def update(
        self,
        boxes: NDArray[np.float64],
        track_ids: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """Update track states and return stable slots.

        Args:
            boxes: Array of shape (N, 4) with bounding boxes.
            track_ids: Array of shape (N,) with ByteTrack IDs.

        Returns:
            Array of shape (N,) with stable slot IDs (0-3), or -1 for unassigned/noise.
        """
        result_slots = np.full_like(track_ids, -1)

        # Increment lost frames for all existing tracks
        for state in self.tracks.values():
            state.frames_since_update += 1

        # Free up slots for tracks that have been lost for too long
        keys_to_remove = []
        for t_id, state in self.tracks.items():
            if state.frames_since_update > self.max_lost_frames:
                if state.assigned_slot is not None and self.slot_occupancy[state.assigned_slot] == t_id:
                    self.slot_occupancy[state.assigned_slot] = None
                keys_to_remove.append(t_id)

        for k in keys_to_remove:
            del self.tracks[k]

        # Process current frame detections
        for i, (box, t_id) in enumerate(zip(boxes, track_ids, strict=False)):
            try:
                # Get 2D court position
                court_pt = self.projector.project_bounding_box_bottom(box)
                cx, cy = float(court_pt[0]), float(court_pt[1])
            except RuntimeError:
                continue  # Homography failed or uncalibrated

            # Filter noise outside court margins (-4m to +4m roughly)
            if not (-4 < cx < self.projector.COURT_WIDTH_M + 4):
                continue
            if not (-4 < cy < self.projector.COURT_LENGTH_M + 4):
                continue

            side = self._get_side(cy)

            # Is this a known track?
            if t_id in self.tracks:
                state = self.tracks[t_id]
                state.last_court_pos = (cx, cy)
                state.frames_since_update = 0
                result_slots[i] = state.assigned_slot if state.assigned_slot is not None else -1
            else:
                # NEW TRACK DETECTED! Need to assign it to a slot.
                best_lost_id = self._find_best_lost_track((cx, cy), side)

                assigned_slot = None
                if best_lost_id is not None:
                    # Inherit the slot from the lost track
                    lost_state = self.tracks[best_lost_id]
                    assigned_slot = lost_state.assigned_slot
                    # Free the old track
                    del self.tracks[best_lost_id]
                    logger.info("Stitched track %d to lost track %d (Slot %s)", t_id, best_lost_id, assigned_slot)
                else:
                    # Claim an empty slot
                    available = self._get_available_slots(side)
                    if available:
                        assigned_slot = available[0]
                        logger.info("Assigned new track %d to slot %d", t_id, assigned_slot)
                    else:
                        # FORCE claim a slot that wasn't updated THIS frame
                        # If a slot is occupied by a track that was NOT updated this frame (frames_since_update > 0),
                        # it means the tracker lost them, so we just steal their slot!
                        side_slots = [0, 1] if side == "near" else [2, 3]
                        for s in side_slots:
                            occupying_tid = self.slot_occupancy[s]
                            if (occupying_tid is not None and 
                                occupying_tid in self.tracks and 
                                self.tracks[occupying_tid].frames_since_update > 0):
                                # Steal the slot!
                                assigned_slot = s
                                del self.tracks[occupying_tid]
                                logger.info("Track %d stole slot %d from stale track %d", t_id, assigned_slot, occupying_tid)
                                break

                        if assigned_slot is None:
                            logger.warning("No available slot on %s side for track %d (likely noise)", side, t_id)

                if assigned_slot is not None:
                    self.slot_occupancy[assigned_slot] = t_id

                self.tracks[t_id] = TrackState(
                    track_id=t_id,
                    last_court_pos=(cx, cy),
                    frames_since_update=0,
                    assigned_slot=assigned_slot
                )
                self.id_to_slot[t_id] = assigned_slot if assigned_slot is not None else -1
                result_slots[i] = assigned_slot if assigned_slot is not None else -1

        # Check for slot conflicts (if somehow 2 active tracks claim same slot)
        # In a perfect world, only 1 track maps to a slot at a time.
        # Simple cleanup: if a slot is occupied by a track that isn't active this frame, it's fine.

        return result_slots
