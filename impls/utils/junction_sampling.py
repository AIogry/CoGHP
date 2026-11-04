"""Training-only, matched resampling of ambiguous junction high-level queries.

The topology is inferred from occupied training cells. It identifies junctions;
it never ranks exits by a shortest-path oracle and is never a model input.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

OFFSET_BIN_EDGES = np.array([0, 50, 100, 150, 300, 600, 1200, np.iinfo(np.int32).max], dtype=np.int64)
EVENT_FIELDS = (
    "episode", "junction", "incoming", "first_exit", "entry_index",
    "last_core_index", "exit_index", "return_index", "episode_end",
)


def _pack(arrays):
    lengths = np.array([len(x) for x in arrays], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    values = np.concatenate(arrays).astype(np.int32) if arrays else np.empty(0, np.int32)
    return values, offsets


def _unpack(values, offsets):
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


@dataclass
class JunctionRouteIndex:
    """Immutable-by-convention geometry/events and exact real-query donors."""

    geometry: dict
    frame_cells: np.ndarray
    frame_visits: np.ndarray
    frame_episodes: np.ndarray
    episode_ends: np.ndarray
    events: dict
    donor_starts: list
    segment_visits: np.ndarray
    segment_cells: np.ndarray
    segment_goals: list
    candidate_segments: np.ndarray
    candidate_bins: np.ndarray
    pre_return_seen: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.pools = defaultdict(list)
        for rid, (segment, bin_id) in enumerate(zip(self.candidate_segments, self.candidate_bins)):
            visit = int(self.segment_visits[segment])
            key = (int(self.events["junction"][visit]), int(self.events["incoming"][visit]),
                   int(self.segment_cells[segment]), int(bin_id))
            self.pools[key].append(rid)
        self.pools = {key: np.asarray(value, np.int32) for key, value in self.pools.items()}

    @property
    def signature(self):
        return self.metadata["signature"]

    @classmethod
    def build(cls, observations, raw_episode_ends, cell_width=4.0, origin_xy=(-4.0, -4.0), core_half_width=1.6):
        started = time.monotonic()
        observations = np.asarray(observations)
        ends = np.asarray(raw_episode_ends, dtype=np.int64)
        if observations.ndim != 2 or observations.shape[1] < 2 or not len(observations):
            raise ValueError("Expected nonempty flat observations with XY in the first two dimensions.")
        if not np.isfinite(observations[:, :2]).all():
            raise ValueError("Nonfinite training XY.")
        if len(ends) == 0 or ends[-1] != len(observations)-1 or np.any(np.diff(ends) <= 0) or ends[0] < 0:
            raise ValueError("Raw episode ends must partition all observations including final states.")
        if cell_width <= 0 or not 0 < core_half_width < cell_width / 2:
            raise ValueError("Invalid cell width/core half-width.")
        origin = np.asarray(origin_xy, dtype=float)
        xy_cells = np.floor((observations[:, :2] - origin + cell_width / 2) / cell_width).astype(np.int32)
        if np.any(xy_cells < 0):
            raise ValueError("Training lattice coordinates precede configured origin.")
        ncols = int(xy_cells[:, 0].max()) + 2
        nrows = int(xy_cells[:, 1].max()) + 2
        cells = xy_cells[:, 1] * ncols + xy_cells[:, 0]
        free = set(map(int, np.unique(cells)))
        neighbors = {}
        for cell in sorted(free):
            row, col = divmod(cell, ncols)
            neighbors[cell] = [r*ncols+c for r, c in ((row-1,col),(row+1,col),(row,col-1),(row,col+1))
                               if 0 <= r < nrows and 0 <= c < ncols and r*ncols+c in free]
        junctions = {cell for cell, adjacent in neighbors.items() if len(adjacent) >= 3}
        centers = origin + xy_cells * cell_width
        is_core = np.max(np.abs(observations[:, :2] - centers), axis=1) <= core_half_width
        core = np.where(is_core, cells, -1)
        frame_visit = np.full(len(observations), -1, np.int32)
        frame_episode = np.empty(len(observations), np.int32)
        records = []
        starts_by_visit = []
        for episode, (lo, hi) in enumerate(zip(np.r_[0, ends[:-1]+1], ends)):
            frame_episode[lo:hi+1] = episode
            hits = np.flatnonzero(core[lo:hi+1] >= 0) + lo
            if not len(hits):
                continue
            hit_cells = core[hits]
            boundaries = np.flatnonzero(np.r_[True, np.diff(hit_cells) != 0])
            run_cells = hit_cells[boundaries]
            run_start = hits[boundaries]
            run_last = hits[np.r_[boundaries[1:]-1, len(hits)-1]]
            next_same = np.full(len(run_cells), -1, np.int32)
            seen = {}
            for k in range(len(run_cells)-1, -1, -1):
                next_same[k] = seen.get(int(run_cells[k]), -1)
                seen[int(run_cells[k])] = k
            for k, junction in enumerate(run_cells):
                junction = int(junction)
                if junction not in junctions:
                    continue
                before = int(run_cells[k-1]) if k else -1
                after = int(run_cells[k+1]) if k+1 < len(run_cells) else -1
                incoming = before if before in neighbors[junction] else -1
                outgoing = after if after in neighbors[junction] else -1
                exit_index = int(run_start[k+1]) if outgoing >= 0 else -1
                return_index = int(run_start[next_same[k]]) if next_same[k] >= 0 else -1
                assigned = hits[boundaries[k]:(boundaries[k+1] if k+1 < len(boundaries) else len(hits))]
                visit = len(records)
                frame_visit[assigned] = visit
                records.append((episode, junction, incoming, outgoing, int(run_start[k]),
                                int(run_last[k]), exit_index, return_index, int(hi)))
                valid_starts = assigned[(assigned <= hi-1) & (assigned + 50 >= exit_index)] if exit_index >= 0 else np.empty(0, np.int64)
                starts_by_visit.append(valid_starts.astype(np.int32))
        event_matrix = np.asarray(records, dtype=np.int32).reshape(-1, len(EVENT_FIELDS))
        events = {name: event_matrix[:, i] for i, name in enumerate(EVENT_FIELDS)}
        pre_seen = np.zeros((len(records), nrows*ncols), bool)
        segment_visits, segment_cells, segment_goals = [], [], []
        candidate_segments, candidate_bins = [], []
        for visit in range(len(records)):
            if events["incoming"][visit] < 0 or events["first_exit"][visit] < 0:
                continue
            exit_index = int(events["exit_index"][visit])
            ret = int(events["return_index"][visit])
            last = int(events["episode_end"][visit])-1
            stop = min(last+1, ret) if ret >= 0 else last+1
            if stop <= exit_index:
                continue
            future_cells = cells[exit_index:stop]
            pre_seen[visit, np.unique(future_cells)] = True
            starts = starts_by_visit[visit]
            if len(starts) == 0:
                continue
            for goal_cell in np.unique(future_cells):
                if goal_cell == events["junction"][visit]:
                    continue
                goals = (np.flatnonzero(future_cells == goal_cell) + exit_index).astype(np.int32)
                segment = len(segment_goals)
                segment_visits.append(visit)
                segment_cells.append(goal_cell)
                segment_goals.append(goals)
                for bin_id, (low, high) in enumerate(zip(OFFSET_BIN_EDGES[:-1], OFFSET_BIN_EDGES[1:])):
                    left = np.searchsorted(goals, starts.astype(np.int64)+low)
                    right = np.searchsorted(goals, starts.astype(np.int64)+high)
                    if np.any(left < right):
                        candidate_segments.append(segment)
                        candidate_bins.append(bin_id)
        geometry = dict(
            cell_width=float(cell_width), origin_xy=origin.tolist(), core_half_width=float(core_half_width),
            free_cells=[list(divmod(cell,ncols)) for cell in sorted(free)],
            junction_cells=[list(divmod(cell,ncols)) for cell in sorted(junctions)],
            adjacency={f"{r},{c}":[list(divmod(x,ncols)) for x in neighbors[cell]]
                       for cell in sorted(free) for r,c in [divmod(cell,ncols)]},
            lattice_shape=[nrows,ncols],
            provenance="occupied TRAIN observation XY cells; orthogonal occupied-cell adjacency; no BFS ranking",
        )
        signature=hashlib.sha256()
        signature.update(b"junction_soft_v1:index-v1")
        signature.update(json.dumps(dict(shape=observations.shape,dtype=observations.dtype.str,
            geometry=geometry,offset_bins=OFFSET_BIN_EDGES.tolist()),sort_keys=True).encode())
        signature.update(memoryview(np.ascontiguousarray(observations)).cast("B"))
        signature.update(memoryview(np.ascontiguousarray(ends)).cast("B"))
        # Include the actual event/donor mapping, so an algorithm change that
        # changes eligible queries cannot silently reuse a checkpoint signature.
        for array in (event_matrix,frame_visit,np.asarray(segment_visits,np.int32),
                      np.asarray(segment_cells,np.int32),np.asarray(candidate_segments,np.int32),
                      np.asarray(candidate_bins,np.int8),pre_seen,*_pack(starts_by_visit),*_pack(segment_goals)):
            signature.update(json.dumps([array.shape,array.dtype.str]).encode())
            if array.size:
                signature.update(memoryview(np.ascontiguousarray(array)).cast("B"))
        metadata = dict(schema_version=1, frames=len(observations), episodes=len(ends),
                        physical_visits=len(records), occupied_cells=len(free), junctions=len(junctions),
                        core_junction_frames=int((frame_visit >= 0).sum()),
                        donor_segments=len(segment_goals), donor_candidate_records=len(candidate_segments),
                        signature=signature.hexdigest(),
                        offset_bin_edges=OFFSET_BIN_EDGES.tolist(), build_seconds=time.monotonic()-started,
                        source="training observations and raw episode boundaries only",
                        goal_rule="selected actual goal index lies after first exit and before first return",
                        start_rule="core frame with nearest50 label at/after first exit",
                        effective_episode_end="raw final observation index minus one, matching compact sampler")
        return cls(geometry,cells.astype(np.int32),frame_visit,frame_episode,ends.astype(np.int32),events,
                   starts_by_visit,np.asarray(segment_visits,np.int32),np.asarray(segment_cells,np.int32),
                   segment_goals,np.asarray(candidate_segments,np.int32),np.asarray(candidate_bins,np.int8),pre_seen,metadata)

    def save(self, prefix):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        start_values, start_offsets = _pack(self.donor_starts)
        goal_values, goal_offsets = _pack(self.segment_goals)
        np.savez_compressed(str(prefix)+".npz", frame_cells=self.frame_cells, frame_visits=self.frame_visits,
            frame_episodes=self.frame_episodes, episode_ends=self.episode_ends,
            event_matrix=np.stack([self.events[k] for k in EVENT_FIELDS],axis=1),
            start_values=start_values,start_offsets=start_offsets,goal_values=goal_values,goal_offsets=goal_offsets,
            segment_visits=self.segment_visits,segment_cells=self.segment_cells,
            candidate_segments=self.candidate_segments,candidate_bins=self.candidate_bins,pre_return_seen=self.pre_return_seen)
        Path(str(prefix)+".json").write_text(json.dumps(dict(metadata=self.metadata,geometry=self.geometry),indent=2)+"\n")

    @classmethod
    def load(cls, prefix):
        prefix = Path(prefix)
        meta = json.loads(Path(str(prefix)+".json").read_text())
        if meta["metadata"]["schema_version"] != 1:
            raise ValueError("Unsupported junction index schema.")
        with np.load(str(prefix)+".npz",allow_pickle=False) as z:
            events={k:z["event_matrix"][:,i].copy() for i,k in enumerate(EVENT_FIELDS)}
            return cls(meta["geometry"],z["frame_cells"].copy(),z["frame_visits"].copy(),z["frame_episodes"].copy(),
                z["episode_ends"].copy(),events,_unpack(z["start_values"].copy(),z["start_offsets"].copy()),
                z["segment_visits"].copy(),z["segment_cells"].copy(),
                _unpack(z["goal_values"].copy(),z["goal_offsets"].copy()),
                z["candidate_segments"].copy(),z["candidate_bins"].copy(),z["pre_return_seen"].copy(),meta["metadata"])


class JunctionHighSampler:
    """Soft matched substitution of only the high actor's real-data queries."""

    def __init__(self,index,observations,probability=0.5,seed=0,min_visits=5,min_episodes=3):
        if not 0 <= probability <= 1 or min_visits < 1 or min_episodes < 1:
            raise ValueError("Invalid probability or independent-support thresholds.")
        if len(observations) != len(index.frame_cells):
            raise ValueError("Observations and junction index size mismatch.")
        self.index=index
        self.observations=np.asarray(observations)
        self.probability=float(probability)
        self.min_visits=int(min_visits)
        self.min_episodes=int(min_episodes)
        self.rng=np.random.default_rng(seed)

    def get_rng_state(self):
        return deepcopy(self.rng.bit_generator.state)

    def set_rng_state(self,state):
        self.rng.bit_generator.state=deepcopy(state)

    @lru_cache(maxsize=65536)
    def _supported_pool(self,junction,incoming,goal_cell,bin_id,outgoing,episode):
        index=self.index
        pool=index.pools.get((junction,incoming,goal_cell,bin_id))
        if pool is None:
            return np.empty(0,np.int32),0,0
        pool_visits=index.segment_visits[index.candidate_segments[pool]]
        accept=(index.events["first_exit"][pool_visits] != outgoing)&(index.events["episode"][pool_visits] != episode)
        candidates=pool[accept];candidate_visits=pool_visits[accept]
        # A visit has exactly one record for a particular goal-cell/time-bin.
        n_visits=len(np.unique(candidate_visits))
        n_episodes=len(np.unique(index.events["episode"][candidate_visits]))
        return candidates,n_visits,n_episodes

    def sample(self,start_indices,goal_indices,target_indices):
        start=np.asarray(start_indices,dtype=np.int64)
        goal=np.asarray(goal_indices,dtype=np.int64)
        targets=np.asarray(target_indices,dtype=np.int64)
        if start.ndim != 1 or goal.shape != start.shape or targets.ndim != 2 or targets.shape[0] != len(start):
            raise ValueError("Expected start/goal vectors and far-to-near target matrix.")
        M=targets.shape[1]
        if M not in (1,2,3):
            raise ValueError("First sampling study supports M in {1,2,3}, fixed 50-step spacing.")
        if np.any(start < 0) or np.any(start >= len(self.index.frame_cells)) or np.any(goal < 0) or np.any(goal >= len(self.index.frame_cells)):
            raise ValueError("Query index outside training dataset.")
        index=self.index
        B=len(start)
        output_start=start.copy();output_goal=goal.copy();output_targets=targets.copy()
        replaced=np.zeros(B,bool);triggered=np.zeros(B,bool);supported=np.zeros(B,bool)
        support_visits=np.zeros(B,np.int32);support_episodes=np.zeros(B,np.int32);donor_visits=np.full(B,-1,np.int32)
        visits=index.frame_visits[start];goalcells=index.frame_cells[goal]
        offset_bin=np.searchsorted(OFFSET_BIN_EDGES,goal-start,side="right")-1
        # Cheap vectorized checks leave only actual ambiguous queries for the
        # per-query donor lookup, rather than looping over all junction frames.
        active=np.flatnonzero(visits >= 0)
        if len(active):
            av=visits[active];event=index.events
            nearest=targets[active,-1]
            same_episode=index.frame_episodes[goal[active]] == event["episode"][av]
            original_chain=np.all(targets[active] == np.minimum(start[active,None]+50*np.arange(M,0,-1),goal[active,None]),axis=1)
            keep=(event["incoming"][av]>=0)&(event["first_exit"][av]>=0)&(event["return_index"][av]>=0)
            keep&=(goalcells[active]!=event["junction"][av])&same_episode&original_chain
            keep&=(nearest>=event["exit_index"][av])&(nearest<event["return_index"][av])&(goal[active]>=event["return_index"][av])
            keep&=~index.pre_return_seen[av,goalcells[active]]
            active=active[keep]
            triggered[active]=True
        for row in active:
            visit=int(visits[row]);event=index.events
            junction=int(event["junction"][visit]);incoming=int(event["incoming"][visit]);outgoing=int(event["first_exit"][visit]);ret=int(event["return_index"][visit]);exit_index=int(event["exit_index"][visit]);goal_cell=int(goalcells[row])
            candidates,n_visits,n_episodes=self._supported_pool(junction,incoming,goal_cell,int(offset_bin[row]),outgoing,int(event["episode"][visit]))
            support_visits[row]=n_visits;support_episodes[row]=n_episodes
            if n_visits < self.min_visits or n_episodes < self.min_episodes:
                continue
            supported[row]=True
            if self.probability == 0 or self.rng.random() >= self.probability:
                continue
            record=int(self.rng.choice(candidates));segment=int(index.candidate_segments[record]);donor=int(index.segment_visits[segment])
            possible_starts=index.donor_starts[donor];possible_goals=index.segment_goals[segment]
            low,high=OFFSET_BIN_EDGES[offset_bin[row]:offset_bin[row]+2]
            left=np.searchsorted(possible_goals,possible_starts.astype(np.int64)+low)
            right=np.searchsorted(possible_goals,possible_starts.astype(np.int64)+high)
            valid=np.flatnonzero(left < right)
            chosen_start=int(self.rng.choice(valid))
            new_start=int(possible_starts[chosen_start]);new_goal=int(possible_goals[self.rng.integers(left[chosen_start],right[chosen_start])])
            new_targets=np.minimum(new_start+50*np.arange(M,0,-1),new_goal)
            # Preserve full 50/100/150 clipping masks even when training M=1.
            old_clip=goal[row]-start[row] < np.array([50,100,150])
            new_clip=new_goal-new_start < np.array([50,100,150])
            if not np.array_equal(old_clip,new_clip):
                raise AssertionError("Temporal bin matching failed to preserve clipping.")
            if index.frame_cells[new_goal] != goal_cell or index.frame_cells[new_start] != junction:
                raise AssertionError("Matched donor moved the source or goal cell.")
            if new_targets[-1] < event["exit_index"][donor] or (event["return_index"][donor] >= 0 and new_goal >= event["return_index"][donor]):
                raise AssertionError("Donor lacks clean first-exit-to-goal evidence.")
            output_start[row]=new_start;output_goal[row]=new_goal;output_targets[row]=new_targets;donor_visits[row]=donor;replaced[row]=True
        ids=np.flatnonzero(replaced)
        metrics={"trigger_fraction":float(triggered.mean()) if B else 0.,"supported_fraction":float(supported.mean()) if B else 0.,
                 "replacement_fraction":float(replaced.mean()) if B else 0.,"trigger_count":float(triggered.sum()),
                 "supported_count":float(supported.sum()),"replacement_count":float(replaced.sum())}
        dx=np.linalg.norm(self.observations[output_start[ids],:2]-self.observations[start[ids],:2],axis=-1)
        full=np.linalg.norm(self.observations[output_start[ids]]-self.observations[start[ids]],axis=-1)
        # OGBench ant observation = complete qpos(15) + qvel(14).
        velocity_available=self.observations.shape[1] == 29
        dv=np.linalg.norm(self.observations[output_start[ids],15:17]-self.observations[start[ids],15:17],axis=-1) if velocity_available else np.empty(0)
        ddt=(output_goal[ids]-output_start[ids])-(goal[ids]-start[ids])
        for name,values in (("start_xy_drift",dx),("start_state_l2_drift",full),("start_xy_velocity_drift",dv),("goal_offset_drift",ddt)):
            metrics[name+"_mean"]=float(values.mean()) if len(values) else 0.
            metrics[name+"_abs_mean"]=float(np.abs(values).mean()) if len(values) else 0.
            metrics[name+"_max_abs"]=float(np.max(np.abs(values))) if len(values) else 0.
        metrics["velocity_diagnostic_available"]=float(velocity_available)
        return dict(start_indices=output_start.astype(np.int32),goal_indices=output_goal.astype(np.int32),target_indices=output_targets.astype(np.int32),
                    replaced=replaced,triggered=triggered,supported=supported,support_visits=support_visits,support_episodes=support_episodes,
                    donor_visit_ids=donor_visits,metrics=metrics)
