"""Passive junction diagnostics from already collected evaluation trajectories.

Geometry is inferred from training occupancy and is used only after evaluation.
There is no policy call, simulator call, shortest-route correctness label, or
additional random draw in this module.
"""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def _cell_key(cell):
    return ','.join(str(int(x)) for x in cell)


def _geometry(geometry):
    """Normalize the portable training-topology schema without a grid oracle."""
    if 'geometry' in geometry:
        geometry = geometry['geometry']
    width = float(geometry['cell_width'])
    half = float(geometry.get('core_half_width', width / 2 - 0.4))
    origin = np.asarray(geometry['origin_xy'], dtype=np.float64)
    if width <= 0 or not 0 < half < width / 2 or origin.shape != (2,):
        raise ValueError('Invalid lattice geometry or stable-core width.')
    free = {tuple(map(int, cell)) for cell in geometry['free_cells']}
    junctions = {tuple(map(int, cell)) for cell in geometry['junction_cells']}
    adjacency = {
        tuple(map(int, key.split(','))): {tuple(map(int, cell)) for cell in neighbors}
        for key, neighbors in geometry['adjacency'].items()
    }
    if not junctions <= free or set(adjacency) != free:
        raise ValueError('Junctions/adjacency do not match occupied training cells.')
    for cell, neighbors in adjacency.items():
        if not neighbors <= free or any(cell not in adjacency[n] for n in neighbors):
            raise ValueError('Training topology must have symmetric adjacency.')
    return width, half, origin, free, junctions, adjacency


def _cell_coordinates(points, width, origin):
    xy = np.asarray(points, dtype=np.float64)[..., :2]
    if xy.shape[-1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Expected finite XY coordinates.')
    col_row = np.floor((xy - origin) / width + 0.5).astype(np.int64)
    return col_row[..., ::-1], origin + col_row * width


def _union_length(intervals):
    """Number of observed transitions in a union of half-open intervals."""
    end = -1
    length = 0
    for lo, hi in sorted(intervals):
        length += max(0, hi - max(lo, end))
        end = max(end, hi)
    return int(length)


def summarize_episode(observations, goal, geometry, *, success=False,
                      terminated=False, truncated=False, episode_index=0):
    """Extract physical core visits from a T+1-state, already executed path.

    A physical visit is one run in the stable-core sequence. Border-only jitter
    is ignored. A nonadjacent next core is explicitly ambiguous; no later core
    is used to guess the first exit. Unknown occupied cells remain separators.
    """
    width, half, origin, free, junctions, adjacency = _geometry(geometry)
    observations = np.asarray(observations)
    if observations.ndim != 2 or len(observations) == 0:
        raise ValueError('Expected a nonempty sequence of full observations.')
    coordinates, centers = _cell_coordinates(observations, width, origin)
    known = np.array([tuple(cell) in free for cell in coordinates], dtype=bool)
    core = np.max(np.abs(observations[:, :2] - centers), axis=1) <= half
    goal_cell = tuple(map(int, _cell_coordinates(np.asarray(goal), width, origin)[0]))
    hits = np.flatnonzero(core)
    events = []
    if len(hits):
        hit_cells = coordinates[hits]
        changes = np.flatnonzero(np.r_[True, np.any(np.diff(hit_cells, axis=0), axis=1)])
        run_cells = [tuple(map(int, cell)) for cell in hit_cells[changes]]
        run_starts = hits[changes]
        run_lasts = hits[np.r_[changes[1:] - 1, len(hits) - 1]]
        next_same = np.full(len(run_cells), -1, dtype=np.int64)
        seen = {}
        for k in range(len(run_cells) - 1, -1, -1):
            cell = run_cells[k]
            next_same[k] = seen.get(cell, -1)
            seen[cell] = k
        visit_counts = defaultdict(int)
        for k, cell in enumerate(run_cells):
            if cell not in junctions:
                continue
            before = run_cells[k - 1] if k else None
            after = run_cells[k + 1] if k + 1 < len(run_cells) else None
            incoming = before if before in adjacency[cell] else None
            first_exit = after if after in adjacency[cell] else None
            entry = int(run_starts[k])
            exit_at = int(run_starts[k + 1]) if first_exit is not None else None
            return_at = int(run_starts[next_same[k]]) if next_same[k] >= 0 else None
            returned = return_at is not None
            incoming_status = ('known' if incoming is not None else
                               'episode_start' if before is None else 'nonadjacent_core')
            exit_status = ('known' if first_exit is not None else
                           'episode_end' if after is None else 'nonadjacent_core')
            if first_exit is None:
                outcome = 'no_exit_observed' if after is None else 'ambiguous_exit'
            elif returned:
                outcome = 'returned'
            elif success:
                outcome = 'no_return_before_success'
            elif truncated:
                outcome = 'no_return_observed_before_truncation'
            elif terminated:
                outcome = 'no_return_observed_before_termination'
            else:
                outcome = 'no_return_observed_before_record_end'
            events.append(dict(
                event_index=len(events), episode_index=int(episode_index),
                junction=list(cell), incoming=None if incoming is None else list(incoming),
                goal_cell=list(goal_cell), goal_cell_in_training_topology=goal_cell in free,
                first_exit=None if first_exit is None else list(first_exit),
                incoming_status=incoming_status, exit_status=exit_status,
                previous_stable_cell=None if before is None else list(before),
                next_stable_cell=None if after is None else list(after),
                entry_step=entry, last_core_step=int(run_lasts[k]),
                first_exit_step=exit_at, first_return_step=return_at,
                visit_index_at_junction=visit_counts[cell],
                first_visit_at_junction=visit_counts[cell] == 0,
                first_exit_latency_steps=None if exit_at is None else exit_at - entry,
                return_excursion_steps=(return_at - exit_at
                                        if exit_at is not None and returned else None),
                observed_steps_after_exit=(len(observations) - 1 - exit_at
                                           if exit_at is not None else None),
                returned=returned, outcome=outcome,
                exit_toward_incoming=(incoming == first_exit
                                      if incoming is not None and first_exit is not None else None),
            ))
            visit_counts[cell] += 1
    known_exit = [event for event in events if event['first_exit'] is not None]
    first_visits = [event for event in events if event['first_visit_at_junction']]
    first_known = [event for event in first_visits if event['first_exit'] is not None]
    intervals = [(event['first_exit_step'], event['first_return_step'])
                 for event in known_exit if event['returned']]
    return dict(
        episode_index=int(episode_index), steps=len(observations) - 1,
        success=bool(success), terminated=bool(terminated), truncated=bool(truncated),
        goal_cell=list(goal_cell), goal_cell_in_training_topology=goal_cell in free,
        unknown_topology_frames=int(np.sum(~known)), noncore_frames=int(np.sum(~core)),
        junction_visits=len(events), distinct_junctions=len({tuple(e['junction']) for e in events}),
        known_exit_visits=len(known_exit), returned_known_exit_visits=sum(e['returned'] for e in known_exit),
        first_junction_visits=len(first_visits), first_known_exit_visits=len(first_known),
        first_visit_returns=sum(e['returned'] for e in first_known),
        no_exit_observed_visits=sum(e['exit_status'] == 'episode_end' for e in events),
        ambiguous_exit_visits=sum(e['exit_status'] == 'nonadjacent_core' for e in events),
        union_return_excursion_steps=_union_length(intervals), events=events,
    )


def aggregate_episodes(episodes, *, first_visits_only=False):
    """Descriptive J x incoming x goal-cell groups; no route is called correct."""
    groups = defaultdict(list)
    success_by_episode = {ep['episode_index']: ep['success'] for ep in episodes}
    for episode in episodes:
        for event in episode['events']:
            if first_visits_only and not event['first_visit_at_junction']:
                continue
            key = (tuple(event['junction']),
                   None if event['incoming'] is None else tuple(event['incoming']),
                   tuple(event['goal_cell']))
            groups[key].append(event)
    rows = []
    for (junction, incoming, goal), events in sorted(groups.items(), key=lambda item: str(item[0])):
        episode_ids = sorted({event['episode_index'] for event in events})
        exited = [event for event in events if event['first_exit'] is not None]
        returned = [event for event in exited if event['returned']]
        counts = defaultdict(int)
        outcomes = defaultdict(int)
        for event in exited:
            counts[_cell_key(event['first_exit'])] += 1
        for event in events:
            outcomes[event['outcome']] += 1
        rows.append(dict(
            junction=list(junction), incoming=None if incoming is None else list(incoming),
            goal_cell=list(goal), visits=len(events), distinct_episodes=len(episode_ids),
            episode_indices=episode_ids, known_exit_visits=len(exited), returned_visits=len(returned),
            return_fraction_given_known_exit=len(returned) / len(exited) if exited else None,
            first_exit_counts=dict(counts), outcome_counts=dict(outcomes),
            mean_first_exit_latency_steps=(float(np.mean([e['first_exit_latency_steps'] for e in exited]))
                                           if exited else None),
            mean_return_excursion_steps=(float(np.mean([e['return_excursion_steps'] for e in returned]))
                                          if returned else None),
            episode_success_fraction=float(np.mean([success_by_episode[i] for i in episode_ids])),
        ))
    return rows


def _scalar_metrics(episodes):
    """Finite counts/rates for CSV; denominators expose undefined-rate cases."""
    known = sum(ep['known_exit_visits'] for ep in episodes)
    first_known = sum(ep['first_known_exit_visits'] for ep in episodes)
    returned = sum(ep['returned_known_exit_visits'] for ep in episodes)
    first_returns = sum(ep['first_visit_returns'] for ep in episodes)
    latencies = [event['first_exit_latency_steps'] for ep in episodes for event in ep['events']
                 if event['first_exit'] is not None]
    excursions = [event['return_excursion_steps'] for ep in episodes for event in ep['events']
                  if event['return_excursion_steps'] is not None]
    return dict(
        episodes=len(episodes), junction_visits=sum(ep['junction_visits'] for ep in episodes),
        known_exit_visits=known, returned_visits=returned,
        return_fraction_given_known_exit=returned / known if known else 0.0,
        return_fraction_defined=int(known > 0), first_known_exit_visits=first_known,
        first_visit_returns=first_returns,
        first_visit_return_fraction=first_returns / first_known if first_known else 0.0,
        first_visit_return_fraction_defined=int(first_known > 0),
        no_exit_observed_visits=sum(ep['no_exit_observed_visits'] for ep in episodes),
        ambiguous_exit_visits=sum(ep['ambiguous_exit_visits'] for ep in episodes),
        unknown_topology_frames=sum(ep['unknown_topology_frames'] for ep in episodes),
        mean_first_exit_latency_steps=float(np.mean(latencies)) if latencies else 0.0,
        mean_return_excursion_steps=float(np.mean(excursions)) if excursions else 0.0,
        returned_excursion_count=len(excursions),
        mean_union_return_excursion_steps=float(np.mean([ep['union_return_excursion_steps'] for ep in episodes])),
    )


def save_behavior_rollouts(output_prefix, trajectories, geometry, *, task_id, checkpoint_step):
    """Persist full executed transitions and goal metadata, return scalar metrics.

    Call immediately after an ordinary evaluate(..., record_trajectory_metadata=True).
    NPZ contains no pickled objects; the JSON contains raw event and grouping
    records, including censoring, so downstream episode bootstrap is possible.
    """
    if not trajectories:
        raise ValueError('Cannot persist an empty set of evaluation episodes.')
    if 'geometry' in geometry:
        geometry = geometry['geometry']
    _geometry(geometry)
    prefix = Path(output_prefix)
    npz_path = prefix.with_suffix('.npz')
    json_path = prefix.with_suffix('.json')
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f'Existing behavior records must not be overwritten: {prefix}')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    arrays = defaultdict(list)
    episodes = []
    for trajectory in trajectories:
        metadata = trajectory.get('_metadata')
        if metadata is None or metadata.get('goal') is None:
            raise ValueError('Trajectory lacks opt-in full goal/seed metadata.')
        observation = np.asarray(trajectory['observation'])
        next_observation = np.asarray(trajectory['next_observation'])
        steps = len(observation)
        if steps == 0 or next_observation.shape != observation.shape:
            raise ValueError('Invalid transition observation lengths.')
        if not np.array_equal(observation[1:], next_observation[:-1]):
            raise ValueError('Trajectory observations are not temporally contiguous.')
        if metadata['task_id'] != task_id:
            raise ValueError('Task ID differs from trajectory metadata.')
        path = np.concatenate([observation[:1], next_observation], axis=0)
        episode = summarize_episode(
            path, metadata['goal'], geometry, success=metadata['success'],
            terminated=metadata['terminated'], truncated=metadata['truncated'],
            episode_index=metadata['episode_index'],
        )
        episode['metadata'] = {
            key: value for key, value in metadata.items() if key != 'goal'
        }
        episode['transition_offset'] = offsets[-1]
        episodes.append(episode)
        offsets.append(offsets[-1] + steps)
        for name in ('observation', 'next_observation', 'action', 'reward', 'done'):
            values = np.asarray(trajectory[name])
            if len(values) != steps or values.dtype.hasobject:
                raise ValueError(f'Invalid numeric transition field: {name}')
            arrays[name].append(values)
        arrays['goal'].append(np.asarray(metadata['goal']))
    if len({ep['episode_index'] for ep in episodes}) != len(episodes):
        raise ValueError('Episode indices must be unique within a task recording.')
    payload = {key: np.concatenate(value, axis=0) for key, value in arrays.items() if key != 'goal'}
    payload['goal'] = np.stack(arrays['goal'])
    payload['episode_offsets'] = np.asarray(offsets, dtype=np.int64)
    payload['episode_indices'] = np.asarray([ep['episode_index'] for ep in episodes], dtype=np.int64)
    metrics = _scalar_metrics(episodes)
    visited = {tuple(event['junction']) for ep in episodes for event in ep['events']}
    report = dict(
        protocol='passive-native-evaluation-junction-behavior-v1',
        task_id=int(task_id), checkpoint_step=int(checkpoint_step), trajectory_file=npz_path.name,
        geometry=geometry, metrics=metrics, episodes=episodes,
        junction_coverage=[dict(junction=list(cell), visited=cell in visited)
                           for cell in sorted(map(tuple, geometry['junction_cells']))],
        groups_all_visits=aggregate_episodes(episodes),
        groups_first_visit_per_junction=aggregate_episodes(episodes, first_visits_only=True),
        interpretation=[
            'Only trajectories from the native evaluation are observed; policy and evaluation RNG are unchanged.',
            'Topology is derived from training occupancy, with stable core entries used to reject border jitter.',
            'A return is an observed event, not proof that an exit was wrong or the return was unnecessary.',
            'No-return-until-timeout is right censored and differs from reaching the goal without returning.',
            'First-visit groups separate initial decisions from subsequent corrective visits.',
            'Per-visit excursion intervals may overlap; episode union length avoids counting a frame twice.',
            'Excursion duration is descriptive elapsed time, not a counterfactual estimate of avoidable cost.',
            'Group return rates are descriptive and conditioned on actually visited contexts; route coverage may differ.',
            'Scalar rates with zero denominators use zero plus an explicit defined flag; JSON group rates use null.',
        ],
    )
    # Exclusive creation prevents quietly replacing an earlier evaluation.
    with npz_path.open('xb') as stream:
        np.savez_compressed(stream, **payload)
    with json_path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return metrics
