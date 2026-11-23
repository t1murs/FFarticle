"""Freeze the post-diagnostic design, then generate the declared parameter points.

This script generates design coordinates only, never virtual-firm outcomes.
The original mathematical specification and parent random-stream keys remain
unchanged. Registered parameter interventions use those parent keys.
"""
from __future__ import annotations
from datetime import datetime, timezone
from itertools import product, combinations
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import scipy
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference_verification/src'))
from family_ai.configuration import (scalar_parameters, resolve_mode,
    canonical_cell_id, canonical_parameter_text, validate_parameters)

MODES = ['A0', 'A1', 'Aplus', 'B', 'Bplus', 'C', 'D']
RULES = ['constant_eta', 'reliance_weighted_eta']
PARAM_FIELDS = ['kappa_0', 'q_0', 'alpha', 'beta', 'eta', 'phi', 'gamma',
                'lambda_q', 'a', 'chi', 'c_V', 'q_V', 'q_H', 'c', 'd', 'w_0']
CONTEXTS = {
    'chi_zero': {'chi': 0.0},
    'reliability_frozen': {'lambda_q': 0.0},
    'chi_zero_reliability_frozen': {'chi': 0.0, 'lambda_q': 0.0},
    'gamma_zero': {'gamma': 0.0},
    'chi_zero_gamma_zero': {'chi': 0.0, 'gamma': 0.0},
    'no_shock': {'d': 0.0},
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def cid(context, interface, L, rule):
    return f'{context}|{interface}|L{L}|{rule}'

def make_conditions(spec, params, family):
    logical = []
    physical = []
    signature_to_id = {}

    def add(context, interface, L, rule, overrides=None):
        overrides = dict(overrides or {})
        p = {**params, **overrides}
        if interface.startswith('F'):
            mask = int(interface[1:], 2)
            chosen = {
                'r_AI': params['r_AI'] if mask & 4 else params['r_reference'],
                'c_AI': params['c_AI'] if mask & 2 else params['c_reference'],
                'epsilon_AI': params['epsilon_AI'] if mask & 1 else 0.0,
            }
            overrides.update(chosen)
            p.update(chosen)
            mode = resolve_mode(spec, 'C', p)
            mode['code'] = interface
        else:
            mode = resolve_mode(spec, interface, p)
        validate_parameters(p)
        # Alias only established whole-trajectory identities. Mode names are
        # excluded; all operative model and numeric interface inputs remain.
        norm_L = 0 if p['a'] == 0 or mode['r'] == 0 else L
        norm_rule = RULES[0] if mode['r'] == 0 else rule
        signature = tuple(float(p[k]) for k in PARAM_FIELDS) + tuple(
            float(mode[k]) for k in ['r', 'c_s', 'epsilon', 'v']) + (norm_L, norm_rule)
        condition = {
            'condition_id': cid(context, interface, L, rule),
            'block': context,
            'interface': interface,
            'mode': mode,
            'L': L,
            'learning_rule': rule,
            'parameters': overrides,
            'variant_parameter_id': str(canonical_cell_id(p)),
            'variant_parameter_sha256': hashlib.sha256(canonical_parameter_text(p).encode()).hexdigest(),
        }
        if signature not in signature_to_id:
            signature_to_id[signature] = len(physical)
            physical.append({**condition, 'physical_id': len(physical)})
        condition['physical_id'] = signature_to_id[signature]
        logical.append(condition)

    for mode, L, rule in product(MODES, [0, 1], RULES):
        add('base', mode, L, rule)
    if family != 'focused':
        for mask, L, rule in product(range(8), [0, 1], RULES):
            add('base', f'F{mask:03b}', L, rule)
        for context, overrides in CONTEXTS.items():
            interfaces = MODES + ['F100'] if context == 'no_shock' else ['B', 'F100']
            for interface, L, rule in product(interfaces, [0, 1], RULES):
                add(context, interface, L, rule, overrides)
    return logical, physical

def make_contrasts(conditions, family):
    lookup = {c['condition_id']: c for c in conditions}
    contrasts = []

    def add(name, group, terms, **metadata):
        combined = {}
        for key, weight in terms:
            assert key in lookup, key
            combined[key] = combined.get(key, 0) + weight
        contrasts.append({
            'contrast_id': name, 'family': group,
            'terms': [{'condition_id': k, 'weight': v} for k, v in combined.items() if v],
            'metadata': metadata, 'precision_primary': True,
        })

    def pair(context, left, right, L, rule, group='core_mode'):
        add(f'{context}|{left}-{right}|L{L}|{rule}', group,
            [(cid(context, left, L, rule), 1), (cid(context, right, L, rule), -1)],
            context=context, left=left, right=right, L=L, learning_rule=rule)

    core_pairs = [('C', 'B'), ('D', 'C'), ('D', 'Bplus'), ('A1', 'A0'),
                  ('Aplus', 'A1'), ('Bplus', 'B')]
    contexts = ['base'] if family == 'focused' else ['base', 'no_shock']
    for context in contexts:
        for (left, right), L, rule in product(core_pairs, [0, 1], RULES):
            pair(context, left, right, L, rule)
        for mode, rule in product(MODES[1:], RULES):
            add(f'{context}|{mode}|L1-L0|{rule}', 'source',
                [(cid(context, mode, 1, rule), 1), (cid(context, mode, 0, rule), -1)],
                context=context, interface=mode, learning_rule=rule)
        for mode, L in product(MODES, [0, 1]):
            add(f'{context}|{mode}|L{L}|constant-weighted', 'learning_rule',
                [(cid(context, mode, L, RULES[0]), 1), (cid(context, mode, L, RULES[1]), -1)],
                context=context, interface=mode, L=L)

    if family == 'focused':
        return contrasts

    # Anchored factorial contrasts. First-order effects hold other components
    # at the reference; higher-order terms expose nonadditivity explicitly.
    for subset in range(1, 8):
        included = [j for j in [4, 2, 1] if subset & j]
        label = '+'.join({4: 'access', 2: 'cost', 1: 'distortion'}[j] for j in included)
        for L, rule in product([0, 1], RULES):
            terms = []
            for mask in range(8):
                if mask & ~subset:
                    continue
                sign = (-1) ** (len(included) - mask.bit_count())
                terms.append((cid('base', f'F{mask:03b}', L, rule), sign))
            add(f'factorial|{label}|L{L}|{rule}', 'component' if len(included) == 1 else 'component_interaction',
                terms, components=included, component_label=label, L=L, learning_rule=rule,
                interpretation='Anchored finite difference, not empirical causal mediation')

    for context in ['base'] + list(CONTEXTS):
        for L, rule in product([0, 1], RULES):
            pair(context, 'F100', 'B', L, rule, 'pure_access')
        for interface, rule in product(['B', 'F100'], RULES):
            # B at base/no_shock is already included in the core source set.
            if interface == 'B' and context in ['base', 'no_shock']:
                continue
            add(f'{context}|{interface}|L1-L0|{rule}', 'source',
                [(cid(context, interface, 1, rule), 1), (cid(context, interface, 0, rule), -1)],
                context=context, interface=interface, learning_rule=rule)
        for rule in RULES:
            terms = [(cid(context, 'F100', 1, rule), 1),
                     (cid(context, 'B', 1, rule), -1),
                     (cid(context, 'F100', 0, rule), -1),
                     (cid(context, 'B', 0, rule), 1)]
            add(f'{context}|Psi_access_source|{rule}', 'principal_interaction', terms,
                context=context, learning_rule=rule,
                interpretation='Pure-access effect under predecessor attribution minus neutral attribution')
        if context != 'base':
            for interface, L, rule in product(['B', 'F100'], [0, 1], RULES):
                add(f'{context}-base|{interface}|L{L}|{rule}', 'mechanism_level',
                    [(cid(context, interface, L, rule), 1), (cid('base', interface, L, rule), -1)],
                    context=context, interface=interface, L=L, learning_rule=rule)
            for rule in RULES:
                terms = []
                for ctx, multiplier in [(context, 1), ('base', -1)]:
                    terms += [(cid(ctx, 'F100', 1, rule), multiplier),
                              (cid(ctx, 'B', 1, rule), -multiplier),
                              (cid(ctx, 'F100', 0, rule), -multiplier),
                              (cid(ctx, 'B', 0, rule), multiplier)]
                add(f'{context}-base|Psi_access_source|{rule}', 'mechanism_interaction', terms,
                    context=context, learning_rule=rule)
    assert len({x['contrast_id'] for x in contrasts}) == len(contrasts)
    return contrasts

def main():
    spec = json.loads((ROOT / 'config/model_spec_stage2_frozen.json').read_text())
    gate = json.loads((ROOT / 'audit/science_preproduction_gate.json').read_text())
    if not gate.get('scientific_gate_passed'):
        raise RuntimeError('Scientific gate has not passed')
    target = ROOT / 'config/experiment_manifest.json'
    if target.exists():
        raise RuntimeError('Design already exists; version any change rather than redraw')
    settings = {
        'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'stage3a_diagnostic_results_seen': True,
        'variance_results_seen': False, 'production_results_seen': False,
        'master_seed': 20261121,
        'design_ids': {'focused': 2, 'joint_original': 3, 'joint_concordant': 4, 'joint_discordant': 5},
        'joint_dimension_order': list(spec['joint_design']['bounds']),
        'sobol': {'d': 19, 'scramble': True, 'bits': 30, 'power': 8,
                  'rng': 'Generator(PCG64(SeedSequence(entropy)))',
                  'entropy': [20261121, 31, 3, 0, 0, 0, 0, 0, 0],
                  'numpy': np.__version__, 'scipy': scipy.__version__},
        'dependence': {'columns_permuted': {'a': 'kappa_0', 'c_AI': 'r_AI'},
                       'tie_rule': 'Stable lexsort by anchor then recorded namespace32 uniform priority',
                       'seed_entropy': '[20261121,32,design_id,0,0,0,0,0,0]',
                       'marginal_values_preserved_exactly': True},
        'pairing': 'Parent canonical cell_id and design_id key every variant; variant parameter identities remain separate.',
        'focused_scope': 'All108basecells and all28logical conditions preserved.',
        'joint_scope': 'All256points in each of3dependence designs. Core28conditions; full2^3unchecked interface factorial; paired mechanism contexts.',
        'contexts': CONTEXTS,
        'primary_measures': 'Frozen primary measures plus prospectively promoted H_post phase means; full paired interactions included in E17.',
        'precision': spec['precision_plan'],
        'scientific_gate_sha256': sha(ROOT / 'audit/science_preproduction_gate.json'),
        'aliases': 'Resolved numeric mode identity; a=0 source-label identity; A0 source/rule identity. No averaging or lossy rounding.',
        'storage': 'All five scientific firm-round fields remain available through an exact indexed lossless representation; physical conditions map to every logical condition. Auxiliary q/rate/count and domain traces regenerate from recorded keys.',
        'negative_control_aliases': {
            'a_zero': 'L0 removes the source term; exact separate a0 tests retained. No extra independent population is generated.',
            'zero_coverage_or_access': 'Unaided A0 behavior for scientific outcomes, with exact separate tests; does not assert all inactive intermediate fields equal across uptake rules.'},
    }
    write_json(ROOT / 'config/design_generation_freeze.json', settings)
    # The settings file above is written before any Sobol/dependence draws.
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(settings['sobol']['entropy'])))
    sobol = qmc.Sobol(d=19, scramble=True, bits=30, rng=rng)
    u = sobol.random_base2(m=8)
    names = list(spec['joint_design']['bounds'])
    lower = np.array([spec['joint_design']['bounds'][k][0] for k in names])
    upper = np.array([spec['joint_design']['bounds'][k][1] for k in names])
    values = lower + u * (upper - lower)
    np.savez_compressed(ROOT / 'config/joint_design_coordinates.npz', unit=u, original=values)
    base_rows = []
    grid = spec['focused_design']['grid']
    for vals in product(*grid.values()):
        p = scalar_parameters(spec, dict(zip(grid, vals)))
        base_rows.append(('focused', 2, len(base_rows) + 1, p))
    dependence_records = {}
    for family, design_id, direction in [('joint_original', 3, 0), ('joint_concordant', 4, 1), ('joint_discordant', 5, -1)]:
        x = values.copy()
        if direction:
            tie_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261121, 32, design_id, 0, 0, 0, 0, 0, 0])))
            priorities = tie_rng.random((2, 256))
            maps = {}
            for j, (target_name, anchor_name) in enumerate([('a', 'kappa_0'), ('c_AI', 'r_AI')]):
                target_col, anchor_col = names.index(target_name), names.index(anchor_name)
                order = np.lexsort((priorities[j], x[:, anchor_col]))
                ranked = np.sort(values[:, target_col])
                if direction < 0:
                    ranked = ranked[::-1]
                x[order, target_col] = ranked
                maps[target_name] = order.tolist()
            dependence_records[family] = {'priorities': priorities.tolist(), 'rank_maps': maps}
        for index, point in enumerate(x):
            p = scalar_parameters(spec, dict(zip(names, map(float, point))))
            base_rows.append((family, design_id, index + 1, p))
    write_json(ROOT / 'config/dependence_rank_record.json', dependence_records)
    cells = []
    for index, (family, design_id, family_index, params) in enumerate(base_rows, start=1):
        parent_id = canonical_cell_id(params)
        conditions, unique = make_conditions(spec, params, family)
        contrasts = make_contrasts(conditions, family)
        cells.append({'base_index': index, 'family': family, 'family_index': family_index,
                      'design_id': design_id, 'parent_cell_id': parent_id,
                      'cell_key': f'd{design_id}_c_{parent_id:016x}',
                      'parameters': params, 'canonical_parameters': canonical_parameter_text(params),
                      'conditions': conditions, 'unique_conditions': unique, 'contrasts': contrasts})
    assert len(cells) == 876
    manifest = {'version': 'Stage3B post-diagnostic pre-production analysis design v1.0',
                'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'specification_sha256': sha(ROOT / 'config/model_spec_stage2_frozen.json'),
                'generation_freeze_sha256': sha(ROOT / 'config/design_generation_freeze.json'),
                'data_origin': 'SCENARIO-ASSUMED design; no firm outcomes generated by this script',
                'base_cells': cells,
                'counts': {'base_cells': len(cells),
                           'logical_conditions': sum(len(c['conditions']) for c in cells),
                           'physical_conditions': sum(len(c['unique_conditions']) for c in cells),
                           'registered_contrasts': sum(len(c['contrasts']) for c in cells)}}
    write_json(target, manifest)
    record = {'status': 'GENERATED_PENDING_INDEPENDENT_AUDIT',
              'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': sha(target), 'counts': manifest['counts'],
              'design_generation_code_sha256': sha(__file__),
              'variance_or_production_runs': 0}
    write_json(ROOT / 'audit/design_generation.json', record)
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
