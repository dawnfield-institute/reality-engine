"""
Experiment 08: Deep Law Detection

Comprehensive search for emergent laws from full trace data.
Looking for any consistent mathematical relationships.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
from scipy.fft import fft, fftfreq
import math
import glob
import os

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
SQRT5 = math.sqrt(5)
XI = 1 + math.pi / 55


def load_latest_trace():
    """Load most recent trace file."""
    traces = glob.glob("results/full_trace_*.json")
    if not traces:
        print("No trace files found!")
        return None
    latest = max(traces, key=os.path.getctime)
    print("Loading: %s" % latest)
    with open(latest, 'r') as f:
        return json.load(f)


def extract_time_series(trace):
    """Extract all time series from trace."""
    series = {
        'step': [],
        'node_count': [],
        'total_value': [],
        'max_value': [],
        'min_value': [],
        'mean_value': [],
        'std_value': [],
        'value_range': [],
        'gini': [],  # inequality measure
        'splits': [],
        'deaths': [],
        'net_change': [],
    }
    
    for s in trace['steps']:
        values = [n['value'] for n in s['nodes']]
        
        series['step'].append(s['step'])
        series['node_count'].append(s['node_count'])
        series['total_value'].append(s['total_value'])
        series['max_value'].append(max(values) if values else 0)
        series['min_value'].append(min(values) if values else 0)
        series['mean_value'].append(np.mean(values) if values else 0)
        series['std_value'].append(np.std(values) if values else 0)
        series['value_range'].append(max(values) - min(values) if values else 0)
        
        # Gini coefficient (inequality)
        if len(values) > 1:
            sorted_v = np.sort(values)
            n = len(sorted_v)
            cumsum = np.cumsum(sorted_v)
            gini = (2 * np.sum((np.arange(1, n+1) * sorted_v))) / (n * np.sum(sorted_v)) - (n + 1) / n
        else:
            gini = 0
        series['gini'].append(gini)
        
        series['splits'].append(1 if s['split_info'] else 0)
        series['deaths'].append(len(s['destroyed']))
        series['net_change'].append(len(s['created']) - len(s['destroyed']))
    
    return {k: np.array(v) for k, v in series.items()}


def find_periodicities(series, name, max_period=200):
    """Find periodic patterns via autocorrelation and FFT."""
    print("\n--- Periodicity in %s ---" % name)
    
    x = series - np.mean(series)
    if np.std(x) == 0:
        print("  Constant series, no periodicity")
        return []
    
    # Autocorrelation
    n = len(x)
    autocorr = np.correlate(x, x, mode='full')[n-1:]
    autocorr = autocorr / autocorr[0]
    
    # Find peaks in autocorrelation
    peaks = []
    for i in range(2, min(max_period, len(autocorr) - 1)):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            if autocorr[i] > 0.1:  # Significant
                peaks.append((i, autocorr[i]))
    
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    if peaks:
        print("  Top periodicities (autocorrelation peaks):")
        for lag, corr in peaks[:5]:
            # Check if near 55 or its multiples/divisors
            near_55 = ""
            if abs(lag - 55) <= 2:
                near_55 = " ← NEAR 55!"
            elif abs(lag - 110) <= 2:
                near_55 = " ← NEAR 2×55!"
            elif abs(lag - 27) <= 1 or abs(lag - 28) <= 1:
                near_55 = " ← NEAR 55/2!"
            print("    lag=%3d: r=%.3f%s" % (lag, corr, near_55))
    else:
        print("  No significant periodicities found")
    
    # FFT for dominant frequencies
    freqs = fftfreq(n)
    spectrum = np.abs(fft(x))
    
    # Find peak frequencies (excluding DC)
    peak_indices = np.argsort(spectrum[1:n//2])[-5:] + 1
    print("  Top FFT frequencies:")
    for idx in reversed(peak_indices):
        freq = freqs[idx]
        period = 1 / freq if freq != 0 else float('inf')
        power = spectrum[idx]
        near = ""
        if abs(period - 55) < 3:
            near = " ← PERIOD ~55!"
        print("    period=%.1f steps (power=%.1f)%s" % (period, power, near))
    
    return peaks


def find_scaling_laws(series):
    """Look for power law and exponential relationships."""
    print("\n" + "=" * 70)
    print("SCALING LAWS")
    print("=" * 70)
    
    step = series['step'][1:]  # Skip step 0
    
    tests = [
        ('node_count', series['node_count'][1:]),
        ('mean_value', series['mean_value'][1:]),
        ('std_value', series['std_value'][1:]),
        ('gini', series['gini'][1:]),
    ]
    
    for name, y in tests:
        if np.any(y <= 0):
            continue
            
        print("\n--- %s vs step ---" % name)
        
        # Linear: y = a*t + b
        slope, intercept, r_lin, _, _ = stats.linregress(step, y)
        print("  Linear: y = %.4f*t + %.4f (r²=%.4f)" % (slope, intercept, r_lin**2))
        
        # Power law: y = a * t^b  →  log(y) = log(a) + b*log(t)
        log_t = np.log(step)
        log_y = np.log(y)
        b, log_a, r_pow, _, _ = stats.linregress(log_t, log_y)
        print("  Power law: y = %.4f * t^%.4f (r²=%.4f)" % (np.exp(log_a), b, r_pow**2))
        
        # Check for special exponents
        if abs(b - 1.0) < 0.1:
            print("    → Exponent ~1 (linear growth)")
        elif abs(b - 0.5) < 0.1:
            print("    → Exponent ~0.5 (square root growth)")
        elif abs(b - (-1)) < 0.1:
            print("    → Exponent ~-1 (inverse/hyperbolic)")
        elif abs(b - PHI_INV) < 0.1:
            print("    → Exponent ~1/φ = 0.618!")
        
        # Exponential: y = a * e^(b*t)  →  log(y) = log(a) + b*t
        b_exp, log_a_exp, r_exp, _, _ = stats.linregress(step, log_y)
        print("  Exponential: y = %.4f * e^(%.6f*t) (r²=%.4f)" % (np.exp(log_a_exp), b_exp, r_exp**2))


def find_ratio_laws(trace):
    """Look for consistent ratios in splits."""
    print("\n" + "=" * 70)
    print("RATIO LAWS")
    print("=" * 70)
    
    # Collect all split ratios with context
    splits = []
    for s in trace['steps']:
        if s['split_info']:
            info = s['split_info']
            splits.append({
                'step': s['step'],
                'parent_value': info['parent_value'],
                'child_values': info['child_values'],
                'ratio': info['ratio'],
                'sibling_ratio': info['child_values'][0] / info['child_values'][1] if info['child_values'][1] > 0 else float('inf'),
            })
    
    if not splits:
        print("No splits found!")
        return
    
    print("\nTotal splits: %d" % len(splits))
    
    # Ratio distribution
    ratios = np.array([s['ratio'] for s in splits])
    sib_ratios = np.array([s['sibling_ratio'] for s in splits])
    sib_ratios = np.minimum(sib_ratios, 1/sib_ratios)  # Normalize to [0, 1]
    
    # Test against special values
    special_values = {
        '1/2': 0.5,
        '1/φ': PHI_INV,
        '1/φ²': PHI_INV**2,
        '1/3': 1/3,
        '2/5': 0.4,
        '1/e': 1/math.e,
        '1/π': 1/math.pi,
        '1/√5': 1/SQRT5,
    }
    
    print("\nRatio clustering near special values:")
    for name, val in sorted(special_values.items(), key=lambda x: x[1]):
        near = np.sum(np.abs(ratios - val) < 0.03) + np.sum(np.abs(ratios - (1-val)) < 0.03)
        pct = 100 * near / len(ratios)
        bar = '█' * int(pct)
        print("  %6s (%.3f): %5.1f%% %s" % (name, val, pct, bar))
    
    # Does ratio depend on parent value?
    parent_vals = np.array([s['parent_value'] for s in splits])
    corr, p_val = stats.pearsonr(parent_vals, ratios)
    print("\nRatio vs parent value: r=%.3f (p=%.4f)" % (corr, p_val))
    if p_val < 0.05:
        print("  → SIGNIFICANT: Larger parents split differently!")
    
    # Does ratio depend on step (time)?
    steps = np.array([s['step'] for s in splits])
    corr_t, p_val_t = stats.pearsonr(steps, ratios)
    print("Ratio vs time: r=%.3f (p=%.4f)" % (corr_t, p_val_t))
    if p_val_t < 0.05:
        print("  → SIGNIFICANT: Ratios evolve over time!")
    
    # Sibling ratio distribution
    print("\nSibling ratios (smaller/larger child):")
    print("  Mean: %.4f" % np.mean(sib_ratios))
    print("  Median: %.4f" % np.median(sib_ratios))
    
    # Test if sibling ratios cluster near 1/φ
    near_phi_sib = np.sum(np.abs(sib_ratios - PHI_INV) < 0.05)
    print("  Near 1/φ (±0.05): %.1f%%" % (100 * near_phi_sib / len(sib_ratios)))


def find_conservation_laws(trace, series):
    """Look for conserved quantities beyond total value."""
    print("\n" + "=" * 70)
    print("CONSERVATION LAWS")
    print("=" * 70)
    
    # Total value conservation (we know this one)
    total = series['total_value']
    print("\n1. Total value: min=%.10f, max=%.10f" % (total.min(), total.max()))
    print("   Conservation error: %.2e" % (total.max() - total.min()))
    
    # Node count × mean value
    product = series['node_count'] * series['mean_value']
    print("\n2. N × mean_value: min=%.6f, max=%.6f" % (product.min(), product.max()))
    print("   Variation: %.2e (should equal total value)" % (product.max() - product.min()))
    
    # Entropy-like quantities
    print("\n3. Entropy-like quantities:")
    
    for s in [trace['steps'][0], trace['steps'][len(trace['steps'])//2], trace['steps'][-1]]:
        values = np.array([n['value'] for n in s['nodes']])
        values = values[values > 0]  # Filter zeros
        if len(values) > 0:
            probs = values / values.sum()
            entropy = -np.sum(probs * np.log(probs))
            print("   Step %3d: H=%.4f (N=%d nodes)" % (s['step'], entropy, len(values)))
    
    # Look for other invariants
    print("\n4. Testing potential invariants:")
    
    # Variance × N
    var_n = series['std_value']**2 * series['node_count']
    print("   Var × N: range [%.6f, %.6f]" % (var_n.min(), var_n.max()))
    
    # Gini × N
    gini_n = series['gini'] * series['node_count']
    print("   Gini × N: range [%.4f, %.4f]" % (gini_n.min(), gini_n.max()))


def find_neighbor_laws(trace):
    """Analyze connectivity patterns."""
    print("\n" + "=" * 70)
    print("CONNECTIVITY LAWS")
    print("=" * 70)
    
    # Degree distribution over time
    degree_history = []
    
    for s in trace['steps'][::10]:  # Sample every 10 steps
        degrees = [len(n['neighbors']) for n in s['nodes']]
        degree_history.append({
            'step': s['step'],
            'mean': np.mean(degrees),
            'max': max(degrees),
            'connected': sum(1 for d in degrees if d > 0),
            'isolated': sum(1 for d in degrees if d == 0),
        })
    
    # Does mean degree change?
    steps = [d['step'] for d in degree_history]
    means = [d['mean'] for d in degree_history]
    
    print("\nDegree evolution:")
    print("  Start: mean=%.3f, max=%d" % (degree_history[0]['mean'], degree_history[0]['max']))
    print("  End: mean=%.3f, max=%d" % (degree_history[-1]['mean'], degree_history[-1]['max']))
    
    if len(steps) > 2:
        slope, _, r, _, _ = stats.linregress(steps, means)
        print("  Trend: slope=%.6f (r²=%.4f)" % (slope, r**2))
    
    # Clustering by value - do similar-valued nodes connect?
    final = trace['steps'][-1]
    connected_pairs = []
    for n in final['nodes']:
        for nb_id in n['neighbors']:
            nb = next((x for x in final['nodes'] if x['id'] == nb_id), None)
            if nb:
                connected_pairs.append((n['value'], nb['value']))
    
    if connected_pairs:
        v1, v2 = zip(*connected_pairs)
        corr, p = stats.pearsonr(v1, v2)
        print("\nValue correlation in connected pairs: r=%.3f (p=%.4f)" % (corr, p))
        if p < 0.05 and corr > 0.3:
            print("  → SIGNIFICANT: Similar values attract!")
        elif p < 0.05 and corr < -0.3:
            print("  → SIGNIFICANT: Opposite values attract!")


def find_55_signatures(series, trace):
    """Specifically hunt for 55-related patterns."""
    print("\n" + "=" * 70)
    print("55-SIGNATURE DETECTION")
    print("=" * 70)
    
    n_steps = len(series['step'])
    
    # Test multiple series
    test_series = ['node_count', 'mean_value', 'gini', 'std_value']
    
    for name in test_series:
        x = series[name]
        if len(x) < 110:
            continue
        
        # Compute autocorrelation at specific lags
        x_norm = x - np.mean(x)
        var = np.var(x)
        
        lags = [11, 22, 27, 28, 55, 89, 110, 144]
        print("\n%s autocorrelation:" % name)
        for lag in lags:
            if lag >= len(x):
                continue
            acf = np.mean(x_norm[:-lag] * x_norm[lag:]) / var
            mark = ""
            if lag == 55:
                mark = " ← F_10"
            elif lag == 89:
                mark = " ← F_11"
            elif lag == 144:
                mark = " ← F_12"
            elif lag == 28:
                mark = " ← 55/2"
            print("  lag %3d: %.4f%s" % (lag, acf, mark))
    
    # Check if events cluster at 55-step intervals
    split_steps = [s['step'] for s in trace['steps'] if s['split_info']]
    if split_steps:
        # Modular analysis - do splits prefer certain positions mod 55?
        mod_55 = [s % 55 for s in split_steps]
        mod_counts = Counter(mod_55)
        
        # Chi-square test for uniformity
        expected = len(split_steps) / 55
        observed = [mod_counts.get(i, 0) for i in range(55)]
        chi2, p = stats.chisquare(observed)
        
        print("\nSplits mod 55 uniformity test:")
        print("  Chi² = %.2f, p = %.4f" % (chi2, p))
        if p < 0.05:
            print("  → NON-UNIFORM: Splits prefer certain phases!")
            top_phases = sorted(mod_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print("  Preferred phases: %s" % [(p, c) for p, c in top_phases])


def main():
    trace = load_latest_trace()
    if not trace:
        return
    
    print("\nExtracting time series...")
    series = extract_time_series(trace)
    
    print("\n" + "=" * 70)
    print("PERIODICITY ANALYSIS")
    print("=" * 70)
    
    for name in ['node_count', 'mean_value', 'gini']:
        find_periodicities(series[name], name)
    
    find_scaling_laws(series)
    find_ratio_laws(trace)
    find_conservation_laws(trace, series)
    find_neighbor_laws(trace)
    find_55_signatures(series, trace)
    
    print("\n" + "=" * 70)
    print("LAW DETECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
