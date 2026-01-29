# Issues Found and Fixed During Manuscript Refinement

## Critical Issues

### 1. Normalization Method Mismatch
**Severity**: High  
**Location**: Section 3.2 (Data Preprocessing and Feature Extraction)  
**Issue**: Manuscript described only min-max normalization, but implementation uses StandardScaler (z-score) followed by min-max scaling.  
**Impact**: Incorrect methodology description could mislead readers about preprocessing steps.  
**Fix**: Updated to accurately describe two-stage normalization: z-score standardization followed by min-max scaling to [0,1].  
**Verification**: Confirmed against `src/agn_general/data_loader.py` lines 154-164.

### 2. Novelty Definition Misrepresentation
**Severity**: Critical  
**Location**: Multiple sections (Abstract, Section 5.3, Discussion, Conclusion)  
**Issue**: 75% novelty rate presented as a performance outcome, but it's threshold-induced: nodes above 25th percentile are classified as novel, meaning exactly 75% are novel by construction.  
**Impact**: Misrepresents the scientific contribution by presenting a definitional property as a performance metric.  
**Fix**: 
- Updated evaluation protocol to explicitly state threshold-based definition
- Rewrote novelty analysis to emphasize distributional separation (mean distances: 0.0269-0.3707) rather than percentage
- Updated Table 2 to include mean distances
- Revised all mentions to focus on distributional properties
**Verification**: Confirmed against `src/agn_general/evaluation.py` line 202: `'is_novel': (min_distances > np.percentile(min_distances, 25))`

### 3. Missing Figure Integration
**Severity**: Medium  
**Location**: Results section  
**Issue**: `network_comparison.png` and `metrics_comparison.png` files exist in `results/plots/` but were not referenced in manuscript.  
**Impact**: Missing visualizations that support claims about network structure preservation.  
**Fix**: Added Figure 1 (network comparison) and Figure 2 (metrics comparison) with proper captions and cross-references. Renumbered subsequent figures.  
**Verification**: Confirmed all PNG files exist: `karate_network_comparison.png`, `facebook_network_comparison.png`, `email_network_comparison.png`, `karate_metrics_comparison.png`, `facebook_metrics_comparison.png`, `email_metrics_comparison.png`.

## Moderate Issues

### 4. Placeholder Formatting
**Severity**: Low  
**Location**: Reproducibility section, Acknowledgments  
**Issue**: Placeholders used brackets `[...]` instead of journal-style `\texttt{}` formatting.  
**Impact**: Inconsistent with journal formatting standards.  
**Fix**: Updated to `\texttt{[repository URL withheld for review]}` and `\texttt{[funding information withheld for review]}`.  
**Verification**: Applied consistently across all placeholder locations.

### 5. Inconsistent Novelty Reporting
**Severity**: Medium  
**Location**: Abstract, Conclusion  
**Issue**: Abstract and conclusion mentioned "75% novelty" without context about threshold definition.  
**Impact**: Could mislead readers about what the percentage represents.  
**Fix**: Updated to emphasize distributional properties while maintaining threshold context.  
**Verification**: All mentions of novelty now include appropriate context.

## Verification Results

### Numerical Accuracy
- ✅ Table 1 values match CSV files exactly (verified against `*_network_metrics_comparison.csv`)
- ✅ Table 2 values match CSV files exactly (verified against `*_novelty_analysis.csv`)
- ✅ All percentage changes verified against computed values

### Figure Integration
- ✅ All figure paths verified (15 PNG files exist: 3 datasets × 5 plot types)
- ✅ All `\includegraphics` paths correct
- ✅ All `\label{}` and `\ref{}` pairs match

### Methodology Alignment
- ✅ Normalization description matches implementation
- ✅ Novelty definition matches code logic
- ✅ All hyperparameters match code configuration

## Recommendations

1. **Before Submission**: Compile LaTeX to verify all figures render correctly and cross-references resolve
2. **Review**: Have co-authors verify that novelty explanation is clear and scientifically honest
3. **Consider**: Adding a supplementary material section if detailed per-node analysis is needed
4. **Future**: Consider reporting additional novelty metrics beyond threshold-based classification (e.g., distance percentiles, distributional statistics)
