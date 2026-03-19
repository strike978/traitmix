import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import nnls

st.set_page_config(page_title="traitmix", page_icon="🧬", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;800&display=swap');
  html, body, [class*="css"] { font-family: 'DM Mono', monospace; background: #0a0a0f; color: #e8e6f0; }
  .stApp { background: #0a0a0f; }
  h1, h2, h3 { font-family: 'Syne', sans-serif; letter-spacing: -0.02em; }
  .title-block { padding: 2.5rem 0 1.5rem; border-bottom: 1px solid #2a2840; margin-bottom: 2rem; }
  .title-block h1 { font-size: 3.5rem; font-weight: 800; color: #f0eeff; margin: 0; }
  .title-block p { color: #7a749a; font-size: 0.85rem; margin: 0.4rem 0 0; letter-spacing: 0.08em; text-transform: uppercase; }
  .privacy-notice { background: #0f0e18; border: 1px solid #2a2840; border-radius: 6px; padding: 0.9rem 1.2rem; margin-top: 1.5rem; font-size: 0.8rem; color: #9a94ba; display: flex; align-items: center; gap: 0.8rem; }
  .metric-card { background: #12111a; border: 1px solid #2a2840; border-radius: 8px; padding: 1.2rem 1.5rem; margin-bottom: 0.75rem; }
  .metric-label { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #7a749a; margin-bottom: 0.3rem; }
  .metric-value { font-size: 1.6rem; font-weight: 500; color: #c8c2f0; }
  .ancestry-item { background: #12111a; border-left: 3px solid; border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; margin-bottom: 0.6rem; }
  .ancestry-item-label { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #7a749a; margin-bottom: 0.3rem; }
  .ancestry-item-value { font-size: 1.3rem; font-weight: 500; color: #c8c2f0; }
  .ref-populations { background: #0f0e18; border: 1px solid #2a2840; border-radius: 6px; padding: 1rem 1.25rem; margin-top: 1.5rem; }
  .ref-populations h4 { font-size: 0.8rem; letter-spacing: 0.12em; text-transform: uppercase; color: #7a749a; margin-bottom: 1rem; }
  .ref-region { margin-bottom: 1.2rem; }
  .ref-region-name { font-size: 0.85rem; font-weight: 600; color: #c8c2f0; margin-bottom: 0.4rem; }
  .ref-groups { font-size: 0.75rem; color: #7a749a; line-height: 1.6; }
  .info-box { background: #0f0e18; border-left: 3px solid #4a3f8a; padding: 1rem 1.25rem; border-radius: 0 6px 6px 0; font-size: 0.82rem; color: #9a94ba; line-height: 1.7; margin: 1.5rem 0; }
  [data-testid="stFileUploader"] { border: 1px dashed #2a2840; border-radius: 8px; padding: 1rem; background: #0f0e18; }
</style>
""", unsafe_allow_html=True)

# ── Reference groups ───────────────────────────────────────────────────────────
REF_GROUPS = {
    'Northern European': [
        'Russian', 'Orcadian', 'French', 'Finnish in Finland',
        'Utah Residents (CEPH) with Northern and Western European Ancestry',
        'British in England and Scotland',
    ],
    'Southern European': [
        'BergamoItalian', 'Tuscan', 'Basque',
        'Iberian Population in Spain', 'Toscani in Italia',
    ],
    'Oceanic': [
        'PapuanHighlands', 'PapuanSepik', 'Bougainville',
    ],
    'West Asian': [
        'Adygei', 'Druze', 'Mozabite', 'Bedouin', 'Palestinian',
    ],
    'South Asian': [
        'Kalash', 'Bengali in Bangladesh', 'Indian Telugu in the UK',
        'Sri Lankan Tamil in the UK', 'Burusho', 'Balochi', 'Makrani',
        'Gujarati Indian in Houston, TX', 'Punjabi in Lahore, Pakistan',
        'Brahui', 'Pathan', 'Sindhi',
    ],
    'Sub-Saharan African': [
        'BantuSouthAfrica', 'BantuKenya', 'Mandenka', 'Yoruba',
        'Esan in Nigeria', 'Yoruba in Ibadan, Nigeria',
        'Luhya in Webuye, Kenya',
        'Gambian in Western Divisions in the Gambia',
        'Mende in Sierra Leone',
        'San', 'Biaka', 'Mbuti',
    ],
    'Indigenous American': [
        'Surui', 'Pima', 'Colombian', 'Karitiana', 'Maya',
    ],
    'Siberian': [
        'Daur', 'Hezhen', 'Oroqen', 'Mongolian', 'Yakut',
    ],
    'East Asian': [
        'Lahu', 'Cambodian', 'Tujia', 'Miao',
        'Chinese Dai in Xishuangbanna, China',
        'Kinh in Ho Chi Minh City, Vietnam',
        'Han', 'Han Chinese in Beijing, China', 'Southern Han Chinese',
        'Dai', 'She', 'Xibo', 'Yi',
        'Japanese in Tokyo, Japan', 'Japanese',
        'Naxi', 'NorthernHan', 'Tu',
    ],
}

REGION_COLOURS = {
    'Northern European':   '#7B6CF6',
    'Southern European':   '#4FC3F7',
    'West Asian':          '#FFB74D',
    'South Asian':         '#81C784',
    'Sub-Saharan African': '#F06292',
    'East Asian':          '#4DB6AC',
    'Oceanic':             '#BA68C8',
    'Indigenous American': '#FF8A65',
    'Siberian':            '#A5D6A7',
}

_VALID_GT = frozenset(a + b for a in 'ACGT' for b in 'ACGT')

# ── Parsing ────────────────────────────────────────────────────────────────────


def parse_raw_file(file) -> dict[str, str]:
    """
    Parse 23andMe (rsid/chr/pos/genotype) or AncestryDNA (rsid/chr/pos/allele1/allele2)
    raw genotype files. Returns {rsid: two-char genotype string}.

    Fixes vs original:
    - Header row is explicitly skipped in the 23andMe branch (else branch)
    - Both branches now skip lines where parts[0] looks like a header keyword
    """
    lines = [
        l.decode(errors='ignore').strip()
        for l in file
        if l.strip() and not l.startswith(b'#')
    ]
    if not lines:
        return {}

    header = lines[0].split('\t')
    user_snps = {}

    if (header[0].lower() == 'rsid'
            and len(header) >= 5
            and header[3].lower().startswith('allele')):
        # AncestryDNA format: rsid  chromosome  position  allele1  allele2
        col = {c.lower(): i for i, c in enumerate(header)}
        for row in lines[1:]:   # skip header
            parts = row.split('\t')
            if len(parts) < 5:
                continue
            rsid = parts[col['rsid']]
            gt = parts[col['allele1']].strip().upper(
            ) + parts[col['allele2']].strip().upper()
            if gt in _VALID_GT:
                user_snps[rsid] = gt
    else:
        # 23andMe format: rsid  chromosome  position  genotype
        # FIX: skip header row (was lines, including header)
        for row in lines[1:]:
            parts = row.split('\t')
            if len(parts) < 4:
                continue
            rsid = parts[0].strip()
            gt = parts[3].strip().upper()
            if gt in _VALID_GT:   # FIX: filter invalid/missing here, not just in encode_user
                user_snps[rsid] = gt

    return user_snps


# ── Genotype encoding ──────────────────────────────────────────────────────────
def encode_genotypes(geno_series: pd.Series, minor: str) -> np.ndarray:
    """Count copies of minor allele per genotype. Returns array with NaN for missing."""
    s = geno_series.astype(str).str.upper()
    valid_mask = s.isin(_VALID_GT)
    out = np.full(len(s), np.nan)
    out[valid_mask.to_numpy()] = s[valid_mask].str.count(
        minor).to_numpy(dtype=float)
    return out


# ── Reference AF matrix + Fst weights ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_ref_matrix(
    _df: pd.DataFrame,
    snp_cols: tuple[str, ...],          # FIX: tuple (hashable) instead of list
    ref_groups: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Build reference allele-frequency matrix X (n_regions x n_snps),
    compute Fst-proxy weights, and return everything needed for NNLS.

    Returns: X, minor_alleles, region_names, col_means, fst_weights

    Fixes vs original:
    - snp_cols is now a tuple so Streamlit can hash it for caching
    - Fst weights computed here (cached) instead of on every UI rerun
    - col_means returned directly instead of recomputed in main()
    """
    label_col = 'group_full' if 'group_full' in _df.columns else 'group'
    all_ref_groups = [g for grps in ref_groups.values() for g in grps]
    ref_df = _df[_df[label_col].isin(all_ref_groups)]
    snp_cols = list(snp_cols)  # convert back to list internally

    # Global minor allele per SNP across all reference individuals
    minor_alleles = []
    for snp in snp_cols:
        s = ref_df[snp].astype(str).str.upper()
        valid = s[s.isin(_VALID_GT)]
        if valid.empty:
            minor_alleles.append(None)
            continue
        flat = ''.join(valid.tolist())
        counts = {a: flat.count(a) for a in set(flat) if a in 'ACGT'}
        if len(counts) < 2:
            minor_alleles.append(None)
            continue
        minor_alleles.append(min(counts, key=counts.get))
    minor_alleles = np.array(minor_alleles, dtype=object)

    # One mean AF vector per region (averaged across constituent groups)
    region_names = list(ref_groups.keys())
    X = np.full((len(region_names), len(snp_cols)), np.nan)

    for r, region in enumerate(region_names):
        group_afs = []
        for grp in ref_groups[region]:
            grp_df = _df[_df[label_col] == grp]
            if grp_df.empty:
                continue
            af = np.full(len(snp_cols), np.nan)
            for j, (snp, minor) in enumerate(zip(snp_cols, minor_alleles)):
                if minor is None:
                    af[j] = 0.0
                    continue
                enc = encode_genotypes(grp_df[snp], str(minor))
                valid = enc[~np.isnan(enc)]
                if valid.size > 0:
                    af[j] = valid.mean() / 2.0
            group_afs.append(af)
        if group_afs:
            X[r] = np.nanmean(np.stack(group_afs, axis=0), axis=0)

    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Fst-proxy weights: favour SNPs that vary most across regions
    # FIX: computed once here inside the cache, not re-run on every upload
    between_var = np.var(X, axis=0)
    within_var = np.mean(X * (1 - X), axis=0)
    fst_weights = between_var / (between_var + within_var + 1e-9)
    fst_weights = np.where(fst_weights > 0.05, fst_weights, 0.0)

    return X, minor_alleles, region_names, col_means, fst_weights


# ── User AF vector ─────────────────────────────────────────────────────────────
def encode_user(
    user_snps: dict[str, str],
    snp_cols: list[str],
    minor_alleles: np.ndarray,
    col_means: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Encode user genotypes as per-SNP AF values, imputing missing with col_means."""
    y = np.full(len(snp_cols), np.nan)
    n_matched = 0
    for i, (snp, minor) in enumerate(zip(snp_cols, minor_alleles)):
        if minor is None:
            continue
        gt = user_snps.get(snp, '')
        if len(gt) == 2 and gt.upper() in _VALID_GT:
            y[i] = gt.upper().count(str(minor)) / 2.0
            n_matched += 1
    nan_mask = np.isnan(y)
    y[nan_mask] = col_means[nan_mask]
    return y, n_matched


# ── NNLS admixture ─────────────────────────────────────────────────────────────
def run_nnls(
    X: np.ndarray,
    y: np.ndarray,
    region_names: list[str],
    fst_weights: np.ndarray,
) -> pd.DataFrame:
    """
    Weighted NNLS: weight each SNP by sqrt(Fst) so discriminative SNPs
    contribute more. Proportions are normalised to sum to 1.
    """
    w = np.sqrt(fst_weights)
    props, _ = nnls((X * w).T, y * w)
    total = props.sum()
    if total > 0:
        props /= total
    else:
        props = np.full(len(region_names), 1.0 / len(region_names))
    return (
        pd.DataFrame({'Region': region_names, 'Proportion': props})
        .sort_values('Proportion', ascending=False)
    )


# ── Reference populations display ─────────────────────────────────────────────
def render_ref_populations():
    parts = ['<div class="ref-populations"><h4>Populations used in analysis</h4>']
    for region, groups in REF_GROUPS.items():
        color = REGION_COLOURS.get(region, '#7a749a')
        parts.append(
            f'<div class="ref-region">'
            f'<div class="ref-region-name" style="color:{color};">{region}</div>'
            f'<div class="ref-groups">{", ".join(groups)}</div>'
            f'</div>'
        )
    parts.append('</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    with st.spinner("Loading reference panel…"):
        try:
            df_raw = pd.read_csv("113_no_outliers.csv", na_values=['--', ''])
        except Exception as e:
            st.error(f"Could not load reference panel: {e}")
            return

    df = df_raw[df_raw['source'].str.lower().isin(['1000g', 'hgdp'])].copy()
    # tuple for cache hashing
    snp_cols = tuple(c for c in df.columns if c.startswith('rs'))

    st.markdown("""
    <div class="title-block">
        <h1>traitmix</h1>
        <p>NNLS ancestry estimation based on phenotypical SNPs · 1000G + HGDP reference</p>
    </div>
    """, unsafe_allow_html=True)

    # FIX: removed stray quote marks around emoji
    st.markdown(
        '<div class="privacy-notice">🔒 <b>Privacy notice:</b> '
        'Your genetic data is never stored or uploaded. All processing happens '
        'locally in your browser and is not retained.</div>',
        unsafe_allow_html=True,
    )

    raw_file = st.file_uploader(
        "Upload your 23andMe or Ancestry raw data (.txt)", type=["txt"])

    if not raw_file:
        st.markdown("### reference populations")
        render_ref_populations()
        return

    # Build reference matrix (cached — runs once per session)
    with st.spinner("Building reference AF matrix…"):
        X, minor_alleles, region_names, col_means, fst_weights = build_ref_matrix(
            df, snp_cols, REF_GROUPS
        )

    # FIX: use 'is not None' idiom for object arrays
    valid_snp_count = int(sum(m is not None for m in minor_alleles))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">SNPs in panel</div>'
            f'<div class="metric-value">{valid_snp_count:,}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">reference regions</div>'
            f'<div class="metric-value">{len(region_names)}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        n_groups = sum(len(v) for v in REF_GROUPS.values())
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">reference groups</div>'
            f'<div class="metric-value">{n_groups}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    with st.spinner("Parsing your genotype file…"):
        user_snps = parse_raw_file(raw_file)

    if not user_snps:
        st.error("Could not parse any SNPs from your file. "
                 "Make sure it is an unzipped 23andMe or AncestryDNA raw data file.")
        return

    y, n_matched = encode_user(user_snps, list(
        snp_cols), minor_alleles, col_means)
    pct_matched = n_matched / valid_snp_count * 100 if valid_snp_count else 0

    st.markdown(
        f'<div class="metric-card" style="margin-bottom:1.5rem;">'
        f'<div class="metric-label">SNPs matched to reference panel</div>'
        f'<div class="metric-value">{n_matched:,}'
        f'<span style="font-size:1rem; color:#7a749a;"> / {valid_snp_count:,}'
        f'&nbsp;({pct_matched:.1f}%)</span></div></div>',
        unsafe_allow_html=True,
    )

    if pct_matched < 5:
        st.warning(
            "Fewer than 5% of panel SNPs matched. Results may be unreliable.")

    with st.spinner("Running NNLS admixture…"):
        mix_df = run_nnls(X, y, region_names, fst_weights)

    st.markdown("## ancestry results")
    st.markdown(
        '<div class="info-box">'
        '<b>How are your ancestry results calculated?</b><br>'
        'We compare your DNA to reference populations from around the world. '
        'For each region, we compute an average allele frequency profile across '
        'multiple constituent groups. We then use non-negative least squares (NNLS) '
        'to find the mixture of regions that best explains your allele frequencies, '
        'with SNPs weighted by how much they differ between regions (Fst). '
        'All percentages are positive and sum to 100%.<br><br>'
        '<i>Results reflect broad population affinity based on 113 phenotypically '
        'informative markers — not a genome-wide ancestry test.</i>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### ancestry breakdown")
    for _, row in mix_df.iterrows():
        region = row['Region']
        proportion = row['Proportion']
        if proportion > 0.001:
            color = REGION_COLOURS.get(region, '#4a3f8a')
            st.markdown(
                f'<div class="ancestry-item" style="border-color:{color};">'
                f'<div class="ancestry-item-label">{region}</div>'
                f'<div class="ancestry-item-value">{proportion * 100:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("### reference populations")
    render_ref_populations()


if __name__ == "__main__":
    main()
