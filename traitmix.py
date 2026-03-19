import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import nnls
import plotly.graph_objects as go

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
  .privacy-notice svg { width: 18px; height: 18px; stroke: #7B6CF6; flex-shrink: 0; }
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
  .stDataFrame { background: #12111a !important; }
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
        'Adygei', 'Druze', 'Sardinian', 'Mozabite', 'Bedouin', 'Palestinian',
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
    ],
    # 'African Hunter-Gatherer': [
    #     'San', 'Biaka', 'Mbuti',
    # ],
    'Indigenous American': [
        'Surui', 'Pima', 'Colombian', 'Karitiana',
        'Maya',
        # 'Peruvian in Lima, Peru',
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
    'Northern European':       '#7B6CF6',
    'Southern European':       '#4FC3F7',
    'West Asian':              '#FFB74D',
    'South Asian':             '#81C784',
    'Sub-Saharan African':     '#F06292',
    'African Hunter-Gatherer': '#AD1457',
    'East Asian':              '#4DB6AC',
    'Oceanic':                 '#BA68C8',
    'Indigenous American':     '#FF8A65',
    'Siberian':                '#A5D6A7',
}

_VALID_GT = frozenset(a + b for a in 'ACGT' for b in 'ACGT')


# ── Parsing ────────────────────────────────────────────────────────────────────
def parse_raw_file(file) -> dict[str, str]:
    lines = [l.decode(errors='ignore').strip()
             for l in file if l.strip() and not l.startswith(b'#')]
    if not lines:
        return {}
    header = lines[0].split('\t')
    user_snps = {}
    if (header[0].lower() == 'rsid' and len(header) >= 5
            and header[3].lower().startswith('allele')):
        col = {c: i for i, c in enumerate(header)}
        for row in lines[1:]:
            parts = row.split('\t')
            if len(parts) < 5:
                continue
            user_snps[parts[col['rsid']]] = parts[col['allele1']] + \
                parts[col['allele2']]
    else:
        for row in lines:
            parts = row.split('\t')
            if len(parts) >= 4:
                user_snps[parts[0]] = parts[3]
    return user_snps


# ── Genotype encoding ──────────────────────────────────────────────────────────
def encode_genotypes(geno_series: pd.Series, minor: str) -> np.ndarray:
    s = geno_series.astype(str).str.upper()
    valid_mask = s.isin(_VALID_GT)
    out = np.full(len(s), np.nan)
    out[valid_mask.to_numpy()] = s[valid_mask].str.count(
        minor).to_numpy(dtype=float)
    return out


# ── Reference AF matrix ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_ref_matrix(
    _df: pd.DataFrame,
    snp_cols: list[str],
    ref_groups: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    One mean AF vector per region, averaged across all constituent groups.
    Returns X (n_regions × n_snps), minor_alleles, region_names.
    """
    label_col = 'group_full' if 'group_full' in _df.columns else 'group'
    all_ref_groups = [g for grps in ref_groups.values() for g in grps]
    ref_df = _df[_df[label_col].isin(all_ref_groups)]

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

    # One AF vector per region
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

    return X, minor_alleles, region_names


# ── User AF vector ─────────────────────────────────────────────────────────────
def encode_user(
    user_snps: dict[str, str],
    snp_cols: list[str],
    minor_alleles: np.ndarray,
    col_means: np.ndarray,
) -> tuple[np.ndarray, int]:
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
) -> pd.DataFrame:
    props, _ = nnls(X.T, y)
    total = props.sum()
    props = props / \
        total if total > 0 else np.full(
            len(region_names), 1.0 / len(region_names))
    return pd.DataFrame({
        'Region': region_names,
        'Proportion': props,
    }).sort_values('Proportion', ascending=False)


# ── Helper: render reference populations block ─────────────────────────────────
def render_ref_populations():
    ref_html = '<div class="ref-populations"><h4>Populations used in analysis</h4>'
    for region, groups in REF_GROUPS.items():
        color = REGION_COLOURS.get(region, '#7a749a')
        ref_html += (
            f'<div class="ref-region">'
            f'<div class="ref-region-name" style="color: {color};">{region}</div>'
            f'<div class="ref-groups">{", ".join(groups)}</div>'
            f'</div>'
        )
    ref_html += '</div>'
    st.markdown(ref_html, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():

    # Load reference panel and SNP columns at the very start
    with st.spinner("Loading reference panel…"):
        try:
            df_raw = pd.read_csv("113_no_outliers.csv")
        except Exception as e:
            st.error(f"Could not load reference panel: {e}")
            return
    df = df_raw[df_raw['source'].str.lower().isin(['1000g', 'hgdp'])].copy()
    snp_cols = [c for c in df.columns if c.startswith('rs')]

    st.markdown("""
    <div class="title-block">
        <h1>traitmix</h1>
        <p>NNLS ancestry estimation based on phenotypical SNPs · 1000G + HGDP reference</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '''<div class="privacy-notice">'🔒' <b>Privacy notice:</b> Your genetic data is never stored or uploaded. All processing happens locally in your browser and is not retained.</div>''',
        unsafe_allow_html=True,
    )

    # ── File uploader ──────────────────────────────────────────────────────────
    raw_file = st.file_uploader(
        "Upload your 23andMe or Ancestry raw data (.txt)", type=["txt"])

    if not raw_file:
        st.markdown("### reference populations")
        render_ref_populations()
        return

    # ── Build reference matrix ─────────────────────────────────────────────────
    with st.spinner("Building reference AF matrix…"):
        X, minor_alleles, region_names = build_ref_matrix(
            df, snp_cols, REF_GROUPS)
        col_means = X.mean(axis=0)

    # Only count SNPs that have a valid minor allele as "in panel"
    # (SNPs where minor is None are uninformative and skipped during matching)
    valid_snp_count = int(np.sum(minor_alleles != None))

    # ── Summary metrics ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">SNPs in panel</div>'
            f'<div class="metric-value">{valid_snp_count:,}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">reference regions</div>'
            f'<div class="metric-value">{len(region_names)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        n_groups = sum(len(v) for v in REF_GROUPS.values())
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">reference groups</div>'
            f'<div class="metric-value">{n_groups}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Parse user file ────────────────────────────────────────────────────────
    with st.spinner("Parsing your genotype file…"):
        user_snps = parse_raw_file(raw_file)

    y, n_matched = encode_user(user_snps, snp_cols, minor_alleles, col_means)
    # Denominator is valid_snp_count — SNPs where minor is None are never
    # matchable and should not count against the user's match rate.
    pct_matched = n_matched / valid_snp_count * 100 if valid_snp_count else 0

    st.markdown(
        f'<div class="metric-card" style="margin-bottom:1.5rem;">'
        f'<div class="metric-label">SNPs matched to reference panel</div>'
        f'<div class="metric-value">{n_matched:,}'
        f'<span style="font-size:1rem; color:#7a749a;"> / {valid_snp_count:,}&nbsp;({pct_matched:.1f}%)</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if pct_matched < 5:
        st.warning(
            "Fewer than 5% of panel SNPs matched. Results may be unreliable.")

    # ── Run NNLS ───────────────────────────────────────────────────────────────
    with st.spinner("Running NNLS admixture…"):
        mix_df = run_nnls(X, y, region_names)

    st.markdown("## ancestry results")
    st.markdown(
        '<div class="info-box">'
        '<b>How are your ancestry results calculated?</b><br>'
        'We analyze your DNA by comparing it to a set of reference populations, each representing a different region of the world. '
        'For each region, we combine genetic data from several groups to create an average DNA profile. '
        'Next, we use a mathematical method to find the mix of these regions that best matches your DNA. '
        'All percentages are positive and add up to 100%.<br><br>'
        '<i>In short: Your results show which regions of the world your DNA most closely matches, based on a selected set of phenotypical trait markers.</i>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Ancestry breakdown ─────────────────────────────────────────────────────
    st.markdown("### ancestry breakdown")
    for _, row in mix_df.iterrows():
        region = row['Region']
        proportion = row['Proportion']
        if proportion > 0.001:
            color = REGION_COLOURS.get(region, '#4a3f8a')
            st.markdown(
                f'<div class="ancestry-item" style="border-color: {color};">'
                f'<div class="ancestry-item-label">{region}</div>'
                f'<div class="ancestry-item-value">{proportion * 100:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Reference populations (shown after results) ────────────────────────────
    st.markdown("### reference populations")
    render_ref_populations()


if __name__ == "__main__":
    main()
