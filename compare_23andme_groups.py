import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import nnls
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="mixy", page_icon="🧬", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background: #0a0a0f;
    color: #e8e6f0;
  }
  .stApp { background: #0a0a0f; }

  h1, h2, h3 { font-family: 'Syne', sans-serif; letter-spacing: -0.02em; }

  .title-block {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #2a2840;
    margin-bottom: 2rem;
  }
  .title-block h1 {
    font-size: 3.5rem;
    font-weight: 800;
    color: #f0eeff;
    margin: 0;
  }
  .title-block p {
    color: #7a749a;
    font-size: 0.85rem;
    margin: 0.4rem 0 0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .metric-card {
    background: #12111a;
    border: 1px solid #2a2840;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
  }
  .metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7a749a;
    margin-bottom: 0.3rem;
  }
  .metric-value {
    font-size: 1.6rem;
    font-weight: 500;
    color: #c8c2f0;
  }

  .info-box {
    background: #0f0e18;
    border-left: 3px solid #4a3f8a;
    padding: 1rem 1.25rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #9a94ba;
    line-height: 1.7;
    margin: 1.5rem 0;
  }

  .stDataFrame { background: #12111a !important; }

  /* Upload widget */
  [data-testid="stFileUploader"] {
    border: 1px dashed #2a2840;
    border-radius: 8px;
    padding: 1rem;
    background: #0f0e18;
  }

  .stButton > button {
    background: #2a2050;
    border: 1px solid #4a3f8a;
    color: #c8c2f0;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    border-radius: 4px;
  }
</style>
""", unsafe_allow_html=True)


# ── Reference groups ───────────────────────────────────────────────────────────
REF_GROUPS = {
    'Northern European':    ['GBR', 'FIN'],
    'Southern European':    ['IBS', 'TSI'],
    'West Asian':           ['Adygei', 'Druze'],
    'African':              ['YRI', 'ESN'],
    'African HG':           ['Mbuti', 'San'],
    'Oceanic':              ['PapuanHighlands', 'PapuanSepik'],
    'South Asian':          ['GIH', 'STU', 'ITU'],
    'East Asian':           ['JPT', 'CHB', 'Dai'],
    'Indigenous American':  ['Pima', 'Colombian'],
}

PALETTE = [
    '#7B6CF6', '#4FC3F7', '#81C784', '#FFB74D',
    '#F06292', '#BA68C8', '#4DB6AC', '#FF8A65', '#A5D6A7',
]


# ── Parsing ────────────────────────────────────────────────────────────────────
def parse_raw_file(file) -> dict[str, str]:
    """Parse 23andMe or Ancestry raw data. Returns {rsid: genotype}."""
    lines = [l.decode(errors='ignore').strip()
             for l in file if l.strip() and not l.startswith(b'#')]
    if not lines:
        return {}

    header = lines[0].split('\t')
    user_snps = {}

    # Ancestry format: rsid, chromosome, position, allele1, allele2
    if header[0].lower() == 'rsid' and len(header) >= 5 and header[3].lower().startswith('allele'):
        col = {c: i for i, c in enumerate(header)}
        for row in lines[1:]:
            parts = row.split('\t')
            if len(parts) < 5:
                continue
            rsid = parts[col['rsid']]
            user_snps[rsid] = parts[col['allele1']] + parts[col['allele2']]
    else:
        # 23andMe format: rsid, chrom, pos, genotype
        for row in lines:
            parts = row.split('\t')
            if len(parts) >= 4:
                user_snps[parts[0]] = parts[3]

    return user_snps


# ── Vectorised genotype encoding ───────────────────────────────────────────────
_VALID_GT = frozenset(
    a + b for a in 'ACGT' for b in 'ACGT'
)


def encode_genotypes(geno_series: pd.Series, minor: str) -> np.ndarray:
    """
    Vectorised: encode a Series of genotype strings as minor-allele counts.
    Invalid / missing entries become NaN.
    """
    s = geno_series.astype(str).str.upper()
    valid_mask = s.isin(_VALID_GT)
    out = np.full(len(s), np.nan)
    valid_s = s[valid_mask]
    # Count occurrences of the minor allele character in the 2-char genotype
    counts = valid_s.str.count(minor).to_numpy(dtype=float)
    out[valid_mask.to_numpy()] = counts
    return out


# ── Core statistics ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_minor_alleles(_df: pd.DataFrame, snp_cols: list[str]) -> dict[str, str | None]:
    """
    For each SNP return the minor allele (the less frequent across all samples).
    Returns None for monomorphic or invalid SNPs.
    """
    minor = {}
    for snp in snp_cols:
        s = _df[snp].astype(str).str.upper()
        valid = s[s.isin(_VALID_GT)]
        if valid.empty:
            minor[snp] = None
            continue
        # Flatten all allele characters
        flat = ''.join(valid.tolist())
        counts = {a: flat.count(a) for a in set(flat) if a in 'ACGT'}
        if len(counts) < 2:
            minor[snp] = None
            continue
        minor[snp] = min(counts, key=counts.get)
    return minor


@st.cache_data(show_spinner=False)
def compute_group_afs(
    _df: pd.DataFrame,
    snp_cols: list[str],
    minor_alleles: dict,
    group_col: str,
) -> dict[str, np.ndarray]:
    """
    Returns {group: allele_freq_vector} where each vector has length len(snp_cols).
    NaN SNPs are imputed with the group-level mean AF.
    """
    group_afs = {}
    groups = _df[group_col].unique()

    for grp in groups:
        sub = _df[_df[group_col] == grp]
        af = np.full(len(snp_cols), np.nan)
        for i, snp in enumerate(snp_cols):
            m = minor_alleles.get(snp)
            if not m:
                af[i] = 0.0
                continue
            enc = encode_genotypes(sub[snp], m)
            valid = enc[~np.isnan(enc)]
            if valid.size > 0:
                af[i] = valid.mean() / 2.0          # diploid → allele freq
        # Impute missing with group mean
        nan_mask = np.isnan(af)
        if nan_mask.any():
            af[nan_mask] = np.nanmean(af) if not np.all(nan_mask) else 0.0
        group_afs[grp] = af

    return group_afs


# ── FST calculation ────────────────────────────────────────────────────────────
def compute_fst_weights(
    group_afs: dict[str, np.ndarray],
    ref_codes: list[str],
    snp_cols: list[str],
    min_fst: float = 0.0,
    top_k: int | None = None,
) -> np.ndarray:
    """
    Compute per-SNP FST across the reference groups using the
    Weir & Cockerham (1984) simplified estimator:

        FST ≈ Var(p) / (p̄ · (1 − p̄))

    where p̄ = mean allele freq across groups, Var(p) = variance across groups.

    Returns a weight vector of length len(snp_cols).
    Optionally zeroes out all but the top-k SNPs by FST.
    """
    # Collect AF matrix: shape (n_ref_groups, n_snps)
    af_matrix = np.stack(
        [group_afs[code] for code in ref_codes if code in group_afs],
        axis=0,
    )  # (G, S)

    p_bar = af_matrix.mean(axis=0)                    # mean across groups
    p_var = af_matrix.var(axis=0)                     # variance across groups

    denom = p_bar * (1.0 - p_bar)
    with np.errstate(invalid='ignore', divide='ignore'):
        fst = np.where(denom > 1e-8, p_var / denom, 0.0)

    fst = np.clip(fst, min_fst, None)

    if top_k is not None and top_k < len(fst):
        threshold = np.partition(fst, -top_k)[-top_k]
        fst = np.where(fst >= threshold, fst, 0.0)

    return fst


# ── Ancestry mean vectors ──────────────────────────────────────────────────────
def build_ancestry_matrix(
    group_afs: dict[str, np.ndarray],
    ref_groups: dict[str, list[str]],
    snp_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    For each named ancestry, average the AF vectors of its constituent groups.
    Returns (matrix X of shape (n_snps, n_ancestries), list of ancestry names).
    """
    names, cols = [], []
    for ancestry, codes in ref_groups.items():
        vecs = [group_afs[c] for c in codes if c in group_afs]
        if not vecs:
            continue
        names.append(ancestry)
        cols.append(np.stack(vecs, axis=0).mean(axis=0))
    X = np.column_stack(cols)   # (n_snps, n_ancestries)
    return X, names


# ── User vector ────────────────────────────────────────────────────────────────
def encode_user(
    user_snps: dict[str, str],
    snp_cols: list[str],
    minor_alleles: dict,
    group_afs: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Encode user genotypes as a minor-allele count vector.
    Missing SNPs are imputed with the global mean AF across all reference groups.
    """
    global_mean = np.nanmean(
        np.stack(list(group_afs.values()), axis=0), axis=0
    ) * 2.0  # convert AF → expected count

    y = np.full(len(snp_cols), np.nan)
    for i, snp in enumerate(snp_cols):
        m = minor_alleles.get(snp)
        gt = user_snps.get(snp, '')
        if m and len(gt) == 2 and gt.upper() in _VALID_GT:
            y[i] = gt.upper().count(m)

    nan_mask = np.isnan(y)
    y[nan_mask] = global_mean[nan_mask]
    return y


# ── FST-weighted NNLS ──────────────────────────────────────────────────────────
def fst_weighted_nnls(
    X: np.ndarray,          # (n_snps, n_ancestries)  reference AF matrix
    y: np.ndarray,          # (n_snps,)               user vector
    fst_weights: np.ndarray,  # (n_snps,)
) -> np.ndarray:
    """
    Solve the weighted NNLS problem:

        minimise  ‖ W(Xβ − y) ‖²   s.t.  β ≥ 0

    by pre-multiplying both sides by the diagonal weight matrix W = diag(√fst).
    This is equivalent to weighting each SNP's contribution by its FST,
    so informationally poor SNPs contribute little to the fit.
    """
    W = np.sqrt(fst_weights)            # √ because we square in the residual
    Xw = X * W[:, np.newaxis]           # broadcast: (n_snps, n_ancestries)
    yw = y * W

    props, _ = nnls(Xw, yw)
    total = props.sum()
    return props / total if total > 0 else props


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_admixture(admix_df: pd.DataFrame) -> go.Figure:
    admix_df = admix_df[admix_df['Proportion'] > 0.001].sort_values(
        'Proportion', ascending=True
    )
    fig = go.Figure(go.Bar(
        x=admix_df['Proportion'] * 100,
        y=admix_df['Ancestry'],
        orientation='h',
        marker=dict(
            color=PALETTE[:len(admix_df)],
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in admix_df['Proportion'] * 100],
        textposition='outside',
        textfont=dict(family='DM Mono', size=11, color='#9a94ba'),
    ))
    fig.update_layout(
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='#0a0a0f',
        font=dict(family='DM Mono', color='#9a94ba'),
        margin=dict(l=160, r=80, t=30, b=30),
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor='#1e1c2e',
            gridwidth=1,
            zeroline=False,
            ticksuffix='%',
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11),
        ),
        height=max(250, len(admix_df) * 52 + 60),
        bargap=0.35,
    )
    return fig


def plot_fst_distribution(fst_weights: np.ndarray) -> go.Figure:
    nonzero = fst_weights[fst_weights > 0]
    fig = go.Figure(go.Histogram(
        x=nonzero,
        nbinsx=60,
        marker_color='#4a3f8a',
        marker_line_width=0,
        opacity=0.85,
    ))
    fig.update_layout(
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='#0a0a0f',
        font=dict(family='DM Mono', color='#9a94ba'),
        xaxis=dict(
            title='FST',
            showgrid=True,
            gridcolor='#1e1c2e',
            zeroline=False,
        ),
        yaxis=dict(
            title='SNP count',
            showgrid=True,
            gridcolor='#1e1c2e',
        ),
        margin=dict(l=60, r=30, t=20, b=50),
        height=200,
        bargap=0.05,
    )
    return fig


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="title-block">
        <h1>mixy</h1>
        <p>FST-weighted admixture · 1000G + HGDP reference panel</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load reference data ────────────────────────────────────────────────────
    with st.spinner("Loading reference panel…"):
        try:
            df_raw = pd.read_csv("113_no_outliers.csv")
        except Exception as e:
            st.error(f"Could not load reference panel: {e}")
            return

    df = df_raw[df_raw['source'].str.lower().isin(['1000g', 'hgdp'])].copy()
    snp_cols = [c for c in df.columns if c.startswith('rs')]
    group_col = 'group_full' if 'group_full' in df.columns else 'group'

    # ── Sidebar controls ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### settings")
        if len(snp_cols) <= 500:
            top_k = len(snp_cols)
            st.markdown(f"Using all {len(snp_cols)} SNPs in panel.")
        else:
            top_k = st.slider(
                "Top-K informative SNPs",
                min_value=500, max_value=len(snp_cols),
                value=min(5000, len(snp_cols)),
                step=500,
                help="Use only the top-K SNPs ranked by FST. Higher = slower but uses more data.",
            )
        min_fst = st.slider(
            "Minimum FST threshold",
            min_value=0.0, max_value=0.5,
            value=0.02, step=0.01,
            help="SNPs below this FST are given zero weight. Removes near-monomorphic sites.",
        )
        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.72rem; color:#4a4470; line-height:1.8;">
        <b>method</b><br>
        FST-weighted NNLS admixture.<br><br>
        Each SNP is weighted by its Weir &amp; Cockerham FST
        across reference populations before solving
        the non-negative least squares problem.<br><br>
        High-FST SNPs — those that actually vary
        between populations — dominate the fit.
        Monomorphic noise is suppressed.
        </div>
        """, unsafe_allow_html=True)

    # ── Pre-compute reference statistics (cached) ──────────────────────────────

    with st.spinner("Computing allele frequencies…"):
        minor_alleles = compute_minor_alleles(df, snp_cols)

        # Gather all unique group codes used in REF_GROUPS
        all_ref_codes = list(
            {c for codes in REF_GROUPS.values() for c in codes})
        df_ref = df[df['group'].isin(all_ref_codes)]

        group_afs = compute_group_afs(df_ref, snp_cols, minor_alleles, 'group')

    # FST across all reference groups
    with st.spinner("Computing FST weights…"):
        present_codes = [c for c in all_ref_codes if c in group_afs]
        fst_weights = compute_fst_weights(
            group_afs, present_codes, snp_cols,
            min_fst=min_fst, top_k=top_k,
        )
        n_informative = int((fst_weights > 0).sum())

    # ── Stats display ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">SNPs in panel</div>
          <div class="metric-value">{len(snp_cols):,}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">informative SNPs (FST &gt; {min_fst})</div>
          <div class="metric-value">{n_informative:,}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        mean_fst = float(
            fst_weights[fst_weights > 0].mean()) if n_informative > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">mean FST (informative)</div>
          <div class="metric-value">{mean_fst:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("FST distribution across SNPs"):
        st.plotly_chart(
            plot_fst_distribution(fst_weights),
            use_container_width=True,
        )

    # ── File upload ────────────────────────────────────────────────────────────
    st.markdown("---")
    raw_file = st.file_uploader(
        "Upload your 23andMe or Ancestry raw data (.txt)",
        type=["txt"],
    )

    if not raw_file:
        st.markdown("""
        <div class="info-box">
          Upload your raw DNA file above to run admixture estimation.<br>
          Supported formats: 23andMe v3/v5, Ancestry.com raw data.
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Parse user file ────────────────────────────────────────────────────────
    with st.spinner("Parsing your genotype file…"):
        user_snps = parse_raw_file(raw_file)

    n_matched = sum(1 for snp in snp_cols if snp in user_snps)
    pct_matched = n_matched / len(snp_cols) * 100 if snp_cols else 0

    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:1.5rem;">
      <div class="metric-label">SNPs matched to reference panel</div>
      <div class="metric-value">{n_matched:,} <span style="font-size:1rem; color:#7a749a;">/ {len(snp_cols):,} &nbsp;({pct_matched:.1f}%)</span></div>
    </div>""", unsafe_allow_html=True)

    if pct_matched < 5:
        st.warning(
            "Fewer than 5% of panel SNPs found in your file. "
            "Results may be unreliable. Check that your file is a supported format."
        )

    # ── Build matrices & run weighted NNLS ────────────────────────────────────
    with st.spinner("Running FST-weighted NNLS…"):
        X, ancestry_names = build_ancestry_matrix(
            group_afs, REF_GROUPS, snp_cols)
        y = encode_user(user_snps, snp_cols, minor_alleles, group_afs)
        props = fst_weighted_nnls(X, y, fst_weights)

    admix_df = pd.DataFrame({
        'Ancestry': ancestry_names,
        'Proportion': props,
    }).sort_values('Proportion', ascending=False)

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown("## admixture results")
    st.markdown("""
    <div class="info-box">
      <b>FST-weighted NNLS</b> — each SNP contributes to the fit in proportion
      to its F<sub>ST</sub> across the reference populations. Uninformative
      (near-monomorphic) SNPs are down-weighted, so the result is driven by
      SNPs that genuinely distinguish populations.
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(plot_admixture(admix_df), use_container_width=True)

    with st.expander("View raw proportions table"):
        st.dataframe(
            admix_df.assign(
                Proportion=lambda d: d['Proportion'].map('{:.4f}'.format)),
            hide_index=True,
            use_container_width=True,
        )

    # ── Top ancestry callout ───────────────────────────────────────────────────
    top_row = admix_df.iloc[0]
    st.markdown(f"""
    <div class="metric-card" style="margin-top:1.5rem; border-color:#4a3f8a;">
      <div class="metric-label">dominant ancestry</div>
      <div class="metric-value" style="color:#c8c2f0;">
        {top_row['Ancestry']}
        <span style="font-size:1rem; color:#7a749a; margin-left:0.75rem;">
          {top_row['Proportion']*100:.1f}%
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
