import matplotlib.pyplot as plt
import numpy as np
import os


def plot_split_distribution(train_df=None, val_df=None, test_df=None,
                            labels=("Entrenamiento", "Validación", "Prueba"),
                            colors=("#c7e9c0", "#9ecae1", "#fb6a4a"),
                            figsize=(10, 1.5),
                            outpath="datos/plots/data_split_distribution.png"):
    # Resolve counts
  
    if train_df is None or val_df is None or test_df is None:
        raise ValueError("train_df, val_df, and test_df must be provided")
    counts = (len(train_df), len(val_df), len(test_df))

    total = sum(counts)
    if total == 0:
        raise ValueError("Total count is zero")

    # percentages
    fracs = [c / total for c in counts]

    # prepare figure
    fig, ax = plt.subplots(figsize=figsize)

    left = 0.0
    for frac, label, color, cnt in zip(fracs, labels, colors, counts):
        ax.barh(0, width=frac, left=left, height=1, color=color, linewidth=0.8)

        pct_text = f"{100*frac:.2f}%\n{label}"

        # compute center of this segment in axis fraction coordinates
        center = left + frac / 2
        ax.text(center, 0, pct_text, ha='center', va='center', fontsize=10)
        left += frac

    # aesthetics
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])


    # place caption under the axis
    fig.subplots_adjust(bottom=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    outdir = os.path.dirname(outpath)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)

def stratified_split(df, test_size=0.25, val_size=0.10, random_state=42):
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(test_size * len(df))
    n_val = int(val_size * len(df))
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    plot_split_distribution(df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx])
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


if __name__ == "__main__":
    # small demo: try to load datos/Resilience_CleanOnly_v1_PREPROCESSED.csv if present
    import pandas as pd
    demo_path = "/home/moya/TP-IAA/datos/Resilience_CleanOnly_v1_PREPROCESSED_v2.csv"
    print(demo_path)
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        train, val, test = stratified_split(df)
        print('Saved plot to datos/plots/data_split_distribution.png')
    else:
        print(f"Demo file {demo_path} not found. Please provide your own DataFrame to test the plotting function.")
  