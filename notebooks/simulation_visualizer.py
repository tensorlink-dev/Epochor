# notebooks/simulation_visualizer.py

"""
This script outlines the visualization logic that might be used in
a Jupyter notebook (simulation.ipynb) to analyze Epochor scoring dynamics.

It would typically load simulation data (e.g., from a CSV or a test run like
test_scoring.py) and generate plots to understand win rates, score gaps,
and leaderboard evolution.

Requires: pandas, matplotlib, numpy (and data from a simulation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Helper Functions (Conceptual - would use actual simulation data) ---

def load_simulation_data(filepath="./test_simulation_results.csv") -> pd.DataFrame:
    """
    Placeholder to load simulation data.
    The data should contain rounds, UIDs, raw_scores, ema_scores, weights.
    For this example, we'll generate some mock data if the file doesn't exist.
    """
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}. Generating mock data instead.")
    
    print(f"Generating mock simulation data as {filepath} was not found.")
    num_rounds = 100
    num_miners = 20
    uids = list(range(num_miners))
    data = []
    np.random.seed(42)
    # Base performance that slowly diverges for some miners
    base_ema = {uid: 0.5 + uid * 0.02 for uid in uids}
    
    for r in range(num_rounds):
        leader_score_this_round = float('inf')
        round_scores = {}
        for uid in uids:
            # Simulate EMA score (loss, lower is better)
            noise = np.random.normal(0, 0.05)
            drift = (uid % 5 - 2) * 0.001 * r # Some improve, some worsen
            ema_score = max(0.01, base_ema[uid] + noise - drift)
            round_scores[uid] = ema_score
            if ema_score < leader_score_this_round:
                leader_score_this_round = ema_score
        
        for uid in uids:
            ema_score = round_scores[uid]
            # Mock inverted score for win rate proxy (higher is better)
            inverted_score = 1 / (1 + ema_score)
            # Mock weight (simplified)
            weight = inverted_score / sum(1 / (1 + s) for s in round_scores.values())
            
            data.append({
                "round": r,
                "uid": uid,
                "ema_loss": ema_score,
                "weight": weight,
                "leader_ema_loss_this_round": leader_score_this_round
            })
            # Update base for next round slightly
            base_ema[uid] = max(0.01, base_ema[uid] + np.random.normal(0, 0.005) - (uid % 3 -1) * 0.0005)

    df = pd.DataFrame(data)
    try: # Try to save for future runs if it was generated
        df.to_csv(filepath, index=False)
        print(f"Mock data saved to {filepath}")
    except Exception as e:
        print(f"Could not save mock data: {e}")
    return df

def calculate_win_rate_and_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a conceptual 'win rate' (e.g., being in top N% or beating a threshold)
    and 'gap from leader' for each miner over time.
    
    This is highly dependent on the definition of "win".
    For this example: "win" = having EMA loss in the best 20% for the round.
    "Gap" = Miner EMA Loss - Leader EMA Loss for the round.
    """
    if df.empty:
        return df

    results = []
    for round_num, group in df.groupby("round"):
        leader_loss = group["ema_loss"].min()
        # Define top 20% threshold for this round
        top_20_percentile_loss = group["ema_loss"].quantile(0.20)
        
        for _, row in group.iterrows():
            uid = row["uid"]
            ema_loss = row["ema_loss"]
            
            is_winner = 1 if ema_loss <= top_20_percentile_loss else 0
            gap_from_leader = ema_loss - leader_loss
            
            results.append({
                "round": round_num,
                "uid": uid,
                "ema_loss": ema_loss,
                "is_winner_this_round": is_winner,
                "gap_from_leader_loss": gap_from_leader,
                "weight": row["weight"]
            })
            
    return pd.DataFrame(results)

# --- Plotting Functions ---

def plot_ema_scores_evolution(df: pd.DataFrame, top_n_uids: int = 5):
    """Plots EMA scores (losses) over rounds for top N UIDs and average."""
    if df.empty:
        print("DataFrame is empty, skipping EMA scores plot.")
        return
    plt.figure(figsize=(12, 6))
    # Find top N UIDs by average EMA loss (lower is better)
    avg_losses = df.groupby("uid")["ema_loss"].mean().nsmallest(top_n_uids).index
    
    for uid in avg_losses:
        subset = df[df["uid"] == uid]
        plt.plot(subset["round"], subset["ema_loss"], label=f"UID {uid}")
    
    # Plot average EMA loss of all miners
    avg_all_miners = df.groupby("round")["ema_loss"].mean()
    plt.plot(avg_all_miners.index, avg_all_miners.values, label="Average All Miners", linestyle='--', color='black')
    
    plt.xlabel("Round")
    plt.ylabel("EMA Score (Loss - Lower is Better)")
    plt.title(f"EMA Score Evolution (Top {top_n_uids} UIDs & Average)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("ema_scores_evolution.png")
    plt.show()

def plot_winrate_vs_gap(df_analysis: pd.DataFrame, uid_to_highlight: Optional[int] = None):
    """
    Visualizes win rate (or average weight) against average gap from leader.
    X-axis: Average Gap from Leader (Loss)
    Y-axis: Overall Win Rate (fraction of rounds in top 20%) or Average Weight
    Size of bubble: Could be number of rounds participated or final EMA.
    """
    if df_analysis.empty:
        print("Analysis DataFrame is empty, skipping winrate vs gap plot.")
        return

    summary = df_analysis.groupby("uid").agg(
        avg_gap_from_leader=("gap_from_leader_loss", "mean"),
        overall_win_rate=("is_winner_this_round", "mean"), # Fraction of rounds won
        avg_weight=("weight", "mean"),
        final_ema_loss=("ema_loss", "last") # Proxy for final standing
    ).reset_index()

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        summary["avg_gap_from_leader"],
        summary["overall_win_rate"], # Using win rate on Y axis
        # summary["avg_weight"], # Alternative: use average weight on Y axis
        s= (1 / (summary["final_ema_loss"] + 0.01)) * 2000, # Size by inverse of final loss (better = bigger)
        alpha=0.6, 
        cmap="viridis",
        c=summary["final_ema_loss"] # Color by final loss
    )

    # Highlight a specific UID if provided
    if uid_to_highlight is not None and uid_to_highlight in summary["uid"].values:
        highlight_data = summary[summary["uid"] == uid_to_highlight]
        plt.scatter(
            highlight_data["avg_gap_from_leader"],
            highlight_data["overall_win_rate"],
            s= (1 / (highlight_data["final_ema_loss"] + 0.01)) * 2000,
            color='red', 
            edgecolor='black', 
            label=f"UID {uid_to_highlight}",
            zorder=5
        )

    for i, row in summary.iterrows():
        plt.text(row["avg_gap_from_leader"], row["overall_win_rate"], str(int(row["uid"])))

    plt.xlabel("Average Gap from Leader (Loss Difference)")
    plt.ylabel("Overall Win Rate (Fraction of Rounds in Top 20% by Loss)")
    # plt.ylabel("Average Weight Received") # If using avg_weight for Y-axis
    plt.title("Miner Performance: Win Rate vs. Gap from Leader")
    plt.colorbar(scatter, label="Final EMA Loss (Lower is Better)")
    plt.grid(True)
    plt.axvline(0, color='grey', linestyle='--', label="Leader Line (Gap=0)")
    plt.legend()
    plt.tight_layout()
    # plt.savefig("winrate_vs_gap.png")
    plt.show()

def plot_leaderboard_evolution(df: pd.DataFrame, top_n: int = 5):
    """Plots how the ranks of the top N miners change over rounds."""
    if df.empty:
        print("DataFrame is empty, skipping leaderboard evolution plot.")
        return
    
    # Create a pivot table: rounds as index, UIDs as columns, EMA loss as values
    pivot_df = df.pivot(index='round', columns='uid', values='ema_loss')
    
    # Calculate ranks for each round (lower EMA loss gets better rank)
    rank_df = pivot_df.rank(axis=1, method='min', ascending=True)

    plt.figure(figsize=(12, 6))
    # Find overall top N UIDs based on their average rank or final EMA
    final_losses = df.groupby("uid")["ema_loss"].last().nsmallest(top_n*2).index # Look at a slightly larger pool
    uids_to_plot = df[df["uid"].isin(final_losses)].groupby("uid")["ema_loss"].mean().nsmallest(top_n).index

    for uid in uids_to_plot:
        if uid in rank_df.columns:
            plt.plot(rank_df.index, rank_df[uid], label=f'UID {uid}')

    plt.xlabel("Round")
    plt.ylabel(f"Rank (among all {len(df['uid'].unique())} miners)")
    plt.title(f"Leaderboard Evolution (Top {top_n} Miners by Avg EMA Loss)")
    plt.gca().invert_yaxis() # Lower rank (e.g., 1st) is better
    plt.legend(title="UID")
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("leaderboard_evolution.png")
    plt.show()


# --- Main execution for stand-alone script ---
if __name__ == "__main__":
    print("Running Epochor Simulation Visualizer Script...")
    # This would be where you load data from your actual simulation run.
    # E.g., from the output of `tests/test_scoring.py` if it saves results.
    simulation_df = load_simulation_data() # Loads mock data if file not found

    if not simulation_df.empty:
        # 1. Plot EMA Score (Loss) Evolution
        plot_ema_scores_evolution(simulation_df, top_n_uids=5)

        # 2. Calculate Win Rate and Gap from Leader for further analysis
        analysis_df = calculate_win_rate_and_gap(simulation_df)
        
        # 3. Plot Win Rate vs. Gap from Leader
        # Highlight a UID if desired, e.g., the one with the best final EMA
        if not analysis_df.empty:
            best_final_uid = analysis_df.loc[analysis_df.groupby('uid')['ema_loss'].last().idxmin(), 'uid']
            plot_winrate_vs_gap(analysis_df, uid_to_highlight=int(best_final_uid) if pd.notna(best_final_uid) else None)
        else:
            print("Analysis dataframe is empty, cannot plot winrate vs gap.")

        # 4. Plot Leaderboard Evolution
        plot_leaderboard_evolution(simulation_df, top_n=5)
        
        print("Visualizations complete. If plots are not showing, ensure you are in a GUI environment or check savefig calls.")
    else:
        print("No simulation data loaded or generated. Exiting.")

    # To make this a true notebook experience, you would run these cells
    # interactively in a Jupyter environment.
