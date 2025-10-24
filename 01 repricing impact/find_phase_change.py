"""
Find the actual phase change where divergences suddenly started appearing
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DB_PATH = os.path.join(DATA_DIR, 'divergences_recovered.db')

conn = sqlite3.connect(DB_PATH)

print("=" * 80)
print("FINDING THE PHASE CHANGE IN DIVERGENCE PATTERN")
print("=" * 80)
print()

# 1. Get overall statistics
print("1. Overall dataset statistics...")
query = """
SELECT
    MIN(block_number) as min_block,
    MAX(block_number) as max_block,
    COUNT(*) as total_divergences,
    COUNT(DISTINCT block_number) as unique_blocks
FROM divergences
"""
stats = pd.read_sql_query(query, conn).iloc[0]
print(f"Block range: {stats['min_block']:,} to {stats['max_block']:,}")
print(f"Total divergences: {stats['total_divergences']:,}")
print(f"Unique blocks: {stats['unique_blocks']:,}")
print()

# 2. Get divergence counts in buckets of 100k blocks
print("2. Divergences by 100k block ranges...")
query = """
SELECT
    CAST(block_number / 100000 AS INTEGER) * 100000 as block_bucket,
    COUNT(*) as divergence_count,
    COUNT(DISTINCT block_number) as blocks_with_divergences,
    MIN(block_number) as min_block,
    MAX(block_number) as max_block,
    MIN(timestamp) as min_timestamp,
    MAX(timestamp) as max_timestamp
FROM divergences
GROUP BY block_bucket
ORDER BY block_bucket
"""
df_buckets = pd.read_sql_query(query, conn)
df_buckets['start_date'] = pd.to_datetime(df_buckets['min_timestamp'], unit='s')
df_buckets['end_date'] = pd.to_datetime(df_buckets['max_timestamp'], unit='s')
print(df_buckets.to_string(index=False))
print()

# 3. Find the exact transition - look at 10k block ranges around suspicious area
# Find the first bucket with significant activity
significant_buckets = df_buckets[df_buckets['divergence_count'] > 1000]
if not significant_buckets.empty:
    transition_bucket = significant_buckets.iloc[0]['block_bucket']
    print(f"3. First significant activity bucket: {transition_bucket:,}")

    # Zoom in with 10k block granularity
    search_start = max(stats['min_block'], transition_bucket - 100000)
    search_end = transition_bucket + 100000

    print(f"   Zooming in on blocks {search_start:,} to {search_end:,}...")
    query = f"""
    SELECT
        CAST(block_number / 10000 AS INTEGER) * 10000 as block_bucket,
        COUNT(*) as divergence_count,
        COUNT(DISTINCT block_number) as blocks_with_divergences,
        MIN(block_number) as min_block,
        MAX(block_number) as max_block,
        MIN(timestamp) as min_timestamp,
        MAX(timestamp) as max_timestamp
    FROM divergences
    WHERE block_number BETWEEN {search_start} AND {search_end}
    GROUP BY block_bucket
    ORDER BY block_bucket
    """
    df_zoom = pd.read_sql_query(query, conn)
    if not df_zoom.empty:
        df_zoom['start_date'] = pd.to_datetime(df_zoom['min_timestamp'], unit='s')
        print()
        print("   10k block granularity around transition:")
        print(df_zoom.to_string(index=False))
        print()

        # Find exact transition block (first block with divergence in significant bucket)
        first_significant = df_zoom[df_zoom['divergence_count'] > 100].iloc[0] if len(df_zoom[df_zoom['divergence_count'] > 100]) > 0 else df_zoom.iloc[0]
        transition_block_approx = first_significant['min_block']

        # Get exact first blocks
        print(f"4. Finding exact transition point around block {transition_block_approx:,}...")
        query = f"""
        SELECT
            block_number,
            COUNT(*) as divergence_count,
            MIN(timestamp) as timestamp,
            GROUP_CONCAT(DISTINCT HEX(divergence_contract)) as contracts,
            GROUP_CONCAT(DISTINCT divergence_types) as divergence_types
        FROM divergences
        WHERE block_number BETWEEN {transition_block_approx - 100} AND {transition_block_approx + 100}
        GROUP BY block_number
        ORDER BY block_number
        LIMIT 50
        """
        df_exact = pd.read_sql_query(query, conn)
        df_exact['datetime'] = pd.to_datetime(df_exact['timestamp'], unit='s')
        print(df_exact.to_string(index=False))
        print()

# 4. Check what changed at the transition
if not significant_buckets.empty:
    first_sig_block = df_exact.iloc[0]['block_number'] if not df_exact.empty else transition_block_approx

    print(f"5. What changed at block {first_sig_block:,}?")
    print()

    # Get sample transactions from before and after
    query = f"""
    SELECT
        'BEFORE' as period,
        AVG(normal_gas_used) as avg_gas,
        AVG(experimental_gas_used) as avg_exp_gas,
        AVG(gas_efficiency_ratio) as avg_efficiency,
        COUNT(DISTINCT HEX(divergence_contract)) as unique_contracts,
        SUM(CASE WHEN oog_occurred = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as oog_pct,
        GROUP_CONCAT(DISTINCT divergence_types) as divergence_types,
        COUNT(*) as count
    FROM divergences
    WHERE block_number < {first_sig_block}

    UNION ALL

    SELECT
        'AFTER' as period,
        AVG(normal_gas_used) as avg_gas,
        AVG(experimental_gas_used) as avg_exp_gas,
        AVG(gas_efficiency_ratio) as avg_efficiency,
        COUNT(DISTINCT HEX(divergence_contract)) as unique_contracts,
        SUM(CASE WHEN oog_occurred = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as oog_pct,
        GROUP_CONCAT(DISTINCT divergence_types) as divergence_types,
        COUNT(*) as count
    FROM divergences
    WHERE block_number >= {first_sig_block}
    """
    df_comparison = pd.read_sql_query(query, conn)
    print("Before vs After Comparison:")
    print(df_comparison.to_string(index=False))
    print()

# 5. Visualize the phase change
print("6. Creating visualization...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot 1: Log scale of divergences by 10k blocks (if we have zoom data)
if 'df_zoom' in locals() and not df_zoom.empty:
    ax1 = axes[0]
    ax1.bar(df_zoom['block_bucket'], df_zoom['divergence_count'],
            width=8000, alpha=0.7, color='steelblue')
    ax1.axvline(x=first_sig_block, color='red', linestyle='--', linewidth=2,
                label=f'Transition at block {first_sig_block:,}')
    ax1.set_xlabel('Block Number', fontsize=12)
    ax1.set_ylabel('Divergence Count', fontsize=12)
    ax1.set_title('Phase Change: Divergences by Block Range (10k granularity)',
                  fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Cumulative divergences
query = """
SELECT
    block_number,
    COUNT(*) as count
FROM divergences
GROUP BY block_number
ORDER BY block_number
"""
df_cumulative = pd.read_sql_query(query, conn)
df_cumulative['cumulative'] = df_cumulative['count'].cumsum()

ax2 = axes[1]
ax2.plot(df_cumulative['block_number'], df_cumulative['cumulative'],
         linewidth=2, color='darkblue')
if 'first_sig_block' in locals():
    ax2.axvline(x=first_sig_block, color='red', linestyle='--', linewidth=2,
                label=f'Transition at block {first_sig_block:,}')
ax2.set_xlabel('Block Number', fontsize=12)
ax2.set_ylabel('Cumulative Divergences', fontsize=12)
ax2.set_title('Cumulative Divergences Over Time', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_dir = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'phase_change_analysis.png')
plt.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"  Saved to: {output_file}")
print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

if 'first_sig_block' in locals() and 'df_exact' in locals() and not df_exact.empty:
    transition_date = df_exact.iloc[0]['datetime']
    print()
    print("SUMMARY:")
    print(f"- Phase change occurred at block {first_sig_block:,}")
    print(f"- Date: {transition_date}")
    print(f"- Before: {df_comparison[df_comparison['period']=='BEFORE']['count'].iloc[0]:,.0f} divergences")
    print(f"- After: {df_comparison[df_comparison['period']=='AFTER']['count'].iloc[0]:,.0f} divergences")
    print()
    print("This likely indicates when the experiment or data collection methodology changed.")

conn.close()
