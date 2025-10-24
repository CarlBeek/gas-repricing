"""
Analyze the divergence uptick around block 23610410
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
print("ANALYZING DIVERGENCE UPTICK AROUND BLOCK 23610410")
print("=" * 80)
print()

# 1. Get divergence counts around the uptick
print("1. Divergences per block around the uptick...")
query = """
SELECT
    block_number,
    COUNT(*) as divergence_count,
    AVG(gas_efficiency_ratio) as avg_efficiency_ratio,
    SUM(CASE WHEN oog_occurred = 1 THEN 1 ELSE 0 END) as oog_count,
    MIN(timestamp) as timestamp
FROM divergences
WHERE block_number BETWEEN 23610400 AND 23610420
GROUP BY block_number
ORDER BY block_number
"""
df_blocks = pd.read_sql_query(query, conn)
df_blocks['datetime'] = pd.to_datetime(df_blocks['timestamp'], unit='s')
print(df_blocks.to_string(index=False))
print()

# 2. Find the spike block
spike_block = df_blocks.loc[df_blocks['divergence_count'].idxmax(), 'block_number']
spike_count = df_blocks.loc[df_blocks['divergence_count'].idxmax(), 'divergence_count']
print(f"Spike detected at block {spike_block} with {spike_count:,} divergences")
print()

# 3. Get top contracts in the spike block
print("3. Top diverging contracts in spike block...")
query = f"""
SELECT
    HEX(divergence_contract) as contract,
    HEX(divergence_function_selector) as function_selector,
    COUNT(*) as count,
    AVG(gas_efficiency_ratio) as avg_efficiency_ratio,
    SUM(CASE WHEN oog_occurred = 1 THEN 1 ELSE 0 END) as oog_count,
    AVG(normal_gas_used) as avg_normal_gas,
    AVG(experimental_gas_used) as avg_exp_gas,
    GROUP_CONCAT(DISTINCT divergence_types) as divergence_types
FROM divergences
WHERE block_number = {spike_block}
    AND divergence_contract IS NOT NULL
GROUP BY divergence_contract, divergence_function_selector
ORDER BY count DESC
LIMIT 20
"""
df_contracts = pd.read_sql_query(query, conn)
print(df_contracts.to_string(index=False))
print()

# 4. Get specific transaction details for top contract
if not df_contracts.empty:
    top_contract = df_contracts.iloc[0]['contract']
    top_function = df_contracts.iloc[0]['function_selector']

    print(f"4. Details for top contract {top_contract[:10]}... function {top_function}...")
    query = f"""
    SELECT
        HEX(tx_hash) as tx_hash,
        tx_index,
        divergence_types,
        normal_gas_used,
        experimental_gas_used,
        gas_efficiency_ratio,
        normal_sload_count,
        normal_sstore_count,
        normal_call_count,
        oog_occurred,
        oog_opcode_name,
        oog_pattern
    FROM divergences
    WHERE block_number = {spike_block}
        AND HEX(divergence_contract) = ?
        AND HEX(divergence_function_selector) = ?
    LIMIT 10
    """
    df_txs = pd.read_sql_query(query, conn, params=(top_contract, top_function))
    print(df_txs.to_string(index=False))
    print()

# 5. Compare operation patterns before/after spike
print("5. Operation patterns comparison...")
query = f"""
SELECT
    CASE
        WHEN block_number < {spike_block} THEN 'Before Spike'
        WHEN block_number = {spike_block} THEN 'During Spike'
        ELSE 'After Spike'
    END as period,
    COUNT(*) as divergence_count,
    AVG(normal_sload_count) as avg_sload,
    AVG(normal_sstore_count) as avg_sstore,
    AVG(normal_call_count) as avg_call,
    AVG(normal_gas_used) as avg_gas,
    AVG(gas_efficiency_ratio) as avg_efficiency_ratio,
    SUM(CASE WHEN oog_occurred = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as oog_pct
FROM divergences
WHERE block_number BETWEEN 23610400 AND 23610420
GROUP BY period
ORDER BY
    CASE period
        WHEN 'Before Spike' THEN 1
        WHEN 'During Spike' THEN 2
        ELSE 3
    END
"""
df_comparison = pd.read_sql_query(query, conn)
print(df_comparison.to_string(index=False))
print()

# 6. Check divergence type distribution in spike
print("6. Divergence types in spike block...")
query = f"""
SELECT
    divergence_types,
    COUNT(*) as count,
    AVG(gas_efficiency_ratio) as avg_efficiency_ratio,
    SUM(CASE WHEN oog_occurred = 1 THEN 1 ELSE 0 END) as oog_count
FROM divergences
WHERE block_number = {spike_block}
GROUP BY divergence_types
ORDER BY count DESC
LIMIT 10
"""
df_div_types = pd.read_sql_query(query, conn)
print(df_div_types.to_string(index=False))
print()

# 7. Get OOG pattern distribution in spike
print("7. OOG patterns in spike block...")
query = f"""
SELECT
    oog_pattern,
    oog_opcode_name,
    COUNT(*) as count
FROM divergences
WHERE block_number = {spike_block}
    AND oog_occurred = 1
GROUP BY oog_pattern, oog_opcode_name
ORDER BY count DESC
LIMIT 10
"""
df_oog = pd.read_sql_query(query, conn)
if not df_oog.empty:
    print(df_oog.to_string(index=False))
else:
    print("No OOG events in spike block")
print()

# 8. Visualizations
print("8. Creating visualizations...")

# Plot 1: Divergences over time with spike highlighted
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax1 = axes[0]
ax1.bar(df_blocks['block_number'], df_blocks['divergence_count'],
        color=['red' if b == spike_block else 'steelblue' for b in df_blocks['block_number']],
        alpha=0.7)
ax1.set_xlabel('Block Number', fontsize=12)
ax1.set_ylabel('Divergence Count', fontsize=12)
ax1.set_title(f'Divergences Around Block {spike_block}\n(Spike highlighted in red)',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.tick_params(axis='x', rotation=45)

# Plot 2: OOG percentage
ax2 = axes[1]
df_blocks['oog_pct'] = (df_blocks['oog_count'] / df_blocks['divergence_count']) * 100
ax2.plot(df_blocks['block_number'], df_blocks['oog_pct'],
         marker='o', linewidth=2, markersize=6)
ax2.axvline(x=spike_block, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Block Number', fontsize=12)
ax2.set_ylabel('OOG Percentage (%)', fontsize=12)
ax2.set_title('Out-of-Gas Percentage by Block', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_dir = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f'uptick_analysis_block_{spike_block}.png')
plt.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"  Saved visualization to: {output_file}")
print()

# Plot 3: Top contracts breakdown
if not df_contracts.empty:
    fig, ax = plt.subplots(figsize=(14, 8))
    top_10 = df_contracts.head(10).copy()
    top_10['label'] = top_10['contract'].str[:10] + '...'

    ax.barh(range(len(top_10)), top_10['count'], alpha=0.7, color='coral')
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10['label'])
    ax.set_xlabel('Divergence Count', fontsize=12)
    ax.set_ylabel('Contract Address', fontsize=12)
    ax.set_title(f'Top 10 Contracts in Block {spike_block}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'top_contracts_block_{spike_block}.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"  Saved contract breakdown to: {output_file}")
    print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print("Summary:")
print(f"- Spike occurred at block {spike_block} ({df_blocks[df_blocks['block_number']==spike_block]['datetime'].iloc[0]})")
print(f"- {spike_count:,} divergences (vs typical ~{df_blocks[df_blocks['block_number']!=spike_block]['divergence_count'].median():.0f})")
if not df_contracts.empty:
    print(f"- Top contract: {top_contract}")
    print(f"- Top function: {top_function}")
    print(f"- This contract accounts for {df_contracts.iloc[0]['count']:,} divergences ({df_contracts.iloc[0]['count']/spike_count*100:.1f}%)")

conn.close()
