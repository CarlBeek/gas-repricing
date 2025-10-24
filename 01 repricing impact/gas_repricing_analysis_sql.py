"""
Gas Repricing Analysis: SQL-Based Approach
===========================================

This script analyzes divergence data using SQL queries instead of loading
the entire dataset into memory. Designed to work with large databases (1GB+).

Usage:
  python gas_repricing_analysis_sql.py                    # Run analysis
  python gas_repricing_analysis_sql.py --recover          # Recover corrupted DB first, then analyze
  python gas_repricing_analysis_sql.py --recover-only     # Only recover DB, don't analyze
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import sys
import os
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 6)

# Database paths - handle being run from different directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

DB_PATH_ORIGINAL = os.path.join(DATA_DIR, 'divergences.db')
DB_PATH_RECOVERED = os.path.join(DATA_DIR, 'divergences_recovered.db')
DB_PATH = DB_PATH_RECOVERED  # Default to recovered DB


class GasRepricingAnalyzer:
    """Analyzer for gas repricing divergence data using SQL queries."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def query(self, sql, params=None):
        """Execute a SQL query and return results as DataFrame."""
        return pd.read_sql_query(sql, self.conn, params=params)

    def execute(self, sql, params=None):
        """Execute a SQL command without returning results."""
        cursor = self.conn.cursor()
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        self.conn.commit()
        return cursor

    # =========================================================================
    # Summary Statistics
    # =========================================================================

    def get_basic_stats(self):
        """Get basic statistics about the dataset."""
        query = """
        SELECT
            COUNT(*) as total_divergences,
            MIN(block_number) as min_block,
            MAX(block_number) as max_block,
            COUNT(DISTINCT block_number) as unique_blocks,
            MIN(timestamp) as min_timestamp,
            MAX(timestamp) as max_timestamp,
            SUM(CASE WHEN oog_occurred = 1 THEN 1 ELSE 0 END) as oog_count
        FROM divergences
        """
        return self.query(query).iloc[0].to_dict()

    def get_divergence_type_counts(self):
        """Get counts for each divergence type."""
        query = """
        SELECT
            SUM(CASE WHEN divergence_types LIKE '%status%' THEN 1 ELSE 0 END) as status_count,
            SUM(CASE WHEN divergence_types LIKE '%gas_pattern%' THEN 1 ELSE 0 END) as gas_pattern_count,
            SUM(CASE WHEN divergence_types LIKE '%state_root%' THEN 1 ELSE 0 END) as state_root_count,
            SUM(CASE WHEN divergence_types LIKE '%logs%' THEN 1 ELSE 0 END) as logs_count,
            SUM(CASE WHEN divergence_types LIKE '%return_data%' THEN 1 ELSE 0 END) as return_data_count,
            COUNT(*) as total
        FROM divergences
        """
        return self.query(query).iloc[0].to_dict()

    def get_gas_stats(self):
        """Get gas usage statistics."""
        query = """
        SELECT
            AVG(normal_gas_used) as avg_normal_gas,
            AVG(experimental_gas_used) as avg_exp_gas,
            AVG(gas_efficiency_ratio) as avg_efficiency_ratio,
            MIN(gas_efficiency_ratio) as min_efficiency_ratio,
            MAX(gas_efficiency_ratio) as max_efficiency_ratio,
            AVG(experimental_gas_used - normal_gas_used) as avg_gas_difference
        FROM divergences
        """
        return self.query(query).iloc[0].to_dict()

    def get_gas_efficiency_percentiles(self, percentiles=[10, 25, 50, 75, 90, 95, 99]):
        """Get percentiles for gas efficiency ratio."""
        # SQLite doesn't have a built-in percentile function, so we calculate it manually
        results = {}
        for p in percentiles:
            query = f"""
            SELECT gas_efficiency_ratio
            FROM divergences
            ORDER BY gas_efficiency_ratio
            LIMIT 1 OFFSET (
                SELECT CAST(COUNT(*) * {p/100.0} AS INTEGER)
                FROM divergences
            )
            """
            result = self.query(query)
            results[f'p{p}'] = result.iloc[0, 0] if not result.empty else None
        return results

    def get_gas_efficiency_distribution(self, bins=100):
        """Get gas efficiency ratio distribution for histogram."""
        query = f"""
        SELECT
            gas_efficiency_ratio,
            COUNT(*) as count
        FROM divergences
        GROUP BY CAST(gas_efficiency_ratio * {bins} AS INTEGER)
        ORDER BY gas_efficiency_ratio
        """
        return self.query(query)

    # =========================================================================
    # OOG Analysis
    # =========================================================================

    def get_oog_pattern_counts(self):
        """Get OOG pattern distribution."""
        query = """
        SELECT
            oog_pattern,
            COUNT(*) as count
        FROM divergences
        WHERE oog_occurred = 1
        GROUP BY oog_pattern
        ORDER BY count DESC
        """
        return self.query(query)

    def get_oog_opcode_counts(self, limit=20):
        """Get top opcodes at OOG point."""
        query = f"""
        SELECT
            oog_opcode_name,
            COUNT(*) as count
        FROM divergences
        WHERE oog_occurred = 1 AND oog_opcode_name IS NOT NULL
        GROUP BY oog_opcode_name
        ORDER BY count DESC
        LIMIT {limit}
        """
        return self.query(query)

    def get_oog_comparison(self):
        """Compare operation patterns between OOG and non-OOG transactions."""
        query = """
        SELECT
            oog_occurred,
            AVG(normal_sload_count) as avg_sload,
            AVG(normal_sstore_count) as avg_sstore,
            AVG(normal_call_count) as avg_call,
            AVG(normal_log_count) as avg_log,
            AVG(normal_gas_used) as avg_gas
        FROM divergences
        GROUP BY oog_occurred
        """
        return self.query(query)

    # =========================================================================
    # Block and Time Analysis
    # =========================================================================

    def get_divergences_per_block(self):
        """Get divergence counts per block."""
        query = """
        SELECT
            block_number,
            COUNT(*) as divergence_count
        FROM divergences
        GROUP BY block_number
        ORDER BY block_number
        """
        return self.query(query)

    def get_divergence_type_by_block(self):
        """Get divergence type counts aggregated by block."""
        query = """
        SELECT
            block_number,
            SUM(CASE WHEN divergence_types LIKE '%status%' THEN 1 ELSE 0 END) as status_count,
            SUM(CASE WHEN divergence_types LIKE '%gas_pattern%' THEN 1 ELSE 0 END) as gas_pattern_count,
            SUM(CASE WHEN divergence_types LIKE '%state_root%' THEN 1 ELSE 0 END) as state_root_count,
            SUM(CASE WHEN divergence_types LIKE '%logs%' THEN 1 ELSE 0 END) as logs_count,
            SUM(CASE WHEN oog_occurred = 1 THEN 1 ELSE 0 END) as oog_count
        FROM divergences
        GROUP BY block_number
        ORDER BY block_number
        """
        return self.query(query)

    # =========================================================================
    # Sampling for Visualization
    # =========================================================================

    def get_sample_for_scatter(self, sample_size=10000, seed=42):
        """Get a random sample for scatter plots."""
        query = f"""
        SELECT
            normal_gas_used,
            experimental_gas_used,
            gas_efficiency_ratio,
            oog_occurred,
            divergence_types
        FROM divergences
        ORDER BY RANDOM()
        LIMIT {sample_size}
        """
        return self.query(query)

    def get_gas_efficiency_by_divergence_type(self):
        """Get gas efficiency ratios grouped by divergence type for box plots."""
        results = {}

        for dtype in ['status', 'gas_pattern', 'state_root', 'logs', 'return_data']:
            query = f"""
            SELECT gas_efficiency_ratio
            FROM divergences
            WHERE divergence_types LIKE '%{dtype}%'
            ORDER BY RANDOM()
            LIMIT 50000
            """
            df = self.query(query)
            if not df.empty:
                results[dtype] = df['gas_efficiency_ratio'].values

        return results

    # =========================================================================
    # Advanced Queries
    # =========================================================================

    def get_top_diverging_contracts(self, limit=20):
        """Get contracts with most divergences."""
        query = f"""
        SELECT
            HEX(divergence_contract) as contract,
            COUNT(*) as divergence_count,
            AVG(gas_efficiency_ratio) as avg_efficiency_ratio,
            SUM(CASE WHEN oog_occurred = 1 THEN 1 ELSE 0 END) as oog_count
        FROM divergences
        WHERE divergence_contract IS NOT NULL
        GROUP BY divergence_contract
        ORDER BY divergence_count DESC
        LIMIT {limit}
        """
        return self.query(query)

    def get_gas_usage_by_operation_counts(self):
        """Analyze gas usage based on operation counts."""
        query = """
        SELECT
            CASE
                WHEN normal_sload_count < 10 THEN '0-9'
                WHEN normal_sload_count < 50 THEN '10-49'
                WHEN normal_sload_count < 100 THEN '50-99'
                WHEN normal_sload_count < 500 THEN '100-499'
                ELSE '500+'
            END as sload_bucket,
            COUNT(*) as count,
            AVG(gas_efficiency_ratio) as avg_efficiency_ratio,
            AVG(normal_gas_used) as avg_gas_used
        FROM divergences
        WHERE normal_sload_count IS NOT NULL
        GROUP BY sload_bucket
        ORDER BY
            CASE sload_bucket
                WHEN '0-9' THEN 1
                WHEN '10-49' THEN 2
                WHEN '50-99' THEN 3
                WHEN '100-499' THEN 4
                ELSE 5
            END
        """
        return self.query(query)

    def get_simple_transfers(self):
        """Get statistics on simple transfers (21000 gas)."""
        query = """
        SELECT
            COUNT(*) as count,
            AVG(gas_efficiency_ratio) as avg_efficiency_ratio
        FROM divergences
        WHERE normal_gas_used <= 21000
        """
        return self.query(query).iloc[0].to_dict()

    def get_efficiency_ratio_tolerance_bands(self):
        """Count transactions within various tolerance bands around 1.0."""
        query = """
        SELECT
            SUM(CASE WHEN gas_efficiency_ratio BETWEEN 0.95 AND 1.05 THEN 1 ELSE 0 END) as within_5pct,
            SUM(CASE WHEN gas_efficiency_ratio BETWEEN 0.90 AND 1.10 THEN 1 ELSE 0 END) as within_10pct,
            SUM(CASE WHEN gas_efficiency_ratio BETWEEN 0.85 AND 1.15 THEN 1 ELSE 0 END) as within_15pct,
            SUM(CASE WHEN gas_efficiency_ratio < 0.85 OR gas_efficiency_ratio > 1.15 THEN 1 ELSE 0 END) as outside_15pct,
            COUNT(*) as total
        FROM divergences
        """
        return self.query(query).iloc[0].to_dict()


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_divergence_type_distribution(analyzer, save_path=None):
    """Plot divergence type distribution."""
    type_counts = analyzer.get_divergence_type_counts()

    labels = []
    counts = []
    for dtype in ['status', 'gas_pattern', 'state_root', 'logs', 'return_data']:
        count = type_counts[f'{dtype}_count']
        if count > 0:
            labels.append(dtype.replace('_', ' ').title())
            counts.append(count)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", len(labels))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('Divergence Type Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved to: {save_path}")

    return fig


def plot_gas_efficiency_histogram(analyzer, bins=100, save_path=None):
    """Plot gas efficiency ratio histogram."""
    # For large datasets, we'll use SQL aggregation
    query = f"""
    SELECT gas_efficiency_ratio
    FROM divergences
    ORDER BY RANDOM()
    LIMIT 100000
    """
    df = analyzer.query(query)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(df['gas_efficiency_ratio'], bins=bins, edgecolor='black', alpha=0.7)

    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Expected (same path)')
    ax.axvline(x=0.95, color='orange', linestyle=':', linewidth=1.5, label='±5% threshold')
    ax.axvline(x=1.05, color='orange', linestyle=':', linewidth=1.5)

    ax.set_yscale('log')
    ax.set_xlabel('Gas Efficiency Ratio', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title('Distribution of Gas Efficiency Ratios\n(Ratio ~1.0 = same execution path, ≠1.0 = different path)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved to: {save_path}")

    return fig


def plot_oog_patterns(analyzer, save_path=None):
    """Plot OOG pattern distribution."""
    oog_patterns = analyzer.get_oog_pattern_counts()

    if oog_patterns.empty:
        print("No OOG data to plot")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Pie chart
    colors = sns.color_palette("Set2", len(oog_patterns))
    ax1.pie(oog_patterns['count'], labels=oog_patterns['oog_pattern'],
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('OOG Pattern Distribution', fontsize=14, fontweight='bold')

    # Bar chart
    oog_patterns.plot(x='oog_pattern', y='count', kind='bar', ax=ax2,
                     color=colors, alpha=0.7, legend=False)
    ax2.set_xlabel('OOG Pattern', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('OOG Pattern Counts', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved to: {save_path}")

    return fig


def plot_oog_opcodes(analyzer, limit=20, save_path=None):
    """Plot top opcodes at OOG point."""
    oog_opcodes = analyzer.get_oog_opcode_counts(limit)

    if oog_opcodes.empty:
        print("No OOG opcode data to plot")
        return None

    fig, ax = plt.subplots(figsize=(14, 8))
    oog_opcodes.plot(x='oog_opcode_name', y='count', kind='barh',
                     ax=ax, color='coral', alpha=0.7, legend=False)
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Opcode', fontsize=12)
    ax.set_title(f'Top {limit} Opcodes at OOG Point', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved to: {save_path}")

    return fig


def plot_divergences_timeline(analyzer, save_path=None):
    """Plot divergences over time."""
    df = analyzer.get_divergences_per_block()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['block_number'], df['divergence_count'], linewidth=1, alpha=0.7)
    ax.set_xlabel('Block Number', fontsize=12)
    ax.set_ylabel('Divergences per Block', fontsize=12)
    ax.set_title('Divergences Over Time (by Block)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved to: {save_path}")

    return fig


def print_summary_report(analyzer):
    """Print a comprehensive summary report."""
    print("=" * 80)
    print("GAS REPRICING ANALYSIS - SUMMARY REPORT")
    print("=" * 80)
    print()

    # Basic stats
    stats = analyzer.get_basic_stats()
    print(f"Total Divergences: {stats['total_divergences']:,}")
    print(f"Block Range: {stats['min_block']:,} to {stats['max_block']:,}")
    print(f"Unique Blocks: {stats['unique_blocks']:,}")

    # Time range
    min_date = datetime.fromtimestamp(stats['min_timestamp'])
    max_date = datetime.fromtimestamp(stats['max_timestamp'])
    print(f"Date Range: {min_date} to {max_date}")
    print(f"Time Span: {(max_date - min_date).days} days")
    print()

    # OOG stats
    oog_pct = (stats['oog_count'] / stats['total_divergences']) * 100
    print(f"Out-of-Gas Occurrences: {stats['oog_count']:,} ({oog_pct:.2f}%)")
    print()

    # Divergence types
    print("=" * 80)
    print("DIVERGENCE TYPE BREAKDOWN")
    print("=" * 80)
    type_counts = analyzer.get_divergence_type_counts()
    total = type_counts['total']

    for dtype in ['status', 'gas_pattern', 'state_root', 'logs', 'return_data']:
        count = type_counts[f'{dtype}_count']
        pct = (count / total) * 100
        print(f"{dtype:20s}: {count:,} ({pct:5.2f}%)")
    print()

    # Gas stats
    print("=" * 80)
    print("GAS STATISTICS")
    print("=" * 80)
    gas_stats = analyzer.get_gas_stats()
    print(f"Average Normal Gas Used: {gas_stats['avg_normal_gas']:,.0f}")
    print(f"Average Experimental Gas Used: {gas_stats['avg_exp_gas']:,.0f}")
    print(f"Average Gas Difference: {gas_stats['avg_gas_difference']:,.0f}")
    print()
    print(f"Gas Efficiency Ratio:")
    print(f"  Mean:   {gas_stats['avg_efficiency_ratio']:.4f}")
    print(f"  Min:    {gas_stats['min_efficiency_ratio']:.4f}")
    print(f"  Max:    {gas_stats['max_efficiency_ratio']:.4f}")
    print()

    # Tolerance bands
    tolerance = analyzer.get_efficiency_ratio_tolerance_bands()
    print("Efficiency Ratio Tolerance Bands:")
    print(f"  Within ±5%:   {tolerance['within_5pct']:,} ({tolerance['within_5pct']/tolerance['total']*100:.2f}%)")
    print(f"  Within ±10%:  {tolerance['within_10pct']:,} ({tolerance['within_10pct']/tolerance['total']*100:.2f}%)")
    print(f"  Within ±15%:  {tolerance['within_15pct']:,} ({tolerance['within_15pct']/tolerance['total']*100:.2f}%)")
    print(f"  Outside ±15%: {tolerance['outside_15pct']:,} ({tolerance['outside_15pct']/tolerance['total']*100:.2f}%)")
    print()

    # Simple transfers
    simple = analyzer.get_simple_transfers()
    simple_pct = (simple['count'] / total) * 100
    print(f"Simple Transfers (≤21000 gas): {simple['count']:,} ({simple_pct:.2f}%)")
    print()


# ============================================================================
# Database Recovery
# ============================================================================

def check_db_integrity(db_path):
    """Check if database is corrupted."""
    if not os.path.exists(db_path):
        return False, "Database file does not exist"

    try:
        result = subprocess.run(
            ['sqlite3', db_path, 'PRAGMA integrity_check;'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return False, f"Error running integrity check: {result.stderr}"

        output = result.stdout.strip()
        if output == "ok":
            return True, "Database integrity OK"
        else:
            # Count number of error lines
            error_lines = [line for line in output.split('\n') if 'invalid' in line.lower() or 'corrupt' in line.lower()]
            return False, f"Database corrupted ({len(error_lines)} errors found)"

    except subprocess.TimeoutExpired:
        return False, "Integrity check timed out"
    except Exception as e:
        return False, f"Error checking integrity: {e}"


def recover_database(source_path, dest_path, backup_old=True):
    """Recover a corrupted SQLite database."""
    print("=" * 80)
    print("DATABASE RECOVERY")
    print("=" * 80)
    print()

    # Check if source exists
    if not os.path.exists(source_path):
        print(f"ERROR: Source database does not exist: {source_path}")
        return False

    # Check source integrity
    print(f"Checking integrity of source database: {source_path}")
    is_ok, message = check_db_integrity(source_path)
    print(f"  {message}")

    if is_ok:
        print("\nSource database is not corrupted. No recovery needed.")
        return True

    print("\nStarting recovery process...")
    print(f"  Source: {source_path}")
    print(f"  Destination: {dest_path}")
    print()

    # Backup existing recovered database if it exists
    if backup_old and os.path.exists(dest_path):
        backup_path = f"{dest_path}.backup"
        print(f"Backing up existing recovered database to: {backup_path}")
        try:
            os.rename(dest_path, backup_path)
            print("  Backup complete")
        except Exception as e:
            print(f"  WARNING: Could not backup existing database: {e}")

    # Remove destination if it exists (and we didn't back it up)
    if os.path.exists(dest_path):
        try:
            os.remove(dest_path)
        except Exception as e:
            print(f"ERROR: Could not remove existing destination: {e}")
            return False

    # Run recovery
    print("\nRunning SQLite recovery (this may take a few minutes)...")
    try:
        # Use .recover command to extract data from corrupted DB
        recover_cmd = f"sqlite3 '{source_path}' '.recover'"
        load_cmd = f"sqlite3 '{dest_path}'"

        # Chain the commands: recover | load into new DB
        full_cmd = f"{recover_cmd} | {load_cmd}"

        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"ERROR: Recovery failed: {result.stderr}")
            return False

        print("  Recovery complete")

    except subprocess.TimeoutExpired:
        print("ERROR: Recovery timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"ERROR: Recovery failed: {e}")
        return False

    # Check if recovered database exists and get stats
    if not os.path.exists(dest_path):
        print("ERROR: Recovered database was not created")
        return False

    # Get file size
    size_bytes = os.path.getsize(dest_path)
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)

    print(f"\nRecovered database created successfully")
    print(f"  Size: {size_gb:.2f} GB ({size_mb:.0f} MB)")

    # Check integrity of recovered database
    print("\nChecking integrity of recovered database...")
    is_ok, message = check_db_integrity(dest_path)
    print(f"  {message}")

    if not is_ok:
        print("WARNING: Recovered database still has issues")
        return False

    # Count rows
    print("\nCounting rows in recovered database...")
    try:
        conn = sqlite3.connect(dest_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM divergences")
        row_count = cursor.fetchone()[0]
        conn.close()
        print(f"  Total rows: {row_count:,}")
    except Exception as e:
        print(f"  Could not count rows: {e}")

    print("\n" + "=" * 80)
    print("RECOVERY COMPLETE")
    print("=" * 80)
    print()

    return True


# ============================================================================
# Main Analysis
# ============================================================================

def run_analysis(db_path, save_plots=True, show_plots=False):
    """Run the main analysis."""
    print("Connecting to database...")

    # Setup output directory if saving plots
    output_dir = None
    if save_plots:
        output_dir = OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_subdir = os.path.join(output_dir, timestamp)
        os.makedirs(output_subdir, exist_ok=True)
        print(f"Plots will be saved to: {output_subdir}\n")
        output_dir = output_subdir

    with GasRepricingAnalyzer(db_path) as analyzer:
        # Print summary report
        print_summary_report(analyzer)

        # Generate visualizations
        print("=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        print("\n1. Plotting divergence type distribution...")
        save_path = os.path.join(output_dir, '01_divergence_type_distribution.png') if output_dir else None
        fig = plot_divergence_type_distribution(analyzer, save_path=save_path)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        print("2. Plotting gas efficiency histogram...")
        save_path = os.path.join(output_dir, '02_gas_efficiency_histogram.png') if output_dir else None
        fig = plot_gas_efficiency_histogram(analyzer, save_path=save_path)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        print("3. Plotting divergences timeline...")
        save_path = os.path.join(output_dir, '03_divergences_timeline.png') if output_dir else None
        fig = plot_divergences_timeline(analyzer, save_path=save_path)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        print("4. Plotting OOG patterns...")
        save_path = os.path.join(output_dir, '04_oog_patterns.png') if output_dir else None
        fig = plot_oog_patterns(analyzer, save_path=save_path)
        if fig:
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

        print("5. Plotting OOG opcodes...")
        save_path = os.path.join(output_dir, '05_oog_opcodes.png') if output_dir else None
        fig = plot_oog_opcodes(analyzer, save_path=save_path)
        if fig:
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        if output_dir:
            print(f"\nAll plots saved to: {output_dir}")
            print(f"Total files: {len([f for f in os.listdir(output_dir) if f.endswith('.png')])}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Gas Repricing Analysis - SQL-based analysis of divergence data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gas_repricing_analysis_sql.py                    # Run analysis, save plots to output/
  python gas_repricing_analysis_sql.py --show-plots       # Display plots interactively
  python gas_repricing_analysis_sql.py --no-save          # Don't save plots, just show
  python gas_repricing_analysis_sql.py --recover          # Recover DB first, then analyze
  python gas_repricing_analysis_sql.py --recover-only     # Only recover DB
  python gas_repricing_analysis_sql.py --check-integrity  # Check DB integrity only
        """
    )

    parser.add_argument('--recover', action='store_true',
                        help='Recover database from divergences.db before running analysis')
    parser.add_argument('--recover-only', action='store_true',
                        help='Only recover the database, do not run analysis')
    parser.add_argument('--check-integrity', action='store_true',
                        help='Check database integrity and exit')
    parser.add_argument('--source-db', default=DB_PATH_ORIGINAL,
                        help=f'Source database path (default: {DB_PATH_ORIGINAL})')
    parser.add_argument('--dest-db', default=DB_PATH_RECOVERED,
                        help=f'Destination database path (default: {DB_PATH_RECOVERED})')
    parser.add_argument('--show-plots', action='store_true',
                        help='Show plots interactively (default: False, just save)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plots to disk')

    args = parser.parse_args()

    # Check integrity only
    if args.check_integrity:
        print("Checking database integrity...")
        is_ok, message = check_db_integrity(args.dest_db)
        print(f"{args.dest_db}: {message}")
        sys.exit(0 if is_ok else 1)

    # Recover database if requested
    if args.recover or args.recover_only:
        success = recover_database(args.source_db, args.dest_db)
        if not success:
            print("\nERROR: Database recovery failed")
            sys.exit(1)

        if args.recover_only:
            sys.exit(0)

    # Run analysis
    db_path = args.dest_db

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found: {db_path}")
        print("\nTry running with --recover flag to recover the database first:")
        print(f"  python {sys.argv[0]} --recover")
        sys.exit(1)

    # Determine save/show behavior
    save_plots = not args.no_save
    show_plots = args.show_plots

    run_analysis(db_path, save_plots=save_plots, show_plots=show_plots)


if __name__ == "__main__":
    main()
