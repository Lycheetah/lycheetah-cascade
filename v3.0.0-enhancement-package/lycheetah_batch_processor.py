"""
lycheetah_batch_processor.py
=============================
High-Throughput Batch Processing for CASCADE System

Process multiple documents efficiently with:
- Parallel execution
- Progress tracking
- Error handling
- Aggregate statistics
- CSV/JSON export

Author: Lycheetah × CASCADE
Version: 2.0.0
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from lycheetah_cascade_core import CascadeSignatureEngine, SignatureBlock


# ============================================================================
# BATCH DATA STRUCTURES
# ============================================================================

@dataclass
class BatchItem:
    """Single item in batch processing"""
    id: str
    content: str
    source: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result from processing single item"""
    item_id: str
    timestamp: datetime
    success: bool
    block: Optional[SignatureBlock] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        if self.success and self.block:
            return {
                'id': self.item_id,
                'timestamp': self.timestamp.isoformat(),
                'success': True,
                'lcs': round(self.block.lcs, 4),
                'truth_pressure': round(self.block.truth_pressure, 4),
                'authenticity': round(self.block.authenticity_score, 4),
                'is_sovereign': self.block.is_sovereign(),
                'trinity_vector': self.block.axiom_vector.tolist(),
                'processing_time': round(self.processing_time, 3)
            }
        else:
            return {
                'id': self.item_id,
                'timestamp': self.timestamp.isoformat(),
                'success': False,
                'error': self.error,
                'processing_time': round(self.processing_time, 3)
            }


@dataclass
class BatchSummary:
    """Summary statistics for batch processing"""
    total_items: int
    successful: int
    failed: int
    total_time: float
    avg_time_per_item: float
    sovereign_count: int
    sovereign_rate: float
    avg_lcs: float
    avg_truth_pressure: float
    avg_authenticity: float
    trinity_stats: Dict


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchProcessor:
    """
    Process multiple documents in batch
    
    Key Features:
    - Parallel execution for speed
    - Progress tracking
    - Robust error handling
    - Flexible input formats
    - Multiple output formats
    """
    
    def __init__(self,
                 engine: Optional[CascadeSignatureEngine] = None,
                 max_workers: int = 4):
        """
        Initialize batch processor
        
        Args:
            engine: Signature engine (creates new if None)
            max_workers: Number of parallel workers
        """
        self.engine = engine if engine else CascadeSignatureEngine()
        self.max_workers = max_workers
        
        # Processing state
        self.current_batch: Optional[List[BatchItem]] = None
        self.results: List[BatchResult] = []
        
    # ------------------------------------------------------------------------
    # INPUT LOADING
    # ------------------------------------------------------------------------
    
    def load_from_directory(self, 
                           directory: str,
                           file_pattern: str = "*.txt") -> List[BatchItem]:
        """
        Load batch items from directory
        
        Args:
            directory: Path to directory
            file_pattern: Glob pattern for files
        
        Returns:
            List of batch items
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        items = []
        for file_path in dir_path.glob(file_pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                items.append(BatchItem(
                    id=file_path.stem,
                    content=content,
                    source=str(file_path),
                    metadata={'filename': file_path.name}
                ))
            except Exception as e:
                print(f"⚠️  Failed to load {file_path}: {e}")
        
        print(f"✓ Loaded {len(items)} items from {directory}")
        return items
    
    def load_from_json(self, filepath: str) -> List[BatchItem]:
        """
        Load batch items from JSON file
        
        Expected format:
        [
            {"id": "doc1", "content": "...", "metadata": {}},
            ...
        ]
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        items = []
        for item_data in data:
            items.append(BatchItem(
                id=item_data['id'],
                content=item_data['content'],
                source=item_data.get('source'),
                metadata=item_data.get('metadata', {})
            ))
        
        print(f"✓ Loaded {len(items)} items from {filepath}")
        return items
    
    def load_from_csv(self, filepath: str, 
                     id_column: str = 'id',
                     content_column: str = 'content') -> List[BatchItem]:
        """Load batch items from CSV file"""
        items = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if id_column not in row or content_column not in row:
                    continue
                
                # Other columns become metadata
                metadata = {k: v for k, v in row.items() 
                          if k not in [id_column, content_column]}
                
                items.append(BatchItem(
                    id=row[id_column],
                    content=row[content_column],
                    metadata=metadata
                ))
        
        print(f"✓ Loaded {len(items)} items from {filepath}")
        return items
    
    # ------------------------------------------------------------------------
    # BATCH PROCESSING
    # ------------------------------------------------------------------------
    
    def process_batch(self,
                     items: List[BatchItem],
                     parallel: bool = True,
                     progress_callback: Optional[Callable] = None) -> List[BatchResult]:
        """
        Process batch of items
        
        Args:
            items: Items to process
            parallel: Use parallel execution
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of batch results
        """
        self.current_batch = items
        self.results = []
        
        print(f"\n{'='*70}")
        print(f"PROCESSING BATCH: {len(items)} items")
        print(f"Parallel: {parallel} (workers: {self.max_workers})")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        if parallel and self.max_workers > 1:
            self.results = self._process_parallel(items, progress_callback)
        else:
            self.results = self._process_sequential(items, progress_callback)
        
        total_time = time.time() - start_time
        
        print(f"\n✓ Batch processing complete")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average: {total_time/len(items):.3f}s per item")
        print(f"  Success: {sum(1 for r in self.results if r.success)}/{len(items)}")
        
        return self.results
    
    def _process_sequential(self,
                           items: List[BatchItem],
                           progress_callback: Optional[Callable]) -> List[BatchResult]:
        """Process items sequentially"""
        results = []
        
        for i, item in enumerate(items, 1):
            result = self._process_single_item(item)
            results.append(result)
            
            if progress_callback:
                progress_callback(i, len(items), result)
            
            if i % 10 == 0 or i == len(items):
                print(f"  Progress: {i}/{len(items)} ({i/len(items)*100:.1f}%)")
        
        return results
    
    def _process_parallel(self,
                         items: List[BatchItem],
                         progress_callback: Optional[Callable]) -> List[BatchResult]:
        """Process items in parallel"""
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._process_single_item, item): item
                for item in items
            }
            
            # Process as they complete
            for future in as_completed(future_to_item):
                result = future.result()
                results.append(result)
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(items), result)
                
                if completed % 10 == 0 or completed == len(items):
                    print(f"  Progress: {completed}/{len(items)} ({completed/len(items)*100:.1f}%)")
        
        return results
    
    def _process_single_item(self, item: BatchItem) -> BatchResult:
        """Process a single item"""
        start_time = time.time()
        
        try:
            # Verify signature
            block = self.engine.verify_provenance(item.content)
            
            processing_time = time.time() - start_time
            
            return BatchResult(
                item_id=item.id,
                timestamp=datetime.now(),
                success=True,
                block=block,
                processing_time=processing_time
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            
            return BatchResult(
                item_id=item.id,
                timestamp=datetime.now(),
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    # ------------------------------------------------------------------------
    # ANALYSIS & REPORTING
    # ------------------------------------------------------------------------
    
    def compute_summary(self) -> BatchSummary:
        """Compute summary statistics from results"""
        if not self.results:
            raise ValueError("No results to summarize")
        
        successful_results = [r for r in self.results if r.success and r.block]
        
        total_items = len(self.results)
        successful = len(successful_results)
        failed = total_items - successful
        
        total_time = sum(r.processing_time for r in self.results)
        avg_time = total_time / total_items if total_items > 0 else 0
        
        if successful_results:
            sovereign_count = sum(1 for r in successful_results if r.block.is_sovereign())
            sovereign_rate = sovereign_count / successful
            
            lcs_values = [r.block.lcs for r in successful_results]
            truth_values = [r.block.truth_pressure for r in successful_results]
            auth_values = [r.block.authenticity_score for r in successful_results]
            
            import numpy as np
            trinity_vectors = np.array([r.block.axiom_vector for r in successful_results])
            
            trinity_stats = {
                'protector_mean': float(np.mean(trinity_vectors[:, 0])),
                'healer_mean': float(np.mean(trinity_vectors[:, 1])),
                'beacon_mean': float(np.mean(trinity_vectors[:, 2])),
                'balance': float(np.mean([np.std(v) for v in trinity_vectors]))
            }
        else:
            sovereign_count = 0
            sovereign_rate = 0.0
            lcs_values = [0]
            truth_values = [0]
            auth_values = [0]
            trinity_stats = {}
        
        import numpy as np
        
        return BatchSummary(
            total_items=total_items,
            successful=successful,
            failed=failed,
            total_time=round(total_time, 2),
            avg_time_per_item=round(avg_time, 3),
            sovereign_count=sovereign_count,
            sovereign_rate=round(sovereign_rate, 3),
            avg_lcs=round(float(np.mean(lcs_values)), 3),
            avg_truth_pressure=round(float(np.mean(truth_values)), 3),
            avg_authenticity=round(float(np.mean(auth_values)), 3),
            trinity_stats=trinity_stats
        )
    
    def print_summary(self):
        """Print human-readable summary"""
        summary = self.compute_summary()
        
        print(f"\n{'='*70}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*70}\n")
        
        print("PROCESSING STATS:")
        print(f"  Total items: {summary.total_items}")
        print(f"  Successful: {summary.successful}")
        print(f"  Failed: {summary.failed}")
        print(f"  Total time: {summary.total_time:.2f}s")
        print(f"  Avg time/item: {summary.avg_time_per_item:.3f}s")
        
        print("\nSIGNATURE STATS:")
        print(f"  Sovereign: {summary.sovereign_count}/{summary.successful} ({summary.sovereign_rate:.1%})")
        print(f"  Avg LCS: {summary.avg_lcs:.3f}")
        print(f"  Avg Truth Pressure: {summary.avg_truth_pressure:.3f}")
        print(f"  Avg Authenticity: {summary.avg_authenticity:.3f}")
        
        if summary.trinity_stats:
            print("\nTRINITY STATS:")
            print(f"  Protector: {summary.trinity_stats['protector_mean']:.3f}")
            print(f"  Healer: {summary.trinity_stats['healer_mean']:.3f}")
            print(f"  Beacon: {summary.trinity_stats['beacon_mean']:.3f}")
            print(f"  Balance: {summary.trinity_stats['balance']:.3f}")
        
        print(f"\n{'='*70}\n")
    
    # ------------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------------
    
    def export_to_json(self, filepath: str):
        """Export results to JSON"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_items': len(self.results),
            'results': [r.to_dict() for r in self.results],
            'summary': self.compute_summary().__dict__
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✓ Exported results to {filepath}")
    
    def export_to_csv(self, filepath: str):
        """Export results to CSV"""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'id', 'timestamp', 'success', 'is_sovereign',
                'lcs', 'truth_pressure', 'authenticity',
                'protector', 'healer', 'beacon',
                'processing_time', 'error'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                if result.success and result.block:
                    row = {
                        'id': result.item_id,
                        'timestamp': result.timestamp.isoformat(),
                        'success': True,
                        'is_sovereign': result.block.is_sovereign(),
                        'lcs': round(result.block.lcs, 4),
                        'truth_pressure': round(result.block.truth_pressure, 4),
                        'authenticity': round(result.block.authenticity_score, 4),
                        'protector': round(result.block.axiom_vector[0], 3),
                        'healer': round(result.block.axiom_vector[1], 3),
                        'beacon': round(result.block.axiom_vector[2], 3),
                        'processing_time': round(result.processing_time, 3),
                        'error': ''
                    }
                else:
                    row = {
                        'id': result.item_id,
                        'timestamp': result.timestamp.isoformat(),
                        'success': False,
                        'is_sovereign': False,
                        'lcs': 0,
                        'truth_pressure': 0,
                        'authenticity': 0,
                        'protector': 0,
                        'healer': 0,
                        'beacon': 0,
                        'processing_time': round(result.processing_time, 3),
                        'error': result.error or ''
                    }
                
                writer.writerow(row)
        
        print(f"✓ Exported results to {filepath}")
    
    # ------------------------------------------------------------------------
    # FILTERING & QUERYING
    # ------------------------------------------------------------------------
    
    def filter_by_sovereignty(self, sovereign: bool = True) -> List[BatchResult]:
        """Filter results by sovereignty status"""
        return [r for r in self.results 
                if r.success and r.block and r.block.is_sovereign() == sovereign]
    
    def filter_by_lcs(self, min_lcs: float, max_lcs: float = 1.0) -> List[BatchResult]:
        """Filter results by LCS range"""
        return [r for r in self.results 
                if r.success and r.block and min_lcs <= r.block.lcs <= max_lcs]
    
    def get_top_n(self, n: int, metric: str = 'authenticity') -> List[BatchResult]:
        """Get top N results by metric"""
        successful = [r for r in self.results if r.success and r.block]
        
        if metric == 'authenticity':
            sorted_results = sorted(successful, 
                                  key=lambda r: r.block.authenticity_score, 
                                  reverse=True)
        elif metric == 'lcs':
            sorted_results = sorted(successful, 
                                  key=lambda r: r.block.lcs, 
                                  reverse=True)
        elif metric == 'truth_pressure':
            sorted_results = sorted(successful, 
                                  key=lambda r: r.block.truth_pressure, 
                                  reverse=True)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return sorted_results[:n]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def batch_verify_directory(directory: str,
                           output_file: str = "batch_results.json",
                           parallel: bool = True) -> BatchSummary:
    """
    Convenience function for quick batch processing
    
    Args:
        directory: Directory with text files
        output_file: Where to save results
        parallel: Use parallel processing
    
    Returns:
        Batch summary
    """
    processor = BatchProcessor()
    items = processor.load_from_directory(directory)
    
    if not items:
        print("No items found to process")
        return None
    
    results = processor.process_batch(items, parallel=parallel)
    processor.print_summary()
    processor.export_to_json(output_file)
    
    return processor.compute_summary()


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LYCHEETAH BATCH PROCESSOR")
    print("="*70 + "\n")
    
    # Create sample batch items
    sample_items = [
        BatchItem(
            id="doc1",
            content="The Protector ensures unconditional sacrifice. The Healer transmutes entropy. The Beacon maintains clarity.",
            metadata={'category': 'authentic'}
        ),
        BatchItem(
            id="doc2",
            content="The Protector anchors defense through guardian protocols.",
            metadata={'category': 'partial'}
        ),
        BatchItem(
            id="doc3",
            content="This system provides security automatically.",
            metadata={'category': 'generic'}
        ),
        BatchItem(
            id="doc4",
            content="The software is fast and reliable.",
            metadata={'category': 'generic'}
        ),
    ]
    
    # Process batch
    processor = BatchProcessor(max_workers=2)
    results = processor.process_batch(sample_items, parallel=True)
    
    # Print summary
    processor.print_summary()
    
    # Export
    processor.export_to_json("/tmp/batch_demo.json")
    processor.export_to_csv("/tmp/batch_demo.csv")
    
    # Query results
    print("\nSOVEREIGN SIGNATURES:")
    sovereign = processor.filter_by_sovereignty(True)
    for r in sovereign:
        print(f"  {r.item_id}: LCS={r.block.lcs:.3f}")
    
    print("\n" + "="*70)
    print("✓ BATCH PROCESSOR DEMONSTRATION COMPLETE")
    print("="*70 + "\n")
