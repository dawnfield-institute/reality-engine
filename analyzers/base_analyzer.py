"""
Base Analyzer Framework for Reality Engine

All analyzers inherit from this to detect emergent phenomena.
Analyzers observe but don't interfere - pure measurement.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class Detection:
    """A detected phenomenon with confidence and properties"""
    type: str
    confidence: float
    location: Optional[tuple] = None
    time: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    equation: Optional[str] = None
    parameters: Optional[Dict[str, float]] = None
    
    def __str__(self):
        conf_str = f"{self.confidence:.1%}"
        if self.equation:
            return f"{self.type} ({conf_str}): {self.equation}"
        return f"{self.type} ({conf_str})"


class BaseAnalyzer(ABC):
    """
    Base class for all reality analyzers.
    
    Analyzers:
    - Observe field states
    - Detect patterns
    - Measure properties
    - Never modify the fields
    
    The engine calls update() each step, passing current state.
    Analyzers return Detection objects for discovered phenomena.
    """
    
    def __init__(self, name: str, min_confidence: float = 0.5):
        """
        Initialize analyzer.
        
        Args:
            name: Analyzer identifier
            min_confidence: Minimum confidence threshold for reporting
        """
        self.name = name
        self.min_confidence = min_confidence
        self.detections = []
        self.history = []
        self.step_count = 0
        
    @abstractmethod
    def analyze(self, state: Dict) -> List[Detection]:
        """
        Analyze current state and return detections.
        
        Args:
            state: Current field state from reality engine
                - 'actual': A field (energy/actualization)
                - 'potential': P field (information/potential)
                - 'memory': M field (persistence)
                - 'temperature': T field (disequilibrium)
                - 'step': Current timestep
                - 'time': Elapsed physical time
                - 'structures': List of detected structures (if available)
        
        Returns:
            List of Detection objects for phenomena found
        """
        pass
    
    def update(self, state: Dict) -> List[Detection]:
        """
        Update analyzer with new state and return new detections.
        
        This is called by the engine each step. Override analyze() instead.
        """
        self.step_count += 1
        
        # Store history (every 10 steps to save memory)
        if self.step_count % 10 == 0:
            self.history.append({
                'step': state.get('step', 0),
                'time': state.get('time', 0.0),
                'summary': self._summarize_state(state)
            })
        
        # Analyze current state
        new_detections = self.analyze(state)
        
        # Filter by confidence threshold
        confident_detections = [
            d for d in new_detections 
            if d.confidence >= self.min_confidence
        ]
        
        # Store all confident detections
        self.detections.extend(confident_detections)
        
        return confident_detections
    
    def _summarize_state(self, state: Dict) -> Dict:
        """Create summary statistics of state for history"""
        summary = {}
        
        # Summarize field statistics
        for field_name in ['actual', 'potential', 'memory', 'temperature']:
            if field_name in state:
                field = state[field_name]
                if torch.is_tensor(field):
                    field_np = field.cpu().numpy() if field.is_cuda else field.numpy()
                else:
                    field_np = np.array(field)
                    
                summary[field_name] = {
                    'mean': float(np.mean(field_np)),
                    'std': float(np.std(field_np)),
                    'max': float(np.max(field_np)),
                    'min': float(np.min(field_np))
                }
        
        # Include structure count if available
        if 'structures' in state:
            summary['structure_count'] = len(state['structures'])
        
        return summary
    
    def get_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        return {
            'analyzer': self.name,
            'steps_analyzed': self.step_count,
            'total_detections': len(self.detections),
            'high_confidence_detections': len([d for d in self.detections if d.confidence > 0.8]),
            'detection_types': self._count_by_type(),
            'average_confidence': float(np.mean([d.confidence for d in self.detections])) if self.detections else 0.0,
            'recent_detections': [self._detection_to_dict(d) for d in self.detections[-100:]],  # Last 100
            'history_length': len(self.history)
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count detections by type"""
        counts = {}
        for d in self.detections:
            counts[d.type] = counts.get(d.type, 0) + 1
        return counts
    
    def _detection_to_dict(self, detection: Detection) -> Dict:
        """Convert Detection to dictionary for serialization"""
        return {
            'type': detection.type,
            'confidence': detection.confidence,
            'location': detection.location,
            'time': detection.time,
            'properties': detection.properties,
            'equation': detection.equation,
            'parameters': detection.parameters
        }
    
    def save_report(self, path: Path):
        """Save analysis report to JSON file"""
        report = self.get_report()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def print_summary(self):
        """Print a human-readable summary of findings"""
        print(f"\n{'='*70}")
        print(f"{self.name.upper()} ANALYSIS")
        print(f"{'='*70}")
        print(f"Steps analyzed: {self.step_count}")
        print(f"Total detections: {len(self.detections)}")
        
        if self.detections:
            print(f"Average confidence: {np.mean([d.confidence for d in self.detections]):.1%}")
            print(f"\nDetection types:")
            for dtype, count in sorted(self._count_by_type().items(), key=lambda x: -x[1]):
                print(f"  • {dtype}: {count}")
            
            print(f"\nRecent high-confidence detections:")
            recent_high = [d for d in self.detections[-20:] if d.confidence > 0.7]
            for d in recent_high[-5:]:
                print(f"  {d}")
        else:
            print("No detections above confidence threshold.")
        print(f"{'='*70}\n")
    
    def reset(self):
        """Clear all detections and history"""
        self.detections = []
        self.history = []
        self.step_count = 0
