"""
Reality Engine Client for Dashboard

This client connects the dashboard UI to the Reality Engine Service.
It handles subscriptions to engine updates and provides a clean API
for the dashboard server to query engine state.
"""

import sys
from pathlib import Path
from typing import Optional, Callable, Dict, Any

# Add engine root to path
ENGINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_ROOT))

from core.reality_service import RealityEngineService, EngineConfig, get_service


class RealityEngineClient:
    """
    Client interface to Reality Engine Service.
    
    The dashboard uses this client to:
    - Start/stop the engine
    - Subscribe to state updates
    - Query current state
    - Get field snapshots
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Initialize client connection to Reality Engine Service.
        
        Args:
            config: Engine configuration (if creating new service)
        """
        # Get or create the global service instance
        self.service = get_service(config)
        self._update_callbacks = []
        
    def initialize(self, mode='big_bang'):
        """
        Initialize the Reality Engine.
        
        Args:
            mode: Initialization mode ('big_bang', 'vacuum', 'custom')
        """
        if not self.service.initialized:
            self.service.initialize(mode=mode)
    
    def start(self):
        """Start the Reality Engine service"""
        if not self.service.initialized:
            self.initialize()
        
        self.service.start()
    
    def stop(self):
        """Stop the Reality Engine service"""
        self.service.stop()
    
    def subscribe_updates(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to engine state updates.
        
        The callback will be called on each engine update with the current state.
        
        Args:
            callback: Function that receives state dictionary
        """
        # Wrap callback to track locally
        self._update_callbacks.append(callback)
        self.service.subscribe(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return self.service.get_current_state()
    
    def get_field_snapshot(self, display_size: tuple = (200, 100)) -> Dict[str, Any]:
        """
        Get field visualization data.
        
        Args:
            display_size: (height, width) for display resolution
            
        Returns:
            Dictionary with energy, entropy, temperature, pressure arrays
        """
        return self.service.get_field_snapshot(display_size)
    
    def get_periodic_table(self) -> Dict[str, Any]:
        """Get current periodic table of emerged elements"""
        state = self.service.get_current_state()
        return state.get('periodic_table', {})
    
    def get_particles(self) -> list:
        """Get list of detected particles"""
        return self.service.particles
    
    def get_stellar_structures(self) -> list:
        """Get list of detected stellar structures"""
        return self.service.stellar_structures
    
    def get_pac_conservation(self) -> float:
        """Get PAC (Particle-Antiparticle Conservation) value"""
        state = self.service.get_current_state()
        return state.get('pac_value', 1.0571)
    
    def get_herniation_stats(self) -> Dict[str, Any]:
        """Get herniation statistics"""
        state = self.service.get_current_state()
        return state.get('herniation_stats', {})
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.service.get_performance_stats()
    
    def is_running(self) -> bool:
        """Check if engine is currently running"""
        return self.service.running
    
    def is_initialized(self) -> bool:
        """Check if engine is initialized"""
        return self.service.initialized
    
    @property
    def iteration(self) -> int:
        """Get current iteration count"""
        return self.service.iteration
    
    @property
    def time(self) -> float:
        """Get current simulation time"""
        return self.service.time
