"""Quick test of reality service with herniation detection"""
from core.reality_service import RealityEngineService, EngineConfig

config = EngineConfig(size=(64, 32), dt=0.01)
service = RealityEngineService(config)
service.initialize('big_bang')

print('Running 2000 steps with herniation detection...')
for i in range(2000):
    service._step_engine()
    if i % 500 == 0:
        stats = service.herniation_detector.get_statistics()
        M_mean = service.engine.current_state.M.mean().item()
        M_max = service.engine.current_state.M.max().item()
        T_mean = service.engine.current_state.T.mean().item()
        print(f"Step {i}: herniations={stats.get('total_count', 0)}, M_mean={M_mean:.4f}, M_max={M_max:.4f}, T={T_mean:.3f}")

structures = (service.engine.current_state.memory > 0.1).sum().item()
print(f"\nFinal structures (M > 0.1): {structures}")
print(f"Total herniations: {service.herniation_detector.total_herniations}")
