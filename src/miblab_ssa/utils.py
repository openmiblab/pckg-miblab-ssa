import psutil
import dask.distributed

def get_memory_limit():
    """Detects available memory per worker in GB."""
    try:
        # 1. Try to get it from a running Dask Client
        client = dask.distributed.get_client()
        worker_info = client.scheduler_info()['workers']
        if worker_info:
            # Get the memory limit of the first worker found
            first_worker = next(iter(worker_info.values()))
            limit_bytes = first_worker['memory_limit']
            return limit_bytes / (1024**3)
    except (ValueError, ImportError, RuntimeError):
        # 2. Fallback: Use system memory (psutil)
        # We use 'available' rather than 'total' to be safe
        stats = psutil.virtual_memory()
        return stats.available / (1024**3)