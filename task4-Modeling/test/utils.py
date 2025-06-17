# Add to utils.py
def memory_monitor(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024 / 1024
        result = func(*args, **kwargs)
        end_mem = process.memory_info().rss / 1024 / 1024
        print(f"Memory used: {end_mem - start_mem:.1f} MB")
        return result
    return wrapper

# Decorate your functions
@memory_monitor
def perform_hypothesis_tests(df):
    # existing code