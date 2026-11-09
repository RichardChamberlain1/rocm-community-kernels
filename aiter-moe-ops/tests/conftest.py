def pytest_configure(config):
    config.addinivalue_line(
        "markers", "kernels_ci: the cheap subset CI runs on AMD GPU runners"
    )
