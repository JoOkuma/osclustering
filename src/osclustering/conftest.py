
def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="Plot the clustering results during tests",
    ) 
