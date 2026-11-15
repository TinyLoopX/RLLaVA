from rllava.distribute.dist_utils import init_dist
from .entrypoint import main

if __name__ == "__main__":
    init_dist()
    main()
