"""Genesis simulation initialization."""
import genesis as gs


def init_genesis():
    """Initialize Genesis with GPU backend."""
    gs.init(backend=gs.gpu)
