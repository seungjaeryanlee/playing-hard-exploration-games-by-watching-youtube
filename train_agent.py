from commons import load_model
from wrappers import make_env
from networks import TDC, CMC


def train_agent():
    # Load embedded network
    tdc = TDC().to(device)
    cmc = CMC().to(device)
    load_models(tdc, cmc)

    # TODO Create checkpoints
    checkpoints = []

    # Create environment
    env = make_env(tdc, cmc, checkpoints)

if __name__ == '__main__':
    main()
