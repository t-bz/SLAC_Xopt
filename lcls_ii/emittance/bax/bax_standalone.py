import argparse

import yaml

from emittance_opt import BAXEmittanceOpt


def main(args):
    # test loading
    if args.config is None:
        raise RuntimeError("add a configuration file name")

    config = yaml.safe_load(open(args.config))
    bax = BAXEmittanceOpt(**config)

    if args.test:
        print("running test beamsize measurement")
        bax.image_diagnostic.test_measurement()
    else:
        # take background measurement
        print("measuring screen background")
        bax.image_diagnostic.measure_background()

        # run bayesian exploration
        print("perfomring Bayesian exploration")
        bax.run_bayesian_exploration()

        # run bax optimization
        print("perfomring emittance optimization via BAX")
        bax.run_emittance_optimization()

        # return results/visualization


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAX Emittance minimization script")

    parser.add_argument("config", nargs="?", type=str, help="config filename")
    parser.add_argument(
        "--test",
        action="store_true",
        help="flag to denote if a test run "
        "should be executed, tests basic "
        "measurement and loading config file",
    )

    args = parser.parse_args()
    main(args)
