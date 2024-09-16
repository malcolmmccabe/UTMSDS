import argparse

def main():
    #########################################################################
    # Create the top-level parser & setup subparsers for each assignment
    parser = argparse.ArgumentParser(description="Visualize your algorithms in different environments and with different configurations.")
    subparsers = parser.add_subparsers(dest='assignment', required=True)
    #########################################################################


    #########################################################################
    # Bandits
    parser_bandits = subparsers.add_parser('bandit', help="Bandits w/ 10-armed testbed")
    parser_bandits.add_argument('--num_iterations', type=int, required=False, default=10000, help="Number of iterations to run")
    parser_bandits.add_argument('--num_simulations', type=int, required=False, default=300, help="Number of simulations to run")
    parser_bandits.add_argument('--k', type=int, required=False, default=10, help="Number of arms in the bandit")
    parser_bandits.set_defaults(func=action_for_bandits)
    #########################################################################


    #########################################################################
    # Parse the arguments, call the function
    args = parser.parse_args()
    args.func(args)
    #########################################################################


def action_for_bandits(args):
    from lib.visualize.bandit import visualize
    visualize(
        num_simulations = args.num_simulations,
        num_iterations = args.num_iterations,
        k = args.k
    )

if __name__ == "__main__":
    main()
