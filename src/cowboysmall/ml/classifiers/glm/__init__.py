

def print_details(iterations, cost, theta):
    print()
    print('       convergence:')
    print()
    print('        iterations: {:> 6}'.format(iterations))
    print('        final cost: {: 12.5f}'.format(cost))
    print()
    print('         intercept: {: 12.5f}'.format(theta[0, 0]))
    for i, c in enumerate(theta[1:, 0]):
        print('   coefficient {:>3}: {: 12.5f}'.format(i + 1, c))
    print()
