

def print_layer_details(layers):
    print('          network:')
    print()
    print('      input layer: {:>5} nodes'.format(layers[0].get_nodes()))
    for layer in layers[1:-1]:
        print('     hidden layer: {:>5} nodes'.format(layer.get_nodes()))
    print('     output layer: {:>5} nodes'.format(layers[-1].get_nodes()))
    print()
    print('         training:')
    print()


def print_epoch_details(epoch, epochs, batch, error, duration):
    print(' epoch {:>6} / {} [batch: {:>3}, error: {:.5f}, duration: {:.5f}]'.format(epoch, epochs, batch, error, duration), end = '\r')
