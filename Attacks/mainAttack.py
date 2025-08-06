from Attacks.pruning import prune_model_l1_unstructured 
from Attacks.quantization import quantization

def attacks(net,TypeAttack,attackparameters):
    '''
    Apply a modification based on the ID and parameters
    :param TypeAttack: ID of the modification
    :param net: network to be altered
    :param parameters: parameters of the modification
    :return: altered NN
    '''
    if TypeAttack=="l1pruning":
        return prune_model_l1_unstructured(net, attackparameters["proportion"])

    elif TypeAttack=="quantization":
        return quantization(net,attackparameters["bits"])
    else:
        print("NotImplemented")
        return net
'''
    attackParameter = {'name': "all", "std":.1}
    network = attacks(network, "noise", attackParameter)
'''