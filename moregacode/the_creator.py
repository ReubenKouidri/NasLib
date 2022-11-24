import torch.nn as nn
from components import ResBlock1, ResBlock2, ResBlock3, MaxPool2D, DenseBlock, Flatten


def create_model(genome):
    """
        :param genome: chromosomes of chromosomes encoding the model:
        [gene0(type=conv), gene1(conv),....., gene6(type=dense)]
        where conv chromosomes look like [out_planes, conv_ker, att_ker, red_ratio],
        and dense chromosomes look like: [num_neurons]
        :return: model architecture
    """
    model = nn.Sequential()

    genome.chromosomes[0].chromosomes['in_planes_0'] = 1
    genome.chromosomes[-1].chromosomes['out_features'] = 9

    rb_types = genome.id[0]  # e.g. (1, 2, 2, 1)
    num_rbs = len(rb_types)  # e.g. 4

    # intra-gene synchronisation
    for i, rb_type in enumerate(rb_types):  # e.g. (0, 1) : (2, 2)
        if rb_type > 1:  # select the RBs with > 1 conv layer
            for j in range(1, rb_type):  # j = 1
                genome.chromosomes[i].chromosomes[f'in_planes_{j}'] = genome.chromosomes[i].chromosomes[f'out_planes_{j - 1}']
                # e.g. for above we have genome.chromosomes[1].seq["in_planes_1"] = genome.chromosomes[1].seq["out_planes_0"]

    # inter-gene synchronisation
    # need to have more than 1 RB gene
    if num_rbs > 1:
        for i, j in enumerate(range(1, num_rbs)):  # e.g. range(1, 4) = (1, 2, 3)
            genome.chromosomes[j].chromosomes["in_planes_0"] = genome.chromosomes[i].chromosomes[f"out_planes_{rb_types[i] - 1}"]
            # e.g. out_planes_1 | t=2 (RB2)

    ##################################################################################################################
    # Now that the genome is synchronised, we can express the chromosomes as model architecture
    ##################################################################################################################
    for i, res in enumerate(genome.chromosomes[:num_rbs]):
        if res.chromosome_type == 'RB1':
            rb1 = ResBlock1(
                in_planes_0=res.chromosomes['in_planes_0'],
                out_planes_0=res.chromosomes['out_planes_0'],
                conv_kernel_size_0=res.chromosomes['conv_ker_0'],
                att_kernel_size=res.chromosomes['att_ker'],
                reduction_ratio=res.chromosomes['r_ratio'],
                mp_ker=res.chromosomes['mp_ker'],
                cbam=True,
                spatial=False
            )
            model.add_module(name=f'RB1_{i}',
                             module=rb1)  # NEED TO INDEX LAYER NAMES SO THEY ARE PLACED IN CORRECT ORDER! OTHERWISE BUG

        elif res.chromosome_type == 'RB2':
            rb2 = ResBlock2(
                in_planes_0=res.chromosomes['in_planes_0'],
                out_planes_0=res.chromosomes['out_planes_0'],
                conv_kernel_size_0=res.chromosomes['conv_ker_0'],
                in_planes_1=res.chromosomes['in_planes_1'],
                out_planes_1=res.chromosomes['out_planes_1'],
                conv_kernel_size_1=res.chromosomes['conv_ker_1'],
                att_kernel_size=res.chromosomes['att_ker'],
                reduction_ratio=res.chromosomes['r_ratio'],
                mp_ker=res.chromosomes['mp_ker'],
                cbam=True,
                spatial=False
            )
            model.add_module(name=f'RB2_{i}', module=rb2)

        elif res.chromosome_type == 'RB3':
            rb3 = ResBlock3(
                in_planes_0=res.chromosomes['in_planes_0'],
                out_planes_0=res.chromosomes['out_planes_0'],
                conv_kernel_size_0=res.chromosomes['conv_ker_0'],
                in_planes_1=res.chromosomes['in_planes_1'],
                out_planes_1=res.chromosomes['out_planes_1'],
                conv_kernel_size_1=res.chromosomes['conv_ker_1'],
                in_planes_2=res.chromosomes['in_planes_2'],
                out_planes_2=res.chromosomes['out_planes_2'],
                conv_kernel_size_2=res.chromosomes['conv_ker_2'],
                att_kernel_size=res.chromosomes['att_ker'],
                reduction_ratio=res.chromosomes['r_ratio'],
                mp_ker=res.chromosomes['mp_ker'],
                cbam=True,
                spatial=False
            )
            model.add_module(name=f'RB3_{i}', module=rb3)

        else:
            raise TypeError("should either be 'RB1', 'RB2' or 'RB3'!")

        # model.add_module(name=f'MP_{i}', module=MaxPool2D(res.chromosomes['mp_ker']))

    model.add_module(name='flatten', module=Flatten())  # flatten before passing to MLP
    # input_features of first dense layer = d**2 * num_out_planes from last conv layer
    # genome.chromosomes[genome.id[0][0] - 1] gives the last RB gene, (genome.id[0][1]-1) = 0 / 1 for RB1 / RB2 resp.
    # rb_types[num_rbs - 1] - 1 should give 0 if RB1, 1 if RB2
    d = int(genome.output_size ** 2 * genome.chromosomes[num_rbs - 1].chromosomes[
        f'out_planes_{int(genome.chromosomes[num_rbs - 1].chromosome_type[-1]) - 1}'])

    genome.chromosomes[num_rbs].chromosomes['in_features'] = d  # the first DB gene's in_features set to d
    for j in range(len(genome.id[0]) + 1, len(genome.id[0]) + genome.id[1]):  # set in/out features for dense layers
        genome.chromosomes[j - 1].chromosomes['out_features'] = genome.chromosomes[j].chromosomes['in_features']

    for i, dense in enumerate(genome.chromosomes[num_rbs:]):
        dropout = False if dense.location == len(genome.chromosomes) else True  # i.e. no dropout for last layer
        db = DenseBlock(in_features=dense.chromosomes['in_features'],
                        out_features=dense.chromosomes['out_features'],
                        relu=True,
                        dropout=dropout
                        )
        model.add_module(name=f'dense_{i}', module=db)

    # can amend genome to take activation as a gene
    model.add_module(name='softmax', module=nn.Softmax(dim=1))  # final layer = activation
    return model
