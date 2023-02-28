import torch
from torch import nn
import numpy as np
from components import ResBlock, DenseBlock, MaxPool2D, Flatten
from ArrhythmiaDataset2D import ArrhythmiaDataset
from torch.utils.data import random_split
import torch.optim as optim
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import csv
import argparse
import torch.multiprocessing as mp
import os


LEARNING_RATE = 0.01


class Experiment:
    def __init__(self):
        self.vallosses = []
        self.valaccs = []
        self.trainlosses = []
        self.learning_rate = 0

    def log_metric(self, metric, value):
        if metric == 'val_loss':
            self.vallosses.append(value)

        elif metric == 'val_acc':
            self.valaccs.append(value)

        elif metric == 'train_loss':
            self.trainlosses.append(value)

    def log_parameter(self, param, value):
        if param == 'learning_rate':
            self.learning_rate = value


def scale_lr(batch_size, lr, min_batch_size):
    return lr * (batch_size / min_batch_size)


def setup(rank, world_size, backend):
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def output_size(input_size, filter_size=1, padding=0, stride=1):
    return 1 + np.floor((input_size - filter_size + 2 * padding) / stride)


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
    for i in np.arange(1, genome.id[0]):  # set the in/out planes for conv layers
        genome.chromosomes[i].chromosomes['in_planes_0'] = genome.chromosomes[i - 1].chromosomes['out_planes']
    for j in np.arange(genome.id[0] + 1, genome.id[0] + genome.id[1]):  # set in/out features for dense layers
        genome.chromosomes[j].chromosomes['in_features'] = genome.chromosomes[j - 1].chromosomes['out_features']

    d = 128  # image dimensions
    for i, res in enumerate(genome.chromosomes[:genome.id[0]]):
        d = output_size(input_size=d, filter_size=res.chromosomes['conv_ker'], padding=0, stride=1)  # conv reduction
        d = output_size(input_size=d, filter_size=res.chromosomes['mp_ker'], padding=0, stride=res.chromosomes['mp_ker'])  # MP reduction
        rb = ResBlock(in_planes_0=res.chromosomes['in_planes_0'], out_planes_0=res.chromosomes['out_planes'],
                      conv_kernel_size_0=res.chromosomes['conv_ker'], att_kernel_size=res.chromosomes['att_ker'],
                      reduction_ratio=4, spatial=False)
        model.add_module(name=f'res{i}', module=rb)
        model.add_module(name=f'mp{i}', module=MaxPool2D(res.chromosomes['mp_ker']))

    model.add_module(name='flatten', module=Flatten())  # flatten before passing to MLP
    # input_features of first dense layer = d**2 * num_out_planes from last conv layer
    d = int(d ** 2 * genome.chromosomes[:genome.id[0]][-1].chromosomes['out_planes'])
    genome.chromosomes[genome.id[0]].chromosomes['in_features'] = d

    for i, dense in enumerate(genome.chromosomes[genome.id[0]:]):
        dropout = True if i < genome.chromosomes[-1].location else False  # i.e. no dropout for last layer
        db = DenseBlock(in_features=dense.chromosomes['in_features'],
                        out_features=dense.chromosomes['out_features'],
                        relu=True,
                        dropout=dropout
                        )
        model.add_module(name=f'dense{i}', module=db)
    model.add_module(name='softmax', module=nn.Softmax(dim=1))  # final layer = activation
    return model


def load_data(train_dir, train_ref, test_dir, test_ref):

    trainset = ArrhythmiaDataset(train_dir, reference_file_csv=train_ref,
                                 leads=2, normalize=False,
                                 smoothen=False, wavelet='mexh'
                                 )

    testset = ArrhythmiaDataset(test_dir, reference_file_csv=test_ref,
                                leads=2, normalize=False,
                                smoothen=False, wavelet='mexh',
                                test=True
                                )

    return trainset, testset


def train(model, optimizer, criterion, trainloader, gpu_id):
    model.train()
    total_loss = 0
    epoch_steps = 0
    for batch_idx, (images, labels) in tqdm(enumerate(trainloader)):
        optimizer.zero_grad()
        images = images.cuda(gpu_id, non_blocking=True)
        labels = labels.cuda(gpu_id, non_blocking=True)

        pred = model(images)

        loss = criterion(pred, labels)
        loss.backward()

        total_loss += loss.item()
        epoch_steps += 1

        optimizer.step()

    return total_loss / epoch_steps


def evaluate(model, criterion, valloader, local_rank):
    # Validation loss
    total_loss = 0.0
    epoch_steps = 0
    total = 0
    correct = 0

    model.eval()
    for i, data in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.cuda(local_rank), labels.cuda(local_rank)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            epoch_steps += 1

    val_acc = correct / total
    val_loss = total_loss / epoch_steps

    return val_loss, val_acc


def test_accuracy(model, testset, answer_path, device="cpu"):

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    with torch.no_grad():
        with open(answer_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Recording', 'Result'])
            for data in testloader:
                images, names = data
                images = images.to(device)
                preds = model(images)
                result = torch.argmax(preds, dim=1) + 1

                for i in range(len(names)):
                    answer = [names[i], result[i].item()]
                    if answer[1] > 9 or answer[1] < 1 or np.isnan(answer[1]):
                        answer[1] = 1
                    writer.writerow(answer)
            csvfile.close()


def run(genome, local_rank, world_size, args):
    """
    This is a single process that is linked to a single GPU
    :param local_rank: The id of the GPU on the current node
    :param world_size: Total number of processes across nodes
    :param args:
    :return:
    """
    experiment = Experiment()
    torch.cuda.set_device(local_rank)

    # The overall rank of this GPU process across multiple nodes
    global_process_rank = args.node_rank * args.gpus + local_rank

    learning_rate = scale_lr(args.replica_batch_size * world_size, LEARNING_RATE)
    experiment.log_parameter("learning_rate", learning_rate)

    print(f"Running DDP model on Global Process with Rank: {global_process_rank }.")
    setup(global_process_rank, world_size, args.backend)

    model = create_model(genome)
    model.cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9)

    # Load training data
    trainset, testset = load_data(train_dir="validation_set",
                                  train_ref="REF.csv",
                                  test_dir="last100",
                                  test_ref="last100.csv"
                                  )
    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_subset, num_replicas=world_size, rank=global_process_rank
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.replica_batch_size,
        sampler=train_sampler,
        num_workers=8,
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=args.replica_batch_size, shuffle=True, num_workers=8
    )

    for epoch in range(args.epochs):
        train_loss = train(
            ddp_model, optimizer, criterion, trainloader, local_rank
        )
        experiment.log_metric("train_loss", train_loss)
        val_loss, val_acc = evaluate(ddp_model, criterion, valloader, local_rank)
        experiment.log_metric("val_loss", val_loss)
        experiment.log_metric("val_acc", val_acc)
    ######################################################################################
    # this should return the csv file with all F1 scores etc...
    test_accuracy(model, testset, answer_path='genetic_answers.csv', device=f'cuda:{local_rank}')
    score(answers_csv_path='genetic_answers.csv', reference_csv_path='top100.csv')

    cleanup()


def score(answers_csv_path, reference_csv_path):
    answers = dict()
    reference = dict()
    A = np.zeros((9, 9), dtype=np.float)
    with open(answers_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            answers.setdefault(row['Recording'], []).append(row['Result'])
        f.close()
    with open(reference_csv_path) as ref:
        reader = csv.DictReader(ref)
        for row in reader:
            reference.setdefault(row['Recording'], []).append([row['First_label'], row['Second_label'], row['Third_label']])
        ref.close()

    for key in answers.keys():
        value = []
        for item in answers[key]:
            predict = int(item)
        for item in reference[key][0]:
            if item == '':
                item = 0
            value.append(int(item))

        if predict in value:
            A[predict-1][predict-1] += 1
        else:
            A[value[0]-1][predict-1] += 1

    F11 = 2 * A[0][0] / (np.sum(A[0, :]) + np.sum(A[:, 0]))
    F12 = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    F13 = 2 * A[2][2] / (np.sum(A[2, :]) + np.sum(A[:, 2]))
    F14 = 2 * A[3][3] / (np.sum(A[3, :]) + np.sum(A[:, 3]))
    F15 = 2 * A[4][4] / (np.sum(A[4, :]) + np.sum(A[:, 4]))
    F16 = 2 * A[5][5] / (np.sum(A[5, :]) + np.sum(A[:, 5]))
    F17 = 2 * A[6][6] / (np.sum(A[6, :]) + np.sum(A[:, 6]))
    F18 = 2 * A[7][7] / (np.sum(A[7, :]) + np.sum(A[:, 7]))
    F19 = 2 * A[8][8] / (np.sum(A[8, :]) + np.sum(A[:, 8]))

    F1 = (F11+F12+F13+F14+F15+F16+F17+F18+F19) / 9

    ## following is calculating scores for 4 types: AF, Block, Premature contraction, ST-segment change.

    Faf = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    Fblock = 2 * (A[2][2] + A[3][3] + A[4][4]) / (np.sum(A[2:5, :]) + np.sum(A[:, 2:5]))
    Fpc = 2 * (A[5][5] + A[6][6]) / (np.sum(A[5:7, :]) + np.sum(A[:, 5:7]))
    Fst = 2 * (A[7][7] + A[8][8]) / (np.sum(A[7:9, :]) + np.sum(A[:, 7:9]))

    # print(A)
    print('Total File Number: ', np.sum(A))

    print("F11: ", F11)
    print("F12: ", F12)
    print("F13: ", F13)
    print("F14: ", F14)
    print("F15: ", F15)
    print("F16: ", F16)
    print("F17: ", F17)
    print("F18: ", F18)
    print("F19: ", F19)
    print("F1: ", F1)

    print("Faf: ", Faf)
    print("Fblock: ", Fblock)
    print("Fpc: ", Fpc)
    print("Fst: ", Fst)

    with open('score.txt', 'w') as score_file:
        # print (A, file=score_file)
        print ('Total File Number: %d\n' %(np.sum(A)), file=score_file)
        print ('F11: %0.3f' %F11, file=score_file)
        print ('F12: %0.3f' %F12, file=score_file)
        print ('F13: %0.3f' %F13, file=score_file)
        print ('F14: %0.3f' %F14, file=score_file)
        print ('F15: %0.3f' %F15, file=score_file)
        print ('F16: %0.3f' %F16, file=score_file)
        print ('F17: %0.3f' %F17, file=score_file)
        print ('F18: %0.3f' %F18, file=score_file)
        print ('F19: %0.3f\n' %F19, file=score_file)
        print ('F1: %0.3f\n' %F1, file=score_file)
        print ('Faf: %0.3f' %Faf, file=score_file)
        print ('Fblock: %0.3f' %Fblock, file=score_file)
        print ('Fpc: %0.3f' %Fpc, file=score_file)
        print ('Fst: %0.3f' %Fst, file=score_file)

        score_file.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str)
    parser.add_argument("-b", "--backend", type=str, default="nccl")
    parser.add_argument(
        "-n",
        "--nodes",
        default=1,
        type=int,
        metavar="N",
        help="total number of compute nodes",
    )
    parser.add_argument(
        "-g", "--gpus", default=2, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "-nr",
        "--node_rank",
        default=0,
        type=int,
        help="ranking within the nodes, starts at 0",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to configs",
    )
    parser.add_argument(
        "--replica_batch_size",
        default=32,
        type=int,
        metavar="N",
        help="number of total epochs to configs",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="8892",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    world_size = args.gpus * args.nodes

    genomes_path = 'x_random_s_22_m_0.05_st_0.5_top_5_genomes.pt'
    best_genome = torch.load(genomes_path)[0]

    # Make sure all nodes can talk to each other on the unprivileged port range
    # (1024-65535) in addition to the master port
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    mp.spawn(
        run, args=(best_genome, world_size, args,), nprocs=args.gpus, join=True,
    )
