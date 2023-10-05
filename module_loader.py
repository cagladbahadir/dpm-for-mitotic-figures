from package_import import *

def set_parameters(**args):
    """
    :param args:
    :return:
    """

    save_dir_model = 'Training_Logs/Date_{dat}/Dataset_{dts}_/Tile_size_{ts}/{soft_vs_hard}_Labels/Model_{model}_/Ensemble_{ensemb}_/Batch_{bch}_Batch_Val_{bch_val}_Lr_{lr}_Mite_Percentage_{mp}_Optimizer_{optim}/'.format(
        dat=str(args['date']),
        model=args['model'],
        ensemb=str(args['ensemble']),
        bch=str(args['batch_size']),
        bch_val = str(args['batch_validation']),
        lr=args['Optimizer_lr'],
        mp=args['mitotic_percentage'],
        optim = args['optimizer'],
        ts = args['tile_size'],
        soft_vs_hard  = args['soft_vs_hard'],
        dts = args['dataset']
        )
    print(save_dir_model)
    pretrained_dir = None

    if not args['inference']:
        if not os.path.isdir(save_dir_model): os.makedirs(save_dir_model)

    model=args['net']

    if args['optimizer']=='Adam':
        optimizer_= optim.Adam(model.parameters(), lr=float(args['Optimizer_lr']))
    elif args['optimizer']=='AdamW':
        optimizer_= optim.AdamW(model.parameters(), lr=float(args['Optimizer_lr']))


    if not args['inference']:

        np.savez(save_dir_model+"/History",
                 train_loss = [],
                 train_accuracy = [],
                 train_accuracy_false_neg = [],
                 train_accuracy_false_pos = [],
                 train_accuracy_true_pos = [],
                 train_accuracy_true_neg = [],
                 train_mite_count = [],

                 validation_loss = [],
                 validation_accuracy = [],
                 validation_accuracy_true_pos = [],
                 validation_accuracy_true_neg = [],
                 validation_accuracy_false_pos = [],
                 validation_accuracy_false_neg = [],
                 validation_mite_count=[],

                 time_loading_training = [],
                 time_validation_forward = [],
                 time_forward = [],
                 time_stamp = []
                 )

    ## Not used
    scheduler = None

    return optimizer_,save_dir_model,scheduler,pretrained_dir


def parser_func():

    ## Training time

    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", default = 1)
    parser.add_argument("--date", default = 'Feb1')
    parser.add_argument("--batch_size", type=int, default = 2)
    parser.add_argument("--batch_validation", type=int, default = 2)
    parser.add_argument("--server_or_local",default = 'local')
    parser.add_argument("--learning_rate", type=float, default = 1e-3)
    parser.add_argument("--optimizer", type=str, default = 'Adam')
    parser.add_argument("--dataset",default = 'canine') #canine or brain
    parser.add_argument("--model_type", default = 'diffusion') #resnet34 or diffusion
    parser.add_argument("--loss_function", default = 'BCE')
    parser.add_argument("--mite_ratio", default = '0.1')
    parser.add_argument("--inference", default = False)
    parser.add_argument("--tile_size", default=64)
    parser.add_argument("--dataset_percentage", default='100')
    parser.add_argument("--soft_vs_hard", default = 'hard')
    parser.add_argument("--temp", default=3.1552)

    ## Inference time

    parser.add_argument("--inference_dataset", default = 'validation')
    parser.add_argument("--real_or_fake_test",default='real')
    parser.add_argument("--seed", default=10)
    parser.add_argument("--test_batch", default=0)
    parser.add_argument("--real_to_generated", default=False)
    return parser

global seeds

seeds = [5, 128, 670, 1023, 9845]


def parameter_tracker():
    list_of_params = ['train_loss', 'train_accuracy',
                      'train_accuracy_false_neg',
                      'train_accuracy_false_pos',
                      'train_accuracy_true_pos',
                      'train_accuracy_true_neg',
                      'train_mite_count',
                      'validation_loss',
                      'validation_accuracy',
                      'validation_accuracy_true_pos',
                      'validation_accuracy_true_neg',
                      'validation_accuracy_false_pos',
                      'validation_accuracy_false_neg',
                      'time_loading_training',
                      'time_validation_forward',
                      'time_forward',
                      'validation_mite_count']

    parameters = {}

    for i in list_of_params:
        parameters[i] = torch.tensor(0.)

    return parameters

