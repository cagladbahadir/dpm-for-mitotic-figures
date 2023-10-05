import torch.nn
from module_loader import *
from dataloader import prepare_slides


warnings.simplefilter(action='ignore', category=RuntimeWarning)
torch.autograd.set_detect_anomaly(True)

## Arguments Parsed
parser = parser_func()
args = parser.parse_args()
print(args)

## Set the random seeds for each ensemble
torch.manual_seed(seeds[int(args.ensemble) - 1])  # Different for each ensemble
random.seed(seeds[int(args.ensemble) - 1])
np.random.seed(seeds[int(args.ensemble) - 1])

## Variables set
min_validation_loss = 1e10
min_validation_loss_slide = 1e10
time_stamp = time.time()

## Data loading and path selections
slides = prepare_slides(args.dataset, args.model_type, args.soft_vs_hard, 'train')

## Device initialize
if args.server_or_local == 'local':
    device = torch.device('cpu')
else:
    device = torch.device("cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
    print(device)

## Model initialize
net = model_select(args.model_type, args.soft_vs_hard, int(args.tile_size))

if args.model_type == 'diffusion' and args.server_or_local == 'server':
    net = torch.nn.DataParallel(net, device_ids=[0, 1]).to(device)
else:
    net.to(device)

## Creating the optimizer and the training folder
optimizer_classification, save_dir_model,scheduler, pretrained_dir  = set_parameters(date = args.date,
                                                                     ensemble = args.ensemble,
                                                                     batch_size = args.batch_size,
                                                                     Optimizer_lr = args.learning_rate,
                                                                     net = net,
                                                                     loss_type = args.loss_function,
                                                                     mitotic_percentage = args.mite_ratio,
                                                                     server = args.server_or_local,
                                                                     model = args.model_type,
                                                                     optimizer = args.optimizer,
                                                                     batch_validation = args.batch_validation,
                                                                     inference = False,
                                                                     tile_size = args.tile_size,
                                                                     dataset_percentage = args.dataset_percentage,
                                                                     soft_vs_hard = args.soft_vs_hard,
                                                                     dataset = args.dataset)

models_distillation = []
print('Device: ', str(device))

for epoch in range(1000):

    t0 = time.time()
    sigm = torch.nn.Sigmoid()

    # Loss, accuracy and other parameters to be tracked during training
    parameters_tracked = parameter_tracker()

    # Training dataset shuffled
    training_ids = np.array(list(slides.keys()))
    indix = np.random.permutation(len(training_ids))
    training_ids = training_ids[indix]

    # Loop over the dataset
    for i in range(len(training_ids)):
        t1 = time.time()
        slide_id = training_ids[i]
        train_tiles, train_labels, time_loading_batch = dataloader.load_data_loop(  slide_id = slide_id,
                                                                                    device = device,
                                                                                    batch_size = args.batch_size,
                                                                                    ratio = args.mite_ratio,
                                                                                    model = args.model_type,
                                                                                    state = 'train',
                                                                                    location = args.server_or_local,
                                                                                    slides = slides,
                                                                                    tile_size = int(args.tile_size),
                                                                                    soft_vs_hard=args.soft_vs_hard,
                                                                                    dataset = args.dataset,
                                                                                    date = args.date,
                                                                                    models_distillation = models_distillation,
                                                                                    temp = args.temp
                                                                                    )

        parameters_tracked['time_loading_training'] += time_loading_batch

        optimizer_classification.zero_grad()
        t2 = time.time()

        if 'diffusion' in args.model_type :
            loss = net(train_tiles,classes = train_labels)
        else:
            prediction,output = forward_pass(net, train_tiles.to(device) ,sigm)
            loss = loss_calculator(prediction, train_labels.to(device), args.loss_function)

        t3 = time.time()-t2

        # backprop
        if args.model_type == 'diffusion' and args.server_or_local == 'server':
            loss.mean().backward()
        else:
            loss.backward()

        optimizer_classification.step()

        parameters_tracked['train_loss'] += loss.mean().item()
        if 'diffusion' in args.model_type  :
             accuracy, true_pos, true_neg, false_pos, false_neg = 0,0,0,0,0
        else:
            accuracy, true_pos, true_neg, false_pos,false_neg = accuracy_calculator(prediction, train_labels)
            accuracy, true_pos, true_neg, false_pos, false_neg = accuracy.to('cpu'), true_pos.to('cpu'), true_neg.to('cpu'), false_pos.to('cpu'),false_neg.to('cpu')

        parameters_tracked['train_accuracy'] += accuracy
        parameters_tracked['train_accuracy_false_neg'] += false_neg
        parameters_tracked['train_accuracy_false_pos'] += false_pos
        parameters_tracked['train_accuracy_true_pos'] += true_pos
        parameters_tracked['train_accuracy_true_neg'] += true_neg
        parameters_tracked['time_forward'] += time.time() - t2

        if i == len(training_ids)-1:
            print('One epoch took: ' + str(time.time() - t1) + " seconds.")
            time_optimization = time_loading_training = time_validation_forward = 0

            print('Epoch: [%d] loss: %.8f   classification accuracy: %.3f' %
                  (epoch + 1, parameters_tracked['train_loss'] / len(training_ids),
                   parameters_tracked['train_accuracy'] / len(training_ids)))

            train_acc_hold = parameters_tracked['train_accuracy']

            t0 = time.time()
            validation_tile_number = 0

            if epoch % 1 == 0:  #
                with torch.no_grad():
                    net.eval()
                    validation_ids =  np.array(list(slides.keys()))

                    for j in range(len(validation_ids)):
                        #print('validation ', j, " ",len(validation_ids))
                        validation_id = validation_ids[j]

                        validation_tile, validation_labels, time_loading = dataloader.load_data_loop(slide_id=validation_id,
                                                                                                    ids = validation_ids,
                                                                                                    device = device,
                                                                                                    batch_size = args.batch_validation,
                                                                                                    ratio = args.mite_ratio,
                                                                                                    model = args.model_type,
                                                                                                    state = 'validation',
                                                                                                    location = args.server_or_local,
                                                                                                    augmentation = '',
                                                                                                    slides = slides,
                                                                                                    tile_size = int(args.tile_size),
                                                                                                    soft_vs_hard=args.soft_vs_hard,
                                                                                                    dataset = args.dataset,
                                                                                                    date=args.date,
                                                                                                    models_distillation=models_distillation,
                                                                                                    temp=args.temp
                                                                                                    )

                        t3 = time.time()

                        if 'diffusion' in args.model_type :
                            validation_loss = net(validation_tile, classes = validation_labels)
                            prediction = torch.clone(validation_tile)
                        else:
                            prediction,output = forward_pass(net, validation_tile, sigm)
                            validation_loss = loss_calculator(prediction, validation_labels, args.loss_function)

                        if 'diffusion' in args.model_type :
                            accuracy, true_pos, true_neg, false_pos, false_neg = 0,0,0,0,0
                        else:
                            accuracy, true_pos, true_neg, false_pos, false_neg = accuracy_calculator(prediction,
                                                                                                      validation_labels)
                            accuracy, true_pos, true_neg, false_pos, false_neg = accuracy.to('cpu'), true_pos.to(
                                 'cpu'), true_neg.to('cpu'), false_pos.to('cpu'), false_neg.to('cpu')

                        parameters_tracked['validation_loss'] += validation_loss.mean().item()
                        parameters_tracked['time_validation_forward'] += time.time() - t3
                        parameters_tracked['validation_accuracy'] += accuracy
                        parameters_tracked['validation_accuracy_false_neg'] += false_neg
                        parameters_tracked['validation_accuracy_false_pos'] += false_pos
                        parameters_tracked['validation_accuracy_true_pos'] += true_pos
                        parameters_tracked['validation_accuracy_true_neg'] += true_neg


                        if j == len(validation_ids)-1:

                            print('Validation took: ' + str(time.time() - t0) + " seconds.")
                            time_validation_forward = 0
                            print('Epoch: [%d] validation loss: %.8f' %
                                  (epoch + 1, parameters_tracked['validation_loss'] / len(validation_ids) ))


                            if args.model_type == 'diffusion' and epoch % 10 == 0:
                                PATH = save_dir_model + "/model_checkpoint_epoch_"+str(epoch)+".pt"

                                torch.save({
                                    'epoch': str(epoch),
                                    'model_state_dict': net.state_dict(),
                                    'optimizer_classification_state_dict': optimizer_classification.state_dict(),
                                }, PATH)

                            if not torch.isnan(parameters_tracked['validation_loss']):

                                convergence = checkpoint_history(epoch=epoch,
                                                                 save_dir_model=save_dir_model,
                                                                 train_loss=parameters_tracked['train_loss'],
                                                                 len_training_ids=len(training_ids),
                                                                 validation_loss=parameters_tracked['validation_loss'],
                                                                 train_accuracy=parameters_tracked['train_accuracy'],
                                                                 train_accuracy_true_pos=parameters_tracked[
                                                                     'train_accuracy_true_pos'],
                                                                 train_accuracy_true_neg=parameters_tracked[
                                                                     'train_accuracy_true_neg'],
                                                                 validation_accuracy=parameters_tracked[
                                                                     'validation_accuracy'],
                                                                 validation_accuracy_true_pos=parameters_tracked[
                                                                     'validation_accuracy_true_pos'],
                                                                 validation_accuracy_true_neg=parameters_tracked[
                                                                     'validation_accuracy_true_neg'],
                                                                 len_validation_ids=len(validation_ids),
                                                                 time_stamp=time.time() - time_stamp,
                                                                 validation_accuracy_false_neg=parameters_tracked[
                                                                     'validation_accuracy_false_neg'],
                                                                 validation_accuracy_false_pos=parameters_tracked[
                                                                     'validation_accuracy_false_pos'],
                                                                 train_accuracy_false_pos=parameters_tracked[
                                                                     'train_accuracy_false_pos'],
                                                                 train_accuracy_false_neg=parameters_tracked[
                                                                     'train_accuracy_false_neg'],
                                                                 mitotic_figure_ratio=args.mite_ratio)

                                if (parameters_tracked['validation_loss'] / (len(validation_ids) ) <= min_validation_loss):
                                    PATH = save_dir_model + "/model_checkpoint.pt"

                                    torch.save({
                                        'epoch': str(epoch),
                                        'model_state_dict': net.state_dict(),
                                        'optimizer_classification_state_dict': optimizer_classification.state_dict(),
                                    }, PATH)

                                    min_validation_loss = parameters_tracked['validation_loss'] / (
                                            len(validation_ids))

                            parameters_tracked['train_loss'] = torch.tensor(0.)
                            parameters_tracked['train_accuracy'] = torch.tensor(0.)
                            parameters_tracked['train_accuracy_false_neg'] = torch.tensor(0.)
                            parameters_tracked['train_accuracy_false_pos'] = torch.tensor(0.)
                            parameters_tracked['train_accuracy_true_pos'] = torch.tensor(0.)
                            parameters_tracked['train_accuracy_true_neg'] = torch.tensor(0.)
                            parameters_tracked['train_mite_count'] = torch.tensor(0.)
                            parameters_tracked['validation_loss'] = torch.tensor(0.)
                            parameters_tracked['validation_accuracy'] = torch.tensor(0.)
                            parameters_tracked['validation_accuracy_true_pos'] = torch.tensor(0.)
                            parameters_tracked['validation_accuracy_true_neg'] = torch.tensor(0.)
                            parameters_tracked['validation_accuracy_false_pos'] = torch.tensor(0.)
                            parameters_tracked['validation_accuracy_false_neg'] = torch.tensor(0.)
                            parameters_tracked['time_loading_training'] = torch.tensor(0.)
                            parameters_tracked['time_validation_forward'] = torch.tensor(0.)
                            parameters_tracked['time_forward'] = torch.tensor(0.)

