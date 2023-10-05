from package_import import *
from dataloader_functions import *
from data.Canine.slide_reader import get_slides
from SlideRunner.dataAccess.database import Database
from data.Meningioma.slide_reader import get_meningioma_slides

def load_data_loop(**args):
    t0 = time.time()

    if args['dataset'] == 'canine':
        slide_, mask_ = canine_data_load(**args)
    elif args['dataset'] == 'brain':
        slide_, mask_ = meningioma_data_load(**args)
    return slide_, mask_, time.time()-t0



def prepare_slides(dataset, model, soft_vs_hard, inference_or_train):

    if  'canine' in dataset:

        database = Database()
        database.open(str('data/Canine/Annotations/MITOS_WSI_CCMCT_ODAEL.sqlite'))

        if  'resnet' in model:
            slidelist_test = ['1', '2', '3','6','9','11','18', '20', '27', '30', '31' ]
        else:
            slidelist_test = [ '4', '7', '8', '12', '13', '14', '15', '17', '19', '21', '22', '23', '24', '25', '26', '28', '29', '32', '34', '35', '36']

        basepath = 'data/Canine/WSI'


        if inference_or_train == 'train':
            lbl_bbox, slides, files = get_slides(slidelist_test=slidelist_test, database=database, basepath=basepath,
                                                 soft_vs_hard=soft_vs_hard, inference_or_train=inference_or_train,
                                                 model=model)
        else:
            lbl_bbox, slides, files = get_slides(slidelist_test=slidelist_test, database=database, basepath=basepath,
                                                 soft_vs_hard=soft_vs_hard,  inference_or_train = inference_or_train, model = model)



    elif dataset == 'brain':
        brain_atlas_annotations = os.listdir('data/Meningioma/Brain_Atlas/Annotations/')
        brain_atlas_slides = os.listdir('data/Meningioma/Brain_Atlas/WSI/')
        slides = get_meningioma_slides(brain_atlas_annotations = brain_atlas_annotations,
                            brain_atlas_slides = brain_atlas_slides,
                            soft_vs_hard=soft_vs_hard,  inference_or_train = inference_or_train)

    return slides
