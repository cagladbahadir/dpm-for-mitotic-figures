## This python file is modified from the original source: https://github.com/DeepPathology/MITOS_WSI_CCMCT
from package_import import *
from tqdm import tqdm

def get_slides(slidelist_test:list, database:"Database", negative_class:int=7, basepath:str='WSI', size:int=256, soft_vs_hard: str='', inference_or_train: str='', model: str=''):

    lbl_bbox = list()
    files = list()
    slides = dict()


    getslides = """SELECT uid, filename FROM Slides"""
    for idx, (currslide, filename) in enumerate(tqdm(database.execute(getslides).fetchall(), desc='Loading slides .. ')):
        if (str(currslide) in slidelist_test) and inference_or_train == 'train': # skip test slides
                continue
        elif (str(currslide)  not in slidelist_test) and inference_or_train == 'test': # keep test slides
                continue
        slides[filename] = dict()

        database.loadIntoMemory(currslide)

        slide_path = basepath + os.sep + filename

        slide = openslide.open_slide(str(slide_path))
        slides[filename]['WSI'] = slide
        level = 0
        down_factor = slide.level_downsamples[level]

        annotations_soft = dict()
        annotations_hard = dict()

        for id, annotation in database.annotations.items():
            annotation.r = 25
            d = 2 * annotation.r / down_factor
            x_min = (annotation.x1 - annotation.r) / down_factor
            y_min = (annotation.y1 - annotation.r) / down_factor
            x_max = x_min + d
            y_max = y_min + d

            cls_hard = 1 if annotation.agreedClass == 2 else 0
            cls_soft = []
            for i_ in annotation.labels:
                if i_.annnotatorId == 1 or i_.annnotatorId == 2:
                    cls_soft.append(i_.classId)

            cls_soft = np.array(cls_soft)
            cls_soft = np.mean(cls_soft == 2)
            cls_soft = thresholder(cls_soft)

            if cls_hard not in annotations_hard:
                annotations_hard[cls_hard] = dict()
                annotations_hard[cls_hard] ['bboxes'] = list()
                annotations_hard[cls_hard] ['label'] = list()
            #
            annotations_hard[cls_hard]['bboxes'].append([int(x_min), int(y_min), int(x_max), int(y_max)])
            annotations_hard[cls_hard]['label'].append(cls_hard)

            if cls_soft not in annotations_soft:
                annotations_soft[cls_soft] = dict()
                annotations_soft[cls_soft] ['bboxes'] = list()
                annotations_soft[cls_soft] ['label'] = list()

            annotations_soft[cls_soft]['bboxes'].append([int(x_min), int(y_min), int(x_max), int(y_max)])
            annotations_soft[cls_soft]['label'].append(cls_soft)

        if  inference_or_train == 'train':
            slides = train_and_validation_assign(annotations_hard, annotations_soft,soft_vs_hard, slides, filename)
        else:
            slides = test_assign(annotations_hard, annotations_soft, soft_vs_hard, slides, filename)

    return lbl_bbox, slides,files

def train_and_validation_assign(annotations_hard,annotations_soft,soft_vs_hard, slides,filename):
    y1_values = np.array(annotations_hard[1]['bboxes'])[:, 1]

    separation = np.percentile(y1_values, 75)

    slides[filename]['train'] = dict()
    slides[filename]['validation'] = dict()

    if soft_vs_hard == 'soft':
        annotations = annotations_soft
    else:
        annotations = annotations_hard

    for class_ in annotations.keys():
        slides[filename]['train'][class_] = []
        slides[filename]['validation'][class_] = []

        indices_train = np.array(annotations[class_]['bboxes'])[:, 1] < separation
        indices_validation = np.array(annotations[class_]['bboxes'])[:, 1] >= separation

        slides[filename]['train'][class_] = np.array(annotations[class_]['bboxes'])[indices_train, :]
        slides[filename]['validation'][class_] = np.array(annotations[class_]['bboxes'])[indices_validation, :]

    return slides


def test_assign(annotations_hard,annotations_soft,soft_vs_hard, slides,filename):

    slides[filename]['test'] = dict()

    if soft_vs_hard == 'soft':
        annotations = annotations_soft
    else:
        annotations = annotations_hard

    for class_ in annotations.keys():
        slides[filename]['test'][class_] = []
        slides[filename]['test'][class_] = np.array(annotations[class_]['bboxes'])
    return slides

def thresholder(score):
    if score < 0.3:
        result = 0.
    elif score >= 0.3 and score <= 0.66:
        result = 0.5
    else:
        result = 1
    return result