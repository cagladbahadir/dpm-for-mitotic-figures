from package_import import *
from tqdm import tqdm
from data.Canine.slide_reader import  thresholder, test_assign, train_and_validation_assign


def brain_atlas_reader(pth):
    tsv_file = open(pth)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    scores = dict()
    if '.txt' in pth:
        for row in read_tsv:
            try:
                float(row[0])
                if 1 not in scores.keys():
                    scores[1] = []
                scores[1].append([row[0],row[1]])
            except Exception as e:
                print(e)
                #print(pth)
        return scores
    for row in read_tsv:
        try:
            score = row[0].split(',')[-2]
            if 'NOT' in score:
                scr = 0
            elif 'INDET' in score:
                scr = 0.5
            elif score == 'MITE':
                scr = 1
            if scr not in scores.keys():
                scores[scr] = []
            locs = row[0].split(',')[3].replace("'","").replace('[','').replace(']','').split(" ")
            scores[scr].append([locs[0],locs[1]])
        except Exception as e:
            print(e)
    return scores


def get_meningioma_slides(brain_atlas_annotations: list, brain_atlas_slides: list,
                            soft_vs_hard: str='',  inference_or_train : str=''):

    slides = dict()

    for slide_name in brain_atlas_slides:

        annotations = [annot for annot in brain_atlas_annotations if slide_name.replace('.ndpi','') in annot]
        slides[slide_name] = dict()

        slide_path = 'data/Meningioma/Brain_Atlas/WSI/' + slide_name

        slide = openslide.open_slide(str(slide_path))
        slides[slide_name]['WSI'] = slide

        annotations_soft = dict()
        annotations_hard = dict()

        for annotation in annotations:
            annotation_read = brain_atlas_reader('data/Meningioma/Brain_Atlas/Annotations/'+annotation)
            for cls in annotation_read.keys():
                cls_hard = int(np.round(cls))
                if cls not in annotations_soft.keys():
                    annotations_soft[cls] = dict()
                    annotations_soft[cls] ['bboxes'] = list()
                    annotations_soft[cls] ['label'] = list()
                if cls_hard not in annotations_hard.keys():
                    annotations_hard[cls_hard] = dict()
                    annotations_hard[cls_hard] ['bboxes'] = list()
                    annotations_hard[cls_hard] ['label'] = list()

                for point in annotation_read[cls]:
                    r = 25
                    d = 2 * r
                    x_min = (int(float(point[0])) - r)
                    y_min = (int(float(point[1])) - r)
                    x_max = x_min + d
                    y_max = y_min + d

                    annotations_hard[cls_hard]['bboxes'].append([int(x_min), int(y_min), int(x_max), int(y_max)])
                    annotations_hard[cls_hard]['label'].append(cls_hard)
                    annotations_soft[cls]['bboxes'].append([int(x_min), int(y_min), int(x_max), int(y_max)])
                    annotations_soft[cls]['label'].append(cls)

        if  inference_or_train == 'train':
            slides = train_and_validation_assign(annotations_hard, annotations_soft,soft_vs_hard, slides, slide_name)
        else:
            slides = test_assign(annotations_hard, annotations_soft, soft_vs_hard, slides, slide_name)

    return slides

