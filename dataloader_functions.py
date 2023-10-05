import torch.nn
from package_import import *
from trainer import forward_pass

def meningioma_data_load( **args):

    # Empty image and label tensors
    if args['model'] == 'diffusion':
        image_ = torch.empty(size=(args['batch_size'], 3, args['tile_size'], args['tile_size']), device=args['device'])
        label_ = torch.empty(size=(args['batch_size'], 1), device=args['device'])
    else:
        image_ = torch.empty(size=(args['batch_size'], 3, 256, 256), device=args['device'])
        label_ = torch.empty(size=(args['batch_size'], 1), device=args['device'])

    # Ratio of mitotic figures in a batch
    ratio = float(args['ratio'])

    for i in range(args['batch_size']):

        if args['soft_vs_hard'] == 'soft':
            classes = [0, 0.5, 1]
        else:
            classes = [0, 1]
        cls = np.random.choice(classes)
        probabilities = []

        if args['state'] == 'train' or args['state'] == 'validation':
            for slide_select in list(args['slides'].keys()):
                try:
                    probabilities.append(len(args['slides'][slide_select][args['state']][cls]))
                except:
                    probabilities.append(0)
            probabilities = probabilities/np.sum(probabilities)
            slide_select = np.random.choice(list(args['slides'].keys()), p=probabilities)

        bboxes = args['slides'][slide_select][args['state']][cls]

        # Chose a random example from the selected class
        indice = np.random.choice(np.arange(len(bboxes)),1)[0]
        bbox = bboxes[indice]
        center_x = bbox[0]+int((bbox[2]-bbox[0])/2)
        center_y = bbox[1] + int((bbox[3] - bbox[1]) / 2)
        x, y = center_x-int(args['tile_size']), center_y-int(args['tile_size'])

        # Read the current tile that has the annotated cell in the center
        img = np.array(args['slides'][slide_select]['WSI'].read_region(location=(x,y), level=0, size=((args['tile_size']*2, args['tile_size']*2))))[:, :, :3]
        lbl = cls

        # Random augmentation on the images
        crop = torchvision.transforms.CenterCrop(args['tile_size'])
        random_angle = np.random.choice(np.arange(270), 1)[0]
        random_translate = [0,0]
        augment_random = np.random.choice([0, 1],1)[0]#, size=1, replace=False, p=[0.5, 0.5])[0]
        if augment_random == 1:

            affine_transformed_img = torchvision.transforms.functional.affine(torch.tensor(img).permute(2, 0, 1),
                                                                              angle=int(random_angle),
                                                                              translate=(
                                                                              random_translate[0], random_translate[1]),
                                                                              scale=1, shear=0)
        else:
            affine_transformed_img = torch.tensor(img).permute(2, 0, 1)
        affine_transformed_img = crop(affine_transformed_img)
        affine_transformed_img = affine_transformed_img.float()/255.

        if 'resnet' in args['model']:
            if int(args['tile_size']) < 256:
                affine_transformed_img = torch.nn.functional.interpolate(affine_transformed_img.unsqueeze(0).float(),
                                                                         scale_factor=(4, 4),
                                                                         mode='bilinear')
            affine_transformed_img = affine_transformed_img.squeeze()

            jitter = torchvision.transforms.ColorJitter(brightness=[0.4, 1.2], contrast=[0.5, 4], saturation=[0.6, 3],
                                                        hue=[-0.15, 0.15])
            #jitter = torchvision.transforms.ColorJitter(brightness=[1,1], contrast=[1,1], saturation=[1,1], hue=[0,0])
            ## This augmentation does not change the image at all
            ## Brightness is the amount of light, it should never be 0 or much higher than 1 [0.4,1.2]
            ## Contrast is the amount that the colors are apart from each other, at 0 it is all gray, [0.5,4]
            ## Saturation is the presence of rgb, at 0 the image is gray scale [0.6,4] 4 is very pink
            ## Hue is between -0.15 and 0.15

            if args['state'] == 'train':
                augment_random = np.random.choice([0, 1], size=1, replace=False, p=[0.5, 0.5])[0]
                if augment_random == 1:
                    affine_transformed_img = jitter(affine_transformed_img)


        affine_transformed_img = affine_transformed_img.squeeze()
        affine_transformed_img = affine_transformed_img.unsqueeze(0).float() / torch.quantile(affine_transformed_img.unsqueeze(0).float(), 0.99, dim=None)

        label_[i] = torch.tensor(lbl)

        image_[i] = affine_transformed_img[0,:,:,:]

    return image_, label_

def condition(scr,cls):
    if cls == 0 and  scr < 0.33:
        return True
    elif cls == 0.5 and scr>=0.33 and scr <= 0.67:
        return True
    elif cls == 1 and scr > 0.68:
        return True
    else:
        return False

def canine_data_load( **args):

    # Empty image and label tensors
    if args['model'] == 'diffusion':
        image_ = torch.empty(size=(args['batch_size'], 3, args['tile_size'], args['tile_size']), device=args['device'])
        label_ = torch.empty(size=(args['batch_size'], 1), device=args['device'])
    else:
        image_ = torch.empty(size=(args['batch_size'], 3, 256, 256), device=args['device'])
        label_ = torch.empty(size=(args['batch_size'], 1), device=args['device'])

    # Ratio of mitotic figures in a batch
    ratio = float(args['ratio'])

    for i in range(args['batch_size']):

        if args['soft_vs_hard'] == 'soft':
            classes = [0, 0.5, 1]
        else:
            classes = [0, 1]
        cls = np.random.choice(classes)
        probabilities = []

        if args['state'] == 'train' or args['state'] == 'validation':
            for slide_select in list(args['slides'].keys()):
                try:
                    probabilities.append(len(args['slides'][slide_select][args['state']][cls]))
                except:
                    probabilities.append(0)
            probabilities = probabilities/np.sum(probabilities)
            slide_select = np.random.choice(list(args['slides'].keys()), p=probabilities)

        bboxes = args['slides'][slide_select][args['state']][cls]

        # Chose a random example from the selected class
        indice = np.random.choice(np.arange(len(bboxes)),1)[0]
        bbox = bboxes[indice]
        center_x = bbox[0]+int((bbox[2]-bbox[0])/2)
        center_y = bbox[1] + int((bbox[3] - bbox[1]) / 2)
        x, y = center_x-int(args['tile_size']), center_y-int(args['tile_size'])

        # Read the current tile that has the annotated cell in the center
        img = np.array(args['slides'][slide_select]['WSI'].read_region(location=(x,y), level=0, size=((args['tile_size']*2, args['tile_size']*2))))[:, :, :3]
        lbl = cls

        # Random augmentation on the images
        crop = torchvision.transforms.CenterCrop(args['tile_size'])
        random_angle = np.random.choice(np.arange(270), 1)[0]
        random_translate = [0,0]
        augment_random = np.random.choice([0, 1],1)[0]#, size=1, replace=False, p=[0.5, 0.5])[0]
        if augment_random == 1:

            affine_transformed_img = torchvision.transforms.functional.affine(torch.tensor(img).permute(2, 0, 1),
                                                                              angle=int(random_angle),
                                                                              translate=(
                                                                              random_translate[0], random_translate[1]),
                                                                              scale=1, shear=0)
        else:
            affine_transformed_img = torch.tensor(img).permute(2, 0, 1)
        affine_transformed_img = crop(affine_transformed_img)
        affine_transformed_img = affine_transformed_img.float()/255.

        if 'resnet' in args['model']:

            if int(args['tile_size']) < 256:
                affine_transformed_img = torch.nn.functional.interpolate(affine_transformed_img.unsqueeze(0).float(),
                                                                         scale_factor=(4, 4),
                                                                         mode='bilinear')
            affine_transformed_img = affine_transformed_img.squeeze()

            jitter = torchvision.transforms.ColorJitter(brightness=[0.4, 1.2], contrast=[0.5, 4], saturation=[0.6, 3],
                                                        hue=[-0.15, 0.15])
            #jitter = torchvision.transforms.ColorJitter(brightness=[1,1], contrast=[1,1], saturation=[1,1], hue=[0,0])
            ## This augmentation does not change the image at all
            ## Brightness is the amount of light, it should never be 0 or much higher than 1 [0.4,1.2]
            ## Contrast is the amount that the colors are apart from each other, at 0 it is all gray, [0.5,4]
            ## Saturation is the presence of rgb, at 0 the image is gray scale [0.6,4] 4 is very pink
            ## Hue is between -0.15 and 0.15

            if args['state'] == 'train':
                augment_random = np.random.choice([0, 1], size=1, replace=False, p=[0.5, 0.5])[0]
                if augment_random == 1:
                    affine_transformed_img = jitter(affine_transformed_img)

        affine_transformed_img = affine_transformed_img.squeeze()
        affine_transformed_img = affine_transformed_img.unsqueeze(0).float() / torch.quantile(affine_transformed_img.unsqueeze(0).float(), 0.99, dim=None)

        label_[i] = torch.tensor(lbl)
        image_[i] = affine_transformed_img[0,:,:,:]

    return image_, label_


def canine_data_load_test(**args):
    slide = args['slide']
    ## Empty image
    tile_size = args['tile_size']
    bbox = args['bbox']
    center_x = bbox[0]+int((bbox[2]-bbox[0])/2)
    center_y = bbox[1] + int((bbox[3] - bbox[1]) / 2)
    x, y = center_x-int(tile_size/2), center_y-int(tile_size/2)
    img = np.array(slide['WSI'].read_region(location=(x,y), level=0, size=((args['tile_size'], args['tile_size']))))[:, :, :3]

    if args['model'] == 'diffusion':
        scaled = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()
        img_ = scaled / torch.quantile(scaled.float(), 0.99, dim=None)

    else:
        scaled = torch.nn.functional.interpolate(torch.tensor(img).permute(2,0,1).unsqueeze(0).float(), scale_factor=(4,4),
                                                 mode='bilinear')
        img_ = scaled / torch.quantile(scaled.float(), 0.99, dim=None)

    return img_


def tsv_reader(tsv_path):
    tsv_file = open(tsv_path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    coords = []

    for row in read_tsv:
        try:
            row = row[0].split(',')
            [x,y] = np.array(row[0].split(" _ ")).astype(float)
            score = float(row[1])/float(row[2])
            coords.append([x,y,score])
        except Exception as e:
            print(e)
    return coords
