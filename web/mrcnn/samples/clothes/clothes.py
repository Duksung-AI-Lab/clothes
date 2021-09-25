import os
import sys
import numpy as np
import skimage.draw
from skimage.io import imread
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("/gdrive/My Drive/maskrcnn/MaskRCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR) ### 경로 확인

from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools import mask as maskUtils
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.model import MaskRCNN

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs") ### 경로 확인

############################################################
#  Configurations
############################################################

class ClothesConfig(Config):
    """Configuration for training on Clothes Dataset(Top, Botton category segmentation custom dataset)
    Derives from the base Config class and overrides values specific
    to the Clothes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "clothes"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + category(top, bottom)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200

    USE_MINI_MASK = True

    # train, val image & annotations path ### 경로 확인
    train_img_dir = "/gdrive/My Drive/maskrcnn/MaskRCNN/samples/clothes/clothes_dataset/train/JPEGImages/"
    train_json_path = "/gdrive/My Drive/maskrcnn/MaskRCNN/samples/clothes/clothes_dataset/train/annotations.json"
    valid_img_dir = "/gdrive/My Drive/maskrcnn/MaskRCNN/samples/clothes/clothes_dataset/val/JPEGImages/"
    valid_json_path = "/gdrive/My Drive/maskrcnn/MaskRCNN/samples/clothes/clothes_dataset/val/annotations.json"
    test_img_dir="/gdrive/My Drive/maskrcnn/MaskRCNN/samples/clothes/clothes_dataset/test/JPEGImages/"
    test_json_path="/gdrive/My Drive/maskrcnn/MaskRCNN/samples/clothes/clothes_dataset/test/annotations.json"

class InferenceConfig(ClothesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

############################################################
#  Dataset
############################################################

class ClothesDataset(utils.Dataset):
    def load_coco(self, image_dir, json_path, class_ids=None,
                  class_map=None, return_coco=False):
        """Load the Clothes dataset.
        """

        coco = COCO(json_path)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("clothes", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "clothes", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco


    def load_keypoint(self, image_id):
        """
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "clothes":
            return super(ClothesDataset, self).load_mask(image_id)

        instance_keypoints = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id(
                "clothes.{}".format(annotation['category_id']))
            if class_id:
                keypoint = annotation['keypoints']

                instance_keypoints.append(keypoint)
                class_ids.append(class_id)

        keypoints = np.stack(instance_keypoints, axis=1)
        class_ids = np.array(class_ids, dtype=np.int32)
        return keypoints, class_ids

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "clothes":
            return super(ClothesDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "clothes.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(ClothesDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        super(ClothesDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

############################################################
#  Train
############################################################

def train(model, config):
    """
    train using clothes dataset
    """
    dataset_train = ClothesDataset()
    dataset_train.load_coco(config.train_img_dir, config.train_json_path)
    dataset_train.prepare()

    dataset_valid = ClothesDataset()
    dataset_valid.load_coco(config.valid_img_dir, config.valid_json_path)
    dataset_valid.prepare()

    model.train(dataset_train, dataset_valid,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='3+')

############################################################
#  Splash
############################################################

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

############################################################
#  model evaluation
############################################################

def evaluation(model, config):

    # read dataset
    dataset_test = ClothesDataset()
    dataset_test.load_coco(config.test_img_dir, config.test_json_path)
    dataset_test.prepare()

    image_ids = np.random.choice(dataset_test.image_ids, 10, replace=False) ### 평가하고자 하는 이미지 개수 지정 / 수정
    inference_config = InferenceConfig()

    # < compute VOC-Style mAP @ IoU=0.5 > #
    # Running on 1743(all val image) images.
    APs=[]
    ARs=[]

    for image_id in image_ids:
        # load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_test, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute mAP(mean Average Precision), AR
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        AR, positive_ids = utils.compute_recall(r["rois"], gt_bbox, iou=0.5) # iou 0.5: PASCAL VOC metric
        ARs.append(AR)
        APs.append(AP)

        mAP = np.mean(APs)
        mAR = np.mean(ARs)

    print("** mAP: ", mAP)
    print("** mAR: ", mAR)

    F1_score = 2*(mAP*mAR)/(mAP+mAR)

    print("** F1_score:", F1_score)
    print("** overlaps: ", overlaps)

    # < Precision-Recall > #
    # draw precision-recall curve
    visualize.plot_precision_recall(AP, precisions, recalls)

    # Grid of ground truth objects and their predictions
    visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
                            overlaps, dataset_test.class_names)

###  test & splash
def detect_and_color_splash(model, config):
        dataset_test = ClothesDataset()
        dataset_test.load_coco(config.test_img_dir, config.test_json_path)
        dataset_test.prepare()


        ### 경로 지정 (test dataset path) 수정
        test_img_dir = "/gdrive/My Drive/maskrcnn/MaskRCNN/samples/clothes/clothes_dataset/test/JPEGImages/"
        images=os.listdir(test_img_dir)

        print(images)
        # Run model detection and generate the color splash effect
        i=0
        for image_file_name in images:
            print("Running on {}".format(image_file_name))
            # Read image
            image = skimage.io.imread(test_img_dir + image_file_name)
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            # Color splash
            splash = color_splash(image, r['masks'])

            # Save output 수정
            splash_file_name = ROOT_DIR+"/samples/clothes/splash_result/splash_{}.jpg".format(image_file_name)
            skimage.io.imsave(splash_file_name, splash)

            # detect visualize
            ax = get_ax(1)
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset_test.class_names, image_file_name, r['scores'], ax=ax)

            print("Splash, detect output Saved to ", image_file_name)
            i+=1

############################################################
#  main
############################################################

if __name__ == "__main__":

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Match R-CNN for Clothes Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    """
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    """
    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ClothesConfig()
    else:
       config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = MaskRCNN(mode="training", config=config,
                         model_dir=args.logs)
    else:
        model = MaskRCNN(mode="inference", config=config,
                         model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, config)
    elif args.command == "splash":
        evaluation(model, config)
        detect_and_color_splash(model, config)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

