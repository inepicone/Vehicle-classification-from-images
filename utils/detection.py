
from detectron2        import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from torchvision.ops   import box_area

# configuration:
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu" 
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

# inicialization:
DET_MODEL = DefaultPredictor(cfg)

def get_vehicle_coordinates(img):
    """
    This function will run an object detector over the the image, get
    the vehicle position in the picture and return it.

    Many things should be taken into account to make it work:
        1. Current model being used can detect up to 80 different objects,
           we're only looking for 'cars' or 'trucks', so you should ignore
           other detected objects.
        2. The object detector may find more than one vehicle in the picture,
           you must then, choose the one with the largest area in the image.
        3. The model can also fail and detect zero objects in the picture,
           in that case, you should return coordinates that cover the full
           image, i.e. [0, 0, width, height].
        4. Coordinates values must be integers, we're making reference to
           a position in a numpy.array, we can't use float values.

    Parameters
    ----------
    img : numpy.ndarray
        Image in RGB format.

    Returns
    -------
    box_coordinates : tuple
        Tuple having bounding box coordinates as (left, top, right, bottom).
        Also known as (x1, y1, x2, y2).
    """
   
    outputs = DET_MODEL(img)

    # We create a mask:   
    mask = (outputs["instances"].pred_classes == 2)|(outputs["instances"].pred_classes == 7)

    # we calculate the x,y for those who are car o truck:   
    box_pred = outputs["instances"].pred_boxes[mask]
    
    # In the cases the picture dont have car or truck --> return the picture
    if box_pred.tensor.size()[0]==0:
      box_coordinates = [0,0,img.shape[1],img.shape[0]]

    else: #outputs['instances'][i].pred_classes==2 or outputs['instances'][i].pred_classes==7:
      box_xy = outputs["instances"].pred_boxes
      biggest_box = box_area(box_xy.tensor)
      max = 0
      for j in range(biggest_box.shape[0]):
        if biggest_box[j]>=max:
          max = biggest_box[j]
          big_box_coord = box_xy[j]

      x1, y1, x2, y2 = big_box_coord.tensor.cpu().numpy()[0][:4]

      # we retun int
      box_coordinates = (int(x1), int(y1), int(x2), int(y2))

    return box_coordinates