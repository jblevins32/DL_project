import torch

class GetData():
    '''
    Parse the data from the YOLO model
    
    Args:
        results: result object of YOLO model from one input image
        ID_list: current list of data
        
    Returns:
        ID_list: updated list of data COLUMNS ARE: Object ID, class assignment, confidence, bounding box coordinates (norm: x topleft, y topleft, x botton right, y bottom right)
    '''
    
    def AddGetData(self, results, ID_list):
        # Gather data on where the tracked items are
        boxes = results[0].boxes.xyxyn # This is the normalized positions of the bounding boxes
        IDs = results[0].boxes.id
        classes = results[0].boxes.cls
        conf = results[0].boxes.conf

        # Remove IDs from the ID tensor that have already been logged. Vectorized for speed.
        ID_remove = torch.isin(IDs,ID_list)
        
        IDs = IDs[~ID_remove].unsqueeze(1)
        classes = classes[~ID_remove].unsqueeze(1)
        conf = conf[~ID_remove].unsqueeze(1)
        boxes = boxes[~ID_remove]

        # Adding all info to list
        return torch.cat((ID_list, torch.cat((IDs,classes,conf,boxes),dim=1)),dim=0)