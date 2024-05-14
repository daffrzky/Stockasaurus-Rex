import yolov5
import cv2

# load pretrained model
model = yolov5.load('yolov5s.pt')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4].detach().cpu().numpy()
    scores = predictions[:, 4].detach().cpu().numpy()
    categories = predictions[:, 5].detach().cpu().numpy()
    
    for box,score,category in zip(boxes, scores, categories):
        x1, y1, x2, y2 = box.astype(int)
        label = model.names[int(category)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLOv5 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()