## Assignment 4 - Segmentation and Object Detection:
Within engineering the use of segmentation and object detection have many applications. For example, segmentation can be used to obtain the approximate vapor fraction from boiling images. This assignment will have you using a couple popular semantic segmentation and object detection models.
For semantic segmentation, you will use boiling images and corresponding masks made using labelme and a popular model UNET (which is provided here so you can just copy and put directly into your code): 

```python
def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)
    
    # Contracting path (Encoder)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    
    # Expansive path (Decoder)
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.Concatenate()([u5, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.Concatenate()([u6, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.Concatenate()([u7, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    
    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)  # Binary output for binary segmentation
    
    model = models.Model(inputs, outputs)
    
    return model
```

For object detection, you will be using a dataset of your choice from [here](https://public.roboflow.com/object-detection) and the model yolov8 from ultralytics. 

---
#### INTRODUCTION
Object detection and segmentation are two computer vision tasks. Object detection is the task of identifying and locating specific objects within an image. These models typically return bounding boxes that capture the region of interest. Segmentation, on the other hand gives a full mask of the object, it specifies which class each pixel belongs too.

#### OBJECT DETECTION
One popular model is the yolo model. There are several yolo models and actually not all of them are soley object detection models. 
