
# Important features and hyperparameters of MAsk RCNN

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:

## Loss

rpn_class_loss : How well the Region Proposal Network separates background with objetcs
rpn_bbox_loss : How well the RPN localize objects
mrcnn_bbox_loss : How well the Mask RCNN localize objects
mrcnn_class_loss : How well the Mask RCNN recognize each class of object
mrcnn_mask_loss : How well the Mask RCNN segment objects

### More in depth
Rpn_class_loss: This corresponds to the loss that is to assigned to improper classification of anchor boxes (presence/absence of any object) by Region proposal network. This should be increased when multiple objects are not being detected by the model in the final output. Increasing this ensures that region proposal network will capture it.
Rpn_bbox_loss: This corresponds to the localization accuracy of the RPN. This is the weight to tune in case, the object is being detected but the bounding box should be corrected
Mrcnn_class_loss: This corresponds to the loss that is assigned to improper classification of object that is present in the region proposal. This is to be increased in case the object is being detected from the image, but misclassified
Mrcnn_bbox_loss: This is the loss, assigned on the localization of the bounding box of the identified class, It is to be increased if correct classification of the object is done, but localization is not precise
Mrcnn_mask_loss: This corresponds to masks created on the identified objects, If identification at pixel level is of importance, this weight is to be increased
The above Hyper-parameters are represented on the block diagram in the following figure.

## Hyper-parameters

https://medium.com/analytics-vidhya/taming-the-hyper-parameters-of-mask-rcnn-3742cb3f0e1b

The following are few Hyper-parameters specific to Mask R-CNN

### Back Bone
The Backbone is the Conv Net architecture that is to be used in the first step of Mask R-CNN. The available options for choice of Backbones include ResNet50, ResNet101, and ResNext 101. This choice should be based on the trade off between training time and accuracy. ResNet50 would take relatively lesser time than the later ones, and has several open source pre-trained weights for huge data sets like coco, which can considerably reduce the training time for different instance segmentation projects. ResNet 101 and ResNext 101 will take more time for training (because of the number of layers), but they tend to be more accurate if there are no pre-trained weights involved and basic parameters like learning rate and number of epochs are well tuned.

An ideal approach would be to start with pre-trained weights available like coco with ResNet 50 and evaluate the performance of the model. This would work faster and better on models which involve detection of real world objects which were trained in the coco dataset. If accuracy is of utmost importance and high computation power is available, the options of ResNet101 and ResNeXt 101 can be explored.

### Train_ROIs_Per_Image
This is the maximum number of ROI’s, the Region Proposal Network will generate for the image, which will further be processed for classification and masking in the next stage. The ideal way is to start with default values if number of instances in the image are unknown. If the number of instances are limited, it can be reduced to reduce the training time.

### Max_GT_Instances
This is the maximum number of instances that can be detected in one image. If the number of instances in the images are limited, this can be set to maximum number of instances that can occur in the image. This helps in reduction of false positives and reduces the training time.

### Detection_Min_Confidence
This is the confidence level threshold, beyond which the classification of an instance will happen. Initialization can be at default and reduced or increased based on the number of instances that are detected in the model. If detection of everything is important and false positives are fine, reduce the threshold to identify every possible instance. If accuracy of detection is important, increase the threshold to ensure that there are minimal false positive by guaranteeing that the model predicts only the instances with very high confidence.

### Image_Min_Dim and Image_Max_Dim
The image size is controlled by these settings. The default settings resize images to squares of size 1024x1024. Smaller images can be used (512x512) can be used to reduce memory requirements and training time. The ideal approach would be to train all the initial models on smaller image sizes for faster updation of weights and use higher sizes during final stage to fine tune the final model parameters.

### Loss weights
Mask RCNN uses a complex loss function which is calculated as the weighted sum of different losses at each and every state of the model. The loss weight hyper parameters corresponds to the weight that the model should assign to each of its stages.
