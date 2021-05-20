
# Summary of most important features and hyperparameters of MAsk RCNN

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:

## Loss

rpn_class_loss : How well the Region Proposal Network separates background with objetcs
rpn_bbox_loss : How well the RPN localize objects
mrcnn_bbox_loss : How well the Mask RCNN localize objects
mrcnn_class_loss : How well the Mask RCNN recognize each class of object
mrcnn_mask_loss : How well the Mask RCNN segment objects