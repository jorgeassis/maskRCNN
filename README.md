### Artificial Intelligence Convolutional Neural Networks maps giant kelp forests from satellite imagery

Marquez, L.; Fragkopoulou, E.; Serrão, E.A.; Cavanaugh, K.; Houskeeper, H.; E.; Assis, J.

<hr style="border: 0.5px">

### Rationale
Ongoing climate change is producing shifts in the distribution and abundance of numerous marine species. Such is the case of marine forests of large brown algae, important ecosystem-structuring species whose low latitude distributional ranges have been shifting across the world. Synthesizing robust time series with long-term observations of these species is therefore vital to understand the drivers shaping ecosystem dynamics and predict responses to ongoing and future climate changes. 

### Approach
Here we demonstrate the use of Mask Regional-Convolutional Neural Networks (MRCNN) developed with Keras and TensorFlow to automatically assimilate data from open-source satellite imagery (Landsat Thematic Mapper) and predict potential regions with marine forests. The analyses focused on the giant kelp Macrocystis pirifera along the shorelines of southern California in the northeastern Pacific, where the MRCNN model accurately identified the floating canopies of the species and determined distributional areas through time (from 1984 onwards). The use of this data on explanatory modelling can contribute to a better understanding of ongoing ecosystem dynamics assisting on future projections. Furthermore, this information can be used to improve management plans and identify avenues for future research.

<br>

![plot](./Figure.png)
Example of 3 pseudo-RGB composites used in independent cross-validation. (left panels) Observed floating canopies of giant kelp (depicted in red). (central panels) Manual annotations of giant kelp made by experts (depicted in red). (right panels) Predicted giant kelp forests with Mask R-CNN (depicted in yellow). Performance of predictions is shown with Jaccard’s index and Dice coefficient.

<hr style="border:1px dashed">
