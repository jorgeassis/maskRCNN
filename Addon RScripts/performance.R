
library(ggplot2)
library(hrbrthemes)
library(gridExtra)

experimentsFolder <- "/Volumes/Bathyscaphe/Mask RCNN for Kelp Detection/Experiments/" # "/Volumes/Jellyfish/Dropbox/Manuscripts/Convolutional Neural Networks for kelp canopy identification/Results/Experiments/"
resultsFolder <- "/Volumes/Jellyfish/Dropbox/Manuscripts/Convolutional Neural Networks for kelp canopy identification/Results/" 

# ---------------------------------------------

mainTheme <- theme(panel.grid.major = element_blank() ,
                   text = element_text(size=12) ,
                   axis.title.y = element_text(margin = margin(t = 0, r = 14, b = 0, l = 0)) ,
                   axis.title.x = element_text(margin = margin(t = 14, r = 0, b = 0, l = 0)) 
                   
                   )

# ---------------------------------------------

# Loss functions

for( i in 1:8) {
  
  experiment <- paste0("J0",i)
  lossResults <- read.table(paste0(experimentsFolder,experiment,"/historyExperimentAll.csv"), sep=",", header=T)
  lossResults <- data.frame(epoch=1:50,lossResults)
  
  p1 <- ggplot() +
    geom_vline(xintercept = 10, linetype="dashed", color = "#919191", size=0.35) +
    geom_line( data=lossResults , aes(x=epoch, y=val_loss) , color="grey") +
    geom_point( data=lossResults , aes(x=epoch, y=val_loss) , shape=21, color="black", fill="#13a9a9", size=3.5) +
    geom_point( data=lossResults[which.min(lossResults$val_loss),] , aes(x=epoch, y=val_loss) , shape=21, color="black", fill="#A91313", size=4) +
    theme_minimal() + mainTheme + 
    xlab("Epoch [n]") + ylab("Loss in independent data") + ggtitle("Overall loss", subtitle = paste0("Grid search experiment [",i,"]"))
  
  p2 <- ggplot() +
    geom_line( data=lossResults , aes(x=epoch, y=val_rpn_bbox_loss) , color="grey") +
    geom_point( data=lossResults , aes(x=epoch, y=val_rpn_bbox_loss) , shape=21, color="black", fill="#13a9a9", size=3.5) +
    theme_minimal() + mainTheme + 
    xlab("Epoch [n]") + ylab(NULL) + ggtitle("BBox loss")
  
  p3 <- ggplot() +
    geom_line( data=lossResults , aes(x=epoch, y=val_mrcnn_mask_loss) , color="grey") +
    geom_point( data=lossResults , aes(x=epoch, y=val_mrcnn_mask_loss) , shape=21, color="black", fill="#13a9a9", size=3.5) +
    theme_minimal() + mainTheme + 
    xlab("Epoch [n]") + ylab(NULL) + ggtitle("Mask loss")
  
  pdf( file=paste0(resultsFolder,experiment," Losses.pdf"), width = 20, height = 8 )
  grid.arrange(p1, p2, p3, nrow = 1)
  dev.off()
  
  pdf( file=paste0(resultsFolder,experiment," OverallLoss.pdf"), width = 12, height = 6 )
  print(p1)
  dev.off()
   
}


# ---------------------------------------------

TestJaccard <- data.frame(matrix(NA,ncol=8,nrow=8))
TestDice <- data.frame(matrix(NA,ncol=8,nrow=8))
SummaryResults <- data.frame()

for( i in 1:8) {
  
  experiment <- paste0("J0",i)
  
  bestThresholdResult <- read.table(paste0(experimentsFolder,experiment,"/accuracyThresholdTest.csv"), sep=",", header=T)
  bestThresholdResult.i <- which(bestThresholdResult$indexJaccard == max(bestThresholdResult$indexJaccard))
  bestThresholdResult.i <- bestThresholdResult.i[length(bestThresholdResult.i)]
  
  rawDataJaccard.i <- read.table(paste0(experimentsFolder,experiment,"/accuracyRaw_",bestThresholdResult$Threshold[bestThresholdResult.i],".csv"), sep=",", header=T)
  rawDataJaccard.i <- rawDataJaccard.i$indexJaccard
  rawDataJaccard.i.Mean <- mean(rawDataJaccard.i)
  rawDataJaccard.i.SD <- sd(rawDataJaccard.i)
  
  bestThresholdResult <- read.table(paste0(experimentsFolder,experiment,"/accuracyThresholdTest.csv"), sep=",", header=T)
  bestThresholdResult.i <- which(bestThresholdResult$indexDice == max(bestThresholdResult$indexDice))
  bestThresholdResult.i <- bestThresholdResult.i[length(bestThresholdResult.i)]
  
  rawDataDice.i <- read.table(paste0(experimentsFolder,experiment,"/accuracyRaw_",bestThresholdResult$Threshold[bestThresholdResult.i],".csv"), sep=",", header=T)
  rawDataDice.i <- rawDataDice.i$indexDice
  rawDataDice.i.Mean <- mean(rawDataDice.i)
  rawDataDice.i.SD <- sd(rawDataDice.i)
  
  SummaryResults <- rbind(SummaryResults,data.frame(JaccardMean = rawDataJaccard.i.Mean,JaccardSD=rawDataJaccard.i.SD, 
                                                    DiceMean = rawDataDice.i.Mean,DiceSD=rawDataDice.i.SD,
                                                    overPrediction = bestThresholdResult$areaDifference[bestThresholdResult.i] / bestThresholdResult$areaPredicted[bestThresholdResult.i]))
  
  for( j in 1:8) {
                                       
    if(i == j) { next }
    
    experiment <- paste0("J0",j)
    
    bestThresholdResult <- read.table(paste0(experimentsFolder,experiment,"/accuracyThresholdTest.csv"), sep=",", header=T)
    bestThresholdResult.j <- which(bestThresholdResult$indexJaccard == max(bestThresholdResult$indexJaccard))
    bestThresholdResult.j <- bestThresholdResult.j[length(bestThresholdResult.j)]
    
    rawDataJaccard.j <- read.table(paste0(experimentsFolder,experiment,"/accuracyRaw_",bestThresholdResult$Threshold[bestThresholdResult.i],".csv"), sep=",", header=T)
    rawDataJaccard.j <- rawDataJaccard.j$indexJaccard

    bestThresholdResult <- read.table(paste0(experimentsFolder,experiment,"/accuracyThresholdTest.csv"), sep=",", header=T)
    bestThresholdResult.j <- which(bestThresholdResult$indexDice == max(bestThresholdResult$indexDice))
    bestThresholdResult.j <- bestThresholdResult.j[length(bestThresholdResult.j)]
    
    rawDataDice.j <- read.table(paste0(experimentsFolder,experiment,"/accuracyRaw_",bestThresholdResult$Threshold[bestThresholdResult.i],".csv"), sep=",", header=T)
    rawDataDice.j <- rawDataDice.j$indexDice

    TestJaccard[i,j] <- wilcox.test(rawDataJaccard.i,rawDataJaccard.j, alternative="two.sided", paired = TRUE)$p.value <= (0.05)
    TestDice[i,j] <- wilcox.test(rawDataDice.i,rawDataDice.j, alternative="two.sided", paired = TRUE)$p.value <= (0.05)
    
  }
}

all.equal(TestJaccard,TestDice)

TestJaccard
TestDice
SummaryResults

