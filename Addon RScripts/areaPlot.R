
dataset <- read.table("/Volumes/Bathyscaphe/Mask RCNN for Kelp Detection/TimeSeriesData/Landsat_037041_Detection/areaEstimates.csv",sep=",", header=T)
head(dataset)

dataset$year <- as.numeric(substr(dataset$Image,13,16))
dataset$month <- as.numeric(substr(dataset$Image,17,18))
dataset$day <- as.numeric(substr(dataset$Image,19,20))
dataset$date <- paste0(as.numeric(substr(dataset$Image,13,16)),"-",as.numeric(substr(dataset$Image,17,18)),"-",as.numeric(substr(dataset$Image,19,20)))
head(dataset)

dataset.aggregated <- aggregate(dataset$Area, by = list( dataset[,4] ), FUN = "max")
dataset.aggregated[ dataset.aggregated[,2] == 0 ,2] <- 20

dataset.aggregated <- dataset.aggregated[dataset.aggregated$Group.1 >= 1989,]
dataset.aggregated

library(ggplot2)

mainTheme <- theme(panel.grid.major = element_blank() ,
                   text = element_text(size=12) ,
                   axis.title.y = element_text(margin = margin(t = 0, r = 16, b = 0, l = 0)) ,
                   axis.title.x = element_text(margin = margin(t = 16, r = 0, b = 0, l = 0)) )

dataset.aggregated <- dataset.aggregated[dataset.aggregated$Group.1 >= 1990,]

plot1 <- ggplot(dataset.aggregated, aes(x=Group.1, y=x)) + 
  scale_x_continuous("Time (year)", labels = as.character(c(seq(1990,2021,by=3),2021)), breaks = c(seq(1990,2021,by=3),2021)) +
  geom_bar(stat = "identity", fill="#61A5C7") + theme_minimal(base_size = 14) + mainTheme + 
  ylab("Maximum kelp coverage (m2)") +
  annotate("segment", x = 1991, y = -40, xend = 1991, yend = -20,color="#CE6B6B",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 1992, y = -40, xend = 1992, yend = -20,color="#CE6B6B",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 1997, y = -40, xend = 1997, yend = -20,color="#8B1010",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 1998, y = -40, xend = 1998, yend = -20,color="#8B1010",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 2015, y = -40, xend = 2015, yend = -20,color="#8B1010",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 2016, y = -40, xend = 2016, yend = -20,color="#8B1010",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +

  annotate("segment", x = 2014, y = 6000, xend = 2014, yend = 6020,color="#8B1010",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 2014, y = 6350, xend = 2014, yend = 6370,color="#CE6B6B",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) + 
  geom_text(data=data.frame(x = c(2014.75,2014.75),y = c(5950,6300),label = c("Very Strong El Niño", "Strong El Niño")), aes( x=x, y=y, label=label) , color="black", size=4 , angle=0 , hjust = 0)

plot1

pdf( file=paste0(resultsFolder," Area.pdf"), width = 12, height = 6 )
print(plot1)
dev.off()
