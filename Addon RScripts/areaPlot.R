# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
#
#
# ----------------------------------------------------------------------

closeAllConnections()
rm(list=(ls()[ls()!="v"]))
gc(reset=TRUE)

library(raster)
library(ggplot2)
library(ncdf4)

listTiles <- list.files("../../Results/Landsat_037041_Detection/Matrices/", full.names = TRUE)
listTilesNames <- list.files("../../Results/Landsat_037041_Detection/Matrices/", full.names = FALSE)
length(listTiles)

# ---------------

datasetStructure <- data.frame()

for(i in 1:length(listTiles)) {
  
  fileName <- gsub("LC08_037041_","",listTilesNames[i])
  fileName <- gsub("LE07_037041_","",fileName)
  fileName <- gsub("LT05_037041_","",fileName)
  fileName <- gsub("LT04_037041_","",fileName)
  fileName <- gsub("\\.csv","",fileName)
  
  year <- as.numeric(substr(fileName,1,4))
  month <- as.numeric(substr(fileName,5,6))
  day <- as.numeric(substr(fileName,7,8))
  
  tile <- as.numeric(substr(fileName,unlist(gregexec("_",fileName))+1,nchar(fileName)))
  
  datasetStructure.i <- data.frame(year=year, month=month, day=day, tile=tile, file=listTiles[i])
  datasetStructure <- rbind(datasetStructure,datasetStructure.i)
  
}

which(is.na(datasetStructure$year))

# ---------------

tilesPerYear <- aggregate(datasetStructure$file, list(datasetStructure$year), length)
mean(tilesPerYear[tilesPerYear[,1] >= 1990,2])
sd(tilesPerYear[tilesPerYear[,1] >= 1990,2])
range(tilesPerYear[tilesPerYear[,1] >= 1990,2])

# ---------------

dataset <- data.frame(year=sort(unique(datasetStructure$year)), images=NA, area=NA)

for(y in sort(unique(datasetStructure$year)) ) {
  
  datasetStructure.y <- datasetStructure[which(datasetStructure$year == y),]
  results.i <- numeric(0)
  
  for( tile in unique(datasetStructure.y$tile)) {
    
        files <- datasetStructure.y[which(datasetStructure.y$tile == tile),"file"]
        stacked <- array(NA,dim=c(1024,1024,length(files)))
        
        for( f in files) {
          matrix <- read.csv(f, header=F)
          matrix[matrix != 0 ] <- 1
          # plot(raster(as.matrix(matrix)))
          stacked[ , , which(files == f) ] <- as.matrix(matrix)
        }
        
        stacked[ is.na(stacked)] <- 0
        stacked <- apply(stacked,1:2,sum)
        stacked[stacked != 0 ] <- 1
        results.i <- c(results.i,sum(stacked != 0, na.rm=T) * 30 * 30)
  }
  
  dataset[ which(dataset$year == y),"images"] <- length(files)
  dataset[ which(dataset$year == y),"area"] <- sum(results.i,na.rm=T)
  
}

datasetBk <- dataset

# ---------------

mainTheme <- theme(panel.grid.major = element_blank() ,
                   text = element_text(size=12) ,
                   axis.title.y = element_text(margin = margin(t = 0, r = 16, b = 0, l = 0)) ,
                   axis.title.x = element_text(margin = margin(t = 16, r = 0, b = 0, l = 0)) )

dataset <- dataset[dataset$year >= 1990,]
dataset$areaHa <- dataset$area * 0.0001
  
plot1 <- ggplot(dataset, aes(x=year, y=areaHa)) + 
  scale_x_continuous("Time (year)", labels = as.character(c(seq(1990,2021,by=3),2021)), breaks = c(seq(1990,2021,by=3),2021)) +
  geom_bar(stat = "identity", fill="#61A5C7") + theme_minimal(base_size = 14) + mainTheme + 
  ylab("Maximum kelp coverage (hectare)") +
  annotate("segment", x = 1991, y = -40, xend = 1991, yend = -20,color="#CE6B6B",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 1992, y = -40, xend = 1992, yend = -20,color="#CE6B6B",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 1997, y = -40, xend = 1997, yend = -20,color="#8B1010",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 1998, y = -40, xend = 1998, yend = -20,color="#8B1010",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 2015, y = -40, xend = 2015, yend = -20,color="#8B1010",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 2016, y = -40, xend = 2016, yend = -20,color="#8B1010",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  
  annotate("segment", x = 2014, y = -40, xend = 2014, yend = -20,color="#8B1010",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) +
  annotate("segment", x = 2014, y = -40, xend = 2014, yend = -20,color="#CE6B6B",arrow = arrow(type = "closed", length = unit(0.02, "npc"))) 
#+ geom_text(data=data.frame(x = c(2014.75,2014.75),y = c(5950,6300),label = c("Very Strong El Niño", "Strong El Niño")), aes( x=x, y=y, label=label) , color="black", size=4 , angle=0 , hjust = 0)

plot1

pdf( file=paste0("Area 2.pdf"), width = 12, height = 6 )
print(plot1)
dev.off()

library(ncdf4)

dataset2 <- "../../Results/Landsat_037041_Detection/kelpCoverage_11N.nc"
dataset2 <- nc_open( dataset2, readunlim=FALSE )
dataset2Years <- ncvar_get( dataset2, "year" ) 
dataset2Location <- ncvar_get( dataset2, "Location" ) 
dataset2Var <- ncvar_get( dataset2, "annual_max" ) 
dataset2Var[dataset2Var < 0] <- 0
dataset2Var <- apply(dataset2Var,2,sum)
nc_close(dataset2)

dataset2Var <- dataset2Var[dataset2Years %in% dataset$year]
dataset2Years <- dataset2Years[dataset2Years %in% dataset$year]
dataset2Years

dataset1Var <- dataset$area[dataset$year %in% dataset2Years]
plot(dataset1Var * 0.001,dataset2Var)
cor(dataset1Var,dataset2Var)










# Older version

dataset <- read.table("../Results/Landsat_037041_Detection/areaEstimates.csv",sep=",", header=T)
head(dataset)

dataset$year <- as.numeric(substr(dataset$Image,13,16))
dataset$month <- as.numeric(substr(dataset$Image,17,18))
dataset$day <- as.numeric(substr(dataset$Image,19,20))
dataset$date <- paste0(as.numeric(substr(dataset$Image,13,16)),"-",as.numeric(substr(dataset$Image,17,18)),"-",as.numeric(substr(dataset$Image,19,20)))
head(dataset)

dataset.aggregated <- aggregate(dataset$Area, by = list( substr(dataset$Image,22,23) , dataset[,4] ), FUN = "max")
dataset.aggregated <- aggregate(dataset.aggregated$x, by = list( dataset.aggregated$Group.2 ), FUN = "sum")
dataset.aggregated[ dataset.aggregated[,2] == 0 ,2] <- 20
range(dataset.aggregated)

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

library(ncdf4)

dataset2 <- "../Results/Landsat_037041_Detection/kelpCoverage_11N.nc"
dataset2 <- nc_open( dataset2, readunlim=FALSE )
dataset2Years <- ncvar_get( dataset2, "year" ) 
dataset2Location <- ncvar_get( dataset2, "Location" ) 
dataset2Var <- ncvar_get( dataset2, "annual_max" ) 
dataset2Var[dataset2Var < 0] <- 0
dataset2Var <- apply(dataset2Var,2,sum)
nc_close(dataset2)

dataset2Var <- dataset2Var[dataset2Years %in% dataset.aggregated$Group.1]
dataset2Years <- dataset2Years[dataset2Years %in% dataset.aggregated$Group.1]
dataset2Years

dataset1Var <- dataset.aggregated$x[dataset.aggregated$Group.1 %in% dataset2Years]
dataset1Var <- (dataset1Var / max(dataset1Var) ) * 25000
plot(dataset1Var,dataset2Var)
cor(dataset1Var,dataset2Var)



