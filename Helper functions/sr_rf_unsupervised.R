sr_data <- read.csv("sr_data.csv")
sr_data <- sr_data[sr_data$kpi_sr_details.sr_hw_product_erp_family=="CRS",]
sr_data <- sr_data[sr_data$kpi_sr_details.identifier=="NULL",]
sample <- sr_data[c(2,4,6,9,11,12,17,20,21,24,28,43,44,50,54,55,57,59,91,92,93,94)]
#18,27,33,45,19,10,22,23,25,36,39,56,58,
#2,4,6,9,10,11,12,15,16,17,19,20,21,22,23,24,25,28,36,39,43,44,50,54,55,56,57,58,59,91,92,93,94
library(randomForest)
sr_rf <- randomForest(sample, type=unsupervised, importance = TRUE, proximity = TRUE)
p <- sr_rf$proximity
#Max, Current sev, Product key, RMA count, Region name, Product family,  
a <- length(p[1,])
clusters <- list()
b <- p[1,]
clusters[1] <- list(names(b[b>=0.5]))

for (i in 2:a){
  b<-p[i,]
  c <- list(names(b[b>=0.5]))[[1]]
  status=0
  for(j in 1:length(clusters)){
    if(i %in% clusters[[j]]){
      status=1
    }
  }
  if(status==0){
    for(j in 1:length(clusters)){
      c <- setdiff(c, clusters[[j]])
    }
    if(!(identical(c, character(0)))){
      print(i)
      clusters[i] <- list(c) #list(names(b[b>=0.5]))[[1]]
    }
  }
}
lists <- Filter(Negate(is.null), clusters)

sr_data$Cluster <- ""
sample$Cluster <- ""

for(j in 1:nrow(sample)){
  for(i in 1:length(lists)){
    if(j %in% lists[[i]]){
      print(j)
      sample[j,]$Cluster <- i
      #sr_data[j,]$Cluster <- i
    }
  }
}

sample_data <- sample[!is.na(sample$Cluster),]
cluster1 = sample_data[sample_data$Cluster==1, ]
cluster2 = sample_data[sample_data$Cluster==2, ]
cluster3 = sample_data[sample_data$Cluster==3, ]
cluster4 = sample_data[sample_data$Cluster==4, ]
cluster5 = sample_data[sample_data$Cluster==5, ]
cluster6 = sample_data[sample_data$Cluster==6, ]
cluster7 = sample_data[sample_data$Cluster==7, ]
cluster8 = sample_data[sample_data$Cluster==8, ]
cluster9 = sample_data[sample_data$Cluster==9, ]
cluster10 = sample_data[sample_data$Cluster==10, ]
cluster11 = sample_data[sample_data$Cluster==11, ]
cluster12 = sample_data[sample_data$Cluster==12, ]
cluster13 = sample_data[sample_data$Cluster==13, ]
summary(cluster1)

Protoplot <- plot(cluster1[,4], cluster1[,7], xlab=names(cluster1)[4], ylab=names(cluster1)[7], main = "Cluster1")
Protoplot <- plot(cluster2[,4], cluster2[,7], xlab=names(cluster2)[4], ylab=names(cluster2)[7], main = "Cluster2")
Protoplot <- plot(cluster3[,4], cluster3[,7], xlab=names(cluster3)[4], ylab=names(cluster3)[7], main = "Cluster3")

library(ggplot2)
cluster1_data <- data.frame(Clusters=c("Cluster1","Cluster1","Cluster1","Cluster1","Cluster1","Cluster2","Cluster2","Cluster2","Cluster2","Cluster2","Cluster3","Cluster3","Cluster3","Cluster3","Cluster3","Cluster4","Cluster4","Cluster4","Cluster4","Cluster4","Cluster5","Cluster5","Cluster5","Cluster5","Cluster5","Cluster6","Cluster6","Cluster6","Cluster6","Cluster6","Cluster7","Cluster7","Cluster7","Cluster7","Cluster7"),
                            Values=c("CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE"), sizes=c(668,261,2468,0,295, 59,19,201,0,33, 9,5,53,0,7, 2,1,12,0,0, 2,1,21,0,1, 1,1,8,0,2, 3,0,7,0,1))
ggplot(data = cluster1_data, aes(x = Features, y = sizes, fill = Values)) + geom_bar(stat="identity")






#Cluster1
cluster1_data <- data.frame(Features=c("customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","sw_platform","sw_platform","sw_platform","sw_platform","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment"),
                            Values=c("CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CRS16 SERIES","CRS4 SERIES", "CRS8 SERIES", "CRSMC SERIES","SPARE", "SYSTEM", "SOFTWARE", "CRS-1", "ASR9000", "CRS-3", "HARDWARE_FAILURE", "ERROR_MESSAGES", "CONFIG_ASSISTANCE", "INSTALL_INSTLL_UPGRD", "SOFTWARE_FAILRE", "INTEROP", "OTHER", "COMM001", "COMM004","COMM002","UNK0900","COMM007","COMM009","OTHERS","XR-Routing-Platforms","Hardware","Other","Router and IOS Architecture","NMS (Network Management Services)", "Routing Protocols (Includes NAT and HSRP)", "Commercial", "Enterprise", "Service Provider", "Unknown Customer Market Segment"), sizes=c(668,261,2468,0,295, 1224,26,310,509,103,1616, 2730,4,3,2, 1693,1320,277,167,165,127,39, 2527,334,276,170,148,139,194, 3364,288,47,21,16,14, 24,12,78,18))
ggplot(data = cluster1_data, aes(x = Features, y = sizes, fill = Values)) + geom_bar(stat="identity")+ggtitle("Cluster 1")


#Cluster2
cluster2_data <- data.frame(Features=c("customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","sw_platform","sw_platform","sw_platform","sw_platform","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_tech_name","sr_tech_name","sr_tech_name","sr_tech_name","sr_tech_name","sr_tech_name","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment"),
                            Values=c("CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CRS16 SERIES","CRS4 SERIES", "CRS8 SERIES", "CRSMC SERIES","SPARE", "SYSTEM", "SOFTWARE", "CRS-1", "ASR9000", "CRS-3", "HARDWARE_FAILURE", "ERROR_MESSAGES", "CONFIG_ASSISTANCE", "INSTALL_INSTLL_UPGRD", "SOFTWARE_FAILRE", "INTEROP", "OTHER", "COMM001", "COMM004","COMM002","UNK0900","COMM007","COMM009","OTHERS","XR-Routing-Platforms","Hardware","NMS (Network Management Services)","Other", "Routing Protocols (Includes NAT and HSRP)","Broadband Cable", "Commercial", "Enterprise", "Service Provider", "Unknown Customer Market Segment"), sizes=c(59,19,201,0,33, 95,3,25,46,7,42, 221,0,0,1, 134,122,18,21,15,7,1, 197,34,29,13,17,14,14, 285,22,3,3,2,1, 1,1,9,0))
ggplot(data = cluster2_data, aes(x = Features, y = sizes, fill = Values)) + geom_bar(stat="identity")+ggtitle("Cluster 2")

#Cluster3
cluster3_data <- data.frame(Features=c("customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","sw_platform","sw_platform","sw_platform","sw_platform","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_tac_hw_family","sr_tac_hw_family","sr_tech_name","sr_tech_name","sr_tech_name","sr_tech_name","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment"),
                            Values=c("CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CRS16 SERIES","CRS4 SERIES", "CRS8 SERIES", "CRSMC SERIES","SPARE", "SYSTEM", "SOFTWARE", "ASR9000","CRS-1", "CRS-3", "ERROR_MESSAGES","HARDWARE_FAILURE", "CONFIG_ASSISTANCE","INTEROP", "INSTALL_INSTLL_UPGRD", "SOFTWARE_FAILRE",  "OTHER", "COMM001", "COMM004","COMM002","UNK0900","COMM008","COMM005","OTHERS","XR-Routing-Platforms","Hardware","Other","Application Networking Services","Broadband Cable","Contact Center Software","Commercial", "Enterprise", "Service Provider", "Unknown Customer Market Segment"), sizes=c(9,5,53,0,7, 17,1,5,11,0,41, 41,0,0,0, 33,28,6,4,2,2,0, 49,8,5,5,2,1,5, 67,6,2,0,0,0, 0,2,0,2))
ggplot(data = cluster3_data, aes(x = Features, y = sizes, fill = Values)) + geom_bar(stat="identity")+ggtitle("Cluster 3")

#Cluster4
cluster4_data <- data.frame(Features=c("customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","sw_platform","sw_platform","sw_platform","sw_platform","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment"),
                            Values=c("CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CRS16 SERIES","CRS4 SERIES", "CRS8 SERIES", "CRSMC SERIES","SPARE", "SYSTEM", "SOFTWARE", "ASR9000", "CRS-1", "CRS-3", "HARDWARE_FAILURE", "ERROR_MESSAGES","SOFTWARE_FAILRE", "CONFIG_ASSISTANCE", "DATA_CORRUPTION", "HARDWARE_DOA", "OTHER", "COMM001", "COMM002","COMM009","UNK0900","COMM003","COMM004","OTHERS","XR-Routing-Platforms","Hardware","Other","Router and IOS Architecture","NMS (Network Management Services)", "Routing Protocols (Includes NAT and HSRP)", "Commercial", "Enterprise", "Service Provider", "Unknown Customer Market Segment"), sizes=c(2,1,12,0,0, 5,0,0,1,0,9, 8,7,0,0,0, 7,7,1,0,0,0,0, 10,3,1,1,0,0,0, 15,0,0,0,0, 0,0,0,0))
ggplot(data = cluster4_data, aes(x = Features, y = sizes, fill = Values)) + geom_bar(stat="identity")+ggtitle("Cluster 4")

#Cluster5
cluster5_data <- data.frame(Features=c("customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","sw_platform","sw_platform","sw_platform","sw_platform","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment"),
                            Values=c("CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CRS16 SERIES","CRS4 SERIES", "CRS8 SERIES", "CRSMC SERIES","SPARE", "SYSTEM", "SOFTWARE", "ASR9000", "CRS-1", "CRS-3", "ERROR_MESSAGES", "HARDWARE_FAILURE", "CONFIG_ASSISTANCE", "INSTALL_INSTLL_UPGRD", "INTEROP", "DATA_CORRUPTION","OTHER", "COMM001", "COMM004","COMM002","UNK0900","COMM007","COMM009","OTHERS","XR-Routing-Platforms","Hardware","Other","Router and IOS Architecture","NMS (Network Management Services)", "Routing Protocols (Includes NAT and HSRP)", "Commercial", "Enterprise", "Service Provider", "Unknown Customer Market Segment"), sizes=c(668,261,2468,0,295, 1224,26,310,509,103,1616, 2730,4,3,2, 1693,1320,277,167,165,127,39, 2527,334,276,170,148,139,194, 3364,288,47,21,16,14, 24,12,78,18))
ggplot(data = cluster5_data, aes(x = Features, y = sizes, fill = Values)) + geom_bar(stat="identity")+ggtitle("Cluster 5")

#Cluster6
cluster6_data <- data.frame(Features=c("customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","sw_platform","sw_platform","sw_platform","sw_platform","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment"),
                            Values=c("CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CRS16 SERIES","CRS4 SERIES", "CRS8 SERIES", "CRSMC SERIES","SPARE", "SYSTEM", "SOFTWARE", "CRS-1", "ASR9000", "CRS-3", "HARDWARE_FAILURE", "ERROR_MESSAGES", "CONFIG_ASSISTANCE", "INSTALL_INSTLL_UPGRD", "SOFTWARE_FAILRE", "INTEROP", "OTHER", "COMM001","UNK0900", "COMM004","COMM002","COMM007","COMM009","OTHERS","XR-Routing-Platforms","Hardware","Application Networking Services","Broadband Cable","Contact Center Software", "Data Center and Storage Networking", "Commercial", "Enterprise", "Service Provider", "Unknown Customer Market Segment"), sizes=c(2,1,21,0,1, 6,0,4,6,0,10, 18,7,1,0,0,0, 14,9,1,1,1,0,0, 16,4,2,1,1,1, 23,3,0,0,0, 0,0,0,0))
ggplot(data = cluster6_data, aes(x = Features, y = sizes, fill = Values)) + geom_bar(stat="identity")+ggtitle("Cluster 6")

#Cluster7
cluster7_data <- data.frame(Features=c("customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","customer_activity_code","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","hw_product_erp_platform","sw_platform","sw_platform","sw_platform","sw_platform","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","problem_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_underlying_cause_code","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_tac_hw_family","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment","sr_cust_mkt_segment"),
                            Values=c("CONFIG", "INSTALL", "OPERATE", "PLANNING", "UPGRADE","CRS16 SERIES","CRS4 SERIES", "CRS8 SERIES", "CRSMC SERIES","SPARE", "SYSTEM", "SOFTWARE", "CRS-1", "ASR9000", "CRS-3", "HARDWARE_FAILURE", "ERROR_MESSAGES", "CONFIG_ASSISTANCE", "INSTALL_INSTLL_UPGRD", "SOFTWARE_FAILRE", "INTEROP", "OTHER", "COMM001", "COMM004","COMM002","UNK0900","COMM007","COMM009","OTHERS","XR-Routing-Platforms","Hardware","Other","Router and IOS Architecture","NMS (Network Management Services)", "Routing Protocols (Includes NAT and HSRP)", "Commercial", "Enterprise", "Service Provider", "Unknown Customer Market Segment"), sizes=c(668,261,2468,0,295, 1224,26,310,509,103,1616, 2730,4,3,2, 1693,1320,277,167,165,127,39, 2527,334,276,170,148,139,194, 3364,288,47,21,16,14, 24,12,78,18))
ggplot(data = cluster7_data, aes(x = Features, y = sizes, fill = Values)) + geom_bar(stat="identity")+ggtitle("Cluster 7")

#CSR & ASR903, CDETS vs Non-CDETS
crs_asr <- data.frame(Product_Family = c("CRS", "CRS", "ASR903", "ASR903"), Values=c("CDETS", "Non-CDETS", "CDETS", "Non-CDETS"), sizes=c(1075, 10116, 2489, 10433))
ggplot(data = crs_asr, aes(x = Product_Family, y = sizes, fill = Values)) + geom_bar(stat="identity", width = 0.3)+ggtitle("CDETS vs NON-CDETS")










sample <- sample[,-c(22,21,18,3,9)]
sample[,c(2:6,8:11,13:17)] <- sapply(sample[,c(2:6,8:11,13:17)], as.numeric)
d <- dist(sample, method = "euclidean")
a <- kmeans(d, 4)
sr_data$new_cluster <- a$cluster
cluster1 = sr_data[sr_data$new_cluster==1, ]





summary(cluster1$kpi_sr_details.sr_technology_group_name)
summary(cluster1$kpi_sr_details.sr_customer_activity_code)
summary(cluster1$kpi_sr_details.sr_underlying_cause_desc)
summary(cluster1$kpi_sr_details.sr_problem_code)
summary(cluster1$kpi_sr_details.cust_theater)
summary(cluster1$kpi_sr_details.sr_hw_product_erp_platform)




