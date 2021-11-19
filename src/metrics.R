library(tidyverse)
library(ggplot2)
library(ggrepel)
require(gridExtra)

params <- read.csv("../params.txt", sep=";")

data_l <- read.csv("../experiment_l.txt", sep=";")
data_l <- merge(data_l, params)
data_l <- data_l %>% mutate(Fscore = 2*Precision*Recall/(Precision+Recall))

data_z <- read.csv("../experiment_z.txt", sep=";")
data_z <- merge(data_z, params)
data_z <- data_z %>% mutate(Fscore = 2*Precision*Recall/(Precision+Recall))


# Loss
l <- data_l %>%
  group_by(Model) %>%
  summarise(loss = mean(Loss), sd = sd(Loss), params = mean(Params)) %>%
  mutate(lower=loss-3*sd, upper=loss+3*sd, architecture=substr(Model, 1, 1))
z <- data_z %>%
  group_by(Model) %>%
  summarise(loss = mean(Loss), sd = sd(Loss), params = mean(Params)) %>%
  mutate(lower=loss-3*sd, upper=loss+3*sd, architecture=substr(Model, 1, 1))
ggplot()+
  geom_line(data=l, mapping=aes(x=params, y=loss), size=1, linetype="solid")+
  #geom_errorbar(data=l, mapping=aes(x=params, y=loss, ymin=lower, ymax=upper))+
  geom_point(data=l, mapping=aes(x=params, y=loss, color=architecture), size=3)+theme_light()+
  geom_line(data=z, mapping=aes(x=params, y=loss), size=1, linetype="dashed")+
  #geom_errorbar(data=z, mapping=aes(x=params+5000, y=loss, ymin=lower, ymax=upper))+
  geom_point(data=z, mapping=aes(x=params, y=loss, color=architecture), size=3)+theme_light()

# Precision
l <- data_l %>%
  group_by(Model) %>%
  summarise(precision = mean(Precision), sd = sd(Precision), params = mean(Params)) %>%
  mutate(lower=precision-3*sd, upper=precision+3*sd, architecture=substr(Model, 1, 1))
z <- data_z %>%
  group_by(Model) %>%
  summarise(precision = mean(Precision), sd = sd(Precision), params = mean(Params)) %>%
  mutate(lower=precision-3*sd, upper=precision+3*sd, architecture=substr(Model, 1, 1))
ggplot()+
  geom_line(data=l, mapping=aes(x=params, y=precision), size=1, linetype="solid")+
  #geom_errorbar(data=l, mapping=aes(x=params, y=precision, ymin=lower, ymax=upper))+
  geom_point(data=l, mapping=aes(x=params, y=precision, color=architecture), size=3)+theme_light()+
  geom_line(data=z, mapping=aes(x=params, y=precision), size=1, linetype="dashed")+
  #geom_errorbar(data=z, mapping=aes(x=params+5000, y=precision, ymin=lower, ymax=upper))+
  geom_point(data=z, mapping=aes(x=params, y=precision, color=architecture), size=3)+theme_light()

# Recall
l <- data_l %>%
  group_by(Model) %>%
  summarise(recall = mean(Recall), sd = sd(Recall), params = mean(Params)) %>%
  mutate(lower=recall-3*sd, upper=recall+3*sd, architecture=substr(Model, 1, 1))
z <- data_z %>%
  group_by(Model) %>%
  summarise(recall = mean(Recall), sd = sd(Recall), params = mean(Params)) %>%
  mutate(lower=recall-3*sd, upper=recall+3*sd, architecture=substr(Model, 1, 1))
ggplot()+
  geom_line(data=l, mapping=aes(x=params, y=recall), size=1, linetype="solid")+
  #geom_errorbar(data=l, mapping=aes(x=params, y=recall, ymin=lower, ymax=upper))+
  geom_point(data=l, mapping=aes(x=params, y=recall, color=architecture), size=3)+theme_light()+
  geom_line(data=z, mapping=aes(x=params, y=recall), size=1, linetype="dashed")+
  #geom_errorbar(data=z, mapping=aes(x=params+5000, y=recall, ymin=lower, ymax=upper))+
  geom_point(data=z, mapping=aes(x=params, y=recall, color=architecture), size=3)+theme_light()

# Fscore
l <- data_l %>%
  group_by(Model) %>%
  summarise(fscore = mean(Fscore), sd = sd(Fscore), params = mean(Params)) %>%
  mutate(lower=fscore-3*sd, upper=fscore+3*sd, architecture=substr(Model, 1, 1))
z <- data_z %>%
  group_by(Model) %>%
  summarise(fscore = mean(Fscore), sd = sd(Fscore), params = mean(Params)) %>%
  mutate(lower=fscore-3*sd, upper=fscore+3*sd, architecture=substr(Model, 1, 1))
ggplot()+
  geom_line(data=l, mapping=aes(x=params, y=fscore), size=1, linetype="solid")+
  #geom_errorbar(data=l, mapping=aes(x=params, y=fscore, ymin=lower, ymax=upper))+
  geom_point(data=l, mapping=aes(x=params, y=fscore, color=architecture), size=3)+theme_light()+
  geom_line(data=z, mapping=aes(x=params, y=fscore), size=1, linetype="dashed")+
  #geom_errorbar(data=z, mapping=aes(x=params+5000, y=fscore, ymin=lower, ymax=upper))+
  geom_point(data=z, mapping=aes(x=params, y=fscore, color=architecture), size=3)+theme_light()

# AUC
l <- data_l %>%
  group_by(Model) %>%
  summarise(auc = mean(AUC), sd = sd(AUC), params = mean(Params)) %>%
  mutate(lower=auc-3*sd, upper=auc+3*sd, architecture=substr(Model, 1, 1))
z <- data_z %>%
  group_by(Model) %>%
  summarise(auc = mean(AUC), sd = sd(AUC), params = mean(Params)) %>%
  mutate(lower=auc-3*sd, upper=auc+3*sd, architecture=substr(Model, 1, 1))
ggplot()+
  geom_line(data=l, mapping=aes(x=params, y=auc), size=1, linetype="solid")+
  geom_errorbar(data=l, mapping=aes(x=params, y=auc, ymin=lower, ymax=upper))+
  geom_point(data=l, mapping=aes(x=params, y=auc, color=architecture), size=3)+theme_light()+
  geom_line(data=z, mapping=aes(x=params, y=auc), size=1, linetype="dashed")+
  geom_errorbar(data=z, mapping=aes(x=params+5000, y=auc, ymin=lower, ymax=upper))+
  geom_point(data=z, mapping=aes(x=params, y=auc, color=architecture), size=3)+theme_light()



###########################################
#                  log                    #
###########################################

# Loss
data_l %>%
  group_by(Model) %>%
  summarise(mean = mean(Loss), sd = sd(Loss), params = mean(Params)) %>%
  mutate(lower=mean-3*sd, upper=mean+3*sd, model=substr(Model, 1, 1)) %>%
  ggplot(aes(x=params, y=mean))+
  geom_errorbar(aes(ymin=lower, ymax=upper), color="red")+
  geom_line(size=1)+
  geom_point(aes(color=model), size=3)+theme_light()

# Precision
data_l %>%
  group_by(Model) %>%
  summarise(mean = mean(Precision), sd = sd(Precision), params = mean(Params)) %>%
  mutate(lower=mean-3*sd, upper=mean+3*sd, model=substr(Model, 1, 1)) %>%
  ggplot(aes(x=params, y=mean))+
  geom_errorbar(aes(ymin=lower, ymax=upper), color="red")+
  geom_line(size=1)+
  geom_point(aes(color=model), size=3)+theme_light()

# Recall
data_l %>%
  group_by(Model) %>%
  summarise(mean = mean(Recall), sd = sd(Recall), params = mean(Params)) %>%
  mutate(lower=mean-3*sd, upper=mean+3*sd, model=substr(Model, 1, 1)) %>%
  ggplot(aes(x=params, y=mean))+
  geom_errorbar(aes(ymin=lower, ymax=upper), color="red")+
  geom_line(size=1)+
  geom_point(aes(color=model), size=3)+theme_light()

# Fscore
data_l %>%
  group_by(Model) %>%
  summarise(mean = mean(Fscore), sd = sd(Fscore), params = mean(Params)) %>%
  mutate(lower=mean-3*sd, upper=mean+3*sd, model=substr(Model, 1, 1)) %>%
  ggplot(aes(x=params, y=mean))+
  geom_errorbar(aes(ymin=lower, ymax=upper), color="red")+
  geom_line(size=1)+
  geom_point(aes(color=model), size=3)+theme_light()

# AUC
data_l %>%
  group_by(Model) %>%
  summarise(mean = mean(AUC), sd = sd(AUC), params = mean(Params)) %>%
  mutate(lower=mean-3*sd, upper=mean+3*sd, model=substr(Model, 1, 1)) %>%
  ggplot(aes(x=params, y=mean))+
  geom_errorbar(aes(ymin=lower, ymax=upper), color="red")+
  geom_line(size=1)+
  geom_point(aes(color=model), size=3)+theme_light()


###########################################
#                   z                     #
###########################################

# Loss
data_z %>%
  group_by(Model) %>%
  summarise(mean = mean(Loss), sd = sd(Loss), params = mean(Params)) %>%
  mutate(lower=mean-3*sd, upper=mean+3*sd, model=substr(Model, 1, 1)) %>%
  ggplot(aes(x=params, y=mean))+
  geom_errorbar(aes(ymin=lower, ymax=upper), color="red")+
  geom_line(size=1)+
  geom_point(aes(color=model), size=3)+theme_light()

# Precision
data_z %>%
  group_by(Model) %>%
  summarise(mean = mean(Precision), sd = sd(Precision), params = mean(Params)) %>%
  mutate(lower=mean-3*sd, upper=mean+3*sd, model=substr(Model, 1, 1)) %>%
  ggplot(aes(x=params, y=mean))+
  geom_errorbar(aes(ymin=lower, ymax=upper), color="red")+
  geom_line(size=1)+
  geom_point(aes(color=model), size=3)+theme_light()

# Recall
data_z %>%
  group_by(Model) %>%
  summarise(mean = mean(Recall), sd = sd(Recall), params = mean(Params)) %>%
  mutate(lower=mean-3*sd, upper=mean+3*sd, model=substr(Model, 1, 1)) %>%
  ggplot(aes(x=params, y=mean))+
  geom_errorbar(aes(ymin=lower, ymax=upper), color="red")+
  geom_line(size=1)+
  geom_point(aes(color=model), size=3)+theme_light()

# Fscore
data_z %>%
  group_by(Model) %>%
  summarise(mean = mean(Fscore), sd = sd(Fscore), params = mean(Params)) %>%
  mutate(lower=mean-3*sd, upper=mean+3*sd, model=substr(Model, 1, 1)) %>%
  ggplot(aes(x=params, y=mean))+
  geom_errorbar(aes(ymin=lower, ymax=upper), color="red")+
  geom_line(size=1)+
  geom_point(aes(color=model), size=3)+theme_light()

# AUC
data_z %>%
  group_by(Model) %>%
  summarise(mean = mean(AUC), sd = sd(AUC), params = mean(Params)) %>%
  mutate(lower=mean-3*sd, upper=mean+3*sd, model=substr(Model, 1, 1)) %>%
  ggplot(aes(x=params, y=mean))+
  geom_errorbar(aes(ymin=lower, ymax=upper), color="red")+
  geom_line(size=1)+
  geom_point(aes(color=model), size=3)+theme_light()


###########################################
#              razlika modelov            #
###########################################

colnames(data_l) <- paste("l_", colnames(data_l), sep="")
colnames(data_z) <- paste("z_", colnames(data_z), sep="")

data <- merge.data.frame(data_l, data_z, by.x = c("l_Model", "l_Idx", "l_Params"), by.y = c("z_Model", "z_Idx", "z_Params"))
colnames(data)[1:3] <- c("Model", "Idx", "Params")

# Precision
data %>%
  group_by(Model) %>%
  summarise(mean_z = mean(z_Precision), mean_l = mean(l_Precision)) %>% 
  mutate(Diff = mean_l - mean_z) %>% 
  summarise(mean = mean(Diff))

# Recall
data %>%
  group_by(Model) %>%
  summarise(mean_z = mean(z_Recall), mean_l = mean(l_Recall)) %>% 
  mutate(Diff = mean_l - mean_z) %>% 
  summarise(mean = mean(Diff))

# Fscore
data %>%
  group_by(Model) %>%
  summarise(mean_z = mean(z_Fscore), mean_l = mean(l_Fscore)) %>% 
  mutate(Diff = mean_l - mean_z) %>% 
  summarise(mean = mean(Diff))

# Loss
data %>%
  group_by(Model) %>%
  summarise(mean_z = mean(z_Loss), mean_l = mean(l_Loss)) %>% 
  mutate(Diff = mean_l - mean_z) %>% 
  summarise(mean = mean(Diff))


###########################################
#            Posamezne arhitekture        #
###########################################

show_loss <- function(arch, color) {
  l <- data_l %>%
    filter(str_detect(Model, paste("^", arch, sep=""))) %>%
    group_by(Model) %>%
    summarise(loss = mean(Loss), sd = sd(Loss), params = mean(Params)) %>%
    mutate(lower=loss-3*sd, upper=loss+3*sd)
  z <- data_z %>%
    filter(str_detect(Model, paste("^", arch, sep=""))) %>%
    group_by(Model) %>%
    summarise(loss = mean(Loss), sd = sd(Loss), params = mean(Params)) %>%
    mutate(lower=loss-3*sd, upper=loss+3*sd)
  plot <- ggplot()+scale_x_continuous(labels=scales::scientific, guide=guide_axis(check.overlap=TRUE))+
    geom_line(data=l, mapping=aes(x=params, y=loss), size=1, linetype="solid")+
    geom_point(data=l, mapping=aes(x=params, y=loss), color=color, size=3, show.legend=FALSE)+theme_light()+
    geom_text(data=l, mapping=aes(x=params, y=loss, label=Model))+
    geom_line(data=z, mapping=aes(x=params, y=loss), size=1, linetype="dashed")+
    geom_point(data=z, mapping=aes(x=params, y=loss), color=color, size=3, show.legend=FALSE)+theme_light()+
    geom_text(data=z, mapping=aes(x=params, y=loss, label=Model))+
    ggtitle(paste("Arhitektura", arch, sep=" "))+ylab("Napaka")+xlab("Število parametrov")
  return(plot)
}
show_prec <- function(arch, color) {
  l <- data_l %>%
    filter(str_detect(Model, paste("^", arch, sep=""))) %>%
    group_by(Model) %>%
    summarise(precision = mean(Precision), sd = sd(Precision), params = mean(Params)) %>%
    mutate(lower=precision-3*sd, upper=precision+3*sd)
  z <- data_z %>%
    filter(str_detect(Model, paste("^", arch, sep=""))) %>%
    group_by(Model) %>%
    summarise(precision = mean(Precision), sd = sd(Precision), params = mean(Params)) %>%
    mutate(lower=precision-3*sd, upper=precision+3*sd)
  plot <- ggplot()+scale_x_continuous(labels=scales::scientific, guide=guide_axis(check.overlap=TRUE))+
    geom_line(data=l, mapping=aes(x=params, y=precision), size=1, linetype="solid")+
    geom_point(data=l, mapping=aes(x=params, y=precision), color=color, size=3, show.legend=FALSE)+theme_light()+
    geom_text(data=l, mapping=aes(x=params, y=precision, label=Model))+
    geom_line(data=z, mapping=aes(x=params, y=precision), size=1, linetype="dashed")+
    geom_point(data=z, mapping=aes(x=params, y=precision), color=color, size=3, show.legend=FALSE)+theme_light()+
    geom_text(data=z, mapping=aes(x=params, y=precision, label=Model))+
    ggtitle(paste("Arhitektura", arch, sep=" "))+ylab("Preciznost")+xlab("Število parametrov")
  return(plot)
}
show_rec <- function(arch, color) {
  l <- data_l %>%
    filter(str_detect(Model, paste("^", arch, sep=""))) %>%
    group_by(Model) %>%
    summarise(recall = mean(Recall), sd = sd(Recall), params = mean(Params)) %>%
    mutate(lower=recall-3*sd, upper=recall+3*sd)
  z <- data_z %>%
    filter(str_detect(Model, paste("^", arch, sep=""))) %>%
    group_by(Model) %>%
    summarise(recall = mean(Recall), sd = sd(Recall), params = mean(Params)) %>%
    mutate(lower=recall-3*sd, upper=recall+3*sd)
  plot <- ggplot()+scale_x_continuous(labels=scales::scientific, guide=guide_axis(check.overlap=TRUE))+
    geom_line(data=l, mapping=aes(x=params, y=recall), size=1, linetype="solid")+
    geom_point(data=l, mapping=aes(x=params, y=recall), color=color, size=3, show.legend=FALSE)+theme_light()+
    geom_text(data=l, mapping=aes(x=params, y=recall, label=Model))+
    geom_line(data=z, mapping=aes(x=params, y=recall), size=1, linetype="dashed")+
    geom_point(data=z, mapping=aes(x=params, y=recall), color=color, size=3, show.legend=FALSE)+theme_light()+
    geom_text(data=z, mapping=aes(x=params, y=recall, label=Model))+
    ggtitle(paste("Arhitektura", arch, sep=" "))+ylab("Priklic")+xlab("Število parametrov")
  return(plot)
}
show_f <- function(arch, color) {
  l <- data_l %>%
    filter(str_detect(Model, paste("^", arch, sep=""))) %>%
    group_by(Model) %>%
    summarise(f = mean(Fscore), sd = sd(Fscore), params = mean(Params)) %>%
    mutate(lower=f-3*sd, upper=f+3*sd)
  z <- data_z %>%
    filter(str_detect(Model, paste("^", arch, sep=""))) %>%
    group_by(Model) %>%
    summarise(f = mean(Fscore), sd = sd(Fscore), params = mean(Params)) %>%
    mutate(lower=f-3*sd, upper=f+3*sd)
  plot <- ggplot()+scale_x_continuous(labels=scales::scientific, guide=guide_axis(check.overlap=TRUE))+
    geom_line(data=l, mapping=aes(x=params, y=f), size=1, linetype="solid")+
    geom_point(data=l, mapping=aes(x=params, y=f), color=color, size=3, show.legend=FALSE)+theme_light()+
    geom_text(data=l, mapping=aes(x=params, y=f, label=Model))+
    geom_line(data=z, mapping=aes(x=params, y=f), size=1, linetype="dashed")+
    geom_point(data=z, mapping=aes(x=params, y=f), color=color, size=3, show.legend=FALSE)+theme_light()+
    geom_text(data=z, mapping=aes(x=params, y=f, label=Model))+
    ggtitle(paste("Arhitektura", arch, sep=" "))+ylab("F vrednost")+xlab("Število parametrov")
  return(plot)
}

# Loss
a <- show_loss("A", "Orange")
b <- show_loss("B", "Green")
c <- show_loss("C", "deepskyblue2")
d <- show_loss("D", "magenta1")
grid.arrange(a, b, c, d, ncol=2, nrow=2)

# Precision
a <- show_prec("A", "Orange")
b <- show_prec("B", "Green")
c <- show_prec("C", "deepskyblue2")
d <- show_prec("D", "magenta1")
grid.arrange(a, b, c, d, ncol=2, nrow=2)

# Recall
a <- show_rec("A", "Orange")
b <- show_rec("B", "Green")
c <- show_rec("C", "deepskyblue2")
d <- show_rec("D", "magenta1")
grid.arrange(a, b, c, d, ncol=2, nrow=2)

# Recall
a <- show_f("A", "Orange")
b <- show_f("B", "Green")
c <- show_f("C", "deepskyblue2")
d <- show_f("D", "magenta1")
grid.arrange(a, b, c, d, ncol=2, nrow=2)

