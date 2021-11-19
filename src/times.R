library(tidyverse)
library(ggplot2)
library(ggrepel)
require(gridExtra)

data_l <- read.csv("../times_l.txt", sep=";")
data_l <- data_l %>% mutate(architecture=substr(Model, 1, 1))
data_z <- read.csv("../times_z.txt", sep=";")
data_z <- data_z %>% mutate(architecture=substr(Model, 1, 1))

ggplot()+
  geom_line(data=l, mapping=aes(x=Params, y=Time), size=1, linetype="solid")+
  geom_point(data=l, mapping=aes(x=Params, y=Time, color=architecture), size=3)+theme_light()+
  geom_line(data=z, mapping=aes(x=Params, y=Time), size=1, linetype="dashed")+
  geom_point(data=z, mapping=aes(x=Params, y=Time, color=architecture), size=3)+theme_light()

show_time <- function(arch, color) {
  l <- data_l %>%
    filter(str_detect(Model, paste("^", arch, sep="")))
  z <- data_z %>%
    filter(str_detect(Model, paste("^", arch, sep="")))
  plot <- ggplot()+scale_x_continuous(labels=scales::scientific, guide=guide_axis(check.overlap=TRUE))+
    geom_line(data=l, mapping=aes(x=Params, y=Time), size=1, linetype="solid")+
    geom_point(data=l, mapping=aes(x=Params, y=Time), color=color, size=3, show.legend=FALSE)+theme_light()+
    geom_text(data=l, mapping=aes(x=Params, y=Time, label=Model))+
    geom_line(data=z, mapping=aes(x=Params, y=Time), size=1, linetype="dashed")+
    geom_point(data=z, mapping=aes(x=Params, y=Time), color=color, size=3, show.legend=FALSE)+theme_light()+
    geom_text(data=z, mapping=aes(x=Params, y=Time, label=Model))+
    ggtitle(paste("Arhitektura", arch, sep=" "))+ylab(paste("\u010C", "as [t]", sep=""))+xlab("Število parametrov")
  return(plot)
}

a <- show_time("A", "Orange")
b <- show_time("B", "Green")
c <- show_time("C", "deepskyblue2")
d <- show_time("D", "magenta1")
grid.arrange(a, b, c, d, ncol=2, nrow=2)
