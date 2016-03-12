library(ggplot2)

df <- read.csv("data/ranks.csv", sep="\t")

df$Candidates <- df$Suggestions

p <- ggplot(data=df, aes(x=Rank, y=Accuracy, color=System))
p <- p + geom_point(aes(shape=Candidates), alpha=0.5)
p <- p + geom_line(aes(linetype=Candidates), alpha=0.5)
ggsave("data/ranks.pdf", p)
dev.off()
