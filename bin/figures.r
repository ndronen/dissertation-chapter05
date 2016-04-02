library(plyr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(gtools)
library(xtable)

prepare_df <- function(df) {
  df$System <- df$Algorithm
  df$Algorithm <- NULL
  df$`Non-word length` <- df$Non.word.Length
  df
}

plot_system_accuracy_at_k_by_length <- function(df, candidates=NULL, max_rank=10, min_length=1, max_length=15, by_system=FALSE) {
  df <- filter(df, (`Non-word length` >= min_length) & (`Non-word length` <= max_length))
  if (!is.null(candidates)) {
    df <- filter(df, Candidates == candidates)
  }
  df <- filter(df, Rank <= max_rank)

  df$`Non-word length` <- factor(df$`Non-word length`)

  p <- ggplot(data=df, aes(x=Rank, y=Accuracy, color=`Non-word length`))
  p <- p + scale_x_continuous(breaks=sort(unique(df$Rank)))

  if (by_system) {
    p <- p + geom_point(aes(shape=System), alpha=0.5)
  } else {
    p <- p + geom_point(alpha=0.5)
  }

  if (by_system) {
    p <- p + geom_line(aes(linetype=System), alpha=0.5)
  } else {
    p <- p + geom_line(alpha=0.5)
  }

  #p <- p + ylim(0.1, 1)

  p
}

plot_system_accuracy_at_k_difference <- function(df, lengths) {
  p <- ggplot(data=df, aes(x=Rank, color=`Non-word length`))
  
  for (len in lengths) {
    df_length <- filter(df, `Non-word length` == len)
    #df_length$`Non-word length` <- factor(df_length$`Non-word length`,
    #                                      levels=mixedsort(unique(df_length$`Non-word length`)))
    #print(levels(df_length$`Non-word length`))
    p <- p + geom_ribbon(data=df_length,
      #aes(ymin=ymin, ymax=ymax, x=Rank, fill=`Non-word length`, linetype=`Non-word length`),
      aes(ymin=ymin, ymax=ymax, x=Rank, fill=`Non-word length`),
      alpha=0.5, color="black")
  }
  p <- p + scale_x_continuous(breaks=sort(unique(tmp$Rank)))
  p <- p + YLIM
  p
}

plot_generated_data_results <- function() {
  df <- read.csv("data/ranks-test.csv", sep="\t")
  df <- prepare_df(df)
  
  # Plot the ConvNet and Aspell results separately.
  p <- plot_system_accuracy_at_k_by_length(
    filter(df, System == "ConvNet (binary)"),
    candidates="No near-miss")
  ggsave("data/ranks-by-length-convnet.pdf", p)
  dev.off()
  
  p <- plot_system_accuracy_at_k_by_length(
    filter(df, System == "Aspell with Jaro-Winkler"),
    candidates="No near-miss")
  ggsave("data/ranks-by-length-aspell-with-jaro-winkler.pdf", p)
  dev.off()
  
  # Then plot the differences.
  df_aspell <- filter(df, System=="Aspell with Jaro-Winkler")
  df_convnet <- filter(df, System=="ConvNet (binary)")
  df_length <- df_aspell
  
  df_length$Accuracy <- df_convnet$Accuracy - df_aspell$Accuracy
  p <- plot_system_accuracy_at_k_by_length(df_length,
    candidates="No near-miss")
    # c(4,6,8,10))
  p <- p + coord_cartesian(ylim=c(-.25,.25))
  ggsave("data/ranks-by-length-difference.pdf", p)
  dev.off()
}
#plot_generated_data_results()

plot_mitton_data_results <- function() {
  df <- read.csv("data/ranks-mitton.csv", sep="\t")
  df <- prepare_df(df)

  # This data frame has results for four different Mitton corpora.

  graphics.off()

  lengths <- c(3,4,5,6,7,8,9,10)

  for (corpus in unique(df$Corpus)) {
    tmp <- subset(df, Corpus == corpus & Non.word.Length %in% lengths & Rank <= 5)

    # Plot the differences.
    df_aspell <- filter(tmp, System=="Aspell with Jaro-Winkler")
    df_convnet <- filter(tmp, System=="ConvNet (binary)")
    df_random_forest <- filter(tmp, System=="Random Forest")

    df_length <- df_aspell
    df_length$`Jaro-Winkler` <- df_aspell$Accuracy
    df_length$`ConvNet (binary)` <- df_convnet$Accuracy
    df_length$`Random Forest` <- df_random_forest$Accuracy

    ylim <- c(-.45, .45)
    if (corpus == "Birbeck") {
      ylim <- c(-.21, .21)
    }

    df_length$Accuracy <- df_length$`Random Forest` - df_length$`Jaro-Winkler`
    p1 <- plot_system_accuracy_at_k_by_length(df_length, min_length=3)
    p1 <- p1 + coord_cartesian(ylim=ylim)
    p1 <- p1 + labs(title=paste0("Random Forest versus Jaro-Winkler - ", corpus))

    df_length$Accuracy <- df_length$`Random Forest` - df_length$`ConvNet (binary)`
    p2 <- plot_system_accuracy_at_k_by_length(df_length, min_length=3)
    p2 <- p2 + coord_cartesian(ylim=ylim)
    p2 <- p2 + labs(title=paste0("Random Forest versus ConvNet (binary) - ", corpus))

    df_length$Accuracy <- df_length$`ConvNet (binary)` - df_length$`Jaro-Winkler`
    p3 <- plot_system_accuracy_at_k_by_length(df_length, min_length=3)
    p3 <- p3 + coord_cartesian(ylim=ylim)
    p3 <- p3 + labs(title=paste0("ConvNet (binary) versus Jaro-Winkler - ", corpus))

    pdf_file = paste0("data/ranks-by-length-difference-", corpus, ".pdf")
    pdf(pdf_file)
    grid.arrange(p1, p2, p3, ncol=1)
    dev.off()
  }
}
plot_mitton_data_results()

make_mitton_tables <- function() {
  df <- read.csv("data/ranks-mitton.csv", sep="\t")
  df <- prepare_df(df)
}
#make_mitton_tables()
