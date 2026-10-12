# Cleaning workspace
rm(list = ls())

# Libraries
library(plyr)
library(ggplot2)
library(nloptr)

# install.packages("Rcpp", type="source")
# install.packages("https://ampl.com/dl/API/rAMPL.tar.gz", repos=NULL, INSTALL_opts=c("--no-multiarch", "--no-staged-install"))

# Quantile regression function
quantreg <- function(solver=NULL, modelDirectory=NULL, df=data.frame(), p) {
  
  # AMPL package
  library(rAMPL)
  
  # Create an AMPL instance
  env <- new(Environment, "C:/Users/Orbolato/AMPL")
  ampl <- new(AMPL, env)
  
  # Set solver
  if (!is.null(solver)) {
    ampl$setOption("solver", solver)
  }
  
  # Read the model file.
  if (is.null(modelDirectory)) {
    modelDirectory <- "optimization-models"
  }
  ampl$read(paste(modelDirectory, "/quantreg.mod", sep=""))
  
  # Maximum number of arrivals and departures
  percentile <- ampl$getParameter('p')
  percentile$set(p)
  a_max <- ampl$getParameter('a_max')
  a_max$set(max(df$actual_arrivals))
  ampl$setData(data.frame(n=1:nrow(df),
                          arr=df$actual_arrivals,
                          dep=df$actual_departures,
                          freq=df$nrow),1,'points')
  
  # Solve the model
  ampl$solve()
  
  # Print out the result
  cat(sprintf("Objective: %f\n", ampl$getObjective("z")$value()))
  
  # Get the values of the decision variable in the data frame
  res <- ampl$getVariable("alpha")$getValues()
  res <- cbind(res,ampl$getVariable("beta")$getValues())
  print(res)
  return(res)
}

# Linear quantile regression function
linear_quantreg <- function(solver=NULL, modelDirectory=NULL, df=data.frame(), p) {
  
  # AMPL package
  library(rAMPL)
  
  # Create an AMPL instance
  env <- new(Environment, "C:/Users/Orbolato/AMPL")
  ampl <- new(AMPL, env)
  
  # Set solver
  if (!is.null(solver)) {
    ampl$setOption("solver", solver)
  }
  
  # Read the model file.
  if (is.null(modelDirectory)) {
    modelDirectory <- "optimization-models"
  }
  ampl$read(paste(modelDirectory, "/linear_quantreg.mod", sep=""))
  
  # Maximum number of arrivals and departures
  percentile <- ampl$getParameter('p')
  percentile$set(p)
  a_max <- ampl$getParameter('a_max')
  a_max$set(max(df$actual_arrivals))
  ampl$setData(data.frame(n=1:nrow(df),
                          arr=df$actual_arrivals,
                          dep=df$actual_departures,
                          freq=df$nrow),1,'points')
  
  # Solve the model
  ampl$solve()
  
  # Print out the result
  cat(sprintf("Objective: %f\n", ampl$getObjective("z")$value()))
  
  # Get the values of the decision variable in the data frame
  res <- ampl$getVariable("alpha")$getValues()
  res <- cbind(res,ampl$getVariable("beta")$getValues())
  print(res)
  return(res)
}

# Linear quantile regression function
linear_quantreg_2 <- function(solver=NULL, modelDirectory=NULL, df=data.frame(), p) {
  
  # AMPL package
  library(rAMPL)
  
  # Create an AMPL instance
  env <- new(Environment, "C:/Users/Orbolato/AMPL")
  ampl <- new(AMPL, env)
  
  # Set solver
  if (!is.null(solver)) {
    ampl$setOption("solver", solver)
  }
  
  # Read the model file.
  if (is.null(modelDirectory)) {
    modelDirectory <- "optimization-models"
  }
  ampl$read(paste(modelDirectory, "/linear_quantreg_2.mod", sep=""))
  
  # Maximum number of arrivals and departures
  percentile <- ampl$getParameter('p')
  percentile$set(p)
  a_max <- ampl$getParameter('a_max')
  a_max$set(max(df$actual_arrivals))
  ampl$setData(data.frame(n=1:nrow(df),
                          arr=df$actual_arrivals,
                          dep=df$actual_departures),1,'points')
  
  # Solve the model
  ampl$solve()
  
  # Print out the result
  cat(sprintf("Objective: %f\n", ampl$getObjective("z")$value()))
  
  # Get the values of the decision variable in the data frame
  res <- ampl$getVariable("alpha")$getValues()
  res <- cbind(res,ampl$getVariable("beta")$getValues())
  print(res)
  return(res)
}

# Loading flight data from ANAC VRA
folder <- "./" # Run this script from the repository root
setwd(folder)
file <- "data/data_gru.Rda"

# Opening file
load(file)

# VFR, IFR and LIRF
df_vfr <- df[df$ifr == 0 & df$lifr == 0,]
df_ifr <- df[df$ifr == 1 & df$lifr == 0,]
df_lifr <- df[df$lifr == 1,]
df_list <- list('VMC' = df_vfr, 'IMC' = df_ifr, 'LIMC' = df_lifr)

# Runway configurations
runway_config <- unique(df$configuration)

# Generating charts
dfc_list <- list()
res_list <- list()
for(i in 1:length(df_list)){
  for(rc in runway_config){
    
    # Frequency counts
    mc <- names(df_list)[i]
    df_mc <- df_list[[i]]
    print(paste0(paste0(mc,' '),rc))
    dfc <- ddply(df_mc[df_mc$configuration == rc,], 
                     c('actual_arrivals', 'actual_departures'), "nrow", .drop = FALSE)
    #dfc <- dfc[dfc$nrow > 1,]
    dfc_list <- append(dfc_list,list(dfc))
    
    # Plots
    strs <- strsplit(rc, split='|',fixed=TRUE)[[1]]
    g <- ggplot(data = dfc, aes(x = actual_arrivals, y = actual_departures)) 
    g <- g + geom_point(aes(size = nrow, colour = nrow), shape = 19)
    g <- g + scale_colour_gradient(low = "green", high="red", guide = 'legend', breaks=seq(0,as.integer(max(dfc$nrow)/10 + 1)*10,by=as.integer(max(dfc$nrow)/10 + 1)))
    g <- g + scale_size_continuous(breaks=seq(0,as.integer(max(dfc$nrow)/10 + 1)*10,by=as.integer(max(dfc$nrow)/10 + 1)))
    g <- g + scale_x_continuous(breaks = min(df_mc[df_mc$configuration == rc,'actual_arrivals']):max(df_mc[df_mc$configuration == rc,'actual_arrivals']),
                                limits = c(min(df_mc[df_mc$configuration == rc,'actual_arrivals'])-1,max(df_mc[df_mc$configuration == rc,'actual_arrivals'])+1))
    g <- g + scale_y_continuous(breaks = min(df_mc[df_mc$configuration == rc,'actual_departures']):max(df_mc[df_mc$configuration == rc,'actual_departures']),
                                limits = c(min(df_mc[df_mc$configuration == rc,'actual_departures'])-1,max(df_mc[df_mc$configuration == rc,'actual_departures'])+1))
    g <- g + labs(title = paste0(paste0(mc,' '),rc), x = 'Arrivals', y = 'Departures', color = 'Freq.',
                  size = 'Freq.')
    g <- g + theme_classic()
    filename <- paste0(paste0(mc,'_'),strs[1])
    filename <- paste0(filename,'_')
    filename <- paste0(filename,strs[2])
    names(dfc_list)[length(dfc_list)] <- filename
    final_filename <- paste0(filename,'.png')
    final_filename <- paste0('images\\',final_filename)
    final_filename <- paste0(folder,final_filename)
    ggsave(final_filename)
    
    # List of results
    for(p in c(0.75,0.9,0.95,0.975,0.99,0.9925,0.995,0.9975)){
      
      # Solving the problem
      res <- linear_quantreg('Gurobi',NULL,dfc,p)
      res <- data.frame(cbind(res$alpha.val,res$beta.val))
      colnames(res) <- c('alpha.val','beta.val')
      res_list <- append(res_list,list(res))
      res_filename <- paste0(filename, '_RCCE_')
      res_filename <- paste0(res_filename, sub('0.','',format(p,nsmall=4)))
      names(res_list)[length(res_list)] <- res_filename
      
      # Line points
      a_max <- max(dfc$actual_arrivals)
      dep_lim <- c(res$alpha.val[1])
      for(k in 1:a_max){
        dep_lim <- append(dep_lim,res$alpha.val[k] + res$beta.val[k]*k)
      }
      data <- data.frame(cbind(0:a_max,dep_lim))
      data <- rbind(data,c(a_max,0))
      colnames(data) <- c('x','y')
      
      # Plots
      g <- ggplot(data = dfc, aes(x = actual_arrivals, y = actual_departures))
      g <- g + geom_point(aes(size = nrow, colour = nrow), shape = 19)
      g <- g + scale_colour_gradient(low = "green", high="red", guide = 'legend', breaks=seq(0,as.integer(max(dfc$nrow)/10 + 1)*10,by=as.integer(max(dfc$nrow)/10 + 1)))
      g <- g + scale_size_continuous(breaks=seq(0,as.integer(max(dfc$nrow)/10 + 1)*10,by=as.integer(max(dfc$nrow)/10 + 1)))
      g <- g + geom_line(data = data, aes(x = x, y = y))
      g <- g + scale_x_continuous(breaks = 0:max(df_mc[df_mc$configuration == rc,'actual_arrivals']),
                                  limits = c(-1,max(df_mc[df_mc$configuration == rc,'actual_arrivals'])+1))
      g <- g + scale_y_continuous(breaks = 0:max(df_mc[df_mc$configuration == rc,'actual_departures']),
                                  limits = c(-1,max(df_mc[df_mc$configuration == rc,'actual_departures'])+1))
      g <- g + labs(title = paste0(paste0(mc,' '),rc), x = 'Arrivals', y = 'Departures', color = 'Freq.',
                    size = 'Freq.')
      g <- g + theme_classic()
      final_filename <- paste0(res_filename,'.png')
      final_filename <- paste0('images\\',final_filename)
      final_filename <- paste0(folder,final_filename)
      ggsave(final_filename)
      
      # Exporting CSV
      write.csv(data, paste0('curves/',paste0(res_filename,'.csv')), row.names=FALSE)
    }
  }
}

# Exporting CSV
write.csv(df_vfr, 'data/vfr_data.csv', row.names=FALSE)
write.csv(df_ifr, 'data/ifr_data.csv', row.names=FALSE)
write.csv(df_lifr, 'data/lifr_data.csv', row.names=FALSE)
write.csv(df, 'data/data.csv', row.names=FALSE)
