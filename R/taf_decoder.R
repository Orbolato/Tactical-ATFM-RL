# Setting work directory
# Work directory: run this script from the repository root

# Filter noisy data
filter_data = function(taf_data){
  remove_rows = c()
  for(row in 1:nrow(taf_data)){
    taf_split <- strsplit(taf_data[row,], " - ")[[1]]
    date <- taf_split[1]
    taf <- taf_split[2]
    year <- substring(date,1,4)
    month <- substring(date,5,6)
    day <- substring(date,7,8)
    hour <- substring(date,9,10)
    time <- paste0(year,"-")
    time <- paste0(time,paste0(month,"-"))
    time <- paste0(time,paste0(day," "))
    time <- paste0(time,paste0(hour,":00"))
    time <- ymd_hm(time, tz='UTC')
    if(substring(taf,5,7) == 'AMD' || substring(taf,5,7) == 'COR'){
      start_day <- substring(taf,22,23)
      start_hour <- substring(taf,24,25)
    } else {
      start_day <- substring(taf,18,19)
      start_hour <- substring(taf,20,21)
    }
    start_time <- paste0(year,"-")
    start_time <- paste0(start_time,paste0(month,"-"))
    start_time <- paste0(start_time,paste0(start_day," "))
    start_time <- paste0(start_time,paste0(start_hour,":00"))
    start_time <- ymd_hm(start_time, tz='UTC')
    if(start_time != time || start_hour != "06"){
      print(start_time)
      remove_rows <- append(remove_rows, row)
    }
  }
  return(data.frame(taf_data[-remove_rows,]))
}

# Split TAF in BECMG
split_taf_string = function(taf_str){
  taf_split <- strsplit(taf_str, " - ")[[1]]
  day <- taf_split[1]
  taf <- taf_split[2]
  taf <- strsplit(taf, "\\ BECMG|\\ FM")[[1]]
  if(length(taf) > 1){
    for(i in 2:length(taf)){
      if(substring(taf[i],1,1) == " "){
        taf[i] <- paste0("BECMG", taf[i])
      } else {
        taf[i] <- paste0("FM", taf[i])
      }
    }
  }
  rmk <- strsplit(taf[length(taf)], " RMK ")[[1]]
  taf[length(taf)] <- rmk[1]
  rmk <- paste0("RMK ", rmk[2])
  taf <- append(taf, rmk)
  if(substring(taf[1],5,7) == 'AMD' || substring(taf[1],5,7) == 'COR'){
    taf[1] <- paste0(paste0(substring(taf[1],1,13),day),substring(taf[1],21))
  } else {
    taf[1] <- paste0(paste0(substring(taf[1],1,9),day),substring(taf[1],17))
  }
  return(taf)
}

# Decode each TAF line (Wind direction - Degrees, Visibility in nm)
decode_forecast_data = function(raw_taf)
{
  print(raw_taf)
  TAF = taf_decoder1(raw_taf)#[2])
  f = TAF[[1]]
  op = as.character(NA)
  start_time = TAF[[4]]
  while (TAF[[2]]){
    TAF = taf_decoder2(TAF[[3]], f, op, start_time)
    f = TAF[[1]]
    op = TAF[[4]]
  }
  
  if(length(raw_taf) > 2){
    for (i in 3:(length(raw_taf))) {
      TAF = taf_decoder2(raw_taf[i], f, op, start_time)
      f = TAF[[1]]
      op = TAF[[4]]
      while (TAF[[2]]){
        TAF = taf_decoder2(TAF[[3]], f, op, start_time)
        f = TAF[[1]]
        op = TAF[[4]]
      }
    }
  }
  f <- subset(f, select = c("time","probability","wind_direction","wind_speed","visibility","ceiling"))
  return(f)
}

# Decoding each TAF data
decode_historical_taf <- function(taf_data){
  taf_lst <- data.frame()
  filtered_taf_data <- filter_data(taf_data)
  for(row in 1:nrow(filtered_taf_data)){
    decoded_taf <- split_taf_string(filtered_taf_data[row,])
    decoded_taf <- decode_forecast_data(decoded_taf)
    decoded_taf <- decoded_taf[1:24,] # Picking only the first 24 hours
    taf_lst <- rbind(taf_lst, decoded_taf)
  }
  rownames(taf_lst) <- NULL
  return(taf_lst)
}

# Preprocess data
preprocess_taf_lst <- function(taf_lst){
  
  # Copying TAF list
  taf_lst_copy <- taf_lst
  
  # Changing column names
  colnames(taf_lst_copy) <- c('time','probability','wind_direction_degrees','wind_speed_knt','visibility_nm','ceiling_ft')
  
  # Generating period information and eliminating NAs
  # taf_lst_copy["period"] <- 0
  for(row in 1:nrow(taf_lst_copy)){
    for(col in colnames(taf_lst_copy)){
      if(is.na(taf_lst_copy[row,col])){
        taf_lst_copy[row,col] <- taf_lst_copy[row-1,col]
      }
    }
    # if(hour(taf_lst_copy$time[row]) >= 3
    #    && hour(taf_lst_copy$time[row]) < 9){
    #   taf_lst_copy[row,"period"] <- 1
    # } else if (hour(taf_lst_copy$time[row]) >= 9
    #            && hour(taf_lst_copy$time[row]) < 15){
    #   taf_lst_copy[row,"period"] <- 2
    # } else if (hour(taf_lst_copy$time[row]) >= 15
    #            && hour(taf_lst_copy$time[row]) < 21){
    #   taf_lst_copy[row,"period"] <- 3
    # } else {
    #   taf_lst_copy[row,"period"] <- 4
    # }
  }
  
  # Initializing new list of TAFs
  new_taf_lst <- data.frame()
  
  # Constructing new list of TAFs for each 15-min period
  for(row in 1:nrow(taf_lst_copy)){
    for(i in 1:4){
      new_taf_lst <- rbind(new_taf_lst,taf_lst_copy[row,])
      new_taf_lst$time[nrow(new_taf_lst)] <- new_taf_lst$time[nrow(new_taf_lst)] + minutes(15 * (i - 1))
    }
  }
  row.names(new_taf_lst) <- NULL
  # colnames(new_taf_lst) <- c('time','probability','wind_direction_degrees','wind_speed_knt','visibility_nm','ceiling_ft')
  
  # Initializing new information
  new_taf_lst[,'weather_condition'] <- 0
  new_taf_lst[,'wind_state'] <- 0
  new_taf_lst[,'headwind_10'] <- new_taf_lst[,'wind_speed_knt'] * cos((new_taf_lst[,'wind_direction_degrees'] - 73) * pi / 180)
  new_taf_lst[,'crosswind_10'] <- new_taf_lst[,'wind_speed_knt'] * sin((new_taf_lst[,'wind_direction_degrees'] - 73) * pi / 180)
  new_taf_lst[,'day'] <- NA
  
  # Converting visibility to km
  # new_taf_lst$visibility_nm <- new_taf_lst$visibility_nm * 1.852
  # colnames(new_taf_lst)[colnames(new_taf_lst) == 'visibility_nm'] <- 'visibility_km'
  
  for(row in 1:nrow(new_taf_lst)){
    
    # Determining period
    if(hour(new_taf_lst$time[row]) >= 6){
      new_taf_lst$day[row] <- paste0(year(new_taf_lst$time[row]),'-')
      new_taf_lst$day[row] <- paste0(new_taf_lst$day[row],paste0(month(new_taf_lst$time[row]),'-'))
      new_taf_lst$day[row] <- paste0(new_taf_lst$day[row],day(new_taf_lst$time[row]))
    } else {
      new_taf_lst$day[row] <- paste0(year(new_taf_lst$time[row] - days(1)),'-')
      new_taf_lst$day[row] <- paste0(new_taf_lst$day[row],paste0(month(new_taf_lst$time[row] - days(1)),'-'))
      new_taf_lst$day[row] <- paste0(new_taf_lst$day[row],day(new_taf_lst$time[row] - days(1)))
    }
    # if(new_taf_lst$period[row] == 4 && hour(new_taf_lst$time[row]) < 21){
    #   new_taf_lst$day[row] <- paste0(new_taf_lst$day[row],paste0(day(new_taf_lst$time[row]-days(1)),'-'))
    # } else {
    #   new_taf_lst$day[row] <- paste0(new_taf_lst$day[row],paste0(day(new_taf_lst$time[row]),'-'))
    # }
    # new_taf_lst$day[row] <- paste0(new_taf_lst$day[row],new_taf_lst$period[row])
    
    # Converting probability to numeric
    if(new_taf_lst[row,'probability'] == 'Equal to or greater than 50%'){
      new_taf_lst[row,'probability'] <- 1
    } else if(new_taf_lst[row,'probability'] == 'Equal to or greater than 40%, but less than 50%'){
      new_taf_lst[row,'probability'] <- 0.4
    } else {
      new_taf_lst[row,'probability'] <- 0.3
    }
    
    # Eliminating NA from ceiling info
    # if(is.na(new_taf_lst[row,'ceiling_ft'])){
    #   new_taf_lst[row,'ceiling_ft'] <- 0
    # }
    
    # Determining weather condition
    if(new_taf_lst[row,'visibility_nm'] >= 3.0 / 1.1508 && new_taf_lst[row,'ceiling_ft'] >= 1000){
      new_taf_lst[row,'weather_condition'] <- 0 # VMC
    } else if(new_taf_lst[row,'visibility_nm'] >= 1.0 / 1.1508 && new_taf_lst[row,'ceiling_ft'] >= 500){
      new_taf_lst[row,'weather_condition'] <- 1 # IMC
    } else {
      new_taf_lst[row,'weather_condition'] <- 2 # LIMC
    }
    
    # Determining wind state
    if( abs(new_taf_lst$crosswind_10[row]) > 20 ){
      new_taf_lst$wind_state[row] <- 3
    } else if(new_taf_lst$headwind_10[row] < -5){
      new_taf_lst$wind_state[row] <- 2
    } else if(new_taf_lst$headwind_10[row] > 5){
      new_taf_lst$wind_state[row] <- 1
    } else {
      new_taf_lst$wind_state[row] <- 0
    }
  }
  new_taf_lst <- new_taf_lst[!duplicated(new_taf_lst),]
  return(new_taf_lst)
}

# Create standard TAF scenarios
std_taf <- function(new_taf_lst){
  new_taf_lst_std <- new_taf_lst
  for(row in 1:nrow(new_taf_lst_std)){
    if(new_taf_lst_std$probability[row] < 1){
      new_taf_lst_std[row,3:(ncol(new_taf_lst_std)-1)] <- new_taf_lst_std[row-1,3:(ncol(new_taf_lst_std)-1)]
    }
  }
  return(new_taf_lst_std)
}

# Loading functions
source("R/forecast_data_decoder_mod.R")

# Loading raw TAF data
taf_data <- read.csv("data\\raw-data\\taf.csv", header=FALSE)
# taf_lst <- read.csv("taf_list.csv", header=TRUE)
# new_taf_lst <- read.csv("new_taf_list.csv", header=TRUE)

# Obtaining decoded TAF data
taf_lst <- decode_historical_taf(taf_data)

# Writing csv
write.csv(taf_lst, file='taf_list_orig.csv', row.names=FALSE)

# Preprocessing data
new_taf_lst <- preprocess_taf_lst(taf_lst)

# Writing csv
write.csv(new_taf_lst, file='new_taf_list.csv', row.names=FALSE)

# Obtaining standard scenario
new_taf_lst_std <- std_taf(new_taf_lst)
write.csv(new_taf_lst_std, file='std_taf.csv', row.names=FALSE)

# Saving workspace
save.image(file='taf_workspace.RData')
