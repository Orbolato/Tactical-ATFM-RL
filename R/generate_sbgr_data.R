# Cleaning workspace
rm(list = ls())

# Libraries
library(plyr)
library(dplyr)
library(ggplot2)
library(nloptr)
library(lubridate)

# Obs.: horários UTC!!!

# Setting work directory
folder <- "./" # Run this script from the repository root
setwd(folder)

# Loading TAF workspace
load('data/taf_workspace.RData')

# Loading flight data from VRA and SIROS
vra <- read.csv("data/raw-data/traffic-data/vra.csv", header=TRUE, sep=";", stringsAsFactors=FALSE, fileEncoding="latin1")
colnames(vra) <- c('Sigla.ICAO.Empresa.Aerea','Empresa.Aerea','Numero.Voo','Codigo.DI',
                   'Codigo.Tipo.Linha','Modelo.Equipamento','Numero.de.Assentos',
                   'Sigla.ICAO.Aeroporto.Origem','Descricao.Aeroporto.Origem',
                   'Partida.Prevista','Partida.Real',
                   'Sigla.ICAO.Aeroporto.Destino','Descricao.Aeroporto.Destino',
                   'Chegada.Prevista','Chegada.Real',
                   'Situacao.Voo','Justificativa','Referencia',
                   'Situacao.Partida','Situacao.Chegada')
vra <- vra[vra$Sigla.ICAO.Aeroporto.Destino == "SBGR" | vra$Sigla.ICAO.Aeroporto.Origem == "SBGR",]
vra <- vra[vra$Situacao.Voo != "CANCELADO",]
vra <- vra[vra$Partida.Prevista != "" & vra$Chegada.Prevista != "" &
             vra$Partida.Real != "" & vra$Chegada.Real != "",]

siros <- data.frame()
for(i in 1:6){
  file <- paste0("data/raw-data/traffic-data/siros_",as.character(i))
  file <- paste0(file, ".csv")
  print(file)
  aux <- read.csv(file, header=TRUE, sep=";", stringsAsFactors=FALSE, fileEncoding="latin1")
  siros <- rbind(siros, aux)
}
siros <- siros[siros$Destino == "SBGR" | siros$Origem == "SBGR",]
siros <- siros[!duplicated(siros),]

# Converting date
vra$Partida.Prevista <- dmy_hm(vra$Partida.Prevista, tz='UTC') + hours(3)
vra$Partida.Real <- dmy_hm(vra$Partida.Real, tz='UTC') + hours(3)
vra$Chegada.Prevista <- dmy_hm(vra$Chegada.Prevista, tz='UTC') + hours(3)
vra$Chegada.Real <- dmy_hm(vra$Chegada.Real, tz='UTC') + hours(3)
siros$Partida.Prevista <- dmy_hm(siros$Partida.Prevista, tz='UTC')
siros$Chegada.Prevista <- dmy_hm(siros$Chegada.Prevista, tz='UTC')

# Loading flight data from FR24
load("data/raw-data/traffic-data/KPI08_2022.rda")
kpi08_2022 <- kpi08[kpi08$real_arr == "SBGR",]
load("data/raw-data/traffic-data/KPI08_2023.rda")
kpi08_2023 <- kpi08[kpi08$real_arr == "SBGR",]
kpi08 <- rbind(kpi08_2022, kpi08_2023)
df_100 <- kpi08[kpi08$variante == "A100",]
df_100 <- df_100[!duplicated(df_100$indicat),]
kpi08 <- kpi08[kpi08$variante == "A40",]
kpi08 <- kpi08[!duplicated(kpi08$indicat),]

# Correcting arrival times by the transit time between 40 NM and landing
transit_40 <- minutes(round(quantile(kpi08[!is.na(kpi08$ref_time), "transit_time"],0.1))) # Mean approach speed of 120 knots
vra$Chegada.Real_40 <- vra$Chegada.Real - transit_40
siros$Chegada.Prevista_40 <- siros$Chegada.Prevista - transit_40

# Correcting arrival times by the transit time between 100 NM and landing
transit_100 <- minutes(round(quantile(df_100[!is.na(df_100$ref_time), "transit_time"],0.1)))
vra$Chegada.Real_100 <- vra$Chegada.Real - transit_100
siros$Chegada.Prevista_100 <- siros$Chegada.Prevista - transit_100

# Filtering
vra_dep <- vra[vra$Sigla.ICAO.Aeroporto.Origem == 'SBGR',]
vra_arr <- vra[vra$Sigla.ICAO.Aeroporto.Destino == 'SBGR',]
siros_dep <- siros[siros$Aeroporto.Origem == 'GUARULHOS - GOVERNADOR ANDRÉ FRANCO MONTORO - GUARULHOS - SP - BRASIL',]
siros_arr <- siros[siros$Aeroporto.Destino == 'GUARULHOS - GOVERNADOR ANDRÉ FRANCO MONTORO - GUARULHOS - SP - BRASIL',]

# Adding fix flags

#1 – EDMUS – East: (-45.38587,-23.18923)
#2 – VUNOX – North: (-46.21324,-22.31790)
#3 – ZARES – West: (-47.95462,-22.76634)
#4 – BUXUK – South: (-47.81587,-23.76391)

wp <- rbind(c(-45.38587,-23.18923),c(-46.21324,-22.31790),c(-47.95462,-22.76634),c(-47.81587,-23.76391))
rownames(wp) <- c('EDMUS','VUNOX','ZARES','BUXUK')
colnames(wp) <- c('lon','lat')

df_100$dist_edmus <- sqrt((df_100$lon - wp['EDMUS','lon'])^2 + (df_100$lat - wp['EDMUS','lat'])^2)
df_100$dist_vunox <- sqrt((df_100$lon - wp['VUNOX','lon'])^2 + (df_100$lat - wp['VUNOX','lat'])^2)
df_100$dist_zares <- sqrt((df_100$lon - wp['ZARES','lon'])^2 + (df_100$lat - wp['ZARES','lat'])^2)
df_100$dist_buxuk <- sqrt((df_100$lon - wp['BUXUK','lon'])^2 + (df_100$lat - wp['BUXUK','lat'])^2)
df_100$fix <- NA

for(i in 1:nrow(df_100)){
  min_dist <- min(df_100$dist_edmus[i], df_100$dist_vunox[i], df_100$dist_zares[i], df_100$dist_buxuk[i])
  if(df_100$dist_edmus[i] == min_dist){
    df_100$fix[i] = 'EDMUS'
  } else if(df_100$dist_vunox[i] == min_dist){
    df_100$fix[i] = 'VUNOX'
  } else if(df_100$dist_zares[i] == min_dist){
    df_100$fix[i] = 'ZARES'
  } else if(df_100$dist_buxuk[i] == min_dist){
    df_100$fix[i] = 'BUXUK'
  }
}

# Determining all aerodromes and the corresponding fixes
all_aerodromes <- cbind(df_100$plan_dep, df_100$fix)
all_aerodromes <- data.frame(all_aerodromes)
colnames(all_aerodromes) <- c('id_icao', 'fix')
all_aerodromes <- all_aerodromes[!duplicated(all_aerodromes$id_icao),]
rownames(all_aerodromes) <- all_aerodromes$id_icao

new_aerodromes <- rbind(c("SBJE",-40.0717,-13.8772),
                        c("SBKP",-47.1344,-23.0075),
                        c("MMSM",-100.9311,22.2543),
                        c("MDSD",-69.6689,18.4297),
                        c("SVPR",-67.9283,10.1497),
                        c("SBVC",-40.8631,-14.8628),
                        c("SBSI",-54.1694,-28.2817),
                        c("SBZM",-47.9647,-19.7647),
                        c("LPBJ",-7.9325,38.0789),
                        c("SBSJ",-45.8628,-23.2281),
                        c("SDSC",-47.9036,-21.8756),
                        c("OTHH",51.6081,25.2731),
                        c("SBPB",-41.7322,-2.8936),
                        c("SBJA",-40.5531,-11.1633),
                        c("LTFM",28.7278,41.2622),
                        c("SILO",-50.8800,-14.2494))

rownames(new_aerodromes) <- c("SBJE","SBKP","MMSM","MDSD","SVPR","SBVC","SBSI","SBZM","LPBJ","SBSJ","SDSC","OTHH",
                              "SBPB","SBJA","LTFM","SILO")

colnames(new_aerodromes) <- c("id_icao","lon","lat")

new_aerodromes <- data.frame(new_aerodromes)

new_aerodromes$lon <- as.numeric(new_aerodromes$lon)
new_aerodromes$lat <- as.numeric(new_aerodromes$lat)

new_aerodromes$fix <- NA

for(i in 1:nrow(new_aerodromes)){
  dist_edmus <- sqrt((new_aerodromes$lon[i] - wp['EDMUS','lon'])^2 + (new_aerodromes$lat[i] - wp['EDMUS','lat'])^2)
  dist_vunox <- sqrt((new_aerodromes$lon[i] - wp['VUNOX','lon'])^2 + (new_aerodromes$lat[i] - wp['VUNOX','lat'])^2)
  dist_zares <- sqrt((new_aerodromes$lon[i] - wp['ZARES','lon'])^2 + (new_aerodromes$lat[i] - wp['ZARES','lat'])^2)
  dist_buxuk <- sqrt((new_aerodromes$lon[i] - wp['BUXUK','lon'])^2 + (new_aerodromes$lat[i] - wp['BUXUK','lat'])^2)
  min_dist <- min(dist_edmus, dist_vunox, dist_zares, dist_buxuk)
  if(dist_edmus == min_dist){
    new_aerodromes$fix[i] = 'EDMUS'
  } else if(dist_vunox == min_dist){
    new_aerodromes$fix[i] = 'VUNOX'
  } else if(dist_zares == min_dist){
    new_aerodromes$fix[i] = 'ZARES'
  } else if(dist_buxuk == min_dist){
    new_aerodromes$fix[i] = 'BUXUK'
  }
}

new_aerodromes <- subset(new_aerodromes, select = -c(lon, lat))

all_aerodromes <- rbind(all_aerodromes, new_aerodromes)

# Determining fixes
vra_arr$fix <- NA
siros_arr$fix <- NA

for(i in 1:nrow(vra_arr)){
  vra_arr$fix[i] <- all_aerodromes[vra_arr$Sigla.ICAO.Aeroporto.Origem[i], 'fix']
  siros_arr$fix[i] <- all_aerodromes[siros_arr$Origem[i], 'fix']
}

# Introducing the missing flights in the KPI 08 database
vra_arr$id_icao <- paste0(vra_arr$Sigla.ICAO.Empresa.Aerea, vra_arr$Numero.Voo)
vra_arr$day <- paste0(year(vra_arr$Chegada.Real),'-',month(vra_arr$Chegada.Real),'-',day(vra_arr$Chegada.Real))
for(i in 1:nrow(vra_arr)){
  n <- nchar(vra_arr$Numero.Voo[i])
  if(n > 4){
    vra_arr$Numero.Voo[i] <- substring(vra_arr$Numero.Voo[i], n-3)
  }
}
vra_arr$flight <- paste0(vra_arr$Sigla.ICAO.Aeroporto.Origem,'-',vra_arr$Numero.Voo,'-',vra_arr$day)

kpi08$day <- paste0(year(kpi08$last_time),'-',month(kpi08$last_time),'-',day(kpi08$last_time))
kpi08$flight_number <- NA
for(i in 1:nrow(kpi08)){
  n <- nchar(kpi08$id_iata[i])
  m <- nchar(kpi08$id_icao[i])
  if(n == 0){
    kpi08$flight_number[i] <- substring(kpi08$id_icao[i], 4)
  } else if(n == 3){
    kpi08$flight_number[i] <- paste0('000',substring(kpi08$id_iata[i], 3))
  } else if(n == 4){
    kpi08$flight_number[i] <- paste0('00',substring(kpi08$id_iata[i], 3))
  } else if(n == 5){
    kpi08$flight_number[i] <- paste0('0',substring(kpi08$id_iata[i], 3))
  } else if(n == 6){
    kpi08$flight_number[i] <- substring(kpi08$id_iata[i], 3)
  } else if(n == 7){
    if(m <= n){
      kpi08$flight_number[i] <- substring(kpi08$id_iata[i], 4)
    } else {
      if(substr(kpi08$id_iata[i],1,3) == substr(kpi08$id_icao[i],1,3)){
        kpi08$flight_number[i] <- substring(kpi08$id_iata[i], 4)
      } else {
        for(j in 3:7){
          aux <- paste0(substr(kpi08$id_iata[i],1,9-j),substr(kpi08$id_iata[i],11-j,7))
          if(aux %in% kpi08$id_iata[kpi08$plan_dep == kpi08$plan_dep[i]]){
            kpi08$flight_number[i] <- substring(aux, 3)
          }
        }
      }
    }
  } else if(n == 8){
    for(j in 4:8){
      aux <- paste0(substr(kpi08$id_iata[i],1,11-j),substr(kpi08$id_iata[i],13-j,8))
      if(aux %in% kpi08$id_iata[kpi08$plan_dep == kpi08$plan_dep[i]]){
        kpi08$flight_number[i] <- substring(aux, 4)
      }
    }
  }
}
kpi08$flight <- paste0(kpi08$plan_dep,'-',kpi08$flight_number,'-',kpi08$day)
vra_kpi08 <- vra_arr[!vra_arr$flight %in% kpi08$flight,]
new_kpi08 <- data.frame(cbind(NA,vra_kpi08$id_icao,NA,
                              vra_kpi08$Sigla.ICAO.Aeroporto.Origem,
                              'SBGR',NA,NA,NA,NA,
                              as.character(vra_kpi08$Chegada.Real_40),
                              as.character(vra_kpi08$Chegada.Real),
                              NA,NA,NA,NA,NA,vra_kpi08$Modelo.Equipamento,
                              NA,NA,vra_kpi08$day,1,40,NA,NA,NA,'A40',
                              NA,NA,NA,NA,NA,NA,NA,NA,NA,
                              vra_kpi08$day,vra_kpi08$Numero.Voo,vra_kpi08$flight))
colnames(new_kpi08) <- colnames(kpi08)
new_kpi08[nchar(new_kpi08$first_time) <= 10, 'first_time'] <- paste0(new_kpi08[nchar(new_kpi08$first_time) <= 10, 'first_time'],
                                                                     " 00:00:00")
new_kpi08[nchar(new_kpi08$last_time) <= 10, 'last_time'] <- paste0(new_kpi08[nchar(new_kpi08$last_time) <= 10, 'last_time'],
                                                                   " 00:00:00")
new_kpi08$first_time <- ymd_hms(new_kpi08$first_time)
new_kpi08$last_time <- ymd_hms(new_kpi08$last_time)
kpi08 <- rbind(kpi08, new_kpi08)
kpi08 <- kpi08[!duplicated(kpi08$last_time),]

kpi08$fix <- NA

for(i in 1:nrow(kpi08)){
  if (kpi08$plan_dep[i] %in% all_aerodromes$id_icao){
    kpi08$fix[i] = all_aerodromes[kpi08$plan_dep[i],'fix']
  }
}

df_100$day <- paste0(year(df_100$last_time),'-',month(df_100$last_time),'-',day(df_100$last_time))
df_100$flight_number <- NA
for(i in 1:nrow(df_100)){
  n <- nchar(df_100$id_iata[i])
  m <- nchar(df_100$id_icao[i])
  if(n == 0){
    df_100$flight_number[i] <- substring(df_100$id_icao[i], 4)
  } else if(n == 3){
    df_100$flight_number[i] <- paste0('000',substring(df_100$id_iata[i], 3))
  } else if(n == 4){
    df_100$flight_number[i] <- paste0('00',substring(df_100$id_iata[i], 3))
  } else if(n == 5){
    df_100$flight_number[i] <- paste0('0',substring(df_100$id_iata[i], 3))
  } else if(n == 6){
    df_100$flight_number[i] <- substring(df_100$id_iata[i], 3)
  } else if(n == 7){
    if(m <= n){
      df_100$flight_number[i] <- substring(df_100$id_iata[i], 4)
    } else {
      if(substr(df_100$id_iata[i],1,3) == substr(df_100$id_icao[i],1,3)){
        df_100$flight_number[i] <- substring(df_100$id_iata[i], 4)
      } else {
        for(j in 3:7){
          aux <- paste0(substr(df_100$id_iata[i],1,9-j),substr(df_100$id_iata[i],11-j,7))
          if(aux %in% df_100$id_iata[df_100$plan_dep == df_100$plan_dep[i]]){
            df_100$flight_number[i] <- substring(aux, 3)
          }
        }
      }
    }
  } else if(n == 8){
    for(j in 4:8){
      aux <- paste0(substr(df_100$id_iata[i],1,11-j),substr(df_100$id_iata[i],13-j,8))
      if(aux %in% df_100$id_iata[df_100$plan_dep == df_100$plan_dep[i]]){
        df_100$flight_number[i] <- substring(aux, 4)
      }
    }
  }
}
df_100$flight <- paste0(df_100$plan_dep,'-',df_100$flight_number,'-',df_100$day)
vra_df_100 <- vra_arr[!vra_arr$flight %in% df_100$flight,]
new_df_100 <- data.frame(cbind(NA,vra_df_100$id_icao,NA,
                               vra_df_100$Sigla.ICAO.Aeroporto.Origem,
                               'SBGR',NA,NA,NA,NA,
                               as.character(vra_df_100$Chegada.Real_40),
                               as.character(vra_df_100$Chegada.Real),
                               NA,NA,NA,NA,NA,vra_df_100$Modelo.Equipamento,
                               NA,NA,vra_df_100$day,1,40,NA,NA,NA,'A40',
                               NA,NA,NA,NA,NA,NA,NA,NA,NA,
                               NA,NA,NA,NA,NA,
                               vra_df_100$day,vra_df_100$Numero.Voo,vra_df_100$flight))
colnames(new_df_100) <- colnames(df_100)
new_df_100[nchar(new_df_100$first_time) <= 10, 'first_time'] <- paste0(new_df_100[nchar(new_df_100$first_time) <= 10, 'first_time'],
                                                                       " 00:00:00")
new_df_100[nchar(new_df_100$last_time) <= 10, 'last_time'] <- paste0(new_df_100[nchar(new_df_100$last_time) <= 10, 'last_time'],
                                                                     " 00:00:00")
new_df_100$first_time <- ymd_hms(new_df_100$first_time)
new_df_100$last_time <- ymd_hms(new_df_100$last_time)

for(i in 1:nrow(new_df_100)){
  if (new_df_100$plan_dep[i] %in% all_aerodromes$id_icao){
    new_df_100$fix[i] = all_aerodromes[new_df_100$plan_dep[i],'fix']
  }
}

df_100 <- rbind(df_100, new_df_100)
df_100 <- df_100[!duplicated(df_100$last_time),]

# Filtering

remove <- c()

for(i in 1:nrow(kpi08)){
  if(is.na(kpi08$fix[i]) | !kpi08$flight[i] %in% df_100$flight){
    remove <- c(remove, i)
  }
}

if(!is.null(remove)){
  kpi08 <- kpi08[-remove,]
}

remove <- c()

for(i in 1:nrow(df_100)){
  if(!df_100$flight[i] %in% kpi08$flight){
    remove <- c(remove, i)
  }
}

if(!is.null(remove)){
  df_100 <- df_100[-remove,]
}

# Loading flight data from FR24
kpi17_full <- data.frame()
folder <- "data/raw-data/traffic-data/KPI17_v2/"
for (i in 2022:2023) {
  file_i <- paste0("KPI17_", as.character(i))
  folder_i <- paste0(folder, as.character(i))
  folder_i <- paste0(folder_i, "/")
  if (i %% 4 == 0) {
    if (i %% 100 == 0 & i %% 400 != 0) {
      month_days = c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    } else {
      month_days = c(31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    }
  } else {
    month_days = c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
  }
  for (j in 1:9) {
    file_ij <- paste0(file_i, "0")
    file_ij <- paste0(file_ij, as.character(j))
    folder_ij <- paste0(folder_i, "0")
    folder_ij <- paste0(folder_ij, as.character(j))
    folder_ij <- paste0(folder_ij, "/")
    for (k in 1:9) {
      file_ijk <- paste0(file_ij, "0")
      file_ijk <- paste0(file_ijk, as.character(k))
      file_ijk <- paste0(file_ijk, ".Rda")
      tryCatch({
        load(paste0(folder_ij, file_ijk))
        kpi17_full <- rbind(kpi17_full, kpi17[kpi17$plan_dep == "SBGR",])
        print(file_ijk)
      }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    }
    for (k in 10:month_days[j]) {
      file_ijk <- paste0(file_ij, as.character(k))
      file_ijk <- paste0(file_ijk, ".Rda")
      tryCatch({
        load(paste0(folder_ij, file_ijk))
        kpi17_full <- rbind(kpi17_full, kpi17[kpi17$plan_dep == "SBGR",])
        print(file_ijk)
      }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    }
  }
  for (j in 10:12) {
    file_ij <- paste0(file_i, as.character(j))
    folder_ij <- paste0(folder_i, as.character(j))
    folder_ij <- paste0(folder_ij, "/")
    for (k in 1:9) {
      file_ijk <- paste0(file_ij, "0")
      file_ijk <- paste0(file_ijk, as.character(k))
      file_ijk <- paste0(file_ijk, ".Rda")
      tryCatch({
        load(paste0(folder_ij, file_ijk))
        kpi17_full <- rbind(kpi17_full, kpi17[kpi17$plan_dep == "SBGR",])
        print(file_ijk)
      }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    }
    for (k in 10:month_days[j]) {
      file_ijk <- paste0(file_ij, as.character(k))
      file_ijk <- paste0(file_ijk, ".Rda")
      tryCatch({
        load(paste0(folder_ij, file_ijk))
        kpi17_full <- rbind(kpi17_full, kpi17[kpi17$plan_dep == "SBGR",])
        print(file_ijk)
      }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    }
  }
}
kpi17_full <- kpi17_full[!duplicated(kpi17_full$indicat),]

# Loading flight data from TATIC
tatic <- data.frame()
folder <- "data/raw-data/traffic-data/BD_TF_Raw_v2/"
for (i in 2022:2023) {
  file_i <- paste0("TF_", as.character(i))
  folder_i <- paste0(folder, as.character(i))
  folder_i <- paste0(folder_i, "/")
  if (i %% 4 == 0) {
    if (i %% 100 == 0 & i %% 400 != 0) {
      month_days = c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    } else {
      month_days = c(31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    }
  } else {
    month_days = c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
  }
  for (j in 1:9) {
    file_ij <- paste0(file_i, "0")
    file_ij <- paste0(file_ij, as.character(j))
    folder_ij <- paste0(folder_i, "0")
    folder_ij <- paste0(folder_ij, as.character(j))
    folder_ij <- paste0(folder_ij, "/")
    for (k in 1:9) {
      file_ijk <- paste0(file_ij, "0")
      file_ijk <- paste0(file_ijk, as.character(k))
      file_ijk <- paste0(file_ijk, ".Rda")
      tryCatch({
        load(paste0(folder_ij, file_ijk))
        tatic <- rbind(tatic, datax_dep_tatic[datax_dep_tatic$adep == "SBGR",])
        print(file_ijk)
      }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    }
    for (k in 10:month_days[j]) {
      file_ijk <- paste0(file_ij, as.character(k))
      file_ijk <- paste0(file_ijk, ".Rda")
      tryCatch({
        load(paste0(folder_ij, file_ijk))
        tatic <- rbind(tatic, datax_dep_tatic[datax_dep_tatic$adep == "SBGR",])
        print(file_ijk)
      }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    }
  }
  for (j in 10:12) {
    file_ij <- paste0(file_i, as.character(j))
    folder_ij <- paste0(folder_i, as.character(j))
    folder_ij <- paste0(folder_ij, "/")
    for (k in 1:9) {
      file_ijk <- paste0(file_ij, "0")
      file_ijk <- paste0(file_ijk, as.character(k))
      file_ijk <- paste0(file_ijk, ".Rda")
      tryCatch({
        load(paste0(folder_ij, file_ijk))
        tatic <- rbind(tatic, datax_dep_tatic[datax_dep_tatic$adep == "SBGR",])
        print(file_ijk)
      }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    }
    for (k in 10:month_days[j]) {
      file_ijk <- paste0(file_ij, as.character(k))
      file_ijk <- paste0(file_ijk, ".Rda")
      tryCatch({
        load(paste0(folder_ij, file_ijk))
        tatic <- rbind(tatic, datax_dep_tatic[datax_dep_tatic$adep == "SBGR",])
        print(file_ijk)
      }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    }
  }
}
tatic <- tatic[!duplicated(tatic$id),]
# tatic <- tatic[tatic$flighttype == 'S',]
tatic <- tatic[!(is.na(tatic$eobt) | is.na(tatic$wpush) | is.na(tatic$cpush) |
                   is.na(tatic$wtaxi) | is.na(tatic$taxi) | is.na(tatic$hold) |
                   is.na(tatic$crwy) | is.na(tatic$cdep) | is.na(tatic$dep)),]
tatic$createdat <- ymd_hms(tatic$createdat)
tatic$eobt <- ymd_hms(tatic$eobt)
tatic$wpush <- ymd_hms(tatic$wpush)
tatic$cpush <- ymd_hms(tatic$cpush)
tatic$wtaxi <- ymd_hms(tatic$wtaxi)
tatic$taxi <- ymd_hms(tatic$taxi)
tatic$hold <- ymd_hms(tatic$hold)
tatic$crwy <- ymd_hms(tatic$crwy)
tatic$cdep <- ymd_hms(tatic$cdep)
tatic$dep <- ymd_hms(tatic$dep)

remove <- c()

for(i in 1:nrow(tatic)){
  if(!(tatic$dep[i] >= tatic$cdep[i] & tatic$cdep[i] >= tatic$crwy[i] & 
       tatic$crwy[i] >= tatic$hold[i] & tatic$hold[i] >= tatic$taxi[i] &
       tatic$taxi[i] >= tatic$wtaxi[i] & tatic$wtaxi[i] >= tatic$cpush[i] &
       tatic$cpush[i] >= tatic$wpush[i])){
    remove <- c(remove, i)
  }
}

tatic <- tatic[-remove,]

# Preprocessing taxi time
taxi_corr <- minutes(round(mean(as.numeric(tatic$dep[!is.na(tatic$cpush) & !is.na(tatic$dep)] - tatic$cpush[!is.na(tatic$cpush) & !is.na(tatic$dep)]))/60))
tatic$cpush[is.na(tatic$cpush)] <- tatic$dep[is.na(tatic$cpush)] - taxi_corr
unimpeded_taxi_time <- minutes(round(quantile(tatic$dep[!is.na(tatic$cpush) & !is.na(tatic$dep)] - tatic$cpush[!is.na(tatic$cpush) & !is.na(tatic$dep)], 0.1)/60))

# Loading METAR
load("data/raw-data/metar_sbgr_2019_2023.Rda")
metar <- metar[metar$valid > ymd_hms("2021-12-31 00:00:00", tz="UTC"),]

# Creating dataframe
col_names <- c('time',
               'scheduled_arrivals_40_edmus','scheduled_arrivals_40_vunox',
               'scheduled_arrivals_40_zares','scheduled_arrivals_40_buxuk',
               'scheduled_arrivals_40',
               'arrival_demand_40_edmus','arrival_demand_40_vunox',
               'arrival_demand_40_zares','arrival_demand_40_buxuk',
               'arrival_demand_40',
               'actual_arrivals_edmus','actual_arrivals_vunox',
               'actual_arrivals_zares','actual_arrivals_buxuk',
               'actual_arrivals',
               'base_arrivals_40_edmus','base_arrivals_40_vunox',
               'base_arrivals_40_zares','base_arrivals_40_buxuk',
               'base_arrivals_40',
               'scheduled_arrivals_100_edmus','scheduled_arrivals_100_vunox',
               'scheduled_arrivals_100_zares','scheduled_arrivals_100_buxuk',
               'scheduled_arrivals_100',
               'arrival_demand_100_edmus','arrival_demand_100_vunox',
               'arrival_demand_100_zares','arrival_demand_100_buxuk',
               'arrival_demand_100',
               'base_arrivals_100_edmus','base_arrivals_100_vunox',
               'base_arrivals_100_zares','base_arrivals_100_buxuk',
               'base_arrivals_100',
               'scheduled_departures','departure_demand',
               'actual_departures','base_departures',
               'configuration','base_configuration',
               'sep_edmus','sep_vunox','sep_zares','sep_buxuk',
               'vsby','ceiling','ifr','lifr',
               'drct','sknt','headwind_10','crosswind_10','wind_state',
               'day')
df <- data.frame(matrix(0,nrow(new_taf_lst),length(col_names)))
colnames(df) <- col_names
df$time <- as.POSIXct(df$time, tz="UTC")

for(row in 1:nrow(df)){
  
  # Printing time
  time <- new_taf_lst$time[row]
  day <- new_taf_lst$day[row]
  
  # Determining time and day
  df$time[row] <- time
  df$day[row] <- day
  print(df$time[row])
  
  # Determining last valid metar
  last_metar <- tail(metar[date(metar$valid) <= date(time) & hour(metar$valid) <= hour(time),],1)
  
  # Determining visibility, ceiling and meteorological condition
  df$vsby[row] <- last_metar$vsby
  df$ceiling[row] <- last_metar$ceiling
  df$ifr[row] <- last_metar$ifr
  df$lifr[row] <- last_metar$lifr
  
  # Determining wind speed (knots), direction (degrees) and wind state (headwind and crosswind)
  df$sknt[row] <- last_metar$wind_speed
  df$drct[row] <- last_metar$wind_direction
  df$headwind_10[row] <- df$sknt[row] * cos((df$drct[row] - 73) * pi / 180)
  df$crosswind_10[row] <- df$sknt[row] * sin((df$drct[row] - 73) * pi / 180)
  if( abs(df$crosswind_10[row]) > 20 ){
    df$wind_state[row] <- 3
  } else if(df$headwind_10[row] < -5){
    df$wind_state[row] <- 2
  } else if(df$headwind_10[row] > 5){
    df$wind_state[row] <- 1
  } else {
    df$wind_state[row] <- 0
  }
  
  # Determining runway configuration
  arr <- kpi08[kpi08$last_time >= time & kpi08$last_time < time + minutes(15),]
  n_10 <- nrow(arr[!is.na(arr$pista) & arr$pista == 73,])
  n_28 <- nrow(arr[!is.na(arr$pista) & arr$pista == 253,])
  if(n_10 >= n_28) {
    df$configuration[row] <- "10R|10L"
  } else {
    df$configuration[row] <- "28L|28R"
  }
  
  # Determining base runway configuration
  arr_base <- kpi08[kpi08$last_time >= time + transit_40 & kpi08$last_time < time + minutes(15) + transit_40,]
  n_10_base <- nrow(arr_base[!is.na(arr_base$pista) & arr_base$pista == 73,])
  n_28_base <- nrow(arr_base[!is.na(arr_base$pista) & arr_base$pista == 253,])
  if(n_10_base >= n_28_base) {
    df$base_configuration[row] <- "10R|10L"
  } else {
    df$base_configuration[row] <- "28L|28R"
  }
  
  # Determining scheduled arrivals at 40 NM (VRA)
  df$scheduled_arrivals_40_edmus[row] <- nrow(siros_arr[siros_arr$Chegada.Prevista_40 >= time &
                                                        siros_arr$Chegada.Prevista_40 < time + minutes(15) & siros_arr$fix == 'EDMUS',])
  df$scheduled_arrivals_40_vunox[row] <- nrow(siros_arr[siros_arr$Chegada.Prevista_40 >= time &
                                                        siros_arr$Chegada.Prevista_40 < time + minutes(15) & siros_arr$fix == 'VUNOX',])
  df$scheduled_arrivals_40_zares[row] <- nrow(siros_arr[siros_arr$Chegada.Prevista_40 >= time &
                                                        siros_arr$Chegada.Prevista_40 < time + minutes(15) & siros_arr$fix == 'ZARES',])
  df$scheduled_arrivals_40_buxuk[row] <- nrow(siros_arr[siros_arr$Chegada.Prevista_40 >= time &
                                                        siros_arr$Chegada.Prevista_40 < time + minutes(15) & siros_arr$fix == 'BUXUK',])
  df$scheduled_arrivals_40[row] <- nrow(siros_arr[siros_arr$Chegada.Prevista_40 >= time &
                                                  siros_arr$Chegada.Prevista_40 < time + minutes(15),])
  
  # Determining demand of arrivals at 40 NM (FR24)
  df$arrival_demand_40_edmus[row] <- nrow(kpi08[kpi08$first_time >= time &
                                        kpi08$first_time < time + minutes(15) & 
                                        kpi08$fix == 'EDMUS',])
  df$arrival_demand_40_vunox[row] <- nrow(kpi08[kpi08$first_time >= time &
                                               kpi08$first_time < time + minutes(15) & 
                                               kpi08$fix == 'VUNOX',])
  df$arrival_demand_40_zares[row] <- nrow(kpi08[kpi08$first_time >= time &
                                               kpi08$first_time < time + minutes(15) & 
                                               kpi08$fix == 'ZARES',])
  df$arrival_demand_40_buxuk[row] <- nrow(kpi08[kpi08$first_time >= time &
                                               kpi08$first_time < time + minutes(15) & 
                                               kpi08$fix == 'BUXUK',])
  df$arrival_demand_40[row] <- nrow(kpi08[kpi08$first_time >= time &
                                         kpi08$first_time < time + minutes(15),])
  
  # Determining actual arrivals (FR24 - KPI 08)
  df$actual_arrivals_edmus[row] <- nrow(kpi08[kpi08$last_time >= time & 
                                                kpi08$last_time < time + minutes(15) & 
                                                kpi08$fix == 'EDMUS',])
  df$actual_arrivals_vunox[row] <- nrow(kpi08[kpi08$last_time >= time & 
                                                kpi08$last_time < time + minutes(15) & 
                                                kpi08$fix == 'VUNOX',])
  df$actual_arrivals_zares[row] <- nrow(kpi08[kpi08$last_time >= time & 
                                                kpi08$last_time < time + minutes(15) & 
                                                kpi08$fix == 'ZARES',])
  df$actual_arrivals_buxuk[row] <- nrow(kpi08[kpi08$last_time >= time & 
                                                kpi08$last_time < time + minutes(15) & 
                                                kpi08$fix == 'BUXUK',])
  df$actual_arrivals[row] <- nrow(kpi08[kpi08$last_time >= time & kpi08$last_time < time + minutes(15),])
  
  # Determining actual arrivals at 40 NM (FR24 - KPI 08 - With translation)
  df$base_arrivals_40_edmus[row] <- nrow(kpi08[kpi08$last_time >= time + transit_40 & 
                                                kpi08$last_time < time + minutes(15) + transit_40 & 
                                                kpi08$fix == 'EDMUS',])
  df$base_arrivals_40_vunox[row] <- nrow(kpi08[kpi08$last_time >= time + transit_40 & 
                                                kpi08$last_time < time + minutes(15) + transit_40 & 
                                                kpi08$fix == 'VUNOX',])
  df$base_arrivals_40_zares[row] <- nrow(kpi08[kpi08$last_time >= time + transit_40 & 
                                                kpi08$last_time < time + minutes(15) + transit_40 & 
                                                kpi08$fix == 'ZARES',])
  df$base_arrivals_40_buxuk[row] <- nrow(kpi08[kpi08$last_time >= time + transit_40 & 
                                                kpi08$last_time < time + minutes(15) + transit_40 & 
                                                kpi08$fix == 'BUXUK',])
  df$base_arrivals_40[row] <- nrow(kpi08[kpi08$last_time >= time + transit_40 & kpi08$last_time < time + minutes(15) + transit_40,])
  
  # Determining scheduled arrivals at 100 NM (VRA)
  df$scheduled_arrivals_100_edmus[row] <- nrow(siros_arr[siros_arr$Chegada.Prevista_100 >= time &
                                                         siros_arr$Chegada.Prevista_100 < time + minutes(15) & siros_arr$fix == 'EDMUS',])
  df$scheduled_arrivals_100_vunox[row] <- nrow(siros_arr[siros_arr$Chegada.Prevista_100 >= time &
                                                         siros_arr$Chegada.Prevista_100 < time + minutes(15) & siros_arr$fix == 'VUNOX',])
  df$scheduled_arrivals_100_zares[row] <- nrow(siros_arr[siros_arr$Chegada.Prevista_100 >= time &
                                                         siros_arr$Chegada.Prevista_100 < time + minutes(15) & siros_arr$fix == 'ZARES',])
  df$scheduled_arrivals_100_buxuk[row] <- nrow(siros_arr[siros_arr$Chegada.Prevista_100 >= time &
                                                         siros_arr$Chegada.Prevista_100 < time + minutes(15) & siros_arr$fix == 'BUXUK',])
  df$scheduled_arrivals_100[row] <- nrow(siros_arr[siros_arr$Chegada.Prevista_100 >= time &
                                                   siros_arr$Chegada.Prevista_100 < time + minutes(15),])
  
  # Determining demand of arrivals at 100 NM (FR24)
  df$arrival_demand_100_edmus[row] <- nrow(df_100[df_100$first_time >= time &
                                               df_100$first_time < time + minutes(15) & 
                                               df_100$fix == 'EDMUS',])
  df$arrival_demand_100_vunox[row] <- nrow(df_100[df_100$first_time >= time &
                                               df_100$first_time < time + minutes(15) & 
                                               df_100$fix == 'VUNOX',])
  df$arrival_demand_100_zares[row] <- nrow(df_100[df_100$first_time >= time &
                                               df_100$first_time < time + minutes(15) & 
                                               df_100$fix == 'ZARES',])
  df$arrival_demand_100_buxuk[row] <- nrow(df_100[df_100$first_time >= time &
                                               df_100$first_time < time + minutes(15) & 
                                               df_100$fix == 'BUXUK',])
  df$arrival_demand_100[row] <- nrow(df_100[df_100$first_time >= time &
                                         df_100$first_time < time + minutes(15),])
  
  # Determining actual arrivals at 100 NM (FR24 - KPI 08 - With translation)
  df$base_arrivals_100_edmus[row] <- nrow(df_100[df_100$last_time >= time + transit_100 & 
                                                   df_100$last_time < time + minutes(15) + transit_100 & 
                                                   df_100$fix == 'EDMUS',])
  df$base_arrivals_100_vunox[row] <- nrow(df_100[df_100$last_time >= time + transit_100 & 
                                                  df_100$last_time < time + minutes(15) + transit_100 & 
                                                  df_100$fix == 'VUNOX',])
  df$base_arrivals_100_zares[row] <- nrow(df_100[df_100$last_time >= time + transit_100 & 
                                                   df_100$last_time < time + minutes(15) + transit_100 & 
                                                   df_100$fix == 'ZARES',])
  df$base_arrivals_100_buxuk[row] <- nrow(df_100[df_100$last_time >= time + transit_100 & 
                                                   df_100$last_time < time + minutes(15) + transit_100 & 
                                                   df_100$fix == 'BUXUK',])
  df$base_arrivals_100[row] <- nrow(df_100[df_100$last_time >= time + transit_100 & df_100$last_time < time + minutes(15) + transit_100,])
  
  # Determining scheduled departures (VRA)
  df$scheduled_departures[row] <- nrow(siros_dep[siros_dep$Partida.Prevista >= time &
                                                   siros_dep$Partida.Prevista < time + minutes(15),])
  
  # Determining demand of departures (TATIC)
  df$departure_demand[row] <- nrow(tatic[tatic$cpush >= time &
                                         tatic$cpush < time + minutes(15),])
  
  # Determining actual departures (FR24 - KPI 17 or TATIC)
  df$actual_departures[row] <- nrow(tatic[tatic$dep >= time &
                                         tatic$dep < time + minutes(15),])
  # df$actual_departures[row] <- nrow(kpi17_full[kpi17_full$time >= time &
                                                 # kpi17_full$time < time + minutes(15),])
  
  # Determining base departures (FR24 - KPI 17 or TATIC - With translation)
  df$base_departures[row] <- nrow(tatic[tatic$dep >= time + unimpeded_taxi_time &
                                         tatic$dep < time + minutes(15) + unimpeded_taxi_time,])
  # df$base_departures[row] <- nrow(kpi17_full[kpi17_full$time >= time + unimpeded_taxi_time &
  #                                                kpi17_full$time < time + minutes(15) + unimpeded_taxi_time,])
  
}
df <- df[df$time >= as.POSIXct("2022-01-02 06:00", tz="UTC") &
           df$time < as.POSIXct("2023-12-30 06:00", tz="UTC"),]
df <- df[df$day != "2023-04-24" & df$day != "2023-04-25" & df$day != "2023-04-26" &
           df$day != "2023-04-27" & df$day != "2023-04-28",]
for (row in 1:nrow(df)) {
  if(hour(df$time[row]) >= 6){
    df$day[row] <- paste0(year(df$time[row]),'-')
    df$day[row] <- paste0(df$day[row],paste0(month(df$time[row]),'-'))
    df$day[row] <- paste0(df$day[row],day(df$time[row]))
  } else {
    df$day[row] <- paste0(year(df$time[row] - days(1)),'-')
    df$day[row] <- paste0(df$day[row],paste0(month(df$time[row] - days(1)),'-'))
    df$day[row] <- paste0(df$day[row],day(df$time[row] - days(1)))
  }
}
df$sep_edmus <- 5
df$sep_vunox <- 5
df$sep_zares <- 5
df$sep_buxuk <- 5

# for(row in 1:nrow(df)){
# 
#   time = df$time[row]
# 
#   print(time)
# 
#   df$actual_arrivals[row] <- nrow(kpi08[kpi08$last_time >= time & kpi08$last_time < time + minutes(15),])
#   df$base_arrivals_40[row] <- nrow(kpi08[kpi08$last_time >= time + transit_40 & kpi08$last_time < time + minutes(15) + transit_40,])
#   
#   # Determining actual arrivals at 100 NM (FR24 - KPI 08 - With translation)
#   df$base_arrivals_100_edmus[row] <- nrow(df_100[df_100$last_time >= time + transit_100 & 
#                                                    df_100$last_time < time + minutes(15) + transit_100 & 
#                                                    df_100$fix == 'EDMUS',])
#   df$base_arrivals_100_vunox[row] <- nrow(df_100[df_100$last_time >= time + transit_100 & 
#                                                    df_100$last_time < time + minutes(15) + transit_100 & 
#                                                    df_100$fix == 'VUNOX',])
#   df$base_arrivals_100_zares[row] <- nrow(df_100[df_100$last_time >= time + transit_100 & 
#                                                    df_100$last_time < time + minutes(15) + transit_100 & 
#                                                    df_100$fix == 'ZARES',])
#   df$base_arrivals_100_buxuk[row] <- nrow(df_100[df_100$last_time >= time + transit_100 & 
#                                                    df_100$last_time < time + minutes(15) + transit_100 & 
#                                                    df_100$fix == 'BUXUK',])
#   df$base_arrivals_100[row] <- nrow(df_100[df_100$last_time >= time + transit_100 & df_100$last_time < time + minutes(15) + transit_100,])
# }
# save(df, file="data/data_gru.Rda")
# save.image("data/sbgr_data.RData")
# write.csv(df, 'data/data.csv', row.names=FALSE)