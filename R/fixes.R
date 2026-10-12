library(leaflet)
#library(dplyr)
#library(purrr)
library(conflicted)
#conflict_prefer("map", "maps")
library(mapview)
library(magick)
library(webshot)

# Work directory: run this script from the repository root

# Loading flight data from FR24
load("data/raw-data/traffic-data/KPI08_2022.rda")
kpi08_2022 <- kpi08[kpi08$real_arr == "SBGR",]
load("data/raw-data/traffic-data/KPI08_2023.rda")
kpi08_2023 <- kpi08[kpi08$real_arr == "SBGR",]
kpi08 <- rbind(kpi08_2022, kpi08_2023)

# Loading STAR data
load("data/all_proc.rda")

sbgr_star <- all_proc[all_proc$airport == 'SBGR' & all_proc$`SID/STAR` == 'STAR',]
sbgr_star_filtered <- sbgr_star[!duplicated(sbgr_star$waypoint),]

edmus_1a <- sbgr_star[sbgr_star$procedure_set == 'EDMUS 1a',]
edmus_1b_E <- sbgr_star[sbgr_star$procedure_set == 'EDMUS 1b' & sbgr_star$entry_exit_waypoint == 'EDMUS',]
edmus_1b_N <- sbgr_star[sbgr_star$procedure_set == 'EDMUS 1b' & sbgr_star$entry_exit_waypoint == 'VUNOX',]
evral_1a_W <- sbgr_star[sbgr_star$procedure_set == 'RNAV EVRAL 1A' & sbgr_star$entry_exit_waypoint == 'ZARES',]
evral_1a_S <- sbgr_star[sbgr_star$procedure_set == 'RNAV EVRAL 1A' & sbgr_star$entry_exit_waypoint == 'BUXUK',]
evral_1b_W <- sbgr_star[sbgr_star$procedure_set == 'RNAV EVRAL 1B' & sbgr_star$entry_exit_waypoint == 'ZARES',]
evral_1b_S <- sbgr_star[sbgr_star$procedure_set == 'RNAV EVRAL 1B' & sbgr_star$entry_exit_waypoint == 'BUXUK',]
evral_1c_W <- sbgr_star[sbgr_star$procedure_set == 'RNAV EVRAL 1C' & sbgr_star$entry_exit_waypoint == 'ZARES',]
evral_1c_S <- sbgr_star[sbgr_star$procedure_set == 'RNAV EVRAL 1C' & sbgr_star$entry_exit_waypoint == 'BUXUK',]
evral_1d_W <- sbgr_star[sbgr_star$procedure_set == 'RNAV EVRAL 1D' & sbgr_star$entry_exit_waypoint == 'ZARES',]
evral_1d_S <- sbgr_star[sbgr_star$procedure_set == 'RNAV EVRAL 1D' & sbgr_star$entry_exit_waypoint == 'BUXUK',]
vunox_1a <- sbgr_star[sbgr_star$procedure_set == 'RNAV VUNOX 1A',]
vunox_1c <- sbgr_star[sbgr_star$procedure_set == 'RNAV VUNOX 1C',]
sanpa_1a <- sbgr_star[sbgr_star$procedure_set == 'RNAV SANPA',]

longlat <- data.frame(sbgr_star_filtered$long, sbgr_star_filtered$lat)
colnames(longlat) <- c('long', 'lat')
rownames(longlat) <- sbgr_star_filtered$waypoint

#1 – EDMUS – East: (-45.38587,-23.18923)
#2 – VUNOX – North: (-46.21324,-22.31790)
#3 – ZARES – West: (-47.95462,-22.76634)
#4 – BUXUK – South: (-47.81587,-23.76391)

map <- leaflet(options = leafletOptions(
  attributionControl = FALSE,
  zoomControl = FALSE
  )) %>%
  addProviderTiles(provider = providers$CartoDB) %>%
  setView(lng = -46.47306 , lat = -23.43556, zoom = 8) %>%
  addCircles(lng = -46.47306 , lat = -23.43556,  color = 'black', weight = 4, radius = 185200, fill = FALSE, stroke = TRUE, fillOpacity = 0.7, dashArray = 5) %>% #100NM
  addCircles(lng = -46.47306 , lat = -23.43556 , color = 'black', weight = 4, radius = 74080, fill = FALSE, stroke = TRUE, fillOpacity = 0.7, dashArray = 5) %>% #40NM
  addCircles(data = longlat, weight = 10, color = 'blue', label = rownames(longlat)) %>%
  addPolylines(lng = edmus_1a$long, lat = edmus_1a$lat, color = 'black', weight = 4, label = 'EDMUS 1A - EAST') %>%
  addPolylines(lng = edmus_1b_E$long, lat = edmus_1b_E$lat, color = 'black', weight = 4, label = 'EDMUS 1B EDMUS - EAST') %>%
  addPolylines(lng = edmus_1b_N$long, lat = edmus_1b_N$lat, color = 'black', weight = 4, label = 'EDMUS 1B VUNOX - NORTH') %>%  
  addPolylines(lng = evral_1a_W$long, lat = evral_1a_W$lat, color = 'black', weight = 4, label = 'EVRAL 1A ZARES - WEST') %>% 
  addPolylines(lng = evral_1a_S$long, lat = evral_1a_S$lat, color = 'black', weight = 4, label = 'EVRAL 1A BUXUK - SOUTH') %>%
  addPolylines(lng = evral_1b_W$long, lat = evral_1b_W$lat, color = 'black', weight = 4, label = 'EVRAL 1B ZARES - WEST') %>% 
  addPolylines(lng = evral_1b_S$long, lat = evral_1b_S$lat, color = 'black', weight = 4, label = 'EVRAL 1B BUXUK - SOUTH') %>% 
  addPolylines(lng = evral_1c_W$long, lat = evral_1c_W$lat, color = 'black', weight = 4, label = 'EVRAL 1C ZARES - WEST') %>% 
  addPolylines(lng = evral_1c_S$long, lat = evral_1c_S$lat, color = 'black', weight = 4, label = 'EVRAL 1C BUXUK - SOUTH') %>% 
  addPolylines(lng = evral_1d_W$long, lat = evral_1d_W$lat, color = 'black', weight = 4, label = 'EVRAL 1D ZARES - WEST') %>% 
  addPolylines(lng = evral_1d_S$long, lat = evral_1d_S$lat, color = 'black', weight = 4, label = 'EVRAL 1D BUXUK - SOUTH') %>% 
  addPolylines(lng = vunox_1a$long, lat = vunox_1a$lat, color = 'black', weight = 4, label = 'VUNOX 1A - NORTH') %>% 
  addPolylines(lng = vunox_1c$long, lat = vunox_1c$lat, color = 'black', weight = 4, label = 'VUNOX 1C - NORTH') %>% 
  addPolylines(lng = sanpa_1a$long, lat = sanpa_1a$lat, color = 'black', weight = 4, label = 'SANPA 1A - NORTH') %>% 
  addCircles(data = data.frame(long = c(-46.47306), lat = c(-23.43556)), weight = 20, color = 'red', label = 'SBGR') %>%
  htmlwidgets::onRender("
    function(el, x) {
      document.getElementsByClassName('leaflet-control-container')[0].style.display = 'none';
      document.getElementsByClassName('leaflet-bottom leaflet-right')[0].style.display = 'none';
    }
  ")
map

mapshot(map, file = "map.pdf", zoom=0.75)  # or .svg

wp <- rbind(c(-45.38587,-23.18923),c(-46.21324,-22.31790),c(-47.95462,-22.76634),c(-47.81587,-23.76391))
rownames(wp) <- c('EDMUS','VUNOX','ZARES','BUXUK')
colnames(wp) <- c('lon','lat')

#kpi08 <- kpi08[kpi08$type == 'ARR_BR',]

df_100 <- kpi08[kpi08$variante == 'A100' & kpi08$real_arr == 'SBGR',]

df_100$dist_edmus <- sqrt((df_100$lon - wp['EDMUS','lon'])^2 + (df_100$lat - wp['EDMUS','lat'])^2)
df_100$dist_vunox <- sqrt((df_100$lon - wp['VUNOX','lon'])^2 + (df_100$lat - wp['VUNOX','lat'])^2)
df_100$dist_zares <- sqrt((df_100$lon - wp['ZARES','lon'])^2 + (df_100$lat - wp['ZARES','lat'])^2)
df_100$dist_buxuk <- sqrt((df_100$lon - wp['BUXUK','lon'])^2 + (df_100$lat - wp['BUXUK','lat'])^2)
df_100$fix <- df_100$variante

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

df_100_edmus <- df_100[df_100$fix == 'EDMUS',]
df_100_vunox <- df_100[df_100$fix == 'VUNOX',]
df_100_zares <- df_100[df_100$fix == 'ZARES',]
df_100_buxuk <- df_100[df_100$fix == 'BUXUK',]

df_40 <- kpi08[kpi08$variante == 'A40' & kpi08$real_arr == 'SBGR',]
df_40$fix <- NA

for(i in 1:nrow(df_100)){
  if(nrow(df_40[df_40$indicat == df_100$indicat[i],]) >= 1){
    df_40$fix[df_40$indicat == df_100$indicat[i]] = df_100$fix[i]
  }
}

remove <- c()

for(i in 1:nrow(df_40)){
  if(is.na(df_40$fix[i])){
    remove <- c(remove, i)
  }
}

df_40 <- df_40[-remove,]

df_40_edmus <- df_40[df_40$fix == 'EDMUS',]
df_40_vunox <- df_40[df_40$fix == 'VUNOX',]
df_40_zares <- df_40[df_40$fix == 'ZARES',]
df_40_buxuk <- df_40[df_40$fix == 'BUXUK',]

map <- leaflet() %>%
  addProviderTiles(provider = providers$CartoDB) %>%
  addCircles(lng = -46.47306 , lat = -23.43556,  color = 'black', weight = 2.5, radius = 185200, fill = FALSE, stroke = TRUE, fillOpacity = 0.7, dashArray = 5) %>% #100NM
  addCircles(lng = -46.47306 , lat = -23.43556 , color = 'black', weight = 2.5, radius = 74080, fill = FALSE, stroke = TRUE, fillOpacity = 0.7, dashArray = 5) %>% #40NM
  addCircles(data = longlat, weight = 5, color = 'cyan', label = rownames(longlat)) %>%
  addPolylines(lng = edmus_1a$long, lat = edmus_1a$lat, color = 'black', weight = 2.5, label = 'EDMUS 1A - EAST') %>%
  addPolylines(lng = edmus_1b_E$long, lat = edmus_1b_E$lat, color = 'black', weight = 2.5, label = 'EDMUS 1B EDMUS - EAST') %>%
  addPolylines(lng = edmus_1b_N$long, lat = edmus_1b_N$lat, color = 'black', weight = 2.5, label = 'EDMUS 1B VUNOX - NORTH') %>%  
  addPolylines(lng = evral_1a_W$long, lat = evral_1a_W$lat, color = 'black', weight = 2.5, label = 'EVRAL 1A ZARES - WEST') %>% 
  addPolylines(lng = evral_1a_S$long, lat = evral_1a_S$lat, color = 'black', weight = 2.5, label = 'EVRAL 1A BUXUK - SOUTH') %>%
  addPolylines(lng = evral_1b_W$long, lat = evral_1b_W$lat, color = 'black', weight = 2.5, label = 'EVRAL 1B ZARES - WEST') %>% 
  addPolylines(lng = evral_1b_S$long, lat = evral_1b_S$lat, color = 'black', weight = 2.5, label = 'EVRAL 1B BUXUK - SOUTH') %>% 
  addPolylines(lng = evral_1c_W$long, lat = evral_1c_W$lat, color = 'black', weight = 2.5, label = 'EVRAL 1C ZARES - WEST') %>% 
  addPolylines(lng = evral_1c_S$long, lat = evral_1c_S$lat, color = 'black', weight = 2.5, label = 'EVRAL 1C BUXUK - SOUTH') %>% 
  addPolylines(lng = evral_1d_W$long, lat = evral_1d_W$lat, color = 'black', weight = 2.5, label = 'EVRAL 1D ZARES - WEST') %>% 
  addPolylines(lng = evral_1d_S$long, lat = evral_1d_S$lat, color = 'black', weight = 2.5, label = 'EVRAL 1D BUXUK - SOUTH') %>% 
  addPolylines(lng = vunox_1a$long, lat = vunox_1a$lat, color = 'black', weight = 2.5, label = 'VUNOX 1A - NORTH') %>% 
  addPolylines(lng = vunox_1c$long, lat = vunox_1c$lat, color = 'black', weight = 2.5, label = 'VUNOX 1C - NORTH') %>% 
  addPolylines(lng = sanpa_1a$long, lat = sanpa_1a$lat, color = 'black', weight = 2.5, label = 'SANPA 1A - NORTH') %>% 
  addCircles(lng = df_40_edmus$lon,lat = df_40_edmus$lat, weight = 0.0001, color = 'red') %>%
  addCircles(lng = df_40_vunox$lon,lat = df_40_vunox$lat, weight = 0.0001, color = 'green') %>%
  addCircles(lng = df_40_zares$lon,lat = df_40_zares$lat, weight = 0.0001, color = 'blue') %>%
  addCircles(lng = df_40_buxuk$lon,lat = df_40_buxuk$lat, weight = 0.0001, color = 'orange') %>%
  addCircles(lng = df_100_edmus$lon,lat = df_100_edmus$lat, weight = 0.0001, color = 'red') %>%
  addCircles(lng = df_100_vunox$lon,lat = df_100_vunox$lat, weight = 0.0001, color = 'green') %>%
  addCircles(lng = df_100_zares$lon,lat = df_100_zares$lat, weight = 0.0001, color = 'blue') %>%
  addCircles(lng = df_100_buxuk$lon,lat = df_100_buxuk$lat, weight = 0.0001, color = 'orange') %>%
  addCircles(lng = aerodromes[aerodromes$fix == 'EDMUS','lon'],
             lat = aerodromes[aerodromes$fix == 'EDMUS','lat'], 
             weight = 10, 
             color = 'red', 
             label = aerodromes[aerodromes$fix == 'EDMUS','id_icao']) %>%
  addCircles(lng = aerodromes[aerodromes$fix == 'VUNOX','lon'],
             lat = aerodromes[aerdromes$fix == 'VUNOX','lat'], 
             weight = 10, 
             color = 'green', 
             label = aerodromes[aerodromes$fix == 'VUNOX','id_icao']) %>%
  addCircles(lng = aerodromes[aerodromes$fix == 'ZARES','lon'],
             lat = aerodromes[aerdromes$fix == 'ZARES','lat'], 
             weight = 10, 
             color = 'blue', 
             label = aerodromes[aerodromes$fix == 'ZARES','id_icao']) %>%
  addCircles(lng = aerodromes[aerodromes$fix == 'BUXUK','lon'],
             lat = aerodromes[aerdromes$fix == 'BUXUK','lat'], 
             weight = 10, 
             color = 'orange', 
             label = aerodromes[aerodromes$fix == 'BUXUK','id_icao']) %>%
  addCircles(data = data.frame(long = c(-46.47306), lat = c(-23.43556)), weight = 10, color = 'purple', label = 'SBGR')
map
