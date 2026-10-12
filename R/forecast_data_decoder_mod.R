library(RCurl)
library(dplyr)
library(lubridate)

options(error=traceback)

# recognize_METAR = function(field)
# {
#   return(grepl("METAR",field,fixed=T))
# }
# 
# extract_METAR = function(field)
# {
#   return(grepl("METAR",field,fixed=T))
# }
# 
# recognize_SPECI = function(field)
# {
#   return(grepl("SPECI",field,fixed=T))
# }
# 
# extract_SPECI = function(field)
# {
#   return(grepl("SPECI",field,fixed=T))
# }

recognize_TAF = function(field)
{
  return(grepl("TAF",field,fixed=T))
}

extract_TAF = function(field)
{
  return(grepl("TAF",field,fixed=T))
}

recognize_AMD = function(field)
{
  return(grepl("AMD",field,fixed=T))
}

extract_AMD = function(field)
{
  return(grepl("AMD",field,fixed=T))
}

recognize_COR = function(field)
{
  return(grepl("COR",field,fixed=T))
}

extract_COR = function(field)
{
  return(grepl("COR",field,fixed=T))
}

recognize_AUTO = function(field)
{
  return(grepl("AUTO",field,fixed=T))
}

extract_AUTO = function(field)
{
  return(grepl("AUTO",field,fixed=T))
}

recognize_Airport_ICAO_id = function(field)
{
  return(grepl("[A-Z][0-9A-Z][0-9A-Z][0-9A-Z]",field))
}

extract_Airport_ICAO_id = function(field)
{
  return(field)
}

recognize_date = function(field)
{
  return(grepl("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]",field))
}

recognize_timestamp = function(field)
{
  return(grepl("[0-9][0-9][0-9][0-9][0-9][0-9]Z",field))
}

extract_date = function(field)
{
  
  day <- field
  time <- paste0(substring(day,1,4),"-")
  time <- paste0(time,paste0(substring(day,5,6),"-"))
  time <- paste0(time,paste0(substring(day,7,8)," "))
  time <- paste0(time,paste0(substring(day,9,10),":00"))
  time <- ymd_hm(time, tz='UTC')

  return(data.frame(time))
}

extract_timestamp = function(field)
{
  day <- as.integer(substr(field,1,2))
  hour <- as.integer(substr(field,3,4))
  minute <- as.integer(substr(field,5,6))

  n = now('UTC')
  tn = ymd_hm(paste0(as.character(year(n)), '-', as.character(month(n)), '-', as.character(day(n)), ' ',  as.character(hour(n)), ':00'), tz = 'UTC')
  
  time = tn
  i = 0
  while((!((hour(time) == hour) && (day(time) == day))) && i <= 12){
    time = time - hours(1)
    i = i+1
  }
  if(!((hour(time) == hour) && (day(time) == day))){
    time = tn
    i = 0
    while((!((hour(time) == hour) && (day(time) == day))) && i <= 36){
      time = time + hours(1)
      i = i+1
    }
    
    if(!((hour(time) == hour) && (day(time) == day))){
      sprintf('Error: NOAs data is inconsistent')
    }
  }

  if(minute >= 30){
    time = time + hours(1)
  }

  return(data.frame(time))
}


recognize_NIL = function(field)
{
  return(grepl("NIL",field,fixed=T))
}

extract_NIL = function(field)
{
  return(grepl("NIL",field,fixed=T))
}


recognize_forecast_period = function(field)
{
  return(grepl("[0-9][0-9][0-9][0-9]\\/[0-9][0-9][0-9][0-9]",field))
}

extract_forecast_period = function(start_time,field)
{
  day1 = as.numeric(substr(field,1,2))
  hour1 = as.numeric(substr(field,3,4))
  day2 = as.numeric(substr(field,6,7))
  hour2 = as.numeric(substr(field,8,9))
  
  # n = now('UTC')
  # tn = ymd_hm(paste0(as.character(year(n)), '-', as.character(month(n)), '-', as.character(day(n)), ' ',  as.character(hour(n)), ':00'), tz = 'UTC')

  tn = start_time

  t1 = tn
  i = 0
  while((!((hour(t1) == hour1) && (day(t1) == day1))) && i <= 12){
    t1 = t1 - hours(1)
    i = i+1
  }
  if(!((hour(t1) == hour1) && (day(t1) == day1))){
    t1 = tn
    i = 0
    while((!((hour(t1) == hour1) && (day(t1) == day1))) && i <= 36){
      t1 = t1 + hours(1)
      i = i+1
    }
    
    if(!((hour(t1) == hour1) && (day(t1) == day1))){
      sprintf('Error: NOAs data is inconsistent')
    }
  }
  
  bool = F
  if(hour2 == 24){
    bool = T
    hour2 = 0
  }
  t2 = t1
  bool2 = F
  bool3 = F
  while(!(bool3 && bool2)){
    t2 = t2 + hours(1)
    
    if(bool && (day(t2-days(1)) == day2)) {
      bool2 = T
    } else if ((!bool) && (day(t2) == day2)){
      bool2 = T
    } else {
      bool2 = F
    }
    
    bool3 = (hour(t2) == hour2)
  }
  
  time = t1
  i = t1
  while(i != t2){
    i = i + hours(1)
    time = c(time, i)
  }

  return(data.frame(time))
}

recognize_CNL = function(field)
{
  return(grepl("CNL",field,fixed=T))
}

extract_CNL = function(field)
{
  return(grepl("CNL",field,fixed=T))
}


recognize_wind = function(field)
{
  return( grepl("00000KT" ,field,fixed=T) || 
          grepl("00000MPS",field,fixed=T) ||
          grepl("00000",field,fixed=T) ||
            
          grepl("[0-9][0-9][0-9][0-9][0-9]KT" , field) ||
          grepl("[0-9][0-9][0-9][0-9][0-9]MPS", field) || 
          grepl("VRB[0-9][0-9]KT" , field) ||
          grepl("VRB[0-9][0-9]MPS", field) ||
          grepl("VRB[0-9][0-9]G[0-9][0-9]KT" , field) ||
          grepl("VRB[0-9][0-9]G[0-9][0-9]MPS", field) ||            
          grepl("[0-9][0-9][0-9][0-9][0-9]G[0-9][0-9]KT" , field) ||
          grepl("[0-9][0-9][0-9][0-9][0-9]G[0-9][0-9]MPS", field) ||
          
          grepl("[0-9][0-9][0-9]P99KT" , field) ||
          grepl("[0-9][0-9][0-9]P99MPS", field) || 
          grepl("VRBP99KT" , field) ||
          grepl("VRBP99MPS", field) ||
          grepl("VRBP99GP99KT" , field) ||
          grepl("VRBP99GP99MPS", field) ||            
          grepl("[0-9][0-9][0-9]P99GP99KT" , field) ||
          grepl("[0-9][0-9][0-9]P99GP99MPS", field) ||
          grepl("VRB[0-9][0-9]GP99KT" , field) ||
          grepl("VRB[0-9][0-9]GP99MPS", field) ||            
          grepl("[0-9][0-9][0-9][0-9][0-9]GP99KT" , field) ||
          grepl("[0-9][0-9][0-9][0-9][0-9]GP99MPS", field) 
        )
}

extract_wind = function(field)
{
  speed = 0
  direction = as.numeric(NA)
  gust_speed = as.numeric(NA)
  if( grepl("[0-9][0-9][0-9][0-9][0-9]KT" , field) ) {
    res = regexec("([0-9][0-9][0-9])([0-9][0-9])KT", field)
    direction = as.numeric(regmatches(field,res)[[1]][[2]])
    speed = as.numeric(regmatches(field,res)[[1]][[3]])
  } else if( grepl("[0-9][0-9][0-9][0-9][0-9]MPS", field) ) {
    res = regexec("([0-9][0-9][0-9])([0-9][0-9])MPS", field)
    direction = as.numeric(regmatches(field,res)[[1]][[2]])
    speed = 1.943844*as.numeric(regmatches(field,res)[[1]][[3]])
    if(speed > 99){
      speed = 100
    }
  } else if( grepl("VRB[0-9][0-9]KT" , field) ) {
    res = regexec("VRB([0-9][0-9])KT", field)
    speed = as.numeric(regmatches(field,res)[[1]][[2]])
  } else if( grepl("VRB[0-9][0-9]MPS", field) ) {
    res = regexec("VRB([0-9][0-9])MPS", field)
    speed = 1.943844*as.numeric(regmatches(field,res)[[1]][[2]])
    if(speed > 99){
      speed = 100
    }
  } else if ( grepl("VRB[0-9][0-9]G[0-9][0-9]KT",field) ) {
    res = regexec("VRB([0-9][0-9])G([0-9][0-9])KT", field)
    speed = as.numeric(regmatches(field,res)[[1]][[2]])
    gust_speed = as.numeric(regmatches(field,res)[[1]][[3]])
  } else if ( grepl("VRB[0-9][0-9]G[0-9][0-9]MPS",field) ) {
    res = regexec("VRB([0-9][0-9])G([0-9][0-9])MPS", field)
    speed = 1.943844*as.numeric(regmatches(field,res)[[1]][[2]])
    if(speed > 99){
      speed = 100
    }
    gust_speed = 1.943844*as.numeric(regmatches(field,res)[[1]][[3]]) 
    if(gust_speed > 99){
      gust_speed = 100
    }    
  } else if( grepl("[0-9][0-9][0-9][0-9][0-9]G[0-9][0-9]KT" , field) ) {
    res = regexec("([0-9][0-9][0-9])([0-9][0-9])G([0-9][0-9])KT" , field)
    direction = as.numeric(regmatches(field,res)[[1]][[2]])
    speed = as.numeric(regmatches(field,res)[[1]][[3]])
    gust_speed = as.numeric(regmatches(field,res)[[1]][[4]])
  } else if( grepl("[0-9][0-9][0-9][0-9][0-9]G[0-9][0-9]MPS", field) ) {
    res = regexec("([0-9][0-9][0-9])([0-9][0-9])G([0-9][0-9])MPS" , field)
    direction = as.numeric(regmatches(field,res)[[1]][[2]])
    speed = 1.943844*as.numeric(regmatches(field,res)[[1]][[3]])
    if(speed > 99){
      speed = 100
    }
    gust_speed = 1.943844*as.numeric(regmatches(field,res)[[1]][[4]])
    if(gust_speed > 99){
      gust_speed = 100
    }
  } else if( grepl("[0-9][0-9][0-9]P99KT" , field) ) {
    res = regexec("([0-9][0-9][0-9])P99KT", field)
    direction = as.numeric(regmatches(field,res)[[1]][[2]])
    speed = 100
  } else if( grepl("[0-9][0-9][0-9]P99MPS", field) ) {
    res = regexec("([0-9][0-9][0-9])P99MPS", field)
    direction = as.numeric(regmatches(field,res)[[1]][[2]])
    speed = 100
  } else if( grepl("VRBP99KT" , field) ) {
    speed = 100
  } else if( grepl("VRBP99MPS", field) ) {
    speed = 100
  } else if ( grepl("VRBP99GP99KT",field) ) {
    speed = 100
    gust_speed = 100
  } else if ( grepl("VRBP99GP99MPS",field) ) {
    speed = 100
    gust_speed = 100   
  } else if( grepl("[0-9][0-9][0-9]P99GP99KT" , field) ) {
    res = regexec("([0-9][0-9][0-9])P99GP99KT" , field)
    direction = as.numeric(regmatches(field,res)[[1]][[2]])
    speed = 100
    gust_speed = 100   
  } else if( grepl("[0-9][0-9][0-9]P99GP99MPS", field) ) {
    res = regexec("([0-9][0-9][0-9])P99GP99MPS" , field)
    direction = as.numeric(regmatches(field,res)[[1]][[2]])
    speed = 100
    gust_speed = 100   
  } else if ( grepl("VRB[0-9][0-9]GP99KT",field) ) {
    res = regexec("VRB([0-9][0-9])GP99KT", field)
    speed = as.numeric(regmatches(field,res)[[1]][[2]])
    gust_speed = 100
  } else if ( grepl("VRB[0-9][0-9]GP99MPS",field) ) {
    res = regexec("VRB([0-9][0-9])GP99MPS", field)
    speed = 1.943844*as.numeric(regmatches(field,res)[[1]][[2]])
    if(speed > 99){
      speed = 100
    }
    gust_speed = 100     
  } else if( grepl("[0-9][0-9][0-9][0-9][0-9]GP99KT" , field) ) {
    res = regexec("([0-9][0-9][0-9])([0-9][0-9])GP99KT" , field)
    direction = as.numeric(regmatches(field,res)[[1]][[2]])
    speed = as.numeric(regmatches(field,res)[[1]][[3]])
    gust_speed = 100
  } else if( grepl("[0-9][0-9][0-9][0-9][0-9]GP99MPS", field) ) {
    res = regexec("([0-9][0-9][0-9])([0-9][0-9])GP99MPS" , field)
    direction = as.numeric(regmatches(field,res)[[1]][[2]])
    speed = 1.943844*as.numeric(regmatches(field,res)[[1]][[3]])
    if(speed > 99){
      speed = 100
    }
    gust_speed = 100
  }
  return(data.frame(direction,speed,gust_speed))
}

recognize_wind_direction_variation = function(field)
{
  return(grepl("[0-9][0-9][0-9]V[0-9][0-9][0-9]",field))
}

extract_wind_direction_variation = function(field)
{
  res = regexec("([0-9][0-9][0-9])V([0-9][0-9][0-9])",field)
  extreme_wind_direction_n = as.numeric(regmatches(field,res)[[1]][[2]])
  extreme_wind_direction_x = as.numeric(regmatches(field,res)[[1]][[3]])
  return(data.frame(extreme_wind_direction_n,extreme_wind_direction_x))
}

recognize_visibility = function(field)
{
  return(grepl("[0-9][0-9][0-9][0-9]",field)||
           grepl(".*SM",field) ||
           grepl("M.*SM",field) ||
           grepl("////",field) ||
           grepl("CAVOK",field,fixed=T))
}

set_NA_if_empty_string = function(s)
{
  if ( s=="" ) {
    return(NA)
  }else{
    return(s)
  }
}

extract_visibility = function(field)
{
  CAVOK = F
  visibility = as.numeric(NA)
  if ( grepl("[0-9][0-9][0-9][0-9]",field) ) {
    res = regexec("([0-9][0-9][0-9][0-9])",field)
    visibility = 0.000539957*as.numeric(regmatches(field,res)[[1]][[2]])
  } else if ( grepl("M(.*)SM",field)) {
    res = regexec("M(.*)SM",field)
    visibility = 0
  } else if (grepl(".*SM",field)) {
    res = regexec("(.*)SM",field)
    visibility = 0.000539957*1609.344*as.numeric(eval(parse(text=regmatches(field,res)[[1]][[2]])))
  } else if ( grepl("CAVOK",field,fixed=T)){
    visibility = 0.000539957*9999
    CAVOK = T
  }
  return(data.frame(visibility, CAVOK))
}

recognize_minimum_visibility = function(field)
{
  return(grepl("[0-9][0-9][0-9][0-9](NE|NW|N|SE|SW|S|E|W)?",field))
}

extract_minimum_visibility = function(field)
{
  res = regexec("([0-9][0-9][0-9][0-9])(NE|NW|N|SE|SW|S|E|W)?",field)
  minimum_visibility = 0.000539957*as.numeric(regmatches(field,res)[[1]][[2]])
  mv_direction = set_NA_if_empty_string(regmatches(field,res)[[1]][[3]])
  return(data.frame(minimum_visibility, mv_direction))
}

recognize_runway_visual_range = function(field)
{
  return(grepl("R[0-9][0-9](L|C|R)?\\/(M|P)?[0-9][0-9][0-9][0-9](U|D|N)?",field))
}

extract_runway_visual_range = function(field)
{
  runway = NA
  runway_visual_range = NA
  tendency = NA
  extreme_value = NA
  
  res = regexec("R([0-9][0-9])(L|C|R)?\\/(M|P)?([0-9][0-9][0-9][0-9])(U|D|N)?",field)
  runway = paste0(regmatches(field,res)[[1]][[2]],regmatches(field,res)[[1]][[3]])
  extreme_value = set_NA_if_empty_string(regmatches(field,res)[[1]][[4]])
  runway_visual_range = as.numeric(regmatches(field,res)[[1]][[5]])
  tendency = set_NA_if_empty_string(regmatches(field,res)[[1]][[6]])
  
  return(data.frame(runway,extreme_value,runway_visual_range,tendency))
}

recognize_weather = function(field)
{
  if(grepl("NSW",field)){
    found = T
  } else if(grepl("//",field)){
    found = T
  } else {
    res = regexec("(\\-|\\+|VC)?(MI|BC|PR|DR|BL|SH|TS|FZ)?(.*)",field)
    supposed_phenomena = regmatches(field,res)[[1]][[4]]
    found = T
    if ( nchar(supposed_phenomena) != 0 && nchar(supposed_phenomena)%%2 == 0 ){
      start = 1
      for ( i in 1:(nchar(supposed_phenomena)/2) ) {
        found = found && (substr(supposed_phenomena,start,start+1) %in% c("DZ","RA","SN","SG","IC","PL","GR","GS","UP","BR","FG","FU","VA","DU","SA","HZ","PY","PO","SQ","FC","SS","DS") )
        start = start + 2
      }
    } else if ( supposed_phenomena=="" ) {
      found = T
    } else {
      found = F
    }
  }
  return(found)
}

extract_weather = function(field)
{
  if(grepl("NSW",field)){
    intensity = ''
    descriptor = ''
    phenomena = ''
    TSorRA = F
  } else if(grepl("//",field)){
    intensity = ''
    descriptor = ''
    phenomena = ''
    TSorRA = as.logical(NA)
  } else {
    # Description abreviation
    D_a = c('MI','BC','PR','DR','BL','SH','TS','FZ')
    # Description legend
    D_l = c('Shallow','Patches of','Partial','Low Drifting','Blowing','Shower','Thunderstorm','Freezing')
    # Phenomena abreviation
    P_a = c("DZ","RA","SN","SG","PL","GR","GS","BR","FG","FU","VA","DU","SA","HZ","PO","SQ","FC","SS","DS")
    # Phenomena legend
    P_l = c("Drizzle","Rain","Snow","Snow Grains","Ice Pellets","Hail","Small Hail and/or Snow Pellets","Mist","Fog","Smoke","Volcanic Ash","Dust (Widespread)","Sand","Haze","Dust/Sand Whirls (Dust Devils)","Squall","Funnel Cloud (Tornado or Waterspout)","Sandstorm","Duststorm")
    
    res = regexec("(\\-|\\+|VC)?(MI|BC|PR|DR|BL|SH|TS|FZ)?(.*)",field)
    i = set_NA_if_empty_string(regmatches(field,res)[[1]][[2]])
  
    d = set_NA_if_empty_string(regmatches(field,res)[[1]][[3]])
    TSorRA = (d %in% c("TS"))
  
    p = regmatches(field,res)[[1]][[4]]
    if (nchar(p) != 0 && nchar(p)%%2 == 0){
      start = 1
      for (j in 1:(nchar(p)/2)) {
        TSorRA = TSorRA || (substr(p,start,start+1) %in% c("RA") )
        start = start + 2
      }
    }
    p = set_NA_if_empty_string(p)
  
  
    if (is.na(i) && (!(is.na(d) && is.na(p)))) {
      intensity ='Moderate'    
    } else if (i == '+'){
      intensity = 'Heavy'  
    } else if (i == '-'){
      intensity = 'Light'  
    } else if (i == 'VC'){
      intensity = 'Vicinity'
    } else{
      intensity = ''
    }
  
    if (is.na(d)) {
      descriptor = ''
    } else{
      descriptor = D_l[which(D_a == d)]
    }
  
    if (is.na(p)) {
      phenomena = ''
    } else {
      phenomena = P_l[which(P_a == substr(p,1,2))]
    
      if (nchar(p) > 2){
        start = 3
        for (j in 2:(nchar(p)/2)) {
          phenomena = paste(phenomena,"and",P_l[which(P_a == substr(p,start,start+1))])
          start = start + 2
        }
      }
    }
  }
  if (intensity == '' && descriptor == '' && phenomena == ''){
    weather = as.character(NA)
  } else if (descriptor == ''){
    weather = paste(intensity,phenomena)
  } else if (phenomena == ''){
    weather = paste(intensity,descriptor)
  } else{
    weather = paste(intensity,descriptor,phenomena)
  }
  
  return(data.frame(weather, TSorRA,stringsAsFactors=FALSE))
}

recognize_clouds = function(field)
{
  return(grepl("(FEW|SCT|BKN|OVC)([0-9][0-9][0-9])(CB|TCU|///)?",field) ||
         grepl("NSC",field) ||
         grepl("VV([0-9][0-9][0-9])",field) ||
           grepl("VV///",field) ||
           grepl("//////",field) )
}

extract_clouds = function(field)
{
  CUorTCU = F
  if ( grepl("(FEW|SCT|BKN|OVC)([0-9][0-9][0-9])(CB|TCU)",field) ) {
    CUorTCU = T
  }
  
  ceiling = 99999
  vertical_visibility = as.numeric(NA)
  cloud_abbreviation = as.character(NA)
  if ( grepl("(FEW|SCT)([0-9][0-9][0-9])(CB|TCU)?",field) ) {
    res = regexec("(FEW|SCT)([0-9][0-9][0-9])(CB|TCU)?",field)
    cloud_abbreviation = regmatches(field,res)[[1]][[2]]
  } else if ( grepl("(FEW|SCT)([0-9][0-9][0-9])(///)",field) ) {
    res = regexec("(FEW|SCT)([0-9][0-9][0-9])(///)",field)
    cloud_abbreviation = regmatches(field,res)[[1]][[2]]
    CUorTCU = as.logical(NA)
  } else if ( grepl("(BKN|OVC)([0-9][0-9][0-9])(CB|TCU|///)?",field) ) {
    res = regexec("(BKN|OVC)([0-9][0-9][0-9])(CB|TCU|///)?",field)
    cloud_abbreviation = regmatches(field,res)[[1]][[2]]
    ceiling = 100*as.numeric(regmatches(field,res)[[1]][[3]])
  } else if ( grepl("(BKN|OVC)([0-9][0-9][0-9])(///)",field) ) {
    res = regexec("(BKN|OVC)([0-9][0-9][0-9])(///)",field)
    cloud_abbreviation = regmatches(field,res)[[1]][[2]]
    ceiling = 100*as.numeric(regmatches(field,res)[[1]][[3]])
    CUorTCU = as.logical(NA)
  } else if ( grepl("NSC",field) ) {
    cloud_abbreviation = "NSC"
    vertical_visibility = 9999
  } else if (grepl("VV([0-9][0-9][0-9])",field) ) {
    res = regexec("VV([0-9][0-9][0-9])",field)
    ceiling = as.numeric(NA)
    vertical_visibility = 100*as.numeric(regmatches(field,res)[[1]][[2]])
    CUorTCU = as.logical(NA)
  } else {
    ceiling = as.numeric(NA)
    CUorTCU = as.logical(NA)
  }
  
  return(data.frame(cloud_abbreviation,ceiling,vertical_visibility,CUorTCU,stringsAsFactors=FALSE))
}

recognize_temperature = function(field)
{
  return(grepl("(M)?([0-9][0-9])/(M)?([0-9][0-9])",field) ||
           grepl("(M)?([0-9][0-9])/",field) )
}

extract_temperature = function(field)
{
  temperature = NA
  temperature_dew = NA
  if ( grepl("(M)?([0-9][0-9])/(M)?([0-9][0-9])",field) ){
    sign = +1
    res = regexec("(M)?([0-9][0-9])/(M)?([0-9][0-9])",field)
    if ( "M" == regmatches(field,res)[[1]][[2]] ) {
      sign = -1
    }
    temperature = sign*as.numeric(regmatches(field,res)[[1]][[3]])
    sign = +1
    if ( "M" == regmatches(field,res)[[1]][[4]] ) {
      sign = -1
    } 
    temperature_dew = sign*as.numeric(regmatches(field,res)[[1]][[5]])    
  } else if ( grepl("(M)?([0-9][0-9])/",field) ) {
    sign = +1
    res = regexec("(M)?([0-9][0-9])/",field)
    if ( "M" == regmatches(field,res)[[1]][[2]] ) {
      sign = -1
    }
    temperature = sign*as.numeric(regmatches(field,res)[[1]][[3]])    
  }
  
  return(data.frame(temperature,temperature_dew))
}

recognize_altimeter = function(field)
{
  return( grepl("Q([0-9][0-9][0-9][0-9])",field) )
}

extract_altimeter = function(field)
{
  res = regexec("Q([0-9][0-9][0-9][0-9])",field)
  pressure = as.numeric(regmatches(field,res)[[1]][[2]])
  # UOM is hectorpascals
  return(data.frame(pressure))
}

recognize_temperature_max = function(field)
{
  return(grepl("TX(M)?([0-9][0-9])\\/([0-9][0-9])([0-9][0-9])Z",field))
}

extract_temperature_max = function(field)
{
  temperature_max = NA
  sign = +1
  res = regexec("TX(M)?([0-9][0-9])\\/([0-9][0-9])([0-9][0-9])Z",field)
  if ( "M" == regmatches(field,res)[[1]][[2]] ) {
    sign = -1
  }
  temperature_max = sign*as.numeric(regmatches(field,res)[[1]][[3]])
  hour_tx = as.numeric(regmatches(field,res)[[1]][[4]])
  minute_tx = as.numeric(regmatches(field,res)[[1]][[5]])    

  return(data.frame(temperature_max,hour_tx,minute_tx))
}

recognize_temperature_min = function(field)
{
  return(grepl("TN(M)?([0-9][0-9])\\/([0-9][0-9])([0-9][0-9])Z",field))
}

extract_temperature_min = function(field)
{
  temperature_min = NA
  sign = +1
  res = regexec("TN(M)?([0-9][0-9])\\/([0-9][0-9])([0-9][0-9])Z",field)
  if ( "M" == regmatches(field,res)[[1]][[2]] ) {
    sign = -1
  }
  temperature_min = sign*as.numeric(regmatches(field,res)[[1]][[3]])
  hour_tn = as.numeric(regmatches(field,res)[[1]][[4]])
  minute_tn = as.numeric(regmatches(field,res)[[1]][[5]])    
  
  return(data.frame(temperature_min,hour_tn,minute_tn))
}

recognize_change_id = function(field)
{
  return(grepl("PROB30",field) ||
           grepl("PROB40",field) ||
           grepl("TEMPO",field) ||
           grepl("BECMG",field) ||
           grepl("FM[0-9][0-9][0-9][0-9][0-9][0-9]",field))
}

extract_change_id = function(start_time,field)
{
  if(grepl("FM[0-9][0-9][0-9][0-9][0-9][0-9]",field)){
    change_indicator = "FM"
    case = 1
    day <- as.integer(substr(field,3,4))
    hour <- as.integer(substr(field,5,6))
    minute <- as.integer(substr(field,7,8))
    
    # n = now('UTC')
    # tn = ymd_hm(paste0(as.character(year(n)), '-', as.character(month(n)), '-', as.character(day(n)), ' ',  as.character(hour(n)), ':00'), tz = 'UTC')
    tn = start_time
    t = tn
    i = 0
    while((!((hour(t) == hour) && (day(t) == day))) && i <= 12){
      t = t - hours(1)
      i = i+1
    }
    if(!((hour(t) == hour) && (day(t) == day))){
      t = tn
      i = 0
      while((!((hour(t) == hour) && (day(t) == day))) && i <= 36){
        t = t + hours(1)
        i = i+1
      }
      if(!((hour(t) == hour) && (day(t) == day))){
        print('Error: NOAs data is inconsistent')
      }
    }
    if(minute >= 30){
      t = t +hours(1)
    }
    time_fm = t
    return(data.frame(change_indicator,case,time_fm,stringsAsFactors=FALSE))
  } else {
    change_indicator = field
    if(grepl("PROB30",field)){
      case = 2
    } else if(grepl("PROB40",field)){
      case = 3
    } else if(grepl("TEMPO",field)){
      case = 4
    } else if(grepl("BECMG",field)){
      case = 5
    }
    return(data.frame(change_indicator,case,stringsAsFactors=FALSE))
  }
}

parse_field = function(groups,index,recognizer,extractor,is_compulsory,field_description)
{
  data = NA
  found_optional_field = F
  index = index[1]
  if ( index <= length(groups) ) {
    field = groups[index]
    if ( recognizer(field) ) {
      data = extractor(field)
      index = index + 1
      found_optional_field = T
    } else {
      if ( is_compulsory ) {
        stop(sprintf("Expected compulsory field '%s', found '%s'.",field_description,field))
      }  
    }
  }
  return(data.frame(data,index,found_optional_field,stringsAsFactors=FALSE))
}


parse_field_2 = function(groups,index,start_time,recognizer,extractor,is_compulsory,field_description)
{
  data = NA
  found_optional_field = F
  index = index[1]
  if ( index <= length(groups) ) {
    field = groups[index]
    if ( recognizer(field) ) {
      data = extractor(start_time,field)
      index = index + 1
      found_optional_field = T
    } else {
      if ( is_compulsory ) {
        stop(sprintf("Expected compulsory field '%s', found '%s'.",field_description,field))
      }  
    }
  }
  return(data.frame(data,index,found_optional_field,stringsAsFactors=FALSE))
}

fix_white_space_in_FMH_visibility = function(taf_string)
{
  res = regexec("\\d(\\s+)\\d\\/\\dSM",taf_string)
  if ( res[[1]][[1]] != -1 ) {
    white_space_position = res[[1]][[2]]
    substr(taf_string,white_space_position,white_space_position) = '+'
  }
  return(taf_string)
}

metar_decoder = function(metar_string)
{
  time = ymd_hm(NA)
  
  probability = as.character(NA)
  
  wind_direction = as.numeric(NA)
  wind_speed = as.numeric(NA)
  gust_speed = as.numeric(NA)
  
  visibility = as.numeric(NA)
  
  ceiling = as.numeric(NA)
  cumulus_nimbus = as.logical(NA)
  
  weather1 = as.character(NA)
  weather2 = as.character(NA)
  weather3 = as.character(NA)
  rain_or_thunderstorm = as.logical(NA)
  
  temperature = as.numeric(NA)
  temperature_max = as.numeric(NA)
  temperature_min = as.numeric(NA)
  temperature_dew = as.numeric(NA)
  
  pressure = as.numeric(NA)
  
  observation = data.frame(time,probability,wind_direction,wind_speed,gust_speed,visibility,ceiling,cumulus_nimbus,weather1,weather2,weather3,rain_or_thunderstorm,temperature,temperature_min,temperature_max,temperature_dew,pressure,stringsAsFactors=FALSE)
  
  metar_string = fix_white_space_in_FMH_visibility(metar_string)
  groups = scan(what=character(),text=metar_string)
  
  df = parse_field(groups,1,recognize_METAR,extract_METAR,F, "METAR")
  df = parse_field(groups,df$index,recognize_SPECI,extract_SPECI,F, "SPECI")
  df = parse_field(groups,df$index,recognize_COR,extract_COR,F, "COR")
  
  #df = parse_field(groups,df$index,recognize_,extract_,F, "")
  
  df = parse_field(groups,df$index,recognize_Airport_ICAO_id,extract_Airport_ICAO_id,T, "Airport ICAO id")
  
  # df = parse_field(groups,df$index,recognize_date,extract_date,T, "Time Stamp")
  df = parse_field(groups,df$index,recognize_timestamp,extract_timestamp,T, "Time Stamp")
  observation$time = df$time
  
  df = parse_field(groups,df$index,recognize_NIL,extract_NIL,F, "NIL")
  
  if ( !df$found_optional_field ) {
    df = parse_field(groups,df$index,recognize_COR,extract_COR,F, "COR")
    df = parse_field(groups,df$index,recognize_AUTO,extract_AUTO,F, "AUTO")
    df = parse_field(groups,df$index,recognize_COR,extract_COR,F, "COR")
    
    observation$probability = 'Equal to 100% (Observation)'
    
    df = parse_field(groups,df$index,recognize_wind,extract_wind,T, "Wind")
    observation$wind_direction = df$direction
    observation$wind_speed = df$speed
    observation$gust_speed = df$gust_speed
    
    df = parse_field(groups,df$index,recognize_wind_direction_variation,extract_wind_direction_variation,F, "Wind Direction Variation")
    
    df = parse_field(groups,df$index,recognize_visibility,extract_visibility,T,"Visibility")
    observation$visibility = df$visibility
    if(df$CAVOK){
      observation$ceiling = 99999
      observation$cumulus_nimbus = F
      observation$rain_or_thunderstorm = F
    } else{
      df = parse_field(groups,df$index,recognize_minimum_visibility,extract_minimum_visibility,F, "Minimum Visibility")
      
      df = parse_field(groups,df$index,recognize_runway_visual_range,extract_runway_visual_range,F, "Runway Visual Range")
      df = parse_field(groups,df$index,recognize_runway_visual_range,extract_runway_visual_range,F, "Runway Visual Range")
      df = parse_field(groups,df$index,recognize_runway_visual_range,extract_runway_visual_range,F, "Runway Visual Range")
      df = parse_field(groups,df$index,recognize_runway_visual_range,extract_runway_visual_range,F, "Runway Visual Range")
      
      observation$rain_or_thunderstorm = F
      df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 1")
      if ( df$found_optional_field ) {
        observation$weather1 = df$weather
        TSorRA= df$TSorRA
        
        df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 2")
        if ( df$found_optional_field ) {
          observation$weather2 = df$weather
          TSorRA = c(TSorRA, df$TSorRA)
          
          df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 3")
          if ( df$found_optional_field ) {
            observation$weather3 = df$weather
            TSorRA = c(TSorRA, df$TSorRA)
          }
        }
        
        TSorRA = TSorRA[which(!is.na(TSorRA))]
        if(length(TSorRA) == 0) {
          observation$rain_or_thunderstorm = as.logical(NA)
        } else {
          for (j in 1:(length(TSorRA))) {
            observation$rain_or_thunderstorm = observation$rain_or_thunderstorm || TSorRA[j]
          }
        }
      }
      
      observation$cumulus_nimbus = F
      df= parse_field(groups,df$index,recognize_clouds,extract_clouds,T,"Clouds 1")
      ceiling = df$ceiling
      CUorTCU = df$CUorTCU
      
      df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 2")
      if ( df$found_optional_field ) {
        ceiling = c(ceiling, df$ceiling)
        CUorTCU = c(CUorTCU, df$CUorTCU)
        
        df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 3")
        if ( df$found_optional_field ) {
          ceiling = c(ceiling, df$ceiling)
          CUorTCU = c(CUorTCU, df$CUorTCU)
          
          df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 4")
          if ( df$found_optional_field ) {
            ceiling = c(ceiling, df$ceiling)
            CUorTCU = c(CUorTCU, df$CUorTCU)
          }
        }
      }
      
      ceiling = ceiling[which(!is.na(ceiling))]
      if(length(ceiling) == 0){
        observation$ceiling = as.numeric(NA)
      } else {
        observation$ceiling = min(ceiling)
      }
      CUorTCU = CUorTCU[which(!is.na(CUorTCU))]
      if(length(CUorTCU) == 0) {
        observation$cumulus_nimbus = as.logical(NA)
      } else {
        observation$cumulus_nimbus = F
        for (j in 1:(length(CUorTCU))) {
          observation$cumulus_nimbus = observation$cumulus_nimbus || CUorTCU[j]
        }
      }
    }
    
    
    df= parse_field(groups,df$index,recognize_temperature,extract_temperature,T,"Temperature and Dew Temperature")
    observation$temperature = df$temperature
    observation$temperature_dew = df$temperature_dew
    
    df= parse_field(groups,df$index,recognize_altimeter,extract_altimeter,T,"Pressure")
    observation$pressure = df$pressure
  }
  
  return(observation)
}

taf_decoder1 = function(taf_string)
{
  time = ymd_hm(NA)
  
  probability = as.character(NA)
  
  wind_direction = as.numeric(NA)
  wind_speed = as.numeric(NA)
  gust_speed = as.numeric(NA)
  
  visibility = as.numeric(NA)
  
  ceiling = as.numeric(NA)
  cumulus_nimbus = as.logical(NA)
  
  weather1 = as.character(NA)
  weather2 = as.character(NA)
  weather3 = as.character(NA)
  rain_or_thunderstorm = as.logical(NA)
  
  temperature = as.numeric(NA)
  temperature_max = as.numeric(NA)
  temperature_min = as.numeric(NA)
  temperature_dew = as.numeric(NA)
  
  pressure = as.numeric(NA)
  
  misplaced_line = F
  misplaced_line_string = ""
  
  taf_string = fix_white_space_in_FMH_visibility(taf_string)
  groups = scan(what=character(),text=taf_string)
  
  df = parse_field(groups,1,recognize_TAF,extract_TAF,F, "TAF")
  df = parse_field(groups,df$index,recognize_TAF,extract_TAF,F, "TAF")
  
  df = parse_field(groups,df$index,recognize_AMD,extract_AMD,F, "AMD")
  
  df = parse_field(groups,df$index,recognize_COR,extract_COR,F, "COR")
  
  df = parse_field(groups,df$index,recognize_Airport_ICAO_id,extract_Airport_ICAO_id,T, "Airport ICAO id")
  
  df = parse_field(groups,df$index,recognize_date,extract_date,T, "Time Stamp")
  
  start_time = df$time
  
  #df = parse_field(groups,df$index,recognize_timestamp,extract_timestamp,T, "Time Stamp")
  
  df = parse_field(groups,df$index,recognize_NIL,extract_NIL,F, "NIL")
  
  if ( df$found_optional_field ) {
    forecast = NA
  } else {
    df = parse_field_2(groups,df$index,start_time,recognize_forecast_period,extract_forecast_period,T, "Forecast Period")
    time = df$time
    
    forecast = data.frame(time,probability,wind_direction,wind_speed,gust_speed,visibility,ceiling,cumulus_nimbus,weather1,weather2,weather3,rain_or_thunderstorm,temperature,temperature_min,temperature_max,temperature_dew,pressure,stringsAsFactors=FALSE)
    
    df = parse_field(groups,df$index,recognize_CNL,extract_CNL,F, "CNL")
    if ( !(df$found_optional_field) ) {
      forecast$probability = 'Equal to or greater than 50%'
      
      df = parse_field(groups,df$index,recognize_wind,extract_wind,T, "Wind")
      forecast$wind_direction = df$direction
      forecast$wind_speed = df$speed
      forecast$gust_speed = df$gust_speed
      
      df = parse_field(groups,df$index,recognize_visibility,extract_visibility,T,"Visibility")
      forecast$visibility = df$visibility
      if(df$CAVOK){
        forecast$ceiling = 99999
        forecast$cumulus_nimbus = F
        forecast$rain_or_thunderstorm = F
      } else{
        forecast$rain_or_thunderstorm = F
        df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 1")
        if ( df$found_optional_field ) {
          forecast$weather1 = df$weather
          TSorRA= df$TSorRA
          
          df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 2")
          if ( df$found_optional_field ) {
            forecast$weather2 = df$weather
            TSorRA = c(TSorRA, df$TSorRA)
            
            df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 3")
            if ( df$found_optional_field ) {
              forecast$weather3 = df$weather
              TSorRA = c(TSorRA, df$TSorRA)
            }
          }
          
          TSorRA = TSorRA[which(!is.na(TSorRA))]
          if(length(TSorRA) == 0) {
            forecast$rain_or_thunderstorm = as.logical(NA)
          } else {
            # for (j in 1:(length(TSorRA))) {
            #   forecast$rain_or_thunderstorm = forecast$rain_or_thunderstorm || TSorRA[j]
            # }
          }
        }
        
        df= parse_field(groups,df$index,recognize_clouds,extract_clouds,T,"Clouds 1")
        ceiling = df$ceiling
        CUorTCU = df$CUorTCU
        
        df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 2")
        if ( df$found_optional_field ) {
          ceiling = c(ceiling, df$ceiling)
          CUorTCU = c(CUorTCU, df$CUorTCU)
          
          df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 3")
          if ( df$found_optional_field ) {
            ceiling = c(ceiling, df$ceiling)
            CUorTCU = c(CUorTCU, df$CUorTCU)
            
            df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 4")
            if ( df$found_optional_field ) {
              ceiling = c(ceiling, df$ceiling)
              CUorTCU = c(CUorTCU, df$CUorTCU)
            }
          }
        }
        
        ceiling = ceiling[which(!is.na(ceiling))]
        if(length(ceiling) == 0){
          forecast$ceiling = as.numeric(NA)
        } else {
          forecast$ceiling = min(ceiling)
        }
        CUorTCU = CUorTCU[which(!is.na(CUorTCU))]
        if(length(CUorTCU) == 0) {
          forecast$cumulus_nimbus = as.logical(NA)
        } else {
          forecast$cumulus_nimbus = F
          # for (j in 1:(length(CUorTCU))) {
          #   forecast$cumulus_nimbus = forecast$cumulus_nimbus || CUorTCU[j]
          # }
        }
      }
      
      
      df= parse_field(groups,df$index,recognize_temperature_max,extract_temperature_max,F,"Maximum Temperature")
      if ( df$found_optional_field ) {
        forecast$temperature_max = df$temperature_max
      }
      
      df= parse_field(groups,df$index,recognize_temperature_min,extract_temperature_min,F,"Minumum Temperature")
      if ( df$found_optional_field ) {
        forecast$temperature_min = df$temperature_min
      }
      
      df= parse_field(groups,df$index,recognize_temperature_max,extract_temperature_max,F,"Maximum Temperature")
      if ( df$found_optional_field ) {
        forecast$temperature_max = df$temperature_max
      }
      
      df = parse_field_2(groups,df$index,start_time,recognize_change_id,extract_change_id,F, "Change Identificator")
      if ( df$found_optional_field ) {
        misplaced_line = T
        misplaced_line_string = groups[(df$index-1)]
        if (df$index <= length(groups)){
          for (i in (df$index):(length(groups))) {
            misplaced_line_string = paste(misplaced_line_string, groups[i], sep = ' ')
          }
        }
      }
      
    }
  }
  
  return(list(forecast, misplaced_line, misplaced_line_string, start_time))
}

taf_decoder2 = function(taf_string, f, only_prob, start_time)
{
  misplaced_line = F
  misplaced_line_string = ""
  if(!is.na(only_prob)){
    taf_string = paste0(only_prob, " ", taf_string)
  }
  only_prob = as.character(NA)
  taf_string = fix_white_space_in_FMH_visibility(taf_string)
  groups = scan(what=character(),text=taf_string)
  
  df = parse_field_2(groups,1,start_time,recognize_change_id,extract_change_id,F, "Change Identificator")
  if (length(groups) == 1 && df$found_optional_field){
    forecast = f
    only_prob = groups[1]
  } else if(df$found_optional_field){
    case = df$case
    if (df$case == 1){
      t = df$time_fm
      time = f$time[which(f$time >= t)]
    } else if (df$case == 2 || df$case == 3){
      df = parse_field_2(groups,df$index,start_time,recognize_change_id,extract_change_id,F, "Change Identificator")
      
      df = parse_field_2(groups,df$index,start_time,recognize_forecast_period,extract_forecast_period,T, "Change Period of Case PROBXX TEMPO")
      time = df$time
    } else if (df$case == 4){
      df = parse_field_2(groups,df$index,start_time,recognize_forecast_period,extract_forecast_period,T, "Change Period of Case TEMPO")
      time = df$time
    } else if (df$case == 5){
      df = parse_field_2(groups,df$index,start_time,recognize_forecast_period,extract_forecast_period,T, "Change Period of Case BECMG")
      t = max(df$time)
      time = f$time[which(f$time >= t)]    
    }
    
    forecast = f[which(f$time == time[1]),]
    for (i in 2:(length(time))) {
      forecast = rbind(forecast, f[which(f$time == time[i]),])
    }
    if(case == 2){
      forecast$probability = 'Equal to or greater than 30%, but less than 40%'
    } else if(case == 3){
      forecast$probability = 'Equal to or greater than 40%, but less than 50%'
    }
    
    
    
    df = parse_field(groups,df$index,recognize_wind,extract_wind,F, "Wind")
    if ( df$found_optional_field ) {
      forecast$wind_direction = df$direction
      forecast$wind_speed = df$speed
      forecast$gust_speed = df$gust_speed
    }
    
    df = parse_field(groups,df$index,recognize_visibility,extract_visibility,F,"Visibility")
    if ( df$found_optional_field ) {
      forecast$visibility = df$visibility
      if(df$CAVOK){
        forecast$ceiling = 99999
        forecast$cumulus_nimbus = F
        forecast$weather1 = as.character(NA)
        forecast$weather2 = as.character(NA)
        forecast$weather3 = as.character(NA)
        forecast$rain_or_thunderstorm = F
      } else{
        df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 1")
        if ( df$found_optional_field ) {
          forecast$weather1 = df$weather
          forecast$weather2 = as.character(NA)
          forecast$weather3 = as.character(NA)
          
          TSorRA= df$TSorRA
          
          df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 2")
          if ( df$found_optional_field ) {
            forecast$weather2 = df$weather
            TSorRA = c(TSorRA,df$TSorRA)
            
            df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 3")
            if ( df$found_optional_field ) {
              forecast$weather3 = df$weather
              TSorRA = c(TSorRA,df$TSorRA)
            }
          }
          
          TSorRA = TSorRA[which(!is.na(TSorRA))]
          if(length(TSorRA) == 0) {
            forecast$rain_or_thunderstorm = as.logical(NA)
          } else {
            forecast$rain_or_thunderstorm = F
            # for (j in 1:(length(TSorRA))) {
            #   forecast$rain_or_thunderstorm = forecast$rain_or_thunderstorm || TSorRA[j]
            # }
          }
        }
        
        df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 1")
        if ( df$found_optional_field ) {
          ceiling = df$ceiling
          CUorTCU = df$CUorTCU
          
          df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 2")
          if ( df$found_optional_field ) {
            ceiling = c(ceiling, df$ceiling)
            CUorTCU = c(CUorTCU, df$CUorTCU)
            
            df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 3")
            if ( df$found_optional_field ) {
              ceiling = c(ceiling, df$ceiling)
              CUorTCU = c(CUorTCU, df$CUorTCU)
              
              df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 4")
              if ( df$found_optional_field ) {
                ceiling = c(ceiling, df$ceiling)
                CUorTCU = c(CUorTCU, df$CUorTCU)
              }
            }
          }
          
          ceiling = ceiling[which(!is.na(ceiling))]
          if(length(ceiling) == 0){
            forecast$ceiling = as.numeric(NA)
          } else {
            forecast$ceiling = min(ceiling)
          }
          CUorTCU = CUorTCU[which(!is.na(CUorTCU))]
          if(length(CUorTCU) == 0) {
            forecast$cumulus_nimbus = as.logical(NA)
          } else {
            forecast$cumulus_nimbus = F
            # for (j in 1:(length(CUorTCU))) {
            #   forecast$cumulus_nimbus = forecast$cumulus_nimbus || CUorTCU[j]
            # }
          }
        }
        
      }
    } else {
      df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 1")
      if ( df$found_optional_field ) {
        for (i in 1:(nrow(forecast))) {
          forecast$weather1[i] = df$weather
          forecast$weather2[i] = as.character(NA)
          forecast$weather3[i] = as.character(NA)
          
        }
        forecast$rain_or_thunderstorm = df$TSorRA
        
        df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 2")
        if ( df$found_optional_field ) {
          for (i in 1:(nrow(forecast))) {
            forecast$weather2[i] = df$weather
          }
          forecast$rain_or_thunderstorm = forecast$rain_or_thunderstorm || df$TSorRA
          
          df= parse_field(groups,df$index,recognize_weather,extract_weather,F,"Weather 3")
          if ( df$found_optional_field ) {
            for (i in 1:(nrow(forecast))) {
            forecast$weather3[i] = df$weather
            }
            forecast$rain_or_thunderstorm = forecast$rain_or_thunderstorm || df$TSorRA
          }
        }
      }
      
      df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 1")
      if ( df$found_optional_field ) {
        ceiling = df$ceiling
        CUorTCU = df$CUorTCU
        
        df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 2")
        if ( df$found_optional_field ) {
          ceiling = c(ceiling, df$ceiling)
          CUorTCU = c(CUorTCU, df$CUorTCU)
          
          df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 3")
          if ( df$found_optional_field ) {
            ceiling = c(ceiling, df$ceiling)
            CUorTCU = c(CUorTCU, df$CUorTCU)
            
            df= parse_field(groups,df$index,recognize_clouds,extract_clouds,F,"Clouds 4")
            if ( df$found_optional_field ) {
              ceiling = c(ceiling, df$ceiling)
              CUorTCU = c(CUorTCU, df$CUorTCU)
            }
          }
        }
        
        ceiling = ceiling[which(!is.na(ceiling))]
        if(length(ceiling) == 0){
          forecast$ceiling = as.numeric(NA)
        } else {
          forecast$ceiling = min(ceiling)
        }
        CUorTCU = CUorTCU[which(!is.na(CUorTCU))]
        if(length(CUorTCU) == 0) {
          forecast$cumulus_nimbus = as.logical(NA)
        } else {
          forecast$cumulus_nimbus = F
          # for (j in 1:(length(CUorTCU))) {
          #   forecast$cumulus_nimbus = forecast$cumulus_nimbus || CUorTCU[j]
          # }
        }
      }
    }
    f2 = forecast
    forecast = f
    for (i in 1:nrow(forecast)) {
      for (j in 1:nrow(f2)) {
        if(forecast$time[i] == f2$time[j]){
          forecast[i,] = f2[j,]
        }
      }
    }
    
    df = parse_field_2(groups,df$index,start_time,recognize_change_id,extract_change_id,F, "Change Identificator")
    if ( df$found_optional_field ) {
      misplaced_line = T
      misplaced_line_string = groups[(df$index-1)]
      if (df$index <= length(groups)){
        for (i in (df$index):(length(groups))) {
          misplaced_line_string = paste(misplaced_line_string, groups[i], sep = ' ')
        }
      }
    }
  } else {
    forecast = f
  }
  
  return(list(forecast, misplaced_line, misplaced_line_string, only_prob))
}

null_data = function(){
  time = ymd_hm(NA)
  
  probability = as.character(NA)
  
  wind_direction = as.numeric(NA)
  wind_speed = as.numeric(NA)
  gust_speed = as.numeric(NA)
  
  visibility = as.numeric(NA)
  
  ceiling = as.numeric(NA)
  cumulus_nimbus = as.logical(NA)
  
  weather1 = as.character(NA)
  weather2 = as.character(NA)
  weather3 = as.character(NA)
  rain_or_thunderstorm = as.logical(NA)
  
  temperature = as.numeric(NA)
  temperature_max = as.numeric(NA)
  temperature_min = as.numeric(NA)
  temperature_dew = as.numeric(NA)
  
  pressure = as.numeric(NA)
  
  aux = data.frame(time,probability,wind_direction,wind_speed,gust_speed,visibility,ceiling,cumulus_nimbus,weather1,weather2,weather3,rain_or_thunderstorm,temperature,temperature_min,temperature_max,temperature_dew,pressure,stringsAsFactors=FALSE)
  data = aux
  for (i in 2:24) {
    data = rbind(data, aux)
  }
  
  return(data)
}

# get_metar_string = function(ICAO_ID){
#   url = sprintf("https://tgftp.nws.noaa.gov/data/observations/metar/stations/%s.TXT",ICAO_ID)
#   metar = scan(what=character(),text=getURL(url),sep="\n")
#   return(metar)
# }
# 
# get_taf_string = function(ICAO_ID){
#   url = sprintf("https://tgftp.nws.noaa.gov/data/forecasts/taf/stations/%s.TXT",ICAO_ID)
#   taf = scan(what=character(),text=getURL(url),sep="\n")
#   return(taf)
# }
# 
# get_forecast_data = function(ICAO_ID)
# {
#   ans = get_metar_string(ICAO_ID)
#   m = metar_decoder(ans[2])
#   
#   ans = get_taf_string(ICAO_ID)
#   TAF = taf_decoder1(ans[2])
#   f = TAF[[1]]
#   op = as.character(NA)
#   while (TAF[[2]]){
#     TAF = taf_decoder2(TAF[[3]], f, op)
#     f = TAF[[1]]
#     op = TAF[[4]]
#   }
#   
#   if(length(ans) > 2){
#     for (i in 3:(length(ans))) {
#       TAF = taf_decoder2(ans[i], f, op)
#       f = TAF[[1]]
#       op = TAF[[4]]
#       while (TAF[[2]]){
#         TAF = taf_decoder2(TAF[[3]], f, op)
#         f = TAF[[1]]
#         op = TAF[[4]]
#       }
#     }
#   }
# 
#   m$temperature_max = f$temperature_max[1]
#   m$temperature_min = f$temperature_min[1]
#   f$temperature = m$temperature
#   f$temperature_dew = m$temperature_dew
#   f$pressure = m$pressure
#   
#   f = f[which(f$time > m$time),]
# 
#   data = null_data()
#   data[1,] = m
#   if(nrow(f) >= 23){
#     for (i in 1:23) {
#       data[(i+1),] = f[i,]
#     }
#   } else {
#     for (i in 1:nrow(f)) {
#       data[(i+1),] = f[i,]
#     }
#   }
#   
#   #data = rename(data, c("Time (YYYY-MM-DD HH:MM)" = "time", "Prabability" = "probability", "Wind Direction (degrees)" = "wind_direction", "Wind Speed (kt)" = "wind_speed", "Gusts Speed (kt)" = "gust_speed", "Visibility (nm)" = "visibility", "Ceiling (ft)" = "ceiling","Cumulus Nimbus" = "cumulus_nimbus", "Weather 1" = "weather1", "Weather 2" = "weather2", "Weather 3" = "weather3", "Rain or Thunderstorm" = "rain_or_thunderstorm", "Temperature (°C)" = "temperature", "Minumum Temperature (°C)" = "temperature_min", "Maximum Temperature (°C)" = "temperature_max", "Dew Point (°C)" = "temperature_dew", "Pressure (hPa)" = "pressure"))
#   return(data)
# }
