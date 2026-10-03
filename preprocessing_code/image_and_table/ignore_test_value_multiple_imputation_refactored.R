library(mice)
library(stringr)

mice_imputation_full <- function(train_file, val_file, test_file, lm_fit_summary_csv, lm_fit_pool_summary, output_path, save_model_name){
  train <- read.csv(train_file, na.strings = c('NaN', '', 'nan'))
  val <- read.csv(val_file, na.strings = c('NaN', '', 'nan'))
  test <- read.csv(test_file, na.strings = c('NaN', '', 'nan'))
  
  # training testing split
  train["divide"] <- "train"
  val["divide"] <- "val"
  test["divide"] <- "test"
  ran <- rbind(train,val,test)
  summary(as.factor(ran$divide))
  
  # only keep patient id and 12 vars + pwv + divide
  names(ran)
  ori_ran <- ran[c(1,7:20)]
  summary(ori_ran)
  
  # during imputation ignore val or test set
  ignore <- !ori_ran$divide=="train"
  summary(ignore)
  summary(as.factor(ori_ran$divide))

  # only keep patient id and 12 vars + pwv
  chabu <- ori_ran[,-15]
  names(chabu)
  summary(is.na(chabu))

  
  # initialize imputation. do not consider patient_id and param35 vars
  init = mice(chabu, maxit=0)
  meth<-init$method
  predM<-init$predictorMatrix
  methods(mice)
  names(chabu)
  predM[,("patient_id")]=0
  predM[,("param35")]=0
  meth[c("param3")]="pmm"
  meth[c("param4")]="pmm"
  meth[c("param5")]="pmm"
  meth[c("param6")]="pmm"
  meth[c("param7")]="pmm"
  meth[c("param8")]="logreg"
  meth[c("param9")]="logreg"
  meth[c("param10")]="logreg"
  meth[c("param11")]="logreg"
  meth[c("param12")]="logreg"
  chabu$param8<-  factor(chabu$param8)
  chabu$param9 <-  factor(chabu$param9)
  chabu$param10 <-  factor(chabu$param10)
  chabu$param11 <-  factor(chabu$param11)
  chabu$param12 <-  factor(chabu$param12)
  summary(chabu)
  summary(chabu$param8)
  summary(chabu$param9)
  summary(chabu$param10)
  
  # start imputation
  imputed =  mice(chabu, method=meth, predictorMatrix=predM,ignore=ignore, m=20)
  
  # post imputation analysis
  fit <- with(imputed,lm(param35~param1+param2+param3+param4+param5+param6+param7+param8+param9+param10+param11+param12))
  f <- summary(fit)
  write.csv(f, lm_fit_summary_csv)
  g <- summary(pool(fit))
  write.csv(g, lm_fit_pool_summary)

  # save imputation results
  for (i in 1:20)
  {
    output_file_name = paste(output_path, '/mice_imputation_full_', toString(i), '.csv', sep='')
    write.csv(complete(imputed, i), output_file_name)
  }

  # save imputation model
  save(imputed, file = save_model_name)
}






