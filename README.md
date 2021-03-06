# Imdb Classification Tidymodels

## Introduction
Since I am a big fan of horror movies I wanted to investigate if you can classify movie data with machine learning.
In this project I deal with:
- A web scraping to get data from idbm
- A brief exploratorive data analysis
- Logistic Regression, Random Forest, XGBoost
- Tidymodles

With my web scraper I have managed not only to load data from one page but also to pull data from linked pages. I highly recommend rvest for web scraping. This is an excellent way to get data form the web. My scraper works for the top 250 movies of imdb, but since I need a lot of data for my project and since it is very time consuming to load the data of the 5000 movies, I was looking for a finished data set. For this reason, I started the ML part of this project with a data set from [Yueming](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset).

## Parts of me EDA
After a quick skim() I got a first overview of the dataset. In the following I analysed NA values, skewness, correlations and I plotted each numeric and factor variable with the help of loops.

## Modelling
In the following I made some final adjustments and split the data into test and train data.
```
set.seed(51069)
all_split <- initial_split(all, strata = horror_movie)
all_train <- training(all_split)
all_test <- testing(all_split)
```
Starting with tidymodels:
First, I started with a recipe. I built up this recipe based on the steps of [Hands-On Machine Learning with R](https://bradleyboehmke.github.io/HOML/engineering.html#proper-implementation) (Bradley Boehmke and Brandon Greenwell).
```
  step_knnimpute(all_predictors(), neighbors = 3) %>%
  step_nzv(all_nominal())  %>%
  step_BoxCox(all_numeric(),-all_outcomes()) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>% 
  step_pca(all_numeric(), -all_outcomes())%>%
  step_other(all_nominal(), -all_outcomes(), threshold = 0.01) %>%
  step_dummy(all_nominal(), -all_outcomes())
```
Afterwards, I used the prep, the juice function and started the workflow.
```
all_prep <- prep(all_rec)
juiced <- juice(all_prep)

wf <- workflow() %>%
  add_recipe(all_rec)

test_bake <- all_prep %>%
  bake(all_test)
```
As resampling method, I decided to do a quick k-fold cross validation instead of bootstrapping.
```
vfold <- all_train %>%
  vfold_cv(v = 5, strata = "horror_movie")
```

### Models
The workflow of the Logistic Regression deviated from the workflow of the other models where I also did some parameter tuning.

For the Random Forest and XGBoost I proceeded as following:

1.    Define a parsnip model
2.    Define parameters using dials package
3.    Combine model and recipe using workflows package
4.    Tune the workflow using tune package
5.    Evaluate tuning results
6.    Select best model for prediction
7.    Prediction of target variable using test data
8.    Finally, save the prediction

In addition, I promise not to go too deep into the optimization of each model, as this would go beyond the scope of these projects.

#### Logistic Regression
```
# Define the model 
logistic_glm <- logistic_reg(mode = "classification") %>%
  set_engine("glm")

# Train the model 
logistic_glm <-
  logistic_reg(mode = "classification") %>%
  set_engine("glm") %>%
  fit(horror_movie ~ ., data = juiced)
```
Looking at the accuracy of the model
```
predictions_glm <- logistic_glm %>%
  predict(new_data = test_bake) %>%
  bind_cols(test_bake %>% select(horror_movie))

predictions_glm %>%
  metrics(horror_movie, .pred_class) %>%
  select(-.estimator) %>%
  filter(.metric == "accuracy") 
```
Looking at the confusion matrices
```
predictions_glm %>%
  conf_mat(horror_movie, .pred_class) %>%
  pluck(1) %>%
  as_tibble() %>%
  ggplot(aes(Prediction, Truth, alpha = n)) +
  geom_tile(show.legend = FALSE) +
  geom_text(aes(label = n), colour = "white", alpha = 1, size = 8)
```
<img width="450" alt="Rplot_logr_con" src="Rplot_logr_con.png">

#### Random Forest

Random Forest:

1.-3. I decide to tune mtry and min_n. Additionally I did some more specifications.
```
# 1.
rf_mod <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()) %>%
  set_mode("classification") %>%
  set_engine("ranger")
 
 # 2. Is not necessary
 
 # 3.
tune_wf <- workflow() %>%
  add_recipe(all_rec) %>%
  add_model(rf_mod)
```
4. Tune the workflow using tune package
```
# 4.
doParallel::registerDoParallel()

tictoc::tic()
set.seed(51069)
tune_res <- tune_grid(
  tune_wf,
  resamples = vfold,
  grid = 10
)
tictoc::toc()

tune_res
```
5.-6. I used roc_auc to evaluate each model
```
# 5.
show_best(tune_res, "roc_auc", n = 10)

# 6.
rf_param_best <- select_best(tune_res, metric = "roc_auc")
rf_model_best <- finalize_model(rf_mod, rf_param_best)
rf_model_finalfit <- fit(rf_model_best, horror_movie ~ ., data = juiced)
```
Looking at the tuning parameters

<img width="450" alt="Rplot_RF" src="Rplot_RF.png">

7. Prediction of target variable using test data
```
# 7.
test_prep <- all_prep %>%
  bake(all_test)

final_wf <- workflow() %>%
  add_recipe(all_rec) %>%
  add_model(rf_model_best)

final_res <- final_wf %>%
  last_fit(all_split)
```
Looking at the accuracy of the model and the the confusion matrices
```
final_res %>%
  collect_metrics()

Con_Mat <- final_res$.predictions 
Con_Mat <- as.data.frame(Con_Mat)%>%
  select(.pred_class, horror_movie)
  
Con_Mat %>%
  conf_mat( horror_movie,.pred_class) %>%
  pluck(1) %>%
  as_tibble() %>%
  ggplot(aes(Prediction, Truth, alpha = n)) +
  geom_tile(show.legend = FALSE) +
  geom_text(aes(label = n), colour = "white", alpha = 1, size = 8)
```
<img width="450" alt="Rplot_rf_con" src="Rplot_rf_con.png">

#### XGBoost Model:

1.-3. I decided to tune each parameter and additionally I made more specifications.
```
# 1.
xgb_model <- boost_tree(
  trees = 1000, 
  tree_depth = tune(), min_n = tune(), 
  loss_reduction = tune(),                     # first three: model complexity
  sample_size = tune(), mtry = tune(),         # randomness
  learn_rate = tune(),                         # step size
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

# 2.
xgb_param <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), all_train),
  learn_rate(),
  size = 30
)

# 3.
xgb_workflow <- 
    workflow() %>% 
    add_recipe(all_rec) %>% 
    add_model(xgb_model)
```
To go more into detail I recommend Bradley Boehmke and Brandon Greenwell’s [tuning strategy for XGBoost](https://bradleyboehmke.github.io/HOML/gbm.html#xgb-tuning-strategy).

4. Tune the workflow using tune package
```
# 4. 
tictoc::tic()
doParallel::registerDoParallel()

set.seed(234)
xgb_res <- tune_grid(
  xgb_workflow,
  resamples = vfold,
  grid = xgb_param,
  control = control_grid(save_pred = TRUE)
)
tictoc::toc()
```
5.-6. I used roc_auc to evaluate each model
```
# 5.
show_best(xgb_res, "roc_auc",  n = 10)

# 6.
xgb_param_best <- select_best(xgb_res, metric = "roc_auc" )
xgb_model_best <- finalize_model(xgb_model, xgb_param_best)
xgb_model_finalfit <- fit(xgb_model_best, horror_movie ~ ., data = juiced)
```
Looking at the tuning parameters

<img width="450" alt="Rplot_xgb" src="Rplot_xgb.png">

7. Prediction of target variable using test data
```
# 7.
final_wf <- workflow() %>%
  add_recipe(all_rec) %>%
  add_model(xgb_model_best)

final_res <- final_wf %>%
  last_fit(all_split)

final_res %>%
  collect_metrics()
```
Looking at confusion matrices and the most important variables

<img width="450" alt="Rplot_xgb_con" src="Rplot_xgb_con.png">

<img width="450" alt="Rplot_xgb_vip" src="Rplot_xgb_vip.png">
 
### Compare the models
#### Accuracy:
-	Log: 0.8809524
-	RF: 0.8841270	
-	XGB: 0.8801587

#### Confusion Matrices:
<img width="900" alt="Rplot_logr_con" src="Rplot_con.png">


## Resume:
The results are very interesting. I thought that these projects could be difficult but I saw a possibility to develop my skills and to deal with a topic I enjoy - this project was a success.

