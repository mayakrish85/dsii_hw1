Data Science II HW 1
================
Maya Krishnamoorthy
2025-02-22

Read and prep CSV files.

``` r
train_df = read_csv("housing_training.csv") %>% janitor::clean_names()
```

    ## Rows: 1440 Columns: 26
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr  (4): Overall_Qual, Kitchen_Qual, Fireplace_Qu, Exter_Qual
    ## dbl (22): Gr_Liv_Area, First_Flr_SF, Second_Flr_SF, Total_Bsmt_SF, Low_Qual_...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
test_df = read_csv("housing_test.csv") %>% janitor::clean_names()
```

    ## Rows: 959 Columns: 26
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr  (4): Overall_Qual, Kitchen_Qual, Fireplace_Qu, Exter_Qual
    ## dbl (22): Gr_Liv_Area, First_Flr_SF, Second_Flr_SF, Total_Bsmt_SF, Low_Qual_...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

Response variable: `sale_price`

## Part a)

**Fit a lasso model on the training data. Report the selected tuning
parameter and the test error. When the 1SE rule is applied, how many
predictors are included in the model?**

### Minimizing CV error

``` r
# Using glmnet
ctrl1 = trainControl(method = "cv", number = 10)

set.seed(2025)
lasso.fit =
  train(sale_price ~ .,
        data = train_df,
        method = "glmnet",
        tuneGrid = expand.grid(alpha = 1, lambda = exp(seq(6, 0, length = 100))),
        trControl = ctrl1)

plot(lasso.fit, xTrans = log)
```

![](ds2_hw1_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
lasso.fit$bestTune
```

    ##    alpha   lambda
    ## 70     1 65.48481

``` r
# coefficients in the final model
coef(lasso.fit$finalModel, lasso.fit$bestTune$lambda)
```

    ## 40 x 1 sparse Matrix of class "dgCMatrix"
    ##                                       s1
    ## (Intercept)                -4.827229e+06
    ## gr_liv_area                 6.538426e+01
    ## first_flr_sf                8.025639e-01
    ## second_flr_sf               .           
    ## total_bsmt_sf               3.541790e+01
    ## low_qual_fin_sf            -4.093893e+01
    ## wood_deck_sf                1.163406e+01
    ## open_porch_sf               1.543115e+01
    ## bsmt_unf_sf                -2.088742e+01
    ## mas_vnr_area                1.090025e+01
    ## garage_cars                 4.083736e+03
    ## garage_area                 8.168146e+00
    ## year_built                  3.233277e+02
    ## tot_rms_abv_grd            -3.616290e+03
    ## full_bath                  -3.840140e+03
    ## overall_qualAverage        -4.853059e+03
    ## overall_qualBelow_Average  -1.245648e+04
    ## overall_qualExcellent       7.549879e+04
    ## overall_qualFair           -1.075093e+04
    ## overall_qualGood            1.212071e+04
    ## overall_qualVery_Excellent  1.357075e+05
    ## overall_qualVery_Good       3.789087e+04
    ## kitchen_qualFair           -2.483806e+04
    ## kitchen_qualGood           -1.720012e+04
    ## kitchen_qualTypical        -2.531083e+04
    ## fireplaces                  1.054387e+04
    ## fireplace_quFair           -7.665787e+03
    ## fireplace_quGood            .           
    ## fireplace_quNo_Fireplace    1.438181e+03
    ## fireplace_quPoor           -5.640226e+03
    ## fireplace_quTypical        -7.011019e+03
    ## exter_qualFair             -3.332982e+04
    ## exter_qualGood             -1.508538e+04
    ## exter_qualTypical          -1.952240e+04
    ## lot_frontage                9.963955e+01
    ## lot_area                    6.042538e-01
    ## longitude                  -3.293051e+04
    ## latitude                    5.507513e+04
    ## misc_val                    8.278839e-01
    ## year_sold                  -5.601503e+02

``` r
# test error
predictions = predict(lasso.fit, newdata = test_df)
mse = mean((predictions - pull(test_df, "sale_price"))^2)
```

The tuning parameter that minimizes CV error is 65.48481. The MSE
associated with this model is 439967917.

### Minimizing for 1SE

``` r
ctrl2 = 
  trainControl(method = "repeatedcv",
               number = 10,
               repeats = 5,
               selectionFunction = "oneSE")

set.seed(2025)

lasso.fit_1se = 
  train(sale_price ~ .,
        data = train_df,
        method = "glmnet",
        tuneGrid = expand.grid(alpha = 1, lambda = exp(seq(6, 0, length = 100))),
        trControl = ctrl2)

# Best lambda for 1SE
lasso.fit_1se$bestTune
```

    ##     alpha   lambda
    ## 100     1 403.4288

``` r
# Getting number of predictors
coeff = coef(lasso.fit_1se$finalModel, lasso.fit_1se$bestTune$lambda)
length(which(coeff != 0)) - 1
```

    ## [1] 36

``` r
# MSE
predictions = predict(lasso.fit_1se, newdata = test_df)
mse = mean((predictions - pull(test_df, "sale_price"))^2); mse
```

    ## [1] 420726548

Using the 1SE rule, the optimal tuning parameter is 403.4288. The MSE is
420726548. There are 36 (non-zero) predictors in this model.

## Part b)

**Fit an elastic net model on the training data. Report the selected
tuning parameters and the test error. Is it possible to apply the 1SE
rule to select the tuning parameters for elastic net? If the 1SE rule is
applicable, implement it to select the tuning parameters. If not,
explain why.**

``` r
enet.fit <- train(sale_price ~ .,
                  data = train_df,
                  method = "glmnet",
                  tuneGrid = expand.grid(alpha = seq(0, 1, length = 20), 
                                         lambda = exp(seq(10, 0, length = 100))),
                  trControl = ctrl1)
enet.fit$bestTune
```

    ##          alpha   lambda
    ## 166 0.05263158 710.2781

``` r
myCol <- rainbow(25)
myPar <- list(superpose.symbol = list(col = myCol),
              superpose.line = list(col = myCol))

plot(enet.fit, par.settings = myPar, xTrans = log)
```

![](ds2_hw1_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
# coefficients in the final model
coef(enet.fit$finalModel, enet.fit$bestTune$lambda)
```

    ## 40 x 1 sparse Matrix of class "dgCMatrix"
    ##                                       s1
    ## (Intercept)                -5.127932e+06
    ## gr_liv_area                 3.863982e+01
    ## first_flr_sf                2.646422e+01
    ## second_flr_sf               2.510967e+01
    ## total_bsmt_sf               3.487656e+01
    ## low_qual_fin_sf            -1.602950e+01
    ## wood_deck_sf                1.237978e+01
    ## open_porch_sf               1.699004e+01
    ## bsmt_unf_sf                -2.067876e+01
    ## mas_vnr_area                1.190003e+01
    ## garage_cars                 4.012079e+03
    ## garage_area                 9.117167e+00
    ## year_built                  3.178099e+02
    ## tot_rms_abv_grd            -3.350809e+03
    ## full_bath                  -3.547688e+03
    ## overall_qualAverage        -5.127109e+03
    ## overall_qualBelow_Average  -1.267432e+04
    ## overall_qualExcellent       7.628805e+04
    ## overall_qualFair           -1.152800e+04
    ## overall_qualGood            1.190476e+04
    ## overall_qualVery_Excellent  1.373009e+05
    ## overall_qualVery_Good       3.755653e+04
    ## kitchen_qualFair           -2.317538e+04
    ## kitchen_qualGood           -1.563756e+04
    ## kitchen_qualTypical        -2.371099e+04
    ## fireplaces                  1.072006e+04
    ## fireplace_quFair           -7.966364e+03
    ## fireplace_quGood            5.597628e+01
    ## fireplace_quNo_Fireplace    1.495223e+03
    ## fireplace_quPoor           -5.908039e+03
    ## fireplace_quTypical        -7.052687e+03
    ## exter_qualFair             -3.205560e+04
    ## exter_qualGood             -1.367021e+04
    ## exter_qualTypical          -1.832736e+04
    ## lot_frontage                9.963098e+01
    ## lot_area                    6.027902e-01
    ## longitude                  -3.515993e+04
    ## latitude                    5.744070e+04
    ## misc_val                    8.545898e-01
    ## year_sold                  -5.600438e+02

``` r
# MSE
enet.pred = predict(enet.fit, newdata = test_df)
mean((enet.pred - pull(test_df, "sale_price"))^2)
```

    ## [1] 437032337

The optimal tuning parameters for the elastic net model are lambda =
316.5799 and alpha = 0.05263158. The test error is 441832112.

``` r
# Trying 1SE method
set.seed(2025)
enet.fit_1se <- train(sale_price ~ .,
                  data = train_df,
                  method = "glmnet",
                  tuneGrid = expand.grid(alpha = seq(0, 1, length = 20), 
                                         lambda = exp(seq(10, 0, length = 100))),
                  trControl = ctrl2)
enet.fit_1se$bestTune
```

    ##    alpha   lambda
    ## 88     0 6554.314

``` r
# MSE
enet.pred_1se = predict(enet.fit_1se, newdata = test_df)
mean((enet.pred_1se - pull(test_df, "sale_price"))^2)
```

    ## [1] 426591709

Using 1SE, the resulting tuning parameters are alpha = 0 and lambda =
6554.314. The resulting test error is 426591709. Since the test errors
are similar for both 1SE and CV methods, we can go ahead and use the 1SE
method as well. **CHECK THIS**
